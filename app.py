import requests
import re
from dotenv import load_dotenv
import os
import sqlparse
import pandas as pd
import json
import io
from flask import Flask, render_template, request, jsonify

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# In-memory storage
uploaded_files = {}  # key = table name, value = DataFrame
uploaded_schema = ""  # stores DBML text

# ------------------------------
# Call Gemini API
# ------------------------------
def call_gemini(prompt: str, temperature: float = 0.7):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY},
            json={"contents": contents, "generationConfig": {"temperature": temperature}}
        )
        resp.raise_for_status()
        data = resp.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            if len(parts) > 0 and "text" in parts[0]:
                return parts[0]["text"]
        return f"⚠️ Respuesta inesperada: {json.dumps(data)}"
    except requests.exceptions.HTTPError as e:
        return f"⚠️ Error HTTP: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"⚠️ Error en la llamada a Gemini: {str(e)}"

# ------------------------------
# Clean SQL response
# ------------------------------
def clean_sql_response(text: str):
    match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r"```(.*?)```", text, re.DOTALL)
    sql = match.group(1).strip() if match else text.strip()
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    sql = re.sub(r'\s+', ' ', sql).strip()
    if sql.endswith(';'):
        sql = sql[:-1]
    # Remove aliases like AS T1, T2
    sql = re.sub(r'\s+as\s+\w+', '', sql, flags=re.IGNORECASE)
    return sql.lower()

# ------------------------------
# Execute SQL on memory
# ------------------------------
def execute_sql_on_memory(sql, files_dict):
    try:
        from pandasql import sqldf
        normalized_files = {k.lower(): v for k, v in files_dict.items() if isinstance(v, pd.DataFrame)}
        pysqldf = lambda q: sqldf(q, normalized_files)
        result = pysqldf(sql)
        if result.empty:
            return json.dumps({"message": "La consulta no devolvió resultados"})
        return result.head(100).to_json(orient='records', force_ascii=False)
    except ImportError:
        return "⚠️ Instala pandasql: pip install pandasql"
    except Exception as e:
        return f"⚠️ Error ejecutando SQL: {str(e)}"

# ------------------------------
# Generate schema from CSV files
# ------------------------------
def generate_schema_from_csvs(files_dict):
    """Genera un schema automático a partir de los DataFrames cargados"""
    schema = {}
    for table_name, df in files_dict.items():
        columns = []
        for col_name in df.columns:
            dtype = df[col_name].dtype
            if dtype == 'int64':
                col_type = 'integer'
            elif dtype == 'float64':
                col_type = 'float'
            elif dtype == 'bool':
                col_type = 'boolean'
            else:
                col_type = 'varchar'
            columns.append({"name": col_name, "type": col_type})
        schema[table_name] = {"columns": columns}
    return schema

# ------------------------------
# Upload route
# ------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_files, uploaded_schema
    files = request.files.getlist("file")
    if not files or all(f.filename == '' for f in files):
        return jsonify({"result": "⚠️ No se seleccionó ningún archivo."}), 400

    results = []
    uploaded_files.clear()
    uploaded_schema = ""

    for file in files:
        if not file or file.filename == '':
            continue
        filename = file.filename
        try:
            if filename.lower().endswith('.txt'):
                uploaded_schema = file.read().decode('utf-8', errors='ignore').strip()
                results.append(f"✓ Schema '{filename}' cargado exitosamente.")
            elif filename.lower().endswith('.csv'):
                content = file.read().decode('utf-8', errors='ignore')
                table_name = os.path.splitext(filename)[0].lower()
                table_name = re.sub(r'[^a-z0-9_]', '_', table_name)
                try:
                    df = pd.read_csv(io.StringIO(content))
                except:
                    try:
                        df = pd.read_csv(io.StringIO(content), sep=';')
                    except:
                        df = pd.read_csv(io.StringIO(content), sep='\t')
                # Convert columns to lowercase
                df.columns = [c.lower() for c in df.columns]
                uploaded_files[table_name] = df
                results.append(f"✓ CSV '{filename}' cargado como tabla '{table_name}' ({len(df)} filas, {len(df.columns)} columnas).")
            else:
                results.append(f"⚠️ Archivo ignorado (solo .csv y .txt): {filename}")
        except Exception as e:
            results.append(f"⚠️ Error procesando {filename}: {str(e)}")
    
    return jsonify({"result": "\n".join(results)})

# ------------------------------
# Parse DBML for Gemini
# ------------------------------
def parse_dbml_to_gemini_schema(dbml_text: str):
    schema = {}
    table_pattern = r"Table\s+(\w+)\s*\{([^}]*)\}"
    for table_match in re.finditer(table_pattern, dbml_text, re.DOTALL | re.IGNORECASE):
        table_name = table_match.group(1).lower()
        body = table_match.group(2)
        columns = []
        for col_line in body.strip().splitlines():
            col_line = col_line.strip()
            if not col_line or col_line.startswith("//") or col_line.startswith("--"):
                continue
            col_match = re.match(r'(\w+)\s+(\w+(?:\([^)]+\))?)', col_line)
            if col_match:
                col_name = col_match.group(1).lower()
                col_type = col_match.group(2).lower()
                columns.append({"name": col_name, "type": col_type})
        if columns:
            schema[table_name] = {"columns": columns}
    ref_pattern = r"Ref:\s*['\"]?(\w+)\.(\w+)['\"]?\s*[<>-]+\s*['\"]?(\w+)\.(\w+)['\"]?"
    relations = []
    for match in re.finditer(ref_pattern, dbml_text, re.IGNORECASE):
        relations.append({
            "from": f"{match.group(1).lower()}.{match.group(2).lower()}",
            "to": f"{match.group(3).lower()}.{match.group(4).lower()}"
        })
    if relations:
        schema["relations"] = relations
    return schema

# ------------------------------
# Convert DBML to Mermaid Diagram
# ------------------------------
def dbml_to_mermaid(dbml_text: str) -> str:
    """
    Convierte schema DBML a diagrama Mermaid ERD (versión compacta)
    """
    lines = []
    lines.append("erDiagram")
    
    # Parse tables
    table_pattern = r"Table\s+(\w+)\s*\{([^}]*)\}"
    tables = {}
    
    for table_match in re.finditer(table_pattern, dbml_text, re.DOTALL | re.IGNORECASE):
        table_name = table_match.group(1)
        body = table_match.group(2)
        
        columns = []
        all_columns = []
        for col_line in body.strip().splitlines():
            col_line = col_line.strip()
            if not col_line or col_line.startswith("//") or col_line.startswith("--"):
                continue
            
            col_match = re.match(r'(\w+)\s+(\w+(?:\([^)]+\))?)(.*)', col_line)
            if col_match:
                col_name = col_match.group(1)
                col_type = col_match.group(2)
                modifiers = col_match.group(3)
                
                # Detect primary key
                is_pk = "primary key" in modifiers.lower()
                
                # Simplificar tipos de datos
                simple_type = col_type.split('(')[0]
                
                col_info = {
                    "name": col_name,
                    "type": simple_type,
                    "is_pk": is_pk
                }
                all_columns.append(col_info)
                
                # Solo agregar PKs y primeras 3 columnas no-PK
                if is_pk or len([c for c in columns if not c['is_pk']]) < 3:
                    columns.append(col_info)
        
        tables[table_name] = {"columns": columns, "total": len(all_columns)}
    
    # Generate Mermaid table definitions (compacto)
    for table_name, table_data in tables.items():
        lines.append(f"    {table_name} {{")
        for col in table_data["columns"]:
            if col['is_pk']:
                lines.append(f"        {col['type']} {col['name']} PK")
            else:
                lines.append(f"        {col['type']} {col['name']}")
        lines.append("    }")
    
    # Parse relationships
    ref_pattern = r"Ref:\s*['\"]?(\w+)\.(\w+)['\"]?\s*([<>-]+)\s*['\"]?(\w+)\.(\w+)['\"]?"
    
    for match in re.finditer(ref_pattern, dbml_text, re.IGNORECASE):
        table1 = match.group(1)
        table2 = match.group(4)
        operator = match.group(3)
        
        # Simplificar relaciones
        if ">" in operator:
            relationship = "||--o{"
        elif "<" in operator:
            relationship = "}o--||"
        else:
            relationship = "||--||"
        
        lines.append(f"    {table1} {relationship} {table2} : \"\"")
    
    return "\n".join(lines)

# ------------------------------
# Generate Diagram route
# ------------------------------
@app.route("/generate-diagram", methods=["POST"])
def generate_diagram():
    if not uploaded_schema:
        return jsonify({"result": "⚠️ No se ha cargado el archivo de schema (.txt)."}), 400
    
    try:
        mermaid_code = dbml_to_mermaid(uploaded_schema)
        return jsonify({"diagram": mermaid_code})
    except Exception as e:
        return jsonify({"result": f"⚠️ Error generando diagrama: {str(e)}"}), 500

# ------------------------------
# Generate SQL route (MODIFICADO)
# ------------------------------
@app.route("/generate-sql", methods=["POST"])
def generate_sql():
    data = request.json
    user_prompt = data.get("prompt")
    
    if not user_prompt:
        return jsonify({"result": "⚠️ Debes enviar un prompt con tu consulta."}), 400

    # Debug: verificar estado
    print(f"DEBUG - Schema cargado: {bool(uploaded_schema)}, Longitud: {len(uploaded_schema) if uploaded_schema else 0}")
    print(f"DEBUG - Archivos CSV: {len(uploaded_files)}, Tablas: {list(uploaded_files.keys())}")

    # Caso 1: Solo schema.txt (sin CSVs)
    if uploaded_schema and len(uploaded_files) == 0:
        parsed_schema = parse_dbml_to_gemini_schema(uploaded_schema)
        sql_prompt = (
            "Eres un experto en SQL. Genera únicamente la consulta SQL pura, "
            "sin comentarios ni aliases. Usa tablas y columnas en minúsculas.\n"
            "NUNCA utilices alias o AS en las tablas.\n"
            "Formatea la consulta con saltos de línea y sangría para que sea fácilmente legible.\n"
            f"Schema:\n```json\n{json.dumps(parsed_schema, indent=2)}\n```\n"
            f"Pregunta del usuario: {user_prompt}"
        )
        
        gemini_response = call_gemini(sql_prompt)
        if gemini_response.startswith("⚠️"):
            return jsonify({"result": gemini_response}), 500

        sql_generated = clean_sql_response(gemini_response)
        
        try:
            sql_generated_formatted = sqlparse.format(
                sql_generated, 
                reindent=True,
                keyword_case='upper'
            )
        except Exception:
            sql_generated_formatted = sql_generated

        return jsonify({
            "sql": sql_generated_formatted,
            "preview": "⚠️ No hay CSVs cargados. Solo se generó el SQL sin ejecutar."
        })

    # Caso 2: Solo CSVs (sin schema.txt)
    elif len(uploaded_files) > 0 and not uploaded_schema:
        # Generar schema automáticamente desde los CSVs
        parsed_schema = generate_schema_from_csvs(uploaded_files)
        sql_prompt = (
            "Eres un experto en SQL. Genera únicamente la consulta SQL pura, "
            "sin comentarios ni aliases. Usa tablas y columnas en minúsculas.\n"
            "NUNCA utilices alias o AS en las tablas.\n"
            "Formatea la consulta con saltos de línea y sangría para que sea fácilmente legible.\n"
            f"Schema (generado automáticamente):\n```json\n{json.dumps(parsed_schema, indent=2)}\n```\n"
            f"Pregunta del usuario: {user_prompt}"
        )
        
        gemini_response = call_gemini(sql_prompt)
        if gemini_response.startswith("⚠️"):
            return jsonify({"result": gemini_response}), 500

        sql_generated = clean_sql_response(gemini_response)
        
        try:
            sql_generated_formatted = sqlparse.format(
                sql_generated, 
                reindent=True,
                keyword_case='upper'
            )
        except Exception:
            sql_generated_formatted = sql_generated

        preview_result = execute_sql_on_memory(sql_generated, uploaded_files)
        if isinstance(preview_result, str) and preview_result.startswith("⚠️"):
            return jsonify({"sql": sql_generated_formatted, "result": preview_result}), 500

        return jsonify({"sql": sql_generated_formatted, "preview": preview_result})

    # Caso 3: CSVs + schema.txt
    elif len(uploaded_files) > 0 and uploaded_schema:
        parsed_schema = parse_dbml_to_gemini_schema(uploaded_schema)
        sql_prompt = (
            "Eres un experto en SQL. Genera únicamente la consulta SQL pura, "
            "sin comentarios ni aliases. Usa tablas y columnas en minúsculas.\n"
            "NUNCA utilices alias o AS en las tablas.\n"
            "Formatea la consulta con saltos de línea y sangría para que sea fácilmente legible.\n"
            f"Schema:\n```json\n{json.dumps(parsed_schema, indent=2)}\n```\n"
            f"Pregunta del usuario: {user_prompt}"
        )
        
        gemini_response = call_gemini(sql_prompt)
        if gemini_response.startswith("⚠️"):
            return jsonify({"result": gemini_response}), 500

        sql_generated = clean_sql_response(gemini_response)
        
        try:
            sql_generated_formatted = sqlparse.format(
                sql_generated, 
                reindent=True,
                keyword_case='upper'
            )
        except Exception:
            sql_generated_formatted = sql_generated

        preview_result = execute_sql_on_memory(sql_generated, uploaded_files)
        if isinstance(preview_result, str) and preview_result.startswith("⚠️"):
            return jsonify({"sql": sql_generated_formatted, "result": preview_result}), 500

        return jsonify({"sql": sql_generated_formatted, "preview": preview_result})

    # Caso 4: No hay nada cargado
    else:
        return jsonify({"result": "⚠️ Debes cargar al menos un archivo CSV o un schema.txt."}), 400

# ------------------------------
# Clear memory
# ------------------------------
@app.route("/clear", methods=["POST"])
def clear_memory():
    global uploaded_files, uploaded_schema
    uploaded_files.clear()
    uploaded_schema = ""
    return jsonify({"result": "✓ Memoria limpiada exitosamente."})

# ------------------------------
# System status
# ------------------------------
@app.route("/status", methods=["GET"])
def system_status():
    status = {
        "schema_loaded": bool(uploaded_schema),
        "tables_count": len(uploaded_files),
        "tables": list(uploaded_files.keys()),
        "total_rows": sum(len(df) for df in uploaded_files.values())
    }
    return jsonify(status)

@app.route("/")
def index():
    return render_template("index.html")

# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)