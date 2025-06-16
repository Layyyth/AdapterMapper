import os
import json
import yaml
from glob import glob

def extract_fields_from_schema(schema_name, schema_obj, parent=""):
    fields = []

    if "properties" in schema_obj:
        for prop, details in schema_obj["properties"].items():
            full_field = f"{parent}.{prop}" if parent else prop

            if "$ref" in details:
                ref = details["$ref"].split("/")[-1]
                fields.append({
                    "schema": schema_name,
                    "field": full_field,
                    "ref": ref
                })
            else:
                fields.append({
                    "schema": schema_name,
                    "field": full_field,
                    "type": details.get("type", "unknown"),
                    "description": details.get("description", "")
                })

                
                if details.get("type") == "object" and "properties" in details:
                    fields.extend(extract_fields_from_schema(schema_name, details, full_field))

                
                elif details.get("type") == "array" and "items" in details:
                    items = details["items"]
                    if "$ref" in items:
                        ref = items["$ref"].split("/")[-1]
                        fields.append({
                            "schema": schema_name,
                            "field": f"{full_field}[]",
                            "ref": ref
                        })
                    elif "properties" in items:
                        fields.extend(extract_fields_from_schema(schema_name, items, f"{full_field}[]"))

    return fields

def parse_openapi_schemas(directory):
    all_fields = []

    files = glob(os.path.join(directory, "*.yaml")) + glob(os.path.join(directory, "*.yml")) + glob(os.path.join(directory, "*.json"))
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                    spec = yaml.safe_load(f)
                else:
                    spec = json.load(f)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                continue

            schemas = spec.get("components", {}).get("schemas", {})
            for schema_name, schema_obj in schemas.items():
                fields = extract_fields_from_schema(schema_name, schema_obj)
                all_fields.extend(fields)

    return all_fields

if __name__ == "__main__":
    
    input_dir = "read-write-api-specs/dist/openapi"

    output_file = "ob_field_schema.json"

    print(" Scanning OpenAPI files...")
    result = parse_openapi_schemas(input_dir)

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(result, out, indent=2)

    print(f" Done. Extracted {len(result)} fields to: {output_file}")
