import json
import argparse

def validate_jsonl(path, required_field="text"):
    total = 0
    valid = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i==1:
                print(line)
            total += 1
            try:
                obj = json.loads(line)
                if required_field not in obj:
                    print(f"❌ Line {i}: Missing required field '{required_field}'")
                else:
                    valid += 1
            except json.JSONDecodeError as e:
                print(f"❌ Line {i}: JSON decode error - {e}")
    
    print(f"\n🧾 Checked {total} lines.")
    print(f"✅ {valid} valid lines with '{required_field}' field.")
    print(f"❌ {total - valid} invalid or incomplete lines.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a JSONL file.")
    parser.add_argument("jsonl_file", help="Path to the JSONL file to validate")
    parser.add_argument("--required_field", default="text", help="Field that must be present in each JSON object")
    args = parser.parse_args()

    validate_jsonl(args.jsonl_file, args.required_field)