import pandas as pd
import json
import argparse
from pathlib import Path

def csv_to_jsonl(input_csv, output_jsonl, text_column="text", chunksize=10000):
    output_path = Path(output_jsonl)
    is_first_chunk = True

    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        mode = "w" if is_first_chunk else "a"
        with output_path.open(mode, encoding="utf-8") as f_out:
            for _, row in chunk.iterrows():
                if pd.isna(row[text_column]):
                    continue  # skip rows with missing text

                row_dict = row.to_dict()
                text_value = row_dict.pop(text_column)
                text_value = text_value.lower().replace('\n', ' ')
                
                json_obj = {
                    "text": text_value,
                    **row_dict  # other columns as metadata
                }
                f_out.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        
        is_first_chunk = False  # flip flag after first chunk

    print(f"✅ Done. JSONL file written to: {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CSV file to JSONL format with 'text' and metadata.")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_jsonl", help="Path to output JSONL file")
    parser.add_argument("--text_column", default="text", help="Name of the column to use as the 'text' field")
    parser.add_argument("--chunksize", type=int, default=10000, help="Number of rows per chunk to process (default: 10000)")

    args = parser.parse_args()

    csv_to_jsonl(args.input_csv, args.output_jsonl, args.text_column, args.chunksize)
