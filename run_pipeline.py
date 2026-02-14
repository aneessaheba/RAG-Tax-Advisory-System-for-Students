"""
Run the full data pipeline: Extract PDFs -> Clean -> Chunk -> Embed -> Load into ChromaDB.
Run this once to prepare the database, then use app.py to chat.
"""
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'tax_rag_data')

STEPS = [
    ("Step 1: Extract text from PDFs", "extract_pdfs_to_json.py"),
    ("Step 2: Clean extracted text", "clean_parsed_json.py"),
    ("Step 3: Split into chunks", "split_clean_json_to_chunks.py"),
    ("Step 4: Embed chunks", "embed_chunks.py"),
    ("Step 5: Load into ChromaDB", "upload_to_chromadb.py"),
]


def main():
    for label, script in STEPS:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}\n")
        script_path = os.path.join(DATA_DIR, script)
        result = subprocess.run([sys.executable, script_path])
        if result.returncode != 0:
            print(f"\nERROR: {label} failed. Fix the issue and re-run.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  Pipeline complete! Run the bot with: python app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
