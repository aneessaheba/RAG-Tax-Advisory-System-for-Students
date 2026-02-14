import os
import csv

RAW_DIR = os.path.join(os.path.dirname(__file__), 'data_raw')
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'document_manifest.csv')

folders = ['irs_publications', 'irs_forms', 'treaties', 'university_guides']

def get_all_pdfs():
    pdfs = set()
    for folder in folders:
        folder_path = os.path.join(RAW_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith('.pdf'):
                pdfs.add((folder, fname))
    return pdfs

def get_manifest_entries():
    entries = set()
    with open(MANIFEST_PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            entries.add((row['folder'], row['filename']))
    return entries

def main():
    pdfs = get_all_pdfs()
    manifest = get_manifest_entries()
    missing_in_manifest = pdfs - manifest
    missing_in_files = manifest - pdfs
    if missing_in_manifest:
        print('PDFs missing in manifest:')
        for folder, fname in sorted(missing_in_manifest):
            print(f'  {folder}/{fname}')
    else:
        print('All PDFs are listed in the manifest.')
    if missing_in_files:
        print('Manifest entries missing PDF files:')
        for folder, fname in sorted(missing_in_files):
            print(f'  {folder}/{fname}')
    else:
        print('All manifest entries have matching PDF files.')

if __name__ == "__main__":
    main()
