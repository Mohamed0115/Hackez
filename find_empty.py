import json

def find_empty_images(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    skipped_filenames = []
    
    for entry in data:
        # Check if tables list is completely empty
        tables = entry.get('tables', [])
        if not tables:
            skipped_filenames.append(entry['filename'])
            continue
            
        # Check if all tables exist but have no text cells or no cells at all
        all_cells_blank = True
        for table in tables:
            cells = table.get('cells', [])
            if cells:
                for cell in cells:
                    # If any cell has text that is not just whitespace, it's not completely blank
                    if cell.get('text', '').strip():
                        all_cells_blank = False
                        break
            if not all_cells_blank:
                break
                
        if all_cells_blank:
            skipped_filenames.append(entry['filename'])

    print(f"🔍 Found {len(skipped_filenames)} skipped/blank images out of {len(data)} total images.")
    if skipped_filenames:
        print(f"List of skipped/blank filenames:")
        for name in skipped_filenames:
            print(f"  - {name}")
    
    # Let's save the skipped filenames to a text file for easy access
    with open('skipped_images.txt', 'w', encoding='utf-8') as f:
        for name in skipped_filenames:
            f.write(f"{name}\n")
            
    print(f"\n✅ Created 'skipped_images.txt' with the list.")

if __name__ == '__main__':
    find_empty_images('final_submission.json')
