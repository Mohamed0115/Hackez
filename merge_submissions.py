import json
import os

# 1. PATHS
ORIGINAL_FILE = 'final_submission.json'
RECOVERY_FILE = 'recovered_data.json'
OUTPUT_FILE   = 'FINAL_MASTER_SUBMISSION.json'

def merge_safely():
    # Load original results
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Load the recovered results
    with open(RECOVERY_FILE, 'r', encoding='utf-8') as f:
        recovered_data = json.load(f)

    # Create a map for quick replacement
    recovery_map = {entry['filename']: entry for entry in recovered_data}
    
    final_list = []
    patched_count = 0

    for entry in original_data:
        fname = entry['filename']
        # If we have a 'Good' version in recovery, use it.
        # Otherwise, keep the original.
        if fname in recovery_map:
            final_list.append(recovery_map[fname])
            patched_count += 1
        else:
            final_list.append(entry)

    # Save to a COMPLETELY NEW FILE
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=4)
    
    print(f"✅ Success! Created {OUTPUT_FILE}")
    print(f"📊 Total images in file: {len(final_list)}")
    print(f"🛠️  Images patched with recovered data: {patched_count}")

if __name__ == "__main__":
    merge_safely()
