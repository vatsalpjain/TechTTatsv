"""
===============================================================================
REMOVE DUPLICATE FILES FROM TRASH FOLDER
===============================================================================

This script:
- Scans `trash_images/` for duplicate files (same filename)
- Keeps only one copy of each unique image
- Removes duplicate files
- Updates the trash report CSV

Usage (from project root):
    python remove_trash_duplicates.py

===============================================================================
"""

import os
from collections import defaultdict
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

TRASH_FOLDER = Path("trash_images")
TRASHED_IMAGES_LIST = TRASH_FOLDER / "trashed_images_list.csv"


def main() -> None:
    print("=" * 80)
    print("REMOVING DUPLICATE FILES FROM TRASH FOLDER")
    print("=" * 80)

    if not TRASH_FOLDER.exists():
        print(f"ERROR: Trash folder not found: {TRASH_FOLDER}")
        return

    # Find all PNG files
    print(f"\nScanning {TRASH_FOLDER}/ for PNG files...")
    all_files = list(TRASH_FOLDER.glob("*.png")) + list(TRASH_FOLDER.glob("*.PNG"))
    
    # Skip the CSV log file
    all_files = [f for f in all_files if f.name != "trashed_images_list.csv" 
                 and f.name != "removed_images_log.csv" 
                 and f.name != "trashed_images_csv_rows.csv"]
    
    print(f"Found {len(all_files):,} image files")

    # Group files by name (case-insensitive)
    files_by_name = defaultdict(list)
    for f in all_files:
        files_by_name[f.name.lower()].append(f)

    # Find duplicates
    duplicates = {name: files for name, files in files_by_name.items() if len(files) > 1}
    
    if not duplicates:
        print("\nNo duplicates found! All files are unique.")
        return

    print(f"\nFound {len(duplicates):,} unique filenames with duplicates")
    
    total_duplicates = sum(len(files) - 1 for files in duplicates.values())
    print(f"Total duplicate files to remove: {total_duplicates:,}")

    # Remove duplicates (keep the first one, remove the rest)
    removed_count = 0
    kept_files = []
    
    print("\nRemoving duplicates...")
    for name_lower, files in duplicates.items():
        # Keep the first file, remove the rest
        kept_file = files[0]
        kept_files.append(kept_file.name)
        
        for duplicate_file in files[1:]:
            try:
                duplicate_file.unlink()
                removed_count += 1
                if removed_count % 100 == 0:
                    print(f"  Removed {removed_count:,} duplicates...")
            except Exception as e:
                print(f"  ERROR removing {duplicate_file.name}: {e}")

    print(f"\nRemoved {removed_count:,} duplicate files")
    print(f"Kept {len(kept_files):,} unique files")

    # Update the trash list CSV if it exists
    if TRASHED_IMAGES_LIST.exists():
        print(f"\nUpdating {TRASHED_IMAGES_LIST}...")
        try:
            import pandas as pd
            # Get all remaining unique files
            remaining_files = sorted(set([
                f.name for f in (list(TRASH_FOLDER.glob("*.png")) + list(TRASH_FOLDER.glob("*.PNG")))
                if f.name not in ["trashed_images_list.csv", "removed_images_log.csv", "trashed_images_csv_rows.csv"]
            ]))
            
            list_df = pd.DataFrame({"Image Index": remaining_files})
            list_df.to_csv(TRASHED_IMAGES_LIST, index=False)
            print(f"Updated CSV with {len(remaining_files):,} unique images")
        except Exception as e:
            print(f"WARNING: Could not update CSV: {e}")

    # Summary
    remaining_files = [f for f in TRASH_FOLDER.glob("*.png")] + [f for f in TRASH_FOLDER.glob("*.PNG")]
    remaining_files = [f for f in remaining_files if f.name not in 
                      ["trashed_images_list.csv", "removed_images_log.csv", "trashed_images_csv_rows.csv"]]
    unique_remaining = len(set(f.name.lower() for f in remaining_files))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Duplicate files removed: {removed_count:,}")
    print(f"Unique files remaining: {unique_remaining:,}")
    print(f"Total files remaining: {len(remaining_files):,}")
    print("\nDone!")


if __name__ == "__main__":
    main()

