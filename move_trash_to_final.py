"""
===============================================================================
MOVE TRASHED IMAGES AND CSV TO FINAL FOLDER
===============================================================================

This script:
- Creates a folder called "final"
- Moves all trashed images from trash_images/ to final/
- Moves the trashed CSV (trashed_images_csv_rows.csv) to final/

Usage (from project root):
    python move_trash_to_final.py

===============================================================================
"""

import shutil
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

TRASH_FOLDER = Path("trash_images")
FINAL_FOLDER = Path("final")
TRASHED_CSV = TRASH_FOLDER / "trashed_images_csv_rows.csv"


def main() -> None:
    print("=" * 80)
    print("MOVING TRASHED IMAGES AND CSV TO FINAL FOLDER")
    print("=" * 80)

    if not TRASH_FOLDER.exists():
        print(f"ERROR: Trash folder not found: {TRASH_FOLDER}")
        return

    # Create final folder
    FINAL_FOLDER.mkdir(exist_ok=True)
    print(f"\nCreated/verified folder: {FINAL_FOLDER}/")

    # Count images to move
    image_files = list(TRASH_FOLDER.glob("*.png")) + list(TRASH_FOLDER.glob("*.PNG"))
    # Exclude CSV files
    image_files = [f for f in image_files if f.name not in 
                   ["trashed_images_list.csv", "removed_images_log.csv", "trashed_images_csv_rows.csv"]]
    
    print(f"Found {len(image_files):,} images to move")

    # Move images
    print("\nMoving images...")
    moved_images = 0
    for img_file in image_files:
        try:
            dest = FINAL_FOLDER / img_file.name
            shutil.move(str(img_file), str(dest))
            moved_images += 1
            if moved_images % 500 == 0:
                print(f"  Moved {moved_images:,} images...")
        except Exception as e:
            print(f"  ERROR moving {img_file.name}: {e}")

    print(f"Moved {moved_images:,} images")

    # Move CSV file
    if TRASHED_CSV.exists():
        print(f"\nMoving CSV file: {TRASHED_CSV.name}")
        try:
            dest_csv = FINAL_FOLDER / TRASHED_CSV.name
            shutil.move(str(TRASHED_CSV), str(dest_csv))
            print(f"Moved CSV to: {dest_csv}")
        except Exception as e:
            print(f"ERROR moving CSV: {e}")
    else:
        print(f"\nWARNING: CSV file not found: {TRASHED_CSV}")

    # Summary
    final_images = len([f for f in FINAL_FOLDER.glob("*.png")] + [f for f in FINAL_FOLDER.glob("*.PNG")])
    final_csvs = len(list(FINAL_FOLDER.glob("*.csv")))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Images moved to final/: {moved_images:,}")
    print(f"CSV files in final/: {final_csvs}")
    print(f"Total files in final/: {final_images + final_csvs}")
    print("\nDone!")


if __name__ == "__main__":
    main()

