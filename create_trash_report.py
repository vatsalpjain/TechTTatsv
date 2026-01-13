"""
===============================================================================
CREATE TRASH REPORT - CSV of Trashed Images
===============================================================================

This script:
- Scans `trash_images/` folder for all PNG images
- Creates a CSV listing all trashed image names
- Extracts rows from `final_clean_validated.csv` that correspond to trashed images
- Saves both CSVs for review

Usage (from project root):
    python create_trash_report.py

Output:
    - trash_images/trashed_images_list.csv (simple list of image names)
    - trash_images/trashed_images_csv_rows.csv (full CSV rows for trashed images)
===============================================================================
"""

import os
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

CSV_FILE = "final_clean_validated.csv"
IMAGE_INDEX_COLUMN = "Image Index"
TRASH_FOLDER = Path("trash_images")

# Output files
TRASHED_IMAGES_LIST = TRASH_FOLDER / "trashed_images_list.csv"
TRASHED_IMAGES_CSV_ROWS = TRASH_FOLDER / "trashed_images_csv_rows.csv"


def main() -> None:
    print("=" * 80)
    print("CREATING TRASH REPORT")
    print("=" * 80)

    # Check trash folder exists
    if not TRASH_FOLDER.exists():
        print(f"ERROR: Trash folder not found: {TRASH_FOLDER}")
        return

    # Find all PNG images in trash
    print(f"\nScanning {TRASH_FOLDER}/ for PNG images...")
    trashed_images = sorted(
        [f.name for f in TRASH_FOLDER.glob("*.png")]
        + [f.name for f in TRASH_FOLDER.glob("*.PNG")]
    )

    if not trashed_images:
        print("WARNING: No images found in trash folder.")
        return

    print(f"Found {len(trashed_images):,} trashed images")

    # Create simple list CSV
    print(f"\nCreating image list CSV: {TRASHED_IMAGES_LIST}")
    list_df = pd.DataFrame({"Image Index": trashed_images})
    list_df.to_csv(TRASHED_IMAGES_LIST, index=False)
    print(f"Saved {len(list_df):,} image names to {TRASHED_IMAGES_LIST}")

    # Load main CSV and extract rows for trashed images
    if not os.path.exists(CSV_FILE):
        print(f"\nWARNING: {CSV_FILE} not found. Skipping CSV row extraction.")
        return

    print(f"\nLoading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False)
        print(f"Loaded {len(df):,} rows from CSV")

        # Extract rows where Image Index matches trashed images
        trashed_set = set(trashed_images)
        trashed_rows = df[df[IMAGE_INDEX_COLUMN].isin(trashed_set)].copy()

        print(f"\nFound {len(trashed_rows):,} CSV rows matching trashed images")

        if len(trashed_rows) > 0:
            print(f"Saving CSV rows to: {TRASHED_IMAGES_CSV_ROWS}")
            trashed_rows.to_csv(TRASHED_IMAGES_CSV_ROWS, index=False)
            print(f"Saved {len(trashed_rows):,} CSV rows to {TRASHED_IMAGES_CSV_ROWS}")
        else:
            print(
                "WARNING: No matching CSV rows found. "
                "This might mean the CSV was already updated to remove these references."
            )

        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total trashed images: {len(trashed_images):,}")
        print(f"CSV rows found for trashed images: {len(trashed_rows):,}")
        print(f"Images without CSV entries: {len(trashed_images) - len(trashed_rows):,}")
        print("\nOutput files:")
        print(f"  - {TRASHED_IMAGES_LIST}")
        print(f"  - {TRASHED_IMAGES_CSV_ROWS}")

    except Exception as e:
        print(f"\nERROR processing CSV: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

