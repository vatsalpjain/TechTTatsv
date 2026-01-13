"""
===============================================================================
RECOVER TRASHED IMAGES FROM ORIGINAL SOURCE FOLDERS
===============================================================================

This script:
- Reads the list of trashed images from trash_images/trashed_images_csv_rows.csv
- Searches for those images in the original source folders (images_001/images/, etc.)
- Copies them back to trash_images/ folder

Usage (from project root):
    python recover_trashed_images.py

===============================================================================
"""

import os
import shutil
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

TRASHED_CSV = "trash_images/trashed_images_csv_rows.csv"
IMAGE_INDEX_COLUMN = "Image Index"

# Original source folders (where images were copied FROM)
SOURCE_FOLDERS = [
    "images_001/images",
    "images_002/images",
    "images_003/images",
    "images_004/images",
    "images_005/images",
    "images_006/images",
]

TRASH_FOLDER = Path("trash_images")


def find_image_in_sources(image_name: str, source_folders: list) -> Path | None:
    """Search for image in source folders"""
    for folder in source_folders:
        if not os.path.exists(folder):
            continue
        # Try both lowercase and uppercase extensions
        for ext in [".png", ".PNG"]:
            img_path = Path(folder) / image_name
            if img_path.exists():
                return img_path
    return None


def main() -> None:
    print("=" * 80)
    print("RECOVERING TRASHED IMAGES FROM ORIGINAL SOURCE FOLDERS")
    print("=" * 80)

    # Load list of trashed images
    if not os.path.exists(TRASHED_CSV):
        print(f"ERROR: {TRASHED_CSV} not found!")
        return

    print(f"\nLoading trashed images list from: {TRASHED_CSV}")
    df_trashed = pd.read_csv(TRASHED_CSV, low_memory=False)
    trashed_images = df_trashed[IMAGE_INDEX_COLUMN].unique().tolist()
    print(f"Found {len(trashed_images):,} unique trashed images to recover")

    # Check source folders
    print("\nChecking source folders...")
    existing_folders = []
    for folder in SOURCE_FOLDERS:
        if os.path.exists(folder):
            existing_folders.append(folder)
            file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            print(f"  {folder}: {file_count:,} files")
        else:
            print(f"  {folder}: NOT FOUND")

    if not existing_folders:
        print("\nERROR: No source folders found!")
        return

    # Recover images
    print(f"\nRecovering images...")
    recovered = 0
    not_found = 0
    already_exists = 0

    TRASH_FOLDER.mkdir(exist_ok=True)

    for idx, image_name in enumerate(trashed_images, 1):
        if idx % 100 == 0:
            print(f"  Progress: {idx:,}/{len(trashed_images):,} (recovered: {recovered:,}, not found: {not_found:,})")

        # Check if already in trash folder
        dest_path = TRASH_FOLDER / image_name
        if dest_path.exists():
            already_exists += 1
            continue

        # Find image in source folders
        source_path = find_image_in_sources(image_name, existing_folders)
        if source_path is None:
            not_found += 1
            continue

        # Copy to trash folder
        try:
            shutil.copy2(source_path, dest_path)
            recovered += 1
        except Exception as e:
            print(f"  ERROR copying {image_name}: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total trashed images: {len(trashed_images):,}")
    print(f"Recovered from source: {recovered:,}")
    print(f"Already in trash: {already_exists:,}")
    print(f"Not found in sources: {not_found:,}")
    print("\nDone!")


if __name__ == "__main__":
    main()

