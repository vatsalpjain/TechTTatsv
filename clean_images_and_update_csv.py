"""
===============================================================================
IMAGE CLEANING FOR MODEL TRAINING
===============================================================================

This script:
- Scans `referenced_images/` for PNG images
- Applies simple automatic checks to flag "dirty" images
  (bad geometry, nearly blank, extremely low contrast, unreadable, etc.)
- Moves dirty images into `trash_images/` (non-destructive delete)
- Updates `final_clean_validated.csv` to drop any rows whose `Image Index`
  has been moved to trash, so the CSV always matches the training set.

The checks are intentionally conservative; you can tune the thresholds below.

Usage (from project root):
    python clean_images_and_update_csv.py

Requirements:
    - pandas
    - pillow
    - numpy
===============================================================================
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

CSV_FILE = "final_clean_validated.csv"
IMAGE_INDEX_COLUMN = "Image Index"

# Folder containing the already validated & copied images
IMAGE_FOLDER = Path("referenced_images")

# Where to move images that we decide to remove
TRASH_FOLDER = Path("trash_images")

# Geometry thresholds
MIN_SIDE = 512  # minimum allowed width/height in pixels
MAX_ASPECT_RATIO = 1.2  # allow up to 20% deviation from square

# Intensity / contrast thresholds (0–255 grayscale)
MIN_STD = 8.0  # very low contrast images are likely blank / bad
MIN_MEAN = 5.0
MAX_MEAN = 250.0

# Batch size for progress reporting (does not change correctness)
BATCH_SIZE = 2_000


def is_image_dirty(path: Path) -> tuple[bool, str]:
    """
    Heuristic checks to decide whether an image is "dirty".

    Returns:
        (is_dirty, reason)
    """
    # Must be PNG
    if path.suffix.lower() != ".png":
        return True, f"Invalid extension: {path.suffix}"

    # Try opening and basic stats
    try:
        with Image.open(path) as img:
            img = img.convert("L")  # ensure grayscale
            w, h = img.size
            arr = np.array(img)
    except (UnidentifiedImageError, OSError) as e:
        return True, f"Unreadable/corrupted: {e}"
    except Exception as e:  # any unexpected error
        return True, f"Open error: {e}"

    # Geometry checks
    if min(w, h) < MIN_SIDE:
        return True, f"Too small: {w}x{h}"

    aspect = max(w, h) / min(w, h)
    if aspect > MAX_ASPECT_RATIO:
        return True, f"Odd aspect ratio: {aspect:.2f}"

    # Intensity / contrast checks (cheap and usually enough to catch
    # almost-blank or completely blown-out images like 00000032_023).
    mean_val = float(arr.mean())
    std_val = float(arr.std())

    if std_val < MIN_STD:
        return True, f"Too low contrast (std={std_val:.2f})"
    if mean_val < MIN_MEAN or mean_val > MAX_MEAN:
        return True, f"Mean too extreme (mean={mean_val:.2f})"

    # If all checks passed, consider image clean
    return False, ""


def main() -> None:
    if not IMAGE_FOLDER.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    TRASH_FOLDER.mkdir(exist_ok=True)

    print("Loading CSV:", CSV_FILE)
    df = pd.read_csv(CSV_FILE)

    # Keep track of images we move to trash
    removed_images: list[str] = []
    reasons: list[str] = []

    image_paths = sorted(IMAGE_FOLDER.glob("*.png")) + sorted(
        IMAGE_FOLDER.glob("*.PNG")
    )

    total = len(image_paths)
    print(f"Found {total:,} images in {IMAGE_FOLDER}/")

    # Backup CSV once at start (only if backup doesn't exist)
    backup_csv = CSV_FILE.replace(".csv", "_before_image_cleaning_backup.csv")
    if not os.path.exists(backup_csv):
        df.to_csv(backup_csv, index=False)
        print(f"Original CSV backed up as: {backup_csv}")

    start_time = time.time()
    # Process images in batches for clearer progress reporting on large datasets
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = image_paths[start:end]
        done = start

        elapsed = time.time() - start_time
        # Estimate remaining time based on processed so far
        if done > 0:
            sec_per_image = elapsed / done
            remaining_images = total - done
            eta_sec = sec_per_image * remaining_images
        else:
            eta_sec = 0.0

        def _fmt(sec: float) -> str:
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h}h {m}m {s}s"
            if m:
                return f"{m}m {s}s"
            return f"{s}s"

        print(
            f"Processing images {start + 1:,}–{end:,}/{total:,} "
            f"(elapsed: {_fmt(elapsed)}, ETA: {_fmt(eta_sec)})..."
        )

        batch_removed = []
        batch_reasons = []
        
        for img_path in batch:
            dirty, reason = is_image_dirty(img_path)
            if not dirty:
                continue

            # Move dirty image to trash (non-destructive)
            dest = TRASH_FOLDER / img_path.name
            img_path.rename(dest)

            batch_removed.append(img_path.name)
            batch_reasons.append(reason)
            removed_images.append(img_path.name)
            reasons.append(reason)

        # Save progress incrementally after each batch
        if batch_removed:
            # Update log file incrementally
            log_path = TRASH_FOLDER / "removed_images_log.csv"
            if log_path.exists():
                existing_log = pd.read_csv(log_path)
                new_log = pd.DataFrame({
                    "Image Index": batch_removed,
                    "reason": batch_reasons,
                })
                log_df = pd.concat([existing_log, new_log], ignore_index=True)
            else:
                log_df = pd.DataFrame({
                    "Image Index": batch_removed,
                    "reason": batch_reasons,
                })
            log_df.to_csv(log_path, index=False)

            # Update CSV incrementally (remove rows for images moved in this batch)
            df = df[~df[IMAGE_INDEX_COLUMN].isin(batch_removed)].copy()
            df.to_csv(CSV_FILE, index=False)
            print(f"  → Moved {len(batch_removed):,} dirty images, CSV updated")

    print(f"\nTotal dirty images moved to trash: {len(removed_images):,}")

    if removed_images:
        # Final summary
        log_path = TRASH_FOLDER / "removed_images_log.csv"
        print(f"Details saved to: {log_path}")

        before_rows = pd.read_csv(backup_csv).shape[0]
        after_rows = len(df)
        print(
            f"Final {CSV_FILE}: {before_rows:,} -> {after_rows:,} rows "
            f"(removed {before_rows - after_rows:,} references)."
        )
    else:
        print("No dirty images detected; CSV left unchanged.")


if __name__ == "__main__":
    main()


