"""Check why trashed images don't match CSV rows"""
import pandas as pd
from pathlib import Path

# Get trashed images
trash_folder = Path('trash_images')
trashed_images = set([f.name for f in trash_folder.glob("*.png")] + [f.name for f in trash_folder.glob("*.PNG")])
print(f"Total trashed images: {len(trashed_images):,}")

# Load CSV
df = pd.read_csv('final_clean_validated.csv', low_memory=False)
csv_images = set(df['Image Index'].unique())
print(f"Total unique images in CSV: {len(csv_images):,}")

# Check overlap
in_both = trashed_images & csv_images
only_in_trash = trashed_images - csv_images
only_in_csv = csv_images - trashed_images

print(f"\nImages in BOTH trash and CSV: {len(in_both):,}")
print(f"Images ONLY in trash (not in CSV): {len(only_in_trash):,}")
print(f"Images ONLY in CSV (not trashed): {len(only_in_csv):,}")

if len(only_in_trash) > 0:
    print(f"\nFirst 10 images in trash but NOT in CSV:")
    print(list(sorted(only_in_trash))[:10])
    
    # Check if these were already removed from CSV
    print("\nThese images were likely already removed from CSV during the interrupted cleaning run.")

