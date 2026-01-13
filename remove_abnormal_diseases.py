"""
Remove abnormal disease entries from final_clean.csv
Removes: XYZ_Disease and 123
Also removes corresponding images from referenced_images/ folder
"""

import pandas as pd
import os
from datetime import datetime

print("=" * 80)
print("REMOVING ABNORMAL DISEASE ENTRIES")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load dataset
print("Loading final_clean.csv...")
df = pd.read_csv('final_clean.csv')
initial_count = len(df)
print(f"✓ Loaded: {initial_count:,} rows\n")

# Define diseases to remove
diseases_to_remove = ['XYZ_Disease', '123']

print("-" * 80)
print("BEFORE REMOVAL:")
print("-" * 80)
for disease in diseases_to_remove:
    count = len(df[df['diseases'] == disease])
    print(f'  "{disease}": {count:,} rows')

total_to_remove = len(df[df['diseases'].isin(diseases_to_remove)])
print(f"\nTotal rows to remove: {total_to_remove:,}")

# Get list of images to remove before filtering
print("\n" + "-" * 80)
print("IDENTIFYING IMAGES TO REMOVE...")
print("-" * 80)

rows_to_remove = df[df['diseases'].isin(diseases_to_remove)]
images_to_remove = rows_to_remove['Image Index'].unique().tolist()

print(f"✓ Found {len(images_to_remove):,} unique images to remove from referenced_images/")

# Remove rows with abnormal diseases
print("\n" + "-" * 80)
print("REMOVING ROWS FROM CSV...")
print("-" * 80)

df_clean = df[~df['diseases'].isin(diseases_to_remove)].copy()
removed_count = initial_count - len(df_clean)

print(f"✓ Removed: {removed_count:,} rows")
print(f"✓ Remaining: {len(df_clean):,} rows")
print(f"✓ Retention rate: {(len(df_clean)/initial_count)*100:.2f}%")

# Remove corresponding images from referenced_images folder
print("\n" + "-" * 80)
print("REMOVING IMAGES FROM referenced_images/ FOLDER...")
print("-" * 80)

referenced_images_folder = 'referenced_images'
images_removed = 0
images_not_found = 0

if os.path.exists(referenced_images_folder):
    for image_name in images_to_remove:
        image_path = os.path.join(referenced_images_folder, image_name)
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                images_removed += 1
            except Exception as e:
                print(f"  ⚠ Could not remove {image_name}: {e}")
        else:
            images_not_found += 1
    
    print(f"✓ Images removed: {images_removed:,}")
    if images_not_found > 0:
        print(f"  (Images not found in folder: {images_not_found:,})")
else:
    print(f"⚠ Folder not found: {referenced_images_folder}")

# Save cleaned dataset
output_file = 'final_clean_validated.csv'
print("\n" + "-" * 80)
print("SAVING CLEANED DATASET...")
print("-" * 80)

df_clean.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")

# Generate summary statistics
print("\n" + "=" * 80)
print("FINAL STATISTICS:")
print("=" * 80)

print(f"\nDataset Summary:")
print(f"  Original rows: {initial_count:,}")
print(f"  Removed rows: {removed_count:,}")
print(f"  Final rows: {len(df_clean):,}")

print(f"\nUnique values:")
print(f"  Unique images: {df_clean['Image Index'].nunique():,}")
print(f"  Unique patients: {df_clean['Patient ID'].nunique():,}")

print(f"\nGender distribution:")
for gender, count in df_clean['Patient Gender'].value_counts(dropna=False).items():
    print(f"  {gender}: {count:,}")

print(f"\nTop 10 diseases (after cleaning):")
for disease, count in df_clean['diseases'].value_counts().head(10).items():
    pct = (count / len(df_clean)) * 100
    print(f"  {disease}: {count:,} ({pct:.1f}%)")

# Also update found_and_valid_images.csv
print("\n" + "-" * 80)
print("UPDATING IMAGE VALIDATION REPORT...")
print("-" * 80)

try:
    df_images = pd.read_csv('found_and_valid_images.csv')
    initial_image_count = len(df_images)
    
    # Get image indices from cleaned dataset
    valid_images = set(df_clean['Image Index'].unique())
    
    # Filter image report
    df_images_clean = df_images[df_images['Image Index'].isin(valid_images)]
    
    df_images_clean.to_csv('found_and_valid_images_cleaned.csv', index=False)
    
    print(f"✓ Original image report: {initial_image_count:,} rows")
    print(f"✓ Cleaned image report: {len(df_images_clean):,} rows")
    print(f"✓ Saved: found_and_valid_images_cleaned.csv")
except Exception as e:
    print(f"⚠ Could not update image report: {e}")

print("\n" + "=" * 80)
print("🎉 CLEANING COMPLETE!")
print("=" * 80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput files:")
print(f"  • {output_file} - Main cleaned dataset")
print(f"  • found_and_valid_images_cleaned.csv - Updated image report")
print("=" * 80)
