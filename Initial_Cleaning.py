# ========================================================================
# COMPLETE DATA CLEANING PIPELINE
# Creates: clean_entry.csv (Clean Data+BBox) and flagged.csv (Dirty Data+BBox)
# ========================================================================

# STEP 1: Install and import required packages
# ---------------------------------------------
!pip install gdown -q

import pandas as pd
import numpy as np
import gdown
import os
from google.colab import drive
import re
from collections import defaultdict

# STEP 2: Mount Google Drive
# ---------------------------
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Create output folders
os.makedirs('/content/cleaned_data', exist_ok=True)
os.makedirs('/content/flagged_data', exist_ok=True)
os.makedirs('/content/logs', exist_ok=True)

print("✓ Setup complete")

# STEP 3: Download CSV files
# ---------------------------
print("\n" + "="*60)
print("DOWNLOADING CSV FILES")
print("="*60)

csv_links = [
    ("https://drive.google.com/file/d/1H6UARor5aXx4_tHACWxE3d4sBY5d3pJF/view?usp=drive_link", "bbox_2.csv"),
    ("https://drive.google.com/file/d/15xqxut8ri1bGc8nMJi7dxv89Xf9QU3BO/view?usp=drive_link", "data_entry_.csv"),
    ("https://drive.google.com/file/d/1KwggxzXcv7DQSH4JN4cTsXooCU1Rehz0/view?usp=drive_link", "data_entry.csv"),
    ("https://drive.google.com/file/d/15RZxzpxgK-V8jYMWb8kzlfJm012KbOxP/view?usp=drive_link", "HACKATHON_CORRUPTED_bbox_list.csv"),
    ("https://drive.google.com/file/d/1Q32l5pH-Fu0JSe8EbVCDIe5xES2Fmo1s/view?usp=drive_link", "HACKATHON_CORRUPTED_data_entry.csv"),
]

def extract_file_id(link):
    if '/file/d/' in link:
        return link.split('/file/d/')[1].split('/')[0]
    return link

def download_file(link, output_name):
    file_id = extract_file_id(link)
    download_url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading {output_name}...")
    gdown.download(download_url, output_name, quiet=False)
    print(f"✓ Downloaded {output_name}")

for link, filename in csv_links:
    try:
        download_file(link, f'/content/{filename}')
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")

# STEP 4: Load all CSV files
# ---------------------------
print("\n" + "="*60)
print("LOADING CSV FILES")
print("="*60)

data_entry = pd.read_csv('/content/data_entry.csv')
print(f"✓ Loaded data_entry.csv: {data_entry.shape}")
print(f"  Columns: {list(data_entry.columns)}")

try:
    data_entry_small = pd.read_csv('/content/data_entry_.csv')
    print(f"✓ Loaded data_entry_.csv (small): {data_entry_small.shape}")
except:
    data_entry_small = None

bbox_2 = pd.read_csv('/content/bbox_2.csv')
print(f"✓ Loaded bbox_2.csv: {bbox_2.shape}")
print(f"  Columns: {list(bbox_2.columns)}")

bbox_corrupted = pd.read_csv('/content/HACKATHON_CORRUPTED_bbox_list.csv')
print(f"✓ Loaded HACKATHON_CORRUPTED_bbox_list.csv: {bbox_corrupted.shape}")
print(f"  Columns: {list(bbox_corrupted.columns)}")

data_entry_corrupted = pd.read_csv('/content/HACKATHON_CORRUPTED_data_entry.csv')
print(f"✓ Loaded HACKATHON_CORRUPTED_data_entry.csv: {data_entry_corrupted.shape}")
print(f"  Columns: {list(data_entry_corrupted.columns)}")

# Normalize column names function
def normalize_columns(df):
    """Standardize column names across different files"""
    col_map = {
        # Image Index variations
        'image index': 'Image Index',
        'imageindex': 'Image Index',
        'image_index': 'Image Index',
        
        # Patient Age variations
        'patient age': 'Patient Age',
        'patientage': 'Patient Age',
        'patient_age': 'Patient Age',
        'age': 'Patient Age',
        
        # Patient Gender variations
        'patient gender': 'Patient Gender',
        'patientgender': 'Patient Gender',
        'patient_gender': 'Patient Gender',
        'gender': 'Patient Gender',
        
        # Finding Labels variations
        'finding labels': 'Finding Labels',
        'findinglabels': 'Finding Labels',
        'finding_labels': 'Finding Labels',
        'labels': 'Finding Labels',
        
        # Finding Label (singular - for bbox)
        'finding label': 'Finding Label',
        'findinglabel': 'Finding Label',
        'finding_label': 'Finding Label',
        'label': 'Finding Label',
        
        # Image dimensions
        'originalimage[width': 'OriginalImage[Width',
        'originalimage[height': 'OriginalImage[Height',
        'originalimagewidth': 'OriginalImage[Width',
        'originalimageheight': 'OriginalImage[Height',
        'width': 'OriginalImage[Width',
        'height': 'OriginalImage[Height',
        
        # Bbox coordinates
        'bbox_x': 'x',
        'bbox_y': 'y',
        'bbox_w': 'w',
        'bbox_h': 'h',
        'bboxx': 'x',
        'bboxy': 'y',
        'bboxw': 'w',
        'bboxh': 'h',
    }
    
    # Create lowercase mapping
    df.columns = df.columns.str.strip()
    new_cols = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in col_map:
            new_cols[col] = col_map[col_lower]
        else:
            new_cols[col] = col
    
    df = df.rename(columns=new_cols)
    return df

# Normalize all dataframes
print("\n🔧 Normalizing column names...")
data_entry = normalize_columns(data_entry)
bbox_2 = normalize_columns(bbox_2)
bbox_corrupted = normalize_columns(bbox_corrupted)
data_entry_corrupted = normalize_columns(data_entry_corrupted)

print("✓ Column normalization complete")
print(f"\ndata_entry columns: {list(data_entry.columns)}")
print(f"bbox_2 columns: {list(bbox_2.columns)}")

# STEP 5: Initialize tracking logs
# ---------------------------------
cleaning_log = {
    'data_entry_issues': [],
    'bbox_issues': [],
    'merged_issues': []
}

# ========================================================================
# PART A: CLEAN DATA_ENTRY (PRIMARY SOURCE OF TRUTH)
# ========================================================================

print("\n" + "="*70)
print("PART A: CLEANING DATA_ENTRY (PRIMARY SOURCE OF TRUTH)")
print("="*70)

def clean_data_entry_primary(df):
    """Clean the main data_entry.csv"""
    df = df.copy()
    initial_count = len(df)
    flagged_rows = []
    
    print(f"\nStarting with {initial_count} rows")
    
    # Issue 1: Missing/Fake images
    print("\n1️⃣ Checking for missing/fake images...")
    missing_patterns = ['missing', 'Missing', 'mock', 'fake', 'Missing_file']
    mask_missing = df['Image Index'].str.contains('|'.join(missing_patterns), case=False, na=False)
    flagged = df[mask_missing].copy()
    flagged['flag_reason'] = 'missing_or_fake_image'
    flagged_rows.append(flagged)
    df = df[~mask_missing]
    print(f"   Flagged: {mask_missing.sum()} missing/fake images")
    
    # Issue 2: Clean Patient Age
    print("\n2️⃣ Cleaning Patient Age...")
    if 'Patient Age' in df.columns:
        # Convert age format (058Y → 58, ?? → NaN)
        df['Patient Age'] = df['Patient Age'].astype(str)
        df['Patient Age'] = df['Patient Age'].str.replace('Y', '').str.replace('?', '').str.strip()
        
        # Flag invalid ages before conversion
        mask_invalid_age = ~df['Patient Age'].str.match(r'^\d+$', na=False)
        flagged = df[mask_invalid_age].copy()
        flagged['flag_reason'] = 'invalid_age_format'
        flagged_rows.append(flagged)
        
        # Convert to numeric
        df['Patient Age'] = pd.to_numeric(df['Patient Age'], errors='coerce')
        
        # Flag unrealistic ages
        mask_unrealistic = (df['Patient Age'] < 0) | (df['Patient Age'] > 120) | df['Patient Age'].isna()
        flagged = df[mask_unrealistic & ~mask_invalid_age].copy()  # Don't double-flag
        flagged['flag_reason'] = 'unrealistic_age'
        flagged_rows.append(flagged)
        
        df = df[~mask_unrealistic]
        print(f"   Flagged: {(mask_invalid_age | mask_unrealistic).sum()} invalid ages")
    
    # Issue 3: Validate image dimensions
    print("\n3️⃣ Validating image dimensions...")
    if 'OriginalImage[Width' in df.columns and 'OriginalImage[Height' in df.columns:
        mask_invalid_dims = (
            (df['OriginalImage[Width'] <= 0) | 
            (df['OriginalImage[Height'] <= 0) |
            df['OriginalImage[Width'].isna() |
            df['OriginalImage[Height'].isna()
        )
        flagged = df[mask_invalid_dims].copy()
        flagged['flag_reason'] = 'invalid_dimensions'
        flagged_rows.append(flagged)
        df = df[~mask_invalid_dims]
        print(f"   Flagged: {mask_invalid_dims.sum()} invalid dimensions")
    
    # Issue 4: Validate Finding Labels (not empty)
    print("\n4️⃣ Validating Finding Labels...")
    if 'Finding Labels' in df.columns:
        mask_no_labels = df['Finding Labels'].isna() | (df['Finding Labels'].str.strip() == '')
        flagged = df[mask_no_labels].copy()
        flagged['flag_reason'] = 'no_finding_labels'
        flagged_rows.append(flagged)
        df = df[~mask_no_labels]
        print(f"   Flagged: {mask_no_labels.sum()} rows with no labels")
        
        # Clean up labels
        df['Finding Labels'] = df['Finding Labels'].str.strip()
    
    # Issue 5: Validate Gender
    print("\n5️⃣ Validating Gender...")
    if 'Patient Gender' in df.columns:
        mask_invalid_gender = ~df['Patient Gender'].isin(['M', 'F'])
        flagged = df[mask_invalid_gender].copy()
        flagged['flag_reason'] = 'invalid_gender'
        flagged_rows.append(flagged)
        df = df[~mask_invalid_gender]
        print(f"   Flagged: {mask_invalid_gender.sum()} invalid gender values")
    
    # Combine all flagged rows
    all_flagged = pd.concat(flagged_rows, ignore_index=True) if flagged_rows else pd.DataFrame()
    
    print(f"\n✅ CLEAN: {len(df)} rows ({len(df)/initial_count*100:.1f}%)")
    print(f"🚩 FLAGGED: {len(all_flagged)} rows ({len(all_flagged)/initial_count*100:.1f}%)")
    
    return df, all_flagged

data_entry_clean, data_entry_flagged = clean_data_entry_primary(data_entry)

# ========================================================================
# PART B: CLEAN BOUNDING BOXES
# ========================================================================

print("\n" + "="*70)
print("PART B: CLEANING BOUNDING BOXES")
print("="*70)

def clean_bbox_table(bbox_df, valid_images_df, source_name):
    """Clean bounding box data"""
    df = bbox_df.copy()
    initial_count = len(df)
    flagged_rows = []
    
    print(f"\n{source_name}: Starting with {initial_count} boxes")
    
    # Check if required columns exist
    required_cols = ['Image Index']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Missing columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return pd.DataFrame(), df.copy()  # Return empty clean, all flagged
    
    # Check what dimension columns we have
    has_bbox_coords = all(col in df.columns for col in ['x', 'y', 'w', 'h'])
    
    # Create image metadata lookup only with available columns
    meta_cols = ['Image Index']
    if 'OriginalImage[Width' in valid_images_df.columns:
        meta_cols.append('OriginalImage[Width')
    if 'OriginalImage[Height' in valid_images_df.columns:
        meta_cols.append('OriginalImage[Height')
    if 'Finding Labels' in valid_images_df.columns:
        meta_cols.append('Finding Labels')
    
    image_meta = valid_images_df[meta_cols].set_index('Image Index').to_dict('index')
    
    # Issue 1: Missing/Fake images
    print(f"1️⃣ Checking for missing/fake images...")
    missing_patterns = ['missing', 'Missing', 'mock', 'fake']
    mask_missing = df['Image Index'].str.contains('|'.join(missing_patterns), case=False, na=False)
    flagged = df[mask_missing].copy()
    flagged['flag_reason'] = f'{source_name}_missing_image'
    flagged_rows.append(flagged)
    df = df[~mask_missing]
    print(f"   Flagged: {mask_missing.sum()}")
    
    # Issue 2: Invalid dimensions (only if columns exist)
    if has_bbox_coords:
        print(f"2️⃣ Checking for invalid box dimensions...")
        mask_invalid_dims = (
            (df['w'] == 9999) | (df['w'] == -1) | (df['w'] <= 0) |
            (df['h'] == 9999) | (df['h'] == -1) | (df['h'] <= 0) |
            df['w'].isna() | df['h'].isna()
        )
        flagged = df[mask_invalid_dims].copy()
        flagged['flag_reason'] = f'{source_name}_invalid_box_dimensions'
        flagged_rows.append(flagged)
        df = df[~mask_invalid_dims]
        print(f"   Flagged: {mask_invalid_dims.sum()}")
        
        # Issue 3: Invalid coordinates
        print(f"3️⃣ Checking for invalid coordinates...")
        mask_invalid_coords = (df['x'] < 0) | (df['y'] < 0) | df['x'].isna() | df['y'].isna()
        flagged = df[mask_invalid_coords].copy()
        flagged['flag_reason'] = f'{source_name}_invalid_coordinates'
        flagged_rows.append(flagged)
        df = df[~mask_invalid_coords]
        print(f"   Flagged: {mask_invalid_coords.sum()}")
        
        # Issue 4: Boxes outside image bounds (only if we have image dimensions)
        if 'OriginalImage[Width' in meta_cols and 'OriginalImage[Height' in meta_cols:
            print(f"4️⃣ Checking if boxes fit inside images...")
            out_of_bounds = []
            valid_boxes = []
            
            for idx, row in df.iterrows():
                img_name = row['Image Index']
                if img_name in image_meta:
                    if 'OriginalImage[Width' in image_meta[img_name] and 'OriginalImage[Height' in image_meta[img_name]:
                        img_width = image_meta[img_name]['OriginalImage[Width']
                        img_height = image_meta[img_name]['OriginalImage[Height']
                        
                        if (row['x'] + row['w'] > img_width) or (row['y'] + row['h'] > img_height):
                            out_of_bounds.append(idx)
                        else:
                            valid_boxes.append(idx)
                    else:
                        valid_boxes.append(idx)
                else:
                    out_of_bounds.append(idx)  # Image not in valid set
            
            if out_of_bounds:
                flagged = df.loc[out_of_bounds].copy()
                flagged['flag_reason'] = f'{source_name}_box_out_of_bounds'
                flagged_rows.append(flagged)
            df = df.loc[valid_boxes] if valid_boxes else pd.DataFrame()
            print(f"   Flagged: {len(out_of_bounds)}")
    else:
        print(f"⚠️  Skipping bbox coordinate validation (columns not found)")
    
    # Issue 5: Finding Label mismatch (only if both columns exist)
    if 'Finding Label' in df.columns and 'Finding Labels' in meta_cols:
        print(f"5️⃣ Checking Finding Label consistency...")
        mismatch = []
        valid_labels = []
        
        for idx, row in df.iterrows():
            img_name = row['Image Index']
            if img_name in image_meta and 'Finding Labels' in image_meta[img_name]:
                img_labels = image_meta[img_name]['Finding Labels']
                box_label = row['Finding Label']
                
                # Check if box label exists in image labels
                if pd.notna(box_label) and pd.notna(img_labels):
                    if box_label not in str(img_labels):
                        mismatch.append(idx)
                    else:
                        valid_labels.append(idx)
                else:
                    valid_labels.append(idx)
            else:
                valid_labels.append(idx)
        
        if mismatch:
            flagged = df.loc[mismatch].copy()
            flagged['flag_reason'] = f'{source_name}_label_mismatch'
            flagged_rows.append(flagged)
        df = df.loc[valid_labels] if valid_labels else pd.DataFrame()
        print(f"   Flagged: {len(mismatch)}")
    
    # Combine all flagged
    all_flagged = pd.concat(flagged_rows, ignore_index=True) if flagged_rows else pd.DataFrame()
    
    print(f"\n✅ CLEAN: {len(df)} boxes ({len(df)/initial_count*100:.1f}%)")
    print(f"🚩 FLAGGED: {len(all_flagged)} boxes ({len(all_flagged)/initial_count*100:.1f}%)")
    
    return df, all_flagged

# Clean bbox_2
print("\n" + "-"*70)
print("CLEANING bbox_2.csv")
print("-"*70)
bbox_2_clean, bbox_2_flagged = clean_bbox_table(bbox_2, data_entry_clean, 'bbox_2')

# Clean corrupted bbox
print("\n" + "-"*70)
print("CLEANING HACKATHON_CORRUPTED_bbox_list.csv")
print("-"*70)
bbox_corrupted_clean, bbox_corrupted_flagged = clean_bbox_table(bbox_corrupted, data_entry_clean, 'bbox_corrupted')

# ========================================================================
# PART C: SALVAGE DATA FROM CORRUPTED DATA_ENTRY
# ========================================================================

print("\n" + "="*70)
print("PART C: SALVAGING DATA FROM CORRUPTED DATA_ENTRY")
print("="*70)

def salvage_corrupted_data_entry(corrupted_df, clean_df):
    """Try to salvage good rows from corrupted data_entry"""
    df = corrupted_df.copy()
    initial_count = len(df)
    
    print(f"Starting with {initial_count} rows")
    
    # Already have these images in clean set
    existing_images = set(clean_df['Image Index'].values)
    
    # Remove duplicates with clean set
    mask_new = ~df['Image Index'].isin(existing_images)
    df = df[mask_new]
    print(f"Removed {(~mask_new).sum()} duplicates already in clean set")
    
    # Apply same cleaning rules
    df_salvaged, df_flagged_salvage = clean_data_entry_primary(df)
    
    return df_salvaged, df_flagged_salvage

data_entry_salvaged, data_entry_salvaged_flagged = salvage_corrupted_data_entry(
    data_entry_corrupted, 
    data_entry_clean
)

print(f"\n✅ SALVAGED: {len(data_entry_salvaged)} additional clean rows")
print(f"🚩 FLAGGED: {len(data_entry_salvaged_flagged)} flagged rows")

# ========================================================================
# PART D: COMBINE CLEAN DATA + BBOX
# ========================================================================

print("\n" + "="*70)
print("PART D: COMBINING CLEAN DATA_ENTRY + BBOX")
print("="*70)

# Combine all clean data_entry sources
all_clean_data_entry = pd.concat([data_entry_clean, data_entry_salvaged], ignore_index=True)
all_clean_data_entry = all_clean_data_entry.drop_duplicates(subset=['Image Index'], keep='first')

print(f"Total clean data_entry records: {len(all_clean_data_entry)}")

# Combine all clean bbox sources
all_clean_bbox = pd.concat([bbox_2_clean, bbox_corrupted_clean], ignore_index=True)

# Determine which columns to use for deduplication based on what's available
dedup_cols = ['Image Index']
if 'Finding Label' in all_clean_bbox.columns:
    dedup_cols.append('Finding Label')
if 'x' in all_clean_bbox.columns:
    dedup_cols.append('x')
if 'y' in all_clean_bbox.columns:
    dedup_cols.append('y')

# Only deduplicate if we have more than just Image Index
if len(dedup_cols) > 1:
    all_clean_bbox = all_clean_bbox.drop_duplicates(subset=dedup_cols, keep='first')
    print(f"Deduplicated using columns: {dedup_cols}")
else:
    all_clean_bbox = all_clean_bbox.drop_duplicates(subset=['Image Index'], keep='first')
    print(f"Deduplicated using only: Image Index")

print(f"Total clean bbox records: {len(all_clean_bbox)}")

# Merge data_entry with bbox (LEFT JOIN - keep all data_entry, add bbox if available)
# First check what columns are actually in the bbox dataframe
print(f"\nBBox columns available: {list(all_clean_bbox.columns)}")

# Only merge if we have bbox data
if len(all_clean_bbox) > 0:
    clean_entry_combined = all_clean_data_entry.merge(
        all_clean_bbox,
        on='Image Index',
        how='left',
        suffixes=('', '_bbox')
    )
    
    # Clean up column names if there are duplicates
    if 'Finding Label' in clean_entry_combined.columns:
        # Rename to avoid confusion
        if 'Finding Labels' in clean_entry_combined.columns:
            clean_entry_combined = clean_entry_combined.rename(columns={
                'Finding Labels': 'diseases',
                'Finding Label': 'bbox_disease'
            })
else:
    # No bbox data, just use data_entry
    clean_entry_combined = all_clean_data_entry.copy()
    print("⚠️  No bbox data to merge")

print(f"\n✅ CLEAN_ENTRY.CSV: {len(clean_entry_combined)} records")
if 'bbox_disease' in clean_entry_combined.columns:
    print(f"   With bounding boxes: {clean_entry_combined['bbox_disease'].notna().sum()}")
    print(f"   Without bounding boxes: {clean_entry_combined['bbox_disease'].isna().sum()}")
else:
    print(f"   No bounding box data available")

# ========================================================================
# PART E: COMBINE FLAGGED DATA + BBOX
# ========================================================================

print("\n" + "="*70)
print("PART E: COMBINING FLAGGED DATA_ENTRY + BBOX")
print("="*70)

# Combine all flagged data_entry sources
all_flagged_data_entry = pd.concat([
    data_entry_flagged, 
    data_entry_salvaged_flagged
], ignore_index=True)
all_flagged_data_entry = all_flagged_data_entry.drop_duplicates(subset=['Image Index'], keep='first')

print(f"Total flagged data_entry records: {len(all_flagged_data_entry)}")

# Combine all flagged bbox sources
all_flagged_bbox = pd.concat([bbox_2_flagged, bbox_corrupted_flagged], ignore_index=True)

# Determine which columns to use for deduplication
dedup_cols_flagged = ['Image Index']
if 'Finding Label' in all_flagged_bbox.columns:
    dedup_cols_flagged.append('Finding Label')
if 'x' in all_flagged_bbox.columns:
    dedup_cols_flagged.append('x')
if 'y' in all_flagged_bbox.columns:
    dedup_cols_flagged.append('y')

# Deduplicate
if len(dedup_cols_flagged) > 1:
    all_flagged_bbox = all_flagged_bbox.drop_duplicates(subset=dedup_cols_flagged, keep='first')
else:
    all_flagged_bbox = all_flagged_bbox.drop_duplicates(subset=['Image Index'], keep='first')

print(f"Total flagged bbox records: {len(all_flagged_bbox)}")

# Merge flagged data_entry with flagged bbox
print(f"\nFlagged BBox columns available: {list(all_flagged_bbox.columns)}")

# Only merge if we have flagged bbox data
if len(all_flagged_bbox) > 0:
    flagged_combined = all_flagged_data_entry.merge(
        all_flagged_bbox,
        on='Image Index',
        how='outer',  # OUTER JOIN - get all flagged records
        suffixes=('', '_bbox')
    )
    
    # Clean up column names
    if 'Finding Label' in flagged_combined.columns:
        if 'Finding Labels' in flagged_combined.columns:
            flagged_combined = flagged_combined.rename(columns={
                'Finding Labels': 'diseases',
                'Finding Label': 'bbox_disease'
            })
else:
    # No flagged bbox data, just use flagged data_entry
    flagged_combined = all_flagged_data_entry.copy()
    print("⚠️  No flagged bbox data to merge")

print(f"\n🚩 FLAGGED.CSV: {len(flagged_combined)} records")

# ========================================================================
# PART F: SAVE FILES
# ========================================================================

print("\n" + "="*70)
print("PART F: SAVING FILES")
print("="*70)

# Save clean_entry.csv
clean_entry_combined.to_csv('/content/cleaned_data/clean_entry.csv', index=False)
print(f"✅ Saved: clean_entry.csv ({len(clean_entry_combined)} rows)")

# Save flagged.csv
flagged_combined.to_csv('/content/flagged_data/flagged.csv', index=False)
print(f"🚩 Saved: flagged.csv ({len(flagged_combined)} rows)")

# Save summary statistics
summary = {
    'Total Original Records': len(data_entry) + len(data_entry_corrupted),
    'Clean Records': len(clean_entry_combined),
    'Flagged Records': len(flagged_combined),
    'Clean Percentage': f"{len(clean_entry_combined)/(len(data_entry) + len(data_entry_corrupted))*100:.2f}%",
}

# Add bbox stats only if column exists
if 'bbox_disease' in clean_entry_combined.columns:
    summary['Clean with BBox'] = clean_entry_combined['bbox_disease'].notna().sum()
    summary['Clean without BBox'] = clean_entry_combined['bbox_disease'].isna().sum()
else:
    summary['Clean with BBox'] = 0
    summary['Clean without BBox'] = len(clean_entry_combined)

summary_df = pd.DataFrame([summary])
summary_df.to_csv('/content/logs/cleaning_summary.csv', index=False)
print(f"📊 Saved: cleaning_summary.csv")

# Save detailed flag reasons
if 'flag_reason' in flagged_combined.columns:
    flag_summary = flagged_combined['flag_reason'].value_counts().reset_index()
    flag_summary.columns = ['Flag Reason', 'Count']
    flag_summary.to_csv('/content/logs/flag_reasons.csv', index=False)
    print(f"📋 Saved: flag_reasons.csv")
    
    print("\nTop 10 flag reasons:")
    print(flag_summary.head(10))

# ========================================================================
# PART G: SAVE TO GOOGLE DRIVE
# ========================================================================

print("\n" + "="*70)
print("PART G: SAVING TO GOOGLE DRIVE")
print("="*70)

# Copy to Drive
import shutil

drive_output = '/content/drive/MyDrive/xray_cleaned_dataset'
os.makedirs(drive_output, exist_ok=True)

shutil.copy('/content/cleaned_data/clean_entry.csv', f'{drive_output}/clean_entry.csv')
shutil.copy('/content/flagged_data/flagged.csv', f'{drive_output}/flagged.csv')
shutil.copytree('/content/logs', f'{drive_output}/logs', dirs_exist_ok=True)

print(f"✓ All files saved to: {drive_output}")

# ========================================================================
# FINAL SUMMARY
# ========================================================================

print("\n" + "="*70)
print("🎉 CLEANING COMPLETE!")
print("="*70)

print(f"""
📊 FINAL STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ CLEAN_ENTRY.CSV
   Total records: {len(clean_entry_combined):,}
   With bounding boxes: {clean_entry_combined['bbox_disease'].notna().sum() if 'bbox_disease' in clean_entry_combined.columns else 0:,}
   Without bounding boxes: {clean_entry_combined['bbox_disease'].isna().sum() if 'bbox_disease' in clean_entry_combined.columns else len(clean_entry_combined):,}

🚩 FLAGGED.CSV
   Total records: {len(flagged_combined):,}
   
📈 OVERALL
   Clean rate: {len(clean_entry_combined)/(len(data_entry) + len(data_entry_corrupted))*100:.1f}%
   
📁 SAVED TO: {drive_output}
   ├── clean_entry.csv       (Ready for ML!)
   ├── flagged.csv           (Review/analyze)
   └── logs/
       ├── cleaning_summary.csv
       └── flag_reasons.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")