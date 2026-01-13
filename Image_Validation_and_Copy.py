"""
================================================================================
IMAGE VALIDATION & REFERENCED IMAGES EXTRACTION
================================================================================

This script:
1. Reads final_clean.csv to get all image references
2. Searches for images in: images_001/images/, images_002/images/, images_003/images/
3. Validates each found image (corruption, format, size)
4. COPIES only clean & valid images to referenced_images/ folder
5. Generates detailed reports

Validation Criteria:
- Format: PNG only (.png, .PNG)
- File size: 10 KB to 100 MB (medical X-rays typical range)
- Corruption: Must be openable by PIL/Pillow
- Referenced: Must exist in final_clean.csv

Output:
- referenced_images/ folder with clean images
- CSV reports for tracking

================================================================================
"""

import pandas as pd
import os
import shutil
from PIL import Image
from pathlib import Path
from datetime import datetime
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_FILE = 'final_clean.csv'
IMAGE_INDEX_COLUMN = 'Image Index'

# Image source folders
IMAGE_FOLDERS = [
    'images_001/images',
    'images_002/images', 
    'images_003/images',
    'images_004/images',
    'images_005/images',
    'images_006/images'
]

# Output folder for valid referenced images
OUTPUT_FOLDER = 'referenced_images'

# Validation thresholds (in bytes)
MIN_FILE_SIZE = 10 * 1024        # 10 KB - smaller files likely corrupted
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB - medical images shouldn't exceed this

# Valid formats
VALID_FORMATS = ['.png', '.PNG']

# Report files
REPORTS = {
    'found_valid': 'found_and_valid_images.csv',
    'missing': 'missing_images.csv',
    'corrupted': 'corrupted_images.csv',
    'invalid_format': 'invalid_format_images.csv',
    'invalid_size': 'invalid_size_images.csv',
    'summary': 'image_validation_summary.txt'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_section(title):
    """Print formatted section"""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)

def format_size(size_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def find_image(image_name, search_folders):
    """
    Search for image in multiple folders
    
    Args:
        image_name: Name of the image file
        search_folders: List of folders to search
        
    Returns:
        Full path if found, None otherwise
    """
    for folder in search_folders:
        image_path = os.path.join(folder, image_name)
        if os.path.exists(image_path):
            return image_path
    return None

def validate_image(image_path):
    """
    Validate image for corruption, format, and size
    
    Args:
        image_path: Full path to image file
        
    Returns:
        dict with validation results
    """
    result = {
        'valid': False,
        'exists': False,
        'format_valid': False,
        'size_valid': False,
        'openable': False,
        'file_size': 0,
        'error': None
    }
    
    # Check existence
    if not os.path.exists(image_path):
        result['error'] = 'File not found'
        return result
    
    result['exists'] = True
    
    # Check format (extension)
    file_ext = os.path.splitext(image_path)[1]
    if file_ext not in VALID_FORMATS:
        result['error'] = f'Invalid format: {file_ext}'
        return result
    
    result['format_valid'] = True
    
    # Check file size
    try:
        file_size = os.path.getsize(image_path)
        result['file_size'] = file_size
        
        if file_size < MIN_FILE_SIZE:
            result['error'] = f'File too small: {format_size(file_size)} (min: {format_size(MIN_FILE_SIZE)})'
            return result
        
        if file_size > MAX_FILE_SIZE:
            result['error'] = f'File too large: {format_size(file_size)} (max: {format_size(MAX_FILE_SIZE)})'
            return result
        
        result['size_valid'] = True
        
    except Exception as e:
        result['error'] = f'Cannot get file size: {str(e)}'
        return result
    
    # Try to open image (corruption check)
    try:
        with Image.open(image_path) as img:
            # Verify it can be loaded
            img.verify()
        
        # Re-open for additional checks (verify() closes the file)
        with Image.open(image_path) as img:
            result['width'] = img.width
            result['height'] = img.height
            result['mode'] = img.mode
        
        result['openable'] = True
        result['valid'] = True
        
    except Exception as e:
        result['error'] = f'Cannot open image (corrupted): {str(e)}'
        return result
    
    return result

# ============================================================================
# MAIN VALIDATION PROCESS
# ============================================================================

def validate_and_copy_images():
    """Main function to validate and copy referenced images"""
    
    # Start timing
    start_time = time.time()
    
    print_header("IMAGE VALIDATION & REFERENCED IMAGES EXTRACTION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output folder
    print_section("Setting up output folder")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"✓ Created/verified: {OUTPUT_FOLDER}/")
    
    # Load CSV
    print_section("Loading CSV reference file")
    print(f"Reading: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded: {len(df):,} image references")
    
    # Get unique image names
    image_list = df[IMAGE_INDEX_COLUMN].unique().tolist()
    print(f"✓ Unique images to validate: {len(image_list):,}")
    
    # Verify image folders exist
    print_section("Checking image source folders")
    existing_folders = []
    for folder in IMAGE_FOLDERS:
        if os.path.exists(folder):
            file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            print(f"✓ {folder}: {file_count:,} files")
            existing_folders.append(folder)
        else:
            print(f"✗ {folder}: NOT FOUND")
    
    if not existing_folders:
        print("\n❌ ERROR: No image folders found!")
        return
    
    # Validation tracking
    results = {
        'found_valid': [],
        'missing': [],
        'corrupted': [],
        'invalid_format': [],
        'invalid_size': []
    }
    
    # Process each image
    print_section("Validating images")
    print(f"Processing {len(image_list):,} images...")
    print(f"Validation criteria:")
    print(f"  - Format: {', '.join(VALID_FORMATS)}")
    print(f"  - Size: {format_size(MIN_FILE_SIZE)} to {format_size(MAX_FILE_SIZE)}")
    print(f"  - Corruption check: Enabled")
    print()
    
    processed = 0
    copied = 0
    
    for idx, image_name in enumerate(image_list, 1):
        # Progress indicator with elapsed time
        if idx % 1000 == 0 or idx == len(image_list):
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(image_list) - idx) / rate if rate > 0 else 0
            print(f"  Progress: {idx:,}/{len(image_list):,} ({idx/len(image_list)*100:.1f}%) | "
                  f"Rate: {rate:.1f} img/sec | "
                  f"Elapsed: {elapsed/60:.1f}m | "
                  f"ETA: {remaining/60:.1f}m")
        
        # Find image
        image_path = find_image(image_name, existing_folders)
        
        if image_path is None:
            # Image not found
            results['missing'].append({
                'Image Index': image_name,
                'Status': 'NOT FOUND'
            })
            continue
        
        # Validate image
        validation = validate_image(image_path)
        processed += 1
        
        if validation['valid']:
            # Valid image - copy to output folder
            try:
                dest_path = os.path.join(OUTPUT_FOLDER, image_name)
                shutil.copy2(image_path, dest_path)
                copied += 1
                
                results['found_valid'].append({
                    'Image Index': image_name,
                    'Source Path': image_path,
                    'File Size': format_size(validation['file_size']),
                    'Width': validation.get('width', 0),
                    'Height': validation.get('height', 0),
                    'Mode': validation.get('mode', 'Unknown'),
                    'Status': 'COPIED TO REFERENCED_IMAGES'
                })
            except Exception as e:
                results['corrupted'].append({
                    'Image Index': image_name,
                    'Source Path': image_path,
                    'Error': f'Copy failed: {str(e)}'
                })
        
        elif not validation['format_valid']:
            # Invalid format
            results['invalid_format'].append({
                'Image Index': image_name,
                'Source Path': image_path,
                'Error': validation['error']
            })
        
        elif not validation['size_valid']:
            # Invalid size
            results['invalid_size'].append({
                'Image Index': image_name,
                'Source Path': image_path,
                'File Size': format_size(validation['file_size']),
                'Error': validation['error']
            })
        
        else:
            # Corrupted
            results['corrupted'].append({
                'Image Index': image_name,
                'Source Path': image_path,
                'Error': validation['error']
            })
    
    # Generate reports
    print_section("Generating reports")
    
    # Save individual reports
    if results['found_valid']:
        df_valid = pd.DataFrame(results['found_valid'])
        df_valid.to_csv(REPORTS['found_valid'], index=False)
        print(f"✓ {REPORTS['found_valid']}: {len(results['found_valid']):,} valid images")
    
    if results['missing']:
        df_missing = pd.DataFrame(results['missing'])
        df_missing.to_csv(REPORTS['missing'], index=False)
        print(f"✗ {REPORTS['missing']}: {len(results['missing']):,} missing images")
    
    if results['corrupted']:
        df_corrupted = pd.DataFrame(results['corrupted'])
        df_corrupted.to_csv(REPORTS['corrupted'], index=False)
        print(f"⚠ {REPORTS['corrupted']}: {len(results['corrupted']):,} corrupted images")
    
    if results['invalid_format']:
        df_format = pd.DataFrame(results['invalid_format'])
        df_format.to_csv(REPORTS['invalid_format'], index=False)
        print(f"⚠ {REPORTS['invalid_format']}: {len(results['invalid_format']):,} invalid format")
    
    # Calculate total elapsed time
    end_time = time.time()
    total_elapsed = end_time - start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    
    # Generate summary report
    print_section("Generating summary report")
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("IMAGE VALIDATION & COPY SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"Total time: {hours}h {minutes}m {seconds}s ({total_elapsed:.1f} seconds)")
    summary_lines.append("")
    
    summary_lines.append("VALIDATION CRITERIA:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"  Format: {', '.join(VALID_FORMATS)}")
    summary_lines.append(f"  Min size: {format_size(MIN_FILE_SIZE)}")
    summary_lines.append(f"  Max size: {format_size(MAX_FILE_SIZE)}")
    summary_lines.append("")
    
    summary_lines.append("RESULTS:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"  Total referenced images in CSV: {len(image_list):,}")
    summary_lines.append(f"  Images found in folders: {processed:,}")
    summary_lines.append(f"  Images missing: {len(results['missing']):,}")
    summary_lines.append("")
    summary_lines.append(f"  ✓ Valid & copied to referenced_images/: {len(results['found_valid']):,}")
    summary_lines.append(f"  ✗ Corrupted/unreadable: {len(results['corrupted']):,}")
    summary_lines.append(f"  ✗ Invalid format: {len(results['invalid_format']):,}")
    summary_lines.append(f"  ✗ Invalid size: {len(results['invalid_size']):,}")
    summary_lines.append("")
    
    # Calculate percentages
    total_found = processed
    if total_found > 0:
        valid_pct = (len(results['found_valid']) / total_found) * 100
        summary_lines.append(f"QUALITY METRICS:")
        summary_lines.append("-" * 80)
        summary_lines.append(f"  Success rate: {valid_pct:.1f}% ({len(results['found_valid']):,}/{total_found:,})")
        summary_lines.append(f"  Missing rate: {(len(results['missing'])/len(image_list))*100:.1f}%")
    
    summary_lines.append("")
    summary_lines.append("OUTPUT:")
    summary_lines.append("PERFORMANCE:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"  Total processing time: {hours}h {minutes}m {seconds}s")
    if processed > 0:
        rate = processed / total_elapsed
        summary_lines.append(f"  Average processing rate: {rate:.2f} images/second")
        summary_lines.append(f"  Time per image: {(total_elapsed/processed)*1000:.1f} milliseconds")
    summary_lines.append("")
    summary_lines.append("-" * 80)
    summary_lines.append(f"  Referenced images folder: {OUTPUT_FOLDER}/")
    summary_lines.append(f"  Total images copied: {copied:,}")
    
    # Get folder size
    if os.path.exists(OUTPUT_FOLDER):
        total_size = sum(
            os.path.getsize(os.path.join(OUTPUT_FOLDER, f)) 
            for f in os.listdir(OUTPUT_FOLDER) 
            if os.path.isfile(os.path.join(OUTPUT_FOLDER, f))
        )
        summary_lines.append(f"  Total folder size: {format_size(total_size)}")
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append("VALIDATION COMPLETE!")
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines)
    
    # Write summary file
    with open(REPORTS['summary'], 'w') as f:
        f.write(summary_text)
    
    # Print summary
    print()
    print(summary_text)
    print(f"\n✓ Summary saved: {REPORTS['summary']}")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        validate_and_copy_images()
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! Image validation and copy completed")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
