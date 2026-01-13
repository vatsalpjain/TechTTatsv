"""
================================================================================
COMPLETE DATA CLEANING PIPELINE FOR MEDICAL IMAGING DATASET
================================================================================

This script performs a comprehensive 5-stage data cleaning process:

STAGE 1: Load and Assess Initial Data
    - Load clean_entry.csv (pre-cleaned data)
    - Load flagged.csv (data with quality issues)

STAGE 2: Analyze Flagged Data Quality Issues
    - Invalid gender formats (female, Male, unknown, etc.)
    - Invalid age formats (text ages, decimals, unrealistic values)
    - Missing disease labels
    - Missing/fake image files

STAGE 3: Clean Flagged Data → cleaned2.0.csv
    - Standardize gender values (F/M or NaN)
    - Convert and validate ages (0-120 years)
    - Keep all rows, set invalid values to NaN

STAGE 4: Extract Completely Clean Data → cleaned3.0.csv
    - Filter rows with NO NaN in key columns
    - Ensure all required fields are populated

STAGE 5: Merge All Clean Data → final_clean.csv
    - Combine clean_entry.csv + cleaned3.0.csv
    - Remove duplicates based on Image Index
    - Final dataset ready for ML/analysis

OUTPUT FILES:
    - cleaned2.0.csv: All flagged data with standardized values
    - cleaned3.0.csv: Only completely clean flagged data
    - final_clean.csv: Combined final dataset (no duplicates)
    - cleaning_report.txt: Detailed statistics and summary

================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILES = {
    'clean_entry': 'clean_entry.csv',
    'flagged': 'flagged.csv'
}

OUTPUT_FILES = {
    'cleaned2': 'cleaned2.0.csv',
    'cleaned3': 'cleaned3.0.csv',
    'final': 'final_clean.csv',
    'report': 'cleaning_report.txt'
}

KEY_COLUMNS = [
    'Patient Gender', 
    'Patient ID', 
    'Image Index', 
    'Patient Age', 
    'diseases', 
    'OriginalImage[Width', 
    'OriginalImage[Height', 
    'View Position', 
    'Random_Code'
]

# Age validation parameters
MIN_VALID_AGE = 0
MAX_VALID_AGE = 120
UNREALISTIC_AGES = [999, 150]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_stage(stage_num, title):
    """Print formatted stage header"""
    print("\n" + "=" * 80)
    print(f"STAGE {stage_num}: {title}")
    print("=" * 80)

def print_section(title):
    """Print formatted section header"""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)

def clean_gender(gender):
    """
    Standardize gender values to F, M, or NaN
    
    Args:
        gender: Original gender value
        
    Returns:
        Standardized gender ('F', 'M', or NaN)
    """
    if pd.isna(gender):
        return np.nan
    
    gender_str = str(gender).strip().lower()
    
    # Map to standard F/M
    if gender_str in ['f', 'female']:
        return 'F'
    elif gender_str in ['m', 'male']:
        return 'M'
    else:
        # Keep unknown, N, or other invalid values as NaN
        return np.nan

def clean_age(age):
    """
    Convert and validate age values
    
    Args:
        age: Original age value
        
    Returns:
        Cleaned age (integer 0-120, or NaN if invalid)
    """
    if pd.isna(age):
        return np.nan
    
    # If already numeric and valid, return as float
    try:
        age_num = float(age)
        # Filter out unrealistic ages
        if age_num < MIN_VALID_AGE or age_num > MAX_VALID_AGE or age_num in UNREALISTIC_AGES:
            return np.nan
        # Handle decimal ages - round to nearest integer
        return round(age_num)
    except (ValueError, TypeError):
        pass
    
    # Handle text ages
    age_str = str(age).strip().lower()
    
    # Map common text ages
    text_age_map = {
        'twenty five': 25,
        'unknown': np.nan,
    }
    
    if age_str in text_age_map:
        return text_age_map[age_str]
    
    # If we can't convert, return NaN
    return np.nan

# ============================================================================
# STAGE 1: LOAD AND ASSESS INITIAL DATA
# ============================================================================

def load_initial_data():
    """Load clean_entry.csv and flagged.csv"""
    print_stage(1, "LOAD AND ASSESS INITIAL DATA")
    
    # Load clean_entry.csv
    print_section("Loading clean_entry.csv")
    df_clean_entry = pd.read_csv(INPUT_FILES['clean_entry'])
    print(f"✓ Loaded: {len(df_clean_entry):,} rows")
    print(f"  Columns: {len(df_clean_entry.columns)}")
    print(f"  Unique images: {df_clean_entry['Image Index'].nunique():,}")
    print(f"  Unique patients: {df_clean_entry['Patient ID'].nunique():,}")
    
    # Load flagged.csv
    print_section("Loading flagged.csv")
    df_flagged = pd.read_csv(INPUT_FILES['flagged'])
    print(f"✓ Loaded: {len(df_flagged):,} rows")
    print(f"  Columns: {len(df_flagged.columns)}")
    
    # Analyze flag reasons
    if 'flag_reason' in df_flagged.columns:
        print("\n  Flag reasons breakdown:")
        for reason, count in df_flagged['flag_reason'].value_counts().items():
            print(f"    - {reason}: {count:,} rows")
    
    return df_clean_entry, df_flagged

# ============================================================================
# STAGE 2: ANALYZE FLAGGED DATA QUALITY ISSUES
# ============================================================================

def analyze_flagged_data(df):
    """Analyze data quality issues in flagged.csv"""
    print_stage(2, "ANALYZE FLAGGED DATA QUALITY ISSUES")
    
    # Gender analysis
    print_section("Gender Value Analysis")
    print("  Unique gender values:")
    for gender, count in df['Patient Gender'].value_counts(dropna=False).items():
        print(f"    - '{gender}': {count:,} rows")
    
    # Age analysis
    print_section("Age Value Analysis")
    print("  Sample of problematic age values:")
    age_counts = df['Patient Age'].value_counts().head(20)
    for age, count in age_counts.items():
        print(f"    - '{age}': {count:,} rows")
    
    # Missing data analysis
    print_section("Missing Data Analysis")
    print("  NaN counts in key columns:")
    for col in KEY_COLUMNS:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            pct = (nan_count / len(df)) * 100
            print(f"    - {col:30s}: {nan_count:5,} ({pct:5.1f}%)")
    
    return df

# ============================================================================
# STAGE 3: CLEAN FLAGGED DATA → cleaned2.0.csv
# ============================================================================

def clean_flagged_data(df):
    """Clean flagged data and save to cleaned2.0.csv"""
    print_stage(3, "CLEAN FLAGGED DATA → cleaned2.0.csv")
    
    df = df.copy()
    initial_rows = len(df)
    
    # Clean Gender
    print_section("Cleaning Gender Column")
    original_valid_gender = df['Patient Gender'].notna().sum()
    df['Patient Gender'] = df['Patient Gender'].apply(clean_gender)
    cleaned_valid_gender = df['Patient Gender'].notna().sum()
    
    print(f"  Before: {original_valid_gender:,} valid gender values")
    print(f"  After:  {cleaned_valid_gender:,} valid gender values (F/M only)")
    print(f"  Improvement: +{cleaned_valid_gender - original_valid_gender:,} standardized values")
    
    # Clean Age
    print_section("Cleaning Age Column")
    original_valid_age = df['Patient Age'].notna().sum()
    df['Patient Age'] = df['Patient Age'].apply(clean_age)
    cleaned_valid_age = df['Patient Age'].notna().sum()
    
    print(f"  Before: {original_valid_age:,} non-null age values")
    print(f"  After:  {cleaned_valid_age:,} valid age values (0-120)")
    
    if cleaned_valid_age > 0:
        print(f"  Age range: {df['Patient Age'].min():.0f} to {df['Patient Age'].max():.0f}")
        print(f"  Mean age: {df['Patient Age'].mean():.1f} years")
    
    # Summary
    print_section("Cleaning Summary")
    print(f"  Total rows retained: {len(df):,} (100% - no rows removed)")
    print(f"  Gender standardized: {df['Patient Gender'].notna().sum():,} valid (F/M)")
    print(f"  Age cleaned: {df['Patient Age'].notna().sum():,} valid (0-120)")
    
    # Save
    df.to_csv(OUTPUT_FILES['cleaned2'], index=False)
    print(f"\n✓ Saved: {OUTPUT_FILES['cleaned2']}")
    
    return df

# ============================================================================
# STAGE 4: EXTRACT COMPLETELY CLEAN DATA → cleaned3.0.csv
# ============================================================================

def extract_completely_clean_data(df):
    """Extract rows with no NaN values in key columns"""
    print_stage(4, "EXTRACT COMPLETELY CLEAN DATA → cleaned3.0.csv")
    
    # Check NaN counts
    print_section("NaN Analysis in Key Columns")
    for col in KEY_COLUMNS:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            pct = (nan_count / len(df)) * 100
            print(f"  {col:30s}: {nan_count:5,} NaN ({pct:5.1f}%)")
    
    # Create filter mask
    print_section("Filtering for Complete Data")
    clean_mask = df[KEY_COLUMNS].notna().all(axis=1)
    
    # Also check that diseases is not empty string
    clean_mask &= df['diseases'].str.strip() != ''
    
    df_clean = df[clean_mask].copy()
    
    rows_removed = len(df) - len(df_clean)
    pct_clean = (len(df_clean) / len(df)) * 100
    
    print(f"  Total rows: {len(df):,}")
    print(f"  Complete rows: {len(df_clean):,}")
    print(f"  Rows removed: {rows_removed:,}")
    print(f"  Clean data percentage: {pct_clean:.1f}%")
    
    # Summary statistics
    print_section("Clean Data Statistics")
    print(f"  Gender distribution:")
    for gender, count in df_clean['Patient Gender'].value_counts().items():
        print(f"    - {gender}: {count:,}")
    
    print(f"\n  Age statistics:")
    print(f"    - Count: {len(df_clean):,}")
    print(f"    - Min: {df_clean['Patient Age'].min():.0f}")
    print(f"    - Max: {df_clean['Patient Age'].max():.0f}")
    print(f"    - Mean: {df_clean['Patient Age'].mean():.1f}")
    
    print(f"\n  View Position distribution:")
    for view, count in df_clean['View Position'].value_counts().items():
        print(f"    - {view}: {count:,}")
    
    print(f"\n  Top 5 diseases:")
    for disease, count in df_clean['diseases'].value_counts().head(5).items():
        print(f"    - {disease}: {count:,}")
    
    # Save
    df_clean.to_csv(OUTPUT_FILES['cleaned3'], index=False)
    print(f"\n✓ Saved: {OUTPUT_FILES['cleaned3']}")
    
    return df_clean

# ============================================================================
# STAGE 5: MERGE ALL CLEAN DATA → final_clean.csv
# ============================================================================

def merge_all_clean_data(df_clean_entry, df_cleaned3):
    """Merge clean_entry.csv and cleaned3.0.csv, remove duplicates"""
    print_stage(5, "MERGE ALL CLEAN DATA → final_clean.csv")
    
    # Check column compatibility
    print_section("Checking Column Compatibility")
    cols1 = set(df_clean_entry.columns)
    cols2 = set(df_cleaned3.columns)
    
    print(f"  clean_entry.csv columns: {len(cols1)}")
    print(f"  cleaned3.0.csv columns: {len(cols2)}")
    
    # Align columns
    if cols1 != cols2:
        print("\n  Column differences found - aligning...")
        only_in_clean_entry = cols1 - cols2
        only_in_cleaned3 = cols2 - cols1
        
        if only_in_clean_entry:
            print(f"    Only in clean_entry: {only_in_clean_entry}")
        if only_in_cleaned3:
            print(f"    Only in cleaned3.0: {only_in_cleaned3}")
        
        # Add missing columns
        all_cols = sorted(cols1.union(cols2))
        for col in all_cols:
            if col not in df_clean_entry.columns:
                df_clean_entry[col] = None
            if col not in df_cleaned3.columns:
                df_cleaned3[col] = None
        
        # Reorder columns
        df_clean_entry = df_clean_entry[all_cols]
        df_cleaned3 = df_cleaned3[all_cols]
        print("    ✓ Columns aligned!")
    
    # Combine datasets
    print_section("Combining Datasets")
    print(f"  clean_entry.csv: {len(df_clean_entry):,} rows")
    print(f"  cleaned3.0.csv:  {len(df_cleaned3):,} rows")
    
    df_combined = pd.concat([df_clean_entry, df_cleaned3], ignore_index=True)
    print(f"  Total after concat: {len(df_combined):,} rows")
    
    # Remove duplicates
    print_section("Removing Duplicates")
    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['Image Index'], keep='first')
    duplicates_removed = initial_count - len(df_combined)
    
    print(f"  Before deduplication: {initial_count:,} rows")
    print(f"  After deduplication:  {len(df_combined):,} rows")
    print(f"  Duplicates removed:   {duplicates_removed:,} rows")
    
    # Verify uniqueness
    is_unique = df_combined['Image Index'].nunique() == len(df_combined)
    print(f"  All images unique: {'✓ YES' if is_unique else '✗ NO'}")
    
    # Final statistics
    print_section("Final Dataset Statistics")
    print(f"  Total rows: {len(df_combined):,}")
    print(f"  Unique images: {df_combined['Image Index'].nunique():,}")
    print(f"  Unique patients: {df_combined['Patient ID'].nunique():,}")
    
    print(f"\n  Gender distribution:")
    for gender, count in df_combined['Patient Gender'].value_counts(dropna=False).items():
        print(f"    - {gender}: {count:,}")
    
    print(f"\n  Age statistics:")
    valid_ages = df_combined['Patient Age'].notna().sum()
    print(f"    - Valid ages: {valid_ages:,}")
    print(f"    - Missing ages: {df_combined['Patient Age'].isna().sum():,}")
    if valid_ages > 0:
        print(f"    - Min: {df_combined['Patient Age'].min():.0f}")
        print(f"    - Max: {df_combined['Patient Age'].max():.0f}")
        print(f"    - Mean: {df_combined['Patient Age'].mean():.1f}")
    
    print(f"\n  View Position distribution:")
    for view, count in df_combined['View Position'].value_counts(dropna=False).items():
        print(f"    - {view}: {count:,}")
    
    print(f"\n  Top 10 diseases:")
    for disease, count in df_combined['diseases'].value_counts().head(10).items():
        print(f"    - {disease}: {count:,}")
    
    # Save
    df_combined.to_csv(OUTPUT_FILES['final'], index=False)
    print(f"\n✓ Saved: {OUTPUT_FILES['final']}")
    
    return df_combined

# ============================================================================
# GENERATE CLEANING REPORT
# ============================================================================

def generate_report(df_clean_entry, df_flagged, df_cleaned2, df_cleaned3, df_final):
    """Generate comprehensive cleaning report"""
    print_stage("FINAL", "GENERATING CLEANING REPORT")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MEDICAL IMAGING DATASET - COMPLETE CLEANING REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Stage 1: Input Data
    report_lines.append("STAGE 1: INPUT DATA")
    report_lines.append("-" * 80)
    report_lines.append(f"clean_entry.csv:  {len(df_clean_entry):,} rows (pre-cleaned data)")
    report_lines.append(f"flagged.csv:      {len(df_flagged):,} rows (data with issues)")
    report_lines.append(f"Total input:      {len(df_clean_entry) + len(df_flagged):,} rows")
    report_lines.append("")
    
    # Stage 2: Flag Analysis
    report_lines.append("STAGE 2: FLAGGED DATA ISSUES")
    report_lines.append("-" * 80)
    if 'flag_reason' in df_flagged.columns:
        for reason, count in df_flagged['flag_reason'].value_counts().items():
            report_lines.append(f"  {reason}: {count:,} rows")
    report_lines.append("")
    
    # Stage 3: Cleaned Flagged Data
    report_lines.append("STAGE 3: CLEANED FLAGGED DATA (cleaned2.0.csv)")
    report_lines.append("-" * 80)
    report_lines.append(f"Total rows: {len(df_cleaned2):,} (100% retained)")
    report_lines.append(f"Valid gender (F/M): {df_cleaned2['Patient Gender'].notna().sum():,}")
    report_lines.append(f"Valid age (0-120): {df_cleaned2['Patient Age'].notna().sum():,}")
    report_lines.append("")
    
    # Stage 4: Completely Clean Data
    report_lines.append("STAGE 4: COMPLETELY CLEAN DATA (cleaned3.0.csv)")
    report_lines.append("-" * 80)
    report_lines.append(f"Total rows: {len(df_cleaned3):,}")
    report_lines.append(f"Percentage of flagged data: {len(df_cleaned3)/len(df_flagged)*100:.1f}%")
    report_lines.append(f"All key columns: 100% complete (no NaN)")
    report_lines.append("")
    
    # Stage 5: Final Merged Data
    report_lines.append("STAGE 5: FINAL MERGED DATA (final_clean.csv)")
    report_lines.append("-" * 80)
    report_lines.append(f"Total unique rows: {len(df_final):,}")
    report_lines.append(f"Unique images: {df_final['Image Index'].nunique():,}")
    report_lines.append(f"Unique patients: {df_final['Patient ID'].nunique():,}")
    report_lines.append(f"Duplicates removed: {(len(df_clean_entry) + len(df_cleaned3)) - len(df_final):,}")
    report_lines.append("")
    
    # Final Quality Metrics
    report_lines.append("FINAL QUALITY METRICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Gender completeness: {df_final['Patient Gender'].notna().sum()/len(df_final)*100:.1f}%")
    report_lines.append(f"Age completeness: {df_final['Patient Age'].notna().sum()/len(df_final)*100:.1f}%")
    report_lines.append(f"Disease labels: {df_final['diseases'].notna().sum()/len(df_final)*100:.1f}%")
    report_lines.append("")
    
    report_lines.append("Gender Distribution:")
    for gender, count in df_final['Patient Gender'].value_counts(dropna=False).items():
        pct = count / len(df_final) * 100
        report_lines.append(f"  {gender}: {count:,} ({pct:.1f}%)")
    report_lines.append("")
    
    valid_ages = df_final['Patient Age'].notna().sum()
    if valid_ages > 0:
        report_lines.append("Age Statistics:")
        report_lines.append(f"  Valid: {valid_ages:,}")
        report_lines.append(f"  Min: {df_final['Patient Age'].min():.0f}")
        report_lines.append(f"  Max: {df_final['Patient Age'].max():.0f}")
        report_lines.append(f"  Mean: {df_final['Patient Age'].mean():.1f}")
        report_lines.append("")
    
    report_lines.append("Top 10 Diseases:")
    for i, (disease, count) in enumerate(df_final['diseases'].value_counts().head(10).items(), 1):
        pct = count / len(df_final) * 100
        report_lines.append(f"  {i:2d}. {disease}: {count:,} ({pct:.1f}%)")
    report_lines.append("")
    
    # Output files
    report_lines.append("OUTPUT FILES GENERATED")
    report_lines.append("-" * 80)
    for key, filename in OUTPUT_FILES.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024 * 1024)  # MB
            report_lines.append(f"  ✓ {filename} ({size:.2f} MB)")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("CLEANING COMPLETE - DATASET READY FOR ANALYSIS!")
    report_lines.append("=" * 80)
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(OUTPUT_FILES['report'], 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ Report saved: {OUTPUT_FILES['report']}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete cleaning pipeline"""
    print("\n" + "=" * 80)
    print("MEDICAL IMAGING DATASET - COMPLETE CLEANING PIPELINE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Stage 1: Load data
        df_clean_entry, df_flagged = load_initial_data()
        
        # Stage 2: Analyze
        df_flagged = analyze_flagged_data(df_flagged)
        
        # Stage 3: Clean flagged data
        df_cleaned2 = clean_flagged_data(df_flagged)
        
        # Stage 4: Extract completely clean data
        df_cleaned3 = extract_completely_clean_data(df_cleaned2)
        
        # Stage 5: Merge all clean data
        df_final = merge_all_clean_data(df_clean_entry, df_cleaned3)
        
        # Generate report
        generate_report(df_clean_entry, df_flagged, df_cleaned2, df_cleaned3, df_final)
        
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! ALL STAGES COMPLETED")
        print("=" * 80)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nFinal dataset: {OUTPUT_FILES['final']} ({len(df_final):,} rows)")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
