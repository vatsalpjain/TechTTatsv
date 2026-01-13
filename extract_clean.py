import pandas as pd
import numpy as np

# Read the cleaned2.0 CSV file
df = pd.read_csv('cleaned2.0.csv')

print(f"Original cleaned2.0.csv rows: {len(df)}")
print(f"\nChecking for NaN/missing values in key columns:")
print("=" * 60)

# Check NaN counts for important columns
key_columns = ['Patient Gender', 'Patient ID', 'Image Index', 'Patient Age', 
               'diseases', 'OriginalImage[Width', 'OriginalImage[Height', 
               'View Position', 'Random_Code']

for col in key_columns:
    nan_count = df[col].isna().sum()
    print(f"{col:30s}: {nan_count:5d} NaN values ({nan_count/len(df)*100:.1f}%)")

# Filter for completely clean rows (no NaN in key columns)
print("\n" + "=" * 60)
print("Filtering for completely clean rows...")
print("=" * 60)

# Create a mask for rows with no NaN values in key columns
clean_mask = df[key_columns].notna().all(axis=1)

# Also check that diseases is not empty string
clean_mask &= df['diseases'].str.strip() != ''

# Extract clean data
df_clean = df[clean_mask].copy()

print(f"\nRows with complete data (no NaN in key columns): {len(df_clean)}")
print(f"Rows removed due to missing data: {len(df) - len(df_clean)}")
print(f"Percentage of clean data: {len(df_clean)/len(df)*100:.1f}%")

# Show summary of clean data
print("\n" + "=" * 60)
print("CLEAN DATA SUMMARY:")
print("=" * 60)
print(f"\nGender distribution:")
print(df_clean['Patient Gender'].value_counts())
print(f"\nAge statistics:")
print(f"  Count: {len(df_clean)}")
print(f"  Min: {df_clean['Patient Age'].min():.0f}")
print(f"  Max: {df_clean['Patient Age'].max():.0f}")
print(f"  Mean: {df_clean['Patient Age'].mean():.1f}")
print(f"\nView Position distribution:")
print(df_clean['View Position'].value_counts())
print(f"\nTop diseases:")
print(df_clean['diseases'].value_counts().head(10))

# Save to cleaned3.0.csv
output_file = 'cleaned3.0.csv'
df_clean.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print(f"✅ Completely clean data saved to: {output_file}")
print(f"   Total rows: {len(df_clean)}")
print("=" * 60)
