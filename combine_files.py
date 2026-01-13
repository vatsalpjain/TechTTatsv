import pandas as pd

# Read both CSV files
print("Reading clean_entry.csv...")
df1 = pd.read_csv('clean_entry.csv')
print(f"  Rows: {len(df1)}")

print("\nReading cleaned3.0.csv...")
df2 = pd.read_csv('cleaned3.0.csv')
print(f"  Rows: {len(df2)}")

# Check columns to ensure they match
print("\nChecking column compatibility...")
print(f"clean_entry.csv columns: {len(df1.columns)}")
print(f"cleaned3.0.csv columns: {len(df2.columns)}")

# Get column differences if any
cols1 = set(df1.columns)
cols2 = set(df2.columns)
if cols1 != cols2:
    print("\nColumn differences found:")
    print(f"  Only in clean_entry: {cols1 - cols2}")
    print(f"  Only in cleaned3.0: {cols2 - cols1}")
    
    # Align columns - keep only common columns or add missing ones
    all_cols = sorted(cols1.union(cols2))
    for col in all_cols:
        if col not in df1.columns:
            df1[col] = None
        if col not in df2.columns:
            df2[col] = None
    
    # Reorder columns to match
    df1 = df1[all_cols]
    df2 = df2[all_cols]
    print("  Columns aligned!")

# Combine the dataframes
print("\nCombining datasets...")
df_combined = pd.concat([df1, df2], ignore_index=True)
print(f"  Total rows after combining: {len(df_combined)}")

# Remove duplicates based on Image Index (unique image identifier)
print("\nRemoving duplicates based on 'Image Index'...")
initial_count = len(df_combined)
df_combined = df_combined.drop_duplicates(subset=['Image Index'], keep='first')
duplicates_removed = initial_count - len(df_combined)
print(f"  Duplicates removed: {duplicates_removed}")
print(f"  Final unique rows: {len(df_combined)}")

# Show summary statistics
print("\n" + "=" * 60)
print("FINAL COMBINED DATASET SUMMARY:")
print("=" * 60)
print(f"Total rows: {len(df_combined)}")
print(f"Unique images: {df_combined['Image Index'].nunique()}")
print(f"Unique patients: {df_combined['Patient ID'].nunique()}")

print(f"\nGender distribution:")
print(df_combined['Patient Gender'].value_counts(dropna=False))

print(f"\nAge statistics:")
print(f"  Count: {df_combined['Patient Age'].notna().sum()}")
print(f"  Missing: {df_combined['Patient Age'].isna().sum()}")
print(f"  Min: {df_combined['Patient Age'].min()}")
print(f"  Max: {df_combined['Patient Age'].max()}")
print(f"  Mean: {df_combined['Patient Age'].mean():.1f}")

print(f"\nView Position distribution:")
print(df_combined['View Position'].value_counts(dropna=False))

print(f"\nTop 10 diseases:")
print(df_combined['diseases'].value_counts().head(10))

# Save to final_clean.csv
output_file = 'final_clean.csv'
df_combined.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print(f"✅ Final combined data saved to: {output_file}")
print(f"   Total rows: {len(df_combined)}")
print(f"   No duplicates: All Image Index values are unique")
print("=" * 60)
