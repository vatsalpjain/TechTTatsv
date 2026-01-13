import pandas as pd
import numpy as np

# Read the flagged CSV file
df = pd.read_csv('flagged.csv')

print(f"Original rows: {len(df)}")
print(f"\nOriginal Gender values:\n{df['Patient Gender'].value_counts(dropna=False)}\n")
print(f"Original Age sample values:\n{df['Patient Age'].value_counts().head(15)}\n")

# 1. Clean Gender column
# Convert various gender formats to F or M
def clean_gender(gender):
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

df['Patient Gender'] = df['Patient Gender'].apply(clean_gender)

# 2. Clean Age column
# Convert text ages to numeric values
def clean_age(age):
    if pd.isna(age):
        return np.nan
    
    # If already numeric and valid, return as float
    try:
        age_num = float(age)
        # Filter out unrealistic ages (negative, 999, 150, etc.)
        if age_num < 0 or age_num > 120 or age_num == 999:
            return np.nan
        # Handle decimal ages like 45.7 - round to nearest integer
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

df['Patient Age'] = df['Patient Age'].apply(clean_age)

# Display cleaning results
print("=" * 60)
print("AFTER CLEANING:")
print("=" * 60)
print(f"\nCleaned Gender values:\n{df['Patient Gender'].value_counts(dropna=False)}\n")
print(f"Cleaned Age statistics:")
print(f"  Valid ages: {df['Patient Age'].notna().sum()}")
print(f"  Missing/Invalid ages: {df['Patient Age'].isna().sum()}")
print(f"  Age range: {df['Patient Age'].min():.0f} to {df['Patient Age'].max():.0f}")
print(f"  Mean age: {df['Patient Age'].mean():.1f}\n")

# Show sample of age distribution
print(f"Age value counts (top 20):\n{df['Patient Age'].value_counts().head(20)}\n")

# Save to new CSV file
output_file = 'cleaned2.0.csv'
df.to_csv(output_file, index=False)

print(f"Cleaned data saved to: {output_file}")
print(f"Total rows in output: {len(df)}")
