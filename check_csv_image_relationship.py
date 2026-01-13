"""Quick script to check CSV row vs image relationship"""
import pandas as pd

df = pd.read_csv('final_clean_validated.csv', low_memory=False)
print(f"Total CSV rows: {len(df):,}")
print(f"Unique Image Index values: {df['Image Index'].nunique():,}")
print(f"\nDifference: {len(df) - df['Image Index'].nunique():,} rows are duplicates")

# Check for duplicates
dupes = df[df.duplicated(subset=['Image Index'], keep=False)]
if len(dupes) > 0:
    print(f"\nRows with duplicate Image Index: {len(dupes):,}")
    print("\nExample - images with multiple rows:")
    example = dupes.groupby('Image Index').size().head(10)
    print(example)
    
    # Show a specific example
    example_img = example.index[0]
    print(f"\nExample: {example_img} appears {example.iloc[0]} times:")
    print(df[df['Image Index'] == example_img][['Image Index', 'Follow-up #', 'Patient ID', 'diseases']].head())
else:
    print("\nNo duplicates found - each Image Index appears exactly once")

