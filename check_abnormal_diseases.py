import pandas as pd

# Load the dataset
df = pd.read_csv('final_clean.csv')

print("=" * 80)
print("CHECKING FOR ABNORMAL DISEASE NAMES")
print("=" * 80)

# Get all unique disease values
diseases = df['diseases'].unique()
print(f"\nTotal unique disease values: {len(diseases)}")

# Define patterns to identify abnormal disease names
abnormal_patterns = ['xyz', '123', 'test', 'fake', 'invalid', 'mock', 'xxx']

# Find abnormal diseases
print("\n" + "-" * 80)
print("ABNORMAL DISEASE NAMES FOUND:")
print("-" * 80)

abnormal_diseases = []
for disease in diseases:
    disease_str = str(disease).lower()
    for pattern in abnormal_patterns:
        if pattern in disease_str:
            count = len(df[df['diseases'] == disease])
            print(f'  "{disease}": {count} rows')
            abnormal_diseases.append(disease)
            break

if not abnormal_diseases:
    print("  None found!")
else:
    print(f"\nTotal abnormal disease types: {len(abnormal_diseases)}")
    total_rows_affected = len(df[df['diseases'].isin(abnormal_diseases)])
    print(f"Total rows with abnormal diseases: {total_rows_affected}")

# Also check for numeric-only disease names
print("\n" + "-" * 80)
print("CHECKING FOR NUMERIC-ONLY DISEASE NAMES:")
print("-" * 80)

numeric_diseases = []
for disease in diseases:
    disease_str = str(disease).strip()
    if disease_str.isdigit():
        count = len(df[df['diseases'] == disease])
        print(f'  "{disease}": {count} rows')
        numeric_diseases.append(disease)

if not numeric_diseases:
    print("  None found!")

# Check for suspicious short names
print("\n" + "-" * 80)
print("CHECKING FOR SUSPICIOUSLY SHORT DISEASE NAMES (< 3 chars):")
print("-" * 80)

short_diseases = []
for disease in diseases:
    disease_str = str(disease).strip()
    if len(disease_str) < 3 and disease_str not in ['nan']:
        count = len(df[df['diseases'] == disease])
        print(f'  "{disease}": {count} rows')
        short_diseases.append(disease)

if not short_diseases:
    print("  None found!")

# Summary of all abnormal diseases to remove
print("\n" + "=" * 80)
print("SUMMARY - DISEASES TO REMOVE:")
print("=" * 80)

all_abnormal = list(set(abnormal_diseases + numeric_diseases + short_diseases))
if all_abnormal:
    total_to_remove = len(df[df['diseases'].isin(all_abnormal)])
    print(f"\nTotal disease types to remove: {len(all_abnormal)}")
    print(f"Total rows to be removed: {total_to_remove}")
    print(f"Remaining rows after removal: {len(df) - total_to_remove}")
    print(f"\nDiseases to remove: {all_abnormal}")
else:
    print("\nNo abnormal diseases found!")

print("\n" + "=" * 80)
