import pandas as pd
import re

# Load the dataset
df = pd.read_csv('final_clean.csv')

print("=" * 80)
print("COMPREHENSIVE DISEASE NAME VALIDATION CHECK")
print("=" * 80)
print(f"Total rows: {len(df):,}\n")

# Get all unique disease values
diseases = df['diseases'].unique()
print(f"Total unique disease values: {len(diseases)}\n")

# Track all problematic diseases
all_problematic = {}

# 1. Check for obvious fake/test patterns
print("-" * 80)
print("1. CHECKING FOR FAKE/TEST PATTERNS (XYZ, 123, test, fake, etc.):")
print("-" * 80)

fake_patterns = ['xyz', '123', 'test', 'fake', 'invalid', 'mock', 'xxx', 'sample', 'demo']
fake_diseases = []

for disease in diseases:
    disease_str = str(disease).lower()
    for pattern in fake_patterns:
        if pattern in disease_str:
            count = len(df[df['diseases'] == disease])
            print(f'  "{disease}": {count:,} rows')
            fake_diseases.append(disease)
            all_problematic[disease] = count
            break

if not fake_diseases:
    print("  ✓ None found!")

# 2. Check for numeric-only disease names
print("\n" + "-" * 80)
print("2. CHECKING FOR NUMERIC-ONLY DISEASE NAMES:")
print("-" * 80)

numeric_diseases = []
for disease in diseases:
    disease_str = str(disease).strip()
    if disease_str.isdigit():
        count = len(df[df['diseases'] == disease])
        print(f'  "{disease}": {count:,} rows')
        numeric_diseases.append(disease)
        all_problematic[disease] = count

if not numeric_diseases:
    print("  ✓ None found!")

# 3. Check for special characters (non-alphanumeric except |, _, space)
print("\n" + "-" * 80)
print("3. CHECKING FOR SPECIAL CHARACTERS (###, ***, ???, etc.):")
print("-" * 80)

special_char_pattern = re.compile(r'[^a-zA-Z0-9|_\s]')
special_diseases = []

for disease in diseases:
    disease_str = str(disease)
    if disease_str != 'nan' and special_char_pattern.search(disease_str):
        count = len(df[df['diseases'] == disease])
        print(f'  "{disease}": {count:,} rows (contains special chars)')
        special_diseases.append(disease)
        all_problematic[disease] = count

if not special_diseases:
    print("  ✓ None found!")

# 4. Check for very long disease names (potential data entry errors)
print("\n" + "-" * 80)
print("4. CHECKING FOR VERY LONG DISEASE NAMES (>100 characters):")
print("-" * 80)

long_diseases = []
for disease in diseases:
    disease_str = str(disease)
    if len(disease_str) > 100:
        count = len(df[df['diseases'] == disease])
        print(f'  "{disease[:80]}...": {count:,} rows (length: {len(disease_str)})')
        long_diseases.append(disease)
        all_problematic[disease] = count

if not long_diseases:
    print("  ✓ None found!")

# 5. Check for suspiciously short names
print("\n" + "-" * 80)
print("5. CHECKING FOR SUSPICIOUSLY SHORT DISEASE NAMES (< 3 chars):")
print("-" * 80)

short_diseases = []
for disease in diseases:
    disease_str = str(disease).strip()
    if len(disease_str) < 3 and disease_str not in ['nan']:
        count = len(df[df['diseases'] == disease])
        print(f'  "{disease}": {count:,} rows')
        short_diseases.append(disease)
        all_problematic[disease] = count

if not short_diseases:
    print("  ✓ None found!")

# 6. Check for diseases with only underscores/pipes
print("\n" + "-" * 80)
print("6. CHECKING FOR MALFORMED DISEASE NAMES (only _, |, or spaces):")
print("-" * 80)

malformed_pattern = re.compile(r'^[_|\s]+$')
malformed_diseases = []

for disease in diseases:
    disease_str = str(disease)
    if disease_str != 'nan' and malformed_pattern.match(disease_str):
        count = len(df[df['diseases'] == disease])
        print(f'  "{disease}": {count:,} rows')
        malformed_diseases.append(disease)
        all_problematic[disease] = count

if not malformed_diseases:
    print("  ✓ None found!")

# 7. Check for unusual capitalization patterns
print("\n" + "-" * 80)
print("7. CHECKING FOR ALL CAPS DISEASE NAMES (potential data entry issues):")
print("-" * 80)

caps_diseases = []
for disease in diseases:
    disease_str = str(disease).strip()
    # Check if all caps and has more than 3 chars
    if disease_str.isupper() and len(disease_str) > 3 and disease_str != 'nan':
        count = len(df[df['diseases'] == disease])
        if count > 0:  # Only report if there are rows
            print(f'  "{disease}": {count:,} rows')
            caps_diseases.append(disease)

if not caps_diseases:
    print("  ✓ None found (or all are valid medical abbreviations)!")

# FINAL SUMMARY
print("\n" + "=" * 80)
print("FINAL SUMMARY - DISEASES TO REMOVE:")
print("=" * 80)

unique_problematic = list(set(all_problematic.keys()))
total_rows_affected = len(df[df['diseases'].isin(unique_problematic)])

if unique_problematic:
    print(f"\n✗ Total problematic disease types found: {len(unique_problematic)}")
    print(f"✗ Total rows to be removed: {total_rows_affected:,}")
    print(f"✓ Rows remaining after removal: {len(df) - total_rows_affected:,}")
    print(f"✓ Percentage retained: {((len(df) - total_rows_affected) / len(df)) * 100:.2f}%")
    
    print("\n" + "-" * 80)
    print("DISEASES TO REMOVE:")
    print("-" * 80)
    for disease in sorted(unique_problematic):
        count = all_problematic[disease]
        print(f"  • \"{disease}\": {count:,} rows")
    
    print("\n" + "-" * 80)
    print("BREAKDOWN BY CATEGORY:")
    print("-" * 80)
    print(f"  Fake/Test patterns: {len(fake_diseases)}")
    print(f"  Numeric-only: {len(numeric_diseases)}")
    print(f"  Special characters: {len(special_diseases)}")
    print(f"  Very long names: {len(long_diseases)}")
    print(f"  Very short names: {len(short_diseases)}")
    print(f"  Malformed: {len(malformed_diseases)}")
    
else:
    print("\n✓ No problematic diseases found!")

print("\n" + "=" * 80)
print("READY TO PROCEED WITH REMOVAL")
print("=" * 80)
