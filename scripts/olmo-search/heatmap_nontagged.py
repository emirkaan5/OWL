import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io


def filter_out_tagged(col_list):
    eval_path = "/home/ekorukluoglu_umass_edu/beam2/BEAM/olmo-search/evals/olmo_eval.csv"
    eval_df = pd.read_csv(eval_path)
    
# Path to the CSV file with evaluation data
csv_data = "/Users/emir/Projects/BEAM/results/direct_probe/EuroLLM-9B-Instruct/masked_one_shot/1984_direct_probe_masked_EuroLLM-9B-Instruct_one-shot.csv"

# Path to the CSV file with book name to index mappings
book_index_csv = "evals/olmo_eval.csv"  # Replace with your actual path

# Load the evaluation data
df = pd.read_csv(csv_data)

# Specify the book name we're analyzing
target_book = "1984"  # The book we want to analyze

# Load the book index mapping
book_df = pd.read_csv(book_index_csv)

# Get indices associated with the target book
target_indices = book_df[book_df['filename'] == target_book]['index'].tolist()

if not target_indices:
    print(f"No indices found for '{target_book}' in the mapping file")
    exit()

print(f"Found {len(target_indices)} indices associated with '{target_book}'")

# Filter the evaluation dataframe to KEEP ONLY the indices for the target book
df_target = df.loc[target_indices]

if df_target.empty:
    print(f"No rows found in evaluation data for the indices of '{target_book}'")
    exit()

print(f"Analyzing {len(df_target)} rows of data for '{target_book}'")

# Define the column groups based on language patterns
en_both_columns = [col for col in df_target.columns if col.startswith('en_') and 'both_match' in col]
translated_both_columns = [
    col for col in df_target.columns if 
    (col.startswith('es_') or col.startswith('vi_') or col.startswith('tr_')) 
    and 'both_match' in col
]
cross_lingual_both_columns = [
    col for col in df_target.columns if
    not col.startswith('en_') and 
    not col.startswith('es_') and 
    not col.startswith('vi_') and 
    not col.startswith('tr_') and
    'both_match' in col
]

# Calculate success rates by properly accounting for True values
def calculate_success_rate(dataframe, column_list):
    if not column_list or dataframe.empty:
        return 0.0
    
    # Count all True values across specified columns
    true_count = 0
    total_count = 0
    
    for col in column_list:
        true_count += dataframe[col].sum()
        total_count += len(dataframe)
    
    # Return percentage
    return (true_count / total_count) * 100 if total_count > 0 else 0.0

# Calculate success rates for each category using the filtered data
en_success_rate = calculate_success_rate(df_target, en_both_columns)
translated_success_rate = calculate_success_rate(df_target, translated_both_columns)
cross_lingual_success_rate = calculate_success_rate(df_target, cross_lingual_both_columns)

# Generate a detailed breakdown of columns for verification
language_breakdown = {
    'English': en_both_columns,
    'Translated': translated_both_columns,
    'Cross-lingual': cross_lingual_both_columns
}

# Print column counts for verification
for category, columns in language_breakdown.items():
    print(f"{category}: {len(columns)} columns")
    if len(columns) > 0:
        print(f"  Sample columns: {columns[:3]}")
        print(f"  True values: {df_target[columns].sum().sum()}")
        print(f"  Total cells: {len(df_target) * len(columns)}")

# Create the bar graph
categories = ['English', 'Translated\n(ES, VI, TR)', 'Cross-lingual']
success_rates = [en_success_rate, translated_success_rate, cross_lingual_success_rate]

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, success_rates, color=['#3274A1', '#E1812C', '#3A923A'])

# Add data labels on top of the bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{success_rates[i]:.1f}%',
             ha='center', va='bottom', fontweight='bold')

# Add titles and labels
plt.title(f'Success Rates for "{target_book}" by Language Category', fontsize=15)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.ylim(0, max(max(success_rates) * 1.2, 5))  # Add some space above the highest bar for labels
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a detailed breakdown in a text box
textstr = '\n'.join((
    f'BOOK: {target_book}',
    f'Data points: {len(df_target)} rows',
    '',
    'ENGLISH:',
    f'  Success Rate: {en_success_rate:.1f}%',
    f'  True Values: {df_target[en_both_columns].sum().sum()}',  
    f'  Total Cells: {len(df_target) * len(en_both_columns)}',
    '',
    'TRANSLATIONS:',
    f'  Success Rate: {translated_success_rate:.1f}%',
    f'  True Values: {df_target[translated_both_columns].sum().sum()}',
    f'  Total Cells: {len(df_target) * len(translated_both_columns)}',
    '',
    'CROSS-LINGUAL:',
    f'  Success Rate: {cross_lingual_success_rate:.1f}%',
    f'  True Values: {df_target[cross_lingual_both_columns].sum().sum()}',
    f'  Total Cells: {len(df_target) * len(cross_lingual_both_columns)}'
))

# Place the text box in the lower right corner
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.figtext(0.72, 0.15, textstr, fontsize=9,
            verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.savefig(f'{target_book}_evaluation_results.png', dpi=300)

# Create a more detailed language-specific analysis
language_prefixes = {
    'en': 'English',
    'es': 'Spanish', 
    'vi': 'Vietnamese',
    'tr': 'Turkish',
    'st': 'Sesotho',
    'yo': 'Yoruba',
    'tn': 'Tswana',
    'ty': 'Tahitian',
    'mai': 'Maithili',
    'mg': 'Malagasy'
}

# Calculate success rate for each language using filtered data
language_rates = {}
for prefix, language in language_prefixes.items():
    cols = [col for col in df_target.columns if col.startswith(f"{prefix}_") and 'both_match' in col]
    if cols:
        language_rates[language] = calculate_success_rate(df_target, cols)

# Filter out languages with zero data
language_rates = {k: v for k, v in language_rates.items() if v > 0}

if language_rates:
    # Create a second figure for language-specific breakdown
    plt.figure(figsize=(12, 8))
    langs = list(language_rates.keys())
    rates = list(language_rates.values())

    # Sort by success rate
    sorted_indices = np.argsort(rates)[::-1]  # Descending order
    sorted_langs = [langs[i] for i in sorted_indices]
    sorted_rates = [rates[i] for i in sorted_indices]

    # Color bars by category
    colors = []
    for lang in sorted_langs:
        if lang == 'English':
            colors.append('#3274A1')  # Blue for English
        elif lang in ['Spanish', 'Vietnamese', 'Turkish']:
            colors.append('#E1812C')  # Orange for translations
        else:
            colors.append('#3A923A')  # Green for cross-lingual

    bars = plt.bar(sorted_langs, sorted_rates, color=colors)

    # Add data labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{sorted_rates[i]:.1f}%',
                 ha='center', va='bottom', fontsize=9)

    plt.title(f'Success Rates for "{target_book}" by Specific Language', fontsize=15)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(max(sorted_rates) * 1.2, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{target_book}_evaluation_by_language.png', dpi=300)

plt.show()