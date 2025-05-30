import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

def clean_book_name(filename):
    """
    Removes “.csv” if present and strips any known suffixes (e.g. “_non_NE”) 
    to yield a base book name.
        • Input: filename such as alice_in_wonderland_non_NE.csv
        • Output: string “alice_in_wonderland_non_NE”
    """
    # Remove the file extension if present.
    if filename.endswith('.csv'):
        name = filename[:-4]
    else:
        name = filename
    # Just extract book name but preserve masked/non_NE flags for categorization
    return name

def create_book_model_heatmaps(occurrence_csv, accuracy_csv, save_path):
    """
    a. Loads the occurrence CSV, expects a column named “filename” plus one of “Occurrences”, “Count”, or “Percentage” (or else uses the second column by default).
    b. Cleans filenames into book names, sorts books by their occurrence metric (highest first).
    c. Loads the aggregated accuracy CSV, expects a column “Book Name” and accuracy columns prefixed “masked_” or “non_NE_” or with no prefix (unmasked).
    d. Splits accuracy data into three DataFrames:
        – masked entries (columns starting with “masked_”)
        – non‑NE entries (columns starting with “non_NE_”)
        – unmasked entries (all other model columns)
    e. Matches book names between occurrence and accuracy data (case‑insensitive), filters to books present in both, and reorders them by occurrence.
    f. For each category, creates a composite heatmap: accuracy layer with a blue‑purple colormap and occurrence layer with a gold colormap, annotates values, and adjusts sizes based on number of books.
    g. Saves PNGs named:
        masked_book_model_accuracy_heatmap.png
        non_NE_book_model_accuracy_heatmap.png
        unmasked_book_model_accuracy_heatmap.png
    h. Raises ValueError if required columns (“filename” or “Book Name”) are missing.
    """
    print(f"Loading occurrence CSV from: {occurrence_csv}")
    df_occ = pd.read_csv(occurrence_csv)
    print(f"Loaded occurrence data with {len(df_occ)} rows")
    
    # Make sure the occurrence CSV contains a "Filename" column.
    if "filename" not in df_occ.columns:
        raise ValueError("Occurrence CSV must have a 'filename' column.")
    
    # Create a new column with the cleaned book name
    df_occ['Book_Name'] = df_occ['filename'].apply(clean_book_name)
    
    # Determine which column to use as the occurrence metric.
    if 'Occurrences' in df_occ.columns:
        metric = 'Occurrences'
    elif 'Count' in df_occ.columns:
        metric = 'Count'
    elif 'Percentage' in df_occ.columns:
        metric = 'Percentage'
    else:
        # Fallback: use the second column.
        metric = df_occ.columns[1]
    
    print(f"Using {metric} as the sorting metric")
    
    # Sort the occurrence DataFrame by the chosen metric (descending).
    df_occ_sorted = df_occ.sort_values(by=metric, ascending=False)
    
    # Get a list of clean book names for ordering
    clean_book_names = df_occ['filename'].apply(
        lambda x: x.replace("_non_NE.csv", "").replace("_unmasked_passages.csv", "")
    ).unique().tolist()
    clean_book_names = [name.replace(".csv", "").replace("_", " ") for name in clean_book_names]
    print(f"Found {len(clean_book_names)} unique book names")
    
    # Create a dictionary mapping clean book names to their occurrence counts
    book_occurrence = {}
    for _, row in df_occ.iterrows():
        clean_name = row['filename'].replace("_non_NE.csv", "").replace("_unmasked_passages.csv", "")
        clean_name = clean_name.replace(".csv", "").replace("_", " ")
        clean_name = clean_name.lower()  # Case insensitive matching
        
        if clean_name in book_occurrence:
            # If this book already has an entry, keep the maximum occurrence value
            book_occurrence[clean_name] = max(book_occurrence[clean_name], row[metric])
        else:
            book_occurrence[clean_name] = row[metric]
    
    print(f"Created occurrence mapping for {len(book_occurrence)} books")
    
    # -------------------------------------------------------------
    # 2. Load Aggregated Accuracy CSV
    # -------------------------------------------------------------
    print(f"Loading accuracy CSV from: {accuracy_csv}")
    df_acc = pd.read_csv(accuracy_csv)
    print(f"Loaded accuracy data with {len(df_acc)} rows and {len(df_acc.columns)} columns")
    
    # Display all book names from the accuracy file for debugging
    print("Book names in accuracy file:", df_acc["Book Name"].tolist())
    
    # Verify the aggregated CSV has a "Book Name" column
    if "Book Name" not in df_acc.columns:
        raise ValueError("Aggregated CSV must contain a 'Book Name' column.")
    
    # -------------------------------------------------------------
    # 3. Create three separate dataframes based on column prefixes
    # -------------------------------------------------------------
    # Create separate dataframes for each category
    df_acc_masked = df_acc.set_index("Book Name").copy()
    df_acc_non_NE = df_acc.set_index("Book Name").copy()
    df_acc_unmasked = df_acc.set_index("Book Name").copy()
    
    # Identify columns for each category
    masked_cols = [col for col in df_acc.columns if col.startswith("masked_")]
    non_NE_cols = [col for col in df_acc.columns if col.startswith("non_NE_")]
    
    # Create list of regular columns (no prefix)
    all_model_cols = [col for col in df_acc.columns if col != "Book Name"]
    base_cols = [col for col in all_model_cols if not col.startswith("masked_") and not col.startswith("non_NE_")]
    
    print(f"Found {len(masked_cols)} masked columns, {len(non_NE_cols)} non_NE columns, {len(base_cols)} unmasked columns")
    
    # Keep only relevant columns for each category
    df_acc_masked = df_acc_masked[masked_cols]
    df_acc_non_NE = df_acc_non_NE[non_NE_cols]
    df_acc_unmasked = df_acc_unmasked[base_cols]
    
    # Remove prefix from masked and non_NE column names for cleaner display
    df_acc_masked.columns = [col.replace("masked_", "") for col in df_acc_masked.columns]
    df_acc_non_NE.columns = [col.replace("non_NE_", "") for col in df_acc_non_NE.columns]
    
    # 4. Sort book names based on occurrence data
    # Create a lookup dictionary for sorting (case-insensitive matching)
    book_rank = {book.lower(): idx for idx, book in enumerate(clean_book_names)}
    print("Book rank dictionary:", book_rank)
    
    # Check which books are being matched
    matched_books = []
    for book in df_acc_masked.index:
        if book.lower() in book_rank:
            matched_books.append(book)
    print(f"Matched books: {matched_books}")
    
    # Filter out books that don't have data and sort by occurrence frequency
    masked_books = [book for book in df_acc_masked.index if book.lower() in book_rank]
    masked_books.sort(key=lambda x: book_rank.get(x.lower(), 999))
    
    non_NE_books = [book for book in df_acc_non_NE.index if book.lower() in book_rank]
    non_NE_books.sort(key=lambda x: book_rank.get(x.lower(), 999))
    
    unmasked_books = [book for book in df_acc_unmasked.index if book.lower() in book_rank]
    unmasked_books.sort(key=lambda x: book_rank.get(x.lower(), 999))
    
    print(f"Books after matching: masked={len(masked_books)}, non_NE={len(non_NE_books)}, unmasked={len(unmasked_books)}")
    
    # Sort the dataframes
    df_acc_masked = df_acc_masked.loc[masked_books]
    df_acc_non_NE = df_acc_non_NE.loc[non_NE_books]
    df_acc_unmasked = df_acc_unmasked.loc[unmasked_books]
    
    # Add an "Occurrences" column to each dataframe showing the occurrence counts
    df_masked_occurrences = pd.DataFrame(index=masked_books)
    df_masked_occurrences["Occurrences"] = [book_occurrence.get(book.lower(), 0) for book in masked_books]
    
    df_non_NE_occurrences = pd.DataFrame(index=non_NE_books)
    df_non_NE_occurrences["Occurrences"] = [book_occurrence.get(book.lower(), 0) for book in non_NE_books]
    
    df_unmasked_occurrences = pd.DataFrame(index=unmasked_books)
    df_unmasked_occurrences["Occurrences"] = [book_occurrence.get(book.lower(), 0) for book in unmasked_books]
    
    # Concatenate the occurrence dataframes with the accuracy dataframes
    df_acc_masked = pd.concat([df_masked_occurrences, df_acc_masked], axis=1)
    df_acc_non_NE = pd.concat([df_non_NE_occurrences, df_acc_non_NE], axis=1)
    df_acc_unmasked = pd.concat([df_unmasked_occurrences, df_acc_unmasked], axis=1)
    
    print(f"Final dataframe sizes: masked={df_acc_masked.shape}, non_NE={df_acc_non_NE.shape}, unmasked={df_acc_unmasked.shape}")
    
    # -------------------------------------------------------------
    # 5. Create the heatmaps with blue-purple colormap
    # -------------------------------------------------------------
    # Create a custom blue-purple colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu',
        ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
        N=256
    )
    
    # Ensure the save directory exists
    print(f"Saving heatmaps to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # Create heatmap for masked data
    if not df_acc_masked.empty:
        print(f"Creating masked heatmap with {len(df_acc_masked)} books and {len(df_acc_masked.columns)} columns")
        
        # Split the dataframe into occurrences and accuracy
        occurrences_df = df_acc_masked[["Occurrences"]]
        accuracy_df = df_acc_masked.drop(columns=["Occurrences"])
        
        # Create a single heatmap with all data
        plt.figure(figsize=(14, len(df_acc_masked)*0.5 + 2))
        
        # Custom color mapping - use gold for occurrences and blue-purple for accuracy
        # First create a masked version of the dataframe where only occurrences are shown
        mask_acc = pd.DataFrame(True, index=df_acc_masked.index, columns=df_acc_masked.columns)
        mask_acc['Occurrences'] = False
        
        # Create a mask for the occurrence column
        mask_occ = pd.DataFrame(True, index=df_acc_masked.index, columns=df_acc_masked.columns)
        mask_occ.loc[:, 'Occurrences'] = False
        mask_occ = ~mask_occ
        
        # Create a single custom colormap for our composite heatmap
        occurrence_cmap = LinearSegmentedColormap.from_list(
            'occurrence_cmap', ['#f5f5f5', '#d8b365'], N=256
        )
        
        # Plot the heatmap with both colormaps
        ax = sns.heatmap(
            df_acc_masked,
            mask=mask_occ,
            annot=True,
            cmap=custom_cmap,
            cbar=True,
            fmt='.1f',
            linewidths=0.5,
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Accuracy (%)"}
        )
        
        # Add the occurrences heatmap on top with a different colorbar
        sns.heatmap(
            df_acc_masked,
            mask=mask_acc,
            annot=True,
            cmap=occurrence_cmap,
            cbar=True,
            fmt='.0f',
            linewidths=0.5,
            vmin=df_acc_masked['Occurrences'].min(),
            vmax=df_acc_masked['Occurrences'].max(),
            cbar_kws={"label": "Occurrences"}
        )
        
        plt.xlabel('Model Names', fontsize=14)
        plt.ylabel('Book Names', fontsize=14)
        plt.title('Masked Entries: Aggregated Accuracy Heatmap (Books Sorted by Occurrence)', fontsize=16)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_file = f"{save_path}/masked_book_model_accuracy_heatmap.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Masked heatmap saved to {save_file}")
    else:
        print("No masked data found, skipping masked heatmap")
    
    # Create heatmap for non_NE data
    if not df_acc_non_NE.empty:
        print(f"Creating non_NE heatmap with {len(df_acc_non_NE)} books and {len(df_acc_non_NE.columns)} columns")
        
        # Split the dataframe into occurrences and accuracy
        occurrences_df = df_acc_non_NE[["Occurrences"]]
        accuracy_df = df_acc_non_NE.drop(columns=["Occurrences"])
        
        # Create a single heatmap with all data
        plt.figure(figsize=(14, len(df_acc_non_NE)*0.5 + 2))
        
        # Custom color mapping - use gold for occurrences and blue-purple for accuracy
        # First create a masked version of the dataframe where only occurrences are shown
        mask_acc = pd.DataFrame(True, index=df_acc_non_NE.index, columns=df_acc_non_NE.columns)
        mask_acc['Occurrences'] = False
        
        # Create a mask for the occurrence column
        mask_occ = pd.DataFrame(True, index=df_acc_non_NE.index, columns=df_acc_non_NE.columns)
        mask_occ.loc[:, 'Occurrences'] = False
        mask_occ = ~mask_occ
        
        # Create a single custom colormap for our composite heatmap
        occurrence_cmap = LinearSegmentedColormap.from_list(
            'occurrence_cmap', ['#f5f5f5', '#d8b365'], N=256
        )
        
        # Plot the heatmap with both colormaps
        ax = sns.heatmap(
            df_acc_non_NE,
            mask=mask_occ,
            annot=True,
            cmap=custom_cmap,
            cbar=True,
            fmt='.1f',
            linewidths=0.5,
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Accuracy (%)"}
        )
        
        # Add the occurrences heatmap on top with a different colorbar
        sns.heatmap(
            df_acc_non_NE,
            mask=mask_acc,
            annot=True,
            cmap=occurrence_cmap,
            cbar=True,
            fmt='.0f',
            linewidths=0.5,
            vmin=df_acc_non_NE['Occurrences'].min(),
            vmax=df_acc_non_NE['Occurrences'].max(),
            cbar_kws={"label": "Occurrences"}
        )
        
        plt.xlabel('Model Names', fontsize=14)
        plt.ylabel('Book Names', fontsize=14)
        plt.title('Non-NE Entries: Aggregated Accuracy Heatmap (Books Sorted by Occurrence)', fontsize=16)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_file = f"{save_path}/non_NE_book_model_accuracy_heatmap.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Non-NE heatmap saved to {save_file}")
    else:
        print("No non_NE data found, skipping non_NE heatmap")
    
    # Create heatmap for unmasked data
    if not df_acc_unmasked.empty:
        print(f"Creating unmasked heatmap with {len(df_acc_unmasked)} books and {len(df_acc_unmasked.columns)} columns")
        
        # Split the dataframe into occurrences and accuracy
        occurrences_df = df_acc_unmasked[["Occurrences"]]
        accuracy_df = df_acc_unmasked.drop(columns=["Occurrences"])
        
        # Create a single heatmap with all data
        plt.figure(figsize=(14, len(df_acc_unmasked)*0.5 + 2))
        
        # Custom color mapping - use gold for occurrences and blue-purple for accuracy
        # First create a masked version of the dataframe where only occurrences are shown
        mask_acc = pd.DataFrame(True, index=df_acc_unmasked.index, columns=df_acc_unmasked.columns)
        mask_acc['Occurrences'] = False
        
        # Create a mask for the occurrence column
        mask_occ = pd.DataFrame(True, index=df_acc_unmasked.index, columns=df_acc_unmasked.columns)
        mask_occ.loc[:, 'Occurrences'] = False
        mask_occ = ~mask_occ
        
        # Create a single custom colormap for our composite heatmap
        occurrence_cmap = LinearSegmentedColormap.from_list(
            'occurrence_cmap', ['#f5f5f5', '#d8b365'], N=256
        )
        
        # Plot the heatmap with both colormaps
        ax = sns.heatmap(
            df_acc_unmasked,
            mask=mask_occ,
            annot=True,
            cmap=custom_cmap,
            cbar=True,
            fmt='.1f',
            linewidths=0.5,
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Accuracy (%)"}
        )
        
        # Add the occurrences heatmap on top with a different colorbar
        sns.heatmap(
            df_acc_unmasked,
            mask=mask_acc,
            annot=True,
            cmap=occurrence_cmap,
            cbar=True,
            fmt='.0f',
            linewidths=0.5,
            vmin=df_acc_unmasked['Occurrences'].min(),
            vmax=df_acc_unmasked['Occurrences'].max(),
            cbar_kws={"label": "Occurrences"}
        )
        
        plt.xlabel('Model Names', fontsize=14)
        plt.ylabel('Book Names', fontsize=14)
        plt.title('Unmasked Entries: Aggregated Accuracy Heatmap (Books Sorted by Occurrence)', fontsize=16)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_file = f"{save_path}/unmasked_book_model_accuracy_heatmap.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Unmasked heatmap saved to {save_file}")
    else:
        print("No unmasked data found, skipping unmasked heatmap")

# -------------------------------
# EXAMPLE USAGE: Adjust these paths as needed.
# -------------------------------
occurrence_csv = 'evals/utils/Percentage_of_Distinct_Passages_per_Filename.csv'
accuracy_csv   = 'evals/utils/aggregated_dp_acc.csv'
save_path      = 'evals/graphs'

create_book_model_heatmaps(occurrence_csv, accuracy_csv, save_path)
