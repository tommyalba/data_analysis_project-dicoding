import pandas as pd
from pathlib import Path
# ======================================================================

def load_and_inspect_ecommerce_data(
    data_dir: str = "data/E-Commerce Public Dataset",
) -> dict:
    """Loads all 9 CSV files from the Olist E-Commerce dataset, 
    cleans duplicate rows, displays row counts before and after deduplication, 
    and identifies potential Primary Key columns.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing the CSV files.
    
    Returns:
    --------
    dict
        A dictionary containing dataset file names as keys and cleaned DataFrames
        as values.
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in directory: {data_path}")
        return {}
    
    cleaned_datasets = {}
    
    for file_path in sorted(csv_files):
        file_name = file_path.name
        
        print(f"\n{'='*65}")
        print(f"Dataset: {file_name}")
        print(f"{'='*65}")
    
        # 1. Load dataset efficiently
        df = pd.read_csv(file_path)
        initial_row_count = len(df)
    
        # 2. Check and remove duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            final_row_count = len(df)
            print(f"[Duplicate Check] Removed {duplicate_count} duplicate rows.")
        else:
            final_row_count = initial_row_count
            print("[Duplicate Check] No duplicate rows found.")
    
        print(f"Row count -> Before: {initial_row_count:,} | After:" f" {final_row_count:,}")
    
        # 3. Identify potential Primary Key columns
        # A strict single-column Primary Key requires 100% unique values and zero nulls.
        potential_pks = []
        for col in df.columns:
            is_unique = df[col].nunique() == final_row_count
            has_no_nulls = df[col].isnull().sum() == 0
        
        if is_unique and has_no_nulls:
            potential_pks.append(col)
    
        if potential_pks:
            print(f"Identified Potential Primary Key(s): {potential_pks}")
        else:
            print(
                "Primary Key Status: No single column satisfies strict uniqueness and"
                " non-null constraints (likely uses a composite key)."
        )
        cleaned_datasets[file_name] = df
    
    print(f"\n{'='*65}")
    print(f"Successfully loaded and processed {len(cleaned_datasets)} datasets.")
    print(f"{'='*65}")
    return cleaned_datasets

# =========================

# Summary of Missing Values
def missing_summary(dataframe):
    """
    Creates a summary of missing values ​​for each column in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to be examined.
    
    Returns:
        pd.DataFrame: Summary of the number and percentage of missing values.
    """
    na_counts = dataframe.isna().sum()[dataframe.isna().sum() > 0]
    na_percentage = dataframe.isna().sum()[dataframe.isna().sum() > 0] / len(dataframe) * 100
    
    summary = pd.DataFrame({
        'Missing Values': na_counts,
        'Percentage': na_percentage.round(2),
        'Total length': len(dataframe)
    })
    return summary