import os
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler

def calculate_moving_averages(df, column='Close', windows=[5, 10, 15, 20, 25, 30]):
    """Calculate moving averages for the specified windows."""
    # First check if the column exists
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in dataframe. Skipping MA calculations.")
        return df
        
    for window in windows:
        df[f'{column}_{window}day_MA'] = df[column].rolling(window=window).mean()
    
    # Note: This will create NaN values for the first (window-1) rows of each MA column
    # These will be handled during the normalization step
    return df

def normalize_numeric_columns(df, exclude_cols=None):
    """Normalize numeric columns using MinMaxScaler, excluding specified columns.
    
    Args:
        df: DataFrame to normalize
        exclude_cols: List of column names to exclude from normalization
    """
    # Initialize exclude_cols if None
    if exclude_cols is None:
        exclude_cols = []
        
    # Get numeric columns (excluding date type and string type)
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Exclude the specified columns from normalization
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    # Drop rows with NaN values in numeric columns (still check all numeric cols)
    df_clean = df.dropna(subset=numeric_cols)
    
    # If we lost rows, print a message
    rows_dropped = len(df) - len(df_clean)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with NaN values")
    
    # Apply MinMaxScaler only to columns that should be normalized
    if len(cols_to_normalize) > 0 and len(df_clean) > 0:
        scaler = MinMaxScaler()
        df_clean[cols_to_normalize] = scaler.fit_transform(df_clean[cols_to_normalize])
    
    return df_clean

def process_csv_file(file_path, output_dir=None):
    """Process a single CSV file according to requirements."""
    # Get the file name without extension
    file_name = os.path.basename(file_path)
    ticker = os.path.splitext(file_name)[0]
    
    # 1. Load file into dataframe and remove rows with NaN or zero Volume
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} with {len(df)} rows")
        
        # Check for empty dataframe
        if df.empty:
            print(f"Warning: Empty dataframe from {file_path}")
            return None, None
        
        # Remove rows where 'Volume' is NaN or zero
        if 'Volume' in df.columns:
            # Count rows before filtering
            initial_row_count = len(df)
            
            # Remove NaN and zero Volume rows
            df = df[df['Volume'].notna() & (df['Volume'] > 0)]
            
            # Report how many rows were removed
            rows_removed = initial_row_count - len(df)
            if rows_removed > 0:
                print(f"Removed {rows_removed} rows with NaN or zero Volume from {file_path}")
                
            if df.empty:
                print(f"Warning: No data left after removing NaN or zero Volume rows from {file_path}")
                return None, None
        else:
            print(f"Warning: 'Volume' column not found in {file_path}")
            
        # 2. Add unique_id column with ticker name
        df['unique_id'] = ticker
        
        # 3. Calculate moving averages
        df = calculate_moving_averages(df)
        
        # 4. Create 'y' column based on Close price increase rate
        df['y'] = pd.NA
        
        # Check if 'Close' column exists
        if 'Close' in df.columns:
            # Calculate percentage increase from previous row
            df['increase_rate'] = df['Close'].pct_change() * 100
            
            df.loc[df['increase_rate'] > 0.55, 'y'] = 1
            df.loc[df['increase_rate'] < -0.5, 'y'] = 0
            
            # Drop the temporary increase_rate column
            df = df.drop(columns=['increase_rate'])
        else:
            print(f"Warning: 'Close' column not found in {file_path}, cannot calculate increase rate")
        
        # 5. Normalize numeric columns (this will also drop rows with NaN)
        # Exclude 'y' column from normalization
        df = normalize_numeric_columns(df, exclude_cols=['y'])
        
        # Check if we still have data after removing NaNs
        if df.empty:
            print(f"Warning: No data left after removing NaN values from {file_path}")
            return None, None
        
        # 6. Save processed dataframe
        output_path = f"{ticker}_ma.csv"
        if output_dir:
            output_path = os.path.join(output_dir, output_path)
        
        df.to_csv(output_path, index=False)
        print(f"Saved processed file with {len(df)} rows to {output_path}")
        
        return df, output_path
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def main(datapath):
    """Process all CSV files in the specified directory and merge them."""
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(datapath, '*.csv'))
    
    # Filter out any *_ma.csv files that might already exist
    csv_files = [f for f in csv_files if not os.path.basename(f).endswith('_ma.csv') and not os.path.basename(f) == 'merged.csv']
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    processed_files = []
    all_dfs = []
    
    # Process each CSV file
    for file_path in csv_files:
        df, output_path = process_csv_file(file_path, datapath)
        if df is not None and output_path is not None:
            processed_files.append(output_path)
            all_dfs.append(df)
    
    # 6. Open all processed *_ma.csv files and merge them
    if processed_files:
        print(f"Opening {len(processed_files)} processed *_ma.csv files for merging")
        
        # Clear the previous list of dataframes to ensure we're only using the saved _ma.csv files
        all_dfs = []
        
        # Open each processed file
        for ma_file in processed_files:
            try:
                df = pd.read_csv(ma_file)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {ma_file}: {e}")
        
        # Merge all dataframes
        if all_dfs:
            print(f"Merging {len(all_dfs)} dataframes")
            merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)
            
            # 7. Sort the merged dataframe by the 'Date' column
            if 'Date' in merged_df.columns:
                print("Sorting the merged dataframe by 'Date' column")
                
                # Convert date column to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(merged_df['Date']):
                    try:
                        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
                    except Exception as e:
                        print(f"Warning: Could not convert 'Date' column to datetime: {e}")
                
                # Sort by date
                merged_df = merged_df.sort_values(by='Date')
                print(f"Sorted dataframe has {len(merged_df)} rows")
            else:
                print("Warning: 'Date' column not found, skipping sorting step")
            
            # Check the merged dataframe
            print(f"Final merged dataframe has {len(merged_df)} rows and {len(merged_df.columns)} columns")
            
            # Save the merged dataframe
            merged_path = os.path.join(datapath, 'merged.csv')
            merged_df.to_csv(merged_path, index=False)
            print(f"All processed dataframes merged and sorted to: {merged_path}")
            return merged_path
        else:
            print("No dataframes to merge after reading processed files.")
            return None
    else:
        print("No processed files to merge.")
        return None

if __name__ == "__main__":
    # Replace this with your actual datapath
    datapath = r"C:\Users\mira\time_experiment\data\kdd17\price_long_50"
    
    # Or use the datapath from variable if running in a context where it's defined
    # datapath variable is assumed to be defined elsewhere in the code
    
    main(datapath)
