import pandas as pd
import numpy as np
import torch
from datetime import datetime
from rope_embedding import ScalarRoPEEmbedding

def add_rope_embeddings_to_df(df, date_column='Date', emb_size=8):
    """
    Add RoPE embeddings to a DataFrame based on date differences.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing a date column
        date_column (str): Name of the column containing dates
        emb_size (int): Size of the RoPE embedding vectors (must be even)
        
    Returns:
        pd.DataFrame: DataFrame with added RoPE embedding columns
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Ensure the date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
        result_df[date_column] = pd.to_datetime(result_df[date_column])
    
    # Find the earliest date
    earliest_date = result_df[date_column].min()
    print(f"Earliest date in the dataset: {earliest_date}")
    
    # Calculate day differences from the earliest date
    day_diffs = (result_df[date_column] - earliest_date).dt.days
    
    # Initialize the RoPE embedding module
    rope_model = ScalarRoPEEmbedding(embedding_dim=emb_size)
    
    # Get the embeddings as numpy array
    embeddings = rope_model.get_numpy_embeddings(day_diffs.values)
    
    # Add embeddings as new columns to the DataFrame
    for i in range(emb_size):
        result_df[f'time_emb{i}'] = embeddings[:, i]
    
    return result_df

def main():
    # Example usage
    # Replace 'your_stock_data.csv' with your actual data file
    df = pd.read_csv('your_stock_data.csv')
    
    # Sample output: print the first few rows of original dataframe
    print("Original DataFrame (first 5 rows):")
    print(df.head())
    
    # Process the dataframe with default embedding size of 8
    processed_df = add_rope_embeddings_to_df(df, emb_size=8)
    
    # Sample output: print the first few rows after adding embeddings
    print("\nProcessed DataFrame with RoPE embeddings (first 5 rows):")
    print(processed_df.head())
    
    # You can save the processed DataFrame to a new CSV if needed
    # processed_df.to_csv('processed_stock_data.csv', index=False)
    
if __name__ == "__main__":
    main()
