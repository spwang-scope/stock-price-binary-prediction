import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from enum import Enum
from embedding_util import add_rope_embeddings_to_df
import random

class DatasetType(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

class StockDataset(Dataset):
    def __init__(self, df, dataset_type=DatasetType.TRAIN, context_window=10, train_split=0.8, val_split=0.1, rope=False):
        """
        Custom dataset for stock price data with context window
        
        Args:
            df (pandas.DataFrame): DataFrame containing preprocessed stock data
            dataset_type (DatasetType): Type of dataset to create (train, validation, or test)
            context_window (int): Number of preceding days to include as context
            train_split (float): Portion of data to use for training
            val_split (float): Portion of data to use for validation
        """
        self.context_window = context_window
        self.neg_pos_ratio = 1
        
        # Copy the dataframe to avoid modifying the original
        self.data = df.copy()
        
        # Ensure data is sorted by date within each ticker
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values(['unique_id', 'Date'])
        
        # Get unique tickers
        self.tickers = self.data['unique_id'].unique()
        
        # Split by ticker to maintain time series integrity within each stock
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for ticker in self.tickers:
            ticker_data = self.data[self.data['unique_id'] == ticker]
            
            # Calculate split indices for this ticker
            total_len = len(ticker_data)
            train_end = int(total_len * train_split)
            val_end = train_end + int(total_len * val_split)
            
            # Split data
            train_dfs.append(ticker_data.iloc[:train_end])
            val_dfs.append(ticker_data.iloc[train_end:val_end])
            test_dfs.append(ticker_data.iloc[val_end:])
        
        # Select the appropriate subset based on dataset_type
        if dataset_type == DatasetType.TRAIN:
            self.data = pd.concat(train_dfs, ignore_index=True)
        elif dataset_type == DatasetType.VALIDATION:
            self.data = pd.concat(val_dfs, ignore_index=True)
        elif dataset_type == DatasetType.TEST:
            self.data = pd.concat(test_dfs, ignore_index=True)
        
        # Ensure we have some data
        if len(self.data) == 0:
            raise ValueError(f"No data available for {dataset_type}")
        
        # Store the original index for reference
        self.data['original_idx'] = self.data.index
        
        # Create ticker-segregated dataframes for efficient retrieval by ticker
        self.ticker_data = {}
        for ticker in self.tickers:
            ticker_subset = self.data[self.data['unique_id'] == ticker]
            if not ticker_subset.empty:
                self.ticker_data[ticker] = ticker_subset
        
        # Extract features, labels, and metadata
        # Exclude 'Date', 'unique_id', 'original_idx', and 'y' from features
        self.unique_ids = self.data['unique_id'].values
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['y', 'original_idx']]
        
        # Store feature column names for reference
        self.feature_cols = feature_cols
        
        # Store the original features and labels
        self.features = self.data[feature_cols].values.astype(np.float32)
        self.labels = self.data['y'].values.astype(np.float32)
        
        # Create a list of valid indices
        self.valid_indices = []
        
        # For each ticker, add indices that have enough preceding context
        for ticker, ticker_df in self.ticker_data.items():
            if len(ticker_df) > context_window:
                # Only include indices that have at least context_window preceding records
                valid_rows = ticker_df.iloc[context_window:]['original_idx'].values
                self.valid_indices.extend(valid_rows)

    def __len__(self):
        # Return the number of valid samples
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a single item with context window.
        
        For each idx, returns:
        - Feature tensor of shape (context_window+1, num_features)
          This includes the current row and context_window preceding rows of SAME ticker
        - Label for the current row
        """

        # Get the actual index in the dataset
        original_idx = self.valid_indices[idx]
        
        # Get the row from the original dataset
        current_row = self.data.loc[self.data['original_idx'] == original_idx].iloc[0]

        # If the label is NaN, try with other random number
        while True:
            if not pd.isna(current_row['y']):
                break
            else:
                idx = random.randint(0, len(self) - 1)
                original_idx = self.valid_indices[idx]
                current_row = self.data.loc[self.data['original_idx'] == original_idx].iloc[0]

        # Get the ticker name
        ticker = current_row['unique_id']
        
        # Get the ticker-specific dataframe
        ticker_df = self.ticker_data[ticker]
        
        # Find the position of the current row in the ticker dataframe
        ticker_idx = ticker_df.index[ticker_df['original_idx'] == original_idx].item()
        ticker_pos = ticker_df.index.get_loc(ticker_idx)
        
        # Ensure we have enough context (should always be true based on valid_indices)
        if ticker_pos < self.context_window:
            # This is a safety check, shouldn't happen
            # If it does, use padding with the earliest available row
            padding_needed = self.context_window - ticker_pos
            earliest_features = ticker_df.iloc[0][self.feature_cols].values.astype(np.float32)
            
            # Get the available context
            context_features = ticker_df.iloc[:ticker_pos+1][self.feature_cols].values.astype(np.float32)
            
            # Create padding
            padding = np.tile(earliest_features, (padding_needed, 1))
            
            # Combine padding with available context
            feature_window = np.vstack([padding, context_features])
        else:
            # Get the appropriate context window
            feature_window = ticker_df.iloc[ticker_pos-self.context_window:ticker_pos+1][self.feature_cols].values.astype(np.float32)
            
            # Get the label
            label = current_row['y']
            
            # Convert to tensors
            feature_tensor = torch.tensor(feature_window, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return feature_tensor, label_tensor

def preprocess_merged_data(file_path):
    """
    Load and preprocess a merged stock data file
    
    Args:
        file_path (str): Path to the merged CSV file
        
    Returns:
        pandas.DataFrame: Preprocessed dataframe with added 'y' column
    """
    # Read the data
    df = pd.read_csv(file_path)
    
    # Ensure Date is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by ticker and date
    df = df.sort_values(['unique_id', 'Date'])
    
    # Drop rows with NaN (will be the first row for each ticker since it has no previous day)
    df = df.dropna()
    
    neg_pos_ratio = (len(df) - df['y'].sum())/(df['y'].sum() + 1e-5)
    
    return df, neg_pos_ratio

def print_df_info(df):
    print(f"Loaded {len(df)} rows after preprocessing")
    print(f"Unique tickers: {df['unique_id'].nunique()}")
    print(f"Features: {[col for col in df.columns if col not in ['Date', 'unique_id', 'y']]}")
    print(f"Class distribution - 1: {df['y'].sum()}, 0: {len(df) - df['y'].sum()}")


def create_dataloaders_from_file(file_path, batch_size=32, context_window=10, train_split=0.8, val_split=0.1, num_workers=4, rope=False):
    """
    Create train, validation, and test DataLoaders from a merged stock data file
    
    Args:
        file_path (str): Path to the merged CSV file
        batch_size (int): Batch size for the DataLoader
        context_window (int): Number of preceding days to include as context
        train_split (float): Portion of data to use for training
        val_split (float): Portion of data to use for validation
        num_workers (int): Number of worker threads for loading data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
   
    # Preprocess the data
    df, neg_pos_ratio = preprocess_merged_data(file_path)

    if rope:
        df = add_rope_embeddings_to_df(df)

    print_df_info(df)
    
    # Create datasets
    train_dataset = StockDataset(
        df, 
        dataset_type=DatasetType.TRAIN, 
        context_window=context_window,
        train_split=train_split,
        val_split=val_split
    )
    
    val_dataset = StockDataset(
        df, 
        dataset_type=DatasetType.VALIDATION, 
        context_window=context_window,
        train_split=train_split,
        val_split=val_split
    )
    
    test_dataset = StockDataset(
        df, 
        dataset_type=DatasetType.TEST, 
        context_window=context_window,
        train_split=train_split,
        val_split=val_split
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, neg_pos_ratio
