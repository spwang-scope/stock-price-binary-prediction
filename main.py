import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time

from models.lstm_model import LSTMModel
from models.tcn_model import TCNModel
from models.transformer_model import TransformerModel
from models.mlp_model import MLPModel
from dataset import create_dataloaders_from_file
from trainer import Trainer

def create_model(model_type, input_size, **kwargs):
    """
    Create a model of the specified type
    
    Args:
        model_type (str): Type of model to create ('lstm', 'tcn', or 'transformer')
        input_size (int): Number of input features
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        model: Created model
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMModel(input_size, **kwargs)
    elif model_type == 'tcn':
        return TCNModel(input_size, **kwargs)
    elif model_type == 'transformer':
        return TransformerModel(input_size, **kwargs)
    elif model_type == 'mlp':
        return MLPModel(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main(args):
    """
    Main function to run the training pipeline
    
    Args:
        args: Command line arguments
    """
    # Create data loaders
    train_loader, val_loader, test_loader, neg_pos_ratio = create_dataloaders_from_file(
        file_path=args.file,
        batch_size=args.batch_size,
        context_window=args.context_window,
        num_workers=args.num_workers,
        rope_embedding_dim=args.rope_embedding_dim
    )
    
    # Determine input size from the first batch
    for features, _ in train_loader:
        input_size = features.shape[2]  # [batch_size, seq_len, input_size]
        break
    
    # Create model
    model_kwargs = {}
    if args.model_type == 'lstm':
        model_kwargs = {
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
    elif args.model_type == 'tcn':
        model_kwargs = {
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'kernel_size': args.kernel_size,
            'dropout': args.dropout
        }
    elif args.model_type == 'transformer':
        model_kwargs = {
            'd_model': args.hidden_size,
            'nhead': args.num_heads,
            'num_encoder_layers': args.num_layers,
            'num_decoder_layers': 1,
            'dim_feedforward': args.hidden_size * 4,
            'dropout': args.dropout
        }
    elif args.model_type == 'mlp':
        model_kwargs = {
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
    
    model = create_model(args.model_type, input_size, **model_kwargs)
    print(f"Created {args.model_type.upper()} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = None
    if args.scheduler == 'reduce':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        bce_weight=neg_pos_ratio
    )
    
    # Train model
    start_time = time.time()
    history = trainer.train(args.epochs, clip_grad_norm=args.clip_grad_norm)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training history
    if args.plot:
        trainer.plot_history()
    
    # Test model
    results = trainer.test(detailed=True)
    
    # Save model
    if args.save_model:
        os.makedirs('models/saved', exist_ok=True)
        model_path = f"models/saved/{args.model_type}_{args.context_window}_{args.hidden_size}.pt"
        trainer.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    return model, trainer, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train time series classification models")
    
    # Data arguments
    parser.add_argument('--file', type=str, required=True, help='Path to the merged CSV file')
    parser.add_argument('--context_window', type=int, default=10, help='Context window size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'tcn', 'transformer', 'mlp'], 
                        help='Type of model to use')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size (TCN only)')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads (Transformer only)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'reduce', 'cosine'], 
                        help='Learning rate scheduler')
    
    # Other arguments
    parser.add_argument('--save_model', action='store_true', help='Save model after training')
    parser.add_argument('--plot', action='store_true', help='Plot training history')

    parser.add_argument('--rope_embedding_dim', type=int, default=0, help='Apply feature dimension of rope embedding (Must be even number)')
    
    args = parser.parse_args()
    
    main(args)