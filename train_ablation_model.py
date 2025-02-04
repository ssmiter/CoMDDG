"""train_CoM_Bidirectional_sorted.py - Training script for the degree-sorted bidirectional Mamba model"""
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import NUM_EPOCHS, WEIGHT_DECAY, LEARNING_RATE, NUM_LAYERS, VAL_SPLIT
from model.CoM_Bidirectional_sorted_cognn_only import CoGNNOnly
from model.utils.loader.data_loader_add_mut_v2 import prepare_enhanced_data

# Configure logging
logging.basicConfig(level=logging.INFO)


def train_sorted_bidirectional_model(dataset='2648'):
    """
    Train the degree-sorted bidirectional model

    Args:
        dataset (str): Dataset identifier
    """
    # Load data
    train_loader, val_loader = prepare_enhanced_data(
        data_path=f"./dataset_process/pkl/data_s{dataset}_add_mutpos_enhanced_struct.pkl",
        batch_size=256,
        val_split=VAL_SPLIT
    )

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get feature dimensions from sample batch
    sample_wild, sample_mutant, _ = next(iter(train_loader))
    node_features = sample_wild.x.shape[1]
    edge_features = sample_wild.edge_attr.shape[1]
    print(f"Node features dimension: {node_features}")
    print(f"Edge features dimension: {edge_features}")

    # Initialize model
    model = CoGNNOnly(
        in_channels=node_features,
        hidden_channels=64,
        out_channels=1,
        # gmb_args={ # case1
        #     'd_model': 64,  # Match hidden_channels
        #     'd_state': 16,
        #     'd_conv': 4,
        #     'expand': 2,
        #     'use_checkpointing': True
        # },

        gmb_args={  # case2
            'd_model': 64,  # Match hidden_channels
            'd_state': 16,
            'd_conv': 2,
            'expand': 1,
            # 'use_checkpointing': True
        },
        num_layers=NUM_LAYERS
    ).to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=20,
        min_lr=0.00001,
        threshold=0.0002
    )

    # Loss function
    mae_criterion = nn.L1Loss()

    best_val_loss = float('inf')
    # best_model_path = f'./training/CoM_Bidirectional_Sorted_S{dataset}_struct.pth'
    # best_model_path = f'./training/CoM_Bidirectional_Sorted_S{dataset}_struct_cognn_Mamba_only_attempt_0.pth'
    best_model_path = f'./training/CoM_Bidirectional_Sorted_S{dataset}_struct_cognn_only_attempt_0.pth'
    def train_epoch():
        model.train()
        total_loss = 0

        for batch_idx, (wild_data, mutant_data, ddg) in enumerate(train_loader):
            # Move data to device
            wild_data = wild_data.to(device)
            mutant_data = mutant_data.to(device)
            ddg = ddg.to(device)

            optimizer.zero_grad()

            # Forward pass now only returns prediction
            output = model(wild_data, mutant_data)
            loss = mae_criterion(output, ddg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        return total_loss / len(train_loader)

    def evaluate(loader):
        model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for wild_data, mutant_data, ddg in loader:
                wild_data = wild_data.to(device)
                mutant_data = mutant_data.to(device)
                ddg = ddg.to(device)

                output = model(wild_data, mutant_data)
                loss = mae_criterion(output, ddg)

                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(ddg.cpu().numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        pcc = np.corrcoef(predictions, targets)[0, 1]

        return total_loss / len(loader), mae, mse, pcc

    # Start training
    print("Starting training for Degree-Sorted Bidirectional Mamba model...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Training phase
        train_loss = train_epoch()

        # Validation phase
        val_loss, val_mae, val_mse, val_pcc = evaluate(val_loader)

        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - start_time

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val MAE: {val_mae:.4f}, '
              f'Val MSE: {val_mse:.4f}, '
              f'Val PCC: {val_pcc:.4f}, '
              f'LR: {current_lr:.6f}, '
              f'Time: {epoch_duration:.2f}s')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'val_mae': val_mae,
                'val_mse': val_mse,
                'val_pcc': val_pcc
            }, best_model_path)
            print(f'Best model saved with validation loss: {best_val_loss:.4f}')

    print("Training completed!")

    # Evaluate best model
    print("\nEvaluating best model...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_val_loss, final_val_mae, final_val_mse, final_val_pcc = evaluate(val_loader)
    print(f'Best Model Metrics:'
          f'\n - Validation Loss: {final_val_loss:.4f}'
          f'\n - Validation MAE: {final_val_mae:.4f}'
          f'\n - Validation MSE: {final_val_mse:.4f}'
          f'\n - Validation PCC: {final_val_pcc:.4f}')


if __name__ == "__main__":
    train_sorted_bidirectional_model(dataset='2648')
