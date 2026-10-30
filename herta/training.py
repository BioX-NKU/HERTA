"""Training, validation and loss functions for multiresolution Hi-C prediction."""

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import pickle

def multiresolution_weighted_mse_loss(predictions, targets, min_stripe_signal=-99999, epoch=None):
    """
    Multi-resolution weighted MSE loss for V-stripe outputs.
    The original function name is preserved for compatibility.
    """
    resolution_weights = {
        '25kb': 10,
        '10kb': 10, 
        '5kb': 10,
    }
    
    stage_weights = {
        'stage1': 1.0,
    }
    
    if epoch is not None:
        print(f"\n=== VSTRIPE LOSS FUNCTION DEBUG (Epoch {epoch}) ===")
        print(f"Available predictions stages: {list(predictions.keys())}")
        print(f"Available targets: {list(targets.keys())}")
    
    total_loss = 0.0
    
    for stage_name, stage_weight in stage_weights.items():
        if stage_name not in predictions:
            if epoch is not None:
                print(f"WARNING: {stage_name} not in predictions!")
            continue
            
        stage_predictions = predictions[stage_name]
        stage_loss = 0.0
        
        for resolution, res_weight in resolution_weights.items():
            target_key = f'target_{resolution}'
            
            if target_key in targets and resolution in stage_predictions:
                pred_vstripe = stage_predictions[resolution]    # [B, vstripe_length]
                target_vstripe = targets[target_key].to(pred_vstripe.device)  # [B, vstripe_length]
                
                if epoch is not None:
                    print(f"  {resolution}:")
                    print(f"    pred_vstripe shape: {pred_vstripe.shape}")
                    print(f"    target_vstripe shape: {target_vstripe.shape}")
                    print(f"    shapes match: {pred_vstripe.shape == target_vstripe.shape}")
                
                # Ensure matching tensor shapes.
                if pred_vstripe.shape != target_vstripe.shape:
                    if epoch is not None:
                        print(f"    ERROR: Shape mismatch! Skipping {resolution}")
                    continue
                
                # Compute MSE directly on V-stripe vectors.
                res_loss = nn.functional.mse_loss(pred_vstripe, target_vstripe)
                weighted_res_loss = res_weight * res_loss
                stage_loss += weighted_res_loss
                
                if epoch is not None:
                    print(f"    raw MSE: {res_loss.item():.6f}")
                    print(f"    weighted MSE: {weighted_res_loss.item():.6f}")
        
        total_loss += stage_weight * stage_loss
    
    if epoch is not None:
        print(f"\nFINAL TOTAL LOSS: {total_loss.item():.6f}")
        print("=" * 60)
    
    return total_loss

def train_pbulk_multilevel(
    train_loader, model, criterion, optimizer, device, min_stripe_signal, epoch=None
):
    """
    Training loop for the 3-track Pbulk+CTCF model.
    The original function name is preserved for compatibility.
    """
    model.train()
    running_loss = 0
    batch_count = 0
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            optimizer.zero_grad()
            
            # Read batch tensors.
            X1 = batch['pbulk'].to(device)      # [B, 80200]
            X2 = batch['ctcf'].to(device)       # [B, 80200]
            
            # Add channel dimensions and concatenate Pbulk with CTCF.
            X1 = X1.unsqueeze(1)                # [B, 80200] -> [B, 1, 80200]
            X2 = X2.unsqueeze(1)                # [B, 80200] -> [B, 1, 80200]
            X = torch.cat((X1, X2), dim=1)      # [B, 2, 80200] 
            
            # Prepare multi-resolution V-stripe targets.
            targets = {
                'target_5kb': batch['target_5kb'].to(device),    # [B, 400]
                'target_10kb': batch['target_10kb'].to(device),  # [B, 200]
                'target_25kb': batch['target_25kb'].to(device)   # [B, 80]
            }
            
            # Forward pass.
            predictions = model(X)
            
            # Debug information.
            if batch_count == 0:
                print(f"\n=== BATCH 0 TRAINING DEBUG ===")
                print(f"Input shapes:")
                print(f"  X1 original: {batch['pbulk'].shape}")
                print(f"  X2 original: {batch['ctcf'].shape}")
                print(f"  X final: {X.shape}")
                print(f"  X range: {X.min():.6f} to {X.max():.6f}")
                
                print(f"\nTarget shapes and ranges:")
                for key, target in targets.items():
                    print(f"  {key}: {target.shape}, range: {target.min():.6f} to {target.max():.6f}")
            
            debug_epoch = epoch if batch_count == 0 else None
            loss = criterion(predictions, targets, min_stripe_signal, debug_epoch)
            
            if batch_count == 0:
                print(f"\nReturned loss: {loss.item():.6f}")
                print("=" * 60)
            
            running_loss += loss.item()
            
            # Backpropagation.
            loss.backward()
            optimizer.step()
            
            batch_count += 1
            
    epoch_loss = running_loss / len(train_loader)
    return model, optimizer, epoch_loss

def validate_pbulk_multilevel(val_loader, model, criterion, device, min_stripe_signal):
    """
    Validation loop for the 3-track Pbulk+CTCF model.
    The original function name is preserved for compatibility.
    """
    model.eval()
    running_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Read batch tensors.
            X1 = batch['pbulk'].to(device)      # [B, 80200]
            X2 = batch['ctcf'].to(device)       # [B, 80200]
            
            # Add channel dimensions and concatenate Pbulk with CTCF.
            X1 = X1.unsqueeze(1)                # [B, 80200] -> [B, 1, 80200]
            X2 = X2.unsqueeze(1)                # [B, 80200] -> [B, 1, 80200]
            X = torch.cat((X1, X2), dim=1)      # [B, 2, 80200] 
            
            # Prepare multi-resolution V-stripe targets.
            targets = {
                'target_5kb': batch['target_5kb'].to(device),    # [B, 400]
                'target_10kb': batch['target_10kb'].to(device),  # [B, 200]
                'target_25kb': batch['target_25kb'].to(device)   # [B, 80]
            }
            
            # Forward pass.
            predictions = model(X)
            
            # Debug information.
            if batch_count == 0:
                print(f"\n=== VALIDATION BATCH 0 DEBUG ===")
                print(f"Val X shape: {X.shape}")
            
            debug_epoch = "VAL" if batch_count == 0 else None
            loss = criterion(predictions, targets, min_stripe_signal, debug_epoch)
            running_loss += loss.item()
            batch_count += 1
            
    epoch_loss = running_loss / len(val_loader)
    return model, epoch_loss

def training_loop_multilevel(
    train,
    validate,
    model,
    ct_list,
    seed,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    epochs,
    patience,
    save_path,
    device,
    min_stripe_signal,
    print_every=1,
    resolution_suffix="multiresolution"  # Additional argument.
):
    """
    Multi-resolution training loop with epoch-aware optional debug logging.
    The original function name is preserved for compatibility.
    """
    # set objects for storing metrics
    counter = 0
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    SAVEPATH = os.path.join(
        save_path, "{}_{}_CTCFmotifScore_seed{}.pth.tar".format("_".join(ct_list), resolution_suffix, seed)
    )

    # Train model
    for epoch in range(epochs):
        print(f"\n{'='*20} EPOCH {epoch} {'='*20}")
        
        # Training step with epoch passed through for optional debug logging.
        model, optimizer, train_loss = train(
            train_loader, model, criterion, optimizer, device, min_stripe_signal, epoch
        )
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(
                valid_loader, model, criterion, device, min_stripe_signal
            )
            valid_losses.append(valid_loss)

        counter += 1

        if valid_loss < best_loss:
            counter = 0
            best_loss = valid_loss
            print("++++saving+++++")
            torch.save(model.state_dict(), SAVEPATH)

        if epoch % print_every == (print_every - 1):
            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.6f}\t"
                f"Valid loss: {valid_loss:.6f}\t"
            )

        if counter > patience:
            print("Early stopping.")
            break

        scheduler.step()

    return model, optimizer, (train_losses, valid_losses)

def train_pbulk_topic_multilevel(
    train_loader, model, criterion, optimizer, device, min_stripe_signal, epoch=None
):
    """
    Training loop for the final 6-track Pbulk+CTCF+Topic model.
    """
    model.train()
    running_loss = 0
    batch_count = 0
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            optimizer.zero_grad()
            
            # Read batch tensors; co-accessibility is not used.
            X1 = batch['pbulk'].to(device)       # [B, 80200]
            X2 = batch['ctcf'].to(device)        # [B, 80200]
            topics = batch['topic'].to(device)   # [B, 8020, 50]
            
            # Add channel dimensions and concatenate Pbulk with CTCF.
            X1 = X1.unsqueeze(1)                 # [B, 80200] -> [B, 1, 80200]
            X2 = X2.unsqueeze(1)                 # [B, 80200] -> [B, 1, 80200]
            X = torch.cat((X1, X2), dim=1)       # [B, 2, 80200]
            
            # Prepare multi-resolution targets.
            targets = {
                'target_5kb': batch['target_5kb'].to(device),
                'target_10kb': batch['target_10kb'].to(device),
                'target_25kb': batch['target_25kb'].to(device)
            }
            
            # Forward pass with Pbulk+CTCF and topic inputs.
            predictions = model(X, topics)
            
            # Debug information.
            if batch_count == 0:
                print(f"\n=== PBULK+TOPIC BATCH 0 TRAINING DEBUG ===")
                print(f"Input shapes:")
                print(f"  X (pbulk+ctcf): {X.shape}")
                print(f"  X range: {X.min():.6f} to {X.max():.6f}")
                print(f"  topics: {topics.shape}, range: {topics.min():.6f} to {topics.max():.6f}")
                
                print(f"\nTarget shapes and ranges:")
                for key, target in targets.items():
                    print(f"  {key}: {target.shape}, range: {target.min():.6f} to {target.max():.6f}")
            
            debug_epoch = epoch if batch_count == 0 else None
            loss = criterion(predictions, targets, min_stripe_signal, debug_epoch)
            
            if batch_count == 0:
                print(f"\nReturned loss: {loss.item():.6f}")
                print("=" * 60)
            
            running_loss += loss.item()
            
            # Backpropagation.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100000)
            optimizer.step()
            
            batch_count += 1
    
    epoch_loss = running_loss / len(train_loader)
    return model, optimizer, epoch_loss

def validate_pbulk_topic_multilevel(val_loader, model, criterion, device, min_stripe_signal):
    """
    Validation loop for the final 6-track Pbulk+CTCF+Topic model.
    """
    model.eval()
    running_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Read batch tensors; co-accessibility is not used.
            X1 = batch['pbulk'].to(device)       # [B, 80200]
            X2 = batch['ctcf'].to(device)        # [B, 80200]
            topics = batch['topic'].to(device)   # [B, 8020, 50]
            
            # Add channel dimensions and concatenate Pbulk with CTCF.
            X1 = X1.unsqueeze(1)
            X2 = X2.unsqueeze(1)
            X = torch.cat((X1, X2), dim=1)
            
            # Prepare multi-resolution targets.
            targets = {
                'target_5kb': batch['target_5kb'].to(device),
                'target_10kb': batch['target_10kb'].to(device),
                'target_25kb': batch['target_25kb'].to(device)
            }
            
            # Forward pass with Pbulk+CTCF and topic inputs.
            predictions = model(X, topics)
            
            # Debug information.
            if batch_count == 0:
                print(f"\n=== PBULK+TOPIC VALIDATION BATCH 0 DEBUG ===")
                print(f"Val X (pbulk+ctcf): {X.shape}")
                print(f"Val topics: {topics.shape}")
                print(f"Val predictions available: {list(predictions.keys())}")
                print(f"Val targets available: {list(targets.keys())}")
            
            debug_epoch = "VAL" if batch_count == 0 else None
            loss = criterion(predictions, targets, min_stripe_signal, debug_epoch)
            
            running_loss += loss.item()
            batch_count += 1
    
    epoch_loss = running_loss / len(val_loader)
    return model, epoch_loss

