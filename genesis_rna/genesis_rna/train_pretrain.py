"""
Pretraining Script for Genesis RNA with AST

Main training loop for RNA foundation model with:
- Multi-task learning (MLM + structure + pairing)
- Adaptive Sparse Training (AST) for efficiency
- Mixed precision training
- Checkpoint management
- Logging and evaluation
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .config import GenesisRNAConfig, TrainingConfig
from .tokenization import RNATokenizer, RNATokenizerConfig, SPECIAL_TOKENS, NUC_VOCAB
from .data import (
    RNAPretrainDataset,
    collate_pretrain_batch,
    create_dummy_dataset,
)
from .model import GenesisRNAModel
from .losses import MultiTaskLoss, compute_metrics
from .ast_wrapper import ASTSampleSelector


def setup_training(
    model_config: GenesisRNAConfig,
    train_config: TrainingConfig,
) -> tuple:
    """
    Set up model, optimizer, and training components.

    Returns:
        (model, optimizer, loss_fn, ast_selector, scaler)
    """
    # Set device
    device = torch.device(train_config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = GenesisRNAModel(model_config)
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    # Create loss function
    loss_fn = MultiTaskLoss(
        mlm_weight=train_config.mlm_loss_weight,
        structure_weight=train_config.structure_loss_weight,
        pair_weight=train_config.pair_loss_weight,
    )

    # Create AST sample selector
    ast_selector = None
    if train_config.use_ast:
        ast_selector = ASTSampleSelector(
            target_activation=train_config.ast_target_activation,
            controller_kp=train_config.ast_controller_kp,
            controller_ki=train_config.ast_controller_ki,
            selection_mode='loss',
            use_pi_controller=True,
        )
        print(f"AST enabled with target activation: {train_config.ast_target_activation}")

    # Create gradient scaler for mixed precision
    scaler = GradScaler() if train_config.fp16 else None

    return model, optimizer, loss_fn, ast_selector, scaler, device


def get_lr_scheduler(
    optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
):
    """Create learning rate scheduler with warmup"""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskLoss,
    ast_selector: Optional[ASTSampleSelector],
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    train_config: TrainingConfig,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dictionary of average metrics
    """
    model.train()

    total_loss = 0
    total_mlm_loss = 0
    total_struct_loss = 0
    total_pair_loss = 0
    total_samples = 0
    total_selected = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        batch_size = batch['input_ids'].size(0)

        # Forward pass (compute outputs for all samples first)
        with autocast(enabled=scaler is not None):
            outputs = model(
                batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

        # AST sample selection
        selected_idx = None
        if ast_selector is not None:
            # Compute per-sample losses for selection
            with torch.no_grad():
                # Use MLM loss as importance metric
                token_losses = torch.nn.functional.cross_entropy(
                    outputs['mlm_logits'].view(-1, outputs['mlm_logits'].size(-1)),
                    batch['mlm_labels'].view(-1),
                    ignore_index=-100,
                    reduction='none'
                ).view(batch_size, -1)

                # Average over sequence
                mask = (batch['mlm_labels'] != -100).float()
                sample_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Select important samples
            selected_idx = ast_selector.select_indices(sample_losses)

            if selected_idx.numel() == 0:
                # No samples selected - skip this batch
                continue

            # Update statistics
            total_selected += selected_idx.numel()

            # Filter batch and outputs to selected samples
            batch_selected = {k: v[selected_idx] if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
            outputs_selected = {k: v[selected_idx] for k, v in outputs.items()}
        else:
            batch_selected = batch
            outputs_selected = outputs
            total_selected += batch_size

        # Compute loss
        with autocast(enabled=scaler is not None):
            loss_dict = loss_fn(outputs_selected, batch_selected)
            loss = loss_dict['loss']

        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            optimizer.step()

        scheduler.step()

        # Update statistics
        total_loss += loss.item() * (selected_idx.numel() if selected_idx is not None else batch_size)
        total_mlm_loss += loss_dict['mlm_loss'].item() * (selected_idx.numel() if selected_idx is not None else batch_size)
        total_struct_loss += loss_dict['structure_loss'].item() * (selected_idx.numel() if selected_idx is not None else batch_size)
        total_pair_loss += loss_dict['pair_loss'].item() * (selected_idx.numel() if selected_idx is not None else batch_size)
        total_samples += batch_size

        # Update progress bar
        activation_rate = (selected_idx.numel() / batch_size) if selected_idx is not None else 1.0
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'act_rate': f"{activation_rate:.2f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}",
        })

    # Compute averages
    avg_metrics = {
        'loss': total_loss / total_selected,
        'mlm_loss': total_mlm_loss / total_selected,
        'structure_loss': total_struct_loss / total_selected,
        'pair_loss': total_pair_loss / total_selected,
        'activation_rate': total_selected / total_samples if ast_selector else 1.0,
    }

    return avg_metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Returns:
        Dictionary of metrics
    """
    model.eval()

    total_loss = 0
    all_metrics = []
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            batch_size = batch['input_ids'].size(0)

            # Forward pass
            outputs = model(
                batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Compute loss
            loss_dict = loss_fn(outputs, batch)

            # Compute metrics
            metrics = compute_metrics(outputs, batch)

            # Accumulate
            total_loss += loss_dict['loss'].item() * batch_size
            all_metrics.append(metrics)
            total_samples += batch_size

    # Average metrics
    avg_metrics = {
        'loss': total_loss / total_samples,
    }

    # Average other metrics
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m.get(key, 0) for m in all_metrics) / len(all_metrics)

    return avg_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: dict,
    save_path: str,
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Pretrain Genesis RNA model")
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--data_path', type=str, help='Path to training data (optional)')
    parser.add_argument('--use_dummy_data', action='store_true', help='Use dummy data for testing')
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--use_ast', action='store_true', default=True, help='Enable AST')
    parser.add_argument('--ast_target_activation', type=float, default=0.4)

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        model_config = GenesisRNAConfig(**config_dict.get('model', {}))
        train_config = TrainingConfig(**config_dict.get('training', {}))
    else:
        # Create default configs
        if args.model_size == 'small':
            from .config import GenesisRNAConfigSmall
            model_config = GenesisRNAConfigSmall()
        elif args.model_size == 'large':
            from .config import GenesisRNAConfigLarge
            model_config = GenesisRNAConfigLarge()
        else:
            model_config = GenesisRNAConfig()

        model_config.vocab_size = len(SPECIAL_TOKENS) + len(NUC_VOCAB)

        train_config = TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            output_dir=str(output_dir),
            use_ast=args.use_ast,
            ast_target_activation=args.ast_target_activation,
        )

    # Save config
    config_save_path = output_dir / 'config.json'
    with open(config_save_path, 'w') as f:
        json.dump({
            'model': model_config.to_dict(),
            'training': train_config.__dict__,
        }, f, indent=2)

    # Create tokenizer
    tokenizer = RNATokenizer()

    # Load or create dataset
    if args.use_dummy_data or not args.data_path:
        print("Creating dummy dataset for testing...")
        train_samples = create_dummy_dataset(
            num_samples=1000,
            min_len=50,
            max_len=200,
            with_structure=True,
        )
        val_samples = create_dummy_dataset(
            num_samples=100,
            min_len=50,
            max_len=200,
            with_structure=True,
        )
    else:
        # TODO: Load real data
        raise NotImplementedError("Real data loading not yet implemented")

    # Create datasets
    train_dataset = RNAPretrainDataset(
        train_samples,
        tokenizer,
        max_len=model_config.max_len,
        mlm_probability=train_config.mlm_probability,
    )

    val_dataset = RNAPretrainDataset(
        val_samples,
        tokenizer,
        max_len=model_config.max_len,
        mlm_probability=train_config.mlm_probability,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_pretrain_batch,
        num_workers=train_config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=collate_pretrain_batch,
        num_workers=train_config.num_workers,
    )

    # Setup training
    model, optimizer, loss_fn, ast_selector, scaler, device = setup_training(
        model_config,
        train_config,
    )

    # Create learning rate scheduler
    num_training_steps = len(train_loader) * train_config.num_epochs
    scheduler = get_lr_scheduler(
        optimizer,
        num_training_steps,
        train_config.warmup_steps,
    )

    # Training loop
    print(f"\nStarting training for {train_config.num_epochs} epochs...")
    print(f"Total training steps: {num_training_steps}")

    best_val_loss = float('inf')

    for epoch in range(train_config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            ast_selector,
            scheduler,
            scaler,
            device,
            epoch,
            train_config,
        )

        print(f"\nEpoch {epoch} - Train metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch} - Val metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model,
                optimizer,
                epoch,
                epoch * len(train_loader),
                {'model': model_config.to_dict(), 'training': train_config.__dict__},
                output_dir / 'best_model.pt',
            )

        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                epoch * len(train_loader),
                {'model': model_config.to_dict(), 'training': train_config.__dict__},
                output_dir / f'checkpoint_epoch_{epoch}.pt',
            )

        # Print AST stats
        if ast_selector:
            ast_stats = ast_selector.get_stats()
            print("\nAST Statistics:")
            for key, value in ast_stats.items():
                print(f"  {key}: {value}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
