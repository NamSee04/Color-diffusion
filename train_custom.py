import argparse
import numpy as np
import pytorch_lightning as pl
from dataset import make_custom_dataloaders
from model import ColorDiffusion
from utils import get_device, load_default_configs
from denoising import Unet, Encoder
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, TQDMProgressBar
import os
import time
from tqdm.auto import tqdm

class CustomProgressCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        self.current_epoch = trainer.current_epoch
        self.total_epochs = trainer.max_epochs
        print(f"\nEpoch {self.current_epoch + 1}/{self.total_epochs}")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Calculate progress
        progress = (batch_idx + 1) / len(trainer.train_dataloader)
        elapsed_time = time.time() - self.epoch_start_time
        estimated_total = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total - elapsed_time
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Print progress
        print(f'\rProgress: [{bar}] {progress:.1%} | '
              f'Time: {elapsed_time:.0f}s | '
              f'ETA: {remaining_time:.0f}s | '
              f'Loss: {trainer.callback_metrics.get("train_loss", 0):.4f}', 
              end='')
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        print(f"\nEpoch {self.current_epoch + 1} completed in {epoch_time:.2f} seconds")
        print(f"Average time per batch: {epoch_time/len(trainer.train_dataloader):.2f} seconds")

def train_custom_model(L_arrays, ab_arrays, output_dir="./checkpoints", 
                      resume_ckpt=None, num_workers=2, log=False):
    """
    Train the color diffusion model on custom L and ab arrays
    
    Args:
        L_arrays: List of numpy arrays of shape (224, 224)
        ab_arrays: List of numpy arrays of shape (224, 224, 2)
        output_dir: Directory to save checkpoints
        resume_ckpt: Path to checkpoint to resume training from
        num_workers: Number of workers for dataloader
        log: Whether to enable logging
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configurations
    enc_config, unet_config, colordiff_config = load_default_configs()
    
    # Create dataloaders
    train_dl, val_dl = make_custom_dataloaders(
        L_arrays, 
        ab_arrays, 
        colordiff_config,
        num_workers=num_workers
    )
    
    # Update config
    colordiff_config["sample"] = False
    colordiff_config["should_log"] = log
    
    # Initialize model components
    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config)
    
    # Load or create model
    if resume_ckpt is not None:
        print(f"Resuming training from checkpoint: {resume_ckpt}")
        model = ColorDiffusion.load_from_checkpoint(
            resume_ckpt,
            strict=True,
            unet=unet,
            encoder=encoder,
            train_dl=train_dl,
            val_dl=val_dl,
            **colordiff_config
        )
    else:
        model = ColorDiffusion(
            unet=unet,
            encoder=encoder,
            train_dl=train_dl,
            val_dl=val_dl,
            **colordiff_config
        )
    
    # Setup checkpointing
    ckpt_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='color_diffusion-{epoch:02d}-{val_loss:.2f}',
        every_n_train_steps=300,
        save_top_k=2,
        save_last=True,
        monitor="val_loss"
    )
    
    # Setup trainer with custom progress callback
    trainer = pl.Trainer(
        max_epochs=colordiff_config["epochs"],
        accelerator=colordiff_config["device"],
        num_sanity_val_steps=1,
        devices="auto",
        log_every_n_steps=1,
        callbacks=[
            CustomProgressCallback(),
            ckpt_callback
        ],
        profiler="simple" if log else None,
        accumulate_grad_batches=colordiff_config["accumulate_grad_batches"],
        enable_progress_bar=False  # Disable default progress bar
    )
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Max epochs: {colordiff_config['epochs']}")
    print(f"Batch size: {colordiff_config['batch_size']}")
    print(f"Device: {colordiff_config['device']}")
    print(f"Total batches per epoch: {len(train_dl)}")
    print("\nStarting training...")
    
    # Train model
    trainer.fit(model, train_dl, val_dl)
    
    return model, trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L_path", required=True, help="Path to .npy file containing L channel arrays")
    parser.add_argument("--ab_path", required=True, help="Path to .npy file containing ab channel arrays")
    parser.add_argument("--output_dir", default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_ckpt", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--log", action="store_true", help="Enable logging")
    args = parser.parse_args()
    
    # Load data
    L_arrays = np.load(args.L_path)
    ab_arrays = np.load(args.ab_path)
    
    # Verify shapes
    assert L_arrays.shape[1:] == (224, 224), f"L arrays should have shape (N, 224, 224), got {L_arrays.shape}"
    assert ab_arrays.shape[1:] == (224, 224, 2), f"ab arrays should have shape (N, 224, 224, 2), got {ab_arrays.shape}"
    assert len(L_arrays) == len(ab_arrays), "Number of L and ab arrays must match"
    
    # Convert to list of arrays
    L_arrays = [L_arrays[i] for i in range(len(L_arrays))]
    ab_arrays = [ab_arrays[i] for i in range(len(ab_arrays))]
    
    # Train model
    model, trainer = train_custom_model(
        L_arrays,
        ab_arrays,
        output_dir=args.output_dir,
        resume_ckpt=args.resume_ckpt,
        num_workers=args.num_workers,
        log=args.log
    ) 