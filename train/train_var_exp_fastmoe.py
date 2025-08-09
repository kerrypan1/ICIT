import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import pandas as pd
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import os
import gc
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from HelperFunctions import plot_samples_sanity_test, plot_loss_over_epochs, evaluate_model, prepare_batch
from Loss import masked_mse_loss
from RadioMapSeerData import RadioMapSeerDataset
from FastMoE import CNNMoE as FastMoECNNMoE

# Config
model_classes = [FastMoECNNMoE]
model_names = ["FastMoE"]  # For saving files
num_experts = [2, 4, 8, 16]  # Test different expert counts
num_epochs = 2  # Reduced for faster training
num_obs = 1000
cuda = 0
data_sampling_stride = 40


if __name__ == "__main__":
    # Fix CUDA multiprocessing issues
    mp.set_start_method('spawn', force=True)
    print("Set multiprocessing start method to 'spawn'")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{cuda}" if cuda_available else "cpu")
    print("Using device:", device, "\n")
    

    torch.cuda.set_device(cuda)

    with torch.cuda.device(cuda):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    

    gc.collect()
    torch.cuda.empty_cache()
    
    print("GPU optimizations enabled\n")
    
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=1,
        rank=0
    )

    # Print initial GPU memory status
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    print(f"Initial memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    print()

    # Dataset filepaths
    gain_filepath = "../../lab1/RadioMapSeer/gain/DPM"
    transmitter_filepath = "../../lab1/RadioMapSeer/png/antennas"
    buildings_filepath = "../../lab1/RadioMapSeer/png/buildings_complete"

    for model_class, model_name in zip(model_classes, model_names):
        train_cumul = []
        val_cumul = []
        time_cumul = []
        
        for num_expert in num_experts:
            model_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"Starting training for {model_name} with {num_obs} observations and {num_expert} experts")
            print(f"CUDA memory before: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved")
            print(f"{'='*60}")
            
            with torch.cuda.device(cuda):
                k = min(2, num_expert)
                model = model_class(
                    in_channels=3,
                    patch_size=16,
                    emb_dim=256,
                    num_experts=num_expert,
                    k=k,
                    out_channels=1,
                    world_size=1  # Single GPU
                ).to(device)
                
                print("Model ready for training")
            
            full_ds = RadioMapSeerDataset(gain_filepath, transmitter_filepath, buildings_filepath, obs=num_obs)
            keep_idx = list(range(0, len(full_ds), data_sampling_stride))
            dataset = Subset(full_ds, keep_idx)

            # Train/Test Split
            train_ratio = 0.8
            num_samples = len(dataset)
            num_train = int(train_ratio * num_samples)
            num_val = num_samples - num_train

            train_set, val_set = random_split(dataset, [num_train, num_val])

            # Data Loading
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True, 
                                    num_workers=4, pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_set, batch_size=64, shuffle=False, 
                                  num_workers=2, pin_memory=True, persistent_workers=True)
            eval_loader = DataLoader(dataset, batch_size=32, shuffle=False, 
                                   num_workers=2, pin_memory=True, persistent_workers=True)
            print("Data Successfully Loaded\n")

            # Sanity Check: Dataset
            plot_samples_sanity_test(dataset)
            print("Dataset sanity check plot saved to images/random_sample_sanity_test.png\n")

            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8, weight_decay=1e-2)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            scaler = GradScaler("cuda")
            aux_loss_weight = 0.001
            train_losses, val_losses = [], []
    
            # Training Loop
            print("Training Loop Initiated\n")
            for epoch in range(1, num_epochs + 1):
                model.train()
                train_loss = 0.0

                for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, disable=True):
                    x, y, bld_mask = prepare_batch(batch, device)
                    
                    if torch.rand(1) < 0.5:
                        x = torch.flip(x, dims=[3])
                        y = torch.flip(y, dims=[3])
                        bld_mask = torch.flip(bld_mask, dims=[3])

                    optimizer.zero_grad(set_to_none=True)

                    with autocast("cuda"):
                        preds, aux_loss = model(x)
                        main_loss = masked_mse_loss(preds, y, bld_mask)
                        loss = main_loss + aux_loss_weight * aux_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item() * x.size(0)

                train_mse = train_loss / len(train_loader.dataset)
                train_losses.append(train_mse)
                print(f"[Epoch {epoch}] Train MSE: {train_mse:.6f}")

                # Validation
                model.eval()
                val_loss = 0.0

                with torch.no_grad(), autocast("cuda"):
                    for batch in val_loader:
                        x, y, bld_mask = prepare_batch(batch, device)
                        preds, aux_loss = model(x)
                        main_loss = masked_mse_loss(preds, y, bld_mask)
                        val_loss += main_loss.item() * x.size(0)

                val_mse = val_loss / len(val_loader.dataset)
                val_losses.append(val_mse)
                print(f"[Epoch {epoch}] Val MSE:   {val_mse:.6f}")
                print(f"[Epoch {epoch}] Aux Loss: {aux_loss.item():.6f}\n")
                
                scheduler.step()

            # Plot Loss Over Epochs
            filepath = f"images/{model_name}_loss_{num_epochs}_epochs_{num_obs}_observations_num_experts_{num_expert}.png"
            plot_loss_over_epochs(train_losses, val_losses, filepath)
            print(f"Loss over epochs plot saved to {filepath}\n")

            print("Evaluating Model\n")
            filepath = f"images/{model_name}_sample_predictions_{num_epochs}_epochs_{num_obs}_observations_num_experts_{num_expert}.png"
            evaluate_model(model, eval_loader, device, filepath)
            print(f"Predictions saved to {filepath}\n")

            train_cumul.append(min(train_losses))
            val_cumul.append(min(val_losses))
            
            time_elapsed = time.time() - model_start_time
            time_cumul.append(time_elapsed)
            
            df = pd.DataFrame({
                'num_experts': num_experts[:len(train_cumul)],
                'train': train_cumul,
                'val': val_cumul,
                'time_elapsed_seconds': time_cumul
            })
            df.to_csv(f"results/{model_name}_{num_epochs}_epochs_results_variable_experts.csv", index=False)
            print(f"Results saved to results/{model_name}_{num_epochs}_epochs_results_variable_experts.csv (time elapsed: {time_elapsed:.2f}s)")

            # Memory cleanup
            del model, optimizer, scaler
            del train_loader, val_loader, eval_loader
            del dataset, train_set, val_set
            del train_losses, val_losses
            
            plt.close('all')
            tqdm._instances.clear()
            gc.collect()
            
            if torch.cuda.is_available():
                with torch.cuda.device(cuda):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            print(f"Memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved")
    dist.destroy_process_group()
    print(f"Completed training for {model_name}")