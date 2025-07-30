import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import pandas as pd
import time

from HelperFunctions import plot_samples_sanity_test, plot_loss_over_epochs, evaluate_model, prepare_batch
from Loss import masked_mse_loss

from RadioMapSeerData import RadioMapSeerDataset

from TransUNet import TransUNet
from UNet import UNet
from CNNMoE import CNNMoE

# Config
model_classes = [TransUNet]
model_names = ["TransUNet"]  # For saving files
num_epochs = 40
num_obser = [2000, 5000]


if __name__ == "__main__":
    # Check CUDA
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "\n")
    torch.backends.cudnn.benchmark = True
    
    # Print initial GPU memory status
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        print(f"Initial memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
        print()

    # Dataset filepaths
    gain_filepath = "../../lab1/RadioMapSeer/gain/DPM"
    transmitter_filepath = "../../lab1/RadioMapSeer/png/antennas"
    buildings_filepath = "../../lab1/RadioMapSeer/png/buildings_complete"

    try:
        for model_class, model_name in zip(model_classes, model_names):
            train_cumul = []
            val_cumul = []
            time_cumul = []
            
            for num_obs in num_obser:
                model_start_time = time.time()
                print(f"\n{'='*60}")
                print(f"Starting training for {model_name} with {num_obs} observations")
                print(f"CUDA memory before: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved")
                print(f"{'='*60}")
                
                try:
                    model = model_class().to(device)
                    dataset = RadioMapSeerDataset(gain_filepath, transmitter_filepath, buildings_filepath, obs=num_obs)

                    # Train/Test Split
                    train_ratio = 0.8
                    num_samples = len(dataset)
                    num_train = int(train_ratio * num_samples)
                    num_val = num_samples - num_train

                    train_set, val_set = random_split(dataset, [num_train, num_val])

                    # DataLoader
                    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
                    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
                    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
                    print("Data Successfully Loaded\n")

                    # Sanity Check: Dataset
                    plot_samples_sanity_test(dataset)
                    print("Dataset sanity check plot saved to images/random_sample_sanity_test.png\n")

                    # Load Model
                    model = model.to(device)

                    # Training Setup
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)
                    scaler = GradScaler("cuda")
                    train_losses, val_losses = [], []

                    # Training Loop
                    print("Training Loop Initiated\n")
                    for epoch in range(1, num_epochs + 1):
                        model.train()
                        train_loss = 0.0

                        for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False):
                            x, y, bld_mask = prepare_batch(batch, device)

                            optimizer.zero_grad(set_to_none=True)

                            with autocast("cuda"):
                                preds = model(x)
                                loss = masked_mse_loss(preds, y, bld_mask)

                            scaler.scale(loss).backward()
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
                            for batch in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation", leave=False):
                                x, y, bld_mask = prepare_batch(batch, device)
                                preds = model(x)
                                val_loss += masked_mse_loss(preds, y, bld_mask).item() * x.size(0)

                        val_mse = val_loss / len(val_loader.dataset)
                        val_losses.append(val_mse)
                        print(f"[Epoch {epoch}] Val MSE:   {val_mse:.6f}\n")

                    # Plot Loss Over Epochs
                    filepath = f"images/{model_name}_loss_{num_epochs}_epochs_{num_obs}_observations.png"
                    plot_loss_over_epochs(train_losses, val_losses, filepath)
                    print(f"Loss over epochs plot saved to {filepath}\n")

                    print("Evaluating Model (sampling only)\n")
                    # Use a smaller subset for evaluation
                    small_eval_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
                    filepath = f"images/{model_name}_sample_predictions_{num_epochs}_epochs_{num_obs}_observations.png"
                    evaluate_model(model, small_eval_loader, device, filepath)
                    print(f"Predictions saved to {filepath}\n")

                    train_cumul.append(train_losses[-1])
                    val_cumul.append(val_losses[-1])
                    
                    # Calculate time elapsed for this observation count
                    time_elapsed = time.time() - model_start_time
                    time_cumul.append(time_elapsed)
                    
                    # Save results continuously after each observation count
                    df = pd.DataFrame({
                        'obs': num_obser[:len(train_cumul)],
                        'train': train_cumul,
                        'val': val_cumul,
                        'time_elapsed_seconds': time_cumul
                    })
                    df.to_csv(f"{model_name}_{num_epochs}_epochs_results.csv", index=False)
                    print(f"Results saved to {model_name}_{num_epochs}_epochs_results.csv (time elapsed: {time_elapsed:.2f}s)")

                    # Comprehensive memory cleanup

                    del model, optimizer, scaler
                    del train_loader, val_loader, eval_loader, small_eval_loader  # Add small_eval_loader
                    del dataset, train_set, val_set
                    del train_losses, val_losses
                    
                    # Clear matplotlib cache
                    import matplotlib.pyplot as plt
                    plt.close('all')
                    
                    # Force garbage collection multiple times
                    import gc
                    gc.collect()
                    gc.collect()  # Second pass
                    
                    # Clear CUDA cache thoroughly
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    
                    # Additional CUDA memory management
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        # Force CUDA to release reserved memory
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        # Reset memory stats to clear fragmentation
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()
                        # Force memory pool cleanup
                        torch.cuda.empty_cache()
                    
                    print(f"Memory after cleanup: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved")
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n❌ CUDA OOM ERROR for {model_name} with {num_obs} observations!")
                        print(f"Error: {e}")
                        print(f"Memory at failure: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
                        
                        # Emergency cleanup
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        # Skip this configuration and continue
                        train_cumul.append(float('nan'))
                        val_cumul.append(float('nan'))
                        continue
                    else:
                        raise e

            print(f"Completed training for {model_name}")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Training failed with exception: {e}")
        print(f"{'='*60}")
        
        # Emergency memory cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Print current memory status
        if torch.cuda.is_available():
            print(f"Memory at error: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
            print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved")
        
        import traceback
        traceback.print_exc()
        raise

#obs,train,val,time_elapsed_seconds
#500,0.00016721332129874877,0.0002095740408980063,3889.3341019153595
#1000,0.00026025832267415,0.000312600789834797,3894.6138134002686
#1500,0.00011935544497050815,0.000102405780081879,3919.5143263339996