import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from HelperFunctions import plot_samples_sanity_test, plot_loss_over_epochs, evaluate_model, prepare_batch
from Loss import masked_mse_loss
from RadioMapSeerData import RadioMapSeerDataset
from TransUNet import TransUNet
from UNet import UNet
from CNNMoE import CNNMoE

# Config - Single model training
model = CNNMoE() 
model_name = "CNNMoE"  # For saving files
num_epochs = 15
num_obs = 1000  # Fixed number of observations

if __name__ == "__main__":
    # Check CUDA and fix device leakage
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "\n")
    
    # Force CUDA to use the correct device and optimize
    if torch.cuda.is_available():
        torch.cuda.set_device(device)  # Fix GPU 0 leakage
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Speed optimization
        
        # ENABLE TENSOR CORES FOR SPEED (1.3x speedup)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear any existing tensors on wrong device
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        # Optional: detect reference cycles causing GPU memory leaks
        try:
            from torch.utils.viz._cycles import warn_tensor_cycles
            warn_tensor_cycles()
            print("✅ Reference cycle detection enabled")
        except:
            pass  # Not available in all PyTorch versions

    # Dataset filepaths
    gain_filepath = "../../lab1/RadioMapSeer/gain/DPM"
    transmitter_filepath = "../../lab1/RadioMapSeer/png/antennas"
    buildings_filepath = "../../lab1/RadioMapSeer/png/buildings_complete"

    # Create model 
    model = model.to(device)
    print(f"Training {model_name} with {num_obs} observations for {num_epochs} epochs\n")

    # Create dataset
    dataset = RadioMapSeerDataset(gain_filepath, transmitter_filepath, buildings_filepath, obs=num_obs)

    # Train/Test Split
    train_ratio = 0.8
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    num_val = num_samples - num_train

    train_set, val_set = random_split(dataset, [num_train, num_val])

    # DataLoader - ALIGNED WITH train.py
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    print("Data Successfully Loaded\n")

    # Sanity Check: Dataset
    plot_samples_sanity_test(dataset)
    print("Dataset sanity check plot saved to images/random_sample_sanity_test.png\n")

    # Training Setup - ALIGNED WITH train.py
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)  # Same as train.py
    scaler = GradScaler("cuda")
    aux_loss_weight = 0.001 
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
                preds, aux_loss = model(x)
                main_loss = masked_mse_loss(preds, y, bld_mask)
                loss = main_loss + aux_loss_weight * aux_loss

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
                preds, aux_loss = model(x)
                main_loss = masked_mse_loss(preds, y, bld_mask)
                val_loss += main_loss.item() * x.size(0)

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
    print("Training completed successfully!") 