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
model = UNet() 
model_name = "UNet"  # For saving files
num_epochs = 40
num_obs = 1000  # Fixed number of observations

if __name__ == "__main__":
    # Check CUDA
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "\n")
    torch.backends.cudnn.benchmark = True

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

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    print("Data Successfully Loaded\n")

    # Sanity Check: Dataset
    plot_samples_sanity_test(dataset)
    print("Dataset sanity check plot saved to images/random_sample_sanity_test.png\n")

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

    print("Skipping full evaluation to avoid memory issues")
    print("Training completed successfully!") 

    # 500 1000 1500 2000 5000
    # 


    # 256 x 256 x 3 

    # --- > 16 x 16 x 3

    # gate (vector)

    # CNN Expert 1 x 1 x channel

    # Attention Layer
    # 768
    # Results with obs = [5, 25, 100, 250, 1000, 2500, 10000]
    #[0.0006425116299525349, 0.0008907116661924396, 0.0007259226216470953, 0.0006376319066584827, 0.00037294928135267973, 0.0003068667825319062, 0.0002632871113533882]
    #[0.0006645574604264711, 0.0008313714056780639, 0.0005999746389146336, 0.0005206517521796825, 0.00034185701824416447, 0.00024641768635847155, 0.00023710251061415856]