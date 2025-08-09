import random
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn.functional as F
from Loss import masked_mse_loss

# Helper function for training and evaluation to load batch onto GPU
def prepare_batch(batch, device):
    transmitters = batch["transmitters"].to(device, non_blocking=True)
    gain_sparse  = batch["gain_sparse"].to(device, non_blocking=True)
    bld_mask     = batch["buildings"].to(device, non_blocking=True)
    gain_full    = batch["gain_full"].to(device, non_blocking=True)
    x = torch.cat([transmitters, gain_sparse, bld_mask], dim=1)
    return x, gain_full, bld_mask

# Helper function to plot a random sample from the dataset for sanity testing
def plot_samples_sanity_test(dataset):
    indices = random.sample(range(len(dataset)), 3)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
    fig.suptitle("Random Sample Sanity Test", fontsize=20)

    for row_idx, idx in enumerate(indices):
        sample = dataset[idx]
        bld = sample["buildings"][0].numpy()
        tx = sample["transmitters"][0].numpy()
        sparse = sample["gain_sparse"][0].numpy()
        full = sample["gain_full"][0].numpy()

        axes[row_idx][0].imshow(bld, cmap="gray")
        axes[row_idx][1].imshow(tx, cmap="hot")
        axes[row_idx][2].imshow(sparse, cmap="viridis")
        axes[row_idx][3].imshow(full, cmap="viridis")

        for col in range(4):
            axes[row_idx][col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig("images/random_sample_sanity_test.png")
    plt.close()

# Helper function to plot the loss over epochs
def plot_loss_over_epochs(train_losses, val_losses, filepath):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Valid')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# Helper function to generate sample predictions
def evaluate_model(model, eval_loader, device, filepath, num_to_show=3):
    model.eval()
    all_preds, all_y, all_b = [], [], []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            transmitters = batch["transmitters"].to(device, non_blocking=True)
            gain_sparse  = batch["gain_sparse"].to(device, non_blocking=True)
            bld_mask     = batch["buildings"].to(device, non_blocking=True)
            gain_full    = batch["gain_full"].to(device, non_blocking=True)

            inp = torch.cat([transmitters, gain_sparse, bld_mask], dim=1)
            model_output = model(inp)
            
            # Handle both old and new model output formats
            if isinstance(model_output, tuple):
                pb, _ = model_output  # Unpack (predictions, aux_loss)
            else:
                pb = model_output  # Old format

            if pb.shape[-2:] != gain_full.shape[-2:]:
                pb = F.interpolate(pb, size=gain_full.shape[-2:], mode='bilinear', align_corners=False)

            pb = pb * (bld_mask < 0.5).float()

            all_preds.append(pb.detach().cpu())
            all_y.append(gain_full.detach().cpu())
            all_b.append(bld_mask.detach().cpu())
            
            # Clear GPU tensors
            del transmitters, gain_sparse, bld_mask, gain_full, inp, pb

    preds = torch.cat(all_preds, dim=0)
    y_all = torch.cat(all_y,    dim=0)
    b_all = torch.cat(all_b,    dim=0)

    mse = masked_mse_loss(preds, y_all, b_all).item()
    print(f"Masked MSE: {mse:.6f}")

    valid_mask = (b_all < 0.5)
    gt_vals = (y_all * valid_mask).view(-1)
    pr_vals = (preds * valid_mask).view(-1)
    gt_vals = gt_vals[gt_vals > 0]
    pr_vals = pr_vals[pr_vals > 0]

    indices = random.sample(range(len(preds)), num_to_show)
    _, axes = plt.subplots(num_to_show, 3, figsize=(12, 4 * num_to_show))

    if num_to_show == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        gt   = y_all[idx, 0]
        pred = preds[idx, 0]
        mask = b_all[idx, 0]

        vmin = 0.0
        vmax = 1.0

        axes[row][0].imshow(gt, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row][0].set_title(f"GT Radiomap (idx={idx})")
        axes[row][0].axis('off')

        axes[row][1].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row][1].set_title("Predicted (masked)")
        axes[row][1].axis('off')

        axes[row][2].imshow(mask, cmap='gray', vmin=0.0, vmax=1.0)
        axes[row][2].set_title("Building Mask")
        axes[row][2].axis('off')

    plt.tight_layout()

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
