import torch
from torch.utils.data import Dataset
import torchvision.io as io

class RadioMapSeerDataset(Dataset):
    def __init__(self, gain_dir, tx_dir, bld_dir, obs):
        self.obs = obs
        self.gain_dir = gain_dir
        self.tx_dir   = tx_dir
        self.bld_dir  = bld_dir
        self.indices  = [(i, j) for i in range(701) for j in range(80)]
        self.samples = [
        (f"{self.bld_dir}/{i}.png",
        f"{self.tx_dir}/{i}_{j}.png",
        f"{self.gain_dir}/{i}_{j}.png")
        for i in range(701) for j in range(80)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        bld_path, tx_path, gain_path = self.samples[idx]

        bld = io.read_image(bld_path, mode=io.ImageReadMode.GRAY).float().div_(255)
        tx  = io.read_image(tx_path, mode=io.ImageReadMode.GRAY).float().div_(255)
        gain= io.read_image(gain_path, mode=io.ImageReadMode.GRAY).float().div_(255)

        _, H, W = gain.shape
        N = H * W

        gen = torch.Generator().manual_seed(idx)
        weights = torch.ones(N, device=gain.device)
        sel = torch.multinomial(weights, self.obs, replacement=False, generator=gen)

        gain_flat   = gain.view(-1)
        sparse_flat = torch.zeros(N, dtype=gain_flat.dtype, device=gain_flat.device)
        sparse_flat.scatter_(0, sel, gain_flat[sel])
        gain_sparse = sparse_flat.view_as(gain)

        return {
            "buildings":    bld,
            "transmitters": tx,
            "gain_sparse":  gain_sparse,
            "gain_full":    gain
        }
