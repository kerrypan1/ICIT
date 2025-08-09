# MSE over non-building pixels
def masked_mse_loss(pred, target, building_mask, eps=1e-8):
    diff_sq = (pred - target).pow(2)
    non_build = (building_mask < 0.5).float() 

    masked_diff = diff_sq * non_build
    sums = masked_diff.sum(dim=[1, 2, 3]) 
    counts = non_build.sum(dim=[1, 2, 3]).clamp_min(eps) 

    return (sums / counts).mean()