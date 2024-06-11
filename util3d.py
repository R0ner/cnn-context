import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

name_legend = {
    "ac": "brown_cricket",
    "bc": "black_cricket",
    "bf": "blow_fly",
    "bl": "buffalo_bettle_larva",
    "bp": "blow_fly_pupa",
    "cf": "curly-wing_fly",
    "gh": "grasshopper",
    "ma": "maggot",
    "ml": "mealworm",
    "pp": "green_bottle_fly_pupa",
    "sl": "soldier_fly_larva",
    "wo": "woodlice",
}


def show_volume(volume, label=None, size=1, fig_axs=None, title=None, **kwargs):
    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()
    volume = np.squeeze(volume)
    if fig_axs is None:
        fig, axs = plt.subplots(
            1, 3, figsize=(size * 3, size * 1), tight_layout=True
        )
    else:
        fig, axs = fig_axs
    if label is not None and title is None:
        title = list(name_legend.values())[label].replace("_", " ")
        fig.suptitle(title)
    if title is not None:
        fig.suptitle(title)
    for i, ax in enumerate(axs):
        ax.imshow(volume.max(i), **kwargs)
    if fig_axs is None:
        plt.show()


def get_saliency3d(model, volumes, device="cpu"):
    # Calculate gradient of higest score w.r.t. input
    volumes = Variable(volumes.data, requires_grad=True)

    # Get predictions (forward pass)
    out = model(volumes.to(device))
    score, indices = torch.max(out, 1)

    # Backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()

    slc = volumes.grad[0].cpu().numpy()[0]

    return slc, score.detach().cpu(), indices.detach().cpu(), out.detach().cpu()

def get_obj_score3d(slc, masks):
    mask = np.squeeze(masks.numpy()).astype(bool)

    N = mask.shape[0] * mask.shape[1] * mask.shape[2]
    N_obj = mask.sum()
    N_bg = N - N_obj
    
    slc_abs = np.abs(slc)

    obj_slc_score = slc_abs[mask].sum() / N_obj
    obj_score = obj_slc_score / (
        obj_slc_score + slc_abs[~mask].sum() / N_bg + 1e-7
    )

    return obj_score

def combine_scans(volumes, masks, targets, threshold=0.05):
    bs = volumes.size(0)
    while bs > 1:
        bs_h = bs // 2
        masks_u = masks[:bs_h] | masks[bs_h : 2 * bs_h]
        masks_i = masks[:bs_h] & masks[bs_h : 2 * bs_h]

        intrs = masks_i.view(bs_h, masks.size(1), -1).sum(-1)
        sizes = masks.view(masks.size(0), masks.size(1), -1).sum(-1)
        fracs = intrs / sizes[:bs_h], intrs / sizes[bs_h : 2 * bs_h]
        keep = (sum(fracs) < 2 * threshold).flatten()

        if keep.any():

            idx_smallest = (fracs[0] > fracs[1]).long().flatten() * bs_h + torch.arange(
                bs_h
            )
            volumes_i = (volumes[idx_smallest] * masks_i)[keep]
            volumes[:bs_h][keep] = (
                volumes[:bs_h][keep] + volumes[bs_h : 2 * bs_h][keep] - volumes_i
            )
            
            targets_i = (targets[idx_smallest] * masks_i)[keep]
            targets[:bs_h][keep] = (
                targets[:bs_h][keep] + targets[bs_h : 2 * bs_h][keep] - targets_i
            )
            masks[:bs_h][keep] = masks_u[keep]
        bs = bs_h
    return volumes, masks, targets