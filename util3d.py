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


def show_volume(volume, label=None, size=1, **kwargs):
    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()
    volume = np.squeeze(volume)
    fig, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(size * 6, size * 3), tight_layout=True
    )
    if label is not None:
        fig.suptitle(list(name_legend.values())[label].replace("_", " "))
    ax0.imshow(volume.max(0), **kwargs)
    ax1.imshow(volume.max(1), **kwargs)
    ax2.imshow(volume.max(2), **kwargs)
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
        obj_slc_score + slc_abs[~mask].sum() / N_bg
    )
    return obj_score
