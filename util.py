import numpy as np
import torch


def train_step(model, imgs, labels, optimizer, criterion, device="cpu", metrics=None):
    optimizer.zero_grad()
    out = model(imgs.to(device))

    loss = criterion(out, labels.type(torch.LongTensor).to(device))
    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if metrics is not None:
        with torch.no_grad():
            score, indices = torch.max(out.cpu(), 1)
            metrics["loss"].append(loss.cpu().detach().item())
            metrics["preds"].append(indices.detach().numpy())
            metrics["scores"].append(score.detach().numpy())
            metrics["labels"].append(labels.numpy())


def eval_step(model, imgs, labels, masks, criterion, device="cpu", metrics=None):
    if metrics is None:
        return

    slc, score, indices, out = get_saliency(model, imgs, device=device)
    obj_score = obj_score(slc, masks)
    with torch.no_grad():
        loss = criterion(out, labels.type(torch.LongTensor))

    metrics["loss"].append(loss.cpu().detach().item())
    metrics["preds"].append(indices.detach().numpy())
    metrics["scores"].append(score.detach().numpy())
    metrics["labels"].append(labels.numpy())
    metrics["obj_scores"].append(obj_score)


def obj_score(slc, masks):
    mask = masks[0, 0].numpy().astype(bool)

    area = mask.shape[0] * mask.shape[1]
    obj_area = mask.sum()

    obj_frac = area / obj_area
    bg_frac = area / (area - obj_area)

    obj_slc_score = np.abs(slc)[mask].sum()
    obj_score = (obj_slc_score * obj_frac) / (
        obj_slc_score * obj_frac + (np.abs(slc).sum() - obj_slc_score) * bg_frac
    )
    return obj_score


def get_saliency(model, imgs, device="cpu"):
    # Calculate gradient of higest score w.r.t. input
    imgs.requires_grad = True

    # Get predictions (forward pass)
    out = model(imgs.to(device))
    score, indices = torch.max(out, 1)

    # Backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()

    # get max along channel axis
    slc = imgs.grad[0].cpu()

    # normalize to [-1..1]
    _, indices = torch.max(torch.abs(slc), dim=0)

    slc_new = torch.zeros(indices.size())

    for channel in range(slc.size()[0]):
        slc_new += slc[channel] * (indices == channel)

    slc_new /= slc_new.abs().max()
    slc = slc_new.numpy()

    return slc, score.cpu(), indices.cpu(), out.cpu()