import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from model import resnet_forward_features


class DummyModel:
    def __init__(self) -> None:
        pass

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


def train_step(
    model,
    imgs,
    labels,
    optimizer,
    criterion,
    masks=None,
    device="cpu",
    metrics=None,
    return_features=False,
    gr=False,
):
    if isinstance(model, DummyModel):
        return

    optimizer.zero_grad()

    if gr:
        loss, loss_ce, out = criterion(imgs.to(device), labels.type(torch.LongTensor).to(device), masks.to(device), model)
    elif return_features:
        out = resnet_forward_features(model, imgs.to(device))
        loss, loss_ce = criterion(
            out, labels.type(torch.LongTensor).to(device), masks.to(device)
        )
        out = out[-1]
    else:
        out = model(imgs.to(device))
        loss = criterion(out, labels.type(torch.LongTensor).to(device))

    loss.backward()

    optimizer.step()

    loss_features = 0
    loss_gr = 0
    if gr:
        loss_total = loss.cpu().detach().item()
        loss_ce = loss_ce.cpu().detach().item()
        loss_gr = loss_total - loss_ce
    elif return_features:
        loss_total = loss.cpu().detach().item()
        loss_ce = loss_ce.cpu().detach().item()
        loss_features = loss_total - loss_ce
    else:
        loss_ce = loss.cpu().detach().item()
        loss_total = loss_ce

    if metrics is not None:
        with torch.no_grad():
            score, indices = torch.max(out.cpu().detach(), 1)
            metrics["loss_total"].append(loss_total)
            metrics["loss_ce"].append(loss_ce)
            metrics["loss_gr"].append(loss_gr)
            metrics["loss_features"].append(loss_features)
            metrics["preds"].append(indices.numpy())
            metrics["scores"].append(score.numpy())
            metrics["labels"].append(labels.numpy())


def eval_step(
    model,
    imgs,
    labels,
    masks,
    criterion,
    device="cpu",
    metrics=None,
    return_features=False,
    gr=False,
):
    if metrics is None or isinstance(model, DummyModel):
        return

    slc, score, indices, out = get_saliency(model, imgs, device=device)
    obj_score = get_obj_score(slc, masks)

    loss_features = 0
    if gr:
        loss, loss_ce, _ = criterion(imgs.to(device), labels.type(torch.LongTensor).to(device), masks.to(device), model)
        loss_total = loss.cpu().detach().item()
        loss_ce = loss_ce.cpu().detach().item()
        loss_gr = loss_total - loss_ce
    else:
        with torch.no_grad():
            
            if not return_features:
                loss_ce = criterion(out, labels.type(torch.LongTensor))
                loss_ce = loss_ce.cpu().detach().item()
                loss_total = loss_ce
            else:
                out = resnet_forward_features(model, imgs.to(device))
                loss_total, loss_ce = criterion(
                    out, labels.type(torch.LongTensor).to(device), masks.to(device)
                )
                loss_total = loss_total.cpu().detach().item()
                loss_ce = loss_ce.cpu().detach().item()
                loss_features = loss_total - loss_ce

    metrics["loss_total"].append(loss_total)
    metrics["loss_ce"].append(loss_ce)
    metrics["loss_gr"].append(loss_gr)
    metrics["loss_features"].append(loss_features)
    metrics["preds"].append(indices.detach().numpy())
    metrics["scores"].append(score.detach().numpy())
    metrics["labels"].append(labels.numpy())
    metrics["obj_scores"].append(obj_score)


def get_obj_score(slc, masks):
    mask = masks[0, 0].numpy().astype(bool)

    N = mask.shape[0] * mask.shape[1]
    N_obj = mask.sum()
    N_bg = N - N_obj

    slc_abs = np.abs(slc)

    obj_slc_score = slc_abs[mask].sum() / (N_obj + 1e-7)
    obj_score = obj_slc_score / (obj_slc_score + slc_abs[~mask].sum() / (N_bg + 1e-7) + 1e-7)
    return obj_score


def get_saliency(model, imgs, device="cpu"):
    imgs = Variable(imgs.data, requires_grad=True)

    # Get predictions (forward pass)
    out = model(imgs.to(device))
    score, indices = torch.max(out, 1)

    # Backward pass to get gradients of score of the predicted class wrt. the input image
    score.backward()

    slc = imgs.grad[0].cpu()

    # normalize to [-1..1]
    _, slc_indices = torch.max(torch.abs(slc), dim=0)

    slc_new = torch.zeros(slc_indices.size())

    for channel in range(slc.size()[0]):
        slc_new += slc[channel] * (slc_indices == channel)

    slc_new /= slc_new.abs().max()
    slc = slc_new.numpy()

    return slc, score.detach().cpu(), indices.detach().cpu(), out.detach().cpu()


def get_performance(metrics: dict[list]) -> dict[float]:
    performance = {}
    for metric in metrics:
        if "loss" in metric:
            performance[f"mean_{metric}"] = np.mean(metrics[metric]).item()
    performance["accuracy"] = np.mean(
        np.concatenate(metrics["preds"]) == np.concatenate(metrics["labels"])
    ).item()
    if "obj_scores" in metrics:
        performance["mean_obj_score"] = np.mean(metrics["obj_scores"]).item()
    return performance


def show_imarray(imarray, ax=None, **kwargs):
    if isinstance(imarray, torch.Tensor):
        imarray = imarray.numpy()
    imarray = np.squeeze(imarray)
    if imarray.ndim > 2:
        imarray = np.moveaxis(imarray, 0, -1)
    if ax is None:
        plt.imshow(imarray, **kwargs)
    else:
        ax.imshow(imarray, **kwargs)
