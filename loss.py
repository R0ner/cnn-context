from typing import Any

import torch
from torch import nn
from torch.autograd import grad
from torch.nn.functional import interpolate


class SuperpixelWeights:
    def __init__(
        self,
        model_type: str,
        normalize: bool = True,
        binary: bool = False,
        binary_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model_type = model_type
        self.normalize = normalize
        self.binary = binary
        self.binary_threshold = binary_threshold
        self.device = device

        if self.model_type != "r18":
            raise NotImplementedError

        self.conv7x7 = nn.Conv2d(
            1, 1, kernel_size=7, stride=2, padding=3, bias=False, padding_mode="reflect"
        ).to(device)
        self.conv7x7.weight = self.conv7x7.weight.requires_grad_(False)
        self.conv7x7.weight *= 0
        self.conv7x7.weight += 1 / (7**2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3x3 = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="reflect"
        ).to(device)
        self.conv3x3_s2 = nn.Conv2d(
            1, 1, kernel_size=3, stride=2, padding=1, bias=False, padding_mode="reflect"
        ).to(device)

        for conv in (self.conv3x3, self.conv3x3_s2):
            conv.weight = conv.weight.requires_grad_(False)
            conv.weight *= 0
            conv.weight += 1 / (3**2)

        self.layer1 = 4 * [self.conv3x3]
        self.layer234 = [self.conv3x3_s2, self.conv3x3] + 2 * [self.conv3x3]

        self.layer1 = nn.Sequential(*self.layer1)
        self.layer234 = nn.Sequential(*self.layer234)

    @staticmethod
    def sp_normalize(sp_weights: list[torch.tensor]) -> list[torch.tensor]:
        sp_normalized = []
        for sp_w in sp_weights:
            sp_w_min = (
                sp_w.view(sp_w.size(0), -1)
                .min(1, keepdim=True)
                .values.view(-1, 1, 1, 1)
            )
            sp_w_max = (
                sp_w.view(sp_w.size(0), -1)
                .max(1, keepdim=True)
                .values.view(-1, 1, 1, 1)
            )
            sp_normalized.append((sp_w - sp_w_min) / (sp_w_max - sp_w_min + 1e-7))
        return sp_normalized

    def __call__(self, masks: torch.tensor) -> list[torch.tensor]:
        sp_weights = []
        x = self.conv7x7(masks.float())
        sp_weights.append(1 - x)

        x = self.maxpool(x)
        x = self.layer1(x)
        sp_weights.append(1 - x)
        x = self.layer234(x)
        sp_weights.append(1 - x)
        x = self.layer234(x)
        sp_weights.append(1 - x)
        x = self.layer234(x)
        sp_weights.append(1 - x)

        if self.normalize:
            sp_weights = self.sp_normalize(sp_weights)

        if self.binary:
            sp_weights = list(
                map(lambda sp_w: (sp_w > self.binary_threshold).float(), sp_weights)
            )

        return sp_weights


class SuperpixelWeightsExact:
    def __init__(
        self, model_type: str, anti_aliasing: bool = True, device: str = "cpu"
    ) -> None:
        self.model_type = model_type
        self.anti_aliasing = anti_aliasing
        self.device = device

        if self.model_type != "r18":
            raise NotImplementedError

    def __call__(self, masks: torch.tensor) -> list[torch.tensor]:
        h, w = masks.shape[-2:]
        sp_weights = []
        for _ in range(5):
            h, w = h // 2 + h % 2, w // 2 + w % 2
            sp_weights.append(
                (
                    ~(
                        interpolate(
                            masks.float(), size=(h, w), mode="bilinear", antialias=True
                        )
                        > 0.1
                    )
                ).float()
            )

        return sp_weights


class SuperpixelCriterion:
    def __init__(
        self,
        model_type: str,
        sp_loss_weight: float = 1,
        layer_weights: str = "constant",
        exact: bool = False,
        normalize: bool = True,
        binary: bool = False,
        binary_threshold: float = 0.5,
        mode: str = "l2",
        device: str = "cpu",
    ) -> None:
        self.model_type = model_type
        self.sp_loss_weight = sp_loss_weight
        self.layer_weights = layer_weights.lower()
        self.exact = exact
        self.normalize = normalize
        self.binary = binary
        self.binary_threshold = binary_threshold
        self.mode = mode
        self.device = device

        self.modes = ("l1", "l2", "elastic")
        self.layer_weight_schemes = ("constant", "geometric", "last")

        assert (
            self.layer_weights in self.layer_weight_schemes
        ), f"'layer_weights' must be one of {self.layer_weight_schemes}"
        assert self.mode in self.modes, f"'mode' must be one of {self.modes}"

        if self.exact:
            self.get_sp_weights = SuperpixelWeightsExact(
                self.model_type, anti_aliasing=True, device=self.device
            )
        else:
            self.get_sp_weights = SuperpixelWeights(
                self.model_type,
                normalize=self.normalize,
                binary=self.binary,
                binary_threshold=self.binary_threshold,
                device=self.device,
            )
        self.ce_criterion = nn.CrossEntropyLoss()  # Cross entropy

        if self.layer_weights == "constant":
            self.get_layer_weight = lambda i: 1
        elif self.layer_weights == "geometric":
            self.get_layer_weight = lambda i: 2 ** (-i)
        elif self.layer_weights == "last":
            self.get_layer_weight = lambda i: i == 0

        if self.mode == "l1":
            # Could use 'identity' instead in case of ReLU.
            self.sp_loss_func = torch.abs
        elif self.mode == "l2":
            self.sp_loss_func = torch.square
        elif self.mode == "elastic":
            self.sp_loss_func = lambda x: (torch.square(x) + torch.abs(x)) / 2

    def __call__(
        self, outs: list[torch.tensor], targets: torch.tensor, masks: torch.tensor
    ) -> torch.tensor:

        # Get feature weights for each layer according to "mode"
        sp_weights = self.get_sp_weights(masks)
        n_layers = len(sp_weights)

        layer_weight_sum = 0
        loss = 0
        for i, (sp, sp_w) in enumerate(zip(outs[:-1], sp_weights)):
            layer_weight = self.get_layer_weight(n_layers - i - 1)
            layer_weight_sum += layer_weight

            # g(l) * sum(w * V(sp)) / (sum(w) * (no. channels))
            # where w is the feature weights, sp are the features, V(x) is the error function, and g(l) is the layer weight.
            # We multiply by the number of channels (sp.size(1)) in the denominator as w is broadcast for multiplication with V(sp).
            # The loss is calculated for each item in the batch and finally the mean is taken over the batch.
            loss += (
                layer_weight
                * (
                    (sp_w * self.sp_loss_func(sp)).view(sp.size(0), -1).sum(1)
                    / (sp_w.view(sp.size(0), -1).sum(1) * sp.size(1) + 1e-7)
                ).mean()
            )

        # Cross entropy loss.
        loss_ce = self.ce_criterion(outs[-1], targets)

        # L_total = L_ce + lambda * L_feature
        # where L_ce is the cross entropy loss, lambda is the feature loss weight, and L_feature is the feature loss.
        loss = loss_ce + loss / layer_weight_sum * self.sp_loss_weight

        return loss, loss_ce


class grCriterion(nn.Module):
    def __init__(
        self, weight=1.0, mode: str = "l2"
    ) -> None:
        super().__init__()
        self.gr_loss_weight = weight
        self.mode = mode

        self.ce_criterion = nn.CrossEntropyLoss()  # Cross entropy

        if self.mode == "l1":
            self.lf = torch.abs
        elif self.mode == "l2":
            self.lf = torch.square
        elif self.mode == "elastic":
            self.lf = lambda x: (torch.square(x) + torch.abs(x)) / 2
    
    def forward(self, x, label, mask, model):
        bs = x.size(0)
        
        x.requires_grad = True
        
        out = model(x)

        # loss_ce = self.ce_criterion(out, label)
        
        # # Compute input gradient
        # saliency = torch.autograd.grad(outputs=loss_ce, inputs=x, create_graph=True)[0]
        
        # x.requires_grad = False

        logits, _ = out.max(1)

        # Compute input gradient
        saliency = grad(outputs=logits, inputs=x, grad_outputs=torch.ones_like(logits), create_graph=True)[0]
        
        x.requires_grad = False
        
        loss_ce = self.ce_criterion(out, label)

        mask_b = ~mask

        # Compute gradient penalty
        loss = (self.lf(mask_b * saliency).view(bs, -1).sum(1) / (mask_b.view(bs, -1).sum(1) * x.size(1) + 1e-7)).mean()
    
        loss = loss_ce + self.gr_loss_weight * loss
        
        return loss, loss_ce, out