"""
Contains some detectors specifically designed for the GTSR dataset
"""
import torch
from torch import Tensor
from pytorch_ood.api import Detector
from torch import nn


class LogicOOD(Detector):
    def __init__(
        self,
        label_net: nn.Module,
        shape_net: nn.Module,
        color_net: nn.Module,
        class_to_shape: dict,
        class_to_color: dict,
        sign_net: nn.Module = None,
        rotation_net: nn.Module = None,
    ):
        self.label_net = label_net
        self.shape_net = shape_net
        self.color_net = color_net
        self.sign_net = sign_net
        self.rotation_net = rotation_net

        self.class_to_shape = class_to_shape
        self.class_to_color = class_to_color

    def fit_features(self, x):
        pass

    def predict_features(self, x: Tensor) -> Tensor:
        raise ValueError

    @torch.no_grad()
    def get_predictions(self, x):

        results = {
            "label": self.label_net(x).cpu(),
            "shape": self.shape_net(x).cpu(),
            "color": self.color_net(x).cpu(),
        }

        if self.sign_net:
            results["sign"] = self.sign_net(x).cpu()

        if self.rotation_net:
            results["rotation"] = self.rotation_net(x).cpu()

        return results

    @torch.no_grad()
    def consistent(self, x, return_predictions=False):
        """
        Determines of the predictions are consistent with the domain knowledge
        """
        p = self.get_predictions(x)

        labels = p["label"].max(dim=1).indices
        shape = torch.tensor([self.class_to_shape[c.item()] for c in labels])
        color = torch.tensor([self.class_to_color[c.item()] for c in labels])

        shape_hat = p["shape"].max(dim=1).indices.cpu()
        color_hat = p["color"].max(dim=1).indices.cpu()

        consistent = (shape_hat == shape) & (color == color_hat)

        if return_predictions:
            return consistent, p

        return -consistent.float()

    @torch.no_grad()
    def predict(self, x):
        consistent, p = self.consistent(x, return_predictions=True)

        values = []

        for key, value in p.items():
            conf = value.softmax(dim=1).max(dim=1).values.cpu()
            # print(f"{conf.shape=}")
            values.append(conf)

        scores = torch.stack(values, dim=1).mean(dim=1)
        return -scores * consistent.float()

    def fit(self, *args, **kwargs):
        pass