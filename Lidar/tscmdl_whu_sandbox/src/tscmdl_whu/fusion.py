"""Feature adapters and the paper-style TSCMDL fusion classifier."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class PointMLPFeatureEncoder(nn.Module):
    """Expose the 1024-D pooled feature before PointMLP's classifier."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        required = (
            "embedding",
            "local_grouper_list",
            "pre_blocks_list",
            "pos_blocks_list",
            "stages",
            "classifier",
        )
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise ValueError(f"PointMLP model is missing attributes: {missing}")
        first_classifier_layer = model.classifier[0]
        if not isinstance(first_classifier_layer, nn.Linear):
            raise ValueError("PointMLP classifier does not start with Linear")
        self.model = model
        self.output_dim = int(first_classifier_layer.in_features)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[1] != 3:
            raise ValueError(f"Expected PointMLP input [B,3,N], got {points.shape}")
        xyz = points.permute(0, 2, 1)
        features = self.model.embedding(points)
        for index in range(self.model.stages):
            xyz, features = self.model.local_grouper_list[index](
                xyz, features.permute(0, 2, 1)
            )
            features = self.model.pre_blocks_list[index](features)
            features = self.model.pos_blocks_list[index](features)
        return F.adaptive_max_pool1d(features, 1).squeeze(dim=-1)


class ResNet50FeatureEncoder(nn.Module):
    """Expose the 2048-D average-pooled feature before ResNet50's classifier."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
            raise ValueError("ResNet50 model does not expose a linear fc layer")
        self.output_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if features.ndim != 2:
            raise ValueError(f"Expected ResNet features [B,D], got {features.shape}")
        return features


class TSCMDLFusionModel(nn.Module):
    """Normalize 1024-D modal features, concatenate, and classify."""

    def __init__(
        self,
        point_encoder: nn.Module,
        image_encoder: nn.Module,
        num_classes: int,
        *,
        point_dim: int = 1024,
        image_dim: int = 2048,
        modal_dim: int = 1024,
        dropout: float = 0.5,
        freeze_backbones: bool = True,
        normalization: str = "batchnorm",
        classifier_hidden_dims: tuple[int, ...] = (512, 256),
        modality: str = "fusion",
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if normalization not in {"batchnorm", "l2"}:
            raise ValueError("normalization must be 'batchnorm' or 'l2'")
        if not classifier_hidden_dims or any(
            dimension <= 0 for dimension in classifier_hidden_dims
        ):
            raise ValueError("classifier_hidden_dims must contain positive values")
        if modality not in {"fusion", "point", "image"}:
            raise ValueError("modality must be 'fusion', 'point', or 'image'")
        self.point_encoder = point_encoder
        self.image_encoder = image_encoder
        self.point_dim = point_dim
        self.image_dim = image_dim
        self.modal_dim = modal_dim
        self.normalization = normalization
        self.classifier_hidden_dims = classifier_hidden_dims
        self.modality = modality
        self.image_projection = nn.Linear(image_dim, modal_dim)
        self.point_projection = (
            nn.Identity() if point_dim == modal_dim else nn.Linear(point_dim, modal_dim)
        )
        self.point_norm = (
            nn.BatchNorm1d(modal_dim)
            if normalization == "batchnorm"
            else nn.Identity()
        )
        self.image_norm = (
            nn.BatchNorm1d(modal_dim)
            if normalization == "batchnorm"
            else nn.Identity()
        )
        classifier_layers: list[nn.Module] = []
        input_dim = 2 * modal_dim
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.extend(
                (
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
            input_dim = hidden_dim
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        self.backbones_frozen = False
        self.set_backbones_trainable(not freeze_backbones)

    def set_backbones_trainable(self, trainable: bool) -> None:
        self.backbones_frozen = not trainable
        for encoder in (self.point_encoder, self.image_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(trainable)
        if self.backbones_frozen:
            self.point_encoder.eval()
            self.image_encoder.eval()

    def train(self, mode: bool = True) -> TSCMDLFusionModel:
        super().train(mode)
        if self.backbones_frozen:
            self.point_encoder.eval()
            self.image_encoder.eval()
        return self

    def encode_modalities(
        self, points: torch.Tensor, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.no_grad if self.backbones_frozen else nullcontext
        with context():
            point_features = self.point_encoder(points)
            image_features = self.image_encoder(images)
        if point_features.shape != (points.shape[0], self.point_dim):
            raise ValueError(
                f"Unexpected point feature shape: {point_features.shape}"
            )
        if image_features.shape != (images.shape[0], self.image_dim):
            raise ValueError(
                f"Unexpected image feature shape: {image_features.shape}"
            )
        return point_features, image_features

    def forward_features(
        self, points: torch.Tensor, images: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        point_raw, image_raw = self.encode_modalities(points, images)
        return self.fuse_features(point_raw, image_raw)

    def fuse_features(
        self,
        point_raw: torch.Tensor,
        image_raw: torch.Tensor,
        modality: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if point_raw.ndim != 2 or point_raw.shape[1] != self.point_dim:
            raise ValueError(f"Unexpected cached point feature shape: {point_raw.shape}")
        if image_raw.ndim != 2 or image_raw.shape[1] != self.image_dim:
            raise ValueError(f"Unexpected cached image feature shape: {image_raw.shape}")
        if point_raw.shape[0] != image_raw.shape[0]:
            raise ValueError("Point and image feature batches have different sizes")
        point_projected = self.point_projection(point_raw)
        image_projected = self.image_projection(image_raw)
        if self.normalization == "l2":
            point_normalized = F.normalize(point_projected, p=2, dim=1)
            image_normalized = F.normalize(image_projected, p=2, dim=1)
        else:
            point_normalized = self.point_norm(point_projected)
            image_normalized = self.image_norm(image_projected)
        active_modality = modality or self.modality
        if active_modality == "point":
            image_normalized = torch.zeros_like(image_normalized)
        elif active_modality == "image":
            point_normalized = torch.zeros_like(point_normalized)
        elif active_modality != "fusion":
            raise ValueError(f"Unsupported modality: {active_modality}")
        fused = torch.cat((point_normalized, image_normalized), dim=1)
        return {
            "point_raw": point_raw,
            "image_raw": image_raw,
            "point_normalized": point_normalized,
            "image_normalized": image_normalized,
            "fused": fused,
        }

    def forward(self, points: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(points, images)["fused"])

    def forward_from_features(
        self,
        point_raw: torch.Tensor,
        image_raw: torch.Tensor,
        modality: str | None = None,
    ) -> torch.Tensor:
        return self.classifier(
            self.fuse_features(point_raw, image_raw, modality)["fused"]
        )
