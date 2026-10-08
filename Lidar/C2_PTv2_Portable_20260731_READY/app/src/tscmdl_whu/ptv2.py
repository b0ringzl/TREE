"""Classification-oriented Point Transformer V2 building blocks.

The implementation follows the PTv2 mode-1 design: grouped vector attention,
position encoding multiplier/bias, and partition-based grid pooling. It uses
native PyTorch operations so the B5 baseline remains runnable in the isolated
Windows environment without the Linux-oriented pointops extension.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def _offset_bounds(offset: torch.Tensor, point_count: int) -> list[tuple[int, int]]:
    if offset.ndim != 1 or offset.numel() == 0:
        raise ValueError("offset must be a non-empty one-dimensional tensor")
    values = [int(value) for value in offset.detach().cpu().tolist()]
    if values[-1] != point_count or any(
        current <= previous for previous, current in zip([0] + values[:-1], values)
    ):
        raise ValueError(f"Invalid packed offsets for {point_count} points: {values}")
    return list(zip([0] + values[:-1], values))


def packed_point_counts(offset: torch.Tensor) -> list[int]:
    """Return per-cloud point counts from cumulative packed offsets."""
    values = [int(value) for value in offset.detach().cpu().tolist()]
    return [current - previous for previous, current in zip([0] + values[:-1], values)]


def knn_query_packed(
    coord: torch.Tensor,
    offset: torch.Tensor,
    neighbours: int,
    *,
    query_chunk_size: int = 2048,
    full_matrix_limit: int = 4096,
) -> torch.Tensor:
    """Compute exact within-cloud kNN indices for packed point clouds."""
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"coord must have shape [N, 3], got {tuple(coord.shape)}")
    if neighbours <= 0:
        raise ValueError("neighbours must be positive")
    if query_chunk_size <= 0 or full_matrix_limit <= 0:
        raise ValueError("query_chunk_size and full_matrix_limit must be positive")

    references: list[torch.Tensor] = []
    with torch.no_grad():
        for start, end in _offset_bounds(offset, coord.shape[0]):
            local = coord[start:end].float()
            count = local.shape[0]
            requested = min(neighbours, count)
            chunks: list[torch.Tensor] = []
            if count <= full_matrix_limit:
                distance = torch.cdist(local, local)
                chunks.append(distance.topk(requested, largest=False).indices)
            else:
                for query in local.split(query_chunk_size):
                    distance = torch.cdist(query, local)
                    chunks.append(distance.topk(requested, largest=False).indices)
            local_reference = torch.cat(chunks, dim=0)
            if requested < neighbours:
                padding = local_reference[:, :1].expand(-1, neighbours - requested)
                local_reference = torch.cat([local_reference, padding], dim=1)
            references.append(local_reference + start)
    return torch.cat(references, dim=0).long()


class PointBatchNorm(nn.Module):
    """Batch normalization for packed point features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return self.norm(values)
        if values.ndim == 3:
            return self.norm(values.transpose(1, 2).contiguous()).transpose(
                1, 2
            ).contiguous()
        raise ValueError(f"Unsupported PointBatchNorm rank: {values.ndim}")


class GroupedLinear(nn.Module):
    """One learned projection per attention group, as defined by PTv2."""

    def __init__(self, in_features: int, groups: int) -> None:
        super().__init__()
        if in_features <= 0 or groups <= 0 or in_features % groups:
            raise ValueError("in_features must be positive and divisible by groups")
        self.in_features = in_features
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(1, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        grouped_shape = (*values.shape[:-1], self.groups, self.in_features // self.groups)
        return (values * self.weight).reshape(grouped_shape).sum(dim=-1)


def drop_path(
    values: torch.Tensor, probability: float, training: bool
) -> torch.Tensor:
    if probability == 0.0 or not training:
        return values
    keep_probability = 1.0 - probability
    shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    random = keep_probability + torch.rand(
        shape, dtype=values.dtype, device=values.device
    )
    random.floor_()
    return values * random / keep_probability


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop-path probability must be in [0, 1)")
        self.probability = probability

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return drop_path(values, self.probability, self.training)


class GroupedVectorAttention(nn.Module):
    """PTv2 grouped vector attention with multiplicative position encoding."""

    def __init__(
        self,
        channels: int,
        groups: int,
        *,
        qkv_bias: bool = True,
        pe_multiplier: bool = True,
        pe_bias: bool = True,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups:
            raise ValueError("channels must be positive and divisible by groups")
        self.channels = channels
        self.groups = groups
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias
        self.query = nn.Sequential(
            nn.Linear(channels, channels, bias=qkv_bias),
            PointBatchNorm(channels),
            nn.ReLU(inplace=True),
        )
        self.key = nn.Sequential(
            nn.Linear(channels, channels, bias=qkv_bias),
            PointBatchNorm(channels),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(channels, channels, bias=qkv_bias)
        if pe_multiplier:
            self.position_multiplier = nn.Sequential(
                nn.Linear(3, channels),
                PointBatchNorm(channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, channels),
            )
        if pe_bias:
            self.position_bias = nn.Sequential(
                nn.Linear(3, channels),
                PointBatchNorm(channels),
                nn.ReLU(inplace=True),
                nn.Linear(channels, channels),
            )
        self.weight_encoding = nn.Sequential(
            GroupedLinear(channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups),
        )
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(
        self, feat: torch.Tensor, coord: torch.Tensor, reference_index: torch.Tensor
    ) -> torch.Tensor:
        if reference_index.ndim != 2 or reference_index.shape[0] != feat.shape[0]:
            raise ValueError("reference_index must have shape [N, K]")
        if reference_index.numel() and (
            reference_index.min() < 0 or reference_index.max() >= feat.shape[0]
        ):
            raise ValueError("reference_index contains an out-of-range point index")

        query = self.query(feat)
        key = self.key(feat)[reference_index]
        value = self.value(feat)[reference_index]
        relative_position = coord[reference_index] - coord.unsqueeze(1)
        relation = key - query.unsqueeze(1)
        if self.pe_multiplier:
            relation = relation * self.position_multiplier(relative_position)
        if self.pe_bias:
            position_bias = self.position_bias(relative_position)
            relation = relation + position_bias
            value = value + position_bias

        weight = self.weight_encoding(relation)
        weight = self.attention_dropout(torch.softmax(weight, dim=1))
        grouped_value = value.reshape(
            value.shape[0],
            value.shape[1],
            self.groups,
            self.channels // self.groups,
        )
        return torch.einsum("nkgi,nkg->ngi", grouped_value, weight).reshape(
            feat.shape[0], self.channels
        )


class PointTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int,
        *,
        qkv_bias: bool,
        pe_multiplier: bool,
        pe_bias: bool,
        attention_dropout: float,
        drop_path_probability: float,
    ) -> None:
        super().__init__()
        self.pre = nn.Linear(channels, channels, bias=False)
        self.pre_norm = PointBatchNorm(channels)
        self.attention = GroupedVectorAttention(
            channels,
            groups,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attention_dropout=attention_dropout,
        )
        self.attention_norm = PointBatchNorm(channels)
        self.post = nn.Linear(channels, channels, bias=False)
        self.post_norm = PointBatchNorm(channels)
        self.activation = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path_probability)

    def forward(
        self, feat: torch.Tensor, coord: torch.Tensor, reference_index: torch.Tensor
    ) -> torch.Tensor:
        identity = feat
        feat = self.activation(self.pre_norm(self.pre(feat)))
        feat = self.attention(feat, coord, reference_index)
        feat = self.activation(self.attention_norm(feat))
        feat = self.post_norm(self.post(feat))
        return self.activation(identity + self.drop_path(feat))


class PointTransformerBlockSequence(nn.Module):
    def __init__(
        self,
        depth: int,
        channels: int,
        groups: int,
        neighbours: int,
        *,
        qkv_bias: bool,
        pe_multiplier: bool,
        pe_bias: bool,
        attention_dropout: float,
        drop_path_probabilities: Sequence[float],
    ) -> None:
        super().__init__()
        if depth <= 0 or len(drop_path_probabilities) != depth:
            raise ValueError("depth must match drop_path_probabilities")
        self.neighbours = neighbours
        self.blocks = nn.ModuleList(
            PointTransformerBlock(
                channels,
                groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attention_dropout=attention_dropout,
                drop_path_probability=float(drop_path_probabilities[index]),
            )
            for index in range(depth)
        )

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        offset: torch.Tensor,
        reference_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference_index is None:
            reference_index = knn_query_packed(coord, offset, self.neighbours)
        for block in self.blocks:
            feat = block(feat, coord, reference_index)
        return feat


class GridPool(nn.Module):
    """Partition points into non-overlapping grids and aggregate each partition."""

    def __init__(
        self, in_channels: int, out_channels: int, grid_size: float
    ) -> None:
        super().__init__()
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        self.grid_size = float(grid_size)
        self.projection = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = PointBatchNorm(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self, coord: torch.Tensor, feat: torch.Tensor, offset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.activation(self.norm(self.projection(feat)))
        pooled_coord_parts: list[torch.Tensor] = []
        pooled_feat_parts: list[torch.Tensor] = []
        pooled_offsets: list[int] = []
        pooled_total = 0

        for start, end in _offset_bounds(offset, coord.shape[0]):
            local_coord = coord[start:end]
            local_feat = feat[start:end]
            origin = local_coord.min(dim=0).values
            voxel = torch.floor((local_coord - origin) / self.grid_size).long()
            _, inverse, counts = torch.unique(
                voxel,
                dim=0,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            cluster_count = int(counts.shape[0])
            coord_sum = local_coord.new_zeros((cluster_count, 3))
            coord_sum.index_add_(0, inverse, local_coord)
            pooled_coord = coord_sum / counts.to(local_coord.dtype).unsqueeze(1)

            pooled_feat = local_feat.new_full(
                (cluster_count, local_feat.shape[1]),
                torch.finfo(local_feat.dtype).min,
            )
            pooled_feat.scatter_reduce_(
                0,
                inverse.unsqueeze(1).expand(-1, local_feat.shape[1]),
                local_feat,
                reduce="amax",
                include_self=True,
            )
            pooled_coord_parts.append(pooled_coord)
            pooled_feat_parts.append(pooled_feat)
            pooled_total += cluster_count
            pooled_offsets.append(pooled_total)

        return (
            torch.cat(pooled_coord_parts, dim=0),
            torch.cat(pooled_feat_parts, dim=0),
            torch.tensor(pooled_offsets, dtype=torch.long, device=coord.device),
        )


class PointTransformerEncoderStage(nn.Module):
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        neighbours: int,
        grid_size: float,
        *,
        qkv_bias: bool,
        pe_multiplier: bool,
        pe_bias: bool,
        attention_dropout: float,
        drop_path_probabilities: Sequence[float],
    ) -> None:
        super().__init__()
        self.pool = GridPool(in_channels, out_channels, grid_size)
        self.blocks = PointTransformerBlockSequence(
            depth,
            out_channels,
            groups,
            neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attention_dropout=attention_dropout,
            drop_path_probabilities=drop_path_probabilities,
        )

    def forward(
        self, coord: torch.Tensor, feat: torch.Tensor, offset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coord, feat, offset = self.pool(coord, feat, offset)
        feat = self.blocks(coord, feat, offset)
        return coord, feat, offset


class PointTransformerV2Classifier(nn.Module):
    """PTv2 encoder with global average pooling for tree-species classification."""

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 3,
        patch_depth: int = 1,
        patch_channels: int = 48,
        patch_groups: int = 6,
        patch_neighbours: int = 8,
        encoder_depths: Sequence[int] = (2, 2, 6, 2),
        encoder_channels: Sequence[int] = (96, 192, 384, 384),
        encoder_groups: Sequence[int] = (12, 24, 48, 48),
        encoder_neighbours: Sequence[int] = (16, 16, 16, 16),
        grid_sizes: Sequence[float] = (0.06, 0.12, 0.24, 0.48),
        qkv_bias: bool = True,
        pe_multiplier: bool = True,
        pe_bias: bool = True,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        classifier_hidden: int = 256,
        classifier_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        stage_count = len(encoder_depths)
        if not all(
            len(values) == stage_count
            for values in (
                encoder_channels,
                encoder_groups,
                encoder_neighbours,
                grid_sizes,
            )
        ):
            raise ValueError("All encoder stage configurations must have equal length")
        if patch_depth <= 0 or classifier_hidden <= 0:
            raise ValueError("patch_depth and classifier_hidden must be positive")
        if not 0.0 <= classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1)")

        self.num_classes = num_classes
        self.patch_neighbours = patch_neighbours
        self.patch_projection = nn.Sequential(
            nn.Linear(in_channels, patch_channels, bias=False),
            PointBatchNorm(patch_channels),
            nn.ReLU(inplace=True),
        )
        total_depth = patch_depth + sum(int(depth) for depth in encoder_depths)
        drop_rates = torch.linspace(0.0, drop_path_rate, total_depth).tolist()
        cursor = 0
        self.patch_blocks = PointTransformerBlockSequence(
            patch_depth,
            patch_channels,
            patch_groups,
            patch_neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attention_dropout=attention_dropout,
            drop_path_probabilities=drop_rates[cursor : cursor + patch_depth],
        )
        cursor += patch_depth

        stages: list[PointTransformerEncoderStage] = []
        input_channels = patch_channels
        for depth, channels, groups, neighbours, grid_size in zip(
            encoder_depths,
            encoder_channels,
            encoder_groups,
            encoder_neighbours,
            grid_sizes,
        ):
            depth = int(depth)
            stages.append(
                PointTransformerEncoderStage(
                    depth,
                    input_channels,
                    int(channels),
                    int(groups),
                    int(neighbours),
                    float(grid_size),
                    qkv_bias=qkv_bias,
                    pe_multiplier=pe_multiplier,
                    pe_bias=pe_bias,
                    attention_dropout=attention_dropout,
                    drop_path_probabilities=drop_rates[cursor : cursor + depth],
                )
            )
            cursor += depth
            input_channels = int(channels)
        self.encoder_stages = nn.ModuleList(stages)
        self.classifier = nn.Sequential(
            nn.Linear(input_channels, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )
        self.last_stage_point_counts: list[list[int]] = []
        self.configuration = {
            "in_channels": in_channels,
            "patch_depth": patch_depth,
            "patch_channels": patch_channels,
            "patch_groups": patch_groups,
            "patch_neighbours": patch_neighbours,
            "encoder_depths": list(encoder_depths),
            "encoder_channels": list(encoder_channels),
            "encoder_groups": list(encoder_groups),
            "encoder_neighbours": list(encoder_neighbours),
            "grid_sizes": list(grid_sizes),
            "qkv_bias": qkv_bias,
            "pe_multiplier": pe_multiplier,
            "pe_bias": pe_bias,
            "attention_dropout": attention_dropout,
            "drop_path_rate": drop_path_rate,
            "classifier_hidden": classifier_hidden,
            "classifier_dropout": classifier_dropout,
            "num_classes": num_classes,
        }

    def forward(
        self, points: torch.Tensor, initial_reference_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError(f"points must have shape [B, N, 3], got {tuple(points.shape)}")
        batch_size, point_count, _ = points.shape
        coord = points.reshape(batch_size * point_count, 3).contiguous()
        feat = self.patch_projection(coord)
        offset = torch.arange(
            1,
            batch_size + 1,
            dtype=torch.long,
            device=points.device,
        ) * point_count

        packed_reference: torch.Tensor | None = None
        if initial_reference_index is not None:
            expected = (batch_size, point_count, self.patch_neighbours)
            if tuple(initial_reference_index.shape) != expected:
                raise ValueError(
                    "initial_reference_index shape mismatch: "
                    f"{tuple(initial_reference_index.shape)} != {expected}"
                )
            base = (
                torch.arange(batch_size, device=points.device)
                .mul(point_count)
                .view(batch_size, 1, 1)
            )
            packed_reference = (
                initial_reference_index.long() + base
            ).reshape(batch_size * point_count, self.patch_neighbours)
        feat = self.patch_blocks(coord, feat, offset, packed_reference)

        stage_counts = [[point_count] * batch_size]
        for stage in self.encoder_stages:
            coord, feat, offset = stage(coord, feat, offset)
            stage_counts.append(packed_point_counts(offset))
        self.last_stage_point_counts = stage_counts

        pooled = [
            feat[start:end].mean(dim=0)
            for start, end in _offset_bounds(offset, feat.shape[0])
        ]
        return self.classifier(torch.stack(pooled, dim=0))

