from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional
from knn_cuda import KNN
from pointnet2_ops.pointnet2_utils import (
    furthest_point_sample,
    gather_operation,
    grouping_operation,
)
from torch import nn, einsum


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        k: int
        use_xyz: boolean
        idx: Tensor, (B, npoint, nsample)

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = gather_operation(
        xyz, furthest_point_sample(xyz_flipped, npoint)
    )  # (B, 3, npoint)
    if idx is None:
        _, idx = KNN(k, transpose_mode=True)(
            xyz_flipped, new_xyz.permute(0, 2, 1).contiguous()
        )
        idx = idx.int()
        # idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points, idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    # new_xyz = torch.mean(xyz, dim=1)
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        _, idx = KNN(k=k, transpose_mode=True)(
            x.transpose(1, 2), x.transpose(1, 2)
        )  # (batch_size, num_points, k)
    device = torch.device("cuda")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = (
        idx + idx_base
    )  # batch_size * num_points * k + range(0, batch_size)*num_points

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[
        idx, :
    ]  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


# def square_distance(src, dst):
#     """
#     Calculate Euclid distance between each two points.
#
#     src^T * dst = xn * xm + yn * ym + zn * zm；
#     sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#     sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#     dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#          = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#
#     Input:
#         src: source points, [B, N, C]
#         dst: target points, [B, M, C]
#     Output:
#         dist: per-point square distance, [B, N, M]
#     """
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist
#
#
# def query_knn(nsample, xyz, new_xyz, include_self=True):
#     """Find k-NN of new_xyz in xyz"""
#     pad = 0 if include_self else 1
#     sqrdists = square_distance(new_xyz, xyz)  # B, S, N
#     idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :, pad: nsample + pad]
#     return idx.int()

# def knn(x, k):
#     inner = -2 * torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
#
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx


class MlpRes(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MlpRes, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class MlpConv(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MlpConv, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=(1, 1),
        stride=(1, 1),
        if_bn=True,
        activation_fn: Optional[Callable] = torch.relu,
    ):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, x):
        out = self.conv(x)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class PointNetSaModuleKNN(nn.Module):
    def __init__(
        self,
        npoint,
        nsample,
        in_channel,
        mlp,
        if_bn=True,
        group_all=False,
        use_xyz=True,
        if_idx=False,
    ):
        """
        Args:
            npoint: int, number of points to sample
            nsample: int, number of points in each local region
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNetSaModuleKNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(
            Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None)
        )
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            points: Tensor, (B, f, N)
            idx: Tensor, (B, npoint, nsample)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_points: Tensor, (B, mlp[-1], npoint)
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz
            )
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(
                xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx
            )

        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]

        if self.if_idx:
            return new_xyz, new_points, idx
        else:
            return new_xyz, new_points


class Transformer(nn.Module):
    def __init__(
        self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4
    ):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1),
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

        self.query_knn = KNN(k=n_knn, transpose_mode=True)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        # idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        _, idx_knn = self.query_knn(pos_flipped, pos_flipped)
        idx_knn = idx_knn.int()
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(
            pos, idx_knn
        )  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum("b c i j, b c i j -> b c i", attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity


class SkipTransformer(nn.Module):
    def __init__(
        self,
        in_channel,
        pos_channel,
        dim=256,
        n_knn=16,
        pos_hidden_dim=64,
        attn_hidden_multiplier=4,
    ):
        super(SkipTransformer, self).__init__()
        self.mlp_v = MlpRes(
            in_dim=in_channel * 2, hidden_dim=in_channel, out_dim=in_channel
        )
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(pos_channel, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1),
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1),
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

        self.query_knn = KNN(k=self.n_knn, transpose_mode=True)

    def forward(self, pos, key, query):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        # idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped, include_self=include_self)
        _, idx_knn = self.query_knn(pos_flipped, pos_flipped)
        idx_knn = idx_knn.int()

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(
            pos, idx_knn
        )  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding  #

        agg = einsum("b c i j, b c i j -> b c i", attention, value)  # b, dim, n
        y = self.conv_end(agg)

        return y + identity


class FeatureExtractor(nn.Module):
    def __init__(self, feat_channel=3, out_dim=1024):
        """Encoder that encodes information of partial point cloud"""
        super(FeatureExtractor, self).__init__()
        self.feat_channel = feat_channel
        self.sa_module_1 = PointNetSaModuleKNN(
            512,
            16,
            self.feat_channel,
            [64, 128],
            group_all=False,
            if_bn=False,
            if_idx=True,
        )
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNetSaModuleKNN(
            128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True
        )
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNetSaModuleKNN(
            None, None, 256, [512, out_dim], group_all=True, if_bn=False
        )

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud[:, 0:3, :].contiguous()
        if self.feat_channel != 3:
            l0_points = point_cloud[:, 3:, :].contiguous()
        else:
            l0_points = point_cloud[:, 0:3, :].contiguous()

        l1_xyz, l1_points, idx1 = self.sa_module_1(
            l0_xyz, l0_points
        )  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(
            l1_xyz, l1_points
        )  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(
            l2_xyz, l2_points
        )  # (B, 3, 1), (B, out_dim, 1)

        return l3_points


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MlpRes(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MlpRes(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MlpRes(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1), nn.ReLU(), nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(
            torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1)
        )  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1.0):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MlpConv(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MlpConv(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])

        self.skip_transformer = SkipTransformer(in_channel=128, pos_channel=3, dim=64)

        self.mlp_ps = MlpConv(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(
            32, 128, up_factor, up_factor, bias=False
        )  # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MlpRes(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MlpConv(in_channel=128, layer_dims=[64, 3])
        self.feat_extract = FeatureExtractor(3, dim_feat)

    def forward(self, pcd_prev, feat_global, k_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            k_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape  # (B, 3, N_prev)
        feat_1 = self.mlp_1(pcd_prev)  # (B, 128, N_prev)
        feat_1 = torch.cat(
            [
                feat_1,
                torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                feat_global.repeat(1, 1, feat_1.size(2)),
            ],
            1,
        )  # (B, 128*2 + 512, N_prev)
        query = self.mlp_2(feat_1)  # (B, 128, N_prev)

        hidden = self.skip_transformer(
            pcd_prev, k_prev if k_prev is not None else query, query
        )  # (B, 128, N_prev) 内部做了差分嵌入

        feat_child = self.mlp_ps(hidden)  # (B, 32, N_prev)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        hidden_up = self.up_sampler(hidden)  # (B, 128, N_prev * up_factor)
        k_curr = self.mlp_delta_feature(
            torch.cat([feat_child, hidden_up], 1)
        )  # (B, 128, N_prev * up_factor)

        delta = torch.tanh(self.mlp_delta(torch.relu(k_curr))) / (
            self.radius**self.i
        )  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)  # (B, 3, N_prev * up_factor)
        pcd_coarse = pcd_child + delta  # (B, 3, N_prev * up_factor)

        feat_coarse = self.feat_extract(pcd_coarse)  # (B, 512, 1)

        return (
            pcd_coarse,
            pcd_child,
            delta,
            k_curr,
            feat_coarse,
        )  # (B, 3, N_prev * up_factor), (B, 128, N_prev * up_factor), (B, 512, 1)


class DGCNN(nn.Module):
    def __init__(self, k_num, in_channel):
        super(DGCNN, self).__init__()
        self.k = k_num

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        batch_size = x.size(0)
        # x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)
        x = nn.functional.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        return x.reshape(batch_size, -1, 1)


class LocalBranch(nn.Module):
    def __init__(self, class_num, i):
        super(LocalBranch, self).__init__()
        self.cls = class_num
        self.radius = 0.2
        self.i = i
        if self.i == 0:
            self.n_agent = 96
        elif self.i == 1:
            self.n_agent = 64
        elif self.i == 2:
            self.n_agent = 32
        self.n_knn = 20
        self.agent_knn = KNN(k=self.n_knn, transpose_mode=False)
        self.agent_cord_mlp = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
        )
        self.agent_label_mlp = nn.Sequential(
            nn.Conv2d(self.cls, 32, kernel_size=1, stride=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
        )
        self.cmp_delta = nn.Sequential(
            nn.Conv1d(64, 128, 1, 1),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1, 1),
        )
        self.label_delta = nn.Sequential(
            nn.Conv1d(64, 128, 1, 1),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1, 1),
            nn.ReLU(),
            nn.Conv1d(64, self.cls, 1, 1),
        )

    def forward(self, pcd_coarse, trans_cord, k_prev):
        b, _, n_pts = pcd_coarse.shape
        if self.i == 0:
            trans_dist = torch.sum((trans_cord.transpose(1, 2)) ** 2, 2)
            _, idx_agent = torch.topk(
                trans_dist, self.n_agent, largest=True
            )  # b, n_agent
        else:
            trans_dist = torch.sum((trans_cord.transpose(1, 2)) ** 2, 2)
            _, idx_tk = torch.topk(
                trans_dist, self.n_agent // 2, largest=True
            )  # b, n_agent
            _, idx_bk = torch.topk(
                trans_dist, self.n_agent // 2, largest=False
            )  # b, n_agent
            idx_agent = torch.cat([idx_tk, idx_bk], dim=1)

        # idx_agent = furthest_point_sample(pcd_coarse[:, 0:3, :].transpose(1, 2).contiguous(), n_agent)  # b, n_agent
        pcd_agent = gather_operation(pcd_coarse.contiguous(), idx_agent.int())
        _, idx_agent_knn = self.agent_knn(
            pcd_coarse[:, 0:3, :], pcd_agent[:, 0:3, :]
        )  # [bs  x k x nq]
        idx_agent_knn = idx_agent_knn.transpose(1, 2).contiguous().int()
        agent_cord_patch = grouping_operation(
            pcd_coarse[:, 0:3, :].contiguous(), idx_agent_knn
        )  # b, 3, agent_num, k
        agent_label_patch = grouping_operation(
            pcd_coarse[:, 3:, :].contiguous(), idx_agent_knn
        )  # b, 12, agent_num, k
        agent_cord_patch = agent_cord_patch - agent_cord_patch[:, :, :, 0].unsqueeze(
            3
        ).repeat(
            1, 1, 1, self.n_knn
        )  # b, 3, k, agent_num
        agent_label_patch = agent_label_patch - agent_label_patch[:, :, :, 0].unsqueeze(
            3
        ).repeat(
            1, 1, 1, self.n_knn
        )  # b, 3, k, agent_num
        agent_cord_patch_feat = self.agent_cord_mlp(
            agent_cord_patch
        )  # (B, 64, agent_num, k)
        agent_label_patch_feat = self.agent_label_mlp(
            agent_label_patch
        )  # (B, 64, agent_num, k)
        agent_cord_patch_feat = torch.max(agent_cord_patch_feat, dim=3)[
            0
        ]  # b, 64, agent_num
        agent_label_patch_feat = torch.max(agent_label_patch_feat, dim=3)[
            0
        ]  # b, 64, agent_num
        child_cmp = torch.tanh(
            self.cmp_delta(torch.relu(agent_cord_patch_feat))
        ) * torch.tensor(
            self.radius
        )  # (B, 3, agent_num)
        child_label = torch.tanh(
            self.label_delta(torch.relu(agent_label_patch_feat))
        ) * torch.tensor(
            self.radius
        )  # (B, 12, agent_num)
        local_trans = torch.cat([child_cmp, child_label], dim=1)  # b, 3+12, agent_num
        pcd_local = pcd_agent + local_trans

        pcd_local = torch.cat([pcd_coarse, pcd_local], dim=2)
        k_prev_agent = gather_operation(k_prev.contiguous(), idx_agent.int())
        k_prev = torch.cat([k_prev, k_prev_agent], dim=2)

        return pcd_local, k_prev  # b, 3+12, agent_num


class LabelBranch(nn.Module):
    def __init__(self, class_num, k_num, dim_feat, up_factor=2, i=0, probability=0.9):
        super(LabelBranch, self).__init__()
        self.i = i
        self.prob = probability
        self.up_factor = up_factor
        self.cls = class_num
        self.mlp_1 = MlpConv(in_channel=self.cls, layer_dims=[64, 128])
        # self.feat_fuse = nn.Linear(dim_feat * 2, dim_feat)
        self.label_feat_extract = DGCNN(k_num, self.cls)
        self.d_k = 1.0
        self.key_start = nn.Conv1d(dim_feat, dim_feat, 1)
        self.query_start = nn.Conv1d(dim_feat, dim_feat, 1)
        self.dropout = nn.Dropout(0.1)
        self.linear_end = nn.Conv1d(dim_feat, dim_feat, 1)
        self.mlp_2 = MlpConv(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])
        self.mlp_ps = MlpConv(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(
            32, 128, up_factor, up_factor, bias=False
        )  # point-wise splitting
        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.skip_transformer = SkipTransformer(
            in_channel=128, pos_channel=self.cls, dim=64
        )
        self.mlp_delta_feature = MlpRes(in_dim=256, hidden_dim=128, out_dim=128)
        self.mlp_delta = MlpConv(in_channel=128, layer_dims=[64, self.cls])

    def forward(self, pcd, feat_pcd, feat_cord, k_cmp):
        xyz = pcd[:, 0:3, :].contiguous()
        pcd_label = pcd[:, 3:, :].contiguous()
        b, _, n = pcd_label.shape
        parent_feat = self.mlp_1(pcd_label)  # b,128,n

        _, idx = KNN(k=20, transpose_mode=True)(
            xyz.transpose(1, 2), xyz.transpose(1, 2)
        )  # (batch_size, num_points, k)
        idx = idx.int()
        label_nn = grouping_operation(pcd_label, idx)  # b, 12, n, k
        label_nn = label_nn.permute(0, 2, 3, 1)
        pcd_label_flip = pcd_label.transpose(1, 2).unsqueeze(2).repeat(1, 1, 20, 1)
        label_nn = (
            torch.cat([label_nn - pcd_label_flip, pcd_label_flip], dim=3)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        feat_label = self.label_feat_extract(label_nn)

        key = feat_label
        query = feat_cord
        value = feat_pcd
        feat_query = self.query_start(query)  # b,256
        feat_key = self.key_start(key)  # b,256
        attention = torch.matmul(feat_query, feat_key.transpose(1, 2)) / np.sqrt(
            self.d_k
        )  # b,512,512
        attention = torch.softmax(attention, -1)  # b,512,512
        attention = self.dropout(attention)  # b,512,512
        value = value  # b,512,1
        out = einsum("b d d, b d i -> b d i", attention, value)  # b,512,1
        out = self.linear_end(out)
        feat_fuse = out  # b,512,1

        parent_feat = torch.cat(
            [
                parent_feat,  # b,128,n
                torch.max(parent_feat, 2, keepdim=True)[0].repeat(
                    1, 1, parent_feat.shape[2]
                ),  # b,128,n
                feat_fuse.repeat(1, 1, parent_feat.shape[2]),
            ],
            dim=1,
        )  # b,512,n
        parent_feat = self.mlp_2(parent_feat)
        child_feat = self.mlp_ps(parent_feat)
        child_feat_up = self.ps(child_feat)
        child_label_up = self.up_sampler(pcd_label)

        child_trans_feat = self.skip_transformer(child_label_up, child_feat_up, k_cmp)
        child_trans_feat = self.mlp_delta_feature(
            torch.cat([child_trans_feat, child_feat_up], 1)
        )
        child_trans = torch.tanh(self.mlp_delta(torch.relu(child_trans_feat))) / (
            self.prob**self.i
        )
        child_label = child_label_up + child_trans

        return child_label, child_label_up, child_trans, child_trans_feat


class Decoder(nn.Module):
    def __init__(
        self,
        class_num,
        k_num=20,
        dim_feat=512,
        num_p0=512,
        radius=1.0,
        up_factors=None,
    ):
        super(Decoder, self).__init__()
        self.cls = class_num
        self.num_p0 = num_p0

        if up_factors is None:
            self.up_factors = [1]
        else:
            self.up_factors = [1] + list(up_factors)

        uppers = []
        for i, factor in enumerate(self.up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))
        self.uppers = nn.ModuleList(uppers)

        self.step_feat_extract = FeatureExtractor(
            feat_channel=self.cls, out_dim=dim_feat
        )

        label_uppers = []
        for i, factor in enumerate(self.up_factors):
            label_uppers.append(
                LabelBranch(
                    class_num,
                    k_num,
                    dim_feat,
                    up_factor=factor,
                    i=i,
                    probability=radius,
                )
            )
        self.label_uppers = nn.ModuleList(label_uppers)

        local_uppers = []
        for i, factor in enumerate(self.up_factors):
            local_uppers.append(LocalBranch(class_num, i=i))
        self.local_uppers = nn.ModuleList(local_uppers)

    def forward(self, feat, partial):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        partial_flip = partial.transpose(1, 2)  # (B, 3+12, n)
        pcd = gather_operation(
            partial_flip.contiguous(),
            furthest_point_sample(
                partial_flip[:, 0:3, :].transpose(1, 2).contiguous(), self.num_p0
            ),
        )  # (B, 3, num_pc)
        arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        k_prev = None
        step_feat = feat
        for i, upper in enumerate(self.uppers):
            pcd_coarse, coarse_child, trans_coarse, k_prev_corse, feat_coarse = upper(
                pcd[:, 0:3, :].contiguous(), step_feat, k_prev
            )  # (B, 3, N_prev * up_factor), (B, 128, N_prev * up_factor), (B, 512, 1)

            pcd_label, label_child, trans_label, k_prev = self.label_uppers[i](
                pcd, step_feat, feat_coarse, k_prev_corse
            )  # (B, 12, N_prev * up_factor), (B, 128, N_prev * up_factor)

            pcd = torch.cat(
                [pcd_coarse, pcd_label], dim=1
            )  # (B, 3 + 12, N_prev * up_factor)
            if i < len(self.uppers) - 1:
                pcd, k_prev = self.local_uppers[i](
                    pcd, trans_coarse, k_prev
                )  # (B, 3 + 12, 1024)
            step_feat = self.step_feat_extract(pcd)

            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd


class CasFusionNet(nn.Module):
    def __init__(
        self,
        class_num,
        dim_feat=1024,
        num_p0=1024,
        radius=1,
        up_factors=(2, 2, 2),
    ):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_p0: int
            radius: searching radius
            up_factors: list of int
        """
        super(CasFusionNet, self).__init__()
        self.cls = class_num
        self.feat_extractor = FeatureExtractor(feat_channel=3, out_dim=dim_feat)
        self.decoder = Decoder(
            class_num,
            k_num=20,
            dim_feat=dim_feat,
            num_p0=num_p0,
            radius=radius,
            up_factors=up_factors,
        )

        self.label_mlp = nn.Sequential(
            nn.Conv1d(dim_feat + 3, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, self.cls, kernel_size=1),
        )

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        b, n, d = point_cloud.shape
        pcd_bnc = point_cloud  # (B, n, 3)
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()  # (B, 3, N)
        feat = self.feat_extractor(point_cloud)  # (B, 512, 1)
        x = torch.cat((point_cloud, feat.repeat(1, 1, n)), dim=1)  # (B, 512+3, n)
        label = self.label_mlp(x)  # (B, 512+3, n) -> (B, 16, n)
        pcd_bnc = torch.cat([pcd_bnc, label.transpose(1, 2)], dim=2)  # (B, n, 3+12)
        out = self.decoder(feat, pcd_bnc)
        return out
