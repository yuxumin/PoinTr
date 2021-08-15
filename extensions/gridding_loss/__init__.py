# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-30 09:56:06
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:19:43
# @Email:  cshzxie@gmail.com

import torch

import gridding_distance


class GriddingDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, min_x, max_x, min_y, max_y, min_z, max_z, pred_cloud, gt_cloud):
        pred_grid, pred_grid_pt_weights, pred_grid_pt_indexes = gridding_distance.forward(
            min_x, max_x, min_y, max_y, min_z, max_z, pred_cloud)
        # print(pred_grid.size())               # torch.Size(batch_size, n_grid_vertices, 8)
        # print(pred_grid_pt_weights.size())    # torch.Size(batch_size, n_pts, 8, 3)
        # print(pred_grid_pt_indexes.size())    # torch.Size(batch_size, n_pts, 8)
        gt_grid, gt_grid_pt_weights, gt_grid_pt_indexes = gridding_distance.forward(
            min_x, max_x, min_y, max_y, min_z, max_z, gt_cloud)
        # print(gt_grid.size())                 # torch.Size(batch_size, n_grid_vertices, 8)
        # print(gt_grid_pt_weights.size())      # torch.Size(batch_size, n_pts, 8, 3)
        # print(gt_grid_pt_indexes.size())      # torch.Size(batch_size, n_pts, 8)

        ctx.save_for_backward(pred_grid_pt_weights, pred_grid_pt_indexes, gt_grid_pt_weights, gt_grid_pt_indexes)
        return pred_grid, gt_grid

    @staticmethod
    def backward(ctx, grad_pred_grid, grad_gt_grid):
        pred_grid_pt_weights, pred_grid_pt_indexes, gt_grid_pt_weights, gt_grid_pt_indexes = ctx.saved_tensors

        grad_pred_cloud = gridding_distance.backward(pred_grid_pt_weights, pred_grid_pt_indexes, grad_pred_grid)
        # print(grad_pred_cloud.size())     # torch.Size(batch_size, n_pts, 3)
        grad_gt_cloud = gridding_distance.backward(gt_grid_pt_weights, gt_grid_pt_indexes, grad_gt_grid)
        # print(grad_gt_cloud.size())       # torch.Size(batch_size, n_pts, 3)

        return None, None, None, None, None, None, grad_pred_cloud, grad_gt_cloud


class GriddingDistance(torch.nn.Module):
    def __init__(self, scale=1):
        super(GriddingDistance, self).__init__()
        self.scale = scale

    def forward(self, pred_cloud, gt_cloud):
        '''
        pred_cloud(b, n_pts1, 3)
        gt_cloud(b, n_pts2, 3)
        '''
        pred_cloud = pred_cloud * self.scale / 2
        gt_cloud = gt_cloud * self.scale / 2

        min_pred_x = torch.min(pred_cloud[:, :, 0])
        max_pred_x = torch.max(pred_cloud[:, :, 0])
        min_pred_y = torch.min(pred_cloud[:, :, 1])
        max_pred_y = torch.max(pred_cloud[:, :, 1])
        min_pred_z = torch.min(pred_cloud[:, :, 2])
        max_pred_z = torch.max(pred_cloud[:, :, 2])

        min_gt_x = torch.min(gt_cloud[:, :, 0])
        max_gt_x = torch.max(gt_cloud[:, :, 0])
        min_gt_y = torch.min(gt_cloud[:, :, 1])
        max_gt_y = torch.max(gt_cloud[:, :, 1])
        min_gt_z = torch.min(gt_cloud[:, :, 2])
        max_gt_z = torch.max(gt_cloud[:, :, 2])

        min_x = torch.floor(torch.min(min_pred_x, min_gt_x)) - 1
        max_x = torch.ceil(torch.max(max_pred_x, max_gt_x)) + 1
        min_y = torch.floor(torch.min(min_pred_y, min_gt_y)) - 1
        max_y = torch.ceil(torch.max(max_pred_y, max_gt_y)) + 1
        min_z = torch.floor(torch.min(min_pred_z, min_gt_z)) - 1
        max_z = torch.ceil(torch.max(max_pred_z, max_gt_z)) + 1

        _pred_clouds = torch.split(pred_cloud, 1, dim=0)
        _gt_clouds = torch.split(gt_cloud, 1, dim=0)
        pred_grids = []
        gt_grids = []
        for pc, gc in zip(_pred_clouds, _gt_clouds):
            non_zeros = torch.sum(pc, dim=2).ne(0)
            pc = pc[non_zeros].unsqueeze(dim=0)
            non_zeros = torch.sum(gc, dim=2).ne(0)
            gc = gc[non_zeros].unsqueeze(dim=0)
            pred_grid, gt_grid = GriddingDistanceFunction.apply(min_x, max_x, min_y, max_y, min_z, max_z, pc, gc)
            pred_grids.append(pred_grid)
            gt_grids.append(gt_grid)

        return torch.cat(pred_grids, dim=0).contiguous(), torch.cat(gt_grids, dim=0).contiguous()


class GriddingLoss(torch.nn.Module):
    def __init__(self, scales=[], alphas=[]):
        super(GriddingLoss, self).__init__()
        self.scales = scales
        self.alphas = alphas
        self.gridding_dists = [GriddingDistance(scale=s) for s in scales]
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, pred_cloud, gt_cloud):
        gridding_loss = None
        n_dists = len(self.scales)

        for i in range(n_dists):
            alpha = self.alphas[i]
            gdist = self.gridding_dists[i]
            pred_grid, gt_grid = gdist(pred_cloud, gt_cloud)

            if gridding_loss is None:
                gridding_loss = alpha * self.l1_loss(pred_grid, gt_grid)
            else:
                gridding_loss += alpha * self.l1_loss(pred_grid, gt_grid)

        return gridding_loss
