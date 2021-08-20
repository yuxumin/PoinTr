import torch
import torch.nn as nn
import math
import numpy as np
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2



tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]

def get_arch(nlevels, npts):
    logmult = int(math.log2(npts/2048))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch

@MODELS.register_module()
class TopNet(nn.Module):
    def __init__(self, config): # node_feature = 8, encoder_feature = 1024, nlevels = 8, num_pred = 2048
        super().__init__()
        self.node_feature = config.node_feature
        self.encoder_feature = config.encoder_feature
        self.nlevels = config.nlevels       
        self.num_pred = config.num_pred

        self.tarch = get_arch(self.nlevels, self.num_pred)
        self.Top_in_channel = self.encoder_feature + self.node_feature
        self.Top_out_channel = self.node_feature
        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_feature,1)
        )
        self.root_layer = nn.Sequential(
            nn.Linear(self.encoder_feature,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256,64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64 , self.node_feature * int(self.tarch[0])),
            nn.Tanh()
        )
        self.leaf_layer = self.get_tree_layer(self.Top_in_channel, 3, int(self.tarch[-1]))
        self.feature_layers = nn.ModuleList([self.get_tree_layer(self.Top_in_channel, self.Top_out_channel, int(self.tarch[d]) ) for d in range(1, self.nlevels - 1)])
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine
    
    @staticmethod
    def get_tree_layer(in_channel, out_channel, node):
        return nn.Sequential(
            nn.Conv1d(in_channel, in_channel//2, 1),
            nn.BatchNorm1d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel//2, in_channel//4, 1),
            nn.BatchNorm1d(in_channel//4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel//4, in_channel//8, 1),
            nn.BatchNorm1d(in_channel//8),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel//8, out_channel * node, 1),
        )

    def forward(self, xyz):
        bs , n , _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=False)[0] # B 1024
        # decoder
        level10 = self.root_layer(feature_global).reshape(-1, self.node_feature ,int(self.tarch[0])) # B 8 node
        outs = [level10,]
        for i in range(1, self.nlevels):
            last_level = outs[-1]
            expand_feature = feature_global.unsqueeze(2).expand(-1,-1,last_level.shape[2])
            if i == self.nlevels - 1:
                layer_feature = self.leaf_layer(torch.cat([expand_feature,last_level],dim=1)).reshape(bs, 3 ,-1)
            else:
                layer_feature = self.feature_layers[i-1](torch.cat([expand_feature,last_level],dim=1)).reshape(bs, self.node_feature, -1)
            outs.append(nn.Tanh()(layer_feature)) 
        return (outs[-1].transpose(1,2).contiguous(), outs[-1].transpose(1,2).contiguous())
