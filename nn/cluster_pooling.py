import torch
from torch_scatter import scatter


class ClusterPooling(torch.nn.Module):

    def __init__(self, target, reduction='mean'):
        super(ClusterPooling, self).__init__()

        self.target = target  # target scale
        self.reduction = reduction

        self.pos_cache = None

    def forward(self, data):

        data.x = scatter(data.x, data['scale' + str(self.target) + '_cluster_map'], dim=0, reduce=self.reduction)
        data.edge_index = data['scale' + str(self.target) + '_edge_index']
        if hasattr(data, 'h'):
            data.h = scatter(data.h, data['scale' + str(self.target) + '_cluster_map'], dim=0, reduce=self.reduction)
        if hasattr(data, 'edge_align'):
            self.pos_cache = data.pos.clone()
            data.pos = data.pos[data['scale' + str(self.target) + '_sample_index']]
            data.edge_align = data['scale' + str(self.target) + '_edge_align']

        return data

    def unpool(self, data):

        data.x = data.x[data['scale' + str(self.target) + '_cluster_map']]
        data.edge_index = data['scale' + str(self.target - 1) + '_edge_index']
        if hasattr(data, 'h'):
            data.h = data.h[data['scale' + str(self.target) + '_cluster_map']]
        if hasattr(data, 'edge_align'):
            data.pos = self.pos_cache
            data.edge_align = data['scale' + str(self.target - 1) + '_edge_align']

        return data
