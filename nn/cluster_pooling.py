import torch
from torch_scatter import scatter


class ClusterPooling(torch.nn.Module):

    def __init__(self, target, reduction='mean'):
        super(ClusterPooling, self).__init__()

        self.target = target  # target scale
        self.reduction = reduction

    def forward(self, data):

        data.x = scatter(data.x, data['scale' + str(self.target) + '_cluster_map'], dim=0, reduce=self.reduction)
        data.edge_index = data['scale' + str(self.target) + '_edge_index']

        return data

    def unpool(self, data):

        data.x = data.x[data['scale' + str(self.target) + '_cluster_map']]
        data.edge_index = data['scale' + str(self.target - 1) + '_edge_index']

        return data
