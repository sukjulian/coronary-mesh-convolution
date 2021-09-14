import nn
import torch_geometric


class SAGEResBlock(nn.ResBlock):

    def __init__(self, in_channels, out_channels, **kwargs):
        convolution = torch_geometric.nn.SAGEConv

        super(SAGEResBlock, self).__init__(convolution, in_channels, out_channels, **kwargs)
