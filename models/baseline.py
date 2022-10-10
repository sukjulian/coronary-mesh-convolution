import torch
from utils.model_tools import parameter_table
import nn


# Base class for different convolutions
class BaselineArchitecture(torch.nn.Module):
    def __init__(self):
        super(BaselineArchitecture, self).__init__()

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_table(self):
        return parameter_table(self)

    def forward(self, data):
        # (N, 3, 3, 3) -> [(N, 6), (N, 6), (N, 9)] (two matrices are symmetric)
        # tmp = [data.feat[:, 0][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        #        data.feat[:, 1][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        #        data.feat[:, 2].reshape(-1, 9)]
        # data.x = torch.hstack((torch.hstack(tmp), data.clone().geo.unsqueeze(1)))
        data.x = torch.hstack((data.clone().norm, data.clone().geo.unsqueeze(1)))

        # Encoder
        data.x = self.conv01(data.x, data.scale0_edge_index)
        data.x = self.conv02(data.x, data.scale0_edge_index)

        # Downstream
        copy0 = data.x.clone()
        data = self.pool1(data)
        data.x = self.conv11(data.x, data.edge_index)
        data.x = self.conv12(data.x, data.edge_index)

        copy1 = data.x.clone()
        data = self.pool2(data)
        data.x = self.conv21(data.x, data.edge_index)
        data.x = self.conv22(data.x, data.edge_index)

        # Upstream
        data = self.pool2.unpool(data)
        data.x = torch.cat((data.x, copy1), dim=1)  # "copy/cat"
        data.x = self.conv13(data.x, data.edge_index)
        data.x = self.conv14(data.x, data.edge_index)
        data.x = self.conv15(data.x, data.edge_index)
        data.x = self.conv16(data.x, data.edge_index)

        # Decoder
        data = self.pool1.unpool(data)
        data.x = torch.cat((data.x, copy0), dim=1)  # "copy/cat"
        data.x = self.conv03(data.x, data.edge_index)
        data.x = self.conv04(data.x, data.edge_index)
        data.x = self.conv05(data.x, data.edge_index)
        data.x = self.conv06(data.x, data.edge_index)

        return data.x.squeeze()


# Attention-scaled graph convolutional (residual) network for comparison with GEM-GCN
class AttGCN(BaselineArchitecture):
    def __init__(self):
        super(AttGCN, self).__init__()

        channels = 94
        kwargs = dict(
            heads=4,
            add_self_loops=False
        )

        # Encoder
        self.conv01 = nn.FeaStResBlock(4, channels, **kwargs)
        self.conv02 = nn.FeaStResBlock(channels, channels, **kwargs)

        # Downstream
        self.pool1 = nn.ClusterPooling(1)
        self.conv11 = nn.FeaStResBlock(channels, channels, **kwargs)
        self.conv12 = nn.FeaStResBlock(channels, channels, **kwargs)

        self.pool2 = nn.ClusterPooling(2)
        self.conv21 = nn.FeaStResBlock(channels, channels, **kwargs)
        self.conv22 = nn.FeaStResBlock(channels, channels, **kwargs)

        # Up-stream
        self.conv13 = nn.FeaStResBlock(channels + channels, channels, **kwargs)
        self.conv14 = nn.FeaStResBlock(channels, channels, **kwargs)
        self.conv15 = nn.FeaStResBlock(channels, channels, **kwargs)
        self.conv16 = nn.FeaStResBlock(channels, channels, **kwargs)

        # Decoder
        self.conv03 = nn.FeaStResBlock(channels + channels, channels, **kwargs)
        self.conv04 = nn.FeaStResBlock(channels, channels, **kwargs)
        self.conv05 = nn.FeaStResBlock(channels, channels, **kwargs)
        self.conv06 = nn.FeaStResBlock(channels, 3, batch_norm=False, relu=False, **kwargs)

        print("{} ({} trainable parameters)".format(self.__class__.__name__, self.count_parameters))


# Isotropic graph convolutional (residual) network for comparison with GEM-GCN
class IsoGCN(BaselineArchitecture):
    def __init__(self):
        super(IsoGCN, self).__init__()

        channels = 180
        kwargs = dict(
            root_weight=False
        )

        # Encoder
        self.conv01 = nn.SAGEResBlock(4, channels, **kwargs)
        self.conv02 = nn.SAGEResBlock(channels, channels, **kwargs)

        # Downstream
        self.pool1 = nn.ClusterPooling(1)
        self.conv11 = nn.SAGEResBlock(channels, channels, **kwargs)
        self.conv12 = nn.SAGEResBlock(channels, channels, **kwargs)

        self.pool2 = nn.ClusterPooling(2)
        self.conv21 = nn.SAGEResBlock(channels, channels, **kwargs)
        self.conv22 = nn.SAGEResBlock(channels, channels, **kwargs)

        # Up-stream
        self.conv13 = nn.SAGEResBlock(channels + channels, channels, **kwargs)
        self.conv14 = nn.SAGEResBlock(channels, channels, **kwargs)
        self.conv15 = nn.SAGEResBlock(channels, channels, **kwargs)
        self.conv16 = nn.SAGEResBlock(channels, channels, **kwargs)

        # Decoder
        self.conv03 = nn.SAGEResBlock(channels + channels, channels, **kwargs)
        self.conv04 = nn.SAGEResBlock(channels, channels, **kwargs)
        self.conv05 = nn.SAGEResBlock(channels, channels, **kwargs)
        self.conv06 = nn.SAGEResBlock(channels, 3, batch_norm=False, relu=False, **kwargs)

        print("{} ({} trainable parameters)".format(self.__class__.__name__, self.count_parameters))
