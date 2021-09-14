import torch
import nn
from utils import parameter_table


# Base class for different convolutions
class Compare(torch.nn.Module):
    def __init__(self):
        super(Compare, self).__init__()

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_table(self):
        return parameter_table.create(self)

    def forward(self, data):
        # (N, 3, 3, 3) -> [(N, 6), (N, 6), (N, 9)] (two matrices are symmetric)
        tmp = [data.feat[:, 0][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
               data.feat[:, 1][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
               data.feat[:, 2].reshape(-1, 9)]
        data.x = torch.hstack((torch.hstack(tmp), data.geo.unsqueeze(1)))

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

        return data.x


# FeaSt convolutional residual network for comparison with the GEM-CNN
class CompareFeaSt(Compare):
    def __init__(self):
        super(CompareFeaSt, self).__init__()

        channels = 115
        heads = 2

        # Encoder
        self.conv01 = nn.FeaStResBlock(22, channels, heads=heads)
        self.conv02 = nn.FeaStResBlock(channels, channels, heads=heads)

        # Downstream
        self.pool1 = nn.ClusterPooling(1)
        self.conv11 = nn.FeaStResBlock(channels, channels, heads=heads)
        self.conv12 = nn.FeaStResBlock(channels, channels, heads=heads)

        self.pool2 = nn.ClusterPooling(2)
        self.conv21 = nn.FeaStResBlock(channels, channels, heads=heads)
        self.conv22 = nn.FeaStResBlock(channels, channels, heads=heads)

        # Up-stream
        self.conv13 = nn.FeaStResBlock(channels + channels, channels, heads=heads)
        self.conv14 = nn.FeaStResBlock(channels, channels, heads=heads)
        self.conv15 = nn.FeaStResBlock(channels, channels, heads=heads)
        self.conv16 = nn.FeaStResBlock(channels, channels, heads=heads)

        # Decoder
        self.conv03 = nn.FeaStResBlock(channels + channels, channels, heads=heads)
        self.conv04 = nn.FeaStResBlock(channels, channels, heads=heads)
        self.conv05 = nn.FeaStResBlock(channels, channels, heads=heads)
        self.conv06 = nn.FeaStResBlock(channels, 3, heads=heads, relu=False)

        print(self.parameter_table())


# FeaSt convolutional residual network for comparison with the GEM-CNN
class CompareSAGE(Compare):
    def __init__(self):
        super(CompareSAGE, self).__init__()

        channels = 116

        # Encoder
        self.conv01 = nn.SAGEResBlock(22, channels)
        self.conv02 = nn.SAGEResBlock(channels, channels)

        # Downstream
        self.pool1 = nn.ClusterPooling(1)
        self.conv11 = nn.SAGEResBlock(channels, channels)
        self.conv12 = nn.SAGEResBlock(channels, channels)

        self.pool2 = nn.ClusterPooling(2)
        self.conv21 = nn.SAGEResBlock(channels, channels)
        self.conv22 = nn.SAGEResBlock(channels, channels)

        # Up-stream
        self.conv13 = nn.SAGEResBlock(channels + channels, channels)
        self.conv14 = nn.SAGEResBlock(channels, channels)
        self.conv15 = nn.SAGEResBlock(channels, channels)
        self.conv16 = nn.SAGEResBlock(channels, channels)

        # Decoder
        self.conv03 = nn.SAGEResBlock(channels + channels, channels)
        self.conv04 = nn.SAGEResBlock(channels, channels)
        self.conv05 = nn.SAGEResBlock(channels, channels)
        self.conv06 = nn.SAGEResBlock(channels, 3, relu=False)

        print(self.parameter_table())
