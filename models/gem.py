import torch
import torch_geometric.transforms as T
from utils import parameter_table
from gem_cnn.transform.scale_mask import ScaleMask
from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.nn.gem_res_net_block import GemResNetBlock
from gem_cnn.nn.pool import ParallelTransportPool
from gem_cnn.utils.matrix_features import so2_feature_to_ambient_vector


# Gauge-equivariant mesh convolutional neural network
class GEM(torch.nn.Module):
    def __init__(self, radii, max_order=2, n_rings=2):
        super(GEM, self).__init__()

        channels = 24
        kwargs = dict(
            n_rings=n_rings,
            band_limit=2,
            num_samples=7,
            checkpoint=True,
            batch=100000,
        )

        # Computed each forward pass
        self.scale_transforms = [
            T.Compose((ScaleMask(i), GemPrecomp(n_rings, max_order, max_r=r)))
            for i, r in enumerate(radii)
        ]

        # Encoder
        self.conv01 = GemResNetBlock(8, channels, 2, max_order, **kwargs)
        self.conv02 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)

        # Downstream
        self.pool1 = ParallelTransportPool(1, unpool=False)
        self.conv11 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)
        self.conv12 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)

        self.pool2 = ParallelTransportPool(2, unpool=False)
        self.conv21 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)
        self.conv22 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)

        # Up-stream
        self.unpool2 = ParallelTransportPool(2, unpool=True)
        self.conv13 = GemResNetBlock(channels + channels, channels, max_order, max_order, **kwargs)
        self.conv14 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)
        self.conv15 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)
        self.conv16 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)

        # Decoder
        self.unpool1 = ParallelTransportPool(1, unpool=True)
        self.conv03 = GemResNetBlock(channels + channels, channels, max_order, max_order, **kwargs)
        self.conv04 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)
        self.conv05 = GemResNetBlock(channels, channels, max_order, max_order, **kwargs)
        self.conv06 = GemResNetBlock(channels, 1, max_order, 1, last_layer=True, **kwargs)

        print(self.parameter_table())

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_table(self):
        return parameter_table.create(self)

    def forward(self, data):
        # Select the scale graphs
        scale_data = [s(data) for s in self.scale_transforms]
        scale_attr = [(d.edge_index, d.precomp, d.connection) for d in scale_data]

        # Geodesics-to-inlet feature in SO(2) representation
        features = data.matrix_features
        N, _, irreps = features.shape  # e.g. [N, 7, 5]
        geodesics = torch.zeros((N, 1, irreps), device=features.device)
        geodesics[:, 0, 0] = data.geo
        x = torch.cat((features, geodesics), dim=1)

        # Encoder
        x = self.conv01(x, *scale_attr[0])
        x = self.conv02(x, *scale_attr[0])

        # Downstream
        copy0 = x.clone()
        x = self.pool1(x, data)
        x = self.conv11(x, *scale_attr[1])
        x = self.conv12(x, *scale_attr[1])

        copy1 = x.clone()
        x = self.pool2(x, data)
        x = self.conv21(x, *scale_attr[2])
        x = self.conv22(x, *scale_attr[2])

        # Upstream
        x = self.unpool2(x, data)
        x = torch.cat((x, copy1), dim=1)  # "copy/cat"
        x = self.conv13(x, *scale_attr[1])
        x = self.conv14(x, *scale_attr[1])
        x = self.conv15(x, *scale_attr[1])
        x = self.conv16(x, *scale_attr[1])

        # Decoder
        x = self.unpool1(x, data)
        x = torch.cat((x, copy0), dim=1)  # "copy/cat"
        x = self.conv03(x, *scale_attr[0])
        x = self.conv04(x, *scale_attr[0])
        x = self.conv05(x, *scale_attr[0])
        x = self.conv06(x, *scale_attr[0])

        # Construct ambient vectors from tangential SO(2) features
        x = so2_feature_to_ambient_vector(x, data.frame).squeeze()

        return x
