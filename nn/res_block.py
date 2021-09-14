import torch
import torch_geometric
from torch.utils.checkpoint import checkpoint


class ResBlock(torch.nn.Module):

    def __init__(self, convolution, in_channels, out_channels, **kwargs):
        super(ResBlock, self).__init__()

        if 'relu' in kwargs:
            if not kwargs.pop('relu'):
                self.act = torch.nn.Identity()
        else:
            self.act = torch.relu

        self.conv0 = convolution(in_channels, out_channels, **kwargs)
        self.bn0 = torch_geometric.nn.BatchNorm(out_channels)
        self.conv1 = convolution(out_channels, out_channels, **kwargs)
        self.bn1 = torch_geometric.nn.BatchNorm(out_channels)

        if in_channels != out_channels:
            self.lin = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.lin = torch.nn.Identity()

    @staticmethod
    def entry_layer(x, edge_index, bn, conv):

        # Dummy wrapper to trick the checkpoint to preserve gradients
        def dummy_conv(signal, connectivity, dummy):
            return conv(signal, connectivity)

        # Re-calculate the convolution on each pass
        check = checkpoint(dummy_conv, x, edge_index,
                           torch.tensor(0., requires_grad=True),
                           preserve_rng_state=False)

        return bn(check)  # without activation

    @staticmethod
    def layer(x, edge_index, bn, conv):
        # Re-calculate the convolution on each pass
        check = checkpoint(conv, x, edge_index, preserve_rng_state=False)

        return bn(check)  # without activation

    def forward(self, x, edge_index):
        y = self.entry_layer(x, edge_index, self.bn0, self.conv0)
        y = torch.relu(y)
        y = self.layer(y, edge_index, self.bn1, self.conv1)

        # Residual connection
        out = y + self.lin(x.unsqueeze(2)).squeeze(2)
        out = self.act(out)

        return out.squeeze()
