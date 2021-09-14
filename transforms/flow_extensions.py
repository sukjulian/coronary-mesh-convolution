from utils.inlet import IndexFinder
import torch
import numpy as np
from utils.remove import remove_vertices


class RemoveFlowExtensions(object):
    """Remove flow extensions based on the geodesic distances to the inlet. Requires geodesics.

    Args:
        factor (float): Multiple of the vessel diameter used as flow extensions.
    """

    def __init__(self, factor=5., delete=False):
        self.factor = factor
        self.delete = delete
        self.inlet = IndexFinder(pytorch=True)

    def __call__(self, data):

        # Compute the vessel diameter
        area = self.inlet.area(data.dir)
        diameter = 2 * np.sqrt(area / np.pi)

        # Vertex mask
        length = self.factor * diameter
        geodesics = data.geo
        vertex_mask = torch.logical_and(torch.gt(geodesics, torch.min(geodesics) + length),
                                        torch.lt(geodesics, torch.max(geodesics) - length))

        if self.delete:

            # Remove vertices from the graph
            data = remove_vertices(data, vertex_mask, dummy_mask=True)

        else:

            # Append the vertex mask
            data.mask = vertex_mask

        return data

    def __repr__(self):
        return '{}(factor={}, delete={})'.format(self.__class__.__name__, self.factor, self.delete)
