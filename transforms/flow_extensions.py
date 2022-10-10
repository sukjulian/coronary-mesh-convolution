from utils.artery_tools import IndexFinder, remove_vertices
import torch
import trimesh
import numpy as np


class RemoveFlowExtensions(object):
    """Remove flow extensions based on the geodesic distances to the inlet. Requires geodesics.

    Args:
        factor (tuple): Multiples of the vessel diameter to remove & use as padding, e.g. (4., 1.).
    """

    def __init__(self, factor=(5., 0)):
        self.factor = factor

        self.inlet = IndexFinder(pytorch=True)

    def inlet_area(self, data):
        if hasattr(data, 'inlet_index'):

            # Inlet vertex mask
            vertex_mask = torch.full((data.num_nodes,), False)
            vertex_mask[data['inlet_index'].long()] = True

            # Determine the inlet mesh
            inlet = remove_vertices(data.clone(), vertex_mask)

            # Use trimesh object for area computation
            area = trimesh.Trimesh(vertices=inlet.pos.numpy(), faces=inlet.face.t().numpy()).area

        else:

            # Read inlet mesh from boundary-condition file
            area = self.inlet.area(data.dir)

        return area

    def __call__(self, data):

        # Compute the vessel diameter
        area = self.inlet_area(data)
        diameter = 2 * np.sqrt(area / np.pi)

        # Truncation
        length = [f * diameter for f in self.factor]
        vertex_mask = torch.logical_and(torch.gt(data.geo, torch.min(data.geo) + length[0]),
                                        torch.lt(data.geo, torch.max(data.geo) - length[0]))

        remove_vertices(data, vertex_mask)

        # Padding
        vertex_mask = torch.logical_and(torch.gt(data.geo, torch.min(data.geo) + length[1]),
                                        torch.lt(data.geo, torch.max(data.geo) - length[1]))

        data.mask = vertex_mask

        return data

    def __repr__(self):
        return '{}(factor={})'.format(self.__class__.__name__, self.factor)
