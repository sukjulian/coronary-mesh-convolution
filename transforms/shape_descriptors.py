from torch_cluster import radius
import torch
from torch_scatter import scatter


class MatrixFeatures(object):
    """Computes a feature descriptor based on the outer product of positions difference and normals difference between
    neighbouring vertices. Requires normals being available for each vertex.

    Args:
        r (float): Neighbourhood radius from which to compute the differences.
    """

    def __init__(self, r):
        self.r = r

    def __call__(self, data):
        index = radius(data.pos, data.pos, self.r)

        difference = data.pos[index[0]] - data.pos[index[1]]  # pointing outward ("source to target")
        distance = torch.linalg.norm(difference, dim=1)

        weight = (self.r - distance)

        difference = difference / self.r
        normal = data.norm[index[0]]  # surrounding vertices ("source to target")

        descriptor = torch.cat([self.feature(difference, difference, index[1], weight).unsqueeze(1),
                                self.feature(normal, normal, index[1], weight).unsqueeze(1),
                                self.feature(difference, normal, index[1], weight).unsqueeze(1)], dim=1)
        data.feat = descriptor

        return data

    @staticmethod
    def feature(a, b, i, weight):
        matrix = torch.bmm(a.view(-1, 3, 1), b.view(-1, 1, 3))  # batch matrix multiplication
        matrix = weight.view(-1, 1, 1) * matrix

        average = scatter(matrix, i, dim=0, reduce='mean')

        # if torch.all(torch.eq(a, b)):  # symmetric
        #     return average[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]  # flattened upper triangular

        # else:
        #     return average.view(-1, 9)  # flatten

        return average

    def __repr__(self):
        return '{}(r={})'.format(self.__class__.__name__, self.r)
