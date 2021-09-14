import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


class RandomRotation(object):
    """Randomly rotate vertex positions and labels.

    Args:
        -
    """

    def __call__(self, data):
        angles = np.random.uniform(0., 360., size=3)

        rotation = R.from_euler('zyx', angles).as_matrix()
        rotation = torch.tensor(rotation, dtype=torch.float32)

        data.pos = torch.mm(data.pos, rotation.t())  # matrix multiplication

        if data.y.shape[1] == 3:
            data.y = torch.mm(data.y, rotation.t())

        if 'feat' in data:
            data.feat = torch.matmul(rotation.view(1, 1, 3, 3), data.feat)
            data.feat = torch.matmul(data.feat, rotation.t().view(1, 1, 3, 3))

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
