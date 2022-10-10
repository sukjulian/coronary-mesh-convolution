import torch_geometric
from models import AttGCN, IsoGCN
from transforms import InletGeodesics, RemoveFlowExtensions, HeatSamplingCluster, MatrixFeatures
from .utils import Experiment


def fit(artery_type, num_epochs, device, gpu):

    # Multiscale radius graph
    ratios = [1., 0.4, 0.1]
    radii = [0.042, 0.06, 0.8] if artery_type == 'single' else [0.022, 0.04, 0.1]

    # Neural network
    model = AttGCN()

    batch_size = 12

    # File name for training weights
    tag = f"baseline_{artery_type}"

    # Precomputed graph transforms
    transforms = [
        torch_geometric.transforms.GenerateMeshNormals(),
        InletGeodesics(),
        RemoveFlowExtensions(factor=(4., 1.)) if artery_type == 'single' else RemoveFlowExtensions(factor=(-1., 1.)),
        HeatSamplingCluster(ratios, radii, loop=True, max_neighbours=512),
        # MatrixFeatures(r=0.042)
    ]

    # Neural network training
    experiment = Experiment(
        model=model,
        dataset=artery_type,
        batch_size=batch_size,
        tag=tag,
        transforms=transforms,
        epochs=num_epochs,
        parallel=gpu
    )
    experiment.run(device)
