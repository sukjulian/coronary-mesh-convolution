from models import DiffusionNet
import torch_geometric
from transforms import InletGeodesics, RemoveFlowExtensions
from .utils import Experiment


def fit(artery_type, num_epochs, device, gpu):

    # Neural network
    model = DiffusionNet()

    batch_size = 1  # current limit

    # File name for training weights
    tag = f"diffusionnet_{artery_type}"

    # Precomputed graph transforms
    transforms = [
        torch_geometric.transforms.GenerateMeshNormals(),
        InletGeodesics(),
        RemoveFlowExtensions(factor=(4., 1.)) if artery_type == 'single' else RemoveFlowExtensions(factor=(-1., 1.))
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
