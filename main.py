from argparse import ArgumentParser
import torch
from exp.gem_gcn import stead
from exp import baseline, diffusionnet


# Command-line arguments
parser = ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    choices=['gem_gcn', 'baseline', 'diffusionnet'],
    required=True
)
parser.add_argument('--artery_type', type=str, choices=['single', 'bifurcating'], required=True)
parser.add_argument('--num_epochs', type=int, default=0)
parser.add_argument('--gpu', type=int, nargs='+', default=[0])
args = parser.parse_args()

# CUDA device
device = torch.device(f'cuda:{args.gpu[0]}' if torch.cuda.is_available() else 'cpu')

# GEM-GCN
if args.model == 'gem_gcn':
    stead.fit(args.artery_type, args.num_epochs, device, args.gpu if len(args.gpu) > 1 else None)

# IsoGCN or AttGCN (set in "exp/baseline.py")
elif args.model == 'baseline':
    baseline.fit(args.artery_type, args.num_epochs, device, args.gpu if len(args.gpu) > 1 else None)

elif args.model == 'diffusionnet':
    diffusionnet.fit(args.artery_type, args.num_epochs, device, args.gpu if len(args.gpu) > 1 else None)
