"""Pinn-Torch: malha, redes, perdas e o laco de treino.

O `Trainer` e as perdas ficavam de fora deste arquivo, entao
`from fisiocomPinn import Trainer` falhava e quem chegava pelo pacote nao
via que havia laco de treino nenhum -- escrevia o proprio. Foi o que
aconteceu em inverse_ecg/problems/crux_3d, que rodou em um nucleo de CPU
por nao ter `device`, e teve o peso de um termo varrido a mao quando
`AdaptiveLossWeights` existe.
"""

from .Grid import Grid, Grid3D, structured_mesh, annular_mesh
from .GridLoss import GridLoss
from .Loss import LOSS
from .Loss_PINN import LOSS_INITIAL, LOSS_PINN
from .Net import FullyConnectedNetwork, FullyConnectedNetworkMod
from .Trainer import AdaptiveLossWeights, Trainer
from .Utils import grad
from .Validator import Validator

__all__ = [
    "AdaptiveLossWeights",
    "FullyConnectedNetwork",
    "FullyConnectedNetworkMod",
    "Grid",
    "Grid3D",
    "GridLoss",
    "LOSS",
    "LOSS_INITIAL",
    "LOSS_PINN",
    "Trainer",
    "Validator",
    "annular_mesh",
    "grad",
    "structured_mesh",
]
