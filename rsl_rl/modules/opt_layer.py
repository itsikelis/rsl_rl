import torch
import torch.nn as nn

from qpth.qp import QPFunction, QPSolvers


class OptLayer(nn.Module):
    def __init__(
        self,
        ns: int,
        nx: int,
        nu: int,
        verbose: int = -1,
        solver=QPSolvers.PDIPM_BATCHED,
        check_Q_spd: bool = True,
    ):
        super().__init__()

        self.ns = ns
        self.nx = nx
        self.nu = nu
        self.qp = QPFunction(verbose, check_Q_spd, solver)

    def forward(self, Q, p, G, h, A, b):
        return self.qp(Q, p, G, h, A, b)
