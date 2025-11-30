import torch
import torch.nn as nn

from qpth.qp import QPFunction, QPSolvers


class ConstrainedLqr(nn.Module):
    def __init__(
        self,
        ns: int,
        nx: int,
        nu: int,
        eps: float = 1e-8,
        verbose: int = 0,
        solver=QPSolvers.PDIPM_BATCHED,
        # solver=QPSolvers.CVXPY,
        check_Q_spd: bool = True,
    ):
        super().__init__()

        self.ns = ns
        self.nx = nx
        self.nu = nu
        self.nvars = self.ns * self.nx + (self.ns - 1) * self.nu

        self.eps = eps  # Regularization to ensure SPD

        self.qp = QPFunction(
            verbose=verbose,
            check_Q_spd=check_Q_spd,
            solver=solver,
            maxIter=20,
        )

    def forward(self, w, A, b, G, h):
        w += self.eps  # Ensure SPD

        nbatch = w.shape[0]

        q_diag = torch.cat(
            (
                torch.zeros((w.shape[0], int(self.nx / 2))) + self.eps,
                w[:, 0 : int(self.nx / 2)],
            ),
            dim=1,
        )
        r_diag = w[:, int(self.nx / 2) : int(self.nx / 2) + self.nu]
        q_fin_diag = 1e4 + w[:, int(self.nx / 2) + self.nu :]

        ## Generate Q matrix
        Q = torch.zeros((nbatch, self.nvars, self.nvars))

        for i in range(self.ns - 1):
            b_idx = i * (self.nx + self.nu)  # Q/R block start index
            for k in range(self.nx):
                diag_idx = b_idx + k
                Q[:, diag_idx, diag_idx] = q_diag[:, k]
            for k in range(self.nu):
                diag_idx = b_idx + self.nx + k
                Q[:, diag_idx, diag_idx] = r_diag[:, k]

        for k in range(self.nx):
            diag_idx = (self.ns - 1) * (self.nx + self.nu) + k
            Q[:, diag_idx, diag_idx] = q_fin_diag[:, k]

        ## Generate p vector
        p = torch.zeros((Q.shape[0], Q.shape[1]))

        ## Solve the LQR problem
        self.sol = self.qp(Q, p, G, h, A, b)

        # Return optimal action
        return self.sol[:, self.nx : self.nx + self.nu]
