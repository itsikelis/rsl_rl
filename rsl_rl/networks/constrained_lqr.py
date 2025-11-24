import torch
import torch.nn as nn

from qpth.qp import QPFunction, QPSolvers


class ConstrainedLqr(nn.Module):
    def __init__(
        self,
        ns: int,
        nx: int,
        nu: int,
        eps: float = 1e-6,
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

        self.qp = QPFunction(verbose=verbose, check_Q_spd=check_Q_spd, solver=solver)

    def forward(self, w, A, b, G, h):
        w += self.eps  # Ensure SPD

        nbatch = w.shape[0]

        q_diag = w[:, 0 : self.nx]
        r_diag = w[:, self.nx : self.nx + self.nu]
        q_fin_diag = w[:, self.nx + self.nu :]

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

        ## If there are no inequality constraints, create a dummy one
        G = torch.zeros((w.shape[0], 1, self.nvars))
        G[:, 0, 0] = 1.0
        h = torch.zeros(w.shape[0], 1)
        h[:, 0] = 1e4

        ## Solve the LQR problem
        self.sol = self.qp(Q, p, G, h, A, b)

        # Return optimal action
        return self.sol[:, self.nx : self.nx + self.nu]
