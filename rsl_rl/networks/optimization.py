import torch
import torch.nn as nn

from qpth.qp import QPFunction, QPSolvers


class Optimizaton(nn.Module):
    def __init__(
        self,
        ns: int,
        nx: int,
        nu: int,
        dt: float,
        w: float,
        ub: list[int],
        lb: list[int],
        verbose: int = 0,
        # solver=QPSolvers.PDIPM_BATCHED,
        solver=QPSolvers.CVXPY,
        check_Q_spd: bool = True,
    ):
        super().__init__()

        self.ns = ns
        self.nx = nx
        self.nu = nu
        self.dt = dt
        self.w = w

        self.ub = ub
        self.lb = lb

        self.nvars = ns * nx + (ns - 1) * nu
        self.n_eq = (ns - 1) * nx + nx
        self.n_ineq = 0

        # Precompute constraint matrices
        self.A_euler, self.b_euler = self.get_explicit_euler_constraints_()
        # self.A_euler, self.b_euler = self.get_semi_implicit_euler_constraints_()
        self.G, self.h = self.get_ineq_constraints_()

        self.qp = QPFunction(verbose=verbose, check_Q_spd=check_Q_spd, solver=solver)

    def forward(self, Q_diag, R_diag, Q_fin_diag, x_init, z_des):

        # Q_diag = policy_input[:nx]
        # R_diag = policy_input[nx : nx + nu]
        # Q_fin_diag = policy_input[nx + nu :]

        H, g = self.get_cost_matrices_(Q_diag, R_diag, Q_fin_diag, z_des)
        H += 1e-8 * torch.eye(H.shape[0])  # Ensure positive definiteness

        A, b = self.get_eq_constraints_(x_init)

        # Solve the optimisation problem
        sol = self.qp(H, g, self.G, self.h, A, b).flatten()

        return sol

    def get_cost_matrices_(self, Q_diag, R_diag, Q_fin_diag, z_des):
        H = torch.zeros((self.nvars, self.nvars))

        Q = torch.diag(Q_diag)
        R = torch.diag(R_diag)
        Q_fin = torch.diag(Q_fin_diag)

        for i in range(self.ns - 1):
            x_start = i * (self.nx + self.nu)
            x_end = x_start + self.nx
            u_start = i * (self.nx + self.nu) + self.nx
            u_end = u_start + self.nu
            H[x_start:x_end, x_start:x_end] = Q
            H[u_start:u_end, u_start:u_end] = R

        H[-self.nx :, -self.nx :] = Q_fin

        g = -H @ z_des

        return H, g

    def get_eq_constraints_(self, x_init):
        # TODO: A_init can be precomputed and b_init is just x_init
        A_init, b_init = self.get_x_init_constraints_(x_init)
        A = torch.vstack((self.A_euler, A_init))
        b = torch.hstack((self.b_euler, b_init))

        return A, b

    def get_explicit_euler_constraints_(self):
        n_eq = (self.ns - 1) * self.nx  # no of Euler equality constraints

        c0 = self.dt * self.A_d_() + torch.eye(self.nx)
        c1 = self.dt * self.B_d_()
        c2 = -torch.eye(self.nx)
        e_mat = torch.hstack((c0, c1, c2))

        A = torch.zeros(n_eq, self.nvars)
        for i in range(self.ns - 1):
            c0 = self.dt * self.A_d_() + torch.eye(self.nx)
            c1 = self.dt * self.B_d_()
            c2 = -torch.eye(self.nx)
            e_mat = torch.hstack((c0, c1, c2))

            row_start = i * self.nx
            row_end = row_start + self.nx
            col_start = i * (self.nx + self.nu)
            col_end = col_start + 2 * self.nx + self.nu
            A[row_start:row_end, col_start:col_end] = e_mat

        b = torch.zeros(n_eq)

        return A, b

    def get_semi_implicit_euler_constraints_(self):
        n_eq = (self.ns - 1) * self.nx  # no of Euler equality constraints

        print(self.A_d_()[0:2])
        exit()
        c0 = torch.vstack((self.dt * self.A_d_()[0:2] - torch.eye(2), -torch.eye(2)))
        print(c0)
        c1 = self.dt * self.B_d_()
        c2 = -torch.eye(self.nx) + self.dt * self.A_d_()
        e_mat = torch.hstack((c0, c1, c2))

        A = torch.zeros(n_eq, self.nvars)
        for i in range(self.ns - 1):
            c0 = self.dt * self.A_d_() + torch.eye(self.nx)
            c1 = self.dt * self.B_d_()
            c2 = -torch.eye(self.nx)
            e_mat = torch.hstack((c0, c1, c2))

            row_start = i * self.nx
            row_end = row_start + self.nx
            col_start = i * (self.nx + self.nu)
            col_end = col_start + 2 * self.nx + self.nu
            A[row_start:row_end, col_start:col_end] = e_mat

        b = torch.zeros(n_eq)

        return A, b

    def get_x_init_constraints_(self, x_init):
        n_eq = self.nx  # no of x_init equality constraints

        A = torch.zeros(n_eq, self.nvars)
        A[0:n_eq, 0:n_eq] = torch.eye(self.nx)

        b = x_init.reshape(-1)

        return A, b

    def get_ineq_constraints_(self):
        I = torch.eye(self.nvars)

        G = torch.vstack((I, -I))
        h = torch.hstack((torch.tensor(self.ub), -torch.tensor(self.lb)))

        return G, h

    def A_d_(self):
        return torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [self.w**2, 0.0, 0.0, 0.0],
                [0.0, self.w**2, 0.0, 0.0],
            ],
            dtype=torch.double,
        )

    def B_d_(self):
        return torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [-(self.w**2), 0.0],
                [0.0, -(self.w**2)],
            ],
            dtype=torch.double,
        )
