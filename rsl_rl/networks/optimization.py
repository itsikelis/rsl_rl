import torch
import torch.nn as nn

from qpth.qp import QPFunction, QPSolvers


class Optimization(nn.Module):
    def __init__(
        self,
        n_batch: int,
        ns: int,
        nx: int,
        nu: int,
        dt: float,
        w: float,
        ub: list[int],
        lb: list[int],
        verbose: int = 0,
        solver=QPSolvers.PDIPM_BATCHED,
        # solver=QPSolvers.CVXPY,
        check_Q_spd: bool = True,
    ):
        super().__init__()

        self.n_batch = n_batch
        self.ns = ns
        self.nx = nx
        self.nu = nu
        self.dt = dt
        self.w = w

        self.ub = []
        self.lb = []
        for _ in range(self.ns - 1):
            self.ub += ub
            self.lb += lb
        self.ub += ub[:nx]
        self.lb += lb[:nx]

        self.nvars = ns * nx + (ns - 1) * nu
        self.n_eq = (ns - 1) * nx + nx
        self.n_ineq = 0

        # Precompute constraint matrices
        self.A_euler, self.b_euler = self.get_explicit_euler_constraints_()
        # self.A_euler, self.b_euler = self.get_semi_implicit_euler_constraints_()
        self.G, self.h = self.get_ineq_constraints_()

        self.qp = QPFunction(verbose=verbose, check_Q_spd=check_Q_spd, solver=solver)

    def forward(self, policy_input, x_init, z_des):

        Q_diag = policy_input[:, 0 : self.nx]
        R_diag = policy_input[:, self.nx : self.nx + self.nu]
        Q_fin_diag = policy_input[:, self.nx + self.nu :]

        H, g = self.get_cost_matrices_(Q_diag, R_diag, Q_fin_diag, z_des)
        H += 1e-8  # Ensure positive definiteness

        A, b = self.get_eq_constraints_(x_init)

        # Solve the optimisation problem
        self.sol = self.qp(
            H,
            g,
            self.G,
            self.h,
            A,
            b,
        ).flatten()

        return self.sol[self.nx : self.nx + self.nu]

    def get_cost_matrices_(self, Q_diag, R_diag, Q_fin_diag, z_des):
        H = torch.zeros((self.n_batch, self.nvars, self.nvars))

        Q = torch.diag_embed(Q_diag)
        R = torch.diag_embed(R_diag)
        Q_fin = torch.diag_embed(Q_fin_diag)

        for i in range(self.ns - 1):
            x_start = i * (self.nx + self.nu)
            x_end = x_start + self.nx
            u_start = i * (self.nx + self.nu) + self.nx
            u_end = u_start + self.nu
            H[:, x_start:x_end, x_start:x_end] = Q
            H[:, u_start:u_end, u_start:u_end] = R

        H[:, -self.nx :, -self.nx :] = Q_fin

        # g = -H @ z_des
        # g = torch.matmul(-H, z_des.reshape((n_envs, -1, 1)))
        g = torch.bmm(-H, z_des.unsqueeze(2))

        return H, g.squeeze()

    def get_eq_constraints_(self, x_init):
        # TODO: A_init can be precomputed and b_init is just x_init
        A_init, b_init = self.get_x_init_constraints_(x_init)
        A = torch.cat((self.A_euler, A_init), dim=1)
        print(self.b_euler.shape, b_init.shape)
        b = torch.cat((self.b_euler, b_init), dim=1)

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

        return A.repeat(self.n_batch, 1, 1), b.repeat(self.n_batch, 1)

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
        A = torch.zeros(self.n_batch, n_eq, self.nvars)
        A[:, 0:n_eq, 0:n_eq] = torch.eye(n_eq)

        b = x_init

        return A, b

    def get_ineq_constraints_(self):
        I = torch.eye(self.nvars)

        G = torch.vstack((I, -I))
        h = torch.hstack((torch.tensor(self.ub), -torch.tensor(self.lb)))

        return G.repeat(self.n_batch, 1, 1), h.repeat(self.n_batch, 1)

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
