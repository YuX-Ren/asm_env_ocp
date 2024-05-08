import numpy as np  
from scipy.linalg import solve  
from acados_template import AcadosModel  
from casadi import SX, vertcat, exp , sum1, DM, inf, types
import casadi as cs

def calculate_motion_mode(temp):
    #decompose the rate_matrix
    eigenvalues, eigenvectors = np.linalg.eig(get_rate_matrix(temp))
    mask = np.abs(eigenvalues) > 1e-5
    order = np.argsort(np.abs(eigenvalues[mask]))

    return eigenvalues[mask][order], eigenvectors[:, mask][:, order]

def get_rate_matrix(temp):
    E = DM([0, 0.4, 1, 0.2])
    B = DM([[inf, 1.5, 1.1, inf],
            [1.5, inf, 10, 0.01],
            [1.1, 10, inf, 1],
            [inf, 0.01, 1, inf]])
    rate_matrix = exp(-(B-E)/temp).T
    
    # eliminate the inf values
    for i in range(4):
        for j in range(4):
            if B[i, j] == inf:
                rate_matrix[i, j] = 0
    for j in range(4):
        rate_matrix[j, j] = - sum1(rate_matrix[:, j])
    return rate_matrix

def get_equilibrium(temp):
    rate_matrix = get_rate_matrix(temp)
    # rate 2 transition
    trans_matrix = rate_matrix/np.max(np.abs(rate_matrix)) + np.eye(4)
    # calculate equilibrium
    eigenvalues, eigenvectors = np.linalg.eig(trans_matrix)
    eigenvectors = np.transpose(eigenvectors)
    targetvector = eigenvectors[np.argmax(eigenvalues)]
    targetvector = targetvector / np.sum(targetvector)
    return targetvector

def export_asm_model() -> AcadosModel:

    model_name = 'asm'
    # set up states & controls
    x1 = SX.sym('x1', 4)
    x = vertcat(x1)

    T = SX.sym('T')

    # xdot
    x1_dot = SX.sym('x1_dot', 4)
    xdot = vertcat(x1_dot)

    # dynamics
    f_expl = vertcat(get_rate_matrix(T)@x1)
    f_impl = xdot - f_expl
    
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = T
    model.name = model_name
    
    # set x0 and x_goal
    x0 = get_equilibrium(1)
    model.x0 = x0
    x_goal = get_equilibrium(2)
    model.x_goal = x_goal
    # motion_mode = calculate_motion_mode(2)
    # b = model.x - x_goal
    # A = motion_mode[1]
    # c = cs.solve(cs.mtimes(A.T, A), cs.mtimes(A.T, b))

    # model.con_h_expr_e = vertcat(c[0], c[1])
    # model.cost_expr_ext_cost_e = c[2]**2

    # store meta information
    model.u_labels = ['$T$']
    model.t_label = '$t$ [s]'

    return model
