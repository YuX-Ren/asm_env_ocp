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
    
    # set x0 and x_goal
    x0 = get_equilibrium(1)
    x_goal = get_equilibrium(2)
    motion_mode = calculate_motion_mode(2)
    
    b = x0 - x_goal
    A = motion_mode[1]
    c = solve(np.dot(A.T, A), np.dot(A.T, b))

    # set up states & controls
    x1 = SX.sym('x1', 4)
    modes = SX.sym('modes', 3)
    x = vertcat(x1, modes)

    T = SX.sym('T')

    # xdot
    x1_dot = SX.sym('x1_dot', 4)
    modes_dot = SX.sym('modes_dot', 3)
    xdot = vertcat(x1_dot, modes_dot)

    # dynamics
    V = get_rate_matrix(T)@x1

    A = motion_mode[1]
    V_mode = cs.solve(cs.mtimes(A.T, A), cs.mtimes(A.T, V))

    f_expl = vertcat(V, V_mode)
    f_impl = xdot - f_expl
    
    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = T
    model.name = model_name
    # cat the x0 and c
    model.x0 = np.concatenate((x0, c.squeeze()))
    model.x_goal = x_goal
    

    # model.con_h_expr_e = vertcat(c[0], c[1])
    # model.con_h_expr_e = vertcat(c[0])
    # model.cost_expr_ext_cost_e = c[0]**2

    # store meta information
    model.u_labels = ['$T$']
    model.t_label = '$t$ [s]'

    return model
