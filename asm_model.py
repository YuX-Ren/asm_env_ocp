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
    dim = 4
    # E = DM([0.37454012 ,0.95071431 ,0.73199394 ,0.59865848 ,0.15601864 ,0.15599452 ,0.05808361 ,0.86617615 ,0.60111501 ,0.70807258])
    # B = DM([[inf        ,0.77280075 ,1.44247621 ,1.12221161 ,0.40483859 ,0.39146313  ,0.23224633 ,1.04823313 ,0.9868735  ,1.03721561], 
    #         [0.77280075 ,inf        ,inf        ,1.8607322  ,1.39955132 ,1.30436207  ,inf        ,1.86006602 ,inf        ,1.56396918], 
    #         [1.44247621 ,inf        ,inf        ,inf        ,0.99318312 ,0.85598009  ,1.44138613 ,1.65368324 ,1.42732618 ,inf       ], 
    #         [1.12221161 ,1.8607322  ,inf        ,inf        ,0.53273063 ,inf ,0.42943193 ,1.50252464 ,inf        ,1.23776833], 
    #         [0.40483859 ,1.39955132 ,0.99318312 ,0.53273063 ,inf        ,inf ,0.20440654 ,inf        ,inf        ,inf       ], 
    #         [0.39146313 ,1.30436207 ,0.85598009 ,inf        ,inf        ,inf ,inf        ,inf        ,inf        ,inf       ], 
    #         [0.23224633 ,inf        ,1.44138613 ,0.42943193 ,0.20440654 ,inf ,inf        ,0.17174544 ,0.57600782 ,0.96255299], 
    #         [1.04823313 ,1.86006602 ,1.65368324 ,1.50252464 ,inf        ,inf ,0.17174544 ,inf        ,1.64993622 ,1.72136577], 
    #         [0.9868735  ,inf        ,1.42732618 ,inf        ,inf        ,inf ,0.57600782 ,1.64993622 ,inf        ,inf       ], 
    #         [1.03721561 ,1.56396918 ,inf        ,1.23776833 ,inf        ,inf ,0.96255299 ,1.72136577 ,inf        ,inf       ]])
    E = DM([0, 0.4, 1, 0.2])
    B = DM([[inf, 1.5, 1.1, inf],
            [1.5, inf, 10, 0.01],
            [1.1, 10, inf, 1],
            [inf, 0.01, 1, inf]])

    rate_matrix = exp(-(B-E)/temp).T
    
    # eliminate the inf values
    for i in range(dim):
        for j in range(dim):
            if B[i, j] == inf:
                rate_matrix[i, j] = 0
    for j in range(dim):
        rate_matrix[j, j] = - sum1(rate_matrix[:, j])
    return rate_matrix

def get_equilibrium(temp):
    dim = 4
    rate_matrix = get_rate_matrix(temp)
    # rate 2 transition
    trans_matrix = rate_matrix/np.max(np.abs(rate_matrix)) + np.eye(dim)
    # calculate equilibrium
    eigenvalues, eigenvectors = np.linalg.eig(trans_matrix)
    eigenvectors = np.transpose(eigenvectors)
    targetvector = eigenvectors[np.argmax(eigenvalues)]
    targetvector = targetvector / np.sum(targetvector)
    return targetvector

def export_asm_model() -> AcadosModel:
    dim = 4
    model_name = 'asm'
    
    # set x0 and x_goal
    x0 = get_equilibrium(1)
    x_goal = get_equilibrium(2)
    motion_mode = calculate_motion_mode(2)
    
    b = x0 - x_goal
    A = motion_mode[1]
    c = solve(np.dot(A.T, A), np.dot(A.T, b))

    # set up states & controls
    x1 = SX.sym('x1', dim)
    modes = SX.sym('modes', dim-1)
    x = vertcat(x1, modes)

    T = SX.sym('T')

    # xdot
    x1_dot = SX.sym('x1_dot', dim)
    modes_dot = SX.sym('modes_dot', dim-1)
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
