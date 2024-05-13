from acados_template import AcadosOcp, AcadosOcpSolver
from asm_model import export_asm_model
import numpy as np
from utils import plot_asm

def main():
    dim = 4
    Tf = 0.2
    ocp = AcadosOcp()

    # set model
    model = export_asm_model()
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = 400
    print(nx, nu)
    # set number of shooting intervals
    ocp.dims.N = N

    # set prediction horizon
    ocp.solver_options.tf = Tf

    # set cost
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost_e = model.x[-1]**2
    # set constraints
    # ocp.constraints.lh_e = np.array([-0.5,-0.5])
    # ocp.constraints.uh_e = np.array([0.5,0.5])
    # ocp.constraints.lh_e = np.array([0.0])
    # ocp.constraints.uh_e = np.array([0.0])
    ocp.constraints.lbu = np.array([0.05])
    ocp.constraints.ubu = np.array([0.2])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.x0 = model.x0
    ocp.constraints.lbx_e = np.array([-0.000000 ,-0.00000 ])
    ocp.constraints.ubx_e = np.array([0.000000 ,0.00000 ])
    ocp.constraints.idxbx_e = np.array([dim, dim+1])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')
    for i, tau in enumerate(np.linspace(0, 1, N)):
            ocp_solver.set(i, 'u', 0.1)
    simX = np.zeros((N+1, 4))
    simA = np.zeros((N+1, 3))
    simU = np.zeros((N, nu))
    ocp.solver_options.nlp_solver_max_iter = 1000
    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')
    print(ocp_solver.get_cost())
    # get solution
    for i in range(N):
        simX[i,:] = np.log(np.abs(ocp_solver.get(i, "x")[:4]))
        # print(ocp_solver.get(i, "x"))
        simA[i,:] = np.log(np.abs(ocp_solver.get(i, "x")[4:7]))
        simU[i,:] = ocp_solver.get(i, "u")
    # print(simA[:,0])
    simX[N,:] = np.log(np.abs(ocp_solver.get(N, "x")[:4]))
    simA[N,:] = np.log(np.abs(ocp_solver.get(N, "x")[4:7]))
    print(simA[N-1,0])
    print(simA[N,0])
    print(simU)
    lbu = ocp.constraints.lbu
    ubu = ocp.constraints.ubu

    plot_asm(np.linspace(0, Tf, N+1), lbu, ubu, simU, simX, simA, latexify=True, time_label=model.t_label, x_labels=model.x_labels, u_labels=model.u_labels)


if __name__ == '__main__':
    main()