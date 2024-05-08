from acados_template import AcadosOcp, AcadosOcpSolver
from asm_model import export_asm_model
import numpy as np
from utils import plot_asm

def main():
    Tf = 0.0001
    ocp = AcadosOcp()

    # set model
    model = export_asm_model()
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = 400

    # set number of shooting intervals
    ocp.dims.N = N

    # set prediction horizon
    ocp.solver_options.tf = Tf

    # set cost
    # ocp.cost.cost_type_e = 'EXTERNAL'
    # ocp.model.cost_expr_ext_cost_e = model.cost_expr_ext_cost_e
    # set constraints
    # ocp.constraints.lh_e = np.array([0,0])
    # ocp.constraints.uh_e = np.array([0,0])
    ocp.constraints.lbu = np.array([1])
    ocp.constraints.ubu = np.array([2])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.x0 = model.x0

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

    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")
    lbu = ocp.constraints.lbu
    ubu = ocp.constraints.ubu

    plot_asm(np.linspace(0, Tf, N+1), lbu, ubu, simU, simX, latexify=True, time_label=model.t_label, x_labels=model.x_labels, u_labels=model.u_labels)


if __name__ == '__main__':
    main()