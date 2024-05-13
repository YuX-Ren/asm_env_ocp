import numpy as np

def get_rate_matrix(temp):
    dim = 4
    E = np.array([0, 0.4, 1, 0.2])
    B = np.array([[np.inf, 1.5, 1.1, np.inf],
                  [1.5, np.inf, 10, 0.01],
                  [1.1, 10, np.inf, 1],
                  [np.inf, 0.01, 1, np.inf]])

    rate_matrix = np.exp(-(B - E)/temp)

    # Eliminate the inf values
    rate_matrix[np.isinf(B)] = 0

    # Set diagonal elements to ensure rows sum to zero
    for j in range(dim):
        rate_matrix[j, j] = -np.sum(rate_matrix[:, j])

    return rate_matrix

def run(temp, time, state):
    rate_matrix = get_rate_matrix(temp)
    dstate = rate_matrix@state * time
    return dstate+state

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



if __name__ == '__main__':
    init_state = get_equilibrium(1)
    cur_state = init_state
    print(init_state)
    target_state = get_equilibrium(2)
    print(target_state)
    action_f = open("action.txt", "r")
    action = action_f.read()
    action_f.close()
    action = action.split('\n')
    action = [float(a) for a in action]
    distances = []
    for act in action:
        cur_state = run(act, 0.2/400, cur_state)
        # KL divergence
        print(act)
        for i in range(5):
            distances.append(np.log(np.sum(cur_state*np.log(cur_state/target_state))))
    for i in np.linspace(0.2, 3, 28000):
        cur_state = run(2, 0.0001, cur_state)
        # KL divergence
        distances.append(np.log(np.sum(cur_state*np.log(cur_state/target_state))))
    distances_t = []
    cur_state = init_state
    for i in np.linspace(0, 3, 30000):
        cur_state = run(2, 0.0001, cur_state)
        # KL divergence
        distances_t.append(np.log(np.sum(cur_state*np.log(cur_state/target_state))))
    t = np.linspace(0, 3, 30000)
    import matplotlib.pyplot as plt
    plt.plot(t, distances_t)
    plt.plot(t, distances)
    plt.savefig("distance_plot.png")