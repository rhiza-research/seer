import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_Bmatrix(capacities):
    """use capacities as admittances to get B matrix
    higher capacity -> higher line flow -> higher admittance.
    """
    nzones = capacities.shape[0]
    B = np.zeros([nzones, nzones])
    # off-diagonal elements
    for i in range(nzones):
        for j in range(nzones):
            if i != j:
                B[i, j] = -1 * capacities[i, j]
    # diagonal elements
    for i in range(nzones):
        B[i, i] = -sum(B[i, :])
    return B

def get_Blinematrix(capacities):
    nzones = capacities.shape[0]
    Bline = np.zeros([nzones*nzones, nzones])
    idx = 0
    for i in range(nzones):
        for j in range(nzones):
            capij = capacities[i, j]
            Bline[idx, i] = capij
            Bline[idx, j] = -capij
            idx += 1
    return Bline

def get_ptdf(capacities):
    B = get_Bmatrix(capacities)
    Bline = get_Blinematrix(capacities)
    # drop first row and column of B
    B = B[1:, 1:]
    # drop first column of Bline
    Bline = Bline[:, 1:]
    Binv = np.linalg.inv(B)
    ptdf = Bline @ Binv
    return ptdf

if __name__ == "__main__":
    # check the ptdf
    nzones = 3
    capacities = np.zeros([nzones, nzones])
    capacities[0, 1] = 10
    capacities[1, 2] = 20
    capacities[2, 0] = 30

    ptdf = get_ptdf(capacities)

    # visualize change in line flows caused by change in injections
    src_node = 0
    dst_node = 1
    dp = np.zeros([nzones])
    dp[src_node] = 1
    dp[dst_node] = -1
    # drop first row of dp
    dp = dp[1:]
    dflows = ptdf*dp
    print(dflows)

    plt.show()


    breakpoint()