import numpy as np
import h5py
import pandas as pd


def confusion_matrix(variable, target) :
    N = [[0,0], [0,0]]   # Make a list of lists (i.e. confusion matrix) for counting successes/failures.
    for i in np.arange(len(target)):
        if (variable[i] == 0.0 and target[i] == 0.0) : N[0][0] += 1
        if (variable[i] == 0.0 and target[i] == 1.0) : N[0][1] += 1
        if (variable[i] == 1.0 and target[i] == 0.0) : N[1][0] += 1
        if (variable[i] == 1.0 and target[i] == 1.0) : N[1][1] += 1
    fracWrong = float(N[0][1]+N[1][0])/float(len(target))
    accuracy = 1.0 - fracWrong
    return N, accuracy, fracWrong

def load_data(name):
    with h5py.File(f'{name}.h5', 'r') as f:
        filename = name.split('/')[1]
        return pd.DataFrame(f[filename][:], dtype=np.float64)


