import numpy as np


def weight_variable_glorot(output_dim):
    '''
    glorot weight initialization method
    '''

    input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range,
                                (input_dim, output_dim))

    return initial
