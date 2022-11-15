import os
import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..')) # ugly trick to import from parent
from DE_library import simulate_ODE


def generate_simple_oscillator():
    initial_time, final_time, time_step_size = 0, 400, 0.05

    np.random.seed = 42  # set seed for reproducibility

    time_steps = np.arange(initial_time, final_time, time_step_size)
    number_of_osilators = 1
    initial_condition = np.zeros(number_of_osilators)

    rng = np.random.RandomState(42)
    W = rng.rand(number_of_osilators)
    K = rng.rand(number_of_osilators, number_of_osilators)

    par = {'W': W, 'K': 1*K}
    phi, _ = simulate_ODE('kuramoto', time_steps, initial_condition, par = par)
    return phi

if __name__ == '__main__':
    results = generate_simple_oscillator()
    np.savetxt('simple_single_oscillator.txt', results)
