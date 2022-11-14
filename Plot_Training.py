import matplotlib.pyplot as plt
import numpy as np

def get_num_result(filename):
    with open(filename, 'r') as f:
        result = f.read().splitlines()

    result = np.asarray(result, dtype = np.int64)
    return result

def plot_training_result(filename):
    result = get_num_result(filename)
    fig, ax = plt.subplots(figsize = (9,7))
    ax.plot(range(1,len(result)+1), result, color = 'black')
    ax.set(title = 'Training Result')
    plt.show()

def print_stat(filename):
    result = get_num_result(filename)
    print(f'Average = {np.mean(result)}, max score = {max(result)}')

if __name__ == "__main__":
    plot_training_result("scores_linear.txt")
