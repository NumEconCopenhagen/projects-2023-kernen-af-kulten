import numpy as np
import matplotlib.pyplot as plt
import math
def print_table(alpha_vec,sigma_vec,hm):
    # a. empty text
    text = ''
    
    # b. top header (the sigma-values)
    text += f'{"":1s}{"":3s}{sigma_vec[0]:6.1f}{"":3s}{sigma_vec[1]:1.1f}{"":3s}{sigma_vec[2]:1.1f}\n'

    
    # c. body
    # Creates a loop over the values in the two vectors, where it calculates the HF/HM ratio for each of the 9 combinations of the values in the vectors
    for i,alpha in enumerate(alpha_vec):
        if i > 0:
            text += '\n' # line shift
        text += f'{alpha:1.2f} ' # left header (alpha values)
        for j, sigma in enumerate(sigma_vec):
            hm.par.alpha=alpha
            hm.par.sigma=sigma
            text += f'{hm.solve_discrete().ratio :6.3f}'
    
    # d. prints the table
    print(text) 
def get_data(alpha_vec, sigma_vec,hm):
    data = []
    for i, alpha in enumerate(alpha_vec):
        row = [alpha]
        for j, sigma in enumerate(sigma_vec):
            hm.par.alpha = alpha
            hm.par.sigma = sigma
            ratio = hm.solve_discrete().ratio
            row.append(ratio)
        data.append(tuple(row))
    return data


#Plots the 3D graph in Question 1
def plotQ1Figure_2(alpha_grid, sigma_grid, ratio_grid, alpha_vec, sigma_vec):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(alpha_grid, sigma_grid, ratio_grid[:, 1:])
    ax.set_xlabel('alpha')
    ax.set_ylabel('sigma')
    ax.set_xticks = alpha_vec
    ax.set_yticks = sigma_vec
    ax.set_zlabel('ratio')
    ax.set_title('Household Specialization Model')
    plt.show()

#Plots the 2D graph in Question 1
def plotQ1figure_1(data, alpha_vec, sigma_vec):
    plt.figure()
    for i, sigma in enumerate(sigma_vec):
        x = [row[0] for row in data]
        y = [row[i+1] for row in data]
        plt.plot(x, y, label=f'sigma = {sigma}')

    plt.xlabel('alpha')
    plt.xticks(alpha_vec)
    plt.ylabel('ratio')
    plt.title('Household Specialization Model')
    plt.legend()
    plt.show()

def plotQ2Figure_1(wf_vector, hm):
    #Creates two empty vectors to be used for the graph
    x_data = []
    y_data = []

    # Calculates the log(ratio) and log wage ratio for each of the wF values and saves the results in the x and y vectors
    for j, wage in enumerate(wf_vector):
        hm.par.wF = wage
        logwratio = math.log(hm.par.wF/hm.par.wM)
        logratio = math.log(hm.solve_discrete().ratio)
        x_data.append(logwratio)
        y_data.append(logratio)

    # creates plot
    plt.plot(x_data, y_data)
    plt.xlabel('log(wF/wM)')
    plt.ylabel('log(ratio)')

    # add wF values as labels for each of the datapoints
    for i in range(len(wf_vector)):
        plt.text(x_data[i], y_data[i], f"wF={wf_vector[i]}")

    # Shows the graph
    plt.show()

def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
        
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y

