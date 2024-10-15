import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def get_parameters():

    """
    input for the model parameters
    """
    print('\n------------------------')
    print("Choose model parameters:")
    print('------------------------\n')

    N_0 = float(input("Initial population N(0): "))
    g = float(input("Growth rate g: "))
    c = float(input("Consumption rate c: "))
    d = float(input("Death rate d: "))
    s = float(input("Supply rate s: "))
    
    return {'N_0': N_0, 'g': g, 'c': c, 'd': d, 's': s}

def CMR_abiotic(N, t, par):

    # DE implementation with quasi-stationary approximation
    R_star = par['s'] / (par['c'] * N)
    dNdt = N * ((par['g'] * par['c'] * R_star) - par['d'])

    return dNdt

def numerical_solution(par):
    
    N_0 = par['N_0']
    time = np.arange(0, 50, 0.1)

    # numerical solution
    N_t = odeint(CMR_abiotic, N_0, time, args=(par,))

    return time, N_t


def CMR_abiotic_Monod(N, t, par):

    # DE implementation with quasi-stationary approximation with Monod function
    R_star = par['s'] / (par['c'] * N)
    mu_max = 7
    k = 7
    mu_R = mu_max * (R_star/(k + R_star))
    dNdt = N * ((par['g'] * par['c'] * mu_R) - par['d'])

    return dNdt

def numerical_solution_Monod(par):
    
    N_0 = par['N_0']
    time = np.arange(0, 50, 0.1)

    # numerical solution
    N_t = odeint(CMR_abiotic_Monod, N_0, time, args=(par,))

    return time, N_t

def analitical_solution(par):

    N_0 = par['N_0']
    time = np.arange(0, 50, 0.1)

    # analytical solution
    N_t = (N_0 - par['g'] * par['s'] / par['d']) * np.exp(-(par['d'] * time)) + (par['g'] * par['s'] / par['d'])

    return time, N_t


def main():
    
    #par = get_parameters_coupled()
    par = {'N_0': 5, 'g': 0.7, 'c': 1.5, 'd': 0.2, 's': 2}

    # getting the solutions
    x_num, y_num = numerical_solution(par)
    x_mon, y_mon = numerical_solution_Monod(par)
    x_th, y_th = analitical_solution(par)

    # plotting
    fig, ax = plt.subplots(1, 1)

    ax.set_title('CMR model with quasi-stationary approximation')
    ax.plot(x_num, y_num, label="Numerical Solution",  color = '#a7c957', linestyle='dashed', linewidth = 1.5)
    ax.plot(x_mon, y_mon, linestyle='dashed', linewidth = 1, color='#bc4749', label="Num. Sol. with Monod")
    ax.plot(x_th, y_th, label="Analytical Solution", color = '#386641', alpha = 0.5, linewidth = 1.5)
    #ax.set_ylim(0, max(max(y_th), max(y_num), max(y_mon))+1)
    ax.grid(alpha = 0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('N(t)')
    ax.legend()

    # Add the differential equation as LaTeX inside a box
    equation_text = r"$\frac{dN}{dt} = N \left( g \cdot c \cdot \frac{s}{cN} - d \right)$"
    
    # Display chosen parameters
    parameters_text = (
        f"Parameters:\n"
        f"N(0) = {par['N_0']}\n"
        f"g = {par['g']}\n"
        f"c = {par['c']}\n"
        f"d = {par['d']}\n"
        f"s = {par['s']}"
    )
    
    # Use text box properties to display the equation and parameters with background and border
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    
    ax.text(0.95, 0.5, equation_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='right', bbox=props)
    
    ax.text(0.95, 0.325, parameters_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='center', horizontalalignment='right', bbox=props)

    plt.show()
    plt.show()


if __name__ == "__main__":
    main()
