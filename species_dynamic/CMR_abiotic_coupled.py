import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def get_parameters_coupled():
    """
    input for the model parameters
    """
    print('\n------------------------')
    print("Choose model parameters:")
    print('------------------------\n')
    N_0 = int(input("Initial population N(0): "))
    R_0 = float(input("Initial resources R(0): "))
    g = float(input("Growth rate g: "))
    c = float(input("Consumption rate c: "))
    d = float(input("Death rate d: "))
    s = float(input("Supply rate s: "))
    
    return {'N_0': N_0, 'R_0': R_0, 'g': g, 'c': c, 'd': d, 's': s}

def CMR_coupled(y, t, par):
    N, R = y
    dNdt = N * (par['g'] * par['c'] * R - par['d'])
    dRdt = par['s'] - par['c'] * R * N
    return [dNdt, dRdt]

def numerical_solution_coupled(par):
    # initial condition
    N, R = par['N_0'], par['R_0']
    y = [N, R]
    time = np.arange(0, 50, 0.1)

    # numerical solution 
    solutions = odeint(CMR_coupled, y, time, args=(par,))

    return time, solutions

def plot_solution(time, solutions, par):
    """
    Function to plot the solution of the coupled CMR model
    """
    fig, ax = plt.subplots(1, 1)

    ax.set_title('CMR model')
    
    # Plot N(t) and R(t)
    ax.plot(time, solutions[:, 0], label="N(t)", c = '#6a994e')
    ax.plot(time, solutions[:, 1], label="R(t)", c = '#fca311')
    #ax.set_ylim(1, 3.2)
    # Dashed lines for equilibrium points
    ax.axhline(par['g']*par['s']/par['d'], 0, max(time), linewidth=0.5, linestyle='dashed', color='black', label = 'stable')
    ax.axhline(par['d']/(par['c']*par['g']), 0, max(time), linewidth=0.5, linestyle='dashed', color='black')

    ax.grid(alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Population')
    ax.legend()

    # Add the differential equation as LaTeX inside a box
    equation_text = (
        r"$\frac{dN}{dt} = N \left( g \cdot c \cdot R - d \right)$" "\n"
        r"$\frac{dR}{dt} = s - c \cdot N \cdot R$")

    
    # Display chosen parameters
    parameters_text = (
        f"Parameters:\n"
        f"N(0) = {par['N_0']}\n"
        f"R(0) = {par['R_0']}\n"
        f"g = {par['g']}\n"
        f"c = {par['c']}\n"
        f"d = {par['d']}\n"
        f"s = {par['s']}"
    )
    
    # Use text box properties to display the equation and parameters with background and border
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    
    ax.text(0.4, 0.91, equation_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='right', bbox=props)
    

    ax.text(0.57, 0.85, parameters_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='center', horizontalalignment='right', bbox=props)

    # Show the plot
    plt.show()

def main():
    #par = get_parameters_coupled()
    par = {'N_0': 10, 'R_0': 5, 'g': 0.5, 'c': 1, 'd': 0.99, 's': 2}

    # getting the solutions
    t, sol = numerical_solution_coupled(par)

    # Call the plot function
    plot_solution(t, sol, par)

if __name__ == "__main__":
    main()
