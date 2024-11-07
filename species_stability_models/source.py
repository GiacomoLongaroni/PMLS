import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.linalg as la


####################################### FUNCTIONS THAT GENERATE MATRICES FROM A GIVEN MODEL################################
def create_random_matrix(S, C, d, sigma): 

    # matrix with -d in the diagonal and zeros otherwise 
    random_matrix = np.zeros((S, S))
    np.fill_diagonal(random_matrix, -d)
    # random uniform matrix (p)
    rand_values = np.random.uniform(0, 1, (S, S))
    # gaussian matrix (m_ij)
    gaussian_values = np.random.normal(0, sigma, (S, S))

    # condition mask (C > p and out of diagonal)
    mask = (rand_values < C) & (np.eye(S) == 0)

    # filtering the matrix adding the m_ij values
    random_matrix[mask] = gaussian_values[mask]
    return random_matrix

def create_pred_prey_matrix(S, C, d, sigma):

    #initializing predator pray matrix 
    pred_prey_matrix = np.zeros((S, S))
    np.fill_diagonal(pred_prey_matrix, -d)
    
    # getting indices of the upper triangle (no diagonal)
    triu_indices = np.triu_indices(S, k=1)
    # extracting values for each entries of the upper triangle p_1
    rand_values = np.random.uniform(0, 1, len(triu_indices[0]))
    # C mask for p_1
    mask = rand_values < C
    i_indices = triu_indices[0][mask]
    j_indices = triu_indices[1][mask]

    # p_2 values 
    p_sign = np.random.uniform(0, 1, len(i_indices))
    signs = np.where(p_sign > 0.5, 1, -1)

    # half gaussian values drawings
    gaussian_values1 = np.abs(np.random.normal(0, sigma, len(i_indices)))
    gaussian_values2 = np.abs(np.random.normal(0, sigma, len(i_indices)))
    pred_prey_matrix[i_indices, j_indices] = signs * gaussian_values1
    pred_prey_matrix[j_indices, i_indices] = -signs * gaussian_values2
    return pred_prey_matrix

def create_competition_mutualism_matrix(S, C, d, sigma):
    # same logic as above functions
    comp_mut_matrix = np.zeros((S, S))
    np.fill_diagonal(comp_mut_matrix, -d)

    triu_indices = np.triu_indices(S, k=1)
    rand_values = np.random.uniform(0, 1, len(triu_indices[0]))
    mask = rand_values < C
    i_indices = triu_indices[0][mask]
    j_indices = triu_indices[1][mask]
    
    p_sign = np.random.uniform(0, 1, len(i_indices))
    signs = np.where(p_sign > 0.5, 1, -1)
    gaussian_values1 = np.abs(np.random.normal(0, sigma, len(i_indices)))
    gaussian_values2 = np.abs(np.random.normal(0, sigma, len(i_indices)))
    comp_mut_matrix[i_indices, j_indices] = signs * gaussian_values1
    comp_mut_matrix[j_indices, i_indices] = signs * gaussian_values2
    return comp_mut_matrix

def create_mutualism_matrix(S, C, d, sigma):

    mut_matrix = np.zeros((S, S))
    np.fill_diagonal(mut_matrix, -d)
    triu_indices = np.triu_indices(S, k=1)
    rand_values = np.random.uniform(0, 1, len(triu_indices[0]))
    mask = rand_values < C
    i_indices = triu_indices[0][mask]
    j_indices = triu_indices[1][mask]
    gaussian_values1 = np.abs(np.random.normal(0, sigma, len(i_indices)))
    gaussian_values2 = np.abs(np.random.normal(0, sigma, len(i_indices)))
    mut_matrix[i_indices, j_indices] = gaussian_values1
    mut_matrix[j_indices, i_indices] = gaussian_values2
    return mut_matrix
####################################################################################################

 # finding eigenvelues of a given matrix
def find_max_real_eigenvalue(M):
    eigenvalues = la.eigvals(M)
    return eigenvalues.real.max()

# scatterplot of the eigenvalues disk
def plot_eigenvalues(ev, par, ax):
    x = ev.real
    y = ev.imag
    col1 = '#e7c6ff'
    col2 = '#bbd0ff'
    col = np.where(x > 0, col2, col1)
    ax.vlines(0, ymin=-1000, ymax=1000, alpha=0.7, colors='black', linewidth=0.7, linestyles='dashed')
    ax.hlines(0, xmin=-150, xmax=500, alpha=0.7, colors='black', linewidth=0.7, linestyles='dashed')
    ax.scatter(x, y, s=7, color=col, label = 'eigenvalues')

    ax.scatter(-1, 0, label = 'd')
    ax.scatter(-1, 3, label = f'n. Species: {par[0]}', s = 0)
    ax.scatter(-1, 3, label = f'σ: {round(par[1],2)}', s = 0)
    ax.scatter(-1, 3, label = f'Connectance: {round(par[2],2)}', s = 0)

    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_xlim(-5, 3)
    ax.set_ylim(-4, 4)
    ax.set_title('Eigenvalues')
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')

# full simulation function
def create_simulation(simulation_function):
    # simulations parameters
    axes_points = 3 # heatmap grid
    n_sim = 200
    species = 100
    d = 1
    sigmas = np.linspace(0.0001, 0.2, axes_points)
    connectances = np.linspace(0.0001, 1, axes_points)

    # initializing heatmap and scatterplot
    eigen_heatmap = np.zeros((axes_points, axes_points))
    probability_scatter = []

    # extracting one couple of parameters
    sigma_mid = np.random.choice(sigmas)
    connectance_mid = np.random.choice(connectances)
    simulation = simulation_function(species, connectance_mid, d, sigma_mid)
    eig = la.eigvals(simulation)

    # creating plots and filling the first with eigenvalues scatter
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle('Randomic')
    plot_eigenvalues(eig, [species, sigma_mid, connectance_mid], axs[0])

    # full set of simulations (ij parameters heatmap indices (sigma connectance), n simulations)
    for i in range(axes_points):
        conn = connectances[i]

        for j in range(axes_points):
            sigma = sigmas[j]
            max_eigs = []
            positive_eigen_count = 0

            for _ in range(n_sim):
                simulation = simulation_function(species, conn, d, sigma)
                # actual simulation
                max_eig = find_max_real_eigenvalue(simulation)
                max_eigs.append(max_eig)

                # evaluating stability
                if max_eig > 0:
                    positive_eigen_count += 1

            eigen_heatmap[i, j] = np.mean(max_eigs)
            complexity = sigma * np.sqrt(conn * species)
            # probability of stability as positive max over total max eigenvalues 
            probability = 1 - positive_eigen_count / n_sim
            probability_scatter.append([complexity, probability])
            
        
    # plotting heatmap and probability scatterplot 
    im = axs[1].imshow(eigen_heatmap, cmap='Purples_r', origin='lower')
    fig.colorbar(im, ax=axs[1])
    axs[1].set_xticks(np.arange(axes_points))
    axs[1].set_xticklabels(np.round(sigmas, 3), rotation=45,  fontsize = 5)
    axs[1].set_yticks(np.arange(axes_points))
    axs[1].set_yticklabels(np.round(connectances, 2), fontsize = 5)
    axs[1].set_xlabel('σ')
    axs[1].set_ylabel('Connectance')
    axs[1].set_title("Heatmap of Max Eigenvalue")

    probability_scatter = np.array(probability_scatter)
    axs[2].scatter(probability_scatter[:, 0], probability_scatter[:, 1], color = '#c8b6ff')
    axs[2].vlines(d, ymin =0, ymax=1, alpha=0.7, colors='black', linewidth=0.7, linestyles='dashed', label = 'd')
    axs[2].set_xlabel('Complexity')
    axs[2].set_ylabel('Probability of stability')
    axs[2].set_title('Probability vs Complexity')
    axs[2].grid(alpha = 0.2)
    axs[2].legend()
    plt.tight_layout()
    plt.show()