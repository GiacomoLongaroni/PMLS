import numpy as np
import matplotlib.pyplot as plt

def simulate_sde(x0, tau, k, sigma, t_max, dt):

    # temporal time steps
    n_steps = int(t_max / dt)
    
    t = np.linspace(0, t_max, n_steps)
    x = np.zeros(n_steps)
    x[0] = x0  # Valore iniziale
    
    # Euler integration 
    for i in range(1, n_steps):
        # extracting gaussian noise
        xi_t = np.random.normal(0, 1)

        # stochastic evolution dx from x(t-1)
        dx = (x[i-1] / tau) * (1 - x[i-1] / k) * dt + np.sqrt(sigma / tau) * np.sqrt(dt) * xi_t

        # computing x(t)
        x[i] = x[i-1] + dx
    
    return t, x


x0 = 1.0   
tau = 1.0  
k = 10.0   
sigma = 0.2
t_max = 15
dt = 0.01  
n_simulations = 10 

# plotting
fig, ax = plt.subplots(1, 1, figsize = (7,4))
colors = plt.cm.Greens(np.linspace(0, 1, n_simulations))

X = []
for j in range(n_simulations):
    t, x = simulate_sde(x0, tau, k, sigma, t_max, dt)
    X.append(x)
    ax.plot(t, x, color=colors[j], alpha=0.7, linewidth=1)

# deterministic logistic
logistic = x0 * np.exp(1 / tau * t) / (1 + (x0 / k) * (np.exp(1 / tau * t) - 1))
ax.plot(t, logistic, color='black', linestyle='dashed')
ax.set_title('Stochastic logistic evolution')
ax.set_xlabel('t')
ax.set_ylabel('x(t)')
ax.grid(alpha=0.4)

#ax[1].hist(np.array(X).reshape(-1), 50)


#plt.tight_layout()  # Migliora la disposizione delle sottotrame

plt.show()

