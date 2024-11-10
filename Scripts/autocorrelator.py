import numpy as np
import matplotlib.pyplot as plt
import sdeint

# Define constants
m = 1.0
omega2 = 5.0  
gamma = 1.0
c = 1.0
EFeld2 = 0.0  # Zero electric field in the example

# Define the system of ODEs 
def deterministic_system(Y, t):
    qx, qy, qz, px, py, pz = Y       # q = position componenet, p = velocity/momentum
    dqx_dt = px / m
    dqy_dt = py / m
    dqz_dt = pz / m
    dpx_dt = -gamma * px + omega2 * py
    dpy_dt = -gamma * py - omega2 * px + EFeld2
    dpz_dt = -gamma * pz
    
    return np.array([dqx_dt, dqy_dt, dqz_dt, dpx_dt, dpy_dt, dpz_dt])

# Define noise terms (Wiener processes)
def noise_system(Y, t):
    noise_matrix = np.zeros((6, 6))
    noise_matrix[3, 3] = c       
    return noise_matrix

# Initial conditions
Y0 = np.array([0.0, 0.0, 0.0, 0.2, 0.2, 0.2])  # Initial positions (qx, qy, qz) and momenta (px, py, pz)

# Time array for the simulation 
t = np.linspace(0, 200, 20001)  

result = sdeint.itoint(deterministic_system, noise_system, Y0, t)

# Extract the results for px and py
px = result[:, 3]
py = result[:, 4]

# Define the normalized autocorrelation function
def autocorrelation(x):
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]  # Keep only the second half (positive lags) else you get something weird :/
    return result / result[0]  # Normalize by the zero-lag value

# Calculate normalized autocorrelation for px and py
acf_px = autocorrelation(px)

y = np.exp(-t) * np.cos(omega2 * t)    #equation   e(-t)- represents damping   cos term represent oscillatiosn with angular freq omega

# Plot the autocorrelation functions and equation
plt.figure()
plt.plot(acf_px, label='ACF px', color='red')
plt.plot(t, y, color='magenta',alpha=0.7)
plt.title('Normalized Autocorrelation Function px')
plt.xlabel('Lag')
plt.ylabel('Normalized Autocorrelation')
plt.legend()
plt.show()
