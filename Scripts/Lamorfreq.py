import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     #motion with damping and thermal noise
                                            #E so high ??   try -1 for circular motion

# Constants
q = 3.20e-19  # Charge of the particle (Coulombs)
m = 6.66e-26  # Mass of the particle (kg)
B = 100000 # Magnetic field strength (Tesla) along the x-axis
E = 2.631e8  # Electric field strength (V/m) along the z-axis
v0 = 170  # Initial velocity in m/s (perpendicular to B)
mobility = 6.17e-8 # Mobility of the ion (m^2/V/s)

# Noise Constants
T = 300  # Temperature in Kelvin
kB = 1.38e-23  # Boltzmann constant (J/K)

# Calculating gamma (damping term)
gamma = (q / m) * mobility

# Calculating the cyclotron frequency
omega_c = q * B / m  # Cyclotron frequency (rad/s)
R = v0 / omega_c    # Radius of motion
t_max = 10 * 2 * np.pi / omega_c  # Simulating for 10 cyclotron periods
N = 10000  # Number of time steps
dt = t_max / N  # Time step

# Time array
t = np.linspace(0, t_max, N)

# Initial conditions
v = np.zeros((N, 3))  # Velocity array (x, y, z)
r = np.zeros((N, 3))  # Position array (x, y, z)
v[0] = [0, v0, 0]  # Initial velocity

# Langevin random force term (thermal noise)
def random_force():
    return np.sqrt(2 * gamma * kB * T / m / dt) * np.random.normal(0, 1, 3)  #as in equation
    #return np.zeros(3)

# Integrating the equations of motion with damping and random forces
for i in range(1, N):
    v_cross_B = np.cross(v[i-1], [B, 0, 0])  # Cross product v(t) x B
    random_term = random_force()  # Random noise term

    # Update velocity (Euler method)
    v[i] = v[i-1] + dt * (-gamma * v[i-1] + (q/m) * (np.array([0, 0, E]) + v_cross_B) + random_term)
    
    # Update position
    r[i] = r[i-1] + v[i] * dt

# Extract x, y, z positions
x, y, z = r[:, 0], r[:, 1], r[:, 2]

# Plotting the 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Damped Helical trajectory with random motion', color='b')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Damped Helical Motion ( with Noise)')
ax.legend()

# Show the plot
plt.show()
print("Cyclotron Frequency =", omega_c)
print('Radius =', R )
print('Time Period = ', t_max / 10)
#print('Lamor Frequency=', omega_c / (2 * np.pi))
