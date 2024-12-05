import numpy as np
import matplotlib.pyplot as plt

# Inputs
N = 10  # Number of EVs
F = 50000  # Maximum capacity
I_options = [8, 16, 32, 48, 64]  # Electric current options
U = 220  # Voltage
D = 12  # Number of time steps
max_power_limit = 350000  # Maximum power limit
Delta_t = 0.5  # Time step duration in hours (30 minutes)
# Number of simulations
nSim = 100

E_required = np.full(N, 50000)
# Initial battery level
B = np.random.randint(0, F, size=N)

# PSO parameters
num_particles = 30
max_iterations = 100
w = 0.5  # Inertia coefficient
phi_p = 0.5  # Personal influence coefficient
phi_g = 0.5  # Global influence coefficient

# Utils
target_values = np.array(I_options)

def nearest_value(x):
    if x < 8:
        return 8
    elif x > 64:
        return 64
    else:
        return target_values[np.abs(target_values - x).argmin()]

# Define the objective function
def objective_function(particle, alpha = 1):
    penalty = 0
    for i in range(N):
        total_energy = np.sum(particle[i, :] * U * Delta_t)
        energy_shortfall = abs(total_energy - E_required[i])
        penalty += energy_shortfall ** alpha
    return -penalty**(1/alpha)



# Initialize storage for results across simulations
column_sums_all = []
I_schedules_all = []

# Run optimization nSim times
for sim in range(nSim):
    # Re-initialize global best values for each simulation
    global_best_value = float('-inf')
    global_best_position = None

    # Re-initialize particle positions and velocities
    particle_positions = np.random.choice(I_options, size=(num_particles, N, D))
    particle_velocities = np.random.uniform(-1, 1, size=(num_particles, N, D))
    personal_best_positions = particle_positions.copy()

    # Evaluate the initial population
    for i in range(num_particles):
        value = objective_function(particle_positions[i])
        if value > global_best_value:
            global_best_value = value
            global_best_position = particle_positions[i]

    # PSO main loop
    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Update velocity
            rp, rg = np.random.uniform(0, 1, size=(2, N, D))
            particle_velocities[i] = (
                w * particle_velocities[i]
                + phi_p * rp * (personal_best_positions[i] - particle_positions[i])
                + phi_g * rg * (global_best_position - particle_positions[i])
            )

            # Update position
            particle_positions[i] = particle_positions[i] + particle_velocities[i]

            # Clip positions to available I_options
            particle_positions[i] = np.vectorize(nearest_value)(particle_positions[i])

            # Evaluate new position
            value = objective_function(particle_positions[i])

            # Update personal best and global best
            f_pi = objective_function(personal_best_positions[i])
            if value > f_pi:
                personal_best_positions[i] = particle_positions[i]
                if f_pi > global_best_value:
                    global_best_value = f_pi
                    global_best_position = particle_positions[i]

            # Update global best
            # if value > global_best_value:
            #     global_best_value = value
            #     global_best_position = particle_positions[i]

    # Simulate the battery level and charging schedule for the best solution
    M_best = np.zeros((N, D))
    M_best[:, 0] = B  # Set initial battery levels
    I_schedule_best = np.zeros((N, D))

    for i in range(N):
        ev_fully_charged = False
        for t in range(1, D):
            if ev_fully_charged:
                I_schedule_best[i, t] = 0
            elif M_best[i, t-1] < F:
                I_t = global_best_position[i, t]
                I_schedule_best[i, t] = I_t
                delta_battery = U * I_t * 0.5  # 0.5 hour for each time step
                M_best[i, t] = min(M_best[i, t-1] + delta_battery, F)
                if M_best[i, t] == F:
                    ev_fully_charged = True
            else:
                I_schedule_best[i, t] = 0
                M_best[i, t] = F

    # Collect results
    column_sums_all.append(M_best.sum(axis=0))
    I_schedules_all.append(I_schedule_best)

# Convert collected results into arrays for analysis
column_sums_all = np.array(column_sums_all)/1000
I_schedules_all = np.array(I_schedules_all)

# Calculate mean and standard deviation
mean_column_sums = column_sums_all.mean(axis=0)
std_column_sums = column_sums_all.std(axis=0)

mean_I_schedules = I_schedules_all.mean(axis=0)
std_I_schedules = I_schedules_all.std(axis=0)

# Plot the mean and variation of column_sum_best
fig, ax = plt.subplots(1, 2, figsize=(14, 4))

# Plot mean and variation of total power
ax[0].plot(range(D), mean_column_sums, label="Mean Total Energy", color="blue", marker='o')
ax[0].fill_between(
    range(D),
    mean_column_sums - std_column_sums,
    mean_column_sums + std_column_sums,
    color="blue",
    alpha=0.3,
    label="±1 Standard Deviation",
)
ax[0].axhline(y=max_power_limit/1000, color="r", linestyle="--", label="Max Power Limit")
ax[0].set_title("Mean Optimized Charging Power", fontsize = 14, weight = 'bold')
time_labels = [f"{5 * t}" for t in range(D)]
ax[0].set_xticks(range(D))
ax[0].set_xticklabels(time_labels, fontsize = 12)
ax[0].set_xlabel("Time [minute]", fontsize = 14, weight = 'bold')
ax[0].set_ylabel("Total Power Consumption (kW)", fontsize = 14, weight = 'bold')
ax[0].grid(True)
ax[0].legend()
markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "X"]

# Plot mean and variation of charging current schedules
for i in range(N):
    ax[1].plot(range(D), mean_I_schedules[i], label=f'Station {i+1}', marker=markers[i])
    ax[1].fill_between(
        range(D),
        mean_I_schedules[i] - std_I_schedules[i],
        mean_I_schedules[i] + std_I_schedules[i],
        alpha=0.1,
    )
ax[1].set_title("Mean Charging Current Schedule for Each Station", fontsize = 14, weight = 'bold')
time_labels = [f"{5 * t}" for t in range(D)]
ax[1].set_xticks(range(D))
ax[1].set_xticklabels(time_labels, fontsize = 12)
ax[1].set_xlabel("Time [minute]", fontsize = 14, weight = 'bold')

# Update y-axis to display the allowed current options
ax[1].set_yticks(I_options, fontsize = 12)
ax[1].set_yticklabels(I_options, fontsize = 12)
ax[1].set_ylabel("Charging Current (A)", fontsize = 14, weight = 'bold')
ax[1].grid(True)
ax[1].legend(loc="upper right", ncol=1)

plt.tight_layout()
plt.savefig('PSO_obj2_sim.jpg', dpi = 600)
plt.show()
