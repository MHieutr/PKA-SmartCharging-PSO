import numpy as np
import random
import matplotlib.pyplot as plt



# Inputs
N = 10  # Maximum number of EVs (capacity)
F = 50000  # Maximum (full) battery capacity in kW
U = 220  # Voltage in Volts
I_options = [8, 16, 32, 48, 64]  # Charging current options in Amps
P = [U * I for I in I_options]  # Power in kW
# Compute Tmin, Tmax, D
D = 13  # Number of time steps
max_power_limit = 350000  # Max allowable power
Delta_t = 0.5  # Time step duration in hours 
E_required = np.full(N, 50000)
# Main genetic algorithm loop
# Number of simulations
nSim = 100

# INPUT
B = np.random.randint(0, F, size=N)  # Random initial battery levels


# Parameters for Genetic Algorithm
population_size = 50
generations = 100
mutation_rate = 0.1

# Define the objective function
def objective_function(particle, alpha = 1):
    penalty = 0
    for i in range(N):
        total_energy = np.sum(particle[i, :] * U * Delta_t)
        energy_shortfall = abs(total_energy - E_required[i])
        penalty += energy_shortfall ** alpha
    return -penalty**(1/alpha)

# Generate initial population (random I_t assignments)
def initialize_population(pop_size, N, D):
    population = []
    for _ in range(pop_size):
        individual = np.random.choice(I_options, size=(N, D))
        population.append(individual)
    return population

# Apply crossover between two parents
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, D-1)
    child1 = np.concatenate([parent1[:, :crossover_point], parent2[:, crossover_point:]], axis=1)
    child2 = np.concatenate([parent2[:, :crossover_point], parent1[:, crossover_point:]], axis=1)
    return child1, child2

# Apply mutation to an individual
def mutate(individual):
    for i in range(N):
        for t in range(D):
            if random.random() < mutation_rate:
                individual[i, t] = random.choice(I_options)
    return individual

# Select the best individuals
def select_population(population, fitness_values, num_select):
    sorted_indices = np.argsort(fitness_values)[::-1]
    selected = [population[i] for i in sorted_indices[:num_select]]
    return selected



# Initialize storage for results across simulations
column_sums_all = []
I_schedules_all = []

for sim in range(nSim):
    
    population = initialize_population(population_size, N, D)

    for generation in range(generations):
        fitness_values = []
        for individual in population:
            M = np.zeros((N, D))

            M[:, 0] = B  # Set initial battery levels

            # Update M based on individual's current selections
            for i in range(N):
                for t in range(1, D):
                    if M[i, t-1] < F:  # Only charge if not fully charged
                        delta_battery = U * individual[i, t]  # Power added
                        M[i, t] = min(M[i, t-1] + delta_battery, F)  # Update battery level
                    else:
                        M[i, t] = 0  # Release space if fully charged

            # Calculate fitness for this individual
            fitness_values.append(objective_function(M))

        # Select best individuals for next generation
        population = select_population(population, fitness_values, population_size // 2)

        # Create next generation via crossover and mutation
        next_population = []
        while len(next_population) < population_size:
            parents = random.sample(population, 2)
            child1, child2 = crossover(parents[0], parents[1])
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        population = next_population

        # Output progress
        best_fitness = max(fitness_values)
#         print(f"Generation {generation+1} - Best Fitness: {best_fitness}")

    # Get the best individual after all generations
    best_individual = population[np.argmax(fitness_values)]

    # Create the final charging matrix M with the best I_t selection
    M_best = np.zeros((N, D))
    #B = np.random.randint(0, F, size=N)  # Random initial battery levels
    M_best[:, 0] = B  # Set initial battery levels

    # Create a matrix to store the charging current schedule for the best individual
    I_schedule_best = np.zeros((N, D))

    # Scheduling I for each EV over the TimeDim
    for i in range(N):
        ev_fully_charged = False  # Flag to track if the EV is fully charged
        for t in range(1, D):
            if ev_fully_charged:
                I_schedule_best[i, t] = 0  # Once fully charged, all future slots remain 0
            elif M_best[i, t-1] < F:  # Only schedule charging if battery is not full
                I_t = best_individual[i, t]  # Get the current from the best individual
                I_schedule_best[i, t] = I_t  # Store the current in the schedule
                delta_battery = U * I_t
                M_best[i, t] = min(M_best[i, t-1] + delta_battery, F)  # Update battery level

                # Check if battery is now fully charged
                if M_best[i, t] == F:
                    ev_fully_charged = True  # Mark EV as fully charged
                    M_best[i, t] = F  # Release the space, EV has finished charging
            else:
                I_schedule_best[i, t] = 0  # No current if fully charged
                M_best[i, t] = 0  # Release the space, EV is fully charged and no longer occupying
                
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
plt.savefig('GA_obj2.jpg', dpi = 600)
#plt.show()