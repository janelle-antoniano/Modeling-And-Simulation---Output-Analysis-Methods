# Overview

Output analysis is the process of examining simulation results to ensure they are accurate and reliable. Different methods are used to reduce bias and analyze system performance effectively. Common techniques include the Welch Method, Replication-Deletion Approach, and Batch Mean Method, which help evaluate data and improve the accuracy of simulation studies.

1. Welch Method
Used to determine the warm-up period (truncation point) for non-terminating simulations, so that initial transient bias is removed before collecting statistics.
How it works:
Run n replications of length m
Compute the average of observation *i* across all replications: Yˉi\bar{Y}_i Yˉi​
Apply a moving average to smooth the curve
Visually identify where the plot stabilizes — that's your truncation point w
Discard all observations before w
Sample Python Code
import numpy as np
import matplotlib.pyplot as plt

# Sample simulation output data
data = [5, 7, 9, 12, 15, 18, 20, 21, 22, 22, 23, 23]

# Welch moving average
window_size = 3
moving_avg = []

for i in range(len(data) - window_size + 1):
   avg = np.mean(data[i:i + window_size])
   moving_avg.append(avg)

# Display result
print("Moving Averages:", moving_avg)

# Plot
plt.plot(data, label="Original Data")
plt.plot(range(window_size - 1, len(data)), moving_avg,
        label="Welch Moving Average")
plt.legend()
plt.show()





2. Replication-Deletion Approach
Used to estimate steady-state performance by running multiple independent replications, each with a warm-up period deleted.
How it works:
Run n independent replications
Delete the first w observations (warm-up) from each
Compute the sample mean Xˉi\bar{X}_i Xˉi​ for each replication
Use these means to compute the overall mean and confidence interval
Sample Python Code

import random
import numpy as np

replications = 5
warmup = 3
results = []

for r in range(replications):

    # Generate random simulation data
    simulation = [random.randint(10, 50) for _ in range(10)]

    # Delete warm-up observations
    steady_state = simulation[warmup:]

    # Compute average
    avg = np.mean(steady_state)
    results.append(avg)

    print(f"Replication {r+1}: {steady_state}")

print("\nAverage per replication:", results)
print("Overall Average:", np.mean(results))








3. Batch Means Method
Used when only one long run is available. It divides the run into batches to create approximately independent observations.
How it works:
Run one long simulation (after warm-up deletion)
Divide remaining observations into k batches of size m
Compute the mean Yˉj\bar{Y}_j Yˉj​ for each batch
Treat batch means as approximately i.i.d. and compute a CI
Sample Python Code
import numpy as np

# Long simulation output
data = [12, 15, 18, 20, 22, 25, 27, 30, 32, 35]

batch_size = 2
batch_means = []

# Divide data into batches
for i in range(0, len(data), batch_size):
   batch = data[i:i + batch_size]
   mean = np.mean(batch)
   batch_means.append(mean)

print("Batch Means:", batch_means)
print("Overall Mean:", np.mean(batch_means))
