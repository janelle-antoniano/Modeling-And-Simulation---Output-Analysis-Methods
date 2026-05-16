import numpy as np
import matplotlib.pyplot as plt

def simulate_queue(n_obs, seed=None):
    """Simulate M/M/1 queue wait times."""
    rng = np.random.default_rng(seed)
    arrival_rate = 0.8
    service_rate = 1.0
    wait_times = []
    last_departure = 0.0
    t = 0.0
    for _ in range(n_obs):
        inter_arrival = rng.exponential(1 / arrival_rate)
        t += inter_arrival
        service = rng.exponential(1 / service_rate)
        start = max(t, last_departure)
        wait = start - t
        last_departure = start + service
        wait_times.append(wait)
    return np.array(wait_times)

def moving_average(data, window=50):
    """Compute centered moving average."""
    return np.convolve(data, np.ones(window) / window, mode='valid')

def welch_method(n_replications=5, n_obs=2000, window=50):
    # Run replications
    all_obs = np.array([
        simulate_queue(n_obs, seed=i)
        for i in range(n_replications)
    ])

    # Average across replications at each time point
    y_bar = all_obs.mean(axis=0)

    # Smooth with moving average
    smoothed = moving_average(y_bar, window=window)

    # Warm-up heuristic: find where smoothed stabilizes
    overall_mean = smoothed[len(smoothed) // 2:].mean()
    tolerance = 0.10 * overall_mean
    warmup_idx = next(
        (i for i in range(len(smoothed))
         if abs(smoothed[i] - overall_mean) < tolerance),
        window
    )

    print(f"Estimated warm-up period: {warmup_idx} observations")
    print(f"Steady-state mean (post warm-up): {overall_mean:.4f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(y_bar, alpha=0.3, label="Avg across replications")
    plt.plot(range(window // 2, window // 2 + len(smoothed)),
             smoothed, label="Moving average", linewidth=2)
    plt.axvline(warmup_idx, color='red', linestyle='--',
                label=f"Warm-up cutoff = {warmup_idx}")
    plt.axhline(overall_mean, color='green', linestyle=':',
                label=f"Steady-state mean = {overall_mean:.2f}")
    plt.xlabel("Observation index")
    plt.ylabel("Wait time")
    plt.title("Welch Method – Warm-up Detection")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return warmup_idx

welch_method()
