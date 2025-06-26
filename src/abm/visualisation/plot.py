"""Functions to plot the results of the ABM."""

import matplotlib.pyplot as plt
import numpy as np
from abm.model import RiotModel
from abm.city_map import CityMap

def run_simulations(
        city_map: CityMap, 
        n_runs: int, 
        n_steps: int, 
        width: int, 
        height: int, 
        n_home: int, 
        n_away: int, 
        entry_home: list[tuple[int, int]], 
        entry_away: list[tuple[int, int]],
        ):
    """Run the model multiple times and return mean and std dev of rioter counts."""
    all_rioters = []

    for _ in range(n_runs):
        agent_data, _, _ = RiotModel.run_riot_model(
            width,
            height,
            n_home,
            n_away,
            entry_home,
            entry_away,
            n_steps,
            city_map,
            animate=False,
            detailed_logging=False,
        )
        all_rioters.append(agent_data["Rioter"].to_numpy())

    all_rioters = np.array(all_rioters)
    mean = np.mean(all_rioters, axis=0)
    std = np.std(all_rioters, axis=0)
    return mean, std

def plot_riot_time_series(
    width: int = 50,
    height: int = 100,
    n_home: int = 4500,
    n_away: int = 500,
    entry_home: list[tuple[int, int]] = [(i, 0) for i in range(14, 19)] + [(i, 0) for i in range(22, 27)],
    entry_away: list[tuple[int, int]] = [(i, 0) for i in range(30, 35)],
    entry_home_list: list[tuple[int, int]] = None,
    entry_away_list: list[tuple[int, int]] = None,
    n_steps: int = 250,
    street_width: int = 7,
    n_streets: int = 3,
    exit_space_height: int = 10,
    street_width_values: list[int] = [5, 10, 15],
    exit_height_values: list[int] = [2, 6, 10],
    n_street_values: list[int] = [1, 3, 5],
    spacing_configs = [
        (range(15, 20), range(29, 34)),  # 2 spaces on each side
        (range(12, 17), range(32, 37)),  # 5 spaces
        (range(7, 12), range(37, 42)),   # 10 spaces
    ],
    n_runs: int = 10,
):
    """
    Plots three subplots showing number of rioters over time for different parameters.
    Each line is the mean of multiple simulations, with shaded standard deviation.
    """
    _, axs = plt.subplots(2, 2, figsize=(18, 6))
    steps = np.arange(n_steps + 1)

    # Subplot 1: Varying street width
    for sw in street_width_values:
        city_map = CityMap(width, height, n_streets=n_streets, street_width=sw, exit_space_height=exit_space_height)
        mean, std = run_simulations(city_map, n_runs, n_steps, width, height, n_home, n_away, entry_home, entry_away)
        max_mean = np.max(mean)
        line, = axs[0,0].plot(steps, mean, label=f"Street Width {sw}")
        axs[0,0].fill_between(steps, mean - std, mean + std, alpha=0.3, color=line.get_color())
        axs[0,0].hlines(max_mean, steps[0], steps[-1], color=line.get_color(), linestyle="--", alpha=0.7)

    axs[0,0].set_title("Street Width")
    axs[0,0].set_xlabel("Step")
    axs[0,0].set_ylabel("Number of Rioters")
    axs[0,0].legend()

    # Subplot 2: Varying exit space height
    for eh in exit_height_values:
        city_map = CityMap(width, height, n_streets=n_streets, street_width=street_width, exit_space_height=eh)
        mean, std = run_simulations(city_map, n_runs, n_steps, width, height, n_home, n_away, entry_home, entry_away)
        max_mean = np.max(mean)
        line, = axs[0,1].plot(steps, mean, label=f"Exit Height {eh}")
        axs[0,1].fill_between(steps, mean - std, mean + std, alpha=0.3, color=line.get_color())
        axs[0,1].hlines(max_mean, steps[0], steps[-1], color=line.get_color(), linestyle="--", alpha=0.7)

    axs[0,1].set_title("Exit Space Height")
    axs[0,1].set_xlabel("Step")
    axs[0,1].set_ylabel("Number of Rioters")
    axs[0,1].legend()

    # Subplot 3: Varying number of streets
    for n in n_street_values:
        city_map = CityMap(width, height, n_streets=n, street_width=street_width, exit_space_height=exit_space_height)
        mean, std = run_simulations(city_map, n_runs, n_steps, width, height, n_home, n_away, entry_home, entry_away)
        max_mean = np.max(mean)
        line, = axs[1,0].plot(steps, mean, label=f"Streets {n}")
        axs[1,0].fill_between(steps, mean - std, mean + std, alpha=0.3, color=line.get_color())
        axs[1,0].hlines(max_mean, steps[0], steps[-1], color=line.get_color(), linestyle="--", alpha=0.7)

    axs[1,0].set_title("Number of Streets")
    axs[1,0].set_xlabel("Step")
    axs[1,0].set_ylabel("Number of Rioters")
    axs[1,0].legend()

    # Subplot 4: Varying exit width
    center_block = [(i, 0) for i in range(22, 27)]
    spacing_labels = ["2 spaces", "5 spaces", "10 spaces"]
    entry_home_list = [
        [(i, 0) for i in left] + center_block for (left, _) in spacing_configs
    ]
    entry_away_list = [
        [(i, 0) for i in right] for (_, right) in spacing_configs
    ]   

    for idx, entry_home in enumerate(entry_home_list):
        entry_away = entry_away_list[idx]
        city_map = CityMap(width, height, n_streets=n_streets, street_width=street_width,
                           exit_space_height=exit_space_height)
        mean, std = run_simulations(city_map, n_runs, n_steps,
                                    width, height, n_home, n_away,
                                    entry_home, entry_away)
        line, = axs[1, 1].plot(steps, mean, label=spacing_labels[idx])
        axs[1, 1].fill_between(steps, mean-std, mean+std, alpha=0.3, color=line.get_color())
        axs[1, 1].hlines(np.max(mean), steps[0], steps[-1], color=line.get_color(), linestyle="--", alpha=0.7)

    axs[1, 1].set_title("Home Entry Spacing")
    axs[1, 1].set_xlabel("Step")
    axs[1, 1].set_ylabel("Number of Rioters")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    plot_riot_time_series()
