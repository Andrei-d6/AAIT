from typing import List, Dict, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import load_runs


def compute_percents(
    runs: Dict[str, List[List[Dict[str, float]]]],
    tasks: List[str]
) -> Dict[str, np.ndarray]:
    """
    Compute the percentage in which each achievemnt is obtained by the agents

    :param runs: Dictionary with episode statistics for each agent (possibly over multiple runs)
        Expected structure:
            runs -> 'Agent1':  [[
                        {'length': 180,
                         'reward': 2.1,
                         ....},

                         {'length': 180,
                          'reward': 2.1,
                         ....},
                    ]]
    :param tasks: Name of the keys of interest (achievements) from the episode stats

    :return: A dictionary with the percentages associated with all the achievements for each agent
    """
    percents = dict()

    for method in runs:

        method_runs = runs[method]

        current_percents = []

        for episodes in method_runs:
            percents_ = np.zeros(len(tasks))

            for episode in episodes:
                for idx, task in enumerate(tasks):
                     percents_[idx] += 1 if episode[task] >= 1 else 0

            percents_ = (percents_ * 100.) /  float(len(episodes))
            current_percents.append(percents_)

        percents[method] = np.mean(current_percents, axis=0)

    return percents


def plot_spectrum(
    inpaths: List[Path],
    labels: List[str],
    outpath: Optional[Path] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (7, 3),
    dpi: int = 300,
    show: bool = False
):
    """
    Plot the spectrum of achievements obtained by agents over a series of episodes

    :param inpaths: Input paths to the episodes statistics for all the agents
    :param outpath: Output path for saving the plot
    :param labels: List of labels for each agent
    :param colors: List of colors for each agent
    :param figsize: Figure size of the final plot
    """

    runs = dict()

    for inpath, label in zip(inpaths, labels):
        runs[label] = load_runs(inpath)

    tasks = sorted(key for key in runs[labels[0]][0][0] if key.startswith('achievement_'))

    percents = compute_percents(runs, tasks)

    legend = {method: label for method, label in zip(runs.keys(), labels)}

    fig, ax = plt.subplots(figsize=figsize)
    centers = np.arange(len(tasks))
    width = 0.7

    for index, (method, label) in enumerate(legend.items()):
        heights = percents[method]
        pos = centers + width * (0.5 / len(legend) + index / len(legend) - 0.5)
        color = colors[index] if colors else None
        ax.bar(pos, heights, width / len(legend), label=label, color=color)

    names = [x[len('achievement_'):].replace('_', ' ').title() for x in tasks]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(
      axis='x', which='both', width=14, length=0.8, direction='inout')
    ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
    ax.set_xticks(centers + 0.0)
    ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')

    ax.set_ylabel('Success Rate (%)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels('0.01 0.1 1 10 100'.split())

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(
      loc='upper center', ncol=10, frameon=False, borderpad=0, borderaxespad=0)

    if outpath is not None:
        Path(outpath).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(outpath, dpi=dpi)

    if show:
        plt.show()