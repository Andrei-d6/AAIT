from typing import List, Optional, Tuple
from pathlib import Path


import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

from utils import load_runs, binning



def plot_counts(
    indir: str,
    outpath: Optional[Path] = None,
    color: Optional[str] = None,
    budget: float = 5e5,
    cols: int = 5,
    figsize: Tuple[int, int] = (2, 1.8),
    dpi: int = 300,
    show: bool = False
):

    runs = load_runs(indir)

    tasks = sorted(key for key in runs[0][0] if key.startswith('achievement_'))
    keys = ['reward', 'length'] + tasks

    borders = np.arange(0, budget, 1e4)
    rows = len(keys) // cols

    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(figsize[0] * cols, figsize[1] * rows)
    )

    for ax, key in zip(axes.flatten(), keys):
        ax.set_title(key.replace('achievement_', '').replace('_', ' ').title())

        x = np.concatenate([np.cumsum([episode['length'] for episode in episodes])
                       for episodes in runs])

        y = np.concatenate([[episode[key] for episode in episodes] for episodes in runs])

        binx, biny = binning(x, y, borders, np.nanmean)
        ax.plot(binx, biny, color=color)

        mins = binning(x, y, borders, np.nanmin)[1]
        maxs = binning(x, y, borders, np.nanmax)[1]
        ax.fill_between(binx, mins, maxs, linewidths=0, alpha=0.2, color=color)

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))

        if maxs.max() == 0:
            ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()

    if outpath is not None:
        Path(outpath).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(outpath, dpi=dpi)

    if show:
        plt.show()