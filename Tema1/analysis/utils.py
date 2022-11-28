from typing import Union, Dict, List
from pathlib import Path

import numpy as np
import warnings
import json


__all__ = ['read_jsonl', 'load_runs', 'binning']


def read_jsonl(path: Union[Path, str]) -> List[Dict[str, float]]:
    """
    Read stats saved in the default jsonl format ('stats.jsonl')

    :param path: Path to the logged statistics

    :return: A list with each episode statistics
    E.g: [{'length': 180,
           'reward': 2.1,
            ....},

           {'length': 180,
            'reward': 2.1,
            ....}]
    """
    with open(path) as json_file:
        json_list = list(json_file)

    return [json.loads(json_str) for json_str in json_list]


def load_runs(indir: str) -> List[List[Dict[str, float]]]:
    """
    Load stats for multiple runs

    :param indir: Path to the agent runs

    :return: A list with statistics for each run
    """
    indir = Path(indir)
    filenames = sorted(list(indir.glob("**/stats.jsonl")))

    return [read_jsonl(filename) for filename in filenames]


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
    xs, ys = np.array(xs), np.array(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    binned = []

    with warnings.catch_warnings():  # Empty buckets become NaN.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for start, stop in zip(borders[:-1], borders[1:]):
            left = (xs <= start).sum()
            right = (xs <= stop).sum()
            if left < right:
                value = reducer(ys[left:right])
            elif binned:
                value = {'nan': np.nan, 'last': binned[-1]}[fill]
            else:
                value = np.nan
            binned.append(value)
    return borders[1:], np.array(binned)