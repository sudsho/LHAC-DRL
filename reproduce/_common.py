"""Shared paths and styling for the reproduce_* scripts."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES, exist_ok=True)

# Paper colour palette
COLOR_LHAC = '#2196F3'
COLOR_GAMIP = '#9E9E9E'
COLOR_GRAY_DARK = '#616161'
COLOR_GRAY_MED = '#9E9E9E'
COLOR_GRAY_LIGHT = '#BDBDBD'
COLOR_GRAY_FAINT = '#E0E0E0'

# Line palette (non-red, used for multi-method comparisons)
LINE_PALETTE = {
    'adaptive_tlo':   '#2196F3',
    'lhac_daf':       '#2196F3',
    'lppo':           '#43A047',
    'envelope_morl':  '#FB8C00',
    'ppo_lagrangian': '#6A1B9A',
    'rcpo':           '#00838F',
    'tlo_in_critic':  '#37474F',
    'weighted_rl':    '#9E9E9E',
}


def out(name: str) -> str:
    """Return absolute path to a figure file under figures/."""
    return os.path.join(FIGURES, name)
