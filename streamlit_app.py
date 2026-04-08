#!/usr/bin/env python3
"""
Streamlit Comparison GUI - Parvez vs LHAC vs Hybrid.

Three scheduling methods across 3 windows x 3 bank configs x 27 datasets.
Includes random dataset generator with interactive sliders.

Launch:
    streamlit run streamlit_comparison.py
"""

import os
import sys
import time
import copy
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
# Add Parvez directory to path for run_parvez_multibank import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
_PARVEZ_DIR = os.path.join(_PROJECT_ROOT, 'Parvez')
if os.path.isdir(_PARVEZ_DIR) and _PARVEZ_DIR not in sys.path:
    sys.path.insert(0, _PARVEZ_DIR)

from daf_lhac_core import (
    FacilityConfiguration, LexicographicRewardEnvironment,
    MultiAgentLHACPPO, CellStatus
)
from evaluate import (
    evaluate_single, TABLE6_DATASETS, ORDERING_STRATEGIES,
    _run_single_ordering, _compute_tardiness,
    WINDOW_BLOCKS, load_agent, StateTruncatingAgent,
    spread_arrivals_to_hours, partition_servers_by_window,
    WINDOW_HOURS, HOURS_PER_DAY
)

# Constants
NUM_CELLS_PER_BANK = 14
TOTAL_DAYS = 65
BANK_NAMES = {1: "052B", 2: "052C", 3: "052D", 4: "052E"}

DATA_DIR = os.path.join(BASE_DIR, 'Data')
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # Jan 2026 Warm start/
PARVEZ_DATA_DIR = os.path.join(PROJECT_ROOT, 'Parvez', 'Data_datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'Models')
INPUT_DIR = os.path.join(PROJECT_ROOT, 'Parvez', 'Input')
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')

BANK_CONFIGS = {
    1: {'num_cells': 14, 'label': '1 Bank (14 cells)', 'excluded': set()},
    2: {'num_cells': 28, 'label': '2 Banks (28 cells)', 'excluded': set()},
    4: {'num_cells': 54, 'label': '4 Banks (54 cells)', 'excluded': {(4, 4), (4, 7)}},
}

WINDOW_CONFIGS = {
    '10d': {'window_length': 10, 'label': '10 Days', 'unit_time_length': 24, 'divide_time': 10},
    '5d':  {'window_length': 5,  'label': '5 Days',  'unit_time_length': 24, 'divide_time': 5},
    '1d':  {'window_length': 1,  'label': '1 Day',   'unit_time_length': 24, 'divide_time': 1},
    '8h':  {'window_length': 8,  'label': '8 hours (sub-day)'},
    '4h':  {'window_length': 4,  'label': '4 hours (sub-day)'},
    '0h':  {'window_length': 0,  'label': 'Real-time (0h)'},
}

DAY_LEVEL_WINDOWS = {'10d', '5d', '1d'}
SUB_DAY_WINDOWS = {'8h', '4h', '0h'}


# ============================================================================
# BankFilteredEnvironment (for LHAC with restricted banks)
# ============================================================================

class BankFilteredEnvironment(LexicographicRewardEnvironment):
    """Environment that restricts valid actions to cells from active banks.

    IMPORTANT: Preserves parent's _cached_valid_actions caching behavior.
    The parent's _get_state() calls get_valid_actions() which populates the cache.
    Then get_valid_actions_deferred() finds this stale cache and returns it.
    This matches the training environment behavior — changing it would give the
    agent fresh valid actions that don't match training, inflating completion rates.
    """

    def __init__(self, facility, servers_df, num_banks=4, **kwargs):
        super().__init__(facility, servers_df, **kwargs)
        self.num_active_banks = num_banks
        self.active_cell_indices = set()
        for bank_idx in range(num_banks):
            if bank_idx in self.bank_cell_indices:
                self.active_cell_indices.update(self.bank_cell_indices[bank_idx])
        self.skip_action = self.total_cells

    def _filter_actions(self, valid_actions):
        filtered = [a for a in valid_actions
                    if a == self.skip_action or a in self.active_cell_indices]
        if self.skip_action not in filtered:
            filtered.append(self.skip_action)
        return filtered

    def get_valid_actions(self):
        # Respect the cache — same as parent class behavior
        if self._cached_valid_actions is not None:
            return self._cached_valid_actions
        parent_valid = super().get_valid_actions()
        filtered = self._filter_actions(parent_valid)
        self._cached_valid_actions = filtered
        return filtered

    def get_valid_actions_deferred(self):
        # Respect the cache — same as parent class behavior
        if self._cached_valid_actions is not None:
            return self._cached_valid_actions
        parent_valid = super().get_valid_actions_deferred()
        filtered = self._filter_actions(parent_valid)
        if hasattr(self, '_efs_cache') and self._efs_cache:
            self._efs_cache = {k: v for k, v in self._efs_cache.items()
                               if k in self.active_cell_indices}
        self._cached_valid_actions = filtered
        return filtered


# ============================================================================
# Rolling Window Helpers
# ============================================================================

def partition_servers_by_day(df, window_days, total_days=TOTAL_DAYS):
    """Partition servers into rolling windows by arrival day.

    Args:
        df: DataFrame with 'ArrivalTime' column (day-level, 1-based).
        window_days: Window size in days (e.g., 10, 5, 1).
        total_days: Total horizon in days. Default: 65.

    Returns:
        List of lists, where windows[i] = [df indices of servers in window i].
    """
    n_windows = math.ceil(total_days / window_days)
    windows = [[] for _ in range(n_windows)]
    for idx in range(len(df)):
        arr_day = int(df.iloc[idx]['ArrivalTime'])
        w_idx = min((arr_day - 1) // window_days, n_windows - 1)
        windows[w_idx].append(idx)
    return windows


def partition_servers_subday(df, window_key, total_days=TOTAL_DAYS):
    """Partition servers for sub-day windows using hourly granularity.

    Args:
        df: DataFrame with 'ArrivalTime' column (day-level, 1-based).
        window_key: One of '8h', '4h', '0h'.
        total_days: Total horizon in days. Default: 65.

    Returns:
        Tuple of (windows, df_h_sorted, whours) where:
        - windows: List of lists of indices into df_h_sorted
        - df_h_sorted: The hourly-spread sorted DataFrame
        - whours: Window size in hours
    """
    _, whours = WINDOW_HOURS[window_key]
    df_h = spread_arrivals_to_hours(df)
    df_h_sorted = df_h.sort_values(
        ['ArrivalTime_hours', 'OTk', 'PTk'],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    if whours == 0:
        # Real-time: one window per unique arrival hour
        unique_hours = sorted(df_h_sorted['ArrivalTime_hours'].unique())
        windows = []
        for h in unique_hours:
            idxs = df_h_sorted[df_h_sorted['ArrivalTime_hours'] == h].index.tolist()
            windows.append(idxs)
    else:
        windows = partition_servers_by_window(df_h_sorted, whours)

    return windows, df_h_sorted, whours


def _run_ordering_rolling_subday(agent, facility, df, windows, df_h_sorted,
                                  whours, sort_fn, num_banks=4,
                                  use_dap=True, use_fpr=True,
                                  progress_callback=None, ordering_idx=0,
                                  total_orderings=5):
    """Run one ordering strategy through sub-day rolling windows with carry-over.

    Similar to _run_ordering_rolling but converts hourly window bounds back to
    day-level for the environment (same logic as evaluate.py lines 932-943).
    """
    full_env = LexicographicRewardEnvironment(
        facility, df, window_length=TOTAL_DAYS,
        use_windowing=False, randomize_order=False,
        use_deferred_actions=use_dap, use_fpr=use_fpr)
    full_env.reset()

    total_assigned = 0
    total_rl_only = 0
    total_tardy = 0
    carryover_indices = []
    active_windows = 0
    total_window_count = len(windows)

    for w_idx, w_server_indices in enumerate(windows):
        if len(w_server_indices) == 0 and len(carryover_indices) == 0:
            continue

        all_indices = carryover_indices + w_server_indices
        if len(all_indices) == 0:
            continue

        active_windows += 1

        # Progress callback
        if progress_callback:
            ordering_name = ORDERING_STRATEGIES[ordering_idx][0] if ordering_idx < len(ORDERING_STRATEGIES) else "?"
            base_pct = ordering_idx / total_orderings * 0.7
            window_pct = (active_windows / max(total_window_count, 1)) * (0.7 / total_orderings)
            progress_callback(
                base_pct + window_pct,
                f"Window {w_idx + 1}/{total_window_count} — ordering: {ordering_name}")

        # Build sub-dataframe from the hourly-sorted frame
        window_rows = df_h_sorted.iloc[all_indices]
        sub_df = pd.DataFrame({
            'k': window_rows['k'].values,
            'type': window_rows['type'].values if 'type' in window_rows.columns else ['artemis'] * len(window_rows),
            'PTk': window_rows['PTk'].values,
            'OTk': window_rows['OTk'].values,
            'RH': window_rows['RH'].values,
            'RL': window_rows['RL'].values,
            'RW': window_rows['RW'].values,
            'DueTime': window_rows['DueTime'].values,
            'ArrivalTime': window_rows['ArrivalTime'].values,
        })

        # Convert hourly window bounds to day-level for the environment
        if whours == 0:
            w_start_day = int(window_rows['ArrivalTime'].min())
            w_end_day = min(int(window_rows['ArrivalTime'].max()) + 1, TOTAL_DAYS)
        else:
            w_start_hour = w_idx * whours + 1
            w_end_hour = min((w_idx + 1) * whours, TOTAL_DAYS * HOURS_PER_DAY)
            w_start_day = max(1, (w_start_hour - 1) // HOURS_PER_DAY + 1)
            w_end_day = min(w_end_hour // HOURS_PER_DAY + 1, TOTAL_DAYS)

        wlen = max(w_end_day - w_start_day + 1, 1)

        # Create window environment with bank filtering
        if num_banks == 4:
            window_env = LexicographicRewardEnvironment(
                facility, sub_df,
                window_length=wlen, use_windowing=True,
                randomize_order=False,
                use_deferred_actions=use_dap, use_fpr=use_fpr)
        else:
            window_env = BankFilteredEnvironment(
                facility, sub_df, num_banks=num_banks,
                window_length=wlen, use_windowing=True,
                randomize_order=False,
                use_deferred_actions=use_dap, use_fpr=use_fpr)

        # Reset, then carry over state from previous windows
        window_env.reset()
        window_env.cell_occupancy = full_env.cell_occupancy.copy()
        window_env.assignments = copy.deepcopy(full_env.assignments)
        window_env.assigned_servers = set()

        # Sort servers by ordering strategy
        if sort_fn is not None:
            window_env.servers.sort(key=sort_fn)
            priority_sorted = sorted(window_env.servers,
                                     key=lambda s: (s['DueTime'], s['ArrivalTime']))
            for rank, s in enumerate(priority_sorted):
                s['priority_rank'] = rank
                s['priority_cost'] = (len(window_env.servers) - rank) * 100000

        # Get state with updated occupancy
        state = window_env._get_state()

        # Run RL agent
        done = False
        while not done:
            if use_dap:
                valid_actions = window_env.get_valid_actions_deferred()
            else:
                valid_actions = window_env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            state, _, done, _, info = window_env.step(action)

        # Count this window's placements
        rl_assigned_this = len(window_env.assigned_servers)
        total_rl_only += rl_assigned_this
        total_assigned += rl_assigned_this
        total_tardy += _compute_tardiness(window_env)

        # Update full env with this window's final state
        full_env.cell_occupancy = window_env.cell_occupancy.copy()
        full_env.assignments = copy.deepcopy(window_env.assignments)

        # Carry-over: unassigned servers that still have time left
        assigned_k_ids = set(window_env.assignments.keys())
        new_carryover = []
        for global_idx in all_indices:
            server_k = int(df_h_sorted.iloc[global_idx]['k'])
            if server_k not in assigned_k_ids:
                arr = int(df_h_sorted.iloc[global_idx]['ArrivalTime'])
                pt = int(df_h_sorted.iloc[global_idx]['PTk'])
                if arr + pt <= TOTAL_DAYS + 1:
                    new_carryover.append(global_idx)
        carryover_indices = new_carryover

    return {
        'total_assigned': total_assigned,
        'total_rl_only': total_rl_only,
        'total_tardy': total_tardy,
        'n_windows': active_windows,
        'full_env': full_env,
    }


def _run_ordering_rolling(agent, facility, df, windows, window_days,
                          sort_fn, num_banks=4, use_dap=True, use_fpr=True,
                          progress_callback=None, ordering_idx=0,
                          total_orderings=5):
    """Run one ordering strategy through all rolling windows with carry-over.

    Ported from evaluate_multibank.py _run_ordering_rolling_mb().
    Creates a full_env to track cumulative cell occupancy and assignments,
    then processes each window sequentially with state carry-over.
    """
    # Full environment for tracking occupancy across windows
    full_env = LexicographicRewardEnvironment(
        facility, df, window_length=TOTAL_DAYS,
        use_windowing=False, randomize_order=False,
        use_deferred_actions=use_dap, use_fpr=use_fpr)
    full_env.reset()

    total_assigned = 0
    total_rl_only = 0
    total_tardy = 0
    carryover_indices = []
    active_windows = 0
    total_window_count = len(windows)

    for w_idx, w_server_indices in enumerate(windows):
        if len(w_server_indices) == 0 and len(carryover_indices) == 0:
            continue

        all_indices = carryover_indices + w_server_indices
        if len(all_indices) == 0:
            continue

        active_windows += 1

        # Progress callback
        if progress_callback:
            ordering_name = ORDERING_STRATEGIES[ordering_idx][0] if ordering_idx < len(ORDERING_STRATEGIES) else "?"
            base_pct = ordering_idx / total_orderings * 0.7
            window_pct = (active_windows / max(total_window_count, 1)) * (0.7 / total_orderings)
            progress_callback(
                base_pct + window_pct,
                f"Window {w_idx + 1}/{total_window_count} — ordering: {ordering_name}")

        # Create sub-dataframe for this window
        sub_rows = df.iloc[all_indices]
        sub_df = sub_rows.reset_index(drop=True)

        # Window bounds
        w_start_day = w_idx * window_days + 1
        w_end_day = min((w_idx + 1) * window_days, TOTAL_DAYS)
        wlen = max(w_end_day - w_start_day + 1, 1)

        # Create window environment with bank filtering
        if num_banks == 4:
            window_env = LexicographicRewardEnvironment(
                facility, sub_df,
                window_length=wlen, use_windowing=True,
                randomize_order=False,
                use_deferred_actions=use_dap, use_fpr=use_fpr)
        else:
            window_env = BankFilteredEnvironment(
                facility, sub_df, num_banks=num_banks,
                window_length=wlen, use_windowing=True,
                randomize_order=False,
                use_deferred_actions=use_dap, use_fpr=use_fpr)

        # Reset, then carry over state from previous windows
        window_env.reset()
        window_env.cell_occupancy = full_env.cell_occupancy.copy()
        window_env.assignments = copy.deepcopy(full_env.assignments)
        window_env.assigned_servers = set()

        # Sort servers by ordering strategy
        if sort_fn is not None:
            window_env.servers.sort(key=sort_fn)
            priority_sorted = sorted(window_env.servers,
                                     key=lambda s: (s['DueTime'], s['ArrivalTime']))
            for rank, s in enumerate(priority_sorted):
                s['priority_rank'] = rank
                s['priority_cost'] = (len(window_env.servers) - rank) * 100000

        # Get state with updated occupancy
        state = window_env._get_state()

        # Run RL agent
        done = False
        while not done:
            if use_dap:
                valid_actions = window_env.get_valid_actions_deferred()
            else:
                valid_actions = window_env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            state, _, done, _, info = window_env.step(action)

        # Count this window's placements
        rl_assigned_this = len(window_env.assigned_servers)
        total_rl_only += rl_assigned_this
        total_assigned += rl_assigned_this
        total_tardy += _compute_tardiness(window_env)

        # Update full env with this window's final state
        full_env.cell_occupancy = window_env.cell_occupancy.copy()
        full_env.assignments = copy.deepcopy(window_env.assignments)

        # Carry-over: unassigned servers that still have time left
        assigned_k_ids = set(window_env.assignments.keys())
        new_carryover = []
        for global_idx in all_indices:
            server_k = int(df.iloc[global_idx]['k'])
            if server_k not in assigned_k_ids:
                arr = int(df.iloc[global_idx]['ArrivalTime'])
                pt = int(df.iloc[global_idx]['PTk'])
                if arr + pt <= TOTAL_DAYS + 1:
                    new_carryover.append(global_idx)
        carryover_indices = new_carryover

    return {
        'total_assigned': total_assigned,
        'total_rl_only': total_rl_only,
        'total_tardy': total_tardy,
        'n_windows': active_windows,
        'full_env': full_env,
    }


# ============================================================================
# Model loading (cached)
# ============================================================================

def load_model_cached(num_banks=4, device_key='cpu'):
    """Load the trained LHAC model (fresh load each time to avoid cache corruption)."""
    import torch
    facility = FacilityConfiguration()

    # Select per-bank fine-tuned model
    BANK_MODEL_MAP = {
        1: 'daf_full_daf_1bank_ft_best.pth',
        2: 'daf_full_daf_2bank_ft_best.pth',
        4: 'daf_full_daf.pth',
    }
    model_file = BANK_MODEL_MAP.get(num_banks, 'daf_full_daf.pth')
    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        # Fallback: search for any model
        model_path = None
        for root, dirs, files in os.walk(MODEL_DIR):
            for f in files:
                if f.endswith('.pt') or f.endswith('.pth'):
                    candidate = os.path.join(root, f)
                    if 'daf_full' in f.lower():
                        model_path = candidate
                        break
                    if model_path is None:
                        model_path = candidate
            if model_path:
                break

    if model_path is None:
        return None, facility, 'cpu', "No model found"

    # Detect dims
    dummy_df = pd.DataFrame({
        'k': [1], 'ArrivalTime': [1], 'DueTime': [65], 'PTk': [5],
        'OTk': [0], 'RH': [1], 'RL': [0], 'RW': [0]
    })
    dummy_env = LexicographicRewardEnvironment(
        facility, dummy_df, window_length=10, use_windowing=True,
        use_deferred_actions=False)
    model_state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n

    # Device selection: passed via device_key parameter (part of cache key)
    device = device_key

    # Detect checkpoint type
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    use_case = checkpoint.get('use_case', False)

    if 'agent1_network' in checkpoint:
        net_keys = checkpoint['agent1_network'].keys()
        if use_case or any('server_encoder' in k for k in net_keys):
            ckpt_state_dim = checkpoint.get('state_dim', 78)
            use_case = True
        else:
            first_key = [k for k in net_keys if 'encoder.0.weight' in k]
            ckpt_state_dim = checkpoint['agent1_network'][first_key[0]].shape[1] if first_key else model_state_dim
    else:
        ckpt_state_dim = model_state_dim
    del checkpoint

    agent, _ = load_agent(model_path, ckpt_state_dim, action_dim,
                          device=torch.device(device), use_case=use_case)

    # Wrap with truncation for DAP
    if ckpt_state_dim < 78:
        agent = StateTruncatingAgent(agent, ckpt_state_dim)

    return agent, facility, device, None


# ============================================================================
# Dataset helpers
# ============================================================================

def find_dataset_path(rel_path):
    """Try multiple locations for benchmark datasets."""
    for base in [DATA_DIR, PARVEZ_DATA_DIR,
                 os.path.join(BASE_DIR, 'Data', 'benchmarks')]:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None


def generate_random_dataset(n_servers, twotc_pct, arrival_pattern, hv_pct,
                            lv_pct, water_pct, avg_pt, due_tightness, seed):
    """Generate a random dataset matching benchmark format."""
    rng = np.random.RandomState(seed)

    horizon = 65  # days

    # Arrival times based on pattern
    if arrival_pattern == 'Uniform':
        arrivals = rng.uniform(1, horizon * 0.7, n_servers).astype(int)
    elif arrival_pattern == 'Right-skewed':
        # More arrivals later
        arrivals = (rng.beta(2, 5, n_servers) * horizon * 0.7 + 1).astype(int)
    else:  # Left-skewed
        # More arrivals earlier
        arrivals = (rng.beta(5, 2, n_servers) * horizon * 0.7 + 1).astype(int)

    arrivals = np.clip(arrivals, 1, horizon - 5)
    arrivals.sort()

    # Processing times
    pts = rng.poisson(avg_pt, n_servers)
    pts = np.clip(pts, 2, 15)

    # OTk (2TC)
    n_2tc = int(n_servers * twotc_pct / 100)
    otk = np.zeros(n_servers, dtype=int)
    idx_2tc = rng.choice(n_servers, n_2tc, replace=False)
    otk[idx_2tc] = 1

    # Resource requirements
    rh = (rng.random(n_servers) < hv_pct / 100).astype(int)
    rl = np.zeros(n_servers, dtype=int)
    # Servers without HV may need LV
    no_hv = rh == 0
    rl[no_hv] = (rng.random(no_hv.sum()) < lv_pct / 100).astype(int)
    rw = (rng.random(n_servers) < water_pct / 100).astype(int)

    # Due times
    if due_tightness == 'Tight':
        lead_time = rng.uniform(15, 22, n_servers)
    else:
        lead_time = rng.uniform(20, 30, n_servers)
    due_times = (arrivals + pts + lead_time).astype(int)
    due_times = np.clip(due_times, arrivals + pts + 1, horizon + 10)

    df = pd.DataFrame({
        'k': np.arange(1, n_servers + 1),
        'ArrivalTime': arrivals,
        'DueTime': due_times,
        'PTk': pts,
        'OTk': otk,
        'RH': rh,
        'RL': rl,
        'RW': rw,
        'type': rng.choice(['artemis', 'athena', 'themis'], n_servers),
    })

    return df


# ============================================================================
# Scheduling runners
# ============================================================================

def run_lhac_scheduling(agent, facility, df, mode, window_key, num_banks,
                         progress_callback=None, eval_mode='full'):
    """Run LHAC (RL-only or Hybrid) with bank filtering.

    Args:
        eval_mode: 'full' for rolling window evaluation (matches batch CSV results),
                   'quick' for single-pass preview (faster but less accurate).
    """
    wlen = WINDOW_CONFIGS[window_key]['window_length']

    if eval_mode == 'quick':
        # ── Single-pass evaluation (fast preview) ───────────────────
        best_result = None
        best_env = None
        best_name = None

        for idx, (name, sort_fn) in enumerate(ORDERING_STRATEGIES):
            if progress_callback:
                progress_callback(idx / len(ORDERING_STRATEGIES) * 0.7,
                                  f"RL ordering: {name}")

            # Create bank-filtered environment
            if num_banks < 4:
                env = BankFilteredEnvironment(
                    facility, df, num_banks=num_banks,
                    window_length=wlen, use_windowing=True,
                    randomize_order=False, use_deferred_actions=True,
                    use_fpr=True)
            else:
                env = LexicographicRewardEnvironment(
                    facility, df, window_length=wlen, use_windowing=True,
                    randomize_order=False, use_deferred_actions=True,
                    use_fpr=True)

            if sort_fn is not None:
                env.servers.sort(key=sort_fn)
                for rank, s in enumerate(sorted(env.servers,
                                                key=lambda s: (s['DueTime'], s['ArrivalTime']))):
                    s['priority_rank'] = rank
                    s['priority_cost'] = (len(env.servers) - rank) * 100000

            state, _ = env.reset()
            if sort_fn is not None:
                env.servers.sort(key=sort_fn)
                for rank, s in enumerate(sorted(env.servers,
                                                key=lambda s: (s['DueTime'], s['ArrivalTime']))):
                    s['priority_rank'] = rank
                    s['priority_cost'] = (len(env.servers) - rank) * 100000
                state = env._get_state()

            done = False
            while not done:
                valid_actions = env.get_valid_actions_deferred()
                action = agent.select_action(state, valid_actions, training=False)
                state, _, done, _, info = env.step(action)

            original_count = len(env.servers) - len(getattr(env, '_retry_queue', []))
            result = {
                'total_servers': original_count,
                'assigned': min(len(env.assignments), original_count),
                'rl_only_completion': info['completion_rate'],
                'rl_only_assigned': info['assigned_servers'],
            }
            result['completion_rate'] = result['assigned'] / result['total_servers'] if result['total_servers'] > 0 else 0
            result['unfinished'] = result['total_servers'] - result['assigned']
            result['tardy'] = _compute_tardiness(env)

            if best_result is None or result['assigned'] > best_result['assigned']:
                best_result = result
                best_name = name
                best_env = env

        if progress_callback:
            progress_callback(0.75, "RL complete")

    else:
        # ── Rolling window evaluation (matches batch CSV results) ───
        best_rolling = None
        best_env = None
        best_name = None
        total_orderings = len(ORDERING_STRATEGIES)

        if window_key in SUB_DAY_WINDOWS:
            # Sub-day windows: use hourly partitioning
            windows, df_h_sorted, whours = partition_servers_subday(df, window_key)

            for idx, (name, sort_fn) in enumerate(ORDERING_STRATEGIES):
                rolling_res = _run_ordering_rolling_subday(
                    agent, facility, df, windows, df_h_sorted, whours,
                    sort_fn, num_banks=num_banks,
                    progress_callback=progress_callback,
                    ordering_idx=idx, total_orderings=total_orderings)

                if best_rolling is None or rolling_res['total_assigned'] > best_rolling['total_assigned']:
                    best_rolling = rolling_res
                    best_name = name
                    best_env = rolling_res['full_env']
        else:
            # Day-level windows: use existing day partitioning
            window_days = wlen
            windows = partition_servers_by_day(df, window_days)

            for idx, (name, sort_fn) in enumerate(ORDERING_STRATEGIES):
                rolling_res = _run_ordering_rolling(
                    agent, facility, df, windows, window_days,
                    sort_fn, num_banks=num_banks,
                    progress_callback=progress_callback,
                    ordering_idx=idx, total_orderings=total_orderings)

                if best_rolling is None or rolling_res['total_assigned'] > best_rolling['total_assigned']:
                    best_rolling = rolling_res
                    best_name = name
                    best_env = rolling_res['full_env']

        if progress_callback:
            progress_callback(0.75, "RL complete (rolling window)")

        total_servers = len(df)
        assigned = best_rolling['total_assigned']
        best_result = {
            'total_servers': total_servers,
            'assigned': assigned,
            'rl_only_completion': assigned / total_servers if total_servers > 0 else 0,
            'rl_only_assigned': best_rolling['total_rl_only'],
            'completion_rate': assigned / total_servers if total_servers > 0 else 0,
            'unfinished': total_servers - assigned,
            'tardy': best_rolling['total_tardy'],
            'n_windows': best_rolling['n_windows'],
        }

    # ── Hybrid CPLEX (common for both modes) ────────────────────────
    best_result['hybrid_cplex_called'] = False
    best_result['hybrid_additional_placed'] = 0
    best_result['hybrid_cplex_runtime_sec'] = 0.0

    if mode == 'hybrid' and best_env is not None and best_result['completion_rate'] < 1.0:
        if progress_callback:
            progress_callback(0.8, "Running CPLEX on residual...")
        try:
            import hybrid_cplex as hc
            from hybrid_cplex import run_hybrid_evaluation

            # Ensure assigned_servers is populated (needed for rolling window mode)
            if not hasattr(best_env, 'assigned_servers') or not best_env.assigned_servers:
                best_env.assigned_servers = set(best_env.assignments.keys())

            orig_nb = hc.NUM_BANKS
            orig_exc = hc.EXCLUDED_CELLS.copy()
            hc.NUM_BANKS = num_banks
            if num_banks < 4:
                hc.EXCLUDED_CELLS = set()

            try:
                tardy_rate = best_result['tardy'] / max(best_result['assigned'], 1)
                hybrid_res = run_hybrid_evaluation(
                    best_env, facility,
                    completion_rate=best_result['completion_rate'],
                    tardiness_rate=tardy_rate,
                    threshold_completion=1.0,
                    threshold_tardiness=0.05,
                    cplex_time_limit=300,
                    verbose=False)

                if hybrid_res and hybrid_res['cplex_called']:
                    oc = best_result['total_servers']
                    fa = min(len(best_env.assignments), oc)
                    best_result['assigned'] = fa
                    best_result['completion_rate'] = fa / oc if oc > 0 else 0
                    best_result['unfinished'] = oc - fa
                    best_result['tardy'] = _compute_tardiness(best_env)
                    best_result['hybrid_cplex_called'] = True
                    best_result['hybrid_additional_placed'] = hybrid_res['cplex_additional_placed']
                    best_result['hybrid_cplex_runtime_sec'] = hybrid_res['cplex_runtime_sec']
            finally:
                hc.NUM_BANKS = orig_nb
                hc.EXCLUDED_CELLS = orig_exc
        except ImportError:
            pass

    best_result['best_ordering'] = best_name
    if progress_callback:
        progress_callback(1.0, "Done!")
    return best_result, best_env


def run_parvez_scheduling(df, window_key, num_banks, progress_callback=None):
    """Run Parvez 2-phase GA+CPLEX."""
    from run_parvez_multibank import run_parvez_single

    if progress_callback:
        progress_callback(0.1, "Starting Parvez GA+CPLEX...")

    # Select bank-specific input files
    if num_banks == 4:
        cell_bank_path = os.path.join(INPUT_DIR, 'CELL_BANK.xlsx')
        block_list_path = os.path.join(INPUT_DIR, 'block_list.xlsx')
    else:
        cell_bank_path = os.path.join(INPUT_DIR, f'CELL_BANK_{num_banks}bank.xlsx')
        block_list_path = os.path.join(INPUT_DIR, f'block_list_{num_banks}bank.xlsx')

    # Save dataset to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        df.to_excel(tmp.name, index=False)
        temp_path = tmp.name

    try:
        result = run_parvez_single(
            dataset_path=temp_path,
            cell_bank_path=cell_bank_path,
            block_list_path=block_list_path,
            window_key=window_key,
            seed=42,
            output_dir=None,
            total_run_time=259200,
            max_iteration_run_time=3600,
            verbose=False,
            num_banks=num_banks)

        if progress_callback:
            progress_callback(1.0, "Parvez complete!")

        # Convert to consistent format
        out = {
            'total_servers': result['n_orders'],
            'assigned': result['n_orders'] - result['unfinished'],
            'unfinished': result['unfinished'],
            'tardy': result['tardy'],
            'completion_rate': result['completion_pct'] / 100.0,
            'rl_only_completion': 0,
            'rl_only_assigned': 0,
            'runtime_sec': result['runtime_sec'],
            'n_windows': result['n_windows'],
            'best_ordering': 'N/A',
            'hybrid_cplex_called': False,
            'hybrid_additional_placed': 0,
            'hybrid_cplex_runtime_sec': 0,
        }
        return out, None  # No env for Parvez
    finally:
        os.unlink(temp_path)


# ============================================================================
# Visualization helpers
# ============================================================================

def build_gantt_data(env, num_banks=4):
    """Extract Gantt data from LHAC environment."""
    if env is None:
        return pd.DataFrame()

    rows = []
    idx_to_bc = {}
    for ci in range(4 * NUM_CELLS_PER_BANK):  # Full 4-bank mapping
        bank = ci // NUM_CELLS_PER_BANK + 1
        pos = ci % NUM_CELLS_PER_BANK + 1
        idx_to_bc[ci] = (bank, pos)

    for server_id, asn in env.assignments.items():
        if 'bank' in asn and 'cell' in asn:
            bank = asn['bank']
            cell = asn['cell']
        elif 'cell_idx' in asn:
            bank, cell = idx_to_bc.get(asn['cell_idx'], (0, 0))
        else:
            continue

        if bank > num_banks:
            continue

        start = asn['start']
        end = asn['end']

        server_data = None
        for s in env.servers:
            if s['k'] == server_id:
                server_data = s
                break

        is_2tc = server_data.get('OTk', 0) == 1 if server_data else False
        due = server_data.get('DueTime', TOTAL_DAYS) if server_data else TOTAL_DAYS
        arrival = server_data.get('ArrivalTime', 1) if server_data else 1

        rows.append({
            'server_id': server_id, 'bank': bank, 'cell': cell,
            'start': start, 'end': end, 'duration': end - start + 1,
            'is_2tc': is_2tc, 'is_tardy': end > due,
            'arrival': arrival, 'due': due,
        })

    return pd.DataFrame(rows)


def build_gantt_chart(gantt_df, num_banks, selected_bank=None):
    """Build Plotly Gantt chart."""
    if gantt_df.empty:
        return go.Figure()

    excluded = BANK_CONFIGS[num_banks]['excluded']

    if selected_bank and selected_bank != "All Banks":
        bank_num = int(selected_bank.split(" ")[1])
        df = gantt_df[gantt_df['bank'] == bank_num].copy()
    else:
        df = gantt_df.copy()

    if df.empty:
        return go.Figure()

    fig = go.Figure()
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    ]

    y_labels, y_map = [], {}
    y_idx = 0
    for bank in range(1, num_banks + 1):
        for cell in range(1, NUM_CELLS_PER_BANK + 1):
            if (bank, cell) in excluded:
                continue
            label = f"B{bank}-C{cell:02d}"
            y_labels.append(label)
            y_map[(bank, cell)] = y_idx
            y_idx += 1

    for _, row in df.iterrows():
        key = (row['bank'], row['cell'])
        if key not in y_map:
            continue
        y_pos = y_map[key]
        color = colors[row['server_id'] % len(colors)]
        border = '#d62728' if row['is_tardy'] else color

        hover = (f"Server {row['server_id']}<br>"
                 f"Bank {row['bank']}, Cell {row['cell']}<br>"
                 f"Days {row['start']}-{row['end']} ({row['duration']}d)<br>"
                 f"{'2TC' if row['is_2tc'] else '1TC'}"
                 f"{'<br><b>TARDY</b>' if row['is_tardy'] else ''}")

        fig.add_trace(go.Bar(
            x=[row['end'] - row['start'] + 1], y=[y_pos],
            base=[row['start'] - 1], orientation='h',
            marker=dict(color=color, line=dict(color=border, width=1.5 if row['is_tardy'] else 0.5)),
            hovertext=hover, hoverinfo='text', showlegend=False,
            text=str(row['server_id']), textposition='inside',
            textfont=dict(size=7, color='white'),
        ))

        if row['is_2tc']:
            adj_key = (row['bank'], row['cell'] + 1)
            if adj_key in y_map:
                fig.add_trace(go.Bar(
                    x=[row['end'] - row['start'] + 1], y=[y_map[adj_key]],
                    base=[row['start'] - 1], orientation='h',
                    marker=dict(color=color, opacity=0.5),
                    hovertext=f"Server {row['server_id']} (2TC adj)",
                    hoverinfo='text', showlegend=False,
                ))

    for week in range(1, 14):
        day = week * 5
        if day <= TOTAL_DAYS:
            fig.add_vline(x=day, line_width=0.5, line_dash="dot",
                          line_color="rgba(0,0,0,0.2)")

    fig.update_layout(
        title="Server Scheduling Gantt Chart",
        xaxis=dict(title="Day", range=[0, TOTAL_DAYS + 1], dtick=5),
        yaxis=dict(tickvals=list(range(len(y_labels))), ticktext=y_labels,
                   autorange='reversed', title="Bank - Cell"),
        height=max(400, len(y_labels) * 22),
        barmode='overlay', bargap=0.15, plot_bgcolor='white',
        margin=dict(l=80, r=20, t=50, b=40),
    )
    return fig


# ============================================================================
# Main Streamlit App
# ============================================================================

def main():
    st.set_page_config(page_title="Production Scheduling Dashboard",
                       page_icon="",
                       layout="wide")

    # Header row: title on left, sponsor logo on right
    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.title("Production Scheduling Dashboard")
    with header_right:
        nsf_path = os.path.join(IMAGES_DIR, 'NSF.jpg')
        if os.path.exists(nsf_path):
            st.image(nsf_path, width=130)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Method
        method = st.radio(
            "Scheduling Method",
            ["LHAC (RL Only)", "Hybrid (RL + CPLEX)"],
            index=1,
            help=("**LHAC** (Lexicographic Hierarchical Actor Critic): "
                  "RL with DAP/CASE/FPR\n\n"
                  "**Hybrid**: LHAC RL + CPLEX residual")
        )

        st.divider()

        # Window
        window_key = st.radio(
            "Lookahead Window",
            ['10d', '5d', '1d'],
            format_func=lambda x: WINDOW_CONFIGS[x]['label'],
            index=0,
            help="Scheduling horizon per iteration."
        )

        st.divider()

        # Banks
        num_banks = st.radio(
            "Number of Banks",
            [4, 2, 1],
            format_func=lambda x: BANK_CONFIGS[x]['label'],
            index=0,
            help="Restrict to a subset of test banks"
        )

        st.divider()

        # Evaluation mode
        eval_mode = st.radio(
            "Evaluation Mode",
            ['full', 'quick'],
            format_func=lambda x: {
                'full': 'Full Evaluation (rolling window)',
                'quick': 'Quick Preview (single pass)',
            }[x],
            index=0,
            help=("**Full Evaluation**: Proper rolling window with carry-over -- "
                  "matches batch CSV results (0 tardiness). "
                  "Runtime: 10d ~1-2min, 5d ~2-5min, 1d ~5-50min.\n\n"
                  "**Quick Preview**: All servers in one pass -- faster "
                  "but may show tardy servers and differ from CSV results.")
        )
        st.divider()

        # Inference device
        import torch as _torch
        gpu_available = _torch.cuda.is_available()
        if gpu_available:
            device_choice = st.radio(
                "Inference Device",
                ['GPU', 'CPU'],
                index=0,
                help=("**GPU**: Fast (~1 min). May be slow if training "
                      "processes are using the GPU.\n\n"
                      "**CPU**: Slower (~5 min) but always reliable.")
            )
            st.session_state['inference_device'] = 'cuda' if device_choice == 'GPU' else 'cpu'
        else:
            st.session_state['inference_device'] = 'cpu'
            st.info("GPU not available — using CPU.")
        st.divider()

        # Dataset source
        st.subheader("Dataset")
        dataset_tab = st.radio(
            "Source",
            ["Benchmark", "Upload Custom", "Generate Random"],
            index=0
        )

    # Main area - dataset selection
    dataset_df = None
    dataset_name = None

    if dataset_tab == "Benchmark":
        options = [(f"{d} | {n} servers | {p} 2TC", rel)
                   for d, n, p, rel in TABLE6_DATASETS]
        selected = st.selectbox("Select benchmark dataset", options,
                                format_func=lambda x: x[0], index=0)
        if selected:
            path = find_dataset_path(selected[1])
            if path:
                dataset_df = pd.read_excel(path)
                dataset_name = selected[0]
            else:
                st.error(f"Dataset not found: {selected[1]}")

    elif dataset_tab == "Upload Custom":
        uploaded = st.file_uploader(
            "Upload .xlsx", type=['xlsx'],
            help="Columns: k, ArrivalTime, DueTime, PTk, OTk, RH, RL, RW")
        if uploaded:
            dataset_df = pd.read_excel(uploaded)
            dataset_name = uploaded.name

    else:  # Generate Random
        st.subheader("Random Dataset Generator")
        col1, col2 = st.columns(2)
        with col1:
            n_servers = st.slider("Number of servers", 50, 500, 200, 10)
            twotc_pct = st.slider("2TC percentage", 0, 50, 10, 5)
            arrival_pattern = st.radio("Arrival pattern",
                                       ["Uniform", "Right-skewed", "Left-skewed"])
            gen_seed = st.number_input("Random seed", value=42, min_value=0)
        with col2:
            hv_pct = st.slider("High voltage %", 0, 100, 60, 5)
            lv_pct = st.slider("Low voltage %", 0, 100, 40, 5)
            water_pct = st.slider("Water cooling %", 0, 100, 30, 5)
            avg_pt = st.slider("Avg processing time (days)", 3, 10, 5)
            due_tight = st.radio("Due date tightness", ["Tight", "Loose"])

        if st.button("Generate Dataset", type="secondary"):
            dataset_df = generate_random_dataset(
                n_servers, twotc_pct, arrival_pattern,
                hv_pct, lv_pct, water_pct, avg_pt, due_tight, gen_seed)
            dataset_name = f"Random {n_servers}srv {twotc_pct}%2TC {arrival_pattern}"
            st.session_state['generated_df'] = dataset_df
            st.session_state['generated_name'] = dataset_name

        if 'generated_df' in st.session_state:
            dataset_df = st.session_state['generated_df']
            dataset_name = st.session_state['generated_name']

    # Dataset preview
    if dataset_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Servers", len(dataset_df))
        with c2:
            n2tc = dataset_df['OTk'].sum() if 'OTk' in dataset_df.columns else 0
            st.metric("2TC Servers", f"{n2tc} ({n2tc/len(dataset_df)*100:.0f}%)")
        with c3:
            method_short = method.split(" (")[0]
            st.metric("Method", method_short)
        with c4:
            st.metric("Config",
                       f"{WINDOW_CONFIGS[window_key]['label']} / {num_banks} bank{'s' if num_banks > 1 else ''}")

        with st.expander("Preview Dataset", expanded=False):
            st.dataframe(dataset_df.head(20), use_container_width=True, height=300)

        # Run button
        if st.button("Run Scheduling", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing...")
            start_time = time.time()

            def update_progress(pct, msg):
                progress_bar.progress(min(pct, 1.0), text=msg)

            try:
                if 'Parvez' in method:
                    with st.spinner("Running Parvez GA+CPLEX (may take minutes)..."):
                        result, env = run_parvez_scheduling(
                            dataset_df, window_key, num_banks, update_progress)
                else:
                    inf_device = st.session_state.get('inference_device', 'cpu')
                    agent, facility, device, err = load_model_cached(num_banks, device_key=inf_device)
                    if err:
                        st.error(f"Model error: {err}")
                        return

                    mode_key = 'hybrid' if 'Hybrid' in method else 'rl_only'
                    spinner_msg = "Running LHAC (rolling window)..." if eval_mode == 'full' else "Running LHAC..."
                    with st.spinner(spinner_msg):
                        result, env = run_lhac_scheduling(
                            agent, facility, dataset_df, mode_key,
                            window_key, num_banks, update_progress,
                            eval_mode=eval_mode)

                elapsed = time.time() - start_time
                progress_bar.progress(1.0, text="Complete!")

                st.session_state['result'] = result
                st.session_state['env'] = env
                st.session_state['elapsed'] = elapsed
                st.session_state['method'] = method
                st.session_state['num_banks'] = num_banks
                st.session_state['window_key'] = window_key
                st.session_state['gantt_df'] = build_gantt_data(env, num_banks) if env else pd.DataFrame()

            except Exception as e:
                progress_bar.progress(1.0, text="Error!")
                st.error(f"Scheduling failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display results
    if 'result' in st.session_state:
        result = st.session_state['result']
        elapsed = st.session_state['elapsed']
        num_banks_r = st.session_state.get('num_banks', 4)
        gantt_df = st.session_state.get('gantt_df', pd.DataFrame())

        st.divider()
        st.header("Scheduling Results")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            comp = result['completion_rate']
            st.metric("Completion Rate", f"{comp:.1%}")
        with c2:
            st.metric("Servers Placed",
                       f"{result['assigned']}/{result['total_servers']}")
        with c3:
            st.metric("Unplaced", result.get('unfinished', 0))
        with c4:
            st.metric("Tardy", result.get('tardy', 0))
        with c5:
            if elapsed < 60:
                st.metric("Runtime", f"{elapsed:.1f} sec")
            else:
                st.metric("Runtime", f"{elapsed / 60:.1f} min")

        # Method-specific info
        col_a, col_b = st.columns(2)
        with col_a:
            if result.get('rl_only_completion', 0) > 0:
                st.markdown(f"**RL-only completion**: {result['rl_only_completion']:.1%}")
            if result.get('best_ordering', 'N/A') != 'N/A':
                st.markdown(f"**Best ordering**: {result['best_ordering']}")
        with col_b:
            if result.get('hybrid_cplex_called', False):
                st.markdown(f"**CPLEX additional**: +{result['hybrid_additional_placed']} servers")
                cplex_sec = result['hybrid_cplex_runtime_sec']
                if cplex_sec < 60:
                    st.markdown(f"**CPLEX runtime**: {cplex_sec:.1f} sec")
                else:
                    st.markdown(f"**CPLEX runtime**: {cplex_sec / 60:.1f} min")
            if result.get('n_windows', 0) > 0:
                st.markdown(f"**Windows processed**: {result['n_windows']}")

        # Gantt chart (only for LHAC methods that have env)
        if not gantt_df.empty:
            st.divider()
            st.header("Gantt Chart")

            excluded = BANK_CONFIGS[num_banks_r]['excluded']
            bank_options = ["All Banks"] + [f"Bank {i} ({BANK_NAMES.get(i, '')})"
                                             for i in range(1, num_banks_r + 1)]
            sel_bank = st.selectbox("Filter by bank", bank_options, index=0)

            fig = build_gantt_chart(gantt_df, num_banks_r, sel_bank)
            st.plotly_chart(fig, use_container_width=True)

            # Bank statistics
            st.divider()
            st.header("Bank-Level Statistics")
            bank_stats = []
            for bank in range(1, num_banks_r + 1):
                bdf = gantt_df[gantt_df['bank'] == bank]
                n_cells = NUM_CELLS_PER_BANK - sum(
                    1 for c in range(1, NUM_CELLS_PER_BANK + 1)
                    if (bank, c) in excluded)
                bank_stats.append({
                    'Bank': f"Bank {bank} ({BANK_NAMES.get(bank, '')})",
                    'Cells': n_cells,
                    'Servers': len(bdf),
                    '2TC': bdf['is_2tc'].sum() if len(bdf) > 0 else 0,
                    'Tardy': bdf['is_tardy'].sum() if len(bdf) > 0 else 0,
                    'Avg Duration': f"{bdf['duration'].mean():.1f}d" if len(bdf) > 0 else "N/A",
                })
            st.dataframe(pd.DataFrame(bank_stats), use_container_width=True, hide_index=True)

            # Cell utilization
            st.divider()
            st.header("Cell Utilization")
            cell_labels, util_data = [], []
            for bank in range(1, num_banks_r + 1):
                for cell in range(1, NUM_CELLS_PER_BANK + 1):
                    if (bank, cell) in excluded:
                        continue
                    cell_labels.append(f"B{bank}-C{cell:02d}")
                    srvs = gantt_df[(gantt_df['bank'] == bank) & (gantt_df['cell'] == cell)]
                    occ = set()
                    for _, r in srvs.iterrows():
                        for d in range(r['start'], r['end'] + 1):
                            occ.add(d)
                    util_data.append(len(occ) / TOTAL_DAYS * 100)

            fig_h = go.Figure(data=go.Bar(
                x=cell_labels, y=util_data,
                marker_color=['#2ca02c' if u > 80 else '#ff7f0e' if u > 50 else '#d62728'
                              for u in util_data],
            ))
            fig_h.update_layout(
                title="Cell Utilization (%)",
                xaxis=dict(title="Cell", tickangle=45),
                yaxis=dict(title="%", range=[0, 105]),
                height=350, margin=dict(l=50, r=20, t=50, b=80),
            )
            st.plotly_chart(fig_h, use_container_width=True)

            # Detailed table
            with st.expander("Detailed Assignment Table", expanded=False):
                display_df = gantt_df[['server_id', 'bank', 'cell', 'start', 'end',
                                       'duration', 'is_2tc', 'is_tardy', 'arrival', 'due'
                                       ]].sort_values(['bank', 'cell', 'start']).reset_index(drop=True)
                display_df.columns = ['Server ID', 'Bank', 'Cell', 'Start', 'End',
                                      'Duration', '2TC', 'Tardy', 'Arrival', 'Due']
                st.dataframe(display_df, use_container_width=True, height=400)
        else:
            if 'Parvez' in st.session_state.get('method', ''):
                st.info("Gantt chart not available for Parvez method (uses separate solver).")

    else:
        st.info("Select a dataset and click **Run Scheduling** to begin.")


if __name__ == '__main__':
    main()
