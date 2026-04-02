#!/usr/bin/env python3
"""
DAF-LHAC Evaluation Pipeline.

Unified evaluation script supporting:
  - Full-horizon and rolling-window evaluation
  - All 6 lookahead windows: 10d, 5d, 1d, 8h, 4h, 0h
  - DAP (Deferred Action Placement) — embeds repair inside RL action selection
  - CASE (Cross-Attention Cell-Server Encoder) network
  - FPR (Feasibility-Preserving Reward) shaping
  - Multi-seed evaluation
  - Multi-ordering inference (best of 5 server orderings)
  - Hybrid LHAC-DAF + CPLEX fallback for residual unplaced servers
  - Parvez Table 6 comparison format

NOTE: When DAP is enabled, greedy post-processing repair is automatically skipped.
DAP's get_valid_actions_deferred() already scans ALL feasible start times per cell,
effectively embedding repair logic INTO the RL agent's action selection.
The pipeline is: RL (DAP+CASE+FPR) → CPLEX if needed. No repair in between.

Usage:
  # LHAC-DAF evaluation (DAP replaces repair)
  python evaluate.py --model Models/daf_full_daf_ckpt_2000.pth --dap --fpr --variant full_daf --windows 10d

  # Hybrid LHAC-DAF + CPLEX (CPLEX fills in any gaps)
  python evaluate.py --model Models/daf_full_daf_ckpt_2000.pth --dap --fpr --variant full_daf --windows 10d --hybrid --hybrid_completion_threshold 1.0

  # Old LHAC baseline (no DAP, with greedy repair)
  python evaluate.py --model Models/lhac_ppo_v1_best.pth --windows 10d
"""

import os
import sys
import copy
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch

# Import from DAF-LHAC core (self-contained)
from daf_lhac_core import (
    FacilityConfiguration, LexicographicRewardEnvironment,
    ImprovedLexicographicDQNAgent, MultiAgentLHACPPO,
    CellStatus, VoltageType,
    TOTAL_TIME_BLOCKS, NUM_BANKS, DATA_DIR, MODEL_DIR
)


# ============================================================================
# Constants
# ============================================================================
HOURS_PER_DAY = 24
TOTAL_DAYS = 65  # 13 weeks × 5 days

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARVEZ_DATA_ROOT = os.path.join(os.path.dirname(_BASE_DIR), '..', 'Parvez', 'Data_datasets')
RESULTS_DIR = os.path.join(_BASE_DIR, 'Results')

# Window configs for rolling-window evaluation (hours)
WINDOW_HOURS = {
    '10d': ('10 days', 10 * 24),
    '5d':  ('5 days',  5 * 24),
    '1d':  ('1 day',   1 * 24),
    '8h':  ('8 hours', 8),
    '4h':  ('4 hours', 4),
    '0h':  ('Real-time', 0),
}

# Window configs for day-level evaluation
WINDOW_BLOCKS = {
    '10d': ('10 days', 10),
    '5d':  ('5 days', 5),
    '1d':  ('1 day', 1),
}

DAY_LEVEL_WINDOWS = {'10d', '5d', '1d'}
SUB_DAY_WINDOWS = {'8h', '4h', '0h'}
ALL_WINDOWS = ['10d', '5d', '1d', '8h', '4h', '0h']
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


# ============================================================================
# Parvez Table 6 reference values
# ============================================================================
PARVEZ_TABLE6 = {
    ('Uniform', 200, '10%'):  (0, 0, 0.05, 2.38),
    ('Uniform', 200, '20%'):  (0, 0, None, None),
    ('Uniform', 200, '30%'):  (0, 0, 0.79, 2.52),
    ('Uniform', 300, '10%'):  (0, 0, 0.39, 4.09),
    ('Uniform', 300, '20%'):  (0, 0, None, None),
    ('Uniform', 300, '30%'):  (0, 0, 216.58, 4.45),
    ('Uniform', 400, '10%'):  (0, 0, 325.86, 92.09),
    ('Uniform', 400, '20%'):  (None, 0, None, None),
    ('Uniform', 400, '30%'):  (140, 0, 840.00, 190.85),
    ('Right-skewed', 200, '10%'): (None, 0, None, None),
    ('Right-skewed', 200, '20%'): (None, 0, None, None),
    ('Right-skewed', 200, '30%'): (None, 0, None, None),
    ('Right-skewed', 300, '10%'): (0, 0, 244.66, 63.42),
    ('Right-skewed', 300, '20%'): (None, 0, None, None),
    ('Right-skewed', 300, '30%'): (0, 0, 610.78, 50.39),
    ('Right-skewed', 400, '10%'): (165, 0, 840.00, 287.23),
    ('Right-skewed', 400, '20%'): (None, 0, None, None),
    ('Right-skewed', 400, '30%'): (281, 0, 840.00, 276.62),
    ('Left-skewed', 200, '10%'): (None, 0, None, None),
    ('Left-skewed', 200, '20%'): (None, 0, None, None),
    ('Left-skewed', 200, '30%'): (None, 0, None, None),
    ('Left-skewed', 300, '10%'): (None, 0, None, None),
    ('Left-skewed', 300, '20%'): (None, 0, None, None),
    ('Left-skewed', 300, '30%'): (None, 0, None, None),
    ('Left-skewed', 400, '10%'): (None, 0, None, None),
    ('Left-skewed', 400, '20%'): (None, 0, None, None),
    ('Left-skewed', 400, '30%'): (None, 0, None, None),
}


# ============================================================================
# Dataset mapping (Parvez's actual datasets)
# ============================================================================
TABLE6_DATASETS = [
    ('Uniform', 200, '10%', 'Uniform/200_orders_10%_uni.xlsx'),
    ('Uniform', 200, '20%', 'Uniform/200_orders_20%_uni.xlsx'),
    ('Uniform', 200, '30%', 'Uniform/200_orders_30%_uni.xlsx'),
    ('Uniform', 300, '10%', 'Uniform/300_orders_10%_uni.xlsx'),
    ('Uniform', 300, '20%', 'Uniform/300_orders_uniform_20%.xlsx'),
    ('Uniform', 300, '30%', 'Uniform/300_orders_30%_uni.xlsx'),
    ('Uniform', 400, '10%', 'Uniform/400_orders_10%_uni.xlsx'),
    ('Uniform', 400, '20%', 'Uniform/400_orders_uniform_20%.xlsx'),
    ('Uniform', 400, '30%', 'Uniform/400_orders_30%_uni.xlsx'),
    ('Left-skewed', 200, '10%', 'Back/200_orders_10%_bk.xlsx'),
    ('Left-skewed', 200, '20%', 'Back/200_orders_20%_bk.xlsx'),
    ('Left-skewed', 200, '30%', 'Back/200_orders_30%_bk.xlsx'),
    ('Left-skewed', 300, '10%', 'Back/300_orders_10%_bk.xlsx'),
    ('Left-skewed', 300, '20%', 'Back/300_orders_back_20%.xlsx'),
    ('Left-skewed', 300, '30%', 'Back/300_orders_30%_bk.xlsx'),
    ('Left-skewed', 400, '10%', 'Back/400_orders_10%_bk.xlsx'),
    ('Left-skewed', 400, '20%', 'Back/400_orders_back_20%.xlsx'),
    ('Left-skewed', 400, '30%', 'Back/400_orders_30%_bk.xlsx'),
    ('Right-skewed', 200, '10%', 'Front/200_orders_10%_fr.xlsx'),
    ('Right-skewed', 200, '20%', 'Front/200_orders_20%_fr.xlsx'),
    ('Right-skewed', 200, '30%', 'Front/200_orders_30%_fr.xlsx'),
    ('Right-skewed', 300, '10%', 'Front/300_orders_10%_fr.xlsx'),
    ('Right-skewed', 300, '20%', 'Front/300_orders_front_20%.xlsx'),
    ('Right-skewed', 300, '30%', 'Front/300_orders_30%_fr.xlsx'),
    ('Right-skewed', 400, '10%', 'Front/400_orders_10%_fr.xlsx'),
    ('Right-skewed', 400, '20%', 'Front/400_orders_front_20%.xlsx'),
    ('Right-skewed', 400, '30%', 'Front/400_orders_30%_fr.xlsx'),
]

# Quick test datasets (3 representative datasets for smoke testing)
QUICK_TEST_DATASETS = [
    ('Uniform', 200, '10%', 'Uniform/200_orders_10%_uni.xlsx'),
    ('Left-skewed', 300, '30%', 'Back/300_orders_30%_bk.xlsx'),
    ('Right-skewed', 400, '20%', 'Front/400_orders_front_20%.xlsx'),
]


# ============================================================================
# Agent loading
# ============================================================================

def load_agent(model_path, state_dim, action_dim, device, use_case=False):
    """Load agent with auto-detection of model type (DQN or Multi-Agent PPO).

    Args:
        model_path: Path to saved checkpoint
        state_dim: Expected state dimension
        action_dim: Action dimension
        device: Torch device
        use_case: Whether the model uses CASE (Cross-Attention) network
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    agent_type = checkpoint.get('agent_type', 'dqn')
    # Auto-detect CASE from checkpoint
    ckpt_use_case = checkpoint.get('use_case', False)
    if ckpt_use_case:
        use_case = True
    print(f"  Agent type: {agent_type}, CASE: {use_case}")

    if agent_type == 'multi_agent_ppo':
        agent = MultiAgentLHACPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            use_case=use_case)
    else:
        agent = ImprovedLexicographicDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device)
    agent.load(model_path)
    return agent, agent_type


class StateTruncatingAgent:
    """Wrapper that truncates state vectors before passing to the underlying agent.

    Used when the environment produces more state features (e.g., 78 with DAP)
    than the model was trained on (e.g., 74 without DAP). The extra features
    (dims 74-77: DAP features) are stripped before inference.
    """

    def __init__(self, agent, truncate_dim):
        self.agent = agent
        self.truncate_dim = truncate_dim

    def select_action(self, state, valid_actions, training=False):
        # Truncate state to model's expected dimension
        if hasattr(state, '__len__') and len(state) > self.truncate_dim:
            state = state[:self.truncate_dim]
        return self.agent.select_action(state, valid_actions, training=training)

    def __getattr__(self, name):
        return getattr(self.agent, name)


# ============================================================================
# Repair mechanisms
# ============================================================================

def _try_place_deferred(env, si, srv):
    """Try placing a server with deferred start (arrival through latest feasible).

    Tries start times from arrival through min(DueTime - PTk, total_time + 1 - PTk).
    Returns True if placed, False otherwise.
    """
    arrival = srv['ArrivalTime']
    pt = srv['PTk']
    is_2tc = srv['OTk'] == 1
    server_id = srv['k']

    if server_id in env.assignments:
        return False

    latest_start = min(srv['DueTime'] - pt, env.total_time + 1 - pt)
    if latest_start < arrival:
        latest_start = env.total_time + 1 - pt
    if latest_start < arrival:
        return False

    for start in range(arrival, latest_start + 1):
        end_time = start + pt
        if end_time > env.total_time + 1:
            break
        for ci in range(env.total_cells):
            if not env._is_cell_compatible(ci, srv):
                continue
            occ = env.cell_occupancy[ci, start:min(end_time, env.total_time + 1)]
            if np.any(occ != 0):
                continue

            if is_2tc:
                if ci not in env.adjacent_cells:
                    continue
                ai = env.adjacent_cells[ci]
                adj_occ = env.cell_occupancy[ai, start:min(end_time, env.total_time + 1)]
                if np.any(adj_occ != 0):
                    continue
                env.cell_occupancy[ci, start:min(end_time, env.total_time + 1)] = server_id
                env.cell_occupancy[ai, start:min(end_time, env.total_time + 1)] = server_id
            else:
                env.cell_occupancy[ci, start:min(end_time, env.total_time + 1)] = server_id

            env.assignments[server_id] = {
                'cell_idx': ci,
                'start': start,
                'end': end_time - 1,
            }
            env.assigned_servers.add(si)
            return True
    return False


def greedy_repair(env, max_passes=3):
    """Multi-pass greedy repair with deferred placement.

    Returns: (repair_count, original_count)
    """
    original_count = len(env.servers) - len(env._retry_queue)
    total_repair = 0

    for pass_num in range(max_passes):
        unassigned = [(si, env.servers[si]) for si in range(original_count)
                      if env.servers[si]['k'] not in env.assignments]
        if not unassigned:
            break

        if pass_num == 0:
            unassigned.sort(key=lambda p: (-p[1]['OTk'], p[1].get('slack', 999)))
        elif pass_num == 1:
            unassigned.sort(key=lambda p: (p[1].get('slack', 999), -p[1]['OTk']))
        else:
            unassigned.sort(key=lambda p: (-p[1]['PTk'], -p[1]['OTk'],
                                           p[1].get('slack', 999)))

        pass_placed = 0
        for si, srv in unassigned:
            if _try_place_deferred(env, si, srv):
                total_repair += 1
                pass_placed += 1
        if pass_placed == 0:
            break

    return total_repair, original_count


def displacement_repair(env, max_swaps=100):
    """Displacement repair: swap unfinished servers with placed ones.

    Returns number of additional servers placed via displacement.
    """
    original_count = len(env.servers) - len(env._retry_queue)
    displaced = 0

    for swap_round in range(max_swaps):
        unfinished = []
        for si in range(original_count):
            srv = env.servers[si]
            if srv['k'] not in env.assignments:
                unfinished.append((si, srv))
        if not unfinished:
            break

        made_progress = False
        unfinished.sort(key=lambda p: (p[1].get('slack', 999), -p[1]['OTk']))

        for uf_si, uf_srv in unfinished:
            if uf_srv['k'] in env.assignments:
                continue

            uf_arrival = uf_srv['ArrivalTime']
            uf_pt = uf_srv['PTk']
            uf_is_2tc = uf_srv['OTk'] == 1
            uf_id = uf_srv['k']

            for ci in range(env.total_cells):
                if not env._is_cell_compatible(ci, uf_srv):
                    continue

                if uf_is_2tc:
                    if ci not in env.adjacent_cells:
                        continue
                    ai = env.adjacent_cells[ci]
                else:
                    ai = None

                latest_start = min(uf_srv['DueTime'] - uf_pt, env.total_time + 1 - uf_pt)
                if latest_start < uf_arrival:
                    latest_start = env.total_time + 1 - uf_pt
                if latest_start < uf_arrival:
                    continue

                for uf_start in range(uf_arrival, latest_start + 1):
                    uf_end = uf_start + uf_pt
                    if uf_end > env.total_time + 1:
                        break

                    occ = env.cell_occupancy[ci, uf_start:min(uf_end, env.total_time + 1)]
                    blocking_ids = set(occ[occ > 0])

                    if ai is not None:
                        adj_occ = env.cell_occupancy[ai, uf_start:min(uf_end, env.total_time + 1)]
                        blocking_ids |= set(adj_occ[adj_occ > 0])

                    if len(blocking_ids) == 0:
                        if ai is not None:
                            env.cell_occupancy[ci, uf_start:min(uf_end, env.total_time + 1)] = uf_id
                            env.cell_occupancy[ai, uf_start:min(uf_end, env.total_time + 1)] = uf_id
                        else:
                            env.cell_occupancy[ci, uf_start:min(uf_end, env.total_time + 1)] = uf_id
                        env.assignments[uf_id] = {'cell_idx': ci, 'start': uf_start, 'end': uf_end - 1}
                        env.assigned_servers.add(uf_si)
                        displaced += 1
                        made_progress = True
                        break

                    if len(blocking_ids) > 2:
                        continue

                    all_displaced = True
                    displaced_info = []

                    for block_id in blocking_ids:
                        block_id = int(block_id)
                        if block_id not in env.assignments:
                            all_displaced = False
                            break

                        basn = env.assignments[block_id]
                        b_ci = basn['cell_idx']
                        b_start = basn['start']
                        b_end = basn['end'] + 1

                        b_srv = None
                        b_si = None
                        for bsi in range(original_count):
                            if env.servers[bsi]['k'] == block_id:
                                b_srv = env.servers[bsi]
                                b_si = bsi
                                break
                        if b_srv is None:
                            all_displaced = False
                            break

                        b_is_2tc = b_srv['OTk'] == 1

                        env.cell_occupancy[b_ci, b_start:min(b_end, env.total_time + 1)][
                            env.cell_occupancy[b_ci, b_start:min(b_end, env.total_time + 1)] == block_id] = 0
                        if b_is_2tc and b_ci in env.adjacent_cells:
                            b_ai = env.adjacent_cells[b_ci]
                            env.cell_occupancy[b_ai, b_start:min(b_end, env.total_time + 1)][
                                env.cell_occupancy[b_ai, b_start:min(b_end, env.total_time + 1)] == block_id] = 0

                        displaced_info.append((block_id, b_ci, b_start, b_end, b_si, b_srv, b_is_2tc))

                    if not all_displaced:
                        for d_id, d_ci, d_s, d_e, d_si, d_srv, d_2tc in displaced_info:
                            env.cell_occupancy[d_ci, d_s:min(d_e, env.total_time + 1)] = d_id
                            if d_2tc and d_ci in env.adjacent_cells:
                                d_ai = env.adjacent_cells[d_ci]
                                env.cell_occupancy[d_ai, d_s:min(d_e, env.total_time + 1)] = d_id
                        continue

                    slot_occ = env.cell_occupancy[ci, uf_start:min(uf_end, env.total_time + 1)]
                    slot_free = not np.any(slot_occ != 0)
                    if ai is not None:
                        adj_slot = env.cell_occupancy[ai, uf_start:min(uf_end, env.total_time + 1)]
                        slot_free = slot_free and not np.any(adj_slot != 0)

                    if slot_free:
                        if ai is not None:
                            env.cell_occupancy[ci, uf_start:min(uf_end, env.total_time + 1)] = uf_id
                            env.cell_occupancy[ai, uf_start:min(uf_end, env.total_time + 1)] = uf_id
                        else:
                            env.cell_occupancy[ci, uf_start:min(uf_end, env.total_time + 1)] = uf_id
                        env.assignments[uf_id] = {'cell_idx': ci, 'start': uf_start, 'end': uf_end - 1}
                        env.assigned_servers.add(uf_si)

                        all_replaced = True
                        for d_id, d_ci, d_s, d_e, d_si, d_srv, d_2tc in displaced_info:
                            del env.assignments[d_id]
                            env.assigned_servers.discard(d_si)
                            if not _try_place_deferred(env, d_si, d_srv):
                                all_replaced = False
                                break

                        if all_replaced:
                            displaced += 1
                            made_progress = True
                            break
                        else:
                            env.cell_occupancy[ci, uf_start:min(uf_end, env.total_time + 1)][
                                env.cell_occupancy[ci, uf_start:min(uf_end, env.total_time + 1)] == uf_id] = 0
                            if ai is not None:
                                env.cell_occupancy[ai, uf_start:min(uf_end, env.total_time + 1)][
                                    env.cell_occupancy[ai, uf_start:min(uf_end, env.total_time + 1)] == uf_id] = 0
                            if uf_id in env.assignments:
                                del env.assignments[uf_id]
                            env.assigned_servers.discard(uf_si)

                            for d_id, d_ci, d_s, d_e, d_si, d_srv, d_2tc in displaced_info:
                                env.cell_occupancy[d_ci, d_s:min(d_e, env.total_time + 1)] = d_id
                                if d_2tc and d_ci in env.adjacent_cells:
                                    d_ai = env.adjacent_cells[d_ci]
                                    env.cell_occupancy[d_ai, d_s:min(d_e, env.total_time + 1)] = d_id
                                env.assignments[d_id] = {'cell_idx': d_ci, 'start': d_s, 'end': d_e - 1}
                                env.assigned_servers.add(d_si)
                    else:
                        for d_id, d_ci, d_s, d_e, d_si, d_srv, d_2tc in displaced_info:
                            env.cell_occupancy[d_ci, d_s:min(d_e, env.total_time + 1)] = d_id
                            if d_2tc and d_ci in env.adjacent_cells:
                                d_ai = env.adjacent_cells[d_ci]
                                env.cell_occupancy[d_ai, d_s:min(d_e, env.total_time + 1)] = d_id

                if uf_srv['k'] in env.assignments:
                    break

        if not made_progress:
            break

    return displaced


def local_search_repair(env, max_iterations=200, seed=42):
    """Local search: random removal/reinsertion."""
    import random as rng
    rng.seed(seed)

    original_count = len(env.servers) - len(env._retry_queue)
    placed_count = 0

    for iteration in range(max_iterations):
        unfinished = []
        for si in range(original_count):
            srv = env.servers[si]
            if srv['k'] not in env.assignments:
                unfinished.append((si, srv))
        if not unfinished:
            break

        uf_si, uf_srv = rng.choice(unfinished)
        uf_id = uf_srv['k']
        uf_arrival = uf_srv['ArrivalTime']
        uf_pt = uf_srv['PTk']
        uf_is_2tc = uf_srv['OTk'] == 1

        candidate_ids = set()
        latest = min(uf_srv['DueTime'] - uf_pt, env.total_time + 1 - uf_pt)
        if latest < uf_arrival:
            latest = env.total_time + 1 - uf_pt
        for start in range(uf_arrival, max(uf_arrival, latest) + 1):
            end = start + uf_pt
            if end > env.total_time + 1:
                break
            for ci in range(env.total_cells):
                if not env._is_cell_compatible(ci, uf_srv):
                    continue
                occ = env.cell_occupancy[ci, start:min(end, env.total_time + 1)]
                for v in occ:
                    if v > 0:
                        candidate_ids.add(int(v))

        if not candidate_ids:
            continue

        victim_id = rng.choice(list(candidate_ids))
        if victim_id not in env.assignments:
            continue

        vasn = env.assignments[victim_id]
        v_ci = vasn['cell_idx']
        v_start = vasn['start']
        v_end = vasn['end'] + 1

        v_srv = None
        v_si = None
        for bsi in range(original_count):
            if env.servers[bsi]['k'] == victim_id:
                v_srv = env.servers[bsi]
                v_si = bsi
                break
        if v_srv is None:
            continue

        v_is_2tc = v_srv['OTk'] == 1

        env.cell_occupancy[v_ci, v_start:min(v_end, env.total_time + 1)][
            env.cell_occupancy[v_ci, v_start:min(v_end, env.total_time + 1)] == victim_id] = 0
        if v_is_2tc and v_ci in env.adjacent_cells:
            v_ai = env.adjacent_cells[v_ci]
            env.cell_occupancy[v_ai, v_start:min(v_end, env.total_time + 1)][
                env.cell_occupancy[v_ai, v_start:min(v_end, env.total_time + 1)] == victim_id] = 0

        del env.assignments[victim_id]
        env.assigned_servers.discard(v_si)

        if _try_place_deferred(env, uf_si, uf_srv):
            if _try_place_deferred(env, v_si, v_srv):
                placed_count += 1
                continue
            else:
                uf_asn = env.assignments[uf_id]
                uf_ci2 = uf_asn['cell_idx']
                uf_s2 = uf_asn['start']
                uf_e2 = uf_asn['end'] + 1
                env.cell_occupancy[uf_ci2, uf_s2:min(uf_e2, env.total_time + 1)][
                    env.cell_occupancy[uf_ci2, uf_s2:min(uf_e2, env.total_time + 1)] == uf_id] = 0
                if uf_is_2tc and uf_ci2 in env.adjacent_cells:
                    uf_ai2 = env.adjacent_cells[uf_ci2]
                    env.cell_occupancy[uf_ai2, uf_s2:min(uf_e2, env.total_time + 1)][
                        env.cell_occupancy[uf_ai2, uf_s2:min(uf_e2, env.total_time + 1)] == uf_id] = 0
                del env.assignments[uf_id]
                env.assigned_servers.discard(uf_si)

        env.cell_occupancy[v_ci, v_start:min(v_end, env.total_time + 1)] = victim_id
        if v_is_2tc and v_ci in env.adjacent_cells:
            v_ai = env.adjacent_cells[v_ci]
            env.cell_occupancy[v_ai, v_start:min(v_end, env.total_time + 1)] = victim_id
        env.assignments[victim_id] = {'cell_idx': v_ci, 'start': v_start, 'end': v_end - 1}
        env.assigned_servers.add(v_si)

    return placed_count


def _compute_tardiness(env):
    """Recompute tardiness for ALL assigned servers."""
    original_count = len(env.servers) - len(env._retry_queue)
    tardy_count = 0
    for si in range(original_count):
        srv = env.servers[si]
        server_id = srv['k']
        if server_id in env.assignments:
            asn = env.assignments[server_id]
            # Use the actual recorded end time from the assignment, not start + PTk.
            # The assignment's 'end' reflects the real completion time (including
            # deferred placement, 2TC scheduling, etc.) and matches the Gantt chart.
            end_time = asn.get('end', asn.get('start', srv['ArrivalTime']) + srv['PTk'])
            if end_time > srv['DueTime']:
                tardy_count += 1
    return tardy_count


# ============================================================================
# Day-level evaluation (10d, 5d, 1d windows)
# ============================================================================

# Server ordering strategies for multi-ordering inference
ORDERING_STRATEGIES = [
    ('2tc_first', None),
    ('slack_first', lambda s: (s.get('slack', s['DueTime'] - s['ArrivalTime'] - s['PTk']),
                               -s['OTk'], s['DueTime'])),
    ('long_pt_first', lambda s: (-s['PTk'], -s['OTk'],
                                  s.get('slack', s['DueTime'] - s['ArrivalTime'] - s['PTk']))),
    ('early_arrival', lambda s: (s['ArrivalTime'], -s['OTk'], -s['PTk'])),
    ('late_due', lambda s: (-s['DueTime'], -s['OTk'], -s['PTk'])),
]


def _run_single_ordering(agent, facility, df, window_length, sort_key_fn,
                          use_dap=False, use_fpr=False,
                          use_displacement=False, hybrid_params=None,
                          skip_repair=False):
    """Run RL agent with a specific server ordering.

    When skip_repair=True (or DAP is ON), greedy post-repair is skipped.
    DAP embeds repair logic inside RL via get_valid_actions_deferred().
    """
    env = LexicographicRewardEnvironment(
        facility, df,
        window_length=window_length,
        use_windowing=True,
        randomize_order=False,
        use_deferred_actions=use_dap,
        use_fpr=use_fpr)

    if sort_key_fn is not None:
        env.servers.sort(key=sort_key_fn)
        priority_sorted = sorted(env.servers,
                                 key=lambda s: (s['DueTime'], s['ArrivalTime']))
        for rank, s in enumerate(priority_sorted):
            s['priority_rank'] = rank
            s['priority_cost'] = (len(env.servers) - rank) * 100000

    state, _ = env.reset()

    if sort_key_fn is not None:
        env.servers.sort(key=sort_key_fn)
        priority_sorted = sorted(env.servers,
                                 key=lambda s: (s['DueTime'], s['ArrivalTime']))
        for rank, s in enumerate(priority_sorted):
            s['priority_rank'] = rank
            s['priority_cost'] = (len(env.servers) - rank) * 100000
        state = env._get_state()

    done = False
    step_count = 0
    while not done:
        # DAP: use deferred valid actions if enabled
        if use_dap:
            valid_actions = env.get_valid_actions_deferred()
        else:
            valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, training=False)
        state, _, done, _, info = env.step(action)
        step_count += 1

    rl_only_comp = info['completion_rate']
    rl_assigned = info['assigned_servers']
    original_count = len(env.servers) - len(getattr(env, '_retry_queue', []))

    # Greedy repair — ONLY for old LHAC (no DAP). When DAP is ON, repair is
    # embedded inside RL via get_valid_actions_deferred(), so skip it.
    repair_count = 0
    if not skip_repair and not use_dap:
        repair_count, original_count = greedy_repair(env)
        # Optional displacement + local search (old LHAC only)
        if use_displacement:
            disp_count = displacement_repair(env, max_swaps=50)
            repair_count += disp_count
            if min(len(env.assignments), original_count) < original_count:
                ls_count = local_search_repair(env, max_iterations=300)
                repair_count += ls_count

    final_assigned = min(len(env.assignments), original_count)
    final_comp = final_assigned / original_count if original_count > 0 else 0.0
    final_unfinished = original_count - final_assigned
    tardy_count = _compute_tardiness(env)

    # Hybrid CPLEX fallback — uses RL-only metrics (no repair in between)
    hybrid_result = None
    if hybrid_params and hybrid_params.get('enabled', False):
        try:
            from hybrid_cplex import run_hybrid_evaluation
            tardiness_rate = tardy_count / final_assigned if final_assigned > 0 else 0.0
            hybrid_result = run_hybrid_evaluation(
                env, facility,
                completion_rate=final_comp,
                tardiness_rate=tardiness_rate,
                threshold_completion=hybrid_params.get('completion_threshold', 0.90),
                threshold_tardiness=hybrid_params.get('tardiness_threshold', 0.05),
                cplex_time_limit=hybrid_params.get('cplex_timelimit', 60),
                verbose=hybrid_params.get('verbose', False))

            if hybrid_result and hybrid_result['cplex_called']:
                final_assigned = min(len(env.assignments), original_count)
                final_comp = final_assigned / original_count if original_count > 0 else 0.0
                final_unfinished = original_count - final_assigned
                tardy_count = _compute_tardiness(env)
        except ImportError:
            print("  WARNING: hybrid_cplex.py not found. CPLEX fallback unavailable.")

    result = {
        'total_servers': original_count,
        'assigned': final_assigned,
        'unfinished': final_unfinished,
        'tardy': tardy_count,
        'completion_rate': final_comp,
        'rl_only_completion': rl_only_comp,
        'rl_only_assigned': rl_assigned,
        'repair_count': repair_count,
        'steps': step_count,
    }

    # Add hybrid-specific fields
    if hybrid_result:
        result['hybrid_cplex_called'] = hybrid_result['cplex_called']
        result['hybrid_additional_placed'] = hybrid_result['cplex_additional_placed']
        result['hybrid_cplex_runtime_sec'] = hybrid_result['cplex_runtime_sec']
    else:
        result['hybrid_cplex_called'] = False
        result['hybrid_additional_placed'] = 0
        result['hybrid_cplex_runtime_sec'] = 0.0

    return result, env


def evaluate_single(agent, facility, df, window_length=10, verbose=False,
                     use_dap=False, use_fpr=False, use_displacement=False,
                     hybrid_params=None, skip_repair=False):
    """Evaluate with multi-ordering inference.

    Runs RL agent with 5 server orderings, keeps best result.
    When DAP is ON, repair is embedded in RL — no post-processing needed.

    For hybrid mode: runs ALL orderings RL-only first, picks best,
    THEN runs CPLEX only on the best ordering (much faster).
    """
    start_time = time.time()

    best_result = None
    best_name = None
    best_env = None

    # Phase 1: Run all orderings RL-only (no CPLEX) to find best ordering
    for name, sort_fn in ORDERING_STRATEGIES:
        result, env_run = _run_single_ordering(agent, facility, df, window_length, sort_fn,
                                       use_dap=use_dap, use_fpr=use_fpr,
                                       use_displacement=use_displacement,
                                       hybrid_params=None,  # No CPLEX in phase 1
                                       skip_repair=skip_repair)
        if best_result is None or result['assigned'] > best_result['assigned']:
            best_result = result
            best_name = name
            best_env = env_run

    # Phase 2: If hybrid is enabled and best RL result is not 100%,
    # call CPLEX directly on the saved best env (NO RL re-run — avoids
    # CUDA non-determinism that caused different/worse placements!)
    if hybrid_params and hybrid_params.get('enabled', False) and best_env is not None:
        comp_threshold = hybrid_params.get('completion_threshold', 0.90)
        if best_result['completion_rate'] < comp_threshold:
            try:
                from hybrid_cplex import run_hybrid_evaluation
                tardy_count = best_result.get('tardy', 0)
                final_assigned = best_result['assigned']
                tardiness_rate = tardy_count / final_assigned if final_assigned > 0 else 0.0

                hybrid_cplex_result = run_hybrid_evaluation(
                    best_env, facility,
                    completion_rate=best_result['completion_rate'],
                    tardiness_rate=tardiness_rate,
                    threshold_completion=hybrid_params.get('completion_threshold', 0.90),
                    threshold_tardiness=hybrid_params.get('tardiness_threshold', 0.05),
                    cplex_time_limit=hybrid_params.get('cplex_timelimit', 300),
                    verbose=hybrid_params.get('verbose', False))

                if hybrid_cplex_result and hybrid_cplex_result['cplex_called']:
                    original_count = best_result['total_servers']
                    final_assigned = min(len(best_env.assignments), original_count)
                    final_comp = final_assigned / original_count if original_count > 0 else 0.0
                    final_unfinished = original_count - final_assigned
                    tardy_count = _compute_tardiness(best_env)

                    best_result['assigned'] = final_assigned
                    best_result['unfinished'] = final_unfinished
                    best_result['completion_rate'] = final_comp
                    best_result['tardy'] = tardy_count
                    best_result['hybrid_cplex_called'] = hybrid_cplex_result['cplex_called']
                    best_result['hybrid_additional_placed'] = hybrid_cplex_result['cplex_additional_placed']
                    best_result['hybrid_cplex_runtime_sec'] = hybrid_cplex_result['cplex_runtime_sec']
            except ImportError:
                print("  WARNING: hybrid_cplex.py not found. CPLEX fallback unavailable.")

    runtime = time.time() - start_time

    out = {
        'total_servers': best_result['total_servers'],
        'assigned': best_result['assigned'],
        'unfinished': best_result['unfinished'],
        'tardy': best_result['tardy'],
        'completion_rate': best_result['completion_rate'],
        'rl_only_completion': best_result['rl_only_completion'],
        'rl_only_assigned': best_result['rl_only_assigned'],
        'repair_count': best_result['repair_count'],
        'runtime_sec': runtime,
        'steps': best_result['steps'],
        'best_ordering': best_name,
        'n_windows': 1,
        'hybrid_cplex_called': best_result.get('hybrid_cplex_called', False),
        'hybrid_additional_placed': best_result.get('hybrid_additional_placed', 0),
        'hybrid_cplex_runtime_sec': best_result.get('hybrid_cplex_runtime_sec', 0.0),
    }

    if verbose:
        print(f"    Total: {out['total_servers']}, "
              f"RL: {out['rl_only_assigned']} ({out['rl_only_completion']:.1%}), "
              f"Final: {out['assigned']}/{out['total_servers']} "
              f"({out['completion_rate']:.1%}), "
              f"Tardy: {out['tardy']}, "
              f"Time: {out['runtime_sec']:.2f}s, "
              f"Best: {best_name}")

    return out


# ============================================================================
# Rolling-window evaluation (sub-day windows: 8h, 4h, 0h)
# ============================================================================

def spread_arrivals_to_hours(df):
    """Convert day-level times to hour-level for sub-day windowing."""
    df_h = df.copy()
    df_h['ArrivalTime_hours'] = (
        (df_h['ArrivalTime'] - 1) * HOURS_PER_DAY + (df_h['k'] % HOURS_PER_DAY)
    ).astype(int)
    df_h['DueTime_hours'] = df_h['DueTime'] * HOURS_PER_DAY
    df_h['PTk_hours'] = df_h['PTk'] * HOURS_PER_DAY
    return df_h


def partition_servers_by_window(df_hourly, window_hours, total_hours=None):
    """Partition servers into rolling windows by hourly arrival time."""
    if total_hours is None:
        total_hours = TOTAL_DAYS * HOURS_PER_DAY

    if window_hours <= 0:
        return [[i] for i in range(len(df_hourly))]

    n_windows = math.ceil(total_hours / window_hours)
    windows = [[] for _ in range(n_windows)]

    for idx, row in df_hourly.iterrows():
        arr_h = row['ArrivalTime_hours']
        w_idx = min((arr_h - 1) // window_hours, n_windows - 1)
        windows[w_idx].append(idx)

    return windows


def evaluate_rolling_window(agent, facility, df, window_key, verbose=False,
                             use_dap=False, use_fpr=False,
                             use_displacement=False, skip_repair=False):
    """Evaluate using rolling-window approach with hourly granularity.

    For sub-day windows (8h, 4h, 0h), converts times to hours.
    When DAP is ON, repair is embedded in RL — no post-processing needed.
    """
    wname, whours = WINDOW_HOURS[window_key]
    total_servers = len(df)

    df_h = spread_arrivals_to_hours(df)
    df_h_sorted = df_h.sort_values(
        ['ArrivalTime_hours', 'OTk', 'PTk'],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    if whours == 0:
        unique_hours = sorted(df_h_sorted['ArrivalTime_hours'].unique())
        windows = []
        for h in unique_hours:
            idxs = df_h_sorted[df_h_sorted['ArrivalTime_hours'] == h].index.tolist()
            windows.append(idxs)
    else:
        windows = partition_servers_by_window(df_h_sorted, whours)

    full_env = LexicographicRewardEnvironment(
        facility, df, window_length=TOTAL_TIME_BLOCKS,
        use_windowing=False, randomize_order=False,
        use_deferred_actions=use_dap, use_fpr=use_fpr)
    full_env.reset()

    start_time = time.time()
    total_assigned = 0
    total_tardy = 0
    total_rl_only = 0
    total_repair = 0
    carryover_indices = []

    for w_idx, w_server_indices in enumerate(windows):
        if len(w_server_indices) == 0 and len(carryover_indices) == 0:
            continue

        all_indices = carryover_indices + w_server_indices
        if len(all_indices) == 0:
            continue

        window_rows = df_h_sorted.iloc[all_indices]

        sub_df = pd.DataFrame({
            'k': window_rows['k'].values,
            'type': window_rows['type'].values,
            'PTk': window_rows['PTk'].values,
            'OTk': window_rows['OTk'].values,
            'RH': window_rows['RH'].values,
            'RL': window_rows['RL'].values,
            'RW': window_rows['RW'].values,
            'DueTime': window_rows['DueTime'].values,
            'ArrivalTime': window_rows['ArrivalTime'].values,
        })

        if whours >= HOURS_PER_DAY:
            w_day_size = whours // HOURS_PER_DAY
            w_start_day = w_idx * w_day_size + 1
            w_end_day = min((w_idx + 1) * w_day_size, TOTAL_TIME_BLOCKS)
        elif whours == 0:
            w_start_day = int(window_rows['ArrivalTime'].min())
            w_end_day = min(int(window_rows['ArrivalTime'].max()) + 1, TOTAL_TIME_BLOCKS)
        else:
            w_start_hour = w_idx * whours + 1
            w_end_hour = min((w_idx + 1) * whours, TOTAL_DAYS * HOURS_PER_DAY)
            w_start_day = max(1, (w_start_hour - 1) // HOURS_PER_DAY + 1)
            w_end_day = min(w_end_hour // HOURS_PER_DAY + 1, TOTAL_TIME_BLOCKS)

        wlen = max(w_end_day - w_start_day + 1, 1)
        window_env = LexicographicRewardEnvironment(
            facility, sub_df,
            window_length=wlen,
            use_windowing=True,
            randomize_order=False,
            use_deferred_actions=use_dap,
            use_fpr=use_fpr)
        window_env.reset()

        window_env.cell_occupancy = full_env.cell_occupancy.copy()
        window_env.assignments = copy.deepcopy(full_env.assignments)
        window_env.assigned_servers = set()

        state = window_env._get_state()
        done = False
        while not done:
            if use_dap:
                valid_actions = window_env.get_valid_actions_deferred()
            else:
                valid_actions = window_env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            state, _, done, _, info = window_env.step(action)

        rl_assigned_this_window = len(window_env.assigned_servers)
        total_rl_only += rl_assigned_this_window

        # Greedy repair — ONLY for old LHAC (no DAP). When DAP is ON,
        # repair is embedded inside RL via get_valid_actions_deferred().
        repair_count = 0
        if not skip_repair and not use_dap:
            original_count_w = len(window_env.servers) - len(window_env._retry_queue)
            for pass_num in range(3):
                unassigned = [(si, window_env.servers[si]) for si in range(original_count_w)
                              if si not in window_env.assigned_servers
                              and window_env.servers[si]['k'] not in window_env.assignments]
                if not unassigned:
                    break
                if pass_num == 0:
                    unassigned.sort(key=lambda p: (-p[1]['OTk'], p[1].get('slack', 999)))
                elif pass_num == 1:
                    unassigned.sort(key=lambda p: (p[1].get('slack', 999), -p[1]['OTk']))
                else:
                    unassigned.sort(key=lambda p: (-p[1]['PTk'], -p[1]['OTk'],
                                                   p[1].get('slack', 999)))
                pass_placed = 0
                for si, srv in unassigned:
                    if _try_place_deferred(window_env, si, srv):
                        repair_count += 1
                        pass_placed += 1
                if pass_placed == 0:
                    break

            if use_displacement:
                disp_count = displacement_repair(window_env, max_swaps=50)
                repair_count += disp_count
        total_repair += repair_count

        final_assigned_this_window = len(window_env.assigned_servers)
        total_assigned += final_assigned_this_window
        total_tardy += _compute_tardiness(window_env)

        full_env.cell_occupancy = window_env.cell_occupancy.copy()
        full_env.assignments = copy.deepcopy(window_env.assignments)

        assigned_k_ids = set(window_env.assignments.keys())
        new_carryover = []
        for local_idx, global_idx in enumerate(all_indices):
            server_k = int(df_h_sorted.iloc[global_idx]['k'])
            if server_k not in assigned_k_ids:
                due = int(df_h_sorted.iloc[global_idx]['DueTime'])
                arr = int(df_h_sorted.iloc[global_idx]['ArrivalTime'])
                pt = int(df_h_sorted.iloc[global_idx]['PTk'])
                if arr + pt <= TOTAL_TIME_BLOCKS + 1:
                    new_carryover.append(global_idx)

        carryover_indices = new_carryover

        if verbose:
            print(f"    Window {w_idx+1}: {len(w_server_indices)} new + "
                  f"{len(carryover_indices)} carry | "
                  f"RL: {rl_assigned_this_window} + Repair: {repair_count} = "
                  f"{final_assigned_this_window}")

    runtime = time.time() - start_time
    total_unfinished = total_servers - total_assigned
    completion_rate = total_assigned / total_servers if total_servers > 0 else 0.0
    rl_only_rate = total_rl_only / total_servers if total_servers > 0 else 0.0

    return {
        'total_servers': total_servers,
        'assigned': total_assigned,
        'unfinished': total_unfinished,
        'tardy': total_tardy,
        'completion_rate': completion_rate,
        'rl_only_completion': rl_only_rate,
        'rl_only_assigned': total_rl_only,
        'repair_count': total_repair,
        'runtime_sec': runtime,
        'n_windows': len([w for w in windows if len(w) > 0]),
    }


# ============================================================================
# Unified evaluation dispatcher
# ============================================================================

def evaluate_dataset(agent, facility, df, window_key, verbose=False,
                     use_dap=False, use_fpr=False, use_displacement=False,
                     hybrid_params=None, skip_repair=False):
    """Dispatch to appropriate evaluation method based on window type.

    When DAP is ON, skip_repair is automatically True (DAP embeds repair in RL).
    """
    # DAP replaces greedy repair — auto-skip when DAP is enabled
    effective_skip_repair = skip_repair or use_dap

    if window_key in DAY_LEVEL_WINDOWS:
        _, wlen = WINDOW_BLOCKS[window_key]
        return evaluate_single(agent, facility, df,
                                window_length=wlen,
                                verbose=verbose,
                                use_dap=use_dap,
                                use_fpr=use_fpr,
                                use_displacement=use_displacement,
                                hybrid_params=hybrid_params,
                                skip_repair=effective_skip_repair)
    else:
        return evaluate_rolling_window(agent, facility, df,
                                        window_key,
                                        verbose=verbose,
                                        use_dap=use_dap,
                                        use_fpr=use_fpr,
                                        use_displacement=use_displacement,
                                        skip_repair=effective_skip_repair)


# ============================================================================
# Display functions
# ============================================================================

def print_results_table(results, window_key):
    """Print results table for a window."""
    print(f"\n{'='*140}")
    print(f"RESULTS: w={window_key} | DAP={'ON' if results[0].get('dap', False) else 'OFF'}")
    print(f"{'='*140}")

    header = (f"{'Pattern':<15} {'Size':>5} {'2TC':>5} | "
              f"{'RL-only%':>9} {'+Repair':>8} {'Final%':>8} {'Unfin':>6} {'Tardy':>6} {'Time(s)':>8}")
    print(header)
    print("-" * 140)

    for r in results:
        line = (f"{r['pattern']:<15} {r['size']:>5} {r['twotc']:>5} | "
                f"{r['rl_only_completion']*100:>8.1f}% "
                f"+{r['repair_count']:>7} "
                f"{r['completion_rate']*100:>7.1f}% "
                f"{r['unfinished']:>6} {r['tardy']:>6} "
                f"{r['runtime_sec']:>7.2f}s")
        print(line)

    print("-" * 140)
    avg_comp = np.mean([r['completion_rate'] for r in results]) * 100
    avg_rl = np.mean([r['rl_only_completion'] for r in results]) * 100
    print(f"Average RL-only: {avg_rl:.1f}% -> Average Final: {avg_comp:.1f}%")


def print_multi_window_comparison(all_results):
    """Print compact multi-window comparison."""
    windows = list(all_results.keys())
    width = 32 + 10 + len(windows) * 10
    print(f"\n{'='*width}")
    print("MULTI-WINDOW COMPARISON: Completion Rate (%)")
    print(f"{'='*width}")

    hdr = f"{'Pattern':<15} {'Size':>5} {'2TC':>5} |"
    for w in windows:
        hdr += f" {w:>8}"
    print(hdr)
    print("-" * width)

    first = windows[0]
    for r in all_results[first]:
        key = (r['pattern'], r['size'], r['twotc'])
        line = f"{r['pattern']:<15} {r['size']:>5} {r['twotc']:>5} |"
        for w in windows:
            match = [wr for wr in all_results[w]
                     if (wr['pattern'], wr['size'], wr['twotc']) == key]
            if match:
                line += f" {match[0]['completion_rate']*100:>7.1f}%"
            else:
                line += f" {'N/A':>8}"
        print(line)

    print("-" * width)
    avg_line = f"{'AVERAGE':<15} {'':>5} {'':>5} |"
    rl_line = f"{'(RL-only avg)':<15} {'':>5} {'':>5} |"
    for w in windows:
        avg = np.mean([r['completion_rate'] for r in all_results[w]]) * 100
        avg_rl = np.mean([r['rl_only_completion'] for r in all_results[w]]) * 100
        avg_line += f" {avg:>7.1f}%"
        rl_line += f" {avg_rl:>7.1f}%"
    print(avg_line)
    print(rl_line)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DAF-LHAC Evaluation Pipeline')
    parser.add_argument('--model', type=str,
                        default=os.path.join(_BASE_DIR, 'Models', 'lhac_ppo_v1_best.pth'),
                        help='Path to trained model')
    parser.add_argument('--windows', type=str, default='10d',
                        help='Comma-separated window names: 10d,5d,1d,8h,4h,0h')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Comma-separated seeds (only affects stochastic repairs)')
    parser.add_argument('--dap', action='store_true',
                        help='Enable Deferred Action Placement (Innovation 1)')
    parser.add_argument('--fpr', action='store_true',
                        help='Enable Feasibility-Preserving Reward (Innovation 3)')
    parser.add_argument('--no_repair', action='store_true',
                        help='Skip repair (RL-only evaluation)')
    parser.add_argument('--displacement', action='store_true',
                        help='Enable displacement + local search repair')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-dataset results')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results CSV')
    parser.add_argument('--variant', type=str, default='daf_lhac',
                        help='Variant name for results tagging')
    parser.add_argument('--quick_test', action='store_true',
                        help='Smoke test: 3 datasets only')
    parser.add_argument('--hybrid', action='store_true',
                        help='Enable CPLEX fallback for residual unplaced servers')
    parser.add_argument('--hybrid_completion_threshold', type=float, default=1.0,
                        help='Completion threshold below which CPLEX is triggered (default: 1.0 = always reach 100%%)')
    parser.add_argument('--hybrid_tardiness_threshold', type=float, default=0.05,
                        help='Tardiness threshold above which CPLEX is triggered')
    parser.add_argument('--hybrid_cplex_timelimit', type=int, default=300,
                        help='CPLEX time limit in seconds (default: 300)')
    args = parser.parse_args()

    window_keys = [w.strip() for w in args.windows.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Validate windows
    for w in window_keys:
        if w not in WINDOW_HOURS:
            print(f"Unknown window: {w}. Valid: {list(WINDOW_HOURS.keys())}")
            sys.exit(1)

    # Select datasets
    datasets = QUICK_TEST_DATASETS if args.quick_test else TABLE6_DATASETS
    data_root = PARVEZ_DATA_ROOT

    # Check Parvez data exists, fall back to Data/
    sample_exists = any(os.path.exists(os.path.join(data_root, d[3])) for d in datasets)
    if not sample_exists:
        data_root = os.path.join(_BASE_DIR, 'Data')
        print("Parvez datasets not found, using Data/ directory")

    # Build hybrid params dict
    hybrid_params = None
    if args.hybrid:
        hybrid_params = {
            'enabled': True,
            'completion_threshold': args.hybrid_completion_threshold,
            'tardiness_threshold': args.hybrid_tardiness_threshold,
            'cplex_timelimit': args.hybrid_cplex_timelimit,
            'verbose': args.verbose,
        }

    # Determine effective repair mode
    effective_skip_repair = args.no_repair or args.dap
    repair_mode = "SKIPPED (DAP embeds repair in RL)" if effective_skip_repair else "ON (greedy repair)"

    print(f"DAF-LHAC Evaluation")
    print(f"  Model: {args.model}")
    print(f"  DAP: {'ON' if args.dap else 'OFF'}")
    print(f"  FPR: {'ON' if args.fpr else 'OFF'}")
    print(f"  Post-RL Repair: {repair_mode}")
    print(f"  Hybrid CPLEX: {'ON' if args.hybrid else 'OFF'}")
    if args.hybrid:
        print(f"    Completion threshold: {args.hybrid_completion_threshold:.0%}")
        print(f"    Tardiness threshold: {args.hybrid_tardiness_threshold:.0%}")
        print(f"    CPLEX time limit: {args.hybrid_cplex_timelimit}s")
    print(f"  Windows: {window_keys}")
    print(f"  Seeds: {seeds}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Data root: {data_root}")

    # Initialize facility
    print("\nInitializing facility...")
    facility = FacilityConfiguration()

    # Load agent
    print(f"Loading model from {args.model}...")
    sample_path = None
    for _, _, _, fname in datasets:
        p = os.path.join(data_root, fname)
        if os.path.exists(p):
            sample_path = p
            break
    if sample_path is None:
        print("ERROR: No datasets found!")
        sys.exit(1)

    sample_df = pd.read_excel(sample_path)

    # Create dummy env WITHOUT DAP to get the MODEL's expected state_dim (74)
    # The existing pre-trained model uses 74 dims; DAP adds 4 more features
    # that only matter after retraining.
    dummy_env_no_dap = LexicographicRewardEnvironment(
        facility, sample_df, window_length=10, use_windowing=True,
        use_deferred_actions=False)
    model_state_dim = dummy_env_no_dap.observation_space.shape[0]  # 74
    action_dim = dummy_env_no_dap.action_space.n  # 55

    # Check if the checkpoint was trained with DAP (state_dim=78) or without (74)
    # Also detect if it uses CASE (Cross-Attention) network
    checkpoint_peek = torch.load(args.model, map_location='cpu', weights_only=False)
    ckpt_use_case = checkpoint_peek.get('use_case', False)

    if 'agent1_network' in checkpoint_peek:
        net_keys = checkpoint_peek['agent1_network'].keys()

        if ckpt_use_case or any('server_encoder' in k for k in net_keys):
            # CASE model: state is split into server(16) + cells(54*12) + global(20)
            # The "state_dim" in checkpoint tells us the flat dim used at training time
            ckpt_state_dim = checkpoint_peek.get('state_dim', 78 if args.dap else 74)
            ckpt_use_case = True
            print(f"  Detected CASE (Cross-Attention) model")
        else:
            # Standard MLP model: check first layer input size
            first_key = [k for k in net_keys if 'encoder.0.weight' in k]
            if first_key:
                ckpt_state_dim = checkpoint_peek['agent1_network'][first_key[0]].shape[1]
            else:
                ckpt_state_dim = model_state_dim
    elif 'q1_network' in checkpoint_peek:
        # DQN model
        first_key = [k for k in checkpoint_peek['q1_network'].keys()
                     if '.0.weight' in k]
        if first_key:
            ckpt_state_dim = checkpoint_peek['q1_network'][first_key[0]].shape[1]
        else:
            ckpt_state_dim = model_state_dim
    else:
        ckpt_state_dim = model_state_dim
    del checkpoint_peek

    # Env state_dim when DAP is ON = 78, but model might expect 74
    env_state_dim = 78 if args.dap else 74
    print(f"  Model state dim: {ckpt_state_dim}, Env state dim: {env_state_dim}, Action dim: {action_dim}")

    # If model was trained without DAP (74) but DAP is ON (78), we need to
    # truncate the state vector to 74 during inference
    need_state_truncation = (ckpt_state_dim < env_state_dim)
    if need_state_truncation:
        print(f"  NOTE: Model expects {ckpt_state_dim} dims, env produces {env_state_dim}. "
              f"Will truncate state to {ckpt_state_dim} during inference.")

    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    agent, agent_type = load_agent(args.model, ckpt_state_dim, action_dim, device,
                                     use_case=ckpt_use_case)
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0

    # Wrap agent with state truncation if needed
    if need_state_truncation:
        agent = StateTruncatingAgent(agent, ckpt_state_dim)
        print(f"  Wrapped agent with state truncation: {env_state_dim} -> {ckpt_state_dim}")

    print(f"  Device: {device}, Agent: {agent_type}")

    # Run evaluations
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_csv_rows = []
    all_results_by_window = {}

    for wkey in window_keys:
        wlabel = WINDOW_HOURS[wkey][0]
        print(f"\n{'='*60}")
        print(f"Window: {wkey} ({wlabel})")
        print(f"{'='*60}")

        window_results = []

        for pattern, size, twotc, filename in datasets:
            filepath = os.path.join(data_root, filename)
            if not os.path.exists(filepath):
                print(f"  SKIP: {filename} not found")
                continue

            df = pd.read_excel(filepath)
            actual_2tc = df['OTk'].sum() / len(df) * 100

            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)

                result = evaluate_dataset(
                    agent, facility, df, wkey,
                    verbose=args.verbose,
                    use_dap=args.dap,
                    use_fpr=args.fpr,
                    use_displacement=args.displacement,
                    hybrid_params=hybrid_params,
                    skip_repair=args.no_repair)

                # Progress line
                hybrid_tag = ""
                if result.get('hybrid_cplex_called', False):
                    hybrid_tag = (f" +{result['hybrid_additional_placed']} CPLEX"
                                  f"({result['hybrid_cplex_runtime_sec']:.1f}s)")
                repair_tag = ""
                if result['repair_count'] > 0:
                    repair_tag = f" +{result['repair_count']} repair"
                print(f"  {pattern:>14} {size:>3} {twotc:>4} seed={seed} -> "
                      f"RL: {result['rl_only_completion']*100:.1f}%"
                      f"{repair_tag}"
                      f"{hybrid_tag} = "
                      f"{result['completion_rate']*100:.1f}% "
                      f"({result['runtime_sec']:.1f}s)")

                row = {
                    'variant': args.variant,
                    'window': wkey,
                    'pattern': pattern,
                    'size': size,
                    'twotc': twotc,
                    'actual_2tc_pct': round(actual_2tc, 1),
                    'seed': seed,
                    'dap': args.dap,
                    'fpr': args.fpr,
                    'total_servers': result['total_servers'],
                    'assigned': result['assigned'],
                    'unfinished': result['unfinished'],
                    'tardy': result['tardy'],
                    'completion_rate': round(result['completion_rate'], 6),
                    'completion_pct': round(result['completion_rate'] * 100, 2),
                    'rl_only_completion': round(result['rl_only_completion'], 6),
                    'rl_only_pct': round(result['rl_only_completion'] * 100, 2),
                    'repair_count': result['repair_count'],
                    'runtime_sec': round(result['runtime_sec'], 2),
                    'n_windows': result.get('n_windows', 1),
                    'hybrid_cplex_called': result.get('hybrid_cplex_called', False),
                    'hybrid_additional_placed': result.get('hybrid_additional_placed', 0),
                    'hybrid_cplex_runtime_sec': round(result.get('hybrid_cplex_runtime_sec', 0.0), 2),
                }
                all_csv_rows.append(row)
                window_results.append(row)

        all_results_by_window[wkey] = window_results

        # Print window summary
        if window_results:
            print_results_table(window_results, wkey)

    # Multi-window comparison
    if len(window_keys) > 1 and all(len(all_results_by_window.get(w, [])) > 0 for w in window_keys):
        print_multi_window_comparison(all_results_by_window)

    # Save CSV
    if all_csv_rows:
        output_path = args.output
        if output_path is None:
            dap_tag = '_dap' if args.dap else ''
            fpr_tag = '_fpr' if args.fpr else ''
            output_path = os.path.join(RESULTS_DIR,
                                        f'{args.variant}{dap_tag}{fpr_tag}_results.csv')
        df_out = pd.DataFrame(all_csv_rows)
        df_out.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        print(f"Total evaluations: {len(all_csv_rows)}")

    print("\nDone!")


if __name__ == '__main__':
    main()
