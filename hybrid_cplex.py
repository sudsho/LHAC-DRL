#!/usr/bin/env python3
"""
Hybrid DAF-LHAC + CPLEX Residual Solver.

After RL (DAF-LHAC) places servers, CPLEX solves the residual problem
for any servers the RL agent could not place.

Approach:
  - RL placements are LOCKED as frozen constraints
  - Pre-filtered variables (only feasible k,t,i,j tuples → ~80% variable reduction)
  - Linear constraints instead of indicator constraints (~10x fewer)
  - yk-conditional constraints (no phantom assignments)
  - Iterative batching for large residual problems (>100 servers)

Key property: RL placements are never displaced, preserving the
non-preemption constraint (once assigned, a server is not moved).

Usage:
    # Called from evaluate.py with --hybrid flag
    python hybrid_cplex.py --model Models/daf_full_daf.pth --dataset Data/test.xlsx
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Constants matching daf_lhac_core.py
NUM_BANKS = 4
NUM_CELLS_PER_BANK = 14
UNIT_TIME = 24
TOTAL_TIME_BLOCKS = 65

# Excluded cells: bank 4, positions 4 and 7
EXCLUDED_CELLS = {(4, 4), (4, 7)}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'Input')
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Input')


def should_call_cplex(completion_rate, tardiness_rate,
                      threshold_completion=0.90,
                      threshold_tardiness=0.05):
    """Decide whether to invoke CPLEX fallback."""
    return (completion_rate < threshold_completion or
            tardiness_rate > threshold_tardiness)


def extract_unplaced_servers(env):
    """Extract unplaced server data from RL environment (original servers only).

    IMPORTANT: env.servers may include retry-queue duplicates (same k value).
    We only iterate over the original servers (before retry entries) and
    de-duplicate by server ID to avoid giving CPLEX duplicate entries.
    """
    assigned_ids = set(env.assignments.keys())
    original_count = len(env.servers) - len(getattr(env, '_retry_queue', []))
    unplaced = []
    seen_ids = set()
    for si in range(original_count):
        server = env.servers[si]
        server_id = server['k']
        if server_id not in assigned_ids and server_id not in seen_ids:
            unplaced.append(server)
            seen_ids.add(server_id)
    return unplaced


def _build_cell_capability_matrices(facility):
    """Build VH, VL, W matrices (4x14) from FacilityConfiguration."""
    from daf_lhac_core import generate_cell_bank_from_b052
    cell_bank_df = generate_cell_bank_from_b052(facility)

    def cell_data_convert(x1):
        x1 = np.array(x1).flatten()
        x1 = np.reshape(x1, (-1, NUM_CELLS_PER_BANK))
        return x1

    VH = cell_data_convert(cell_bank_df['HV Availability'].values)
    VL = cell_data_convert(cell_bank_df['LV Availability'].values)
    W = cell_data_convert(cell_bank_df['Water Cooling Availability'].values)
    return VH, VL, W


def _build_idx_to_bank_cell(env):
    """Build mapping from RL cell_idx to (bank_id, position)."""
    idx_to_bc = {}
    bc_to_idx = {}
    for idx, cell in env.idx_to_cell.items():
        bank_id = cell.bank_id
        position = cell.position
        idx_to_bc[idx] = (bank_id, position)
        bc_to_idx[(bank_id, position)] = idx
    return idx_to_bc, bc_to_idx


def _get_blocked_cell_times(env, idx_to_bc):
    """Get all (bank, position, time) tuples blocked by placed servers.

    Reads directly from env.cell_occupancy (the ground truth) to ensure
    complete coverage. This catches ALL occupied cells including:
    - Primary cells of 1TC/2TC servers
    - Adjacent cells of 2TC servers
    - Any cell marked by the RL env during step()

    Previous approach (reading from env.assignments + computing adj from OTk)
    could miss cells if assignment metadata was incomplete.

    Returns:
        blocked_cells: set of (bank, pos, t) occupied by placed servers
        blocked_by_server: dict {(bank, pos, t): server_id} who blocks it
    """
    blocked_cells = set()
    blocked_by_server = {}  # (bank, pos, t) -> server_id

    # Read directly from cell_occupancy (ground truth)
    for cell_idx in range(env.total_cells):
        if cell_idx not in idx_to_bc:
            continue
        bank, pos = idx_to_bc[cell_idx]
        for t in range(1, env.total_time + 1):
            if t < env.cell_occupancy.shape[1]:
                val = env.cell_occupancy[cell_idx, t]
                if val > 0:  # Occupied by a server (>0 = server_id, -1 = fab block)
                    sid = int(val)
                    key = (bank, pos, t)
                    blocked_cells.add(key)
                    blocked_by_server[key] = sid

    return blocked_cells, blocked_by_server


def _get_fab_blocks(env, idx_to_bc):
    """Get fabrication block times from environment."""
    fab_blocks = set()
    for cell_idx in range(env.total_cells):
        if cell_idx not in idx_to_bc:
            continue
        bank_id, position = idx_to_bc[cell_idx]
        for t in range(1, env.total_time + 1):
            if env.cell_occupancy[cell_idx, t] == -1:
                fab_blocks.add((bank_id, position, t))
    return fab_blocks


def _is_cell_compatible(bank, pos, server, VH, VL, W):
    """Check if a cell is compatible with a server (voltage, water)."""
    if (bank, pos) in EXCLUDED_CELLS:
        return False
    rh = int(server['RH'])
    rl = int(server['RL'])
    rw = int(server['RW'])
    vh = VH[bank - 1][pos - 1]
    vl = VL[bank - 1][pos - 1]
    w = W[bank - 1][pos - 1]
    if rh == 1 and vh == 0:
        return False
    if rl == 1 and vl == 0:
        return False
    if rw == 1 and w == 0:
        return False
    return True


def _prefilter_feasible_tuples(servers, VH, VL, W, blocked_cells, fab_blocks,
                                total_time, server_id_map_reverse=None):
    """Pre-filter to find only feasible (k, t, i, j) tuples.

    Eliminates impossible variable assignments upfront:
    - Cell incompatible with server (voltage/water)
    - Time before server arrival
    - Cell-time blocked by RL or fabrication
    - Excluded cells (bank 4, pos 4/7)

    For 2TC servers: creates x-variables for BOTH primary cell (i,j)
    AND adjacent cell (i,j+1). This is critical for:
    - Correct one_per_cell constraints (prevents 2TC adjacent cell conflicts)
    - Satisfiable processing time constraint: sum(x) = PTk*(1+OTk)*yk

    Returns:
        feasible_tuples: set of (k, t, i, j) - includes primary AND adjacent cells
        feasible_cells: dict {k: set of (i,j)} - ALL cells used (primary + adjacent)
        feasible_times: dict {k: list of t} - feasible times per server
        primary_cells: dict {k: set of (i,j)} - only PRIMARY cells (for c2 constraints)
    """
    I = range(1, NUM_BANKS + 1)
    J = range(1, NUM_CELLS_PER_BANK + 1)

    all_blocked = blocked_cells | fab_blocks

    feasible_tuples = set()
    feasible_cells = {}   # k -> set of ALL (i,j) used (primary + adjacent)
    feasible_times = {}   # k -> list of valid t
    primary_cells = {}    # k -> set of (i,j) that are PRIMARY cells only

    for idx_k, server in enumerate(servers):
        k = idx_k + 1
        arrival = int(server['ArrivalTime'])
        otk = int(server.get('OTk', 0))

        # Valid times: from arrival onwards
        valid_times = list(range(max(1, arrival), total_time + 1))
        feasible_times[k] = valid_times

        # Find primary cells (the cell chosen for placement)
        pfc = set()
        for i in I:
            for j in J:
                if not _is_cell_compatible(i, j, server, VH, VL, W):
                    continue
                # For 2TC, also check adjacent cell exists and is compatible
                if otk == 1:
                    adj_j = j + 1
                    if adj_j > NUM_CELLS_PER_BANK:
                        continue
                    if not _is_cell_compatible(i, adj_j, server, VH, VL, W):
                        continue
                pfc.add((i, j))

        primary_cells[k] = pfc

        # All cells = primary + adjacent (for 2TC)
        fc = set(pfc)
        if otk == 1:
            for (i, j) in pfc:
                fc.add((i, j + 1))
        feasible_cells[k] = fc

        # Build feasible tuples (iterate primary cells, add both for 2TC)
        for t in valid_times:
            for (i, j) in pfc:
                if (i, j, t) not in all_blocked:
                    if otk == 1:
                        adj_j = j + 1
                        if (i, adj_j, t) in all_blocked:
                            continue
                        # Add BOTH primary and adjacent cell tuples
                        feasible_tuples.add((k, t, i, j))
                        feasible_tuples.add((k, t, i, adj_j))
                    else:
                        feasible_tuples.add((k, t, i, j))

    return feasible_tuples, feasible_cells, feasible_times, primary_cells



def build_cplex_residual(env, facility, servers_to_place,
                          blocked_cells, fab_blocks,
                          VH, VL, W,
                          rl_warmstart_hints=None,
                          unit_time_length=UNIT_TIME,
                          forced_server_ids=None):
    """Build CPLEX model inputs for the residual/expanded problem.

    Args:
        env: LexicographicRewardEnvironment
        facility: FacilityConfiguration
        servers_to_place: list of server dicts (unplaced + optionally blocking RL servers)
        blocked_cells: set of (bank, pos, t) blocked by LOCKED RL servers
        fab_blocks: set of (bank, pos, t) fabrication blocks
        VH, VL, W: cell capability matrices
        rl_warmstart_hints: dict {server_id: {'bank','cell','start','end'}} for warm-starting
        unit_time_length: time granularity
        forced_server_ids: set of server IDs that MUST be placed (yk=1 hard constraint).
                          Used for displaced blockers that must be re-placed.

    Returns:
        dict with CPLEX model inputs, or None if no servers to place
    """
    if not servers_to_place:
        return None

    total_time = env.total_time

    # --- Pre-filter feasible tuples ---
    feasible_tuples, feasible_cells, feasible_times, primary_cells = _prefilter_feasible_tuples(
        servers_to_place, VH, VL, W, blocked_cells, fab_blocks, total_time)

    if not feasible_tuples:
        return None

    K = list(range(1, len(servers_to_place) + 1))
    T = list(range(1, total_time + 1))
    I = list(range(1, NUM_BANKS + 1))
    J = list(range(1, NUM_CELLS_PER_BANK + 1))

    server_id_map = {}
    PTk = []
    OTk = []
    cost_list = []

    for idx_k, server in enumerate(servers_to_place):
        cplex_k = idx_k + 1
        server_id_map[cplex_k] = server['k']
        PTk.append(int(server['PTk']))
        OTk.append(int(server['OTk']))

        slack = max(0, server.get('DueTime', total_time) - server['ArrivalTime'] - server['PTk'])
        cost_list.append(max(1, 100 - slack * 5))

    # --- Due times (for tardiness penalty in CPLEX objective) ---
    DueTime = []
    for server in servers_to_place:
        DueTime.append(int(server.get('DueTime', total_time)))

    # --- Warm-start values ---
    warmstart_values = {}
    if rl_warmstart_hints:
        for cplex_k, server in zip(K, servers_to_place):
            sid = server['k']
            if sid in rl_warmstart_hints:
                warmstart_values[cplex_k] = rl_warmstart_hints[sid]

    # --- Forced servers (must be placed, yk=1) ---
    forced_k = set()
    if forced_server_ids:
        for cplex_k, server in zip(K, servers_to_place):
            if server['k'] in forced_server_ids:
                forced_k.add(cplex_k)

    # --- Block list path ---
    block_list_path = os.path.join(INPUT_DIR, 'block_list.xlsx')
    if not os.path.exists(block_list_path):
        block_list_path = None

    return {
        'K': K,
        'T': T,
        'I': I,
        'J': J,
        'VH': VH,
        'VL': VL,
        'W': W,
        'PTk': PTk,
        'OTk': OTk,
        'cost_list': cost_list,
        'DueTime': DueTime,
        'feasible_tuples': feasible_tuples,
        'feasible_cells': feasible_cells,
        'feasible_times': feasible_times,
        'primary_cells': primary_cells,
        'blocked_cells': blocked_cells,
        'fab_blocks': fab_blocks,
        'block_list_path': block_list_path,
        'server_id_map': server_id_map,
        'servers': servers_to_place,
        'warmstart_values': warmstart_values,
        'forced_k': forced_k,
        'unit_time_length': unit_time_length,
        'total_time': total_time,
    }


def solve_residual_with_cplex(cplex_inputs, time_limit=300, verbose=True):
    """Solve the residual problem with CPLEX (improved v2).

    Key improvements over v1:
    - Pre-filtered variables (only feasible tuples)
    - Linear constraints instead of indicator constraints
    - yk-conditional constraints (no phantom assignments)
    - MIP warm-start from RL placements

    Args:
        cplex_inputs: dict from build_cplex_residual()
        time_limit: Maximum CPLEX solve time in seconds
        verbose: Print progress

    Returns:
        dict: {original_server_id: {'bank': int, 'cell': int, 'start': int, 'end': int}}
        or None if no solution found
    """
    try:
        from docplex.mp.model import Model
    except ImportError:
        print("WARNING: docplex not installed. CPLEX fallback unavailable.")
        return None

    K = cplex_inputs['K']
    T = cplex_inputs['T']
    I = cplex_inputs['I']
    J = cplex_inputs['J']
    PTk = cplex_inputs['PTk']
    OTk = cplex_inputs['OTk']
    cost_list = cplex_inputs['cost_list']
    feasible_tuples = cplex_inputs['feasible_tuples']
    feasible_cells = cplex_inputs['feasible_cells']
    feasible_times = cplex_inputs['feasible_times']
    primary_cells_map = cplex_inputs.get('primary_cells', feasible_cells)  # fallback
    server_id_map = cplex_inputs['server_id_map']
    warmstart_values = cplex_inputs.get('warmstart_values', {})
    forced_k = cplex_inputs.get('forced_k', set())
    total_time = cplex_inputs['total_time']

    if verbose:
        total_possible = len(K) * len(T) * len(I) * len(J)
        print(f"\n  CPLEX v2 Solver: {len(K)} servers, "
              f"{len(feasible_tuples):,} feasible vars "
              f"(reduced from {total_possible:,}, "
              f"{100*(1-len(feasible_tuples)/max(1,total_possible)):.0f}% reduction)")

    # --- Build CPLEX model ---
    mdl = Model('DAF_LHAC_Residual_v2')

    if time_limit > 0:
        mdl.parameters.timelimit = time_limit
    mdl.parameters.threads = 8
    mdl.parameters.mip.tolerances.mipgap = 0.001
    mdl.parameters.randomseed = 1
    # Emphasis on finding feasible solutions quickly
    mdl.parameters.emphasis.mip = 1  # Feasibility emphasis

    # For displacement models (with forced servers), use additional heuristics
    # to improve the warm-start solution via local search rather than pure B&B.
    if forced_k:
        # Switch to solution polishing after 60s of normal B&B.
        # Polishing uses local search to improve the current integer solution
        # and is much more effective than B&B when the LP bound is weak.
        polish_time = 60
        mdl.parameters.mip.polishafter.time = polish_time
        # RINS heuristic: fix most variables to their warm-start values and
        # solve a much smaller sub-MIP. Very effective for rearrangement.
        mdl.parameters.mip.strategy.rinsheur = 5  # Every 5 nodes
        # Local branching: solve sub-MIPs in neighborhood of current solution
        mdl.parameters.mip.strategy.lbheur = 1

    # === Decision variables (PRE-FILTERED) ===

    # x[k,t,i,j] - only for feasible tuples
    x = mdl.binary_var_dict(feasible_tuples, name='x')

    # ckij[k,i,j] - only for feasible cells
    ckij_keys = []
    for k in K:
        for (i, j) in feasible_cells.get(k, set()):
            ckij_keys.append((k, i, j))
    ckij = mdl.binary_var_dict(ckij_keys, name='ckij')

    # c2[k,i,j] for adjacent pairs (2TC only) — use PRIMARY cells only
    c2_keys = []
    for k in K:
        if OTk[k-1] == 1:  # Only for 2TC servers
            for (i, j) in primary_cells_map.get(k, set()):
                # Primary cell j paired with adjacent j+1 (both in feasible_cells)
                c2_keys.append((k, i, j))
    c2 = mdl.binary_var_dict(c2_keys, name='c2')

    # yk[k] - server assigned indicator
    yk = mdl.binary_var_dict(K, name='Yk')

    # ckt[k,t] - server active at time t
    ckt_keys = []
    for k in K:
        for t in feasible_times.get(k, []):
            ckt_keys.append((k, t))
    ckt = mdl.binary_var_dict(ckt_keys, name='ckt')

    # c3[k,t] - server starts at time t
    c3_keys = []
    for k in K:
        ptk = PTk[k-1]
        for t in feasible_times.get(k, []):
            # Can only start if there's enough time for processing
            if t + ptk - 1 <= total_time:
                c3_keys.append((k, t))
    c3 = mdl.binary_var_dict(c3_keys, name='c3')

    if verbose:
        print(f"  Variables: x={len(feasible_tuples)}, ckij={len(ckij_keys)}, "
              f"c2={len(c2_keys)}, yk={len(K)}, ckt={len(ckt_keys)}, c3={len(c3_keys)}")
        print(f"  Total: {len(feasible_tuples)+len(ckij_keys)+len(c2_keys)+len(K)+len(ckt_keys)+len(c3_keys):,}")

    # === Forced servers: yk=1 (must be placed) ===
    # Used for displaced blockers that MUST be re-placed. Their original
    # positions are available (freed during displacement), so yk=1 is always feasible.
    if forced_k:
        for k in forced_k:
            if k in yk:
                mdl.add_constraint(yk[k] == 1, ctname=f'forced_k{k}')
        if verbose:
            print(f"  Forced servers (yk=1): {len(forced_k)}")

    # === Objective: minimize weighted unfinished + tardiness penalty ===
    # Tardiness penalty discourages CPLEX from placing servers after their due dates.
    # Weight is small enough that placing tardy is always better than not placing:
    #   max_tardy * weight < min(cost_list) → 65 * 0.01 = 0.65 < 1
    DueTime = cplex_inputs.get('DueTime', [total_time] * len(K))
    tardiness_weight = 0.01
    tardiness_terms = []
    for k in K:
        ptk = PTk[k-1]
        due_k = DueTime[k-1]
        for t in feasible_times.get(k, []):
            end_t = t + ptk - 1
            if end_t > due_k and (k, t) in c3:
                tardy_amount = end_t - due_k
                tardiness_terms.append(c3[k, t] * tardy_amount)

    if tardiness_terms:
        mdl.minimize(
            mdl.sum((1 - yk[k]) * cost_list[k - 1] for k in K)
            + tardiness_weight * mdl.sum(tardiness_terms))
        if verbose:
            print(f"  Tardiness penalty: {len(tardiness_terms)} tardy start-time vars, weight={tardiness_weight}")
    else:
        mdl.minimize(mdl.sum((1 - yk[k]) * cost_list[k - 1] for k in K))

    # === Constraints ===

    # 1. One server per cell per time (only for time-cell combos with variables)
    # Group feasible tuples by (t, i, j)
    from collections import defaultdict
    tij_to_servers = defaultdict(list)
    for (k, t, i, j) in feasible_tuples:
        tij_to_servers[(t, i, j)].append(k)

    for (t, i, j), servers_at_tij in tij_to_servers.items():
        if len(servers_at_tij) > 1:
            mdl.add_constraint(
                mdl.sum(x[k, t, i, j] for k in servers_at_tij) <= 1,
                ctname=f'one_per_cell_t{t}_i{i}_j{j}')

    # 2. Each server uses at most (1+OTk) cells per time per bank
    # Group by (k, t, i)
    kti_to_cells = defaultdict(list)
    for (k, t, i, j) in feasible_tuples:
        kti_to_cells[(k, t, i)].append(j)

    for (k, t, i), js in kti_to_cells.items():
        if len(js) > 1:
            mdl.add_constraint(
                mdl.sum(x[k, t, i, j] for j in js) <= (1 + OTk[k-1]),
                ctname=f'max_cells_bank_k{k}_t{t}_i{i}')

    # 3. Each server uses at most (1+OTk) cells per time across all banks
    kt_to_cells = defaultdict(list)
    for (k, t, i, j) in feasible_tuples:
        kt_to_cells[(k, t)].append((i, j))

    for (k, t), ijs in kt_to_cells.items():
        if len(ijs) > 1:
            mdl.add_constraint(
                mdl.sum(x[k, t, i, j] for (i, j) in ijs) <= (1 + OTk[k-1]),
                ctname=f'max_cells_total_k{k}_t{t}')

    # 4. Processing time constraint (linked to yk)
    for k in K:
        ft = feasible_times.get(k, [])
        if not ft:
            mdl.add_constraint(yk[k] == 0, ctname=f'no_feasible_k{k}')
            continue

        # Sum of all x variables for this server across all feasible (t,i,j)
        x_sum_terms = []
        for t in ft:
            for (i, j) in feasible_cells.get(k, set()):
                if (k, t, i, j) in feasible_tuples:
                    x_sum_terms.append(x[k, t, i, j])

        if x_sum_terms:
            mdl.add_constraint(
                mdl.sum(x_sum_terms) == PTk[k-1] * (1 + OTk[k-1]) * yk[k],
                ctname=f'proc_time_k{k}')
        else:
            mdl.add_constraint(yk[k] == 0, ctname=f'no_vars_k{k}')

    # 5. Cell assignment: ckij links server to cell (YK-CONDITIONAL)
    # sum(ckij[k,i,j]) = (1 + OTk) * yk[k]  (FIX: multiply by yk!)
    for k in K:
        fc = feasible_cells.get(k, set())
        if fc:
            mdl.add_constraint(
                mdl.sum(ckij[k, i, j] for (i, j) in fc if (k, i, j) in ckij) ==
                (1 + OTk[k-1]) * yk[k],
                ctname=f'cell_assign_k{k}')
        else:
            mdl.add_constraint(yk[k] == 0, ctname=f'no_cells_k{k}')

    # 6. x <= ckij (LINEAR replacement for add_if_then indicator!)
    # If cell (i,j) not selected for server k, no time assignment there
    for (k, t, i, j) in feasible_tuples:
        if (k, i, j) in ckij:
            mdl.add_constraint(x[k, t, i, j] <= ckij[k, i, j],
                             ctname=f'x_le_ckij_k{k}_t{t}_i{i}_j{j}')

    # 7. Adjacent cell constraint for 2TC (YK-CONDITIONAL)
    for k in K:
        if OTk[k-1] == 1:
            # c2 linearization
            for (kk, i, j) in c2_keys:
                if kk != k:
                    continue
                if (k, i, j) in ckij and (k, i, j+1) in ckij:
                    mdl.add_constraint(c2[k, i, j] <= ckij[k, i, j])
                    mdl.add_constraint(c2[k, i, j] <= ckij[k, i, j+1])
                    mdl.add_constraint(c2[k, i, j] >= ckij[k, i, j] + ckij[k, i, j+1] - 1)

            # Exactly one adjacent pair when placed (multiply by yk!)
            c2_for_k = [c2[kk, i, j] for (kk, i, j) in c2_keys if kk == k]
            if c2_for_k:
                mdl.add_constraint(
                    mdl.sum(c2_for_k) == yk[k],
                    ctname=f'adj_pair_k{k}')
            else:
                # No adjacent pairs possible -> can't place 2TC server
                mdl.add_constraint(yk[k] == 0, ctname=f'no_adj_k{k}')
        else:
            # 1TC: no adjacent pair needed
            pass

    # 8. Contiguity: c3[k,t] = 1 means server k starts at time t (YK-CONDITIONAL)
    # sum(c3[k,t]) = yk[k]  (FIX: was == 1, now conditional on yk!)
    for k in K:
        c3_for_k = [(kk, t) for (kk, t) in c3_keys if kk == k]
        if c3_for_k:
            mdl.add_constraint(
                mdl.sum(c3[kk, t] for (kk, t) in c3_for_k) == yk[k],
                ctname=f'one_start_k{k}')
        else:
            mdl.add_constraint(yk[k] == 0, ctname=f'no_start_k{k}')

    # 9. c3 -> ckt linkage (which time blocks are active)
    # If c3[k,t]=1, then ckt[k,tt]=1 for tt in [t, t+PTk-1], 0 otherwise
    for k in K:
        ptk = PTk[k-1]
        c3_for_k = [(t) for (kk, t) in c3_keys if kk == k]
        ft = feasible_times.get(k, [])
        ft_set = set(ft)

        for start_t in c3_for_k:
            # If this start time is chosen:
            # Active times = [start_t, start_t + ptk - 1]
            # Inactive times = everything else in feasible_times

            for tt in ft:
                if (k, tt) in ckt:
                    if start_t <= tt < start_t + ptk:
                        # Should be active
                        mdl.add_constraint(
                            ckt[k, tt] >= c3[k, start_t],
                            ctname=f'ckt_on_k{k}_s{start_t}_t{tt}')
                    else:
                        # Should be inactive
                        mdl.add_constraint(
                            ckt[k, tt] <= 1 - c3[k, start_t],
                            ctname=f'ckt_off_k{k}_s{start_t}_t{tt}')

    # 10. x <= ckt (LINEAR replacement for add_if_then indicator!)
    for (k, t, i, j) in feasible_tuples:
        if (k, t) in ckt:
            mdl.add_constraint(x[k, t, i, j] <= ckt[k, t],
                             ctname=f'x_le_ckt_k{k}_t{t}_i{i}_j{j}')

    # 11. Block list constraints (if available)
    if cplex_inputs.get('block_list_path') and os.path.exists(cplex_inputs['block_list_path']):
        try:
            df_block = pd.read_excel(cplex_inputs['block_list_path'])
            unit_time_length = cplex_inputs['unit_time_length']
            i_ = df_block['i'].to_numpy().flatten()
            j_ = df_block['j'].to_numpy().flatten()
            st_ = df_block['st'].to_numpy().flatten()
            tt_ = df_block['tt'].to_numpy().flatten()

            week_length = int((5 * 24) / unit_time_length)
            for n in range(len(i_)):
                start_time = (st_[n] - 1) * week_length + 1
                start_time = max(start_time, 1)
                end_time = tt_[n] * week_length
                end_time = min(end_time, T[-1])

                bi = int(i_[n])
                bj = int(j_[n])
                for t in range(int(start_time), int(end_time) + 1):
                    for k in K:
                        if (k, t, bi, bj) in feasible_tuples:
                            mdl.add_constraint(x[k, t, bi, bj] == 0)
        except Exception as e:
            if verbose:
                print(f"  WARNING: Could not load block_list: {e}")

    # === MIP Warm-start ===
    if warmstart_values:
        try:
            warmstart = mdl.new_solution()
            ws_count = 0
            for cplex_k, hint in warmstart_values.items():
                bank = hint.get('bank')
                cell = hint.get('cell')
                start = hint.get('start')
                end = hint.get('end')

                # Skip incomplete hints (e.g., RL assignments without bank/cell)
                if bank is None or cell is None or start is None or end is None:
                    continue

                warmstart.add_var_value(yk[cplex_k], 1)

                # Set ckij
                if (cplex_k, bank, cell) in ckij:
                    warmstart.add_var_value(ckij[cplex_k, bank, cell], 1)

                # Set c3 (start time)
                if (cplex_k, start) in c3:
                    warmstart.add_var_value(c3[cplex_k, start], 1)

                # Set ckt and x
                otk = OTk[cplex_k - 1]
                for t in range(start, end + 1):
                    if (cplex_k, t) in ckt:
                        warmstart.add_var_value(ckt[cplex_k, t], 1)
                    if (cplex_k, t, bank, cell) in feasible_tuples:
                        warmstart.add_var_value(x[cplex_k, t, bank, cell], 1)
                    # 2TC adjacent
                    if otk == 1 and (cplex_k, bank, cell + 1) in ckij:
                        warmstart.add_var_value(ckij[cplex_k, bank, cell + 1], 1)
                        if (cplex_k, t, bank, cell + 1) in feasible_tuples:
                            warmstart.add_var_value(x[cplex_k, t, bank, cell + 1], 1)

                ws_count += 1

            mdl.add_mip_start(warmstart)
            if verbose:
                print(f"  Warm-start: {ws_count} server placements provided as hints")
        except Exception as e:
            if verbose:
                print(f"  WARNING: Warm-start failed: {e}")

    # === Solve ===
    if verbose:
        num_constraints = mdl.number_of_constraints
        print(f"  Constraints: {num_constraints:,}")
        # mdl.print_information()

    solve_start = time.time()
    try:
        solution = mdl.solve(log_output=verbose)
    except Exception as e:
        error_str = str(e)
        if 'Community Edition' in error_str or 'DOcplexLimits' in type(e).__name__:
            print(f"  CPLEX Community Edition limit hit. Using iterative solve...")
            return _solve_iterative(cplex_inputs, time_limit, verbose)
        else:
            print(f"  CPLEX solve error: {e}")
            return None
    solve_time = time.time() - solve_start

    if solution is None:
        if verbose:
            print(f"  CPLEX: No solution found in {solve_time:.1f}s")
            print(f"  Status: {mdl.get_solve_status()}")
        return None

    # === Parse solution ===
    assignments = {}
    for k in K:
        if yk[k].solution_value > 0.5:
            bank_assigned = None
            cell_assigned = None
            start_t = None
            end_t = None

            is_2tc = OTk[k-1] == 1

            if is_2tc:
                # For 2TC: use c2 variable to identify the ACTUAL primary cell.
                # c2[k,i,j] = 1 means (i,j) is the primary and (i,j+1) is adjacent.
                # This avoids the ambiguity where a position can be BOTH a valid
                # primary (for one pair) AND adjacent (for another pair).
                for (kk, i, j) in c2_keys:
                    if kk == k and c2[kk, i, j].solution_value > 0.5:
                        bank_assigned = i
                        cell_assigned = j
                        break

                if bank_assigned is not None:
                    # Get time range from x-vars for the primary cell
                    for (kk, t, i, j) in feasible_tuples:
                        if kk == k and i == bank_assigned and j == cell_assigned:
                            if x[k, t, i, j].solution_value > 0.5:
                                if start_t is None:
                                    start_t = t
                                    end_t = t
                                else:
                                    start_t = min(start_t, t)
                                    end_t = max(end_t, t)
            else:
                # For 1TC: original logic (no ambiguity)
                for (kk, t, i, j) in feasible_tuples:
                    if kk != k:
                        continue
                    if x[k, t, i, j].solution_value > 0.5:
                        if bank_assigned is None:
                            bank_assigned = i
                            cell_assigned = j
                            start_t = t
                            end_t = t
                        else:
                            if i == bank_assigned and j == cell_assigned:
                                end_t = max(end_t, t)

            if bank_assigned is not None and start_t is not None:
                original_id = server_id_map[k]
                assignments[original_id] = {
                    'bank': bank_assigned,
                    'cell': cell_assigned,
                    'start': start_t,
                    'end': end_t,
                    '_2tc': is_2tc,
                }

    # === Validate: check for internal conflicts between CPLEX assignments ===
    _validate_cplex_assignments(assignments, verbose)

    if verbose:
        print(f"  CPLEX: {len(assignments)}/{len(K)} servers placed "
              f"in {solve_time:.1f}s (gap: {mdl.solve_details.mip_relative_gap:.4f})")

    return assignments


def _validate_cplex_assignments(assignments, verbose=False):
    """Check that CPLEX assignments don't overlap with each other.

    This detects parser bugs: if two assignments claim the same cell-time,
    the parser mis-read the CPLEX solution.
    """
    # Build occupancy map from just the CPLEX assignments
    cell_time_owner = {}  # (bank, pos, t) -> server_id

    conflicts = []
    for sid, asn in assignments.items():
        bank = asn['bank']
        cell = asn['cell']
        start = asn['start']
        end = asn['end']
        is_2tc = asn.get('_2tc', False)

        cells_to_check = [(bank, cell)]
        if is_2tc:
            cells_to_check.append((bank, cell + 1))

        for (b, c) in cells_to_check:
            for t in range(start, end + 1):
                key = (b, c, t)
                if key in cell_time_owner:
                    other = cell_time_owner[key]
                    conflicts.append((sid, other, b, c, t))
                else:
                    cell_time_owner[key] = sid

    if conflicts and verbose:
        print(f"  PARSER VALIDATION: {len(conflicts)} internal conflicts "
              f"in CPLEX assignments!")
        for sid1, sid2, b, c, t in conflicts[:10]:
            asn1 = assignments[sid1]
            asn2 = assignments[sid2]
            print(f"    Conflict: server {sid1} ({asn1['bank']},{asn1['cell']}"
                  f" t={asn1['start']}-{asn1['end']} 2tc={asn1.get('_2tc')}) "
                  f"vs server {sid2} ({asn2['bank']},{asn2['cell']}"
                  f" t={asn2['start']}-{asn2['end']} 2tc={asn2.get('_2tc')}) "
                  f"at cell ({b},{c}) t={t}")

    return len(conflicts) == 0


def _solve_iterative(cplex_inputs, total_time_limit=300, verbose=True):
    """Iterative batched solve for large problems or Community Edition.

    Solves in batches of ~80 servers, sorted by tightest deadline.
    After each batch, MERGES results into env and recomputes blocked cells
    from env.cell_occupancy (ensures consistency between batches).
    """
    if verbose:
        print("  Iterative solve: splitting into batches...")

    env = cplex_inputs['_env']
    facility = cplex_inputs['_facility']
    servers = cplex_inputs['servers']
    total_time = cplex_inputs['total_time']
    idx_to_bc, _ = _build_idx_to_bank_cell(env)

    # Sort by tightest deadline (urgency)
    sorted_servers = sorted(servers,
        key=lambda s: s.get('DueTime', total_time) - s['ArrivalTime'] - s['PTk'])

    batch_size = 80
    batches = []
    for i in range(0, len(sorted_servers), batch_size):
        batches.append(sorted_servers[i:i + batch_size])

    all_assignments = {}
    per_batch_limit = max(60, total_time_limit // len(batches))

    for batch_idx, batch in enumerate(batches):
        # Skip servers already placed by previous batches
        batch = [s for s in batch if s['k'] not in env.assignments]
        if not batch:
            continue

        if verbose:
            print(f"  Batch {batch_idx + 1}/{len(batches)}: "
                  f"{len(batch)} servers, time limit {per_batch_limit}s")

        # Recompute blocked cells from ACTUAL env state (includes all previous batch merges)
        current_blocked, _ = _get_blocked_cell_times(env, idx_to_bc)

        # Pass through forced_server_ids for this batch
        parent_forced = cplex_inputs.get('forced_server_ids_original', set())
        batch_forced = {s['k'] for s in batch} & parent_forced if parent_forced else None

        sub_inputs = build_cplex_residual(
            env, facility,
            batch, current_blocked, cplex_inputs['fab_blocks'],
            cplex_inputs['VH'], cplex_inputs['VL'], cplex_inputs['W'],
            unit_time_length=cplex_inputs['unit_time_length'],
            forced_server_ids=batch_forced)

        if sub_inputs is None:
            continue

        # Carry over internal refs needed by recursive iterative
        sub_inputs['_env'] = env
        sub_inputs['_facility'] = facility

        batch_result = solve_residual_with_cplex(
            sub_inputs, time_limit=per_batch_limit, verbose=False)

        if batch_result:
            all_assignments.update(batch_result)
            # MERGE into env immediately so next batch sees updated cell_occupancy
            merge_cplex_assignments(env, batch_result)

    if verbose:
        print(f"  Iterative: {len(all_assignments)} total servers placed")

    return all_assignments if all_assignments else None


def merge_cplex_assignments(env, cplex_assignments):
    """Merge CPLEX assignments back into RL environment state.

    Idempotent: silently skips servers already in env.assignments.
    """
    if not cplex_assignments:
        return 0

    _, bc_to_idx = _build_idx_to_bank_cell(env)

    placed_count = 0
    for server_id, asn in cplex_assignments.items():
        # Skip already-placed servers (idempotent for iterative merge)
        if server_id in env.assignments:
            continue

        bank = asn['bank']
        cell_pos = asn['cell']
        start = asn['start']
        end = asn['end']

        cell_idx = bc_to_idx.get((bank, cell_pos))
        if cell_idx is None:
            print(f"  WARNING: No cell_idx for bank={bank}, pos={cell_pos}")
            continue

        # Look up server data for 2TC check
        server_data = None
        original_count = len(env.servers) - len(getattr(env, '_retry_queue', []))
        for si in range(original_count):
            if env.servers[si]['k'] == server_id:
                server_data = env.servers[si]
                break

        is_2tc = server_data and server_data.get('OTk', 0) == 1
        adj_idx = bc_to_idx.get((bank, cell_pos + 1)) if is_2tc else None

        # Verify no conflicts on PRIMARY cell
        conflict = False
        for t in range(start, end + 1):
            if t < env.cell_occupancy.shape[1] and env.cell_occupancy[cell_idx, t] > 0:
                existing = int(env.cell_occupancy[cell_idx, t])
                if existing != server_id:
                    print(f"  WARNING: Conflict at cell_idx={cell_idx}, t={t} "
                          f"(occupied by {existing}, trying to place {server_id})")
                    conflict = True
                    break

        # Also check ADJACENT cell for 2TC
        if not conflict and is_2tc and adj_idx is not None:
            for t in range(start, end + 1):
                if t < env.cell_occupancy.shape[1] and env.cell_occupancy[adj_idx, t] > 0:
                    existing = int(env.cell_occupancy[adj_idx, t])
                    if existing != server_id:
                        print(f"  WARNING: Adj conflict at adj_idx={adj_idx}, t={t} "
                              f"(occupied by {existing}, trying to place 2TC {server_id})")
                        conflict = True
                        break

        if conflict:
            print(f"    DEBUG merge: sid={server_id}, bank={bank}, cell={cell_pos}, "
                  f"start={start}, end={end}, 2tc={is_2tc}, adj_idx={adj_idx}")
            continue

        # Update cell occupancy (primary)
        for t in range(start, end + 1):
            if t < env.cell_occupancy.shape[1]:
                env.cell_occupancy[cell_idx, t] = server_id

        # Update adjacent cell for 2TC
        if is_2tc and adj_idx is not None:
            for t in range(start, end + 1):
                if t < env.cell_occupancy.shape[1]:
                    env.cell_occupancy[adj_idx, t] = server_id

        # Update assignments dict
        env.assignments[server_id] = {
            'cell_idx': cell_idx,
            'bank': bank,
            'cell': cell_pos,
            'start': start,
            'end': end,
        }

        if hasattr(env, 'assigned_servers'):
            for si, s in enumerate(env.servers):
                if s['k'] == server_id:
                    env.assigned_servers.add(si)
                    break

        placed_count += 1

    return placed_count



def run_hybrid_evaluation(env, facility, completion_rate, tardiness_rate,
                          threshold_completion=0.90,
                          threshold_tardiness=0.05,
                          cplex_time_limit=300,
                          unit_time_length=UNIT_TIME,
                          verbose=True):
    """Top-level hybrid evaluation: RL + CPLEX residual solve.

    After RL places servers, CPLEX solves a residual MIP for any unplaced
    servers while keeping all RL placements locked as frozen constraints.
    This preserves non-preemption: no RL-placed server is ever displaced.

    Args:
        env: LexicographicRewardEnvironment after RL
        facility: FacilityConfiguration
        completion_rate: Current completion rate (0-1)
        tardiness_rate: Current tardiness rate (0-1)
        threshold_completion: Min acceptable completion (below triggers CPLEX)
        threshold_tardiness: Max acceptable tardiness (above triggers CPLEX)
        cplex_time_limit: CPLEX time limit in seconds
        unit_time_length: Time granularity
        verbose: Print progress

    Returns:
        dict with hybrid result metrics
    """
    # Use original server count (excluding retry-queue duplicates)
    total_servers = len(env.servers) - len(getattr(env, '_retry_queue', []))

    result = {
        'cplex_called': False,
        'cplex_additional_placed': 0,
        'cplex_runtime_sec': 0.0,
        'final_completion_rate': completion_rate,
        'final_tardiness_rate': tardiness_rate,
        'unplaced_before_cplex': total_servers - len(env.assignments),
        'unplaced_after_cplex': total_servers - len(env.assignments),
    }

    if not should_call_cplex(completion_rate, tardiness_rate,
                              threshold_completion, threshold_tardiness):
        if verbose:
            print(f"  Hybrid: No CPLEX needed "
                  f"(completion={completion_rate:.1%}, "
                  f"tardiness={tardiness_rate:.1%})")
        return result

    if verbose:
        print(f"  Hybrid: Triggering CPLEX "
              f"(completion={completion_rate:.1%} < {threshold_completion:.1%})")

    # === Setup ===
    idx_to_bc, bc_to_idx = _build_idx_to_bank_cell(env)
    VH, VL, W = _build_cell_capability_matrices(facility)
    fab_blocks = _get_fab_blocks(env, idx_to_bc)

    cplex_start = time.time()

    # ================================================================
    # CPLEX Residual Solve (RL placements LOCKED)
    # ================================================================
    unplaced = extract_unplaced_servers(env)
    if not unplaced:
        if verbose:
            print(f"  Hybrid: No unplaced servers")
        return result

    if verbose:
        print(f"\n  === CPLEX Residual Solve ({len(unplaced)} unplaced, RL locked) ===")

    blocked_cells, blocked_by_server = _get_blocked_cell_times(env, idx_to_bc)

    cplex_inputs = build_cplex_residual(
        env, facility, unplaced, blocked_cells, fab_blocks,
        VH, VL, W, unit_time_length=unit_time_length)

    cplex_assignments = None
    if cplex_inputs:
        # Store env/facility refs for potential iterative solve
        cplex_inputs['_env'] = env
        cplex_inputs['_facility'] = facility

        # Adaptive time limit: more time for more servers
        solve_limit = min(cplex_time_limit, max(120, len(unplaced) * 10))

        if len(unplaced) > 100:
            # Very large problem: use iterative approach
            if verbose:
                print(f"  Large residual ({len(unplaced)} servers) -> iterative solve")
            cplex_assignments = _solve_iterative(cplex_inputs, solve_limit, verbose)
        else:
            cplex_assignments = solve_residual_with_cplex(
                cplex_inputs, time_limit=solve_limit, verbose=verbose)

    # Merge CPLEX results
    cplex_placed = 0
    if cplex_assignments:
        cplex_placed = merge_cplex_assignments(env, cplex_assignments)

    cplex_time = time.time() - cplex_start

    if verbose:
        final_completion = len(env.assignments) / total_servers
        print(f"  CPLEX result: +{cplex_placed} servers, "
              f"completion {final_completion:.1%}, time {cplex_time:.1f}s")

    result.update(_compute_final_metrics(env, total_servers, cplex_placed, cplex_time))
    return result


def _compute_final_metrics(env, total_servers, additional_placed, cplex_time):
    """Compute final metrics after CPLEX residual solve."""
    final_assigned = len(env.assignments)
    final_completion = final_assigned / total_servers if total_servers > 0 else 1.0

    tardy_count = 0
    for server_id, asn in env.assignments.items():
        server_data = None
        for s in env.servers:
            if s['k'] == server_id:
                server_data = s
                break
        if server_data:
            due = server_data.get('DueTime', env.total_time)
            if asn['end'] > due:
                tardy_count += 1
    final_tardiness = tardy_count / final_assigned if final_assigned > 0 else 0.0

    return {
        'cplex_called': True,
        'cplex_additional_placed': additional_placed,
        'cplex_runtime_sec': cplex_time,
        'final_completion_rate': final_completion,
        'final_tardiness_rate': final_tardiness,
        'unplaced_after_cplex': total_servers - final_assigned,
    }


# =============================================================================
# STANDALONE TEST
# =============================================================================

def _test_hybrid():
    """Quick standalone test of the hybrid pipeline."""
    print("=" * 60)
    print("HYBRID CPLEX RESIDUAL SOLVER - STANDALONE TEST")
    print("=" * 60)

    try:
        from daf_lhac_core import (
            FacilityConfiguration, LexicographicRewardEnvironment,
            MultiAgentLHACPPO, DATA_DIR
        )
    except ImportError:
        print("ERROR: Cannot import daf_lhac_core. Run from DAF-LHAC/ directory.")
        return

    import glob
    datasets = glob.glob(os.path.join(DATA_DIR, '*.xlsx'))
    if not datasets:
        print("ERROR: No datasets found in Data/")
        return

    test_file = datasets[0]
    print(f"\nTest dataset: {os.path.basename(test_file)}")

    facility = FacilityConfiguration()
    df = pd.read_excel(test_file)
    print(f"  Servers: {len(df)}")

    env = LexicographicRewardEnvironment(
        facility, df,
        window_length=10,
        use_windowing=True,
        use_deferred_actions=False)
    env.reset()

    # Simulate partial RL: assign first 50%
    import random
    random.seed(42)
    total = len(env.servers)
    to_assign = total // 2

    print(f"  Simulating partial RL: placing {to_assign}/{total} servers")

    placed = 0
    for si in range(total):
        if placed >= to_assign:
            break
        server = env.servers[si]
        valid = env.get_valid_actions()
        valid_cells = [a for a in range(env.total_cells) if valid[a] == 1]
        if valid_cells:
            action = random.choice(valid_cells)
            env.step(action)
            placed += 1
        else:
            env.step(env.total_cells)

    while not env._is_done():
        env.step(env.total_cells)

    completion = len(env.assignments) / total
    print(f"  RL completion: {completion:.1%} ({len(env.assignments)}/{total})")

    result = run_hybrid_evaluation(
        env, facility,
        completion_rate=completion,
        tardiness_rate=0.0,
        threshold_completion=1.0,
        cplex_time_limit=120,
        verbose=True)

    print(f"\n  Result: {result}")
    print("\nTest complete.")


if __name__ == '__main__':
    _test_hybrid()
