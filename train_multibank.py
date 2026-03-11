#!/usr/bin/env python3
"""
train_multibank.py - Train LHAC-DAF models for different bank configurations.

Trains the full DAF-LHAC (DAP + CASE + FPR) model for 1-bank, 2-bank, or 4-bank
configurations using BankFilteredEnvironment to mask out cells from inactive banks.

Key design:
  - Same 55-action architecture (54 FUL cells + 1 skip) for ALL bank configs
  - BankFilteredEnvironment masks out cells from inactive banks via valid-action filtering
  - State dim: 78 (same for all configs — inactive bank cells show empty capacity)
  - Action masking during training teaches the model to ONLY use available cells
  - Curriculum learning: small -> medium -> large problems

Bank configurations:
  1 bank  = 14 cells (bank 0 only)
  2 banks = 28 cells (banks 0-1)
  4 banks = 54 cells (banks 0-3, with positions 4,7 excluded from bank 3)

Usage:
  python train_multibank.py --num-banks 1 --episodes 25000
  python train_multibank.py --num-banks 2 --episodes 25000
  python train_multibank.py --num-banks 4 --episodes 25000       # Same as original
  python train_multibank.py --num-banks 1 --episodes 5000 --quick  # Quick test
  python train_multibank.py --num-banks 2 --resume Models/daf_full_daf_2bank_ckpt_10000.pth --resume-episode 10000

Rolling window training:
  python train_multibank.py --num-banks 1 --rolling-window --window-days 10
  (Uses rolling window during training so model learns window-boundary behavior)
"""

import os
import sys
import time
import random
import copy
import math
import argparse
import numpy as np
import torch

# Ensure this folder's modules are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from daf_lhac_core import (
    FacilityConfiguration, LexicographicRewardEnvironment,
    MultiAgentLHACPPO, CrossAttentionPPONetwork,
    CellStatus, generate_orders, TOTAL_TIME_BLOCKS,
    MODEL_DIR, DATA_DIR, BASE_DIR
)

# Import BankFilteredEnvironment from evaluate_multibank
from evaluate_multibank import BankFilteredEnvironment, BANK_CONFIGS


# ============================================================================
# Dataset loading (same as train_daf.py but uses local Data/ symlink)
# ============================================================================

def load_training_datasets(data_dir=None):
    """Load all available datasets for training."""
    import pandas as pd

    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, 'Data', 'benchmarks')
        if not os.path.exists(data_dir):
            data_dir = DATA_DIR

    datasets = []

    # Load from benchmarks directory
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if not f.endswith('.xlsx'):
                continue
            path = os.path.join(data_dir, f)
            try:
                df = pd.read_excel(path)
                required = ['k', 'PTk', 'OTk', 'RH', 'RW', 'DueTime', 'ArrivalTime']
                if all(c in df.columns for c in required):
                    datasets.append(df)
            except Exception:
                pass

    # Also look for Parvez datasets
    parvez_dirs = [
        os.path.join(SCRIPT_DIR, '..', '..', 'DATA_GOLALI', 'Input_data'),
        os.path.join(os.path.dirname(DATA_DIR), 'DATA_GOLALI', 'Input_data'),
    ]
    for parvez_root in parvez_dirs:
        for subdir in ['Back', 'Front', 'Uniform']:
            parvez_dir = os.path.join(parvez_root, subdir)
            if os.path.exists(parvez_dir):
                for f in sorted(os.listdir(parvez_dir)):
                    if not f.endswith('.xlsx'):
                        continue
                    path = os.path.join(parvez_dir, f)
                    try:
                        df = pd.read_excel(path)
                        required = ['k', 'PTk', 'OTk', 'RH', 'RW', 'DueTime', 'ArrivalTime']
                        if all(c in df.columns for c in required):
                            datasets.append(df)
                    except Exception:
                        pass

    return datasets


def create_synthetic_datasets(num_datasets=200, num_banks=4):
    """Generate synthetic training datasets.

    For fewer banks, we generate proportionally smaller problems since
    there are fewer cells available.
    """
    datasets = []
    patterns = ['back', 'front', 'uniform']

    # Adjust max size based on bank count:
    # 4 banks (54 cells) -> up to 400 servers
    # 2 banks (28 cells) -> up to 220 servers (roughly proportional)
    # 1 bank  (14 cells) -> up to 120 servers
    num_cells = BANK_CONFIGS[num_banks]['num_cells']
    max_size = min(400, int(num_cells * 7.5))  # ~7.5 servers per cell max
    min_size = 30

    sizes = list(range(min_size, max_size + 1, 10))
    two_tc_pcts = [0.1, 0.15, 0.2, 0.25, 0.3]

    for i in range(num_datasets):
        size = random.choice(sizes)
        pattern = random.choice(patterns)
        two_tc = random.choice(two_tc_pcts)
        df = generate_orders(size, pattern, two_tc, 1000 + i)
        datasets.append(df)

    # Hard instances (scaled by capacity)
    hard_sizes = sorted(set([
        max_size,
        int(max_size * 0.875),
        int(max_size * 0.75),
        int(max_size * 0.625),
    ]))
    hard_configs = []
    for sz in hard_sizes:
        for pat in patterns:
            for tc in [0.10, 0.20, 0.30]:
                hard_configs.append((sz, pat, tc))

    for size, pattern, two_tc in hard_configs:
        for seed_offset in range(20):
            df = generate_orders(size, pattern, two_tc, 5000 + seed_offset)
            datasets.append(df)

    return datasets


def create_variations(real_dfs, num_variations=200, num_banks=4):
    """Create variations of real datasets, scaled for bank count."""
    import pandas as pd
    variations = []
    if not real_dfs:
        return variations

    num_cells = BANK_CONFIGS[num_banks]['num_cells']
    max_size = min(400, int(num_cells * 7.5))

    for i in range(num_variations):
        base_df = random.choice(real_dfs)
        lo = max(30, min(len(base_df) // 3, max_size))
        hi = min(len(base_df), max_size)
        if lo > hi:
            lo = hi
        sample_size = random.randint(lo, hi)
        if sample_size >= len(base_df):
            var_df = base_df.copy()
        else:
            indices = sorted(random.sample(range(len(base_df)), sample_size))
            var_df = base_df.iloc[indices].copy().reset_index(drop=True)
        var_df['k'] = range(1, len(var_df) + 1)
        if random.random() < 0.3:
            noise = np.random.randint(-2, 3, size=len(var_df))
            var_df['ArrivalTime'] = np.clip(
                var_df['ArrivalTime'] + noise, 1, 60).astype(int)
        variations.append(var_df)
    return variations


# ============================================================================
# Curriculum manager (adapted for variable bank counts)
# ============================================================================

class CurriculumManager:
    """Gradually increase training complexity, scaled for bank count."""

    def __init__(self, all_datasets, num_banks=4):
        self.all_training_data = all_datasets
        self.num_banks = num_banks
        num_cells = BANK_CONFIGS[num_banks]['num_cells']
        self.max_size = min(400, int(num_cells * 7.5))

        self.data_by_size = {}
        for df in all_datasets:
            size = len(df)
            if size not in self.data_by_size:
                self.data_by_size[size] = []
            self.data_by_size[size].append(df)

    def get_training_data(self, episode):
        # Scale targets proportionally to bank capacity
        scale = self.max_size / 400.0  # 1.0 for 4-bank, ~0.55 for 2-bank, ~0.3 for 1-bank

        if episode < 500:
            target = random.randint(30, max(30, int(60 * scale)))
        elif episode < 1000:
            target = random.randint(max(30, int(50 * scale)), max(40, int(120 * scale)))
        elif episode < 2000:
            target = random.randint(max(40, int(80 * scale)), max(60, int(250 * scale)))
        elif episode < 3000:
            target = random.randint(max(60, int(150 * scale)), max(80, int(350 * scale)))
        elif episode < 5000:
            target = random.randint(max(80, int(200 * scale)), self.max_size)
        else:
            if random.random() < 0.8:
                target = random.randint(max(100, int(300 * scale)), self.max_size)
            else:
                target = random.randint(max(40, int(100 * scale)), self.max_size)

        if target in self.data_by_size:
            return random.choice(self.data_by_size[target])

        tolerance = int(target * 0.2)
        for size in range(max(30, target - tolerance),
                          min(self.max_size + 1, target + tolerance + 1)):
            if size in self.data_by_size:
                return random.choice(self.data_by_size[size])

        larger = [df for df in self.all_training_data if len(df) >= target]
        if larger:
            base = random.choice(larger)
            indices = sorted(random.sample(range(len(base)), min(target, len(base))))
            sub = base.iloc[indices].copy().reset_index(drop=True)
            sub['k'] = range(1, len(sub) + 1)
            return sub

        return random.choice(self.all_training_data)


# ============================================================================
# Rolling window training helper
# ============================================================================

def partition_servers_by_day(df, window_days, total_days=None):
    """Partition servers into rolling windows by arrival day."""
    if total_days is None:
        total_days = TOTAL_TIME_BLOCKS
    n_windows = math.ceil(total_days / window_days)
    windows = [[] for _ in range(n_windows)]
    for idx in range(len(df)):
        arr_day = int(df.iloc[idx]['ArrivalTime'])
        w_idx = min((arr_day - 1) // window_days, n_windows - 1)
        windows[w_idx].append(idx)
    return windows


def train_episode_rolling(agent, facility, df, window_days, num_banks=4,
                           use_dap=False, use_fpr=False):
    """Train one episode using rolling windows with carry-over.

    This teaches the model to handle window boundaries during training,
    matching how it will be evaluated.
    """
    windows = partition_servers_by_day(df, window_days)

    # Create a full environment to track occupancy across windows
    if num_banks < 4:
        full_env = BankFilteredEnvironment(
            facility, df, num_banks=num_banks, quiet=True,
            window_length=window_days, use_windowing=True,
            randomize_order=True,
            use_deferred_actions=use_dap, use_fpr=use_fpr)
    else:
        full_env = LexicographicRewardEnvironment(
            facility, df,
            window_length=window_days, use_windowing=True,
            randomize_order=True,
            use_deferred_actions=use_dap, use_fpr=use_fpr)

    total_r1, total_r2 = 0.0, 0.0
    total_assigned = set()
    carry_over_indices = []

    for w_idx, server_indices in enumerate(windows):
        if not server_indices and not carry_over_indices:
            continue

        # Combine current window servers + carry-over from previous windows
        all_indices = carry_over_indices + server_indices
        if not all_indices:
            continue

        # Create sub-dataframe for this window
        sub_df = df.iloc[all_indices].copy().reset_index(drop=True)

        # Create sub-environment
        if num_banks < 4:
            sub_env = BankFilteredEnvironment(
                facility, sub_df, num_banks=num_banks, quiet=True,
                window_length=window_days, use_windowing=True,
                randomize_order=True,
                use_deferred_actions=use_dap, use_fpr=use_fpr)
        else:
            sub_env = LexicographicRewardEnvironment(
                facility, sub_df,
                window_length=window_days, use_windowing=True,
                randomize_order=True,
                use_deferred_actions=use_dap, use_fpr=use_fpr)

        # Copy occupancy from full_env to sub_env
        sub_env.cell_occupancy = np.copy(full_env.cell_occupancy)

        # Run RL on this window
        state, _ = sub_env.reset()
        done = False
        while not done:
            if use_dap:
                valid_actions = sub_env.get_valid_actions_deferred()
            else:
                valid_actions = sub_env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=True)
            next_state, rewards, done, _, info = sub_env.step(action)
            r1, r2 = rewards
            agent.store_transition(state, action, r1, r2, done, valid_actions)
            total_r1 += r1
            total_r2 += r2
            state = next_state

        # Copy occupancy back to full_env
        full_env.cell_occupancy = np.copy(sub_env.cell_occupancy)

        # Track assigned servers (map sub indices back to original)
        for sub_idx in sub_env.assigned_servers:
            orig_idx = all_indices[sub_idx]
            total_assigned.add(orig_idx)

        # Carry over unassigned servers to next window
        carry_over_indices = []
        for sub_idx in range(len(sub_df)):
            if sub_idx not in sub_env.assigned_servers:
                orig_idx = all_indices[sub_idx]
                # Only carry if server can still fit in remaining time
                srv = df.iloc[orig_idx]
                arr = int(srv['ArrivalTime'])
                pt = int(srv['PTk'])
                if arr + pt <= TOTAL_TIME_BLOCKS + 1:
                    carry_over_indices.append(orig_idx)

    total_servers = len(df)
    completion_rate = len(total_assigned) / total_servers if total_servers > 0 else 0
    tardy = 0  # Would need to compute if needed

    info = {
        'completion_rate': completion_rate,
        'total_servers': total_servers,
        'unfinished_servers': total_servers - len(total_assigned),
        'tardy_servers': tardy,
    }

    return total_r1, total_r2, info


# ============================================================================
# Training loop
# ============================================================================

def train_multibank(agent, facility, curriculum_manager,
                     num_banks=4, num_episodes=25000,
                     window_length=10, use_rolling=False, window_days=10,
                     save_path=None, checkpoint_interval=2000,
                     eval_interval=1000, eval_datasets=None,
                     use_dap=True, use_fpr=True, variant_name='full_daf',
                     start_episode=0):
    """Train LHAC-DAF with bank-filtered environment.

    Args:
        num_banks: Number of active banks (1, 2, or 4)
        use_rolling: If True, use rolling window training (matches evaluation)
        window_days: Days per window when use_rolling=True
    """
    bank_label = BANK_CONFIGS[num_banks]['label']
    num_cells = BANK_CONFIGS[num_banks]['num_cells']

    print(f"\n{'='*60}")
    print(f"LHAC-DAF TRAINING: {variant_name} | {bank_label}")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes} (starting from {start_episode})")
    print(f"Banks: {num_banks} ({num_cells} cells)")
    print(f"Window length: {window_length}")
    print(f"Rolling window: {'ON (' + str(window_days) + ' days)' if use_rolling else 'OFF'}")
    print(f"DAP: {'ON' if use_dap else 'OFF'}")
    print(f"FPR: {'ON' if use_fpr else 'OFF'}")
    print(f"CASE: {'ON' if agent.use_case else 'OFF'}")
    print(f"Device: {agent.device}")
    print(f"Save path: {save_path}")

    # Metrics
    completion_rates = []
    tardy_rates = []
    episode_rewards1 = []
    perfect_episodes = []
    best_eval_completion = 0.0

    phase2_activated = False
    phase2_episode = None
    start_time = time.time()
    episodes_per_update = 4

    for episode in range(start_episode, num_episodes):
        df = curriculum_manager.get_training_data(episode)

        if use_rolling:
            # Rolling window training
            episode_r1, episode_r2, info = train_episode_rolling(
                agent, facility, df, window_days,
                num_banks=num_banks,
                use_dap=use_dap, use_fpr=use_fpr)
        else:
            # Standard single-pass training with bank filtering
            if num_banks < 4:
                env = BankFilteredEnvironment(
                    facility, df, num_banks=num_banks, quiet=True,
                    window_length=window_length,
                    use_windowing=True,
                    randomize_order=True,
                    use_deferred_actions=use_dap,
                    use_fpr=use_fpr)
            else:
                env = LexicographicRewardEnvironment(
                    facility, df,
                    window_length=window_length,
                    use_windowing=True,
                    randomize_order=True,
                    use_deferred_actions=use_dap,
                    use_fpr=use_fpr)

            state, _ = env.reset()
            episode_r1 = 0
            episode_r2 = 0
            done = False

            while not done:
                if use_dap:
                    valid_actions = env.get_valid_actions_deferred()
                else:
                    valid_actions = env.get_valid_actions()
                action = agent.select_action(state, valid_actions, training=True)
                next_state, rewards, done, _, info = env.step(action)
                r1, r2 = rewards
                agent.store_transition(state, action, r1, r2, done, valid_actions)
                episode_r1 += r1
                episode_r2 += r2
                state = next_state

        # Record metrics
        episode_rewards1.append(episode_r1)
        completion_rates.append(info['completion_rate'])
        tardy_rate = (info['tardy_servers'] / info['total_servers'] * 100
                      if info['total_servers'] > 0 else 0)
        tardy_rates.append(tardy_rate)
        if info['unfinished_servers'] == 0 and info['tardy_servers'] == 0:
            perfect_episodes.append(episode)

        # PPO update
        if (episode + 1) % episodes_per_update == 0:
            loss1, loss2 = agent.update()

        # Phase 2 activation (tardiness minimization)
        if not phase2_activated and episode > 8000:
            recent = (np.mean(completion_rates[-500:])
                      if len(completion_rates) >= 500
                      else np.mean(completion_rates))
            if recent > 0.90:
                print(f"\nPHASE 2 ACTIVATED at episode {episode}! "
                      f"Completion: {recent:.1%}")
                agent.activate_phase2()
                phase2_activated = True
                phase2_episode = episode

        # Update exploration
        agent.update_epsilon()
        agent.update_tolerance()
        agent.step_schedulers()

        # Progress report every 100 episodes
        if (episode + 1) % 100 == 0:
            recent_comp = (np.mean(completion_rates[-100:])
                           if len(completion_rates) >= 100
                           else np.mean(completion_rates))
            recent_tardy = (np.mean(tardy_rates[-100:])
                            if len(tardy_rates) >= 100
                            else np.mean(tardy_rates))
            recent_perfect = sum(1 for e in perfect_episodes
                                 if e >= episode - 100)
            elapsed = time.time() - start_time
            eps_done = episode + 1 - start_episode
            eta = ((num_episodes - episode - 1) /
                   (eps_done / elapsed) / 60) if eps_done > 0 else 0
            phase_str = "[P2]" if phase2_activated else "[P1]"
            print(f"Ep {episode+1:5d} {phase_str} | "
                  f"Comp: {recent_comp:.1%} | "
                  f"Tardy: {recent_tardy:.1f}% | "
                  f"Perfect: {recent_perfect}/100 | "
                  f"eps: {agent.epsilon:.3f} | "
                  f"tau: {agent.tolerance:.4f} | "
                  f"ETA: {eta:.1f}m | "
                  f"Banks: {num_banks}")

        # Checkpoint
        if save_path and (episode + 1) % checkpoint_interval == 0:
            ckpt = save_path.replace('.pth', f'_ckpt_{episode+1}.pth')
            agent.save(ckpt)
            print(f"  [CHECKPOINT] Saved to {ckpt}")

        # Evaluation
        if eval_datasets and (episode + 1) % eval_interval == 0:
            eval_comp = _eval_during_training(
                agent, facility, eval_datasets, window_length,
                num_banks, episode, use_dap, use_fpr)
            if eval_comp > best_eval_completion and save_path:
                best_eval_completion = eval_comp
                best_path = save_path.replace('.pth', '_best.pth')
                agent.save(best_path)
                print(f"  [NEW BEST] Eval completion: {eval_comp:.1%}")

    # Final save
    if save_path:
        os.makedirs(os.path.dirname(save_path)
                     if os.path.dirname(save_path) else '.', exist_ok=True)
        agent.save(save_path)
        meta = save_path.replace('.pth', '_meta.pth')
        torch.save({
            'num_episodes': num_episodes,
            'num_banks': num_banks,
            'num_cells': num_cells,
            'completion_rates': completion_rates,
            'tardy_rates': tardy_rates,
            'perfect_episodes': perfect_episodes,
            'phase2_activated': phase2_activated,
            'phase2_episode': phase2_episode,
            'window_length': window_length,
            'use_rolling': use_rolling,
            'window_days': window_days,
            'variant': variant_name,
            'use_dap': use_dap,
            'use_fpr': use_fpr,
            'use_case': agent.use_case,
            'agent_type': 'multi_agent_ppo',
        }, meta)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE: {variant_name} | {bank_label}")
    print(f"{'='*60}")
    print(f"Total episodes: {num_episodes}")
    print(f"Banks: {num_banks} ({num_cells} cells)")
    print(f"Perfect episodes: {len(perfect_episodes)} "
          f"({len(perfect_episodes)/max(num_episodes,1)*100:.1f}%)")
    if len(completion_rates) >= 100:
        print(f"Final completion: {np.mean(completion_rates[-100:]):.1%}")
        print(f"Final tardiness:  {np.mean(tardy_rates[-100:]):.1f}%")
    print(f"Best eval completion: {best_eval_completion:.1%}")
    print(f"Training time: {(time.time()-start_time)/60:.1f} minutes")

    return agent


def _eval_during_training(agent, facility, eval_datasets, window_length,
                            num_banks, episode, use_dap, use_fpr):
    """Quick evaluation during training with bank filtering."""
    total_comp = 0
    total_servers = 0

    for df in eval_datasets[:5]:
        if num_banks < 4:
            env = BankFilteredEnvironment(
                facility, df, num_banks=num_banks, quiet=True,
                window_length=window_length,
                use_windowing=True,
                randomize_order=False,
                use_deferred_actions=use_dap,
                use_fpr=use_fpr)
        else:
            env = LexicographicRewardEnvironment(
                facility, df,
                window_length=window_length,
                use_windowing=True,
                randomize_order=False,
                use_deferred_actions=use_dap,
                use_fpr=use_fpr)

        state, _ = env.reset()
        done = False
        while not done:
            if use_dap:
                valid_actions = env.get_valid_actions_deferred()
            else:
                valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            state, _, done, _, info = env.step(action)

        # Quick greedy repair (only using active cells)
        if num_banks < 4:
            active_cells = env.active_cell_indices
        else:
            active_cells = set(range(env.total_cells))

        original_count = len(env.servers) - len(env._retry_queue)
        for si in range(original_count):
            if si in env.assigned_servers:
                continue
            srv = env.servers[si]
            arrival = srv['ArrivalTime']
            pt = srv['PTk']
            is_2tc = srv['OTk'] == 1
            server_id = srv['k']
            if server_id in env.assignments:
                continue
            latest = min(srv['DueTime'] - pt, env.total_time + 1 - pt)
            if latest < arrival:
                latest = env.total_time + 1 - pt
            if latest < arrival:
                continue
            placed = False
            for start in range(arrival, latest + 1):
                end_time = start + pt
                if end_time > env.total_time + 1:
                    break
                for ci in active_cells:  # Only try active bank cells
                    if not env._is_cell_compatible(ci, srv):
                        continue
                    occ = env.cell_occupancy[ci, start:min(end_time, env.total_time + 1)]
                    if np.any(occ != 0):
                        continue
                    if is_2tc:
                        if ci not in env.adjacent_cells:
                            continue
                        ai = env.adjacent_cells[ci]
                        if ai not in active_cells:
                            continue  # Adjacent must also be in active bank
                        adj_occ = env.cell_occupancy[ai, start:min(end_time, env.total_time + 1)]
                        if np.any(adj_occ != 0):
                            continue
                        env.cell_occupancy[ci, start:min(end_time, env.total_time + 1)] = server_id
                        env.cell_occupancy[ai, start:min(end_time, env.total_time + 1)] = server_id
                    else:
                        env.cell_occupancy[ci, start:min(end_time, env.total_time + 1)] = server_id
                    env.assignments[server_id] = {
                        'cell_idx': ci, 'start': start, 'end': end_time - 1}
                    env.assigned_servers.add(si)
                    placed = True
                    break
                if placed:
                    break

        total_comp += len(env.assigned_servers)
        total_servers += original_count

    comp_rate = total_comp / total_servers if total_servers > 0 else 0
    print(f"  [EVAL ep={episode+1}] Completion: {comp_rate:.1%} "
          f"({num_banks}-bank, {total_comp}/{total_servers})")
    return comp_rate


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train LHAC-DAF for multi-bank configurations')
    parser.add_argument('--num-banks', type=int, default=1,
                        choices=[1, 2, 4],
                        help='Number of active banks (1, 2, or 4)')
    parser.add_argument('--variant', type=str, default='full_daf',
                        choices=['baseline_lhac', 'dap_only', 'case_only',
                                 'fpr_only', 'dap_case', 'dap_fpr',
                                 'case_fpr', 'full_daf'],
                        help='DAF-LHAC variant to train (default: full_daf)')
    parser.add_argument('--episodes', type=int, default=25000,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--window-length', type=int, default=10,
                        help='Lookahead window length for state features')
    parser.add_argument('--rolling-window', action='store_true',
                        help='Use rolling window during training')
    parser.add_argument('--window-days', type=int, default=10,
                        help='Days per window when using rolling window')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Model save path (auto-generated if None)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (fewer datasets, 5000 episodes)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint-interval', type=int, default=2000)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--resume-episode', type=int, default=0,
                        help='Episode to resume from')
    parser.add_argument('--finetune', type=str, default=None,
                        help='Path to pre-trained model to finetune '
                        '(e.g., 4-bank model for 2-bank finetuning)')
    parser.add_argument('--finetune-lr', type=float, default=1e-4,
                        help='Learning rate for finetuning (lower than training)')
    args = parser.parse_args()

    num_banks = args.num_banks
    bank_label = BANK_CONFIGS[num_banks]['label']
    num_cells = BANK_CONFIGS[num_banks]['num_cells']

    # Quick mode adjustments
    if args.quick:
        args.episodes = min(args.episodes, 5000)

    # Variant config
    VARIANTS = {
        'baseline_lhac': {'dap': False, 'case': False, 'fpr': False},
        'dap_only':      {'dap': True,  'case': False, 'fpr': False},
        'case_only':     {'dap': False, 'case': True,  'fpr': False},
        'fpr_only':      {'dap': False, 'case': False, 'fpr': True},
        'dap_case':      {'dap': True,  'case': True,  'fpr': False},
        'dap_fpr':       {'dap': True,  'case': False, 'fpr': True},
        'case_fpr':      {'dap': False, 'case': True,  'fpr': True},
        'full_daf':      {'dap': True,  'case': True,  'fpr': True},
    }
    config = VARIANTS[args.variant]
    use_dap = config['dap']
    use_case = config['case']
    use_fpr = config['fpr']

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"LHAC-DAF Multi-Bank Training Setup")
    print(f"{'='*60}")
    print(f"  Variant: {args.variant}")
    print(f"  Banks: {num_banks} ({bank_label})")
    print(f"  DAP: {use_dap}, CASE: {use_case}, FPR: {use_fpr}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Rolling window: {'ON (' + str(args.window_days) + ' days)' if args.rolling_window else 'OFF'}")
    if args.finetune:
        print(f"  Finetuning from: {args.finetune}")
        print(f"  Finetune LR: {args.finetune_lr}")

    # Initialize facility (always full 4-bank, masking handles bank restriction)
    facility = FacilityConfiguration()
    n_all_cells = len(facility.all_cells)
    print(f"\n  Facility: {n_all_cells} cells, "
          f"action_space will be {n_all_cells + 1}")

    # Load datasets
    print("\nLoading datasets...")
    real_dfs = load_training_datasets()
    print(f"  Real datasets: {len(real_dfs)}")

    n_syn = 50 if args.quick else 200
    n_var = 50 if args.quick else 200
    synthetic = create_synthetic_datasets(n_syn, num_banks)
    variations = create_variations(real_dfs, n_var, num_banks)
    all_datasets = real_dfs + synthetic + variations
    print(f"  Synthetic: {len(synthetic)} (max_size={BANK_CONFIGS[num_banks]['num_cells'] * 7})")
    print(f"  Variations: {len(variations)}")
    print(f"  Total: {len(all_datasets)}")

    # Eval datasets (use moderate-size real datasets)
    max_eval = min(400, int(num_cells * 7.5))
    eval_datasets = [df for df in real_dfs
                     if len(df) >= min(100, max_eval // 2)
                     and len(df) <= max_eval][:5]
    if not eval_datasets:
        eval_datasets = real_dfs[:5]
    print(f"  Eval hold-out: {len(eval_datasets)} "
          f"(sizes: {[len(d) for d in eval_datasets]})")

    # Create curriculum
    curriculum = CurriculumManager(all_datasets, num_banks)

    # Determine state/action dims using BankFilteredEnvironment
    sample_df = all_datasets[0]
    if num_banks < 4:
        dummy_env = BankFilteredEnvironment(
            facility, sample_df, num_banks=num_banks, quiet=True,
            window_length=args.window_length,
            use_windowing=True,
            use_deferred_actions=use_dap)
    else:
        dummy_env = LexicographicRewardEnvironment(
            facility, sample_df,
            window_length=args.window_length,
            use_windowing=True,
            use_deferred_actions=use_dap)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    print(f"\n  State dim: {state_dim}, Action dim: {action_dim}")
    print(f"  Active cells: {num_cells} / {n_all_cells}")

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"  Device: {device}")

    # Create agent
    lr = args.finetune_lr if args.finetune else args.lr
    agent = MultiAgentLHACPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr1=lr,
        lr2=lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.02,
        epochs_per_update=4,
        mini_batch_size=256,
        device=device,
        use_case=use_case)

    # Load pre-trained model if finetuning or resuming
    if args.finetune:
        agent.load(args.finetune)
        print(f"\n  Loaded pre-trained model from {args.finetune}")
        print(f"  Finetuning with lr={lr} for {num_banks}-bank")
    elif args.resume:
        agent.load(args.resume)
        print(f"\n  Resumed from {args.resume}")

    # Save path
    save_path = args.save_path
    if save_path is None:
        bank_suffix = f'_{num_banks}bank' if num_banks != 4 else ''
        rolling_suffix = '_rolling' if args.rolling_window else ''
        ft_suffix = '_ft' if args.finetune else ''
        save_path = os.path.join(
            SCRIPT_DIR, 'Models',
            f'daf_{args.variant}{bank_suffix}{rolling_suffix}{ft_suffix}.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"  Save path: {save_path}")

    # Train
    agent = train_multibank(
        agent, facility, curriculum,
        num_banks=num_banks,
        num_episodes=args.episodes,
        window_length=args.window_length,
        use_rolling=args.rolling_window,
        window_days=args.window_days,
        save_path=save_path,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        eval_datasets=eval_datasets,
        use_dap=use_dap,
        use_fpr=use_fpr,
        variant_name=args.variant,
        start_episode=args.resume_episode)

    print(f"\nModel saved to: {save_path}")
    print(f"Best model saved to: {save_path.replace('.pth', '_best.pth')}")
    print(f"\nTo evaluate this model:")
    print(f"  python evaluate_multibank.py --banks {num_banks} "
          f"--model {save_path}")


if __name__ == '__main__':
    main()
