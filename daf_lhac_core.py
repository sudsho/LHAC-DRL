#!/usr/bin/env python3
"""
RL Warm-Start Hybrid Solver for CTO Server Test Cell Scheduling

Three modes:
  - parvez_only: GA Phase 1 (pymoo) + CPLEX Phase 2 (baseline)
  - rl_only:     Lexicographic DQN assigns all servers (fast, no CPLEX)
  - hybrid:      Lexicographic DQN warm-start + CPLEX polishing (main contribution)

Usage:
  python rl_warmstart_hybrid.py --mode hybrid --dataset_path Data/400_orders_back_20%.xlsx
  python rl_warmstart_hybrid.py --mode generate_data --orders 400 --arrival back
"""

import os
import sys
import math
import copy
import time as tm
import random
import argparse
import warnings
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# =============================================================================

NUM_BANKS = 4
NUM_CELLS_PER_BANK = 14
UNIT_TIME = 24          # hours per time block
TOTAL_TIME_BLOCKS = 65  # 13 weeks * 5 days
DIVIDE_TIME = 10        # blocks per lookahead window

# Server type processing times (in 24-hour blocks)
ARTEMIS_PT = 7   # 154 hours / 24
THEMIS_PT = 4    # 80 hours / 24
ATHENA_PT = 3    # 60 hours / 24

# Server type ratios from the calibrated industry-partner distribution
ARTEMIS_RATIO = 12 / 31
THEMIS_RATIO = 8 / 31
# ATHENA_RATIO = 1 - ARTEMIS_RATIO - THEMIS_RATIO

# Power consumption per server type (kWh * unit_time_length)
POWER = {
    'artemis_1tc': 8.2 * UNIT_TIME,
    'artemis_2tc': 16.4 * UNIT_TIME,
    'athena_1tc': 10.0 * UNIT_TIME,
    'athena_2tc': 15.0 * UNIT_TIME,
    'themis_1tc': 7.9 * UNIT_TIME,
    'themis_2tc': 14.0 * UNIT_TIME,
}

# Due date distribution (from Parvez paper)
DUE_DATE_FRACTIONS = [1/31, 1/31, 2/31, 1/31, 1/31, 1/31, 1/31, 6/31, 17/31]
DUE_DATE_DAYS = [27, 29, 40, 42, 47, 51, 52, 54, 65]

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'Input')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MODEL_DIR = os.path.join(BASE_DIR, 'Models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

# =============================================================================
# SECTION 2: B052 FACILITY CONFIGURATION
# =============================================================================

class VoltageType(Enum):
    HV = "HV"
    LV = "LV"
    BOTH = "BOTH"

class CoolingType(Enum):
    AIR = "AIR"
    WATER = "WATER"

class CellStatus(Enum):
    FUL = "FUL"   # Fully operational
    FAB = "FAB"   # Fabrication / blocked

@dataclass
class TestCell:
    cell_id: str
    bank_id: int
    position: int
    voltage_type: VoltageType
    cooling_type: CoolingType
    status: CellStatus
    supports_4_frames: bool
    power_400v_connectors: int
    power_208v_connectors: int
    water_hoses: int

    @property
    def total_power_connectors(self) -> int:
        return self.power_400v_connectors + self.power_208v_connectors

    def is_compatible(self, voltage_need: str, cooling_need: str) -> bool:
        if self.status != CellStatus.FUL:
            return False
        if voltage_need == "HV":
            voltage_ok = self.voltage_type in (VoltageType.HV, VoltageType.BOTH)
        elif voltage_need == "LV":
            voltage_ok = self.voltage_type in (VoltageType.LV, VoltageType.BOTH)
        else:
            voltage_ok = True
        if cooling_need == "WATER":
            cooling_ok = self.water_hoses > 0
        else:
            cooling_ok = True
        return voltage_ok and cooling_ok

    def __repr__(self):
        return (f"Cell({self.cell_id}, Bank{self.bank_id}, "
                f"400V:{self.power_400v_connectors}, 208V:{self.power_208v_connectors}, "
                f"Water:{self.water_hoses}, {self.status.value})")

@dataclass
class TestBank:
    bank_id: int
    bank_name: str
    cells: List[TestCell] = field(default_factory=list)

    def get_ful_cells(self) -> List[TestCell]:
        return [c for c in self.cells if c.status == CellStatus.FUL]

    def get_water_cooling_cells(self) -> List[TestCell]:
        return [c for c in self.cells if c.water_hoses > 0]

    def get_utilization(self) -> float:
        total = len(self.cells)
        available = len(self.get_ful_cells())
        return (total - available) / total if total > 0 else 0


class FacilityConfiguration:
    """Complete test facility configuration based on B052 specifications"""

    def __init__(self):
        self.banks: List[TestBank] = []
        self.all_cells: OrderedDict = OrderedDict()
        self._initialize_b052_configuration()

    def _initialize_b052_configuration(self):
        # Bank 1 (052B) - 14 cells
        bank1 = TestBank(bank_id=1, bank_name="052B")
        bank1_specs = [
            # (position, suffix, 4frames, pwrcfg, 400v, 208v, water, status)
            # All cells are FUL (usable). Temporal blocking from block_list.xlsx.
            (1, "01A", False, "LV,HV", 6, 6, 0, "FUL"),
            (2, "02A", True, "LV,HV", 6, 6, 8, "FUL"),
            (3, "03A", True, "LV,HV", 6, 6, 8, "FUL"),
            (4, "04A", False, "LV,HV", 6, 6, 0, "FUL"),
            (5, "05A", True, "LV,HV", 4, 6, 0, "FUL"),
            (6, "06A", True, "LV,HV", 6, 6, 0, "FUL"),
            (7, "07A", False, "LV,HV", 4, 6, 0, "FUL"),
            (8, "08A", True, "LV,HV", 6, 6, 0, "FUL"),
            (9, "09A", False, "LV,HV", 6, 6, 4, "FUL"),
            (10, "10A", True, "LV,HV", 6, 6, 4, "FUL"),
            (11, "11A", True, "LV,HV", 6, 6, 4, "FUL"),
            (12, "12A", False, "LV,HV", 6, 6, 4, "FUL"),
            (13, "13A", True, "LV,HV", 6, 6, 4, "FUL"),
            (14, "14A", True, "LV,HV", 6, 6, 4, "FUL"),
        ]
        for pos, suffix, frames, pwrcfg, p400, p208, water, status in bank1_specs:
            cell = TestCell(
                cell_id=f"052B{suffix}", bank_id=1, position=pos,
                voltage_type=self._parse_voltage(pwrcfg),
                cooling_type=CoolingType.WATER if water > 0 else CoolingType.AIR,
                status=CellStatus[status], supports_4_frames=frames,
                power_400v_connectors=p400, power_208v_connectors=p208,
                water_hoses=water
            )
            bank1.cells.append(cell)
            self.all_cells[cell.cell_id] = cell
        self.banks.append(bank1)

        # Bank 2 (052C) - 14 cells
        bank2 = TestBank(bank_id=2, bank_name="052C")
        bank2_specs = [
            (1, "01A", False, "HV", 6, 0, 4, "FUL"),
            (2, "02A", True, "LV,HV", 6, 4, 4, "FUL"),
            (3, "03A", True, "LV,HV", 6, 4, 4, "FUL"),
            (4, "04A", False, "HV", 4, 0, 4, "FUL"),
            (5, "05A", True, "LV,HV", 6, 6, 4, "FUL"),
            (6, "06A", True, "LV,HV", 6, 6, 4, "FUL"),
            (7, "07A", False, "LV,HV", 6, 6, 4, "FUL"),
            (8, "08A", True, "LV,HV", 6, 6, 4, "FUL"),
            (9, "09A", False, "LV,HV", 6, 6, 4, "FUL"),
            (10, "10A", True, "LV,HV", 6, 6, 4, "FUL"),
            (11, "11A", True, "LV,HV", 6, 6, 4, "FUL"),
            (12, "12A", False, "LV,HV", 6, 6, 4, "FUL"),
            (13, "13A", True, "LV,HV", 6, 6, 4, "FUL"),
            (14, "14A", True, "LV,HV", 6, 6, 4, "FUL"),
        ]
        for pos, suffix, frames, pwrcfg, p400, p208, water, status in bank2_specs:
            cell = TestCell(
                cell_id=f"052C{suffix}", bank_id=2, position=pos,
                voltage_type=self._parse_voltage(pwrcfg),
                cooling_type=CoolingType.WATER if water > 0 else CoolingType.AIR,
                status=CellStatus[status], supports_4_frames=frames,
                power_400v_connectors=p400, power_208v_connectors=p208,
                water_hoses=water
            )
            bank2.cells.append(cell)
            self.all_cells[cell.cell_id] = cell
        self.banks.append(bank2)

        # Bank 3 (052D) - 14 cells
        bank3 = TestBank(bank_id=3, bank_name="052D")
        bank3_specs = [
            (1, "01A", False, "LV,HV", 6, 4, 0, "FUL"),
            (2, "02A", True, "LV,HV", 6, 4, 8, "FUL"),
            (3, "03A", True, "LV,HV", 4, 4, 8, "FUL"),
            (4, "04A", False, "LV,HV", 6, 4, 0, "FUL"),
            (5, "05A", True, "LV,HV", 6, 4, 4, "FUL"),
            (6, "06A", True, "LV,HV", 6, 4, 4, "FUL"),
            (7, "07A", False, "LV,HV", 4, 4, 4, "FUL"),
            (8, "08A", True, "LV,HV", 6, 4, 4, "FUL"),
            (9, "09A", False, "LV,HV", 6, 4, 4, "FUL"),
            (10, "10A", True, "LV,HV", 6, 4, 4, "FUL"),
            (11, "11A", True, "LV,HV", 6, 4, 4, "FUL"),
            (12, "12A", False, "LV,HV", 4, 4, 4, "FUL"),
            (13, "13A", True, "LV,HV", 4, 4, 4, "FUL"),
            (14, "14A", True, "LV,HV", 4, 4, 4, "FUL"),
        ]
        for pos, suffix, frames, pwrcfg, p400, p208, water, status in bank3_specs:
            cell = TestCell(
                cell_id=f"052D{suffix}", bank_id=3, position=pos,
                voltage_type=self._parse_voltage(pwrcfg),
                cooling_type=CoolingType.WATER if water > 0 else CoolingType.AIR,
                status=CellStatus[status], supports_4_frames=frames,
                power_400v_connectors=p400, power_208v_connectors=p208,
                water_hoses=water
            )
            bank3.cells.append(cell)
            self.all_cells[cell.cell_id] = cell
        self.banks.append(bank3)

        # Bank 4 (052E) - 12 usable cells (positions 4 and 7 not usable)
        bank4 = TestBank(bank_id=4, bank_name="052E")
        bank4_specs = [
            (1, "01A", False, "LV,HV", 6, 6, 4, "FUL"),
            (2, "02A", True, "LV,HV", 6, 6, 4, "FUL"),
            (3, "03A", False, "LV,HV", 6, 6, 4, "FUL"),
            # position 4 is not a usable test cell
            (5, "05A", True, "LV,HV", 6, 6, 4, "FUL"),
            (6, "06A", True, "LV,HV", 6, 6, 4, "FUL"),
            # position 7 is not a usable test cell
            (8, "08A", True, "LV,HV", 6, 6, 4, "FUL"),
            (9, "09A", True, "LV,HV", 6, 6, 4, "FUL"),
            (10, "10A", False, "LV,HV", 6, 6, 4, "FUL"),
            (11, "11A", True, "LV,HV", 6, 6, 4, "FUL"),
            (12, "12A", False, "LV,HV", 6, 6, 4, "FUL"),
            (13, "13A", True, "LV,HV", 4, 6, 4, "FUL"),
            (14, "14A", False, "LV,HV", 6, 6, 4, "FUL"),
        ]
        for pos, suffix, frames, pwrcfg, p400, p208, water, status in bank4_specs:
            cell = TestCell(
                cell_id=f"052E{suffix}", bank_id=4, position=pos,
                voltage_type=self._parse_voltage(pwrcfg),
                cooling_type=CoolingType.WATER if water > 0 else CoolingType.AIR,
                status=CellStatus[status], supports_4_frames=frames,
                power_400v_connectors=p400, power_208v_connectors=p208,
                water_hoses=water
            )
            bank4.cells.append(cell)
            self.all_cells[cell.cell_id] = cell
        self.banks.append(bank4)

    def _parse_voltage(self, pwrcfg: str) -> VoltageType:
        if "HV" in pwrcfg and "LV" in pwrcfg:
            return VoltageType.BOTH
        elif "HV" in pwrcfg:
            return VoltageType.HV
        elif "LV" in pwrcfg:
            return VoltageType.LV
        return VoltageType.BOTH

    def get_summary(self) -> Dict:
        total_cells = len(self.all_cells)
        ful_cells = sum(1 for c in self.all_cells.values() if c.status == CellStatus.FUL)
        fab_cells = sum(1 for c in self.all_cells.values() if c.status == CellStatus.FAB)
        water_cells = sum(1 for c in self.all_cells.values() if c.water_hoses > 0)
        return {
            'total_cells': total_cells,
            'total_banks': len(self.banks),
            'cells_per_bank': [len(b.cells) for b in self.banks],
            'ful_cells': ful_cells,
            'fab_cells': fab_cells,
            'water_cooling_cells': water_cells,
        }

    def get_compatible_cells(self, voltage_need: str, cooling_need: str) -> List[TestCell]:
        return [c for c in self.all_cells.values()
                if c.is_compatible(voltage_need, cooling_need)]

    def get_adjacent_pairs_for_2tc(self, voltage_need: str, cooling_need: str) -> List[Tuple[TestCell, TestCell]]:
        pairs = []
        for bank in self.banks:
            ful_cells = sorted(bank.get_ful_cells(), key=lambda c: c.position)
            for i in range(len(ful_cells) - 1):
                c1, c2 = ful_cells[i], ful_cells[i + 1]
                if c2.position == c1.position + 1:
                    if c1.is_compatible(voltage_need, cooling_need) and \
                       c2.is_compatible(voltage_need, cooling_need):
                        pairs.append((c1, c2))
        return pairs


def generate_cell_bank_from_b052(facility: FacilityConfiguration) -> pd.DataFrame:
    """Convert B052 facility config to Parvez's CELL_BANK.xlsx format.

    Returns DataFrame with 56 rows (4 banks x 14 cells) with columns:
    Serial No, HV Availability, LV Availability, Water Cooling Availability
    """
    rows = []
    serial = 1
    for bank in facility.banks:
        # Build a lookup by position for this bank
        cell_by_pos = {c.position: c for c in bank.cells}
        for pos in range(1, NUM_CELLS_PER_BANK + 1):
            if pos in cell_by_pos:
                cell = cell_by_pos[pos]
                hv = 1 if cell.voltage_type in (VoltageType.HV, VoltageType.BOTH) else 0
                lv = 1 if cell.voltage_type in (VoltageType.LV, VoltageType.BOTH) else 0
                w = 1 if cell.water_hoses > 0 else 0
            else:
                # Missing position (e.g., bank 4 positions 4, 7)
                hv, lv, w = 0, 0, 0
            rows.append({
                'Serial No': serial,
                'HV Availability': hv,
                'LV Availability': lv,
                'Water Cooling Availability': w,
            })
            serial += 1
    return pd.DataFrame(rows)


def save_cell_bank(facility: FacilityConfiguration, path: str):
    """Generate and save CELL_BANK.xlsx"""
    df = generate_cell_bank_from_b052(facility)
    df.to_excel(path, index=False)
    print(f"CELL_BANK.xlsx saved to {path} ({len(df)} rows)")


# =============================================================================
# SECTION 3: ORDER DATA GENERATOR
# =============================================================================

def generate_orders(num_orders: int, arrival_pattern: str = 'back',
                    two_tc_pct: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic order data matching Parvez's format.

    Args:
        num_orders: Number of orders to generate
        arrival_pattern: 'back', 'front', or 'uniform'
        two_tc_pct: Fraction of 2TC (OTk=1) servers
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: k, type, PTk, OTk, RH, RL, RW, DueTime, ArrivalTime
    """
    random.seed(seed)
    np.random.seed(seed)

    t_o = num_orders
    total_time = TOTAL_TIME_BLOCKS

    art = int(t_o * ARTEMIS_RATIO)
    the = int(t_o * THEMIS_RATIO)
    ath = t_o - (art + the)

    def _generate_type_list(count, pt, two_tc_pct_type, rh_frac, rw_frac):
        random.seed(42)
        OTk = [1] * int(count * two_tc_pct_type) + [0] * (count - int(count * two_tc_pct_type))
        random.shuffle(OTk)

        RH = [0] * int(count * (1 - rh_frac)) + [1] * (count - int(count * (1 - rh_frac)))
        RL = [1 - rh for rh in RH]  # Complementary
        tmp = list(zip(RL, RH))
        random.shuffle(tmp)
        RL, RH = zip(*tmp)
        RL, RH = list(RL), list(RH)

        RW = [1] * int(count * rw_frac) + [0] * (count - int(count * rw_frac))
        random.shuffle(RW)

        return OTk, RH, RL, RW

    # Themis: 87.5% low voltage, 50% water
    random.seed(42)
    the_OTk = [1] * int(the * two_tc_pct) + [0] * (the - int(the * two_tc_pct))
    random.shuffle(the_OTk)
    the_RH, the_RL = [], []
    for _ in range(int(the * 0.875)):
        the_RL.append(1); the_RH.append(0)
    for _ in range(the - int(the * 0.875)):
        the_RL.append(0); the_RH.append(1)
    tmp = list(zip(the_RL, the_RH))
    random.shuffle(tmp)
    the_RL, the_RH = zip(*tmp)
    the_RL, the_RH = list(the_RL), list(the_RH)
    the_RW = [1] * int(the * 0.5) + [0] * (the - int(the * 0.5))
    random.shuffle(the_RW)

    themis_list = []
    for i in range(the):
        themis_list.append({
            'k': None, 'type': 'themis', 'PTk': THEMIS_PT,
            'OTk': the_OTk[i], 'RH': the_RH[i], 'RL': the_RL[i], 'RW': the_RW[i],
            'DueTime': None, 'ArrivalTime': None
        })

    # Athena: 3/11 low voltage, 4/11 water
    random.seed(42)
    ath_OTk = [1] * int(ath * two_tc_pct) + [0] * (ath - int(ath * two_tc_pct))
    random.shuffle(ath_OTk)
    ath_RH, ath_RL = [], []
    for _ in range(int(ath * (3 / 11))):
        ath_RL.append(1); ath_RH.append(0)
    for _ in range(ath - int(ath * (3 / 11))):
        ath_RL.append(0); ath_RH.append(1)
    tmp = list(zip(ath_RL, ath_RH))
    random.shuffle(tmp)
    ath_RL, ath_RH = zip(*tmp)
    ath_RL, ath_RH = list(ath_RL), list(ath_RH)
    ath_RW = [1] * int(ath * (4 / 11)) + [0] * (ath - int(ath * (4 / 11)))
    random.shuffle(ath_RW)

    athena_list = []
    for i in range(ath):
        athena_list.append({
            'k': None, 'type': 'athena', 'PTk': ATHENA_PT,
            'OTk': ath_OTk[i], 'RH': ath_RH[i], 'RL': ath_RL[i], 'RW': ath_RW[i],
            'DueTime': None, 'ArrivalTime': None
        })

    # Artemis: 8/12 low voltage, 8/12 water
    random.seed(42)
    art_OTk = [1] * int(art * two_tc_pct) + [0] * (art - int(art * two_tc_pct))
    random.shuffle(art_OTk)
    art_RH, art_RL = [], []
    for _ in range(int(art * (8 / 12))):
        art_RL.append(1); art_RH.append(0)
    for _ in range(art - int(art * (8 / 12))):
        art_RL.append(0); art_RH.append(1)
    tmp = list(zip(art_RL, art_RH))
    random.shuffle(tmp)
    art_RL, art_RH = zip(*tmp)
    art_RL, art_RH = list(art_RL), list(art_RH)
    art_RW = [1] * int(art * (8 / 12)) + [0] * (art - int(art * (8 / 12)))
    random.shuffle(art_RW)

    artemis_list = []
    for i in range(art):
        artemis_list.append({
            'k': None, 'type': 'artemis', 'PTk': ARTEMIS_PT,
            'OTk': art_OTk[i], 'RH': art_RH[i], 'RL': art_RL[i], 'RW': art_RW[i],
            'DueTime': None, 'ArrivalTime': None
        })

    # Sort by type priority (artemis first, then themis, then athena)
    random.seed(42)
    f_14_31 = artemis_list + themis_list[:int(the * 0.25)]
    random.shuffle(f_14_31)
    f_17_31 = athena_list + themis_list[int(the * 0.25):]
    random.shuffle(f_17_31)

    final = f_14_31 + f_17_31
    for i in range(len(final)):
        final[i]['k'] = i + 1

    # Due dates
    tmp2_day = [d * 24 for d in DUE_DATE_DAYS]
    tmp2 = [min(math.floor(d / UNIT_TIME), total_time) for d in tmp2_day]

    j = 0
    for idx in range(len(DUE_DATE_FRACTIONS)):
        for k_idx in range(j, j + int(t_o * DUE_DATE_FRACTIONS[idx])):
            if k_idx < len(final):
                final[k_idx]['DueTime'] = tmp2[idx]
        j += int(t_o * DUE_DATE_FRACTIONS[idx])
    for k_idx in range(j, t_o):
        final[k_idx]['DueTime'] = tmp2[-1]

    # Arrival times based on pattern
    if arrival_pattern == 'back':
        # Back-loaded: bulk arrives late
        tmp1 = [0.0425, 0.0425, 0.0425, 0.0425,
                0.0825, 0.0825, 0.0825, 0.0825,
                0.125, 0.125, 0.125, 0.125]
        tmp2_arr = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56]
    elif arrival_pattern == 'front':
        # Front-loaded: bulk arrives early
        tmp1 = [0.125, 0.125, 0.125, 0.125,
                0.0825, 0.0825, 0.0825, 0.0825,
                0.0425, 0.0425, 0.0425, 0.0425]
        tmp2_arr = [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56]
    else:  # uniform
        # Uniform spread
        for i in range(t_o):
            s = math.ceil((i + 1) * 60 / t_o)
            if s > 60:
                s = 60
            final[i]['ArrivalTime'] = s
        df = pd.DataFrame(final)
        return df

    j = 0
    for idx in range(len(tmp1)):
        start = tmp2_arr[idx]
        end = start + 5
        total_size = int(t_o * tmp1[idx])
        block_size = math.ceil(total_size / 5)
        start_time = np.arange(start, end)
        start_time = np.repeat(start_time, block_size)
        start_time = start_time[:total_size]
        for k_idx in range(j, j + total_size):
            if k_idx < len(final):
                final[k_idx]['ArrivalTime'] = int(start_time[k_idx - j])
        j += total_size
    for k_idx in range(j, t_o):
        final[k_idx]['ArrivalTime'] = 60

    df = pd.DataFrame(final)
    return df


# =============================================================================
# SECTION 4: DATA READING (port of read_file.py)
# =============================================================================

def file_read(cell_bank_file_path: str, order_file_path: str,
              unit_time_length: int = UNIT_TIME) -> tuple:
    """Read facility and order data from Excel files.

    Port of Paper_1_Codes/local_dependencies/read_file.py
    """
    df = pd.read_excel(cell_bank_file_path)
    df1 = pd.read_excel(order_file_path)

    def cell_data_convert(x1):
        x1 = pd.DataFrame(x1).to_numpy().flatten()
        x1 = np.reshape(x1, (-1, NUM_CELLS_PER_BANK))
        return x1

    serial = cell_data_convert(df['Serial No'])
    VH = cell_data_convert(df['HV Availability'])
    VL = cell_data_convert(df['LV Availability'])
    W = cell_data_convert(df['Water Cooling Availability'])

    K = pd.DataFrame(df1['k']).to_numpy().flatten()
    PTk = pd.DataFrame(df1['PTk']).to_numpy().flatten()
    OTk = pd.DataFrame(df1['OTk']).to_numpy().flatten()
    RH = pd.DataFrame(df1['RH']).to_numpy().flatten()
    RL = pd.DataFrame(df1['RL']).to_numpy().flatten()
    RW = pd.DataFrame(df1['RW']).to_numpy().flatten()
    TYPE = pd.DataFrame(df1['type']).to_numpy().flatten()
    DueTime = pd.DataFrame(df1['DueTime']).to_numpy().flatten()
    arrival_time = pd.DataFrame(df1['ArrivalTime']).to_numpy().flatten()

    time_val = 24 * 5 * 13  # 24h * 5days * 13weeks
    time_val = int(time_val / unit_time_length)
    T = np.arange(1, time_val + 1, dtype='int32')

    for i in range(len(K)):
        K[i] = i + 1

    # Power consumption per order
    power_cell = [0] * len(TYPE)
    for i in range(len(TYPE)):
        t = TYPE[i]
        otk = OTk[i]
        if t == "artemis":
            power_cell[i] = POWER['artemis_2tc'] if otk == 1 else POWER['artemis_1tc']
        elif t == "athena":
            power_cell[i] = POWER['athena_2tc'] if otk == 1 else POWER['athena_1tc']
        elif t == "themis":
            power_cell[i] = POWER['themis_2tc'] if otk == 1 else POWER['themis_1tc']

    # Availability matrix: atk[k][t] = 1 if order k is available at time t
    atk = np.zeros((len(K), time_val), dtype='int32')
    for i in range(len(K)):
        atk[i][:arrival_time[i]] = 0
        atk[i][arrival_time[i] - 1:] = 1

    return (serial, VH, VL, W, K, PTk, OTk, RH, RL, RW, TYPE,
            DueTime, arrival_time, time_val, T, K, power_cell, atk)


def file_read_from_facility(facility: FacilityConfiguration,
                            order_file_path: str,
                            unit_time_length: int = UNIT_TIME) -> tuple:
    """Read data using B052 facility config instead of CELL_BANK.xlsx."""
    # Generate cell bank data from facility
    cell_bank_df = generate_cell_bank_from_b052(facility)

    def cell_data_convert(x1):
        x1 = np.array(x1).flatten()
        x1 = np.reshape(x1, (-1, NUM_CELLS_PER_BANK))
        return x1

    VH = cell_data_convert(cell_bank_df['HV Availability'].values)
    VL = cell_data_convert(cell_bank_df['LV Availability'].values)
    W = cell_data_convert(cell_bank_df['Water Cooling Availability'].values)
    serial = cell_data_convert(cell_bank_df['Serial No'].values)

    df1 = pd.read_excel(order_file_path)

    K = pd.DataFrame(df1['k']).to_numpy().flatten()
    PTk = pd.DataFrame(df1['PTk']).to_numpy().flatten()
    OTk = pd.DataFrame(df1['OTk']).to_numpy().flatten()
    RH = pd.DataFrame(df1['RH']).to_numpy().flatten()
    RL = pd.DataFrame(df1['RL']).to_numpy().flatten()
    RW = pd.DataFrame(df1['RW']).to_numpy().flatten()
    TYPE = pd.DataFrame(df1['type']).to_numpy().flatten()
    DueTime = pd.DataFrame(df1['DueTime']).to_numpy().flatten()
    arrival_time = pd.DataFrame(df1['ArrivalTime']).to_numpy().flatten()

    time_val = 24 * 5 * 13
    time_val = int(time_val / unit_time_length)
    T = np.arange(1, time_val + 1, dtype='int32')

    for i in range(len(K)):
        K[i] = i + 1

    power_cell = [0] * len(TYPE)
    for i in range(len(TYPE)):
        t = TYPE[i]
        otk = OTk[i]
        if t == "artemis":
            power_cell[i] = POWER['artemis_2tc'] if otk == 1 else POWER['artemis_1tc']
        elif t == "athena":
            power_cell[i] = POWER['athena_2tc'] if otk == 1 else POWER['athena_1tc']
        elif t == "themis":
            power_cell[i] = POWER['themis_2tc'] if otk == 1 else POWER['themis_1tc']

    atk = np.zeros((len(K), time_val), dtype='int32')
    for i in range(len(K)):
        atk[i][:arrival_time[i]] = 0
        atk[i][arrival_time[i] - 1:] = 1

    return (serial, VH, VL, W, K, PTk, OTk, RH, RL, RW, TYPE,
            DueTime, arrival_time, time_val, T, K, power_cell, atk)


# =============================================================================
# SECTION 5: UTILITY FUNCTIONS (direct ports from Parvez)
# =============================================================================

def generate_list_feed_model(time_val: int, divide_time: int,
                             arrival_time: np.ndarray, K: np.ndarray) -> list:
    """Partition orders into rolling windows by arrival time.

    Port of generate_list_feed_model_file.py
    """
    list_feed_model = [[] for _ in range(math.ceil(time_val / divide_time))]
    for i in range(math.ceil(time_val / divide_time)):
        start = divide_time * i + 1
        end = divide_time * i + divide_time
        for j in range(len(arrival_time)):
            if arrival_time[j] >= start and arrival_time[j] <= end:
                list_feed_model[i].append(K[j])
    return list_feed_model


def modified_processing_time(arrival_order_list: list, block_size: int,
                             arrival_time: np.ndarray,
                             PTk: np.ndarray) -> tuple:
    """Split processing times across windows for orders spanning boundaries.

    Port of modified_processing_time_file.py
    """
    remaining_processing_time = np.zeros(len(arrival_time), dtype='int32')
    mod_pt = copy.deepcopy(PTk)

    for j in range(len(arrival_order_list)):
        start_time = block_size * j + 1
        end_time = block_size * j + block_size
        for i in range(len(arrival_order_list[j])):
            order_no = arrival_order_list[j][i]
            if (arrival_time[order_no - 1] + PTk[order_no - 1] - 1) > end_time:
                mod_pt[order_no - 1] = end_time - (arrival_time[order_no - 1] - 1)
                remaining_processing_time[order_no - 1] = PTk[order_no - 1] - mod_pt[order_no - 1]

    return (mod_pt, remaining_processing_time)


def cost_calculation(k_: list, DueTime: np.ndarray,
                     arrival_time: np.ndarray) -> dict:
    """Calculate priority cost for each order based on due date urgency.

    Port of cost_calculation_file.py
    """
    lslsls = []
    for i in k_:
        lslsls.append((i, DueTime[i - 1], arrival_time[i - 1]))
    lslsls.sort(key=lambda x: (x[1], x[2]), reverse=False)
    cost_list = dict()
    for i in range(len(lslsls)):
        k, _, _ = lslsls[i]
        cost_list[k] = (len(lslsls) - i) * 100000
    return copy.deepcopy(cost_list)


def extract_solution_data(cplex_solution) -> tuple:
    """Parse CPLEX solution to extract bank/cell/time assignments.

    Port of extract_solution_data_file.py
    """
    if cplex_solution == 0 or cplex_solution is None:
        return ({}, {}, [], [])

    k_i_list = {}
    k_j_list = {}
    k_t_list = {}
    order_list = []
    testing_utilization_data = []

    for i in cplex_solution.iter_var_values():
        if round(i[1]) == 1:
            s = str(i[0])
            if s[0] == "x" or s[0] == "X":
                sp = s.split("_")
                if sp[1] in k_i_list:
                    k_i_list[sp[1]].append(int(sp[3]))
                    k_j_list[sp[1]].append(int(sp[4]))
                    k_t_list[sp[1]].append(int(sp[2]))
                else:
                    k_i_list[sp[1]] = [int(sp[3])]
                    k_j_list[sp[1]] = [int(sp[4])]
                    k_t_list[sp[1]] = [int(sp[2])]
                testing_utilization_data.append(
                    f"{sp[1]} {sp[2]} {sp[3]} {sp[4]}")
                order_list.append(int(sp[1]))

    for key in k_i_list:
        k_i_list[key] = list(np.unique(np.array(k_i_list[key])))
    for key in k_j_list:
        k_j_list[key] = list(np.unique(np.array(k_j_list[key])))
    for key in k_t_list:
        k_t_list[key] = list(np.unique(np.array(k_t_list[key])))

    order_list = list(np.unique(np.array(order_list)))

    for xx in order_list:
        print(f"Order No= {xx}  bank= {k_i_list[str(xx)]}  cell= {k_j_list[str(xx)]}")

    return (k_i_list, k_j_list, order_list, testing_utilization_data)


def calculate_utilization(returned_data: list, time_block_length: int,
                          utilization_type: str = "") -> dict:
    """Calculate cell utilization from solution data.

    Port of calculate_utilization_file.py
    """
    temp_data_dict = {}
    for i in range(1, NUM_BANKS + 1):
        for j in range(1, NUM_CELLS_PER_BANK + 1):
            temp_data_dict[f"{i},{j}"] = 0

    block_data_dict = {}
    for item in returned_data:
        sp = item.split(" ")
        key = f"{sp[2]},{sp[3]}"
        if key in block_data_dict:
            block_data_dict[key].append(int(sp[1]))
        else:
            block_data_dict[key] = [int(sp[1])]

    for key in block_data_dict:
        block_data_dict[key] = list(np.unique(np.array(block_data_dict[key])))

    print(f"{utilization_type} Utilizations:")
    for key in block_data_dict:
        bank = int(key.split(",")[0])
        cell = int(key.split(",")[1])
        usage = round((len(block_data_dict[key]) / time_block_length) * 100, 4)
        print(f"  Bank={bank} Cell={cell} Usage={usage}%")
        temp_data_dict[key] = len(block_data_dict[key])

    return temp_data_dict


def separate_heuristics_data(K_f, order_with_remaining_time, start_limit,
                              end_limit, arrival_time, DueTime, VH, VL,
                              RH_f, RL_f, W, RW_f, I, J, ki, kj,
                              PTk_f, OTk_f, solution_method):
    """Separate 2TC servers for GA heuristic.

    Port of separate_heuristics_data.py
    """
    total_order = len(K_f) - len(order_with_remaining_time)

    K_h, RH_h, RL_h, RW_h, PTk_h, OTk_h = [], [], [], [], [], []

    if end_limit > TOTAL_TIME_BLOCKS:
        end_limit = TOTAL_TIME_BLOCKS

    for i in range(len(K_f[:total_order])):
        if OTk_f[i] == 1:
            K_h.append(K_f[i])
            RH_h.append(RH_f[i])
            RL_h.append(RL_f[i])
            RW_h.append(RW_f[i])
            PTk_h.append(PTk_f[i])
            OTk_h.append(OTk_f[i])

    if solution_method == 'decom':
        return (K_h, order_with_remaining_time, start_limit, end_limit,
                arrival_time, DueTime, VH, VL, RH_h, RL_h, W, RW_h,
                I, J, ki, kj, PTk_h, OTk_h, K_f, PTk_f)
    elif solution_method == 'heu_only':
        return (K_f[:total_order], order_with_remaining_time, start_limit,
                end_limit, arrival_time, DueTime, VH, VL,
                RH_f[:total_order], RL_f[:total_order], W,
                RW_f[:total_order], I, J, ki, kj,
                PTk_f[:total_order], OTk_f[:total_order], K_f, PTk_f)
    else:
        print("No solution option provided")
        return None


# =============================================================================
# SECTION 6: GA HEURISTIC (port of Parvez's pymoo GA)
# =============================================================================

def read_block_data(file_directory: str) -> np.ndarray:
    """Read fabrication block data from Excel.

    Port of Paper_1_Codes/heu/read_block_data.py
    Returns array with columns: [bank, cell, start_block, end_block]
    """
    try:
        df = pd.read_excel(file_directory)
        data = pd.DataFrame(df).to_numpy().flatten()
        data = np.reshape(data, (-1, 5))
        data = np.delete(data, 0, 1)  # Remove serial number column
        for i in range(len(data)):
            data[i][3] = data[i][3] * 5  # Convert weeks to days
        return data
    except Exception as e:
        print(f"Problem reading block data: {e}")
        return None


def read_block_data_as_dict(block_data: np.ndarray) -> dict:
    """Convert block data array to dict keyed by cell linear index.

    Returns dict: linear_cell_index -> available_from_time
    """
    block_dict = {}
    if block_data is None:
        return block_dict
    for row in block_data:
        bank, cell = int(row[0]), int(row[1])
        start_block, end_block = int(row[2]), int(row[3])
        linear_idx = (bank - 1) * NUM_CELLS_PER_BANK + cell
        block_dict[linear_idx] = end_block
    return block_dict


def _find_compatible_cell_for_ga(VH, VL, W, RH, RL, RW, I, J, K):
    """Find compatible cells for each order (GA version, cell-centric).

    Port of Paper_1_Codes/heu/find_compatible_cell.py
    Returns:
        orders_with_compatible_cell: dict {linear_cell_idx: [compatible order IDs]}
        actualCellHeuCell: dict {heu_cell_num: linear_cell_idx}
    """
    orders_with_compatible_cell = {}
    actualCellHeuCell = {}
    heu_sum = 1
    for i in I:
        for j in J:
            temp_list = []
            for k in K:
                if (VH[i-1][j-1] >= RH[K.index(k)] and
                    VL[i-1][j-1] >= RL[K.index(k)] and
                    W[i-1][j-1] >= RW[K.index(k)]):
                    temp_list.append(k)
            # Skip bank 4 positions 4, 7 (not usable)
            if not (i == 4 and j == 4) and not (i == 4 and j == 7):
                orders_with_compatible_cell[(i-1)*14+j] = temp_list
                actualCellHeuCell[heu_sum] = (i-1)*14+j
                heu_sum += 1
    return orders_with_compatible_cell, actualCellHeuCell


def _find_compatible_cell_for_initial(VH, VL, W, RH, RL, RW, I, J, K):
    """Find compatible cells for each order (initial solution version, order-centric).

    Port of initial_solution.py's find_compatible_cell
    Returns dict: {order_id: ["bank,cell", ...]}
    """
    orders_with_compatible_cell = {}
    for k in K:
        temp_list = []
        for i in I:
            for j in J:
                if (VH[i-1][j-1] >= RH[K.index(k)] and
                    VL[i-1][j-1] >= RL[K.index(k)] and
                    W[i-1][j-1] >= RW[K.index(k)]):
                    temp_list.append(f"{i},{j}")
        orders_with_compatible_cell[k] = temp_list
    return orders_with_compatible_cell


def _sort_order_by_due_date(K, DueTime, PTk, OTk, arrival_time):
    """Sort orders by due date for initial solution generation."""
    arrival_np = np.asarray(arrival_time).reshape(len(arrival_time), 1)
    OTk_np = np.asarray(OTk).reshape(len(OTk), 1)
    PTk_np = np.asarray(PTk).reshape(len(PTk), 1)
    K_np = np.asarray(K).reshape(len(K), 1)
    DueTime_np = np.asarray(DueTime).reshape(len(DueTime), 1)
    K_Due = np.concatenate((K_np, DueTime_np, PTk_np, OTk_np, arrival_np), axis=1)
    unsorted_ = copy.deepcopy(K_Due)
    Order_Duetime = K_Due[K_Due[:, 1].argsort()]
    return Order_Duetime, unsorted_


def _num_to_bank_cell(val):
    """Convert linear cell index to (bank, cell) tuple."""
    bank = int((val - 1) / 14) + 1
    cell = int(val) - (bank - 1) * 14
    return (bank, cell)


def _calculate_assignment(machines_orders, total_machine, time_len,
                          block_data_dict, start_limit, end_limit,
                          K_main, total_order, ArrivalTime, DueTime,
                          OTk, PTk, actualCellHeuCell):
    """Calculate time-indexed assignment from GA's order sequence.

    Port of heuristics.py calculate_assignment
    """
    # Remove duplicate assignments
    h_set = set()
    for i in range(len(machines_orders)):
        for j in range(len(machines_orders[i])):
            order = machines_orders[i][j]
            if order != 0:
                if order in h_set:
                    machines_orders[i][j] = 0
                else:
                    h_set.add(order)

    final_array = np.zeros((total_machine, time_len), dtype=int)

    # Mark blocked periods
    for machine in range(len(final_array)):
        cell_linear = actualCellHeuCell[machine + 1]
        if cell_linear in block_data_dict:
            machine_available_from = block_data_dict[cell_linear]
            if machine_available_from <= start_limit:
                continue
            end = machine_available_from - start_limit
            final_array[machine][0:end] = -1

    # Assign orders to time slots
    for machine_index in range(len(machines_orders)):
        cell_linear = actualCellHeuCell[machine_index + 1]
        if cell_linear in block_data_dict:
            machine_available_from = block_data_dict[cell_linear]
            if machine_available_from > end_limit:
                continue

        for order_index in range(len(machines_orders[machine_index])):
            order = machines_orders[machine_index][order_index]
            if order != 0 and order != -1 and order <= total_order:
                actual_order_no = K_main[order - 1]
                additional_cell_requirement = OTk[order - 1]
                order_arrival_time = ArrivalTime[actual_order_no - 1]
                order_processing_time = PTk[order - 1]

                if start_limit > order_arrival_time:
                    order_arrival_time = start_limit

                if additional_cell_requirement == 1:
                    try:
                        free_from1 = np.where(final_array[machine_index] == 0)[0][0]
                        free_from2 = np.where(final_array[machine_index + 1] == 0)[0][0]
                        free_from = max(free_from1, free_from2)
                        free_from = max(free_from, order_arrival_time - start_limit)
                        while True:
                            sum1 = np.sum(final_array[machine_index][free_from:free_from + order_processing_time])
                            sum2 = np.sum(final_array[machine_index + 1][free_from:free_from + order_processing_time])
                            if sum1 == 0 and sum2 == 0:
                                break
                            free_from += 1
                        final_array[machine_index][free_from:free_from + order_processing_time] = actual_order_no
                        final_array[machine_index + 1][free_from:free_from + order_processing_time] = actual_order_no
                    except:
                        break

                if additional_cell_requirement == 0:
                    try:
                        free_from = np.where(final_array[machine_index] == 0)[0][0]
                        free_from = max(free_from, order_arrival_time - start_limit)
                        while True:
                            s = np.sum(final_array[machine_index][free_from:free_from + order_processing_time])
                            if s == 0:
                                break
                            free_from += 1
                        final_array[machine_index][free_from:free_from + order_processing_time] = actual_order_no
                    except:
                        break

    return final_array


def _calculate_return_value(final_array, actualCellHeuCell, start_limit, end_limit):
    """Convert time-indexed array back to assignment dict.

    Returns dict: {order_id: [bank, cell, start_time, end_time]}
    """
    final_assignment = {}
    for machine in range(len(final_array)):
        actual_machine = actualCellHeuCell[machine + 1]
        (bank, cell) = _num_to_bank_cell(actual_machine)
        unique = np.unique(final_array[machine])
        for order in unique:
            if order <= 0:
                continue
            start = np.where(final_array[machine] == order)[0][0]
            end = np.where(final_array[machine] == order)[0][-1]
            final_assignment[int(order)] = [bank, cell,
                                            start_limit + start, start_limit + end]
    return final_assignment


def _initial_order_generation(block_data_from_file, sorted_orders,
                               orders_with_compatible_cell, start_time,
                               end_time, K_main):
    """Generate one initial solution for GA population.

    Port of initial_solution.py initial_order_generation
    """
    # Use first 54 cells (4 banks * 14 - 2 missing = 54, but block_data may be 42 for 3 banks)
    total_cells = NUM_BANKS * NUM_CELLS_PER_BANK  # 56
    time_len = end_time - start_time + 1
    unassigned = []

    final_array = np.zeros((total_cells, time_len), dtype=int)

    # Set blocked periods
    if block_data_from_file is not None:
        for x in block_data_from_file:
            row = (int(x[0]) - 1) * 14 + (int(x[1]) - 1)
            if row >= total_cells:
                continue
            start_position = int(x[2])
            end_position = int(x[3])
            if start_time > end_position:
                continue
            if start_time > start_position:
                start_position = start_time
            if end_time < end_position:
                end_position = end_time
            try:
                final_array[row][start_position - start_time:end_position - start_time + 1] = -1
            except:
                pass

    np.random.shuffle(sorted_orders)

    for x in range(len(sorted_orders)):
        order_no = int(sorted_orders[x][0])
        processing_time = int(sorted_orders[x][2])
        otk = int(sorted_orders[x][3])
        arrival = int(sorted_orders[x][4]) - start_time

        compatible_cell = orders_with_compatible_cell[order_no]
        compatible_cell_available_time = []
        for i in compatible_cell:
            bank = int(i.split(",")[0])
            cell = int(i.split(",")[1])
            row = (bank - 1) * 14 + (cell - 1)
            try:
                available_from = np.where(final_array[row] == 0)[0][0]
                compatible_cell_available_time.append([row, available_from])
            except:
                compatible_cell_available_time.append([row, end_time])
        compatible_cell_available_time = np.asarray(compatible_cell_available_time, dtype=int)

        assigned = 0

        if otk == 0:
            for i in range(len(compatible_cell_available_time)):
                row = compatible_cell_available_time[i][0]
                available_from = compatible_cell_available_time[i][1]
                start = max(available_from, arrival)
                while start + processing_time < end_time:
                    if np.sum(final_array[row][start:start + processing_time]) == 0:
                        final_array[row][start:start + processing_time] = order_no
                        assigned = 1
                        break
                    start += 1
                try:
                    np.where(final_array == order_no)[0][0]
                    break
                except:
                    continue
            if assigned == 0:
                for i in range(len(compatible_cell_available_time)):
                    row = compatible_cell_available_time[i][0]
                    available_from = compatible_cell_available_time[i][1]
                    start = max(available_from, arrival)
                    if start + processing_time >= end_time and start < end_time:
                        if np.sum(final_array[row][start:start + processing_time]) == 0:
                            final_array[row][start:start + processing_time] = order_no
                            break

        if otk == 1:
            for i in range(len(compatible_cell_available_time) - 1):
                row = compatible_cell_available_time[i][0]
                row2 = compatible_cell_available_time[i + 1][0]
                if row == (row2 - 1) and (row != 13) and (row != 27) and (row != 41) and (row != 55):
                    available_from = compatible_cell_available_time[i][1]
                    available_from2 = compatible_cell_available_time[i + 1][1]
                    start = max(max(available_from, available_from2), arrival)
                    while start + processing_time < end_time:
                        if (np.sum(final_array[row][start:start + processing_time]) == 0 and
                            np.sum(final_array[row + 1][start:start + processing_time]) == 0):
                            final_array[row][start:start + processing_time] = order_no
                            final_array[row + 1][start:start + processing_time] = order_no
                            assigned = 1
                            break
                        start += 1
                    try:
                        np.where(final_array == order_no)[0][0]
                        break
                    except:
                        continue
            if assigned == 0:
                for i in range(len(compatible_cell_available_time) - 1):
                    row = compatible_cell_available_time[i][0]
                    row2 = compatible_cell_available_time[i + 1][0]
                    if row == (row2 - 1) and (row != 13) and (row != 27) and (row != 41) and (row != 55):
                        available_from = compatible_cell_available_time[i][1]
                        available_from2 = compatible_cell_available_time[i + 1][1]
                        start = max(max(available_from, available_from2), arrival)
                        if start + processing_time >= end_time and start < end_time:
                            if (np.sum(final_array[row][start:start + processing_time]) == 0 and
                                np.sum(final_array[row + 1][start:start + processing_time]) == 0):
                                final_array[row][start:start + processing_time] = order_no
                                final_array[row + 1][start:start + processing_time] = order_no
                                break

    for i in K_main:
        try:
            np.where(final_array == i)[0][0]
        except:
            unassigned.append(i)

    return final_array, unassigned


def _get_initial_solution(VH, VL, W, RH, RL, RW, I, J, K_main, DueTime,
                          PTk, OTk, ArrivalTime, start_time, end_time,
                          block_data_from_file, possible_order_per_row,
                          orders_with_compatible_cell_ga, actualCellHeuCell):
    """Generate one initial solution for GA population."""
    DueTime_k_main = [DueTime[k - 1] for k in K_main]
    ArrivalTime_k_main = [ArrivalTime[k - 1] for k in K_main]

    sorted_orders, _ = _sort_order_by_due_date(
        K=K_main, DueTime=DueTime_k_main, PTk=PTk,
        OTk=OTk, arrival_time=ArrivalTime_k_main)

    orders_with_compatible_cell_init = _find_compatible_cell_for_initial(
        VH, VL, W, RH, RL, RW, I, J, K_main)

    initial_assignment, unassigned = _initial_order_generation(
        block_data_from_file, copy.deepcopy(sorted_orders),
        orders_with_compatible_cell_init, start_time, end_time, K_main)

    # Convert to GA chromosome format
    total_machine = len(actualCellHeuCell)
    arr2 = np.zeros((total_machine, possible_order_per_row), dtype=int)

    for i in range(min(len(initial_assignment), total_machine)):
        if i >= len(actualCellHeuCell):
            break
        unique = np.unique(initial_assignment[actualCellHeuCell.get(i+1, i) - 1]
                          if (i+1) in actualCellHeuCell and actualCellHeuCell[i+1] - 1 < len(initial_assignment)
                          else np.array([0]))
        arr_index = 0
        for j in range(len(unique)):
            if unique[j] != -1 and unique[j] != 0:
                if arr_index < possible_order_per_row:
                    arr2[i][arr_index] = K_main.index(unique[j]) + 1
                    arr_index += 1

    # Place unassigned orders
    for order in unassigned:
        if order not in K_main:
            continue
        otk_val = OTk[K_main.index(order)]
        order_idx = K_main.index(order) + 1
        placed = False
        for m in range(total_machine):
            try:
                pos = np.where(arr2[m] == 0)[0][0]
                if otk_val == 0:
                    arr2[m][pos] = order_idx
                    placed = True
                    break
                elif otk_val == 1 and m + 1 < total_machine:
                    arr2[m][pos] = order_idx
                    placed = True
                    break
            except:
                continue

    return arr2.flatten()


def _get_solution_set(VH, VL, W, RH, RL, RW, I, J, K_main, DueTime,
                      PTk, OTk, ArrivalTime, start_time, end_time,
                      block_data_from_file, possible_order_per_row,
                      total_generation, orders_with_compatible_cell_ga,
                      actualCellHeuCell):
    """Generate initial population for GA."""
    total_solution = []
    for i in range(total_generation):
        solution = _get_initial_solution(
            VH, VL, W, RH, RL, RW, I, J, K_main, DueTime,
            PTk, OTk, ArrivalTime, start_time, end_time,
            block_data_from_file, possible_order_per_row,
            orders_with_compatible_cell_ga, actualCellHeuCell)
        if len(total_solution) == 0:
            total_solution = np.expand_dims(solution, axis=0)
        else:
            total_solution = np.append(
                total_solution, np.expand_dims(solution, axis=0), axis=0)
    return total_solution


def heu_model(K_h, order_with_remaining_time, start_limit, end_limit,
              arrival_time, DueTime, VH, VL, RH_h, RL_h, W, RW_h,
              I, J, ki, kj, PTk_h, OTk_h, K_main, PTk_main,
              block_list_path=None):
    """Full GA heuristic for 2TC server assignment.

    Port of Paper_1_Codes/heu/heuristics.py
    Uses pymoo GA with Dask for parallel evaluation.
    """
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.optimize import minimize
        from pymoo.termination.robust import RobustTermination
        from pymoo.termination.ftol import SingleObjectiveSpaceTermination
        from pymoo.termination import get_termination
        from pymoo.core.termination import TerminateIfAny
        from pymoo.core.population import Population
        from pymoo.core.evaluator import Evaluator
    except ImportError:
        print("WARNING: pymoo not installed. GA heuristic unavailable.")
        print("Install with: pip install pymoo")
        return {}, list(K_h)

    # Read block data
    if block_list_path is None:
        block_list_path = os.path.join(INPUT_DIR, 'block_list.xlsx')
    block_data = read_block_data(block_list_path)
    block_data_dict = read_block_data_as_dict(block_data)

    time_len = end_limit - start_limit + 1
    total_order = len(K_h)

    if total_order == 0:
        return {}, []

    K_main_list = list(K_main) if not isinstance(K_main, list) else K_main
    ArrivalTime = list(arrival_time) if not isinstance(arrival_time, list) else arrival_time

    # Find compatible cells
    orders_with_compatible_cell, actualCellHeuCell = _find_compatible_cell_for_ga(
        VH, VL, W, RH_h, RL_h, W, RW_h,
        I, J, K_h)

    total_machine = len(actualCellHeuCell)
    possible_order = math.ceil(time_len / min(PTk_h) if len(PTk_h) > 0 else time_len)

    print(f"GA Setup: {total_order} orders, {total_machine} machines, "
          f"window [{start_limit}-{end_limit}]")

    # Generate initial solutions
    initial_solution = _get_solution_set(
        VH, VL, W, RH_h, RL_h, RW_h, I, J, K_h,
        DueTime, PTk_h, OTk_h, ArrivalTime,
        start_limit, end_limit, block_data,
        possible_order, 100,
        orders_with_compatible_cell, actualCellHeuCell)

    # Define GA optimization problem
    class HeuristicProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=total_machine * possible_order,
                n_obj=1,
                xl=0,
                xu=total_order + 1)

        def _evaluate(self, x, out, *args, **kwargs):
            solution_matrix = np.reshape(x.astype(int), (total_machine, possible_order))
            solution_matrix[solution_matrix > total_order] = 0
            solution_matrix[solution_matrix < 0] = 0

            final_array = _calculate_assignment(
                solution_matrix, total_machine, time_len,
                block_data_dict, start_limit, end_limit,
                K_h, total_order, ArrivalTime, DueTime,
                OTk_h, PTk_h, actualCellHeuCell)

            # Objective: minimize unassigned orders (weighted by cost)
            assigned_count = 0
            total_tardiness = 0
            for order in K_h:
                try:
                    rows = np.where(final_array == order)
                    if len(rows[0]) > 0:
                        assigned_count += 1
                        end_time = rows[1].max() + start_limit
                        if end_time > DueTime[order - 1]:
                            total_tardiness += (end_time - DueTime[order - 1])
                except:
                    pass

            unassigned_count = total_order - assigned_count
            out["F"] = unassigned_count * 1000 + total_tardiness

    problem = HeuristicProblem()

    # Set up Dask for parallel evaluation
    try:
        from dask.distributed import Client
        client = Client(threads_per_worker=1, n_workers=4)
        print(f"Dask cluster: {client.dashboard_link}")
    except:
        client = None
        print("Running GA without Dask parallelization")

    # Create population from initial solutions
    pop = Population.new("X", initial_solution)
    Evaluator().eval(problem, pop)

    algorithm = GA(
        pop_size=100,
        sampling=pop,
        eliminate_duplicates=True,
    )

    termination = TerminateIfAny(
        RobustTermination(SingleObjectiveSpaceTermination(tol=1e-6), period=100),
        get_termination("n_gen", 4000))

    res = minimize(
        problem, algorithm, termination,
        seed=47, verbose=True)

    if client is not None:
        client.close()
        print("Dask shutdown")

    # Extract final solution
    final_solution = np.reshape(res.X.astype(int), (total_machine, possible_order))
    final_solution[final_solution > total_order] = 0
    final_solution[final_solution < 0] = 0

    final_array = _calculate_assignment(
        final_solution, total_machine, time_len,
        block_data_dict, start_limit, end_limit,
        K_h, total_order, ArrivalTime, DueTime,
        OTk_h, PTk_h, actualCellHeuCell)

    final_assignment = _calculate_return_value(
        final_array, actualCellHeuCell, start_limit, end_limit)

    # Find unassigned
    unassigned = [order for order in K_h if order not in final_assignment]
    print(f"GA Result: {len(final_assignment)} assigned, {len(unassigned)} unassigned")

    return final_assignment, unassigned


def modify_heu_output(heuristic_assignment, heu_unassigned, OTk):
    """Filter GA output to keep only 2TC assignments.

    Port of Paper_1_Codes/heu/modify_heu_out.py
    """
    modified_heu_assignment = {}
    for heu_ass in heuristic_assignment:
        if OTk[heu_ass - 1] == 1:
            modified_heu_assignment[heu_ass] = heuristic_assignment[heu_ass]

    modified_heu_unassigned = []
    for order in heu_unassigned:
        if OTk[order - 1] == 1:
            modified_heu_unassigned.append(order)

    return modified_heu_assignment, modified_heu_unassigned


# =============================================================================
# SECTION 7: RL ENVIRONMENT — LexicographicRewardEnvironment
# =============================================================================

class LexicographicRewardEnvironment(gym.Env):
    """Gym environment for CTO server scheduling with lexicographic rewards.

    State Space (~52 dims with windowing):
      - Current server features: OTk, RH, RW, PTk_norm, DueTime_norm, priority, urgency
      - System state: current_time, server_progress, completion_metrics
      - Cell availability by type
      - Future lookahead (next 5 servers, window stats)
      - Tardiness risk and slack information

    Action Space: Discrete(total_usable_cells + 1)
      - Actions 0 to N-1: Assign to specific cell index
      - Action N: Skip server

    Rewards: Tuple (R1, R2)
      - R1: Completion reward (+10 assign, -15 skip when valid exists)
      - R2: Tardiness reward (+3 early, -3 tardy)

    Based on RL 18 notebook with improvements for warm-start quality.
    """

    def __init__(self, facility, servers_df, window_length=10,
                 use_windowing=True, block_list_path=None,
                 randomize_order=False,
                 use_deferred_actions=False,
                 use_fpr=False,
                 use_structured_state=False):
        super().__init__()

        self.facility = facility
        self.window_length = window_length
        self.use_windowing = use_windowing
        self.randomize_order = randomize_order

        # DAF-LHAC feature flags
        self.use_deferred_actions = use_deferred_actions  # Innovation 1: DAP
        self.use_fpr = use_fpr                            # Innovation 3: FPR
        self.use_structured_state = use_structured_state  # Innovation 2: CASE
        self._efs_cache = {}        # {cell_idx: earliest_feasible_start}
        self._cached_phi = 0.0      # Feasibility potential cache for FPR
        self.gamma = 0.99           # Discount factor for potential shaping

        # Build cell index mapping (only FUL cells)
        self.ful_cells = [c for c in facility.all_cells.values()
                          if c.status == CellStatus.FUL]
        self.total_cells = len(self.ful_cells)
        self.cell_to_idx = {cell.cell_id: i for i, cell in enumerate(self.ful_cells)}
        self.idx_to_cell = {i: cell for i, cell in enumerate(self.ful_cells)}

        # Build adjacent cell pairs for 2TC
        self.adjacent_cells = {}
        for idx in range(self.total_cells):
            cell = self.idx_to_cell[idx]
            bank_id = cell.bank_id
            position = cell.position
            for adj_idx in range(self.total_cells):
                adj_cell = self.idx_to_cell[adj_idx]
                if adj_cell.bank_id == bank_id and adj_cell.position == position + 1:
                    self.adjacent_cells[idx] = adj_idx
                    break

        # Load block list for fabrication periods
        self.block_periods = {}  # {cell_idx: [(start, end), ...]}
        if block_list_path is None:
            block_list_path = os.path.join(INPUT_DIR, 'block_list.xlsx')
        if os.path.exists(block_list_path):
            self._load_block_list(block_list_path)

        # Pre-compute per-bank cell lists (avoids rebuilding in _get_state)
        self.bank_cell_indices = {}
        for bank_idx in range(NUM_BANKS):
            self.bank_cell_indices[bank_idx] = [
                i for i in range(self.total_cells)
                if self.idx_to_cell[i].bank_id == bank_idx + 1
            ]

        # Process servers DataFrame
        self.original_df = servers_df.copy()
        self._prepare_servers(servers_df)

        # Observation and action spaces
        self.state_dim = 78 if use_deferred_actions else 74  # +4 DAP features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.total_cells + 1)  # +1 for skip

        # Window parameters
        self.total_time = TOTAL_TIME_BLOCKS
        if use_windowing:
            self.window_start = 0
            self.window_end = min(window_length, self.total_time)
        else:
            self.window_start = 0
            self.window_end = self.total_time

        # State tracking
        self.cell_occupancy = np.zeros((self.total_cells, self.total_time + 1), dtype=int)
        self._init_block_occupancy()

        self.current_server_idx = 0
        self.assigned_servers = set()
        self.skipped_servers = set()
        self.assignments = {}
        self.tardiness_count = 0
        self._cached_valid_actions = None

    def _load_block_list(self, path):
        """Load fabrication block periods from block_list.xlsx."""
        try:
            block_data = read_block_data(path)
            if block_data is None:
                return
            for row in block_data:
                bank, cell = int(row[0]), int(row[1])
                start_block, end_block = int(row[2]), int(row[3])
                # Find matching cell in our FUL cells
                for idx, c in self.idx_to_cell.items():
                    if c.bank_id == bank and c.position == cell:
                        if idx not in self.block_periods:
                            self.block_periods[idx] = []
                        self.block_periods[idx].append((start_block, end_block))
                        break
        except:
            pass

    def _init_block_occupancy(self):
        """Mark fabrication-blocked periods in occupancy grid."""
        for cell_idx, periods in self.block_periods.items():
            for start, end in periods:
                s = max(0, start)
                e = min(end + 1, self.total_time + 1)
                self.cell_occupancy[cell_idx, s:e] = -1

    def _prepare_servers(self, df):
        """Prepare server data from DataFrame."""
        self.servers = []
        required_cols = ['PTk', 'OTk', 'RH', 'RL', 'RW', 'DueTime', 'ArrivalTime']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        for idx, row in df.iterrows():
            server = {
                'idx': idx,
                'k': row.get('k', idx + 1),
                'PTk': int(row['PTk']),
                'OTk': int(row['OTk']),
                'RH': int(row['RH']),
                'RL': int(row.get('RL', 0)),
                'RW': int(row['RW']),
                'DueTime': int(row['DueTime']),
                'ArrivalTime': int(row['ArrivalTime']),
                'type': row.get('type', 'unknown'),
            }
            self.servers.append(server)

        # Compute slack for all servers
        for s in self.servers:
            s['slack'] = s['DueTime'] - s['ArrivalTime'] - s['PTk']

        if self.randomize_order:
            # During training: randomize with bias toward smart orderings
            # that prevent fragmentation. Key insight: 2TC + long-PT servers
            # should be placed FIRST to secure adjacent pairs before fragmentation.
            r = random.random()
            if r < 0.35:
                # 2TC-first + slack sort: prevents fragmentation (best strategy)
                # 2TC servers sorted by slack (tightest first), then 1TC by slack
                self.servers.sort(key=lambda s: (-s['OTk'], s['slack'], -s['PTk']))
            elif r < 0.55:
                # Slack sort (canonical) with 2TC tiebreaker
                self.servers.sort(key=lambda s: (s['slack'], -s['OTk'], s['DueTime']))
            elif r < 0.70:
                # Long-PT first (Artemis→Themis→Athena), then by slack
                self.servers.sort(key=lambda s: (-s['PTk'], -s['OTk'], s['slack']))
            elif r < 0.85:
                random.shuffle(self.servers)
            else:
                self.servers.sort(key=lambda s: (s['ArrivalTime'], s['slack']))
        else:
            # Inference: 2TC-first sort to prevent fragmentation
            # This is the key ordering that prevents adjacent pair fragmentation
            self.servers.sort(key=lambda s: (-s['OTk'], s['slack'], -s['PTk']))

        # Calculate priority ranks
        priority_sorted = sorted(self.servers,
                                 key=lambda s: (s['DueTime'], s['ArrivalTime']))
        for rank, s in enumerate(priority_sorted):
            s['priority_rank'] = rank
            s['priority_cost'] = (len(self.servers) - rank) * 100000

    def _is_cell_compatible(self, cell_idx, server):
        """Check if a cell is structurally compatible with a server (ignoring time)."""
        cell = self.idx_to_cell[cell_idx]
        if cell.status != CellStatus.FUL:
            return False
        needs_hv = server['RH'] == 1
        needs_lv = server.get('RL', 0) == 1
        needs_water = server['RW'] == 1
        if needs_hv and cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
            return False
        if needs_lv and not needs_hv and cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
            return False
        if needs_water and cell.water_hoses == 0:
            return False
        is_2tc = server['OTk'] == 1
        if is_2tc:
            if cell_idx not in self.adjacent_cells:
                return False
            adj_idx = self.adjacent_cells[cell_idx]
            adj_cell = self.idx_to_cell[adj_idx]
            if adj_cell.status != CellStatus.FUL:
                return False
            if needs_hv and adj_cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
                return False
            if needs_lv and not needs_hv and adj_cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
                return False
            if needs_water and adj_cell.water_hoses == 0:
                return False
        return True

    def reset(self, seed=None, options=None):
        """Reset environment to initial state.

        CRITICAL: Must re-prepare server list from original DataFrame to prevent
        exponential episode growth from retry mechanism extending self.servers.
        """
        super().reset(seed=seed)

        self.cell_occupancy = np.zeros((self.total_cells, self.total_time + 1), dtype=int)
        self._init_block_occupancy()

        self.current_server_idx = 0
        self.assigned_servers = set()
        self.skipped_servers = set()
        self.assignments = {}
        self.tardiness_count = 0
        self._cached_valid_actions = None  # Invalidated on step/reset

        # Retry mechanism: after each pass, re-queue skipped servers
        # _retry_count tracks pass number: 0=first pass, 1=first retry, 2=second retry
        self._retry_count = 0
        self._retry_pass = False  # Kept for backward compat with state[55]
        self._retry_queue = []

        # Re-prepare server list from original data to prevent retry extensions
        # from accumulating across episodes (was causing 372->41K step explosion)
        self._prepare_servers(self.original_df)

        if self.use_windowing:
            self.window_start = 0
            self.window_end = min(self.window_length, self.total_time)

        state = self._get_state()
        info = self._get_info()
        return state, info

    def _get_state(self):
        """Build state observation vector (74 dims).

        Dims 0-9:   Current server features
        Dims 10-14: System progress
        Dims 15-19: Cell availability by type
        Dims 20-23: Per-bank capacity
        Dims 24-43: Future lookahead (next 5 servers)
        Dims 44-47: Window statistics
        Dims 48-51: Completion/tardiness rates + window bounds
        Dims 52-57: Global capacity awareness
        Dims 58-73: Enhanced features (temporal profile, compatibility, pressure)
        """
        state = np.zeros(self.state_dim, dtype=np.float32)

        if self.current_server_idx >= len(self.servers):
            return state

        server = self.servers[self.current_server_idx]
        total = len(self.servers) - len(self._retry_queue)  # Original server count
        remaining = len(self.servers) - self.current_server_idx  # Remaining including retries
        arrival = server['ArrivalTime']
        pt = server['PTk']
        end_time = arrival + pt
        slack = server['DueTime'] - arrival - pt

        # Current server features (dims 0-9)
        state[0] = server['OTk']
        state[1] = server['RH']
        state[2] = server['RW']
        state[3] = pt / 10.0
        state[4] = server['DueTime'] / self.total_time
        state[5] = arrival / self.total_time
        state[6] = server['priority_rank'] / max(total, 1)
        state[7] = np.clip(slack / self.total_time, -1, 1)
        state[8] = 1.0 if slack < 0 else 0.0  # urgency flag
        state[9] = server['RL']

        # System progress (dims 10-14)
        state[10] = self.current_server_idx / max(total, 1)
        state[11] = len(self.assigned_servers) / max(total, 1)
        state[12] = len(self.skipped_servers) / max(total, 1)
        state[13] = self.tardiness_count / max(total, 1)
        state[14] = arrival / self.total_time

        # Cell availability summary (dims 15-19) — uses cached valid_actions
        valid_actions = self.get_valid_actions()
        num_valid = sum(1 for a in valid_actions if a < self.total_cells)
        state[15] = num_valid / max(self.total_cells, 1)

        hv_avail = lv_avail = water_avail = both_avail = 0
        for a in valid_actions:
            if a >= self.total_cells:
                continue
            cell = self.idx_to_cell[a]
            if cell.voltage_type == VoltageType.HV:
                hv_avail += 1
            elif cell.voltage_type == VoltageType.LV:
                lv_avail += 1
            elif cell.voltage_type == VoltageType.BOTH:
                both_avail += 1
            if cell.water_hoses > 0:
                water_avail += 1
        state[16] = hv_avail / max(self.total_cells, 1)
        state[17] = lv_avail / max(self.total_cells, 1)
        state[18] = both_avail / max(self.total_cells, 1)
        state[19] = water_avail / max(self.total_cells, 1)

        # Per-bank capacity (dims 20-23) — uses pre-computed bank_cell_indices
        t_slice_end = min(end_time, self.total_time + 1)
        for bank_idx in range(NUM_BANKS):
            bank_cells = self.bank_cell_indices[bank_idx]
            if not bank_cells:
                continue
            busy = 0
            for ci in bank_cells:
                if np.any(self.cell_occupancy[ci, arrival:t_slice_end] != 0):
                    busy += 1
            state[20 + bank_idx] = 1.0 - (busy / len(bank_cells))

        # Future lookahead - next 5 servers (dims 24-43)
        for look_idx in range(5):
            future_idx = self.current_server_idx + 1 + look_idx
            base_dim = 24 + look_idx * 4
            if future_idx < total:
                fs = self.servers[future_idx]
                state[base_dim] = fs['PTk'] / 10.0
                state[base_dim + 1] = fs['DueTime'] / self.total_time
                state[base_dim + 2] = fs['OTk']
                state[base_dim + 3] = (fs['DueTime'] - fs['ArrivalTime'] - fs['PTk']) / self.total_time

        # Window statistics (dims 44-47)
        state[44] = remaining / max(total, 1)
        twotc_remaining = sum(1 for i in range(self.current_server_idx, len(self.servers))
                              if self.servers[i]['OTk'] == 1)
        state[45] = twotc_remaining / max(remaining, 1)
        if remaining > 0:
            lookahead_end = min(self.current_server_idx + 20, len(self.servers))
            avg_slack = np.mean([
                self.servers[i]['DueTime'] - self.servers[i]['ArrivalTime'] - self.servers[i]['PTk']
                for i in range(self.current_server_idx, lookahead_end)
            ]) / self.total_time
            state[46] = np.clip(avg_slack, -1, 1)

        # Completion and tardiness rates so far (dims 48-51)
        processed = self.current_server_idx
        if processed > 0:
            state[48] = len(self.assigned_servers) / processed
            state[49] = self.tardiness_count / processed
        state[50] = self.window_start / self.total_time
        state[51] = self.window_end / self.total_time

        # === Global capacity awareness features (dims 52-57) ===
        # Dim 52: Total remaining demand / available capacity (utilization ratio)
        # Uses FULL time horizon (not just window) for accurate capacity estimation
        if remaining > 0:
            total_demand_blocks = sum(
                self.servers[i]['PTk'] * (2 if self.servers[i]['OTk'] == 1 else 1)
                for i in range(self.current_server_idx, len(self.servers))
            )
            t_now = arrival
            t_cap_end = self.total_time + 1  # Full horizon, not window_end
            free_blocks = int(np.sum(self.cell_occupancy[:, t_now:t_cap_end] == 0))
            state[52] = np.clip(total_demand_blocks / max(free_blocks, 1), 0.0, 3.0)

        # Dim 53: Fraction of remaining orders that are 2TC
        state[53] = twotc_remaining / max(remaining, 1)

        # Dim 54: Available adjacent pairs for 2TC placement
        adj_pairs_free = 0
        if arrival < self.cell_occupancy.shape[1]:
            for ci, ai in self.adjacent_cells.items():
                if (self.cell_occupancy[ci, arrival] == 0 and
                        self.cell_occupancy[ai, arrival] == 0):
                    adj_pairs_free += 1
        state[54] = adj_pairs_free / max(len(self.adjacent_cells), 1)

        # Dim 55: Retry pass indicator (0.0=first pass, 0.5=retry 1, 1.0=retry 2)
        # Kept as float for backward compat: >0.5 still means "in retry"
        state[55] = min(self._retry_count / 2.0, 1.0)

        # Dim 56: Current server's slack normalized
        state[56] = np.clip(slack / 20.0, -1.0, 1.0)

        # Dim 57: Is current server 2TC? (binary)
        state[57] = float(server['OTk'] == 1)

        # === Enhanced features (dims 58-73) ===

        # Dims 58-61: Temporal cell occupancy profile (next 4 time windows of 5 blocks)
        # How full is the facility in the near future?
        for w in range(4):
            t_start = arrival + w * 5
            t_end_w = min(t_start + 5, self.total_time + 1)
            if t_start < self.total_time and t_start < t_end_w:
                total_slots = self.total_cells * (t_end_w - t_start)
                free_count = int(np.sum(self.cell_occupancy[:, t_start:t_end_w] == 0))
                state[58 + w] = free_count / max(total_slots, 1)

        # Dim 62: Server-cell compatibility ratio (structural, ignoring time)
        # What fraction of cells can this server ever use?
        compatible_count = sum(1 for ci in range(self.total_cells)
                               if self._is_cell_compatible(ci, server))
        state[62] = compatible_count / max(self.total_cells, 1)

        # Dim 63: 2TC demand pressure = remaining_2tc / available_adjacent_pairs
        state[63] = np.clip(twotc_remaining / max(adj_pairs_free, 1), 0.0, 5.0)

        # Dims 64-67: Per-bank cells free during this server's processing window
        for bank_idx in range(NUM_BANKS):
            bank_cells = self.bank_cell_indices[bank_idx]
            if not bank_cells:
                continue
            free_for_server = sum(
                1 for ci in bank_cells
                if np.all(self.cell_occupancy[ci, arrival:t_slice_end] == 0))
            state[64 + bank_idx] = free_for_server / max(len(bank_cells), 1)

        # Dims 68-71: Slack distribution of remaining servers (min, 25th, 50th, 75th pctile)
        if remaining > 0:
            remaining_slacks = [
                self.servers[i].get('slack', 0)
                for i in range(self.current_server_idx, len(self.servers))
            ]
            if len(remaining_slacks) > 0:
                q = np.percentile(remaining_slacks, [0, 25, 50, 75])
                for qi in range(4):
                    state[68 + qi] = np.clip(q[qi] / 30.0, -1.0, 1.0)

        # Dim 72: Tardiness fraction of assigned servers so far
        if len(self.assigned_servers) > 0:
            state[72] = self.tardiness_count / len(self.assigned_servers)

        # Dim 73: Time pressure = remaining servers / remaining time blocks
        remaining_time = max(self.total_time - arrival, 1)
        state[73] = np.clip(remaining / remaining_time, 0.0, 5.0)

        # Dims 74-77: DAP features (deferred action placement)
        if self.use_deferred_actions and len(self._efs_cache) > 0:
            efs_delays = [efs - arrival for efs in self._efs_cache.values()]
            state[74] = min(efs_delays) / max(self.total_time, 1)  # Min delay
            state[75] = max(efs_delays) / max(self.total_time, 1)  # Max delay
            state[76] = np.mean(efs_delays) / max(self.total_time, 1)  # Mean delay
            deferred_count = sum(1 for d in efs_delays if d > 0)
            state[77] = deferred_count / max(len(efs_delays), 1)  # Fraction deferred

        return np.nan_to_num(state, nan=0.0)

    def get_valid_actions(self):
        """Return list of valid actions for current server (cached)."""
        if self._cached_valid_actions is not None:
            return self._cached_valid_actions

        if self.current_server_idx >= len(self.servers):
            self._cached_valid_actions = [self.total_cells]
            return self._cached_valid_actions

        server = self.servers[self.current_server_idx]
        valid = []
        is_2tc = server['OTk'] == 1
        needs_hv = server['RH'] == 1
        needs_lv = server.get('RL', 0) == 1
        needs_water = server['RW'] == 1

        arrival = server['ArrivalTime']
        pt = server['PTk']
        end_time = arrival + pt

        for cell_idx in range(self.total_cells):
            cell = self.idx_to_cell[cell_idx]

            # Check FUL status
            if cell.status != CellStatus.FUL:
                continue

            # Check voltage compatibility
            if needs_hv and cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
                continue
            if needs_lv and not needs_hv and cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
                continue

            # Check water cooling
            if needs_water and cell.water_hoses == 0:
                continue

            # Check time availability
            if end_time > self.total_time + 1:
                continue
            occ_slice = self.cell_occupancy[cell_idx, arrival:min(end_time, self.total_time + 1)]
            if np.any(occ_slice != 0):  # Reject occupied (>0) AND FAB-blocked (-1) cells
                continue

            # 2TC: check adjacent cell
            if is_2tc:
                if cell_idx not in self.adjacent_cells:
                    continue
                adj_idx = self.adjacent_cells[cell_idx]
                adj_cell = self.idx_to_cell[adj_idx]

                if adj_cell.status != CellStatus.FUL:
                    continue
                if needs_hv and adj_cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
                    continue
                if needs_lv and not needs_hv and adj_cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
                    continue
                if needs_water and adj_cell.water_hoses == 0:
                    continue

                adj_occ = self.cell_occupancy[adj_idx, arrival:min(end_time, self.total_time + 1)]
                if np.any(adj_occ != 0):  # Reject occupied (>0) AND FAB-blocked (-1) cells
                    continue

            valid.append(cell_idx)

        # Always allow skip
        valid.append(self.total_cells)
        self._cached_valid_actions = valid
        return valid

    # ===================================================================
    # DAP (Deferred Action Placement) — Innovation 1 of DAF-LHAC
    # ===================================================================

    def _find_earliest_feasible_start(self, cell_idx, arrival, latest_start,
                                       pt, is_2tc):
        """Find earliest feasible start time for a cell using sliding window.

        Uses cumulative-sum trick for O(T) scan per cell.
        For 2TC servers, combines primary + adjacent cell occupancy.

        Returns:
            int or None: earliest feasible start, or None if infeasible.
        """
        max_end = min(latest_start + pt, self.total_time + 1)
        if max_end <= arrival or latest_start < arrival:
            return None

        # Build binary occupancy: 1 = occupied/blocked, 0 = free
        occ_slice = self.cell_occupancy[cell_idx, arrival:max_end]
        occ_primary = (occ_slice != 0).astype(np.int32)

        if is_2tc:
            adj_idx = self.adjacent_cells[cell_idx]
            adj_slice = self.cell_occupancy[adj_idx, arrival:max_end]
            occ_combined = occ_primary | (adj_slice != 0).astype(np.int32)
        else:
            occ_combined = occ_primary

        window_len = len(occ_combined)
        if window_len < pt:
            return None

        # Sliding window via cumulative sum: find first window of pt zeros
        cumsum = np.cumsum(occ_combined)
        cumsum_padded = np.concatenate(([0], cumsum))
        window_sums = cumsum_padded[pt:] - cumsum_padded[:len(cumsum_padded) - pt]

        zero_windows = np.where(window_sums == 0)[0]
        if len(zero_windows) == 0:
            return None

        efs_offset = zero_windows[0]
        efs = arrival + efs_offset

        if efs > latest_start:
            return None

        return int(efs)

    def get_valid_actions_deferred(self):
        """DAP: Return valid actions checking ALL feasible start times.

        For each cell, scans from arrival to latest_feasible_start for any
        free window of duration PT. Stores earliest feasible start (EFS)
        in self._efs_cache for use in step().

        Action space is unchanged: Discrete(total_cells + 1).
        """
        if self._cached_valid_actions is not None:
            return self._cached_valid_actions

        if self.current_server_idx >= len(self.servers):
            self._cached_valid_actions = [self.total_cells]
            return self._cached_valid_actions

        server = self.servers[self.current_server_idx]
        valid = []
        self._efs_cache = {}

        is_2tc = server['OTk'] == 1
        needs_hv = server['RH'] == 1
        needs_lv = server.get('RL', 0) == 1
        needs_water = server['RW'] == 1
        arrival = server['ArrivalTime']
        pt = server['PTk']

        # Latest feasible start: must finish by due date or horizon end
        latest_start = min(
            server['DueTime'] - pt,
            self.total_time + 1 - pt
        )
        # If due date is too tight, try horizon-only constraint
        if latest_start < arrival:
            latest_start = self.total_time + 1 - pt
        if latest_start < arrival:
            # Cannot fit at all
            valid.append(self.total_cells)
            self._cached_valid_actions = valid
            return valid

        for cell_idx in range(self.total_cells):
            cell = self.idx_to_cell[cell_idx]

            # Structural compatibility checks (same as get_valid_actions)
            if cell.status != CellStatus.FUL:
                continue
            if needs_hv and cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
                continue
            if needs_lv and not needs_hv and cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
                continue
            if needs_water and cell.water_hoses == 0:
                continue

            # 2TC structural check
            if is_2tc:
                if cell_idx not in self.adjacent_cells:
                    continue
                adj_idx = self.adjacent_cells[cell_idx]
                adj_cell = self.idx_to_cell[adj_idx]
                if adj_cell.status != CellStatus.FUL:
                    continue
                if needs_hv and adj_cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
                    continue
                if needs_lv and not needs_hv and adj_cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
                    continue
                if needs_water and adj_cell.water_hoses == 0:
                    continue

            # DAP: check ALL start times for this cell
            efs = self._find_earliest_feasible_start(
                cell_idx, arrival, latest_start, pt, is_2tc)

            if efs is not None:
                valid.append(cell_idx)
                self._efs_cache[cell_idx] = efs

        # Always allow skip
        valid.append(self.total_cells)
        self._cached_valid_actions = valid
        return valid

    def _completion_potential(self):
        """Estimate fraction of remaining servers that can still be placed.

        Used for potential-based reward shaping: Phi(s) = estimated completable fraction.
        Potential-based shaping is provably optimal-policy preserving.
        """
        remaining = len(self.servers) - self.current_server_idx
        if remaining == 0:
            return 0.0
        # Quick heuristic: check next ~20 servers for any valid placement
        check_end = min(self.current_server_idx + 20, len(self.servers))
        completable = 0
        for i in range(self.current_server_idx, check_end):
            s = self.servers[i]
            t = s['ArrivalTime']
            if t < self.cell_occupancy.shape[1]:
                # Check if at least one compatible cell is free at arrival
                for ci in range(self.total_cells):
                    if self._is_cell_compatible(ci, s):
                        pt_end = min(t + s['PTk'], self.total_time + 1)
                        if not np.any(self.cell_occupancy[ci, t:pt_end] != 0):
                            completable += 1
                            break
        return completable / max(check_end - self.current_server_idx, 1)

    def step(self, action):
        """Execute action and return (state, rewards, done, truncated, info).

        Features a multi-pass retry mechanism (up to 2 retries):
          Pass 0: initial pass through all servers
          Pass 1: retry with same order (facility state has evolved)
          Pass 2: retry with urgency-first order (tightest slack first)
        This breaks the 77% ceiling from single-pass sequential processing.
        """
        self._cached_valid_actions = None  # Invalidate cache on state change

        if self.current_server_idx >= len(self.servers):
            return self._get_state(), (0.0, 0.0), True, False, self._get_info()

        server = self.servers[self.current_server_idx]
        # DAP: use deferred valid actions when enabled
        if self.use_deferred_actions:
            valid_actions = self.get_valid_actions_deferred()
        else:
            valid_actions = self.get_valid_actions()
        r1, r2 = 0.0, 0.0

        # FPR: cache potential BEFORE action
        if self.use_fpr:
            phi_before = self._cached_phi

        # Fix: store the actual executed action (not the originally invalid one)
        if action not in valid_actions:
            action = self.total_cells  # Force skip if invalid

        if action < self.total_cells:
            # Assign server to cell
            cell = self.idx_to_cell[action]
            arrival = server['ArrivalTime']
            pt = server['PTk']
            server_id = server['k']

            # DAP: use earliest feasible start if available, else arrival
            if self.use_deferred_actions and action in self._efs_cache:
                start_time = self._efs_cache[action]
            else:
                start_time = arrival
            end_time = start_time + pt

            # Occupy the cell from start_time (may be deferred)
            self.cell_occupancy[action, start_time:min(end_time, self.total_time + 1)] = server_id

            # 2TC: occupy adjacent cell too
            if server['OTk'] == 1 and action in self.adjacent_cells:
                adj_idx = self.adjacent_cells[action]
                self.cell_occupancy[adj_idx, start_time:min(end_time, self.total_time + 1)] = server_id

            self.assigned_servers.add(self.current_server_idx)
            self.assignments[server_id] = {
                'cell_idx': action,
                'bank': cell.bank_id,
                'cell': cell.position,
                'start': start_time,
                'end': end_time - 1,
            }

            # R1: Completion reward — higher for harder-to-place orders
            is_2tc = server['OTk'] == 1
            slack = server.get('slack', server['DueTime'] - server['ArrivalTime'] - server['PTk'])
            r1 = 12.0 if is_2tc else 10.0

            # Bonus for assigning tight-deadline orders
            if slack < 0:
                r1 += 5.0  # Critical urgency
            elif slack < 5:
                r1 += 3.0  # Tight deadline

            # Extra bonus during retry pass (recovered a previously-skipped server)
            if self._retry_count > 0:
                r1 += 8.0  # Strong incentive to recover skipped servers
                if self._retry_count == 2:
                    r1 += 4.0  # Extra bonus for last-chance recovery

            # Fragmentation penalty: penalize 1TC placements that break adjacent
            # pairs needed by future 2TC servers. This addresses the root cause
            # of the 77% completion plateau — 1TC servers fragmenting the cell
            # grid so later 2TC servers become unplaceable.
            if not is_2tc and action in self.adjacent_cells:
                # This 1TC server is in a cell that's part of an adjacent pair
                adj_idx = self.adjacent_cells[action]
                adj_arrival = self.cell_occupancy[adj_idx, arrival:min(end_time, self.total_time + 1)]
                pair_was_free = not np.any(adj_arrival > 0)  # Free = no occupied servers

                if pair_was_free:
                    # We just broke a free adjacent pair — check if 2TC servers remain
                    remaining_2tc = sum(
                        1 for i in range(self.current_server_idx, len(self.servers))
                        if self.servers[i]['OTk'] == 1
                        and i not in self.assigned_servers
                    )
                    if remaining_2tc > 0:
                        # Count total free adjacent pairs at this time
                        free_pairs = 0
                        for ci, ai in self.adjacent_cells.items():
                            if ci == action:
                                continue  # Skip the pair we just broke
                            if (self.cell_occupancy[ci, arrival] <= 0 and
                                    self.cell_occupancy[ai, arrival] <= 0):
                                free_pairs += 1
                        # Scarcity-based penalty: fewer free pairs → harsher penalty
                        if remaining_2tc > 0 and free_pairs <= remaining_2tc:
                            # Scarce: more 2TC demand than available pairs
                            r1 -= 4.0
                        elif free_pairs <= remaining_2tc * 2:
                            # Tight: pairs are getting scarce
                            r1 -= 2.0
                        else:
                            # Mild: still enough pairs, small warning
                            r1 -= 0.5

            # Also check reverse direction: is this cell the adjacent partner?
            if not is_2tc:
                for ci, ai in self.adjacent_cells.items():
                    if ai == action:
                        # action is the right cell of pair (ci, ai)
                        ci_occ = self.cell_occupancy[ci, arrival:min(end_time, self.total_time + 1)]
                        if not np.any(ci_occ != 0):  # Free = no occupied servers or FAB blocks
                            remaining_2tc = sum(
                                1 for i in range(self.current_server_idx, len(self.servers))
                                if self.servers[i]['OTk'] == 1
                                and i not in self.assigned_servers
                            )
                            if remaining_2tc > 0:
                                free_pairs = 0
                                for ci2, ai2 in self.adjacent_cells.items():
                                    if ai2 == action or ci2 == action:
                                        continue
                                    if (self.cell_occupancy[ci2, arrival] <= 0 and
                                            self.cell_occupancy[ai2, arrival] <= 0):
                                        free_pairs += 1
                                if remaining_2tc > 0 and free_pairs <= remaining_2tc:
                                    r1 -= 4.0
                                elif free_pairs <= remaining_2tc * 2:
                                    r1 -= 2.0
                                else:
                                    r1 -= 0.5
                        break  # Each cell can only be the right partner of one pair

            # R2: Tardiness reward — denser signal
            if end_time <= server['DueTime']:
                slack_ratio = (server['DueTime'] - end_time) / max(server['PTk'], 1)
                r2 = 3.0 + min(slack_ratio, 3.0)  # More reward for more slack
            else:
                tardiness = end_time - server['DueTime']
                r2 = -3.0 - tardiness * 0.5
                self.tardiness_count += 1

        else:
            # Skip server
            self.skipped_servers.add(self.current_server_idx)
            num_valid_cells = sum(1 for a in valid_actions if a < self.total_cells)

            if num_valid_cells > 0:
                # Option-ratio skip penalty: penalty proportional to wasted options
                option_ratio = num_valid_cells / max(self.total_cells, 1)
                r1 = -8.0 - 12.0 * option_ratio  # -8 to -20 based on wasted options

                # Extra penalty if this server will be very hard to place later
                slack = server.get('slack', server['DueTime'] - server['ArrivalTime'] - server['PTk'])
                if slack < server['PTk']:
                    r1 -= 7.0  # Skipping a server that has very little slack left

                # Harsher penalty during retry pass (fewer chances remaining)
                if self._retry_count == 1:
                    r1 -= 5.0
                elif self._retry_count == 2:
                    r1 -= 10.0  # Last chance — strong penalty for skipping

                r2 = -2.0
            else:
                # No valid cells — truly forced skip
                r1 = -1.0
                r2 = 0.0

        # Advance to next server
        self.current_server_idx += 1

        # ---- RETRY MECHANISM (up to 2 retry passes) ----
        # After each pass ends, re-queue skipped servers for another attempt.
        # The facility state has evolved (other servers placed/skipped), so
        # previously-unavailable slots may now be free.
        # Pass 0 → retry_count becomes 1 (same ordering, second chance)
        # Pass 1 → retry_count becomes 2 (urgency-first: tightest slack first)
        # Pass 2 → done (max 2 retries to prevent episode explosion)
        pass_done = (self.current_server_idx >= len(self.servers)
                     and self._retry_count < 2)
        if pass_done and len(self.skipped_servers) > 0:
            # Collect skipped server data and re-append them
            skipped_indices = sorted(self.skipped_servers)
            retry_servers = [self.servers[si] for si in skipped_indices]

            self._retry_count += 1
            self._retry_pass = True  # For backward compat with state[55]

            if self._retry_count == 1:
                # First retry: 2TC-first with tight slack (prioritize hard-to-place)
                retry_servers.sort(key=lambda s: (-s['OTk'], s['slack'], -s['PTk']))
            elif self._retry_count == 2:
                # Second retry: urgency-first ordering (tightest slack first)
                # This complements the original ordering and catches servers
                # that were skipped due to ordering-related blind spots
                retry_servers.sort(key=lambda s: (s['slack'], -s['OTk'], s['DueTime']))

            # Track cumulative retry queue size for original_total calculation
            self._retry_queue.extend(retry_servers)
            # Append retry servers to the server list
            self.servers.extend(retry_servers)
            self.skipped_servers.clear()  # Reset — will track this retry pass skips

        # Check if truly done (all passes exhausted or no skips remain)
        done = self.current_server_idx >= len(self.servers)

        # Terminal rewards — graduated completion bonus
        if done:
            completion_rate = len(self.assigned_servers) / max(len(self.servers) - len(self._retry_queue), 1)
            # Use original server count for completion rate
            original_total = len(self.servers) - len(self._retry_queue)
            unfinished = original_total - len(self.assigned_servers)

            if completion_rate >= 1.0:
                r1 += 100.0
            elif completion_rate >= 0.98:
                r1 += 60.0
            elif completion_rate >= 0.95:
                r1 += 30.0
            elif completion_rate >= 0.90:
                r1 += 10.0

            # Per-unfinished penalty (capped to prevent gradient explosion)
            r1 -= min(unfinished * 5.0, 200.0)

            # Tardiness terminal
            if self.tardiness_count == 0 and len(self.assigned_servers) > 0:
                r2 += 30.0
            else:
                r2 -= self.tardiness_count * 2.0

        # FPR: Potential-based reward shaping (Innovation 3)
        if self.use_fpr and not done:
            phi_after = self._compute_feasibility_potential()
            self._cached_phi = phi_after
            shaping = self.gamma * phi_after - phi_before
            r1 = r1 + 8.0 * shaping  # Scale to match R1 magnitude

        state = self._get_state()
        info = self._get_info()
        return state, (r1, r2), done, False, info

    # ===================================================================
    # FPR (Feasibility-Preserving Reward) — Innovation 3 of DAF-LHAC
    # ===================================================================

    def _compute_feasibility_potential(self):
        """Compute Phi(s): fraction of remaining servers still feasibly placeable.

        Samples up to 30 future servers (deterministic stride) and checks
        if at least one cell has a feasible start time using DAP logic.

        Returns:
            float in [0, 1]
        """
        remaining_start = self.current_server_idx + 1
        remaining_end = len(self.servers)
        if remaining_start >= remaining_end:
            return 0.0

        total_remaining = remaining_end - remaining_start
        max_sample = 30
        if total_remaining > max_sample:
            stride = total_remaining // max_sample
            check_indices = list(range(remaining_start, remaining_end, stride))[:max_sample]
        else:
            check_indices = list(range(remaining_start, remaining_end))

        feasible_count = 0
        for idx in check_indices:
            srv = self.servers[idx]
            if srv['k'] in self.assignments:
                feasible_count += 1
                continue

            arrival = srv['ArrivalTime']
            pt = srv['PTk']
            is_2tc = srv['OTk'] == 1
            needs_hv = srv['RH'] == 1
            needs_lv = srv.get('RL', 0) == 1
            needs_water = srv['RW'] == 1

            latest = min(srv['DueTime'] - pt, self.total_time + 1 - pt)
            if latest < arrival:
                latest = self.total_time + 1 - pt
            if latest < arrival:
                continue  # Infeasible regardless

            found = False
            for ci in range(self.total_cells):
                cell = self.idx_to_cell[ci]
                # Quick structural check
                if cell.status != CellStatus.FUL:
                    continue
                if needs_hv and cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
                    continue
                if needs_lv and not needs_hv and cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
                    continue
                if needs_water and cell.water_hoses == 0:
                    continue
                if is_2tc and ci not in self.adjacent_cells:
                    continue

                efs = self._find_earliest_feasible_start(
                    ci, arrival, latest, pt, is_2tc)
                if efs is not None:
                    found = True
                    break  # Early exit: at least one cell works

            if found:
                feasible_count += 1

        return feasible_count / max(len(check_indices), 1)

    def _get_info(self):
        """Return info dict with current metrics."""
        # Use original server count (exclude retry duplicates)
        original_total = len(self.servers) - len(self._retry_queue)
        assigned = len(self.assigned_servers)
        return {
            'total_servers': original_total,
            'assigned_servers': assigned,
            'unfinished_servers': original_total - assigned,
            'skipped_servers': len(self.skipped_servers),
            'tardy_servers': self.tardiness_count,
            'completion_rate': assigned / max(original_total, 1),
            'assignments': self.assignments,
        }

    def get_assignments_for_warmstart(self):
        """Return assignments in format suitable for CPLEX warm-start.

        Returns dict: {server_id: [bank, cell, start_time, end_time]}
        """
        result = {}
        for server_id, info in self.assignments.items():
            result[server_id] = [info['bank'], info['cell'],
                                 info['start'], info['end']]
        return result


# =============================================================================
# SECTION 8: LEXICOGRAPHIC DQN AGENT
# =============================================================================

class NoisyLinear(nn.Module):
    """Factorized Noisy Linear layer for learned exploration.

    Replaces epsilon-greedy with parametric noise that the network learns
    to control, enabling state-dependent exploration.
    Reference: Fortunato et al., "Noisy Networks for Exploration", ICLR 2018.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DQNNetwork(nn.Module):
    """Dueling DQN with Noisy Networks.

    Architecture:
      Shared encoder -> Value stream (state value V(s))
                     -> Advantage stream (per-action advantage A(s,a))
      Q(s,a) = V(s) + A(s,a) - mean(A)

    Dueling separates 'is this state good?' from 'is this action good?',
    which is critical for scheduling where assign-vs-skip is the key choice.
    Noisy Networks replace epsilon-greedy with learned exploration.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super(DQNNetwork, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Shared encoder (deterministic layers)
        encoder_layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hd))
            encoder_layers.append(nn.LayerNorm(hd))
            encoder_layers.append(nn.ReLU())
            prev_dim = hd
        self.encoder = nn.Sequential(*encoder_layers)

        # Value stream (with noisy output)
        self.value_hidden = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
        )
        self.value_out = NoisyLinear(128, 1)

        # Advantage stream (with noisy output)
        self.advantage_hidden = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
        )
        self.advantage_out = NoisyLinear(128, output_dim)

        # Xavier initialization for deterministic layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        features = self.encoder(x)
        value = self.value_out(self.value_hidden(features))
        advantage = self.advantage_out(self.advantage_hidden(features))
        # Dueling aggregation: Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        self.value_out.reset_noise()
        self.advantage_out.reset_noise()


class NStepBuffer:
    """N-step return accumulator for better credit assignment.

    Collects n transitions and computes n-step discounted returns:
      R_n = r_0 + gamma*r_1 + ... + gamma^(n-1)*r_{n-1}
    Then stores (s_0, a_0, R_n_r1, R_n_r2, s_n, done_n) in replay buffer.

    This bridges per-step rewards (+10) with terminal rewards (+100)
    by looking n steps ahead, improving Q-value accuracy.
    """

    def __init__(self, n=3, gamma=0.99):
        from collections import deque
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def append(self, state, action, r1, r2, next_state, done, valid_actions):
        """Add transition to n-step buffer."""
        self.buffer.append((state, action, r1, r2, next_state, done, valid_actions))

    def is_ready(self):
        """Check if buffer has enough transitions for n-step return."""
        return len(self.buffer) == self.n

    def get(self):
        """Compute n-step return and return flattened transition."""
        r1_nstep = 0.0
        r2_nstep = 0.0
        for i, (_, _, r1, r2, _, d, _) in enumerate(self.buffer):
            r1_nstep += (self.gamma ** i) * r1
            r2_nstep += (self.gamma ** i) * r2
            if d:
                # Episode ended before n steps — truncate
                return (self.buffer[0][0], self.buffer[0][1],
                        r1_nstep, r2_nstep,
                        self.buffer[i][4], True, self.buffer[0][6])
        # Full n-step: use last transition's next_state
        last = self.buffer[-1]
        return (self.buffer[0][0], self.buffer[0][1],
                r1_nstep, r2_nstep,
                last[4], last[5], self.buffer[0][6])

    def flush(self):
        """Flush remaining transitions at episode end (partial n-step)."""
        results = []
        while len(self.buffer) > 0:
            r1_nstep = 0.0
            r2_nstep = 0.0
            for i, (_, _, r1, r2, _, d, _) in enumerate(self.buffer):
                r1_nstep += (self.gamma ** i) * r1
                r2_nstep += (self.gamma ** i) * r2
                if d:
                    results.append((self.buffer[0][0], self.buffer[0][1],
                                    r1_nstep, r2_nstep,
                                    self.buffer[i][4], True, self.buffer[0][6]))
                    break
            else:
                last = self.buffer[-1]
                results.append((self.buffer[0][0], self.buffer[0][1],
                                r1_nstep, r2_nstep,
                                last[4], last[5], self.buffer[0][6]))
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """Experience replay with prioritized sampling."""

    def __init__(self, capacity=1000000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state, action, r1, r2, next_state, done, valid_actions):
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        experience = (state, action, r1, r2, next_state, done, valid_actions)

        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            return None

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones(self.size) / self.size
        else:
            probs = probs / probs_sum

        indices = np.random.choice(self.size, min(batch_size, self.size),
                                   p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        states = torch.FloatTensor(np.array([s[0] for s in samples]))
        actions = torch.LongTensor([s[1] for s in samples])
        r1s = torch.FloatTensor([s[2] for s in samples])
        r2s = torch.FloatTensor([s[3] for s in samples])
        next_states = torch.FloatTensor(np.array([s[4] for s in samples]))
        dones = torch.FloatTensor([float(s[5]) for s in samples])
        weights = torch.FloatTensor(weights)

        return (states, actions, r1s, r2s, next_states, dones,
                weights, indices)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def __len__(self):
        return self.size


class ImprovedLexicographicDQNAgent:
    """Lexicographic DQN agent with dual Q-networks.

    Q1 optimizes completion (primary objective).
    Q2 optimizes tardiness (secondary objective, lexicographic).

    Action selection:
      1. Compute Q1 values for all valid actions
      2. Find actions within tolerance of max Q1
      3. Among those, select action maximizing Q2
    """

    def __init__(self, state_dim, action_dim, lr=5e-5, gamma=0.99,
                 epsilon_start=0.3, epsilon_end=0.01, epsilon_decay=0.9997,
                 tolerance_start=0.05, tolerance_end=0.005,
                 tolerance_decay=0.9999, target_update_freq=200,
                 device=None, phase2_boost=False):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gamma2 = gamma  # May change in phase 2
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tolerance = tolerance_start
        self.tolerance_end = tolerance_end
        self.tolerance_decay = tolerance_decay
        self.target_update_freq = target_update_freq
        self.train_step_count = 0
        self.stochastic_eval = False  # Set True for stochastic evaluation (mean +/- SD)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Q1 network (completion)
        self.q1_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = DQNNetwork(state_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1_network.state_dict())
        self.q1_optimizer = optim.Adam(self.q1_network.parameters(), lr=lr)
        self.q1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.q1_optimizer, T_max=24000, eta_min=1e-6)

        # Q2 network (tardiness)
        self.q2_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = DQNNetwork(state_dim, action_dim).to(self.device)
        self.q2_target.load_state_dict(self.q2_network.state_dict())
        self.q2_optimizer = optim.Adam(self.q2_network.parameters(), lr=lr)
        self.q2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.q2_optimizer, T_max=24000, eta_min=1e-6)

        # Experience replay with N-step returns
        self.memory = PrioritizedReplayBuffer(capacity=1000000)
        self.n_step_buffer = NStepBuffer(n=3, gamma=gamma)

        # Phase tracking
        self.phase2_active = False

    def activate_phase2(self):
        """Activate phase 2: increase gamma for Q2, focus on tardiness."""
        self.phase2_active = True
        self.gamma2 = 0.999
        # Reduce epsilon faster in phase 2
        self.epsilon = max(self.epsilon, 0.1)
        self.epsilon_decay = 0.9995

    def push_transition(self, state, action, r1, r2, next_state, done, valid_actions):
        """Push transition through N-step buffer into replay memory."""
        self.n_step_buffer.append(state, action, r1, r2, next_state, done, valid_actions)
        if self.n_step_buffer.is_ready():
            nstep = self.n_step_buffer.get()
            self.memory.push(*nstep)
        if done:
            # Flush remaining partial n-step transitions at episode end
            for nstep in self.n_step_buffer.flush():
                self.memory.push(*nstep)
            self.n_step_buffer.reset()

    def get_lexicographic_actions(self, state, valid_actions):
        """Select action using lexicographic Q-value ordering."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q1_values = self.q1_network(state_tensor).cpu().numpy()[0]
            q2_values = self.q2_network(state_tensor).cpu().numpy()[0]

        # Mask invalid actions
        mask = np.full(self.action_dim, -1e9)
        for a in valid_actions:
            if a < self.action_dim:
                mask[a] = 0
        q1_masked = q1_values + mask
        q2_masked = q2_values + mask

        # Step 1: Find max Q1 among valid actions
        max_q1 = q1_masked.max()

        # Step 2: Find actions within tolerance of max Q1
        threshold = max_q1 - abs(max_q1) * self.tolerance - 1e-8
        near_optimal = [a for a in valid_actions
                        if a < self.action_dim and q1_masked[a] >= threshold]

        if not near_optimal:
            near_optimal = valid_actions

        # Step 3: Among near-optimal for Q1, pick best Q2
        best_action = max(near_optimal, key=lambda a: q2_masked[a] if a < self.action_dim else -1e9)
        return best_action

    def select_action(self, state, valid_actions, training=True):
        """Select action with epsilon-greedy + Noisy Network exploration."""
        # Reset noise for exploration (Noisy Networks provide state-dependent noise)
        if training:
            self.q1_network.reset_noise()
            self.q2_network.reset_noise()
        # Epsilon-greedy as safety fallback (low epsilon since NoisyNets handle exploration)
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        if self.stochastic_eval and not training:
            return self._stochastic_lexicographic_action(state, valid_actions)
        return self.get_lexicographic_actions(state, valid_actions)

    def _stochastic_lexicographic_action(self, state, valid_actions):
        """Boltzmann sampling from Q2 among Q1-filtered near-optimal actions.

        Used for stochastic evaluation to generate mean +/- SD results.
        """
        import torch.nn.functional as F
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q1 = self.q1_network(state_tensor).squeeze(0)
            q2 = self.q2_network(state_tensor).squeeze(0)

        # Step 1: Q1 filtering (same as deterministic)
        q1_masked = torch.full_like(q1, -1e9)
        for a in valid_actions:
            if a < self.action_dim:
                q1_masked[a] = q1[a]
        max_q1 = q1_masked.max()
        threshold = max_q1 - abs(max_q1) * self.tolerance - 1e-8
        near_optimal = [a for a in valid_actions
                        if a < self.action_dim and q1_masked[a] >= threshold]
        if not near_optimal:
            near_optimal = valid_actions

        # Step 2: Boltzmann sampling from Q2 values over near-optimal set
        q2_vals = torch.tensor([q2[a].item() if a < self.action_dim else -1e9
                                for a in near_optimal])
        # Temperature=1.0 for moderate stochasticity
        probs = F.softmax(q2_vals, dim=0)
        idx = torch.multinomial(probs, 1).item()
        return near_optimal[idx]

    def select_action_single_agent(self, state, valid_actions, training=False):
        """Use only Q1 network (completion), no lexicographic filtering.

        Ablation mode: single-agent RL — uses Agent 1 (completion) only.
        No Q2 network involvement, no lexicographic coordination.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q1_values = self.q1_network(state_tensor).cpu().numpy()[0]

        # Mask invalid actions
        mask = np.full(self.action_dim, -1e9)
        for a in valid_actions:
            if a < self.action_dim:
                mask[a] = 0
        q1_masked = q1_values + mask

        if self.stochastic_eval and not training:
            # Boltzmann sampling from Q1 values
            import torch.nn.functional as F
            q1_vals = torch.tensor([q1_masked[a] if a < self.action_dim else -1e9
                                    for a in valid_actions])
            probs = F.softmax(q1_vals, dim=0)
            idx = torch.multinomial(probs, 1).item()
            return valid_actions[idx]

        # Greedy: pick action with highest Q1
        return valid_actions[int(np.argmax([q1_masked[a] if a < self.action_dim else -1e9
                                            for a in valid_actions]))]

    def train_step(self, batch_size=512, use_prioritization=True):
        """Perform one training step on both Q-networks."""
        if len(self.memory) < batch_size:
            return 0.0, 0.0

        batch = self.memory.sample(batch_size)
        if batch is None:
            return 0.0, 0.0

        states, actions, r1s, r2s, next_states, dones, weights, indices = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        r1s = r1s.to(self.device)
        r2s = r2s.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Reset noise for training forward passes
        self.q1_network.reset_noise()
        self.q1_target.reset_noise()
        self.q2_network.reset_noise()
        self.q2_target.reset_noise()

        # Train Q1 (Double DQN: online selects action, target evaluates)
        q1_current = self.q1_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q1_next_online = self.q1_network(next_states)
            q1_best_actions = q1_next_online.argmax(1)
            q1_next = self.q1_target(next_states).gather(1, q1_best_actions.unsqueeze(1)).squeeze(1)
            q1_target_vals = r1s + self.gamma * q1_next * (1 - dones)

        td_errors_q1 = (q1_current - q1_target_vals).abs().detach().cpu().numpy()
        loss_q1 = (weights * F.smooth_l1_loss(q1_current, q1_target_vals, reduction='none')).mean()

        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.q1_network.parameters(), 1.0)
        self.q1_optimizer.step()

        # Train Q2 (Double DQN: online selects action, target evaluates)
        q2_current = self.q2_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2_next_online = self.q2_network(next_states)
            q2_best_actions = q2_next_online.argmax(1)
            q2_next = self.q2_target(next_states).gather(1, q2_best_actions.unsqueeze(1)).squeeze(1)
            q2_target_vals = r2s + self.gamma2 * q2_next * (1 - dones)

        loss_q2 = (weights * F.smooth_l1_loss(q2_current, q2_target_vals, reduction='none')).mean()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.q2_network.parameters(), 1.0)
        self.q2_optimizer.step()

        # Update priorities
        if use_prioritization:
            self.memory.update_priorities(indices, td_errors_q1)

        # Update target networks
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.q1_target.load_state_dict(self.q1_network.state_dict())
            self.q2_target.load_state_dict(self.q2_network.state_dict())

        return loss_q1.item(), loss_q2.item()

    def step_schedulers(self):
        """Step LR schedulers once per episode (NOT per training step)."""
        self.q1_scheduler.step()
        self.q2_scheduler.step()

    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_tolerance(self):
        """Decay lexicographic tolerance — decays from the start.

        With tolerance_start=0.05, Q1 (completion) strongly dominates.
        Gradual decay ensures increasingly strict lexicographic ordering.
        """
        self.tolerance = max(self.tolerance_end, self.tolerance * self.tolerance_decay)

    def save(self, path):
        """Save agent state to file."""
        torch.save({
            'q1_network': self.q1_network.state_dict(),
            'q2_network': self.q2_network.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'epsilon': self.epsilon,
            'tolerance': self.tolerance,
            'phase2_active': self.phase2_active,
        }, path)
        print(f"Agent saved to {path}")

    def load(self, path):
        """Load agent state from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q1_network.load_state_dict(checkpoint['q1_network'])
        self.q2_network.load_state_dict(checkpoint['q2_network'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.tolerance = checkpoint.get('tolerance', 0.001)
        self.phase2_active = checkpoint.get('phase2_active', True)
        print(f"Agent loaded from {path}")


# =============================================================================
# SECTION 8b: MULTI-AGENT LHAC-PPO — Two Independent PPO Agents with
#             Lexicographic Coordination
#
# Architecture (novel multi-agent contribution for LHAC paper):
#   Agent 1 (Completion Agent): Independent PPO network (encoder + π₁ + V₁)
#     - Trained purely on R1 (completion rewards)
#     - Learns to maximize server assignment completion rate
#
#   Agent 2 (Tardiness Agent): Independent PPO network (encoder + π₂ + V₂)
#     - Trained purely on R2 (tardiness rewards)
#     - Learns to minimize tardiness of assigned servers
#
#   Lexicographic Coordinator:
#     - Agent 1 proposes near-optimal actions: A₁(s) = {a | π₁(a|s) ≥ max-τ}
#     - Agent 2 selects best action from A₁(s) according to π₂
#     - τ decays over training → increasingly strict lexicographic ordering
#
# Key novelty vs LPPO (Zhang et al., IEEE TVT 2023):
#   - LPPO: single agent with shared encoder + dual heads (single-agent)
#   - LHAC: two fully independent agents with lexicographic coordination
#           (multi-agent) — each has own encoder, own optimizer, own trajectory
#   - Applied to discrete combinatorial scheduling (vs continuous driving)
#   - Server retry mechanism + curriculum learning
# =============================================================================


class SingleObjectivePPONetwork(nn.Module):
    """Independent PPO network for a single objective.

    Each agent in the multi-agent LHAC system has its own complete network:
      Encoder: state_dim → 512 (LayerNorm+ReLU) → 256 (LayerNorm+ReLU)
      Policy:  256 → 128 (ReLU) → action_dim (logits)
      Value:   256 → 128 (ReLU) → 1

    Having independent encoders (rather than shared) allows each agent to
    learn objective-specific state representations:
      - Agent 1 focuses on capacity/availability features for completion
      - Agent 2 focuses on slack/timing features for tardiness
    """

    def __init__(self, state_dim, action_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Independent encoder (NOT shared between agents)
        encoder_layers = []
        prev_dim = state_dim
        for hd in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hd))
            encoder_layers.append(nn.LayerNorm(hd))
            encoder_layers.append(nn.ReLU())
            prev_dim = hd
        self.encoder = nn.Sequential(*encoder_layers)

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Smaller init for policy output (more uniform initial policy)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        # Smaller init for value output
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)

    def forward(self, x, valid_mask=None):
        """Forward pass returning policy logits and state value.

        Args:
            x: state tensor [batch, state_dim]
            valid_mask: binary mask [batch, action_dim] (1=valid, 0=invalid)

        Returns:
            logits: policy logits [batch, action_dim] (masked if valid_mask given)
            value: state value [batch]
        """
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        features = self.encoder(x)
        logits = self.policy(features)
        value = self.value(features).squeeze(-1)

        if valid_mask is not None:
            logits = logits + (valid_mask - 1) * 1e9

        return logits, value

    def get_dist(self, x, valid_mask):
        """Get policy distribution over valid actions."""
        logits, value = self.forward(x, valid_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value


# =========================================================================
# CASE: Cross-Attention Cell-Server Encoder (Innovation 2 of DAF-LHAC)
# =========================================================================

class CrossAttentionPPONetwork(nn.Module):
    """Cross-Attention Cell-Server Encoder for per-cell action scoring.

    Instead of a flat MLP that treats the state as a 74/78-dim vector,
    CASE structures the observation into three components:
      - Server query (16 dims): current server features
      - Cell keys (54 x 12 dims each): per-cell features
      - Global context (20 dims): aggregate facility state

    The server query attends to cell keys via multi-head cross-attention,
    producing per-cell compatibility scores. Combined with global context,
    this generates per-cell action logits directly from pairwise features.

    Architecture:
      Server Encoder:  Linear(16, 128) -> LayerNorm -> ReLU -> Linear(128, 128)
      Cell Encoder:    Linear(12, 128) -> LayerNorm -> ReLU -> Linear(128, 128)  [shared]
      Global Encoder:  Linear(20, 128) -> LayerNorm -> ReLU -> Linear(128, 128)
      Cross-Attention: MultiheadAttention(embed=128, heads=4, batch_first=True)
                       + residual connection + LayerNorm
      Fusion:          Linear(256, 256) -> LayerNorm -> ReLU  [attended_server || global]
      Policy:          Bilinear(cell_embed=128, fused=256) -> per-cell logits
                       + Linear(256, 1) -> skip logit
      Value:           Linear(256, 128) -> ReLU -> Linear(128, 1)

    Parameters: ~400K (vs ~550K for flat MLP)
    """

    # Feature dimensions
    SERVER_DIM = 16
    CELL_DIM = 12
    GLOBAL_DIM = 20
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_CELLS = 54  # Total FUL cells (4 banks)

    def __init__(self, action_dim, embed_dim=128, num_heads=4, num_cells=54):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_cells = num_cells

        # Server encoder: 16 -> embed_dim
        self.server_encoder = nn.Sequential(
            nn.Linear(self.SERVER_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Cell encoder: 12 -> embed_dim (shared across all cells)
        self.cell_encoder = nn.Sequential(
            nn.Linear(self.CELL_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Global encoder: 20 -> embed_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(self.GLOBAL_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Cross-attention: server query attends to cell keys
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Fusion: [attended_server || global] -> 256
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
        )

        # Policy head: bilinear per-cell scoring
        # Score for cell i = cell_embed[i]^T W fused + bias
        self.policy_bilinear = nn.Bilinear(embed_dim, embed_dim * 2, 1)
        # Skip action logit from fused context
        self.policy_skip = nn.Linear(embed_dim * 2, 1)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Smaller init for policy outputs
        nn.init.orthogonal_(self.policy_bilinear.weight, gain=0.01)
        nn.init.orthogonal_(self.policy_skip.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, server_feat, cell_feat, global_feat, valid_mask=None):
        """Forward pass with structured observation.

        Args:
            server_feat: [batch, 16] current server features
            cell_feat:   [batch, 54, 12] per-cell features
            global_feat: [batch, 20] global context features
            valid_mask:  [batch, action_dim] (1=valid, 0=invalid)

        Returns:
            logits: [batch, action_dim] policy logits
            value:  [batch] state value
        """
        batch_size = server_feat.shape[0]

        # Encode
        server_embed = self.server_encoder(server_feat)  # [B, 128]
        cell_embed = self.cell_encoder(cell_feat)         # [B, 54, 128]
        global_embed = self.global_encoder(global_feat)   # [B, 128]

        # Cross-attention: server query attends to cell keys
        # query: [B, 1, 128], key/value: [B, 54, 128]
        query = server_embed.unsqueeze(1)  # [B, 1, 128]
        attended, attn_weights = self.cross_attention(
            query, cell_embed, cell_embed)  # [B, 1, 128]

        # Residual + LayerNorm
        attended = self.attn_norm(attended.squeeze(1) + server_embed)  # [B, 128]

        # Fusion: concat attended server + global context
        fused = self.fusion(torch.cat([attended, global_embed], dim=-1))  # [B, 256]

        # Policy: per-cell logits via bilinear scoring
        # cell_embed: [B, 54, 128], fused: [B, 256]
        # Expand fused for bilinear: [B*54, 256]
        fused_expanded = fused.unsqueeze(1).expand(-1, self.num_cells, -1)
        fused_flat = fused_expanded.reshape(-1, fused.shape[-1])
        cell_flat = cell_embed.reshape(-1, self.embed_dim)

        cell_logits = self.policy_bilinear(cell_flat, fused_flat)  # [B*54, 1]
        cell_logits = cell_logits.reshape(batch_size, self.num_cells)  # [B, 54]

        # Skip action logit
        skip_logit = self.policy_skip(fused)  # [B, 1]

        # Combine: [cell_logits, skip_logit]
        logits = torch.cat([cell_logits, skip_logit], dim=-1)  # [B, 55]

        # Numerical stability: clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        # Apply valid action mask
        if valid_mask is not None:
            # Use -1e8 instead of -1e9 for numerical stability
            logits = logits + (valid_mask - 1) * 1e8

        # Safety: replace any NaN with 0
        if torch.isnan(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-1e8)

        # Value
        value = self.value_head(fused).squeeze(-1)  # [B]
        value = torch.nan_to_num(value, nan=0.0)

        return logits, value

    def forward_flat(self, x, valid_mask=None):
        """Forward pass that accepts flat state and splits into structured.

        This allows CASE to work with the existing flat state vector (74/78 dims)
        by extracting the relevant features and structuring them.

        Args:
            x: flat state tensor [batch, state_dim] (74 or 78 dims)
            valid_mask: [batch, action_dim]

        Returns:
            logits, value (same as forward)
        """
        # NaN safety: replace any NaN in input
        x = torch.nan_to_num(x, nan=0.0)
        server_feat, cell_feat, global_feat = self._split_flat_state(x)
        return self.forward(server_feat, cell_feat, global_feat, valid_mask)

    def _split_flat_state(self, x):
        """Split flat state into structured components.

        Maps the flat 74/78-dim state to:
          - server_feat [B, 16]: current server properties
          - cell_feat [B, 54, 12]: per-cell features (approximated from aggregate)
          - global_feat [B, 20]: global facility state

        Since the flat state has aggregate cell info (not per-cell), we construct
        approximate per-cell features from available data. The cross-attention
        will learn to extract useful cell-server compatibility from these.
        """
        batch_size = x.shape[0]

        # Server features [16 dims]:
        # [0] server_arrival_norm  = x[0]   (arrival / total_time)
        # [1] server_pt_norm       = x[1]   (PTk / total_time)
        # [2] server_due_norm      = x[2]   (DueTime / total_time)
        # [3] server_is_2tc        = x[57]  (OTk == 1)
        # [4] server_slack_norm    = x[56]  (slack / 20)
        # [5] server_needs_hv      = x[3]   (RH)
        # [6] server_needs_lv      = x[4]   (RL)
        # [7] server_needs_water   = x[5]   (RW)
        # [8] server_compat_ratio  = x[62]  (compatible cells fraction)
        # [9] server_priority_norm = x[6]   (priority / max_priority)
        # [10] server_remaining_frac = x[8] (remaining / total)
        # [11] server_progress     = x[10]  (assigned / total)
        # [12] server_time_frac    = x[11]  (current time / total)
        # [13] server_retry_level  = x[55]  (retry pass indicator)
        # [14] server_2tc_pressure = x[63]  (2tc demand / available pairs)
        # [15] server_time_pressure = x[73] (remaining / remaining_time)
        server_indices = [0, 1, 2, 57, 56, 3, 4, 5, 62, 6, 8, 10, 11, 55, 63, 73]
        server_feat = torch.stack([x[:, i] for i in server_indices], dim=-1)

        # Global features [20 dims]:
        # [0-3]  bank_free_fraction = x[64:68]
        # [4]    total_free_cell_frac = x[15]
        # [5]    adj_pairs_free = x[54]
        # [6]    capacity_ratio = x[52]
        # [7]    completion_progress = x[10]
        # [8-11] temporal_profile = x[58:62]  (next 4 windows occupancy)
        # [12]   tardiness_frac = x[72]
        # [13]   time_pressure = x[73]
        # [14-17] slack_distribution = x[68:72]  (min, 25th, 50th, 75th)
        # [18]   2tc_demand_remaining = x[53]
        # [19]   window_progress = x[9]
        global_indices = [64, 65, 66, 67, 15, 54, 52, 10, 58, 59, 60, 61,
                          72, 73, 68, 69, 70, 71, 53, 9]
        # Safely index (some might be out of bounds if state < 74)
        global_feat = torch.zeros(batch_size, 20, device=x.device)
        for i, gi in enumerate(global_indices):
            if gi < x.shape[1]:
                global_feat[:, i] = x[:, gi]

        # Cell features [54, 12 dims]:
        # Since flat state only has aggregate cell info, we construct
        # approximate per-cell features. The network learns cell-specific
        # patterns through training.
        # For each cell i:
        # [0-2] Voltage one-hot: derived from x[16:52] (per-bank cell status)
        # [3]   Bank position norm: i / num_cells
        # [4]   Water cooling flag: approximated from bank info
        # [5]   Has adjacent pair: known from facility config
        # [6]   Free in server window: approximated from aggregate
        # [7]   EFS delay norm: from x[76] if available (DAP)
        # [8]   Valid for server: approximated from x[62] (compat ratio)
        # [9]   Tardiness impact: 0 (no per-cell info in flat state)
        # [10]  Future occupancy: from temporal profile
        # [11]  Structural compat: from compat ratio
        #
        # NOTE: When using structured state (use_structured_state=True),
        # the environment will provide real per-cell features instead.
        cell_feat = torch.zeros(batch_size, self.num_cells, 12, device=x.device)

        # Fill with position encoding + aggregate approximation
        for ci in range(self.num_cells):
            cell_feat[:, ci, 3] = ci / self.num_cells  # Position encoding
            # Bank indicator (roughly: cells 0-13 = bank1, etc)
            bank_idx = ci // 14
            if bank_idx < 4 and (64 + bank_idx) < x.shape[1]:
                cell_feat[:, ci, 6] = x[:, 64 + bank_idx]  # Bank free fraction
            cell_feat[:, ci, 8] = x[:, 62] if 62 < x.shape[1] else 0  # Compat ratio
            # Temporal (average over windows)
            if 58 < x.shape[1]:
                cell_feat[:, ci, 10] = (x[:, 58] + x[:, 59]) / 2 if 59 < x.shape[1] else x[:, 58]

        return server_feat, cell_feat, global_feat

    def get_dist(self, x, valid_mask):
        """Get policy distribution — accepts flat state for compatibility."""
        logits, value = self.forward_flat(x, valid_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value


class MultiAgentLHACPPO:
    """Multi-Agent LHAC with PPO backbone.

    Novel multi-agent lexicographic architecture:
      - Agent 1 (Completion): Fully independent PPO, trained on R1
      - Agent 2 (Tardiness):  Fully independent PPO, trained on R2
      - Lexicographic Coordinator: Agent 1 filters → Agent 2 selects

    Key differences from single-agent LPPO:
      1. Two INDEPENDENT networks (separate encoders, parameters, optimizers)
      2. Each agent trained ONLY on its own objective reward
      3. Coordination only at action selection time (not in loss function)
      4. Each agent can learn objective-specific state representations
      5. Independent training allows different learning rates and schedules

    This is a genuinely multi-agent system where:
      - Agent 1 acts as a "gatekeeper" — filters actions for completion safety
      - Agent 2 acts as a "refinement" — picks best tardy-minimizing action
        from Agent 1's approved set
    """

    def __init__(self, state_dim, action_dim, lr1=3e-4, lr2=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 entropy_coef=0.02, value_coef=0.5, max_grad_norm=0.5,
                 tau=0.05, tau_end=0.005, tau_decay=0.9999,
                 epochs_per_update=4, mini_batch_size=256,
                 device=None, use_case=False):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.use_case = use_case  # Innovation 2: Cross-Attention Cell-Server Encoder

        # τ: lexicographic tolerance (dynamic threshold for action filtering)
        self.tolerance = tau
        self.tolerance_start = tau
        self.tolerance_end = tau_end
        self.tolerance_decay = tau_decay

        self.epochs_per_update = epochs_per_update
        self.mini_batch_size = mini_batch_size
        self.phase2_active = False
        self.stochastic_eval = False  # Set True for stochastic evaluation (mean +/- SD)
        self.use_adaptive_tau = True  # Set False for ablation (uses fixed self.tolerance)

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # ===== AGENT 1: Completion Agent (independent PPO) =====
        if self.use_case:
            self.agent1_network = CrossAttentionPPONetwork(
                action_dim).to(self.device)
        else:
            self.agent1_network = SingleObjectivePPONetwork(
                state_dim, action_dim).to(self.device)
        self.agent1_optimizer = optim.Adam(
            self.agent1_network.parameters(), lr=lr1, eps=1e-5)
        self.agent1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.agent1_optimizer, T_max=30000, eta_min=1e-6)

        # ===== AGENT 2: Tardiness Agent (independent PPO) =====
        if self.use_case:
            self.agent2_network = CrossAttentionPPONetwork(
                action_dim).to(self.device)
        else:
            self.agent2_network = SingleObjectivePPONetwork(
                state_dim, action_dim).to(self.device)
        self.agent2_optimizer = optim.Adam(
            self.agent2_network.parameters(), lr=lr2, eps=1e-5)
        self.agent2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.agent2_optimizer, T_max=30000, eta_min=1e-6)

        # On-policy trajectory buffer (shared — both agents see same trajectory)
        self.trajectory = []

        # Small epsilon for safety exploration
        self.epsilon = 0.05
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9998

    def _build_valid_mask(self, valid_actions, batch=False):
        """Build valid action mask tensor."""
        if batch:
            # valid_actions is a list of lists
            masks = torch.zeros(len(valid_actions), self.action_dim,
                                device=self.device)
            for i, va in enumerate(valid_actions):
                for a in va:
                    if a < self.action_dim:
                        masks[i, a] = 1.0
            return masks
        else:
            mask = torch.zeros(1, self.action_dim, device=self.device)
            for a in valid_actions:
                if a < self.action_dim:
                    mask[0, a] = 1.0
            return mask

    def _compute_adaptive_tau(self, state):
        """Compute congestion-aware adaptive τ from state features.

        Novel contribution: Instead of a fixed or monotonically-decaying τ,
        τ adapts to the current system congestion level. When the facility
        is congested (high occupancy, tight deadlines), Agent 1 (completion)
        gets stricter control (smaller τ → fewer actions for Agent 2).
        When capacity is abundant, Agent 2 (tardiness) gets more freedom
        (larger τ → more actions to optimize tardiness).

        Congestion indicators extracted from state:
          - state[10]: completion_progress (assigned/total servers)
          - state[11]: current_time_frac (time elapsed)
          - state[15]: total free cell fraction
          - state[52]: capacity_ratio (free cells in horizon / total cells)
          - state[55]: retry pass indicator

        Formula:
          congestion = (1 - free_cell_frac) * (1 - capacity_ratio)
          adaptive_tau = tau_base * (1 - congestion_weight * congestion)

        High congestion → tau shrinks → Agent 1 dominates (protect completion)
        Low congestion → tau stays high → Agent 2 can optimize tardiness
        """
        # Extract congestion features from state vector
        free_cell_frac = max(0.0, min(1.0, float(state[15])))
        capacity_ratio = max(0.0, min(1.0, float(state[52])))
        completion_progress = max(0.0, min(1.0, float(state[10])))
        retry_level = float(state[55])  # 0.0=pass 0, 0.5=retry 1, 1.0=retry 2
        is_retry_pass = retry_level > 0.1  # True for any retry pass

        # Congestion: high when cells are occupied and capacity is tight
        congestion = (1.0 - free_cell_frac) * (1.0 - capacity_ratio)

        # During retry pass, be strict (protect completion of remaining servers)
        if is_retry_pass:
            # Stricter on later retries
            min_congestion = 0.7 if retry_level < 0.8 else 0.85
            congestion = max(congestion, min_congestion)

        # Late in episode with low completion → very strict
        if completion_progress < 0.5 and float(state[11]) > 0.3:
            congestion = max(congestion, 0.6)

        # Adaptive τ: base tolerance scaled by congestion
        # congestion=0 → tau = tau_base (full freedom for Agent 2)
        # congestion=1 → tau = tau_base * 0.1 (Agent 1 dominates)
        congestion_weight = 0.9  # How much congestion reduces τ
        adaptive_tau = self.tolerance * (1.0 - congestion_weight * congestion)

        # Floor: never let τ go below a minimum
        return max(adaptive_tau, 0.001)

    def select_action(self, state, valid_actions, training=True):
        """Multi-agent lexicographic action selection with adaptive τ.

        Algorithm:
          1. Compute congestion-aware τ from current state
          2. Agent 1 (completion) computes π₁(a|s) over valid actions
          3. Filter: A₁(s) = {a ∈ valid | π₁(a|s) ≥ max π₁ - τ_adaptive}
          4. Agent 2 (tardiness) computes π₂(a|s) over A₁(s) only
          5. Sample/argmax from Agent 2's filtered distribution

        This is multi-agent because Agent 1 and Agent 2 independently
        evaluate the state with their own encoders and policies.
        τ_adaptive adjusts based on facility congestion (novel contribution).
        """
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Compute τ: adaptive (congestion-aware) or fixed (ablation)
        if self.use_adaptive_tau:
            current_tau = self._compute_adaptive_tau(state)
        else:
            current_tau = self.tolerance  # Fixed tau for ablation

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        valid_mask = self._build_valid_mask(valid_actions)

        with torch.no_grad():
            self.agent1_network.eval()
            self.agent2_network.eval()

            # Agent 1: compute completion-focused policy
            dist1, _ = self.agent1_network.get_dist(state_tensor, valid_mask)
            probs1 = dist1.probs  # [1, action_dim]

            # Lexicographic filtering with adaptive τ
            max_prob1 = probs1.max(dim=-1, keepdim=True)[0]
            threshold = max_prob1 - current_tau
            filtered_mask = (probs1 >= threshold).float() * valid_mask

            # Safety: if no action survives filtering, fall back to all valid
            if filtered_mask.sum() == 0:
                filtered_mask = valid_mask

            # Agent 2: select from Agent 1's filtered set
            dist2, _ = self.agent2_network.get_dist(state_tensor, filtered_mask)

            if training or self.stochastic_eval:
                action = dist2.sample().item()
            else:
                action = dist2.probs.argmax(dim=-1).item()

            self.agent1_network.train()
            self.agent2_network.train()

        return action

    def select_action_single_agent(self, state, valid_actions, training=False):
        """Use only Agent 1 (completion), no lexicographic filtering.

        Ablation mode: single-agent RL — uses completion agent only.
        No Agent 2 involvement, no lexicographic coordination, no adaptive tau.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        valid_mask = self._build_valid_mask(valid_actions)

        with torch.no_grad():
            self.agent1_network.eval()

            # Agent 1 only: compute completion-focused policy
            dist1, _ = self.agent1_network.get_dist(state_tensor, valid_mask)

            if training or self.stochastic_eval:
                action = dist1.sample().item()
            else:
                action = dist1.probs.argmax(dim=-1).item()

            self.agent1_network.train()

        return action

    def store_transition(self, state, action, r1, r2, done, valid_actions):
        """Store transition — both agents share the same trajectory."""
        self.trajectory.append({
            'state': state,
            'action': action,
            'r1': r1,
            'r2': r2,
            'done': done,
            'valid_actions': valid_actions,
        })

    @staticmethod
    def compute_gae(rewards, values, dones, gamma, lam):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        T = len(rewards)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        returns = advantages + values
        return advantages, returns

    def _update_single_agent(self, network, optimizer, states_t, actions_t,
                             returns_t, advantages_t, valid_masks,
                             old_log_probs):
        """PPO update for a single agent (Agent 1 or Agent 2).

        Each agent is trained INDEPENDENTLY on its own objective:
          - Agent 1 uses R1 advantages (completion)
          - Agent 2 uses R2 advantages (tardiness)

        This is the key difference from LPPO: no shared loss, no coupled
        gradients, no single optimizer. Each agent optimizes its own objective.
        """
        T = len(states_t)
        total_loss = 0.0
        n_updates = 0

        for epoch in range(self.epochs_per_update):
            indices = np.arange(T)
            np.random.shuffle(indices)

            for start in range(0, T, self.mini_batch_size):
                end = min(start + self.mini_batch_size, T)
                idx = indices[start:end]

                mb_states = states_t[idx]
                mb_actions = actions_t[idx]
                mb_returns = returns_t[idx]
                mb_adv = advantages_t[idx]
                mb_old_lp = old_log_probs[idx]
                mb_masks = valid_masks[idx]

                # Forward pass through THIS agent's network
                dist, values = network.get_dist(mb_states, mb_masks)
                new_lp = dist.log_prob(mb_actions)
                # Safety: clamp log_probs to prevent explosion
                new_lp = torch.clamp(new_lp, min=-20.0, max=0.0)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate loss
                log_ratio = new_lp - mb_old_lp
                log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
                ratio = torch.exp(log_ratio)
                # NaN safety
                ratio = torch.nan_to_num(ratio, nan=1.0)
                surr_a = ratio * mb_adv
                surr_b = torch.clamp(ratio, 1 - self.clip_eps,
                                     1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr_a, surr_b).mean()

                # Value loss
                value_loss = F.mse_loss(values, mb_returns)

                # Total loss for THIS agent
                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), self.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / max(n_updates, 1)

    def update(self):
        """Multi-agent PPO update: train Agent 1 and Agent 2 independently.

        Each agent does its own PPO update with its own rewards:
          - Agent 1: PPO update with R1 (completion) rewards and advantages
          - Agent 2: PPO update with R2 (tardiness) rewards and advantages

        The agents are COMPLETELY INDEPENDENT during training. Lexicographic
        coordination only happens at action selection time.

        Includes reward normalization (running stats) to prevent loss explosion
        when episode lengths or reward magnitudes vary across curriculum stages.
        """
        if len(self.trajectory) < 16:
            self.trajectory = []
            return 0.0, 0.0

        # Extract trajectory data
        states = np.array([t['state'] for t in self.trajectory])
        actions = np.array([t['action'] for t in self.trajectory])
        r1s = np.array([t['r1'] for t in self.trajectory], dtype=np.float32)
        r2s = np.array([t['r2'] for t in self.trajectory], dtype=np.float32)
        dones = np.array([t['done'] for t in self.trajectory], dtype=np.float32)
        valid_actions_list = [t['valid_actions'] for t in self.trajectory]

        # === Reward normalization (prevents loss explosion) ===
        # Scale rewards to have roughly unit variance. This is critical for PPO
        # because reward magnitudes vary wildly (10-28 per step, 100 terminal)
        # and episode lengths vary 100-400 across curriculum stages.
        r1_std = max(r1s.std(), 1.0)
        r2_std = max(r2s.std(), 1.0)
        r1s_norm = r1s / r1_std
        r2s_norm = r2s / r2_std

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        valid_masks = self._build_valid_mask(valid_actions_list, batch=True)

        # ===== AGENT 1: Compute values, GAE, old log probs =====
        with torch.no_grad():
            _, values1 = self.agent1_network.get_dist(states_t, valid_masks)
            values1_np = values1.cpu().numpy()
            dist1, _ = self.agent1_network.get_dist(states_t, valid_masks)
            old_lp1 = dist1.log_prob(actions_t)

        adv1, ret1 = self.compute_gae(r1s_norm, values1_np, dones,
                                       self.gamma, self.gae_lambda)
        if len(adv1) > 1:
            adv1 = (adv1 - adv1.mean()) / (adv1.std() + 1e-8)

        ret1_t = torch.FloatTensor(ret1).to(self.device)
        adv1_t = torch.FloatTensor(adv1).to(self.device)

        # ===== AGENT 2: Compute values, GAE, old log probs =====
        with torch.no_grad():
            _, values2 = self.agent2_network.get_dist(states_t, valid_masks)
            values2_np = values2.cpu().numpy()
            dist2, _ = self.agent2_network.get_dist(states_t, valid_masks)
            old_lp2 = dist2.log_prob(actions_t)

        adv2, ret2 = self.compute_gae(r2s_norm, values2_np, dones,
                                       self.gamma, self.gae_lambda)
        if len(adv2) > 1:
            adv2 = (adv2 - adv2.mean()) / (adv2.std() + 1e-8)

        ret2_t = torch.FloatTensor(ret2).to(self.device)
        adv2_t = torch.FloatTensor(adv2).to(self.device)

        # ===== Train Agent 1 independently on R1 =====
        loss1 = self._update_single_agent(
            self.agent1_network, self.agent1_optimizer,
            states_t, actions_t, ret1_t, adv1_t, valid_masks, old_lp1)

        # ===== Train Agent 2 independently on R2 =====
        loss2 = self._update_single_agent(
            self.agent2_network, self.agent2_optimizer,
            states_t, actions_t, ret2_t, adv2_t, valid_masks, old_lp2)

        # Clear trajectory
        self.trajectory = []
        return loss1, loss2

    def activate_phase2(self):
        """Activate Phase 2: Agent 2 gets boosted learning rate."""
        self.phase2_active = True
        # Boost Agent 2's learning rate when Phase 2 activates
        for param_group in self.agent2_optimizer.param_groups:
            param_group['lr'] = min(param_group['lr'] * 2, 1e-3)

    def step_schedulers(self):
        """Step LR schedulers for both agents (once per episode)."""
        self.agent1_scheduler.step()
        self.agent2_scheduler.step()

    def update_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_tolerance(self):
        """Decay lexicographic tolerance τ (tighter filtering over time)."""
        self.tolerance = max(self.tolerance_end,
                             self.tolerance * self.tolerance_decay)

    def save(self, path):
        """Save both agents and coordinator state."""
        torch.save({
            'agent1_network': self.agent1_network.state_dict(),
            'agent1_optimizer': self.agent1_optimizer.state_dict(),
            'agent2_network': self.agent2_network.state_dict(),
            'agent2_optimizer': self.agent2_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'tolerance': self.tolerance,
            'epsilon': self.epsilon,
            'phase2_active': self.phase2_active,
            'use_case': self.use_case,
            'agent_type': 'multi_agent_ppo',
        }, path)
        print(f"Multi-Agent LHAC saved to {path}")

    def load(self, path):
        """Load both agents and coordinator state."""
        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=False)
        self.agent1_network.load_state_dict(checkpoint['agent1_network'])
        self.agent2_network.load_state_dict(checkpoint['agent2_network'])
        if 'agent1_optimizer' in checkpoint:
            try:
                self.agent1_optimizer.load_state_dict(
                    checkpoint['agent1_optimizer'])
            except Exception:
                pass
        if 'agent2_optimizer' in checkpoint:
            try:
                self.agent2_optimizer.load_state_dict(
                    checkpoint['agent2_optimizer'])
            except Exception:
                pass
        self.tolerance = checkpoint.get('tolerance', 0.05)
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.phase2_active = checkpoint.get('phase2_active', False)
        print(f"Multi-Agent LHAC loaded from {path}")


# =============================================================================
# SECTION 9: LHAC INFERENCE — RL-based window solver
# =============================================================================

# Cache for loaded agent to avoid re-loading every window
_cached_agent = None
_cached_model_path = None


def LHAC_solve_window(K_f, T_f, I, J, VH, VL, W, RH_f, RL_f, RW_f,
                       PTk_f, OTk_f, atk_f, arrival_time, DueTime,
                       ki_prev, kj_prev, order_with_remaining_time,
                       remaining_time_feed, start_limit, end_limit,
                       cost_list, facility,
                       model_path=None):
    """Solve one window using trained LHAC agent (DQN or Multi-Agent PPO).

    Auto-detects model type from checkpoint and loads appropriate agent.

    Returns:
        rl_assignments: dict {server_id: [bank, cell, start_time, end_time]}
        rl_time: float, seconds taken
    """
    global _cached_agent, _cached_model_path

    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'lhac_model.pth')

    start_t = tm.time()

    # Build DataFrame for the RL environment
    rows = []
    for idx in range(len(K_f)):
        k = K_f[idx]
        rows.append({
            'k': k,
            'type': 'unknown',
            'PTk': PTk_f[idx],
            'OTk': OTk_f[idx],
            'RH': RH_f[idx],
            'RL': RL_f[idx],
            'RW': RW_f[idx],
            'DueTime': DueTime[k - 1],
            'ArrivalTime': max(arrival_time[k - 1], start_limit),
        })
    df = pd.DataFrame(rows)

    if len(df) == 0:
        return {}, tm.time() - start_t

    # Load or create agent (auto-detect DQN vs Multi-Agent PPO)
    agent = None
    if os.path.exists(model_path):
        if _cached_agent is not None and _cached_model_path == model_path:
            agent = _cached_agent
        else:
            # Determine dimensions from a dummy env
            dummy_env = LexicographicRewardEnvironment(facility, df,
                                                        window_length=end_limit - start_limit + 1,
                                                        use_windowing=False)
            state_dim = dummy_env.observation_space.shape[0]
            action_dim = dummy_env.action_space.n

            # Auto-detect agent type from checkpoint
            checkpoint = torch.load(model_path, map_location='cpu',
                                    weights_only=False)
            agent_type = checkpoint.get('agent_type', 'dqn')

            if agent_type == 'multi_agent_ppo':
                agent = MultiAgentLHACPPO(
                    state_dim=state_dim, action_dim=action_dim,
                    device='cpu')
            else:
                agent = ImprovedLexicographicDQNAgent(
                    state_dim=state_dim, action_dim=action_dim,
                    device='cpu')
            agent.load(model_path)
            agent.epsilon = 0.0  # Greedy inference

            _cached_agent = agent
            _cached_model_path = model_path
            print(f"LHAC model loaded from {model_path} (type={agent_type})")

    if agent is None:
        # No trained model — use greedy fallback
        print("No LHAC model found, using greedy heuristic fallback")
        rl_assignments = greedy_heuristic_solve(
            K_f, T_f, VH, VL, W, RH_f, RL_f, RW_f,
            PTk_f, OTk_f, arrival_time, DueTime,
            start_limit, end_limit, facility)
        return rl_assignments, tm.time() - start_t

    # Create environment for this window
    env = LexicographicRewardEnvironment(
        facility, df,
        window_length=end_limit - start_limit + 1,
        use_windowing=False)

    # Lock carry-over servers in occupancy grid
    for z in order_with_remaining_time:
        if str(z) in ki_prev and str(z) in kj_prev:
            banks = ki_prev[str(z)]
            cells = kj_prev[str(z)]
            rem = remaining_time_feed[z - 1]
            block_len = min(rem, len(T_f))
            for bank_val in banks:
                for cell_val in cells:
                    # Find matching cell in environment
                    for cell_idx in range(env.total_cells):
                        c = env.idx_to_cell[cell_idx]
                        if c.bank_id == bank_val and c.position == cell_val:
                            end_block = start_limit + block_len
                            env.cell_occupancy[cell_idx, start_limit:end_block] = z
                            break

    # Run agent greedily
    state, _ = env.reset()
    # Restore occupancy after reset (reset clears it)
    for z in order_with_remaining_time:
        if str(z) in ki_prev and str(z) in kj_prev:
            banks = ki_prev[str(z)]
            cells = kj_prev[str(z)]
            rem = remaining_time_feed[z - 1]
            block_len = min(rem, len(T_f))
            for bank_val in banks:
                for cell_val in cells:
                    for cell_idx in range(env.total_cells):
                        c = env.idx_to_cell[cell_idx]
                        if c.bank_id == bank_val and c.position == cell_val:
                            end_block = start_limit + block_len
                            env.cell_occupancy[cell_idx, start_limit:end_block] = z
                            break

    state = env._get_state()
    done = False
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, training=False)
        state, _, done, _, info = env.step(action)

    # === Greedy repair pass on env's occupancy grid ===
    # After the RL agent has made all decisions (including retries), sweep over
    # remaining unassigned servers and greedily place them in any valid cell.
    # This catches servers the RL agent missed due to suboptimal skip decisions.
    repair_count = 0
    # Build list of (original_index, server_dict) for unassigned servers
    # (env.servers may have retry duplicates, so use original count)
    original_count = len(env.servers) - len(env._retry_queue)
    unassigned_env_servers = []
    for si in range(original_count):
        if si not in env.assigned_servers:
            unassigned_env_servers.append((si, env.servers[si]))

    # Sort: 2TC servers first (they are harder to place), then by slack
    unassigned_env_servers.sort(
        key=lambda pair: (-pair[1]['OTk'], pair[1].get('slack', 999)))

    for orig_idx, srv in unassigned_env_servers:
        arrival = srv['ArrivalTime']
        pt = srv['PTk']
        end_time = arrival + pt
        is_2tc = srv['OTk'] == 1
        server_id = srv['k']

        if end_time > env.total_time + 1:
            continue
        if server_id in env.assignments:
            continue  # Already assigned (safety check)

        # Try each cell greedily
        for cell_idx in range(env.total_cells):
            if not env._is_cell_compatible(cell_idx, srv):
                continue

            # Check time availability for primary cell
            occ_slice = env.cell_occupancy[cell_idx, arrival:min(end_time, env.total_time + 1)]
            if np.any(occ_slice != 0):  # Reject occupied (>0) AND FAB-blocked (-1) cells
                continue

            if is_2tc:
                if cell_idx not in env.adjacent_cells:
                    continue
                adj_idx = env.adjacent_cells[cell_idx]
                adj_occ = env.cell_occupancy[adj_idx, arrival:min(end_time, env.total_time + 1)]
                if np.any(adj_occ != 0):  # Reject occupied (>0) AND FAB-blocked (-1) cells
                    continue
                # Place 2TC server
                env.cell_occupancy[cell_idx, arrival:min(end_time, env.total_time + 1)] = server_id
                env.cell_occupancy[adj_idx, arrival:min(end_time, env.total_time + 1)] = server_id
            else:
                # Place 1TC server
                env.cell_occupancy[cell_idx, arrival:min(end_time, env.total_time + 1)] = server_id

            cell = env.idx_to_cell[cell_idx]
            env.assignments[server_id] = {
                'cell_idx': cell_idx,
                'bank': cell.bank_id,
                'cell': cell.position,
                'start': arrival,
                'end': end_time - 1,
            }
            env.assigned_servers.add(orig_idx)  # Mark as assigned
            repair_count += 1
            break

    if repair_count > 0:
        print(f"  Greedy repair (env): +{repair_count} servers placed directly")

    rl_assignments = env.get_assignments_for_warmstart()

    rl_assigned_count = len(rl_assignments)
    rl_total = len(K_f)

    # === Multi-pass: Greedy recovery on unassigned orders ===
    unassigned_ids = [k for k in K_f if k not in rl_assignments]

    if unassigned_ids and len(unassigned_ids) > 0:
        # Extract cell free times from RL's occupancy grid
        initial_cell_free_times = {}
        for cell_idx in range(env.total_cells):
            c = env.idx_to_cell[cell_idx]
            occ = env.cell_occupancy[cell_idx, :]
            # Find the last occupied time slot
            occupied_mask = occ != 0
            if occupied_mask.any():
                last_occupied = int(np.where(occupied_mask)[0][-1]) + 1
            else:
                last_occupied = 0
            initial_cell_free_times[(c.bank_id, c.position)] = max(last_occupied, start_limit)

        # Build sub-lists for unassigned orders only
        unassigned_indices = [K_f.index(k) for k in unassigned_ids]
        greedy_K = unassigned_ids
        greedy_T = T_f
        greedy_RH = [RH_f[i] for i in unassigned_indices]
        greedy_RL = [RL_f[i] for i in unassigned_indices]
        greedy_RW = [RW_f[i] for i in unassigned_indices]
        greedy_PTk = [PTk_f[i] for i in unassigned_indices]
        greedy_OTk = [OTk_f[i] for i in unassigned_indices]

        greedy_assignments = greedy_heuristic_solve(
            greedy_K, greedy_T, VH, VL, W,
            greedy_RH, greedy_RL, greedy_RW,
            greedy_PTk, greedy_OTk,
            arrival_time, DueTime,
            start_limit, end_limit, facility,
            initial_cell_free_times=initial_cell_free_times)

        # Merge greedy assignments into RL result
        if greedy_assignments:
            rl_assignments.update(greedy_assignments)
            print(f"  Greedy recovery: +{len(greedy_assignments)} orders "
                  f"(total {len(rl_assignments)}/{rl_total})")

    elapsed = tm.time() - start_t

    final_completion = len(rl_assignments) / rl_total if rl_total > 0 else 1.0
    print(f"LHAC: {rl_assigned_count}/{rl_total} RL + "
          f"{len(rl_assignments) - rl_assigned_count} greedy = "
          f"{len(rl_assignments)}/{rl_total} ({final_completion:.1%}), "
          f"tardy={info['tardy_servers']}, time={elapsed:.2f}s")

    return rl_assignments, elapsed


# =============================================================================
# SECTION 10: GREEDY HEURISTIC FALLBACK
# =============================================================================

def greedy_heuristic_solve(K_f, T_f, VH, VL, W, RH_f, RL_f, RW_f,
                           PTk_f, OTk_f, arrival_time, DueTime,
                           start_limit, end_limit, facility,
                           initial_cell_free_times=None):
    """Priority-sorted first-fit greedy heuristic.

    Used as standalone fallback or as recovery pass after RL.
    Args:
        initial_cell_free_times: optional dict {(bank_id, position): free_time}
            from RL occupancy grid. If provided, greedy scheduling starts from
            the RL's occupancy state rather than empty cells.
    Returns dict: {server_id: [bank, cell, start_time, end_time]}
    """
    if len(K_f) == 0:
        return {}

    # Build server info list
    servers = []
    for idx in range(len(K_f)):
        k = K_f[idx]
        slack = DueTime[k - 1] - max(arrival_time[k - 1], start_limit) - PTk_f[idx]
        servers.append({
            'k': k,
            'idx': idx,
            'PTk': PTk_f[idx],
            'OTk': OTk_f[idx],
            'RH': RH_f[idx],
            'RL': RL_f[idx],
            'RW': RW_f[idx],
            'DueTime': DueTime[k - 1],
            'ArrivalTime': max(arrival_time[k - 1], start_limit),
            'slack': slack,
        })

    # Sort by slack (tightest deadline first)
    servers.sort(key=lambda s: s['slack'])

    # Build cell list from facility
    ful_cells = [c for c in facility.all_cells.values() if c.status == CellStatus.FUL]
    cell_free_times = np.zeros(len(ful_cells))

    # Initialize from RL occupancy if provided
    if initial_cell_free_times is not None:
        for ci, cell in enumerate(ful_cells):
            key = (cell.bank_id, cell.position)
            if key in initial_cell_free_times:
                cell_free_times[ci] = initial_cell_free_times[key]

    # Build adjacency map
    adjacent = {}
    for i, cell in enumerate(ful_cells):
        for j, adj_cell in enumerate(ful_cells):
            if adj_cell.bank_id == cell.bank_id and adj_cell.position == cell.position + 1:
                adjacent[i] = j
                break

    assignments = {}

    for server in servers:
        best_assignment = None
        min_tardiness = float('inf')
        is_2tc = server['OTk'] == 1
        needs_hv = server['RH'] == 1
        needs_water = server['RW'] == 1

        for cell_idx, cell in enumerate(ful_cells):
            # Voltage check
            if needs_hv and cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
                continue
            if not needs_hv and cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
                continue
            # Water check
            if needs_water and cell.water_hoses == 0:
                continue

            if is_2tc:
                if cell_idx not in adjacent:
                    continue
                adj_idx = adjacent[cell_idx]
                adj_cell = ful_cells[adj_idx]
                if needs_hv and adj_cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
                    continue
                if not needs_hv and adj_cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
                    continue
                if needs_water and adj_cell.water_hoses == 0:
                    continue

                start_time = max(server['ArrivalTime'],
                                 cell_free_times[cell_idx],
                                 cell_free_times[adj_idx])
                end_time = start_time + server['PTk']

                if end_time <= end_limit + 1:
                    tardiness = max(0, end_time - server['DueTime'])
                    if tardiness < min_tardiness:
                        min_tardiness = tardiness
                        best_assignment = ('2tc', cell_idx, adj_idx, start_time, end_time)
            else:
                start_time = max(server['ArrivalTime'], cell_free_times[cell_idx])
                end_time = start_time + server['PTk']

                if end_time <= end_limit + 1:
                    tardiness = max(0, end_time - server['DueTime'])
                    if tardiness < min_tardiness:
                        min_tardiness = tardiness
                        best_assignment = ('1tc', cell_idx, start_time, end_time)

        if best_assignment is not None:
            if best_assignment[0] == '2tc':
                _, ci, ai, st, et = best_assignment
                cell_free_times[ci] = et
                cell_free_times[ai] = et
                assignments[server['k']] = [ful_cells[ci].bank_id,
                                            ful_cells[ci].position,
                                            int(st), int(et) - 1]
            else:
                _, ci, st, et = best_assignment
                cell_free_times[ci] = et
                assignments[server['k']] = [ful_cells[ci].bank_id,
                                            ful_cells[ci].position,
                                            int(st), int(et) - 1]

    print(f"Greedy: {len(assignments)}/{len(K_f)} assigned")
    return assignments


# =============================================================================
# SECTION 10b: POST-HOC CONSTRAINT VALIDATOR
# =============================================================================

def validate_solution(env):
    """Post-hoc validation of ALL Parvez constraints on the current solution.

    Checks constraints from Parvez's OD-DBP-LA formulation:
    1. Cell capacity — no double-booking (at most 1 server per cell per time)
    2. 2TC adjacency — 2TC servers use exactly 2 adjacent cells in same bank
    3. Voltage HV compatibility
    4. Voltage LV compatibility
    5. Water cooling compatibility
    6. No assignment before arrival time
    7. Processing time — server occupies cell for exactly PTk time slots
    8. Contiguity — server processing is one contiguous block (no gaps)
    9. FAB blocking — no server placed on blocked cells during blocked periods
    10. Cell status — server only placed on FUL (functional) cells
    11. 1TC single cell — 1TC servers use exactly 1 cell
    12. Excluded cells — bank 4 positions 4,7 not usable (status != FUL)

    Returns:
        list of violation strings (empty = all valid)
    """
    violations = []
    occ = env.cell_occupancy

    # Build map: server_id -> list of (cell_idx, time_slot)
    server_cells = {}
    for ci in range(env.total_cells):
        for t in range(occ.shape[1]):
            sid = int(occ[ci, t])
            if sid > 0:
                if sid not in server_cells:
                    server_cells[sid] = []
                server_cells[sid].append((ci, t))

    # Constraint 1: Cell capacity — no double-booking
    # Our model stores a single int per (cell, time), so double-booking would
    # mean one server silently overwrote another. Detect by checking that the
    # occupancy grid is consistent with the assignments dict.
    assigned_cell_times = {}  # (ci, t) -> server_id
    for sid, slot_list in server_cells.items():
        for ci, t in slot_list:
            key = (ci, t)
            if key in assigned_cell_times and assigned_cell_times[key] != sid:
                violations.append(
                    f"Double-booking: cell {ci} t={t} has servers "
                    f"{assigned_cell_times[key]} and {sid}")
            assigned_cell_times[key] = sid

    # Check each assigned server
    for srv in env.servers:
        server_id = srv['k']
        if server_id not in server_cells:
            continue  # Unassigned — not a violation

        slots = server_cells[server_id]
        cells_used = set(ci for ci, _ in slots)
        times_used = sorted(set(t for _, t in slots))

        if not times_used:
            continue

        start_time = times_used[0]
        end_time = times_used[-1] + 1
        arrival = srv['ArrivalTime']
        pt = srv['PTk']
        is_2tc = srv['OTk'] == 1

        # Constraint 6: No assignment before arrival
        if start_time < arrival:
            violations.append(
                f"Server {server_id}: starts at t={start_time} before arrival t={arrival}")

        # Constraint 7: Processing time — each cell must have exactly PTk slots
        expected_slots_per_cell = pt
        for ci in cells_used:
            cell_slots = sum(1 for c, t in slots if c == ci)
            if cell_slots != expected_slots_per_cell:
                violations.append(
                    f"Server {server_id}: cell {ci} has {cell_slots} slots, "
                    f"expected {expected_slots_per_cell}")

        # Constraint 8: Contiguity — time slots must be consecutive with no gaps
        if len(times_used) != (end_time - start_time):
            missing = set(range(start_time, end_time)) - set(times_used)
            violations.append(
                f"Server {server_id}: non-contiguous processing, "
                f"missing time slots {sorted(missing)}")

        # Constraint 2: 2TC adjacency — must use exactly 2 adjacent cells
        if is_2tc:
            if len(cells_used) != 2:
                violations.append(
                    f"Server {server_id}: 2TC but uses {len(cells_used)} cells (need 2)")
            else:
                c1, c2 = sorted(cells_used)
                is_adjacent = False
                if c1 in env.adjacent_cells and env.adjacent_cells[c1] == c2:
                    is_adjacent = True
                if c2 in env.adjacent_cells and env.adjacent_cells[c2] == c1:
                    is_adjacent = True
                if not is_adjacent:
                    violations.append(
                        f"Server {server_id}: 2TC cells {c1},{c2} are not adjacent")
                # Check both cells have the same time slots
                c1_times = sorted(t for ci, t in slots if ci == c1)
                c2_times = sorted(t for ci, t in slots if ci == c2)
                if c1_times != c2_times:
                    violations.append(
                        f"Server {server_id}: 2TC cells {c1},{c2} have "
                        f"mismatched time slots")

        # Constraint 11: 1TC servers must use exactly 1 cell
        elif not is_2tc and len(cells_used) != 1:
            violations.append(
                f"Server {server_id}: 1TC but uses {len(cells_used)} cells")

        # Constraints 3-5: Voltage HV/LV & water cooling compatibility
        # Constraint 10: Cell status must be FUL
        for ci in cells_used:
            cell = env.idx_to_cell[ci]
            if cell.status != CellStatus.FUL:
                violations.append(
                    f"Server {server_id}: placed on cell {ci} with "
                    f"status {cell.status} (need FUL)")
            if srv.get('RH', 0) == 1 and cell.voltage_type not in (VoltageType.HV, VoltageType.BOTH):
                violations.append(
                    f"Server {server_id}: needs HV but cell {ci} is {cell.voltage_type}")
            if srv.get('RL', 0) == 1 and srv.get('RH', 0) == 0:
                if cell.voltage_type not in (VoltageType.LV, VoltageType.BOTH):
                    violations.append(
                        f"Server {server_id}: needs LV but cell {ci} is {cell.voltage_type}")
            if srv.get('RW', 0) == 1 and cell.water_hoses == 0:
                violations.append(
                    f"Server {server_id}: needs water cooling but cell {ci} has none")

    # Constraint 9: FAB blocking — no server placed on blocked cells
    if hasattr(env, 'block_periods'):
        for blocked_ci, periods in env.block_periods.items():
            for bs, be in periods:
                for t in range(bs, min(be + 1, occ.shape[1])):
                    val = int(occ[blocked_ci, t])
                    if val > 0:
                        violations.append(
                            f"Server {val}: placed on FAB-blocked cell "
                            f"{blocked_ci} at t={t} (block {bs}-{be})")

    return violations


# =============================================================================
# SECTION 11: CPLEX ILP WITH WARM-START
# =============================================================================

def run_model_with_warmstart(K, T, I, J, VH, VL, RH, RL, W, RW,
                              power_cell, PTk, OTk, atk,
                              block_No, start_limit, end_limit,
                              panda_df, power_df, block_list_file_dir,
                              ki={}, kj={}, order_with_remaining_time=[],
                              remaining_time=[], unit_time_length=0,
                              model_execution_time=0, cost_list=[],
                              rl_assignments=None):
    """CPLEX ILP model with RL warm-start (soft hint).

    Port of function.py with GA hard constraints REPLACED by MIP warm-start.
    Carry-over constraints remain as hard constraints.
    Updated to use 4 banks.
    """
    from docplex.mp.model import Model

    mdl = Model('CVRP_WarmStart')

    # Parameters
    if model_execution_time > 0:
        mdl.parameters.timelimit = model_execution_time
    mdl.parameters.threads = 16
    mdl.parameters.mip.tolerances.mipgap = 0.0001
    mdl.parameters.randomseed = 1

    C = 8
    B = unit_time_length * 160
    B2 = unit_time_length * 176
    D = 2
    M1 = 1000000

    # Decision variables
    A = [(k, t, i, j) for k in K for t in T for i in I for j in J]
    x = mdl.binary_var_dict(A, name='x')

    CCC = [(k, i, j) for k in K for i in I for j in J]
    ckij = mdl.binary_var_dict(CCC, name='ckij')

    C2 = [(k, i, j) for k in K for i in I for j in J[:-1]]
    c2 = mdl.binary_var_dict(C2, name='c2')

    YY = [(t, i) for t in T for i in I]
    y = mdl.binary_var_dict(YY, name='y')
    YY2 = [(t, i) for t in T for i in I]
    y2 = mdl.binary_var_dict(YY2, name='y2')

    Ckt = [(k, t) for k in K for t in T]
    ckt = mdl.binary_var_dict(Ckt, name='ckt')
    C3 = [(k, t) for k in K for t in T]
    c3 = mdl.binary_var_dict(C3, name='c3')

    YK = [k for k in K]
    yk = mdl.binary_var_dict(YK, name='Yk')

    power_k = [k for k in K]
    power = mdl.continuous_var_dict(power_k, name="individual_power_consumption")
    total_power = mdl.continuous_var(name="TotalPowerConsumption")

    # Objective: minimize weighted unfinished
    mdl.minimize(mdl.sum((1 - yk[k]) * cost_list[k] for k in K))

    # Power calculation
    for k in K:
        power[k] = mdl.sum(power_cell[K.index(k)] * x[k, t, i, j]
                           for t in T for i in I for j in J)

    # Constraint: at most one order per cell per time period
    mdl.add_constraints(
        mdl.sum(x[k, t, i, j] for k in K) <= 1
        for t in T for i in I for j in J)

    # Constraint: server occupies at most (1+OTk) cells per bank per time
    mdl.add_constraints(
        mdl.sum(x[k, t, i, j] for j in J) <= (1 + OTk[K.index(k)])
        for k in K for t in T for i in I)
    mdl.add_constraints(
        mdl.sum(x[k, t, i, j] for i in I for j in J) <= (1 + OTk[K.index(k)])
        for k in K for t in T)

    # Processing time constraint
    mdl.add_constraints(
        mdl.sum(
            mdl.sum(x[k, t, i, j] for i in I for j in J) * atk[K.index(k)][t - 1]
            for t in T
        ) == (PTk[K.index(k)] * (1 + OTk[K.index(k)]) * yk[k])
        for k in K)

    # Arrival time constraint
    for k in K:
        cumulative = np.array([atk[K.index(k)][t - 1] for t in T])
        cumulative = np.cumsum(cumulative)
        time_to_be_used = len(cumulative) - np.count_nonzero(cumulative)
        mdl.add_constraint(
            mdl.sum(x[k, t, i, j] for i in I for j in J for t in T[:time_to_be_used]) == 0)

    # Cell assignment constraint
    for k in K:
        for t in T:
            for i in I:
                for j in J:
                    mdl.add_if_then(ckij[k, i, j] == 0, x[k, t, i, j] == 0)

    mdl.add_constraints(
        mdl.sum(ckij[k, i, j] for i in I for j in J) == (1 + OTk[K.index(k)])
        for k in K)

    # 2TC adjacency constraint
    for k in K:
        for i in I:
            for j in J[:-1]:
                mdl.add_constraint(c2[k, i, j] <= ckij[k, i, j])
                mdl.add_constraint(c2[k, i, j] <= ckij[k, i, j + 1])
                mdl.add_constraint(c2[k, i, j] >= ckij[k, i, j] + ckij[k, i, j + 1] - 1)

    mdl.add_constraints(
        mdl.sum(c2[k, i, j] for i in I for j in J[:-1]) == 1 * OTk[K.index(k)]
        for k in K)

    # Contiguity constraints
    mdl.add_constraints(mdl.sum(c3[k, t] for t in T) == 1 for k in K)

    for k in K:
        for t in T[:(len(T) - 1 + 1)]:
            for tt in range(start_limit, t):
                mdl.add_if_then(c3[k, t] == 1, ckt[k, tt] == 0)
            tmp_time = t + PTk[K.index(k)]
            if tmp_time >= T[-1]:
                tmp_time = T[-1] + 1
            for tt in range(t, tmp_time):
                mdl.add_if_then(c3[k, t] == 1, ckt[k, tt] == 1)
            for tt in range(tmp_time, T[-1] + 1):
                mdl.add_if_then(c3[k, t] == 1, ckt[k, tt] == 0)
        for t in T[(len(T) - 1 + 1):]:
            mdl.add_constraint(c3[k, t] == 0)

    for k in K:
        for t in T:
            for i in I:
                for j in J:
                    mdl.add_if_then(ckt[k, t] == 0, x[k, t, i, j] == 0)

    # Voltage/water compatibility
    for k in K:
        for t in T:
            for i in I:
                for j in J:
                    if (((VH[i-1][j-1] == 0) and (RH[K.index(k)] == 1)) or
                        ((VL[i-1][j-1] == 0) and (RL[K.index(k)] == 1)) or
                        ((W[i-1][j-1] == 0) and (RW[K.index(k)] == 1))):
                        mdl.add_constraint(x[k, t, i, j] == 0)

    # Fabrication blocking constraints
    df_block = pd.read_excel(block_list_file_dir)
    i_ = pd.DataFrame(df_block['i']).to_numpy().flatten()
    j_ = pd.DataFrame(df_block['j']).to_numpy().flatten()
    st_ = pd.DataFrame(df_block['st']).to_numpy().flatten()
    tt_ = pd.DataFrame(df_block['tt']).to_numpy().flatten()

    week_length = int((5 * 24) / unit_time_length)
    return_fabrication_block_data = []

    for k in K:
        for n in range(len(i_)):
            start_time_block = (st_[n] - 1) * week_length + 1
            if start_time_block < start_limit:
                start_time_block = start_limit
            end_time_block = tt_[n] * week_length
            if end_time_block > end_limit:
                end_time_block = end_limit
            time_range = list(range(start_time_block, end_time_block + 1))
            i_val = i_[n]
            j_val = j_[n]
            if i_val != 4:
                mdl.add_constraint(mdl.sum(x[k, t, i_val, j_val] for t in time_range) == 0)
            for t in time_range:
                return_fabrication_block_data.append(f"{k} {t} {i_val} {j_val}")
                panda_df.at[f"{i_val},{j_val}", str(t)] = "Blocked"

    # Carry-over constraints (HARD — from previous window's CPLEX solution)
    initial_remaining_time = copy.deepcopy(remaining_time)
    for z in order_with_remaining_time:
        bank = ki[str(z)]
        cell = kj[str(z)]
        time_period = len(T)
        if time_period < remaining_time[z - 1]:
            block_length = time_period
        else:
            block_length = remaining_time[z - 1]
        for t in T[:block_length]:
            for i_val in bank:
                for j_val in cell:
                    mdl.add_constraint(x[z, t, i_val, j_val] == 1)
        remaining_time[z - 1] = remaining_time[z - 1] - block_length

    # ======================================================================
    # MIP WARM-START from RL assignments (SOFT HINT — replaces GA hard lock)
    # ======================================================================
    if rl_assignments and len(rl_assignments) > 0:
        from docplex.mp.solution import SolveSolution
        warmstart = SolveSolution(mdl)

        for server_id, (bank_h, cell_h, start_h, end_h) in rl_assignments.items():
            # Set x variables for this assignment
            for time_h in range(start_h, end_h + 1):
                if (server_id, time_h, bank_h, cell_h) in x:
                    warmstart.add_var_value(x[server_id, time_h, bank_h, cell_h], 1)
                # 2TC: adjacent cell (cell+1)
                if server_id in K and OTk[K.index(server_id)] == 1:
                    adj_cell = cell_h + 1
                    if (server_id, time_h, bank_h, adj_cell) in x:
                        warmstart.add_var_value(x[server_id, time_h, bank_h, adj_cell], 1)
            warmstart.add_var_value(yk[server_id], 1)

        # Unassigned by RL -> hint yk=0
        for k in K:
            if k not in rl_assignments:
                warmstart.add_var_value(yk[k], 0)

        mdl.add_mip_start(warmstart)
        print(f"MIP warm-start added: {len(rl_assignments)} RL assignments")

    # Solve
    mdl.print_information()
    try:
        solution = mdl.solve(log_output=True)
    except Exception as cplex_err:
        print(f"CPLEX solve error: {cplex_err}")
        print("NOTE: CPLEX Community Edition is limited to 1000 variables.")
        print("A full CPLEX license is required for production runs.")
        solution = None

    # Check for infeasibility
    optimality_gap = 0
    total_solve_time = 0
    unassigned = list(K)  # Default: all unassigned on failure

    if solution is not None:
        import docplex.mp.conflict_refiner as cr
        solve_status = mdl.get_solve_status()
        print(f"Solve status: {solve_status}")
        if solve_status.name == 'INFEASIBLE_SOLUTION':
            cref = cr.ConflictRefiner()
            print('Conflict refinement:')
            cref.refine_conflict(mdl, display=True).display()

        # Extract unassigned
        unassigned = []
        for var_name, var in yk.items():
            if round(mdl.solution.get_value(var)) == 0.0:
                unassigned.append(var_name)

        # Build data representation
        for i_sol in solution.iter_var_values():
            if round(i_sol[1]) == 1:
                main_string = str(i_sol[0])
                sp = main_string.split("_")
                if sp[0] in ("x", "X"):
                    row_name = f"{sp[3]},{sp[4]}"
                    column_name = sp[2]
                    order_name = int(sp[1])
                    panda_df.at[row_name, column_name] = order_name

        # Power consumption tracking
        order_column_name = f"OrderForLookahead={block_No}"
        consumption_column_name = f"PowerConsumptionForLookahead={block_No}"
        assigned_order = []
        power_consumption_for_order = []
        gap = []
        for var_name, var in power.items():
            assigned_order.append(int(var_name))
            power_consumption_for_order.append(mdl.solution.get_value(var))
            gap.append('|')

        assigned_order.append("Total Power Consumption=")
        gap.append('|')

        df1 = pd.DataFrame({order_column_name: assigned_order})
        df2 = pd.DataFrame({consumption_column_name: power_consumption_for_order})
        df_gap = pd.DataFrame({"LookaheadEnd": gap})
        data_frame = pd.concat([df1, df2, df_gap], axis=1)
        power_df = pd.concat([power_df, data_frame], axis=1)

        optimality_gap = mdl.solve_details.mip_relative_gap
        total_solve_time = mdl.solve_details.time

    print(f"Solution for block={block_No}")
    print(f"Optimality gap: {optimality_gap}, Solve time: {total_solve_time}s")

    return (solution, remaining_time, return_fabrication_block_data,
            (total_solve_time, optimality_gap), copy.deepcopy(unassigned),
            panda_df, power_df)


# =============================================================================
# SECTION 12: PARVEZ-ONLY ILP (original function.py with GA hard constraints)
# =============================================================================

def run_model_parvez(K, T, I, J, VH, VL, RH, RL, W, RW,
                     power_cell, PTk, OTk, atk,
                     block_No, start_limit, end_limit,
                     panda_df, power_df, block_list_file_dir,
                     ki={}, kj={}, order_with_remaining_time=[],
                     remaining_time=[], unit_time_length=0,
                     model_execution_time=0, cost_list=[],
                     heuristic_assignment={}, heu_unassigned=[]):
    """CPLEX ILP model with GA hard constraints (Parvez baseline).

    Exact port of function.py, updated to 4 banks.
    """
    from docplex.mp.model import Model

    mdl = Model('CVRP_Parvez')

    if model_execution_time > 0:
        mdl.parameters.timelimit = model_execution_time
    mdl.parameters.threads = 16
    mdl.parameters.mip.tolerances.mipgap = 0.0001
    mdl.parameters.randomseed = 1

    C = 8
    B = unit_time_length * 160
    B2 = unit_time_length * 176
    M1 = 1000000

    A = [(k, t, i, j) for k in K for t in T for i in I for j in J]
    x = mdl.binary_var_dict(A, name='x')

    CCC = [(k, i, j) for k in K for i in I for j in J]
    ckij = mdl.binary_var_dict(CCC, name='ckij')

    C2 = [(k, i, j) for k in K for i in I for j in J[:-1]]
    c2 = mdl.binary_var_dict(C2, name='c2')

    YY = [(t, i) for t in T for i in I]
    y = mdl.binary_var_dict(YY, name='y')
    YY2 = [(t, i) for t in T for i in I]
    y2 = mdl.binary_var_dict(YY2, name='y2')

    Ckt = [(k, t) for k in K for t in T]
    ckt = mdl.binary_var_dict(Ckt, name='ckt')
    C3 = [(k, t) for k in K for t in T]
    c3 = mdl.binary_var_dict(C3, name='c3')

    YK = [k for k in K]
    yk = mdl.binary_var_dict(YK, name='Yk')

    power_k = [k for k in K]
    power = mdl.continuous_var_dict(power_k, name="individual_power_consumption")
    total_power = mdl.continuous_var(name="TotalPowerConsumption")

    # Objective
    mdl.minimize(mdl.sum((1 - yk[k]) * cost_list[k] for k in K))

    for k in K:
        power[k] = mdl.sum(power_cell[K.index(k)] * x[k, t, i, j]
                           for t in T for i in I for j in J)

    # All constraints identical to warm-start version
    mdl.add_constraints(
        mdl.sum(x[k, t, i, j] for k in K) <= 1
        for t in T for i in I for j in J)

    mdl.add_constraints(
        mdl.sum(x[k, t, i, j] for j in J) <= (1 + OTk[K.index(k)])
        for k in K for t in T for i in I)
    mdl.add_constraints(
        mdl.sum(x[k, t, i, j] for i in I for j in J) <= (1 + OTk[K.index(k)])
        for k in K for t in T)

    mdl.add_constraints(
        mdl.sum(
            mdl.sum(x[k, t, i, j] for i in I for j in J) * atk[K.index(k)][t - 1]
            for t in T
        ) == (PTk[K.index(k)] * (1 + OTk[K.index(k)]) * yk[k])
        for k in K)

    for k in K:
        cumulative = np.array([atk[K.index(k)][t - 1] for t in T])
        cumulative = np.cumsum(cumulative)
        time_to_be_used = len(cumulative) - np.count_nonzero(cumulative)
        mdl.add_constraint(
            mdl.sum(x[k, t, i, j] for i in I for j in J for t in T[:time_to_be_used]) == 0)

    for k in K:
        for t in T:
            for i in I:
                for j in J:
                    mdl.add_if_then(ckij[k, i, j] == 0, x[k, t, i, j] == 0)

    mdl.add_constraints(
        mdl.sum(ckij[k, i, j] for i in I for j in J) == (1 + OTk[K.index(k)])
        for k in K)

    for k in K:
        for i in I:
            for j in J[:-1]:
                mdl.add_constraint(c2[k, i, j] <= ckij[k, i, j])
                mdl.add_constraint(c2[k, i, j] <= ckij[k, i, j + 1])
                mdl.add_constraint(c2[k, i, j] >= ckij[k, i, j] + ckij[k, i, j + 1] - 1)

    mdl.add_constraints(
        mdl.sum(c2[k, i, j] for i in I for j in J[:-1]) == 1 * OTk[K.index(k)]
        for k in K)

    mdl.add_constraints(mdl.sum(c3[k, t] for t in T) == 1 for k in K)

    for k in K:
        for t in T[:(len(T) - 1 + 1)]:
            for tt in range(start_limit, t):
                mdl.add_if_then(c3[k, t] == 1, ckt[k, tt] == 0)
            tmp_time = t + PTk[K.index(k)]
            if tmp_time >= T[-1]:
                tmp_time = T[-1] + 1
            for tt in range(t, tmp_time):
                mdl.add_if_then(c3[k, t] == 1, ckt[k, tt] == 1)
            for tt in range(tmp_time, T[-1] + 1):
                mdl.add_if_then(c3[k, t] == 1, ckt[k, tt] == 0)
        for t in T[(len(T) - 1 + 1):]:
            mdl.add_constraint(c3[k, t] == 0)

    for k in K:
        for t in T:
            for i in I:
                for j in J:
                    mdl.add_if_then(ckt[k, t] == 0, x[k, t, i, j] == 0)

    for k in K:
        for t in T:
            for i in I:
                for j in J:
                    if (((VH[i-1][j-1] == 0) and (RH[K.index(k)] == 1)) or
                        ((VL[i-1][j-1] == 0) and (RL[K.index(k)] == 1)) or
                        ((W[i-1][j-1] == 0) and (RW[K.index(k)] == 1))):
                        mdl.add_constraint(x[k, t, i, j] == 0)

    # Blocking data
    df_block = pd.read_excel(block_list_file_dir)
    i_ = pd.DataFrame(df_block['i']).to_numpy().flatten()
    j_ = pd.DataFrame(df_block['j']).to_numpy().flatten()
    st_ = pd.DataFrame(df_block['st']).to_numpy().flatten()
    tt_ = pd.DataFrame(df_block['tt']).to_numpy().flatten()

    week_length = int((5 * 24) / unit_time_length)
    return_fabrication_block_data = []

    for k in K:
        for n in range(len(i_)):
            start_time_block = (st_[n] - 1) * week_length + 1
            if start_time_block < start_limit:
                start_time_block = start_limit
            end_time_block = tt_[n] * week_length
            if end_time_block > end_limit:
                end_time_block = end_limit
            time_range = list(range(start_time_block, end_time_block + 1))
            i_val = i_[n]
            j_val = j_[n]
            if i_val != 4:
                mdl.add_constraint(mdl.sum(x[k, t, i_val, j_val] for t in time_range) == 0)
            for t in time_range:
                return_fabrication_block_data.append(f"{k} {t} {i_val} {j_val}")
                panda_df.at[f"{i_val},{j_val}", str(t)] = "Blocked"

    # Carry-over constraints (HARD)
    initial_remaining_time = copy.deepcopy(remaining_time)
    for z in order_with_remaining_time:
        bank = ki[str(z)]
        cell = kj[str(z)]
        time_period = len(T)
        if time_period < remaining_time[z - 1]:
            block_length = time_period
        else:
            block_length = remaining_time[z - 1]
        for t in T[:block_length]:
            for i_val in bank:
                for j_val in cell:
                    mdl.add_constraint(x[z, t, i_val, j_val] == 1)
        remaining_time[z - 1] = remaining_time[z - 1] - block_length

    # GA HARD CONSTRAINTS (Parvez original lines 290-306)
    for order_h in heuristic_assignment:
        bank_h = heuristic_assignment[order_h][0]
        cell_h = heuristic_assignment[order_h][1]
        start_h = heuristic_assignment[order_h][2]
        end_h = heuristic_assignment[order_h][3]
        for time_h in range(start_h, end_h + 1):
            mdl.add_constraint(x[order_h, time_h, bank_h, cell_h] == 1)
            if OTk[K.index(order_h)] == 1:
                mdl.add_constraint(x[order_h, time_h, bank_h, cell_h - 1] == 1)

    # Force unassigned by GA to zero
    for unassigned_order in heu_unassigned:
        mdl.add_constraint(
            mdl.sum(x[unassigned_order, t, i, j] for t in T for i in I for j in J) == 0)

    # Solve
    mdl.print_information()
    try:
        solution = mdl.solve(log_output=True)
    except Exception as cplex_err:
        print(f"CPLEX solve error: {cplex_err}")
        print("NOTE: CPLEX Community Edition is limited to 1000 variables.")
        solution = None

    optimality_gap = 0
    total_solve_time = 0
    unassigned = list(K)

    if solution is not None:
        import docplex.mp.conflict_refiner as cr
        solve_status = mdl.get_solve_status()
        print(f"Solve status: {solve_status}")
        if solve_status.name == 'INFEASIBLE_SOLUTION':
            cref = cr.ConflictRefiner()
            cref.refine_conflict(mdl, display=True).display()

        unassigned = []
        for var_name, var in yk.items():
            if round(mdl.solution.get_value(var)) == 0.0:
                unassigned.append(var_name)

        for i_sol in solution.iter_var_values():
            if round(i_sol[1]) == 1:
                main_string = str(i_sol[0])
                sp = main_string.split("_")
                if sp[0] in ("x", "X"):
                    row_name = f"{sp[3]},{sp[4]}"
                    column_name = sp[2]
                    order_name = int(sp[1])
                    panda_df.at[row_name, column_name] = order_name

        order_column_name = f"OrderForLookahead={block_No}"
        consumption_column_name = f"PowerConsumptionForLookahead={block_No}"
        assigned_order = []
        power_consumption_for_order = []
        gap = []
        for var_name, var in power.items():
            assigned_order.append(int(var_name))
            power_consumption_for_order.append(mdl.solution.get_value(var))
            gap.append('|')
        assigned_order.append("Total Power Consumption=")
        gap.append('|')

        df1 = pd.DataFrame({order_column_name: assigned_order})
        df2 = pd.DataFrame({consumption_column_name: power_consumption_for_order})
        df_gap = pd.DataFrame({"LookaheadEnd": gap})
        data_frame = pd.concat([df1, df2, df_gap], axis=1)
        power_df = pd.concat([power_df, data_frame], axis=1)

        optimality_gap = mdl.solve_details.mip_relative_gap
        total_solve_time = mdl.solve_details.time

    print(f"Solution for block={block_No}")
    print(f"Optimality gap: {optimality_gap}, Solve time: {total_solve_time}s")

    return (solution, remaining_time, return_fabrication_block_data,
            (total_solve_time, optimality_gap), copy.deepcopy(unassigned),
            panda_df, power_df)


# =============================================================================
# SECTION 13: MAIN PIPELINE
# =============================================================================

def main_function(mode, dataset_path, facility, divide_time=DIVIDE_TIME,
                  model_path=None, cplex_time_limit=3600,
                  unit_time_length=UNIT_TIME, solution_method='decom',
                  block_list_path=None):
    """Main pipeline with 3 modes: parvez_only, rl_only, hybrid.

    Port of Paper_1_Codes/main.ipynb window loop with mode switching.
    """
    if block_list_path is None:
        block_list_path = os.path.join(INPUT_DIR, 'block_list.xlsx')

    # Read data using facility config
    (serial, VH, VL, W, K, PTk, OTk, RH, RL, RW, TYPE,
     DueTime, arrival_time, time_val, T, _, power_cell, atk) = \
        file_read_from_facility(facility, dataset_path, unit_time_length)

    # Index sets (4 banks, 14 cells per bank)
    I = np.arange(1, NUM_BANKS + 1)
    J = np.arange(1, NUM_CELLS_PER_BANK + 1)

    # Window partitioning
    list_feed_model = generate_list_feed_model(time_val, divide_time, arrival_time, K)
    processing_time_feed, remaining_time_feed = modified_processing_time(
        list_feed_model, divide_time, arrival_time, PTk)

    # State tracking
    ki = {}
    kj = {}
    order_with_remaining_time = []
    unassigned = []
    olist = []

    # Total runtime management
    total_run_time = cplex_time_limit * len(list_feed_model)
    max_iteration_run_time = cplex_time_limit

    # Output DataFrames for data representation
    row_names = [f"{i},{j}" for i in I for j in J]
    col_names = [str(t) for t in T]
    panda_df = pd.DataFrame(index=row_names, columns=col_names)
    power_df = pd.DataFrame()

    # Metrics collection
    window_metrics = []
    total_rl_time = 0.0
    total_cplex_time = 0.0
    all_unassigned = set()

    print(f"\n{'='*80}")
    print(f"RUNNING MODE: {mode.upper()}")
    print(f"Dataset: {dataset_path}")
    print(f"Orders: {len(K)}, Windows: {len(list_feed_model)}, "
          f"Window size: {divide_time}")
    print(f"{'='*80}\n")

    for i in range(len(list_feed_model)):
        # Build window data
        K_f = []
        start_limit = divide_time * i + 1
        end_limit = divide_time * i + divide_time
        if end_limit > time_val:
            end_limit = time_val

        T_f = list(range(start_limit, end_limit + 1))

        RH_f, RL_f, RW_f, power_cell_f = [], [], [], []
        PTk_f, OTk_f, atk_f = [], [], []

        # Add orders from this window
        for j in list_feed_model[i]:
            K_f.append(j)
            RH_f.append(RH[j - 1])
            RL_f.append(RL[j - 1])
            RW_f.append(RW[j - 1])
            power_cell_f.append(power_cell[j - 1])
            if processing_time_feed[j - 1] > len(T_f):
                remaining_time_feed[j - 1] = (remaining_time_feed[j - 1] +
                                                processing_time_feed[j - 1] - len(T_f))
                processing_time_feed[j - 1] = len(T_f)
            PTk_f.append(processing_time_feed[j - 1])
            OTk_f.append(OTk[j - 1])
            atk_f.append(atk[j - 1])

        # Add previously unassigned orders
        for j in unassigned:
            K_f.append(j)
            RH_f.append(RH[j - 1])
            RL_f.append(RL[j - 1])
            RW_f.append(RW[j - 1])
            power_cell_f.append(power_cell[j - 1])
            if divide_time >= PTk[j - 1]:
                processing_time_feed[j - 1] = PTk[j - 1]
                remaining_time_feed[j - 1] = 0
            else:
                processing_time_feed[j - 1] = divide_time
                remaining_time_feed[j - 1] = PTk[j - 1] - divide_time
            if processing_time_feed[j - 1] > len(T_f):
                remaining_time_feed[j - 1] = (remaining_time_feed[j - 1] +
                                                processing_time_feed[j - 1] - len(T_f))
                processing_time_feed[j - 1] = len(T_f)
            PTk_f.append(processing_time_feed[j - 1])
            OTk_f.append(OTk[j - 1])
            atk_f.append(atk[j - 1])
        unassigned = []

        # Add orders with remaining time from previous window
        for j in order_with_remaining_time:
            K_f.append(j)
            RH_f.append(RH[j - 1])
            RL_f.append(RL[j - 1])
            RW_f.append(RW[j - 1])
            power_cell_f.append(power_cell[j - 1])
            if len(T_f) < remaining_time_feed[j - 1]:
                PTk_f.append(len(T_f))
            else:
                PTk_f.append(remaining_time_feed[j - 1])
            OTk_f.append(OTk[j - 1])
            atk_f.append(atk[j - 1])

        if len(K_f) == 0:
            print(f"Window {i+1}: No orders, skipping")
            window_metrics.append({
                'window': i + 1, 'orders': 0, 'assigned': 0,
                'unfinished': 0, 'tardy': 0,
                'rl_time': 0, 'cplex_time': 0, 'gap': 0
            })
            continue

        # Cost calculation
        cost_list = cost_calculation(copy.deepcopy(K_f), DueTime, arrival_time)

        # Time limit check
        if total_run_time > max_iteration_run_time:
            model_execution_time = max_iteration_run_time
        else:
            model_execution_time = total_run_time
        if model_execution_time < 5:
            print("Time budget exhausted, stopping")
            break

        print(f"\n--- Window {i+1}/{len(list_feed_model)} "
              f"[{start_limit}-{end_limit}] | "
              f"{len(K_f)} orders ---")

        solution = None
        solution_data = (0, 0)
        returned_fabrication_data = []
        returned_test_utilization = []
        rl_time_window = 0.0

        # ===== MODE: PARVEZ_ONLY =====
        if mode == 'parvez_only':
            # GA Phase 1
            heu_unassigned = []
            heu_start = tm.time()
            sep_data = separate_heuristics_data(
                K_f, order_with_remaining_time, start_limit, end_limit,
                arrival_time, DueTime, VH, VL, RH_f, RL_f, W, RW_f,
                I, J, ki, kj, PTk_f, OTk_f, solution_method)

            if sep_data is not None:
                heuristic_assignment, heu_unassigned = heu_model(
                    *sep_data, block_list_path=block_list_path)
            else:
                heuristic_assignment, heu_unassigned = {}, []

            heu_time = tm.time() - heu_start

            # Filter to 2TC only
            heuristic_assignment, heu_unassigned = modify_heu_output(
                heuristic_assignment, heu_unassigned, OTk)

            # Update processing times for GA assignments
            for it in heuristic_assignment:
                assignment_ = heuristic_assignment[it]
                processed_time = assignment_[3] - assignment_[2] + 1
                remaining_to_add = PTk_f[K_f.index(it)] - processed_time
                PTk_f[K_f.index(it)] = processed_time
                remaining_time_feed[it - 1] = remaining_time_feed[it - 1] + remaining_to_add

            # CPLEX Phase 2
            (solution, remaining_time_feed, returned_fabrication_data,
             solution_data, unassigned, panda_df, power_df) = run_model_parvez(
                K_f, T_f, I, J, VH, VL, RH_f, RL_f, W, RW_f,
                power_cell_f, PTk_f, OTk_f, atk_f,
                (i + 1), start_limit, end_limit,
                panda_df, power_df, block_list_path,
                ki, kj, order_with_remaining_time,
                remaining_time_feed, unit_time_length,
                model_execution_time, cost_list,
                heuristic_assignment, heu_unassigned)
            total_cplex_time += solution_data[0]

        # ===== MODE: RL_ONLY =====
        elif mode == 'rl_only':
            rl_assignments, rl_time_window = LHAC_solve_window(
                K_f, T_f, I, J, VH, VL, W, RH_f, RL_f, RW_f,
                PTk_f, OTk_f, atk_f, arrival_time, DueTime,
                ki, kj, order_with_remaining_time,
                remaining_time_feed, start_limit, end_limit,
                cost_list, facility, model_path)
            total_rl_time += rl_time_window

            # Convert RL assignments to solution-like tracking
            # Update ki, kj, olist from RL assignments
            ki_new = {}
            kj_new = {}
            olist_new = []
            for server_id, (bank, cell, start, end) in rl_assignments.items():
                ki_new[str(server_id)] = [bank]
                kj_new[str(server_id)] = [cell]
                olist_new.append(server_id)
                # Update remaining time
                processed = end - start + 1
                if remaining_time_feed[server_id - 1] > 0:
                    remaining_time_feed[server_id - 1] -= processed
                    if remaining_time_feed[server_id - 1] < 0:
                        remaining_time_feed[server_id - 1] = 0

            unassigned = [k for k in K_f if k not in rl_assignments]
            # Sort unassigned by deadline urgency (tightest slack first)
            # so they get priority in the next window
            unassigned.sort(key=lambda k: DueTime[k-1] - arrival_time[k-1] - PTk[k-1])
            ki = ki_new
            kj = kj_new
            olist = olist_new
            solution_data = (rl_time_window, 0)

        # ===== MODE: HYBRID =====
        elif mode == 'hybrid':
            # RL Phase
            rl_assignments, rl_time_window = LHAC_solve_window(
                K_f, T_f, I, J, VH, VL, W, RH_f, RL_f, RW_f,
                PTk_f, OTk_f, atk_f, arrival_time, DueTime,
                ki, kj, order_with_remaining_time,
                remaining_time_feed, start_limit, end_limit,
                cost_list, facility, model_path)
            total_rl_time += rl_time_window

            # CPLEX Phase with warm-start
            (solution, remaining_time_feed, returned_fabrication_data,
             solution_data, unassigned, panda_df, power_df) = run_model_with_warmstart(
                K_f, T_f, I, J, VH, VL, RH_f, RL_f, W, RW_f,
                power_cell_f, PTk_f, OTk_f, atk_f,
                (i + 1), start_limit, end_limit,
                panda_df, power_df, block_list_path,
                ki, kj, order_with_remaining_time,
                remaining_time_feed, unit_time_length,
                model_execution_time, cost_list,
                rl_assignments)
            total_cplex_time += solution_data[0]

        # Extract solution data (for parvez_only and hybrid modes)
        if mode != 'rl_only':
            if solution is not None:
                ki, kj, olist, returned_test_utilization = extract_solution_data(solution)
            elif mode == 'hybrid' and rl_assignments:
                # CPLEX failed but RL assignments exist — fall back to RL results
                print(f"WARNING: CPLEX failed in window {i+1}, using RL assignments as fallback")
                ki_new = {}
                kj_new = {}
                olist_new = []
                for server_id, (bank, cell, start, end) in rl_assignments.items():
                    ki_new[str(server_id)] = [bank]
                    kj_new[str(server_id)] = [cell]
                    olist_new.append(server_id)
                    processed = end - start + 1
                    if remaining_time_feed[server_id - 1] > 0:
                        remaining_time_feed[server_id - 1] -= processed
                        if remaining_time_feed[server_id - 1] < 0:
                            remaining_time_feed[server_id - 1] = 0
                unassigned = [k for k in K_f if k not in rl_assignments]
                ki = ki_new
                kj = kj_new
                olist = olist_new
                returned_test_utilization = []
            else:
                print(f"WARNING: No solution in window {i+1}")
                ki, kj, olist, returned_test_utilization = {}, {}, [], []

        # Calculate utilization
        if returned_test_utilization:
            calculate_utilization(returned_test_utilization,
                                 len(T_f), f"Window {i+1}")

        # Update order_with_remaining_time for next window
        total_run_time -= solution_data[0]
        order_with_remaining_time = []
        for elem in olist:
            if remaining_time_feed[elem - 1] > 0:
                order_with_remaining_time.append(elem)

        # Record window metrics
        num_unassigned = len(unassigned) if isinstance(unassigned, list) else 0
        for u in (unassigned if isinstance(unassigned, list) else []):
            all_unassigned.add(u)

        window_metrics.append({
            'window': i + 1,
            'orders': len(K_f),
            'assigned': len(K_f) - num_unassigned,
            'unfinished': num_unassigned,
            'rl_time': rl_time_window,
            'cplex_time': solution_data[0],
            'gap': solution_data[1] if len(solution_data) > 1 else 0,
        })

        print(f"Window {i+1} done: {len(K_f)-num_unassigned}/{len(K_f)} assigned, "
              f"{num_unassigned} unfinished, "
              f"remaining={len(order_with_remaining_time)}")

    # Compile results
    results = compile_results(window_metrics, mode, len(K),
                              total_rl_time, total_cplex_time,
                              all_unassigned, DueTime)

    # Save outputs
    save_outputs(panda_df, power_df, results, mode)

    return results


# =============================================================================
# SECTION 14: RESULTS & OUTPUT
# =============================================================================

def compile_results(window_metrics, mode, total_orders,
                    total_rl_time, total_cplex_time,
                    all_unassigned, DueTime):
    """Compile and display results from all windows."""
    results = {
        'mode': mode,
        'total_orders': total_orders,
        'window_metrics': window_metrics,
        'total_rl_time': total_rl_time,
        'total_cplex_time': total_cplex_time,
        'total_runtime': total_rl_time + total_cplex_time,
        'total_unfinished': len(all_unassigned),
        'unfinished_orders': list(all_unassigned),
    }

    print(f"\n{'='*80}")
    print(f"RESULTS — MODE: {mode.upper()}")
    print(f"{'='*80}")
    print(f"Total orders:     {total_orders}")
    print(f"Total unfinished: {len(all_unassigned)}")
    print(f"Completion rate:  {(total_orders - len(all_unassigned)) / total_orders:.1%}")

    if mode == 'rl_only':
        print(f"RL time:          {total_rl_time:.2f}s")
    elif mode == 'hybrid':
        print(f"RL time:          {total_rl_time:.2f}s")
        print(f"CPLEX time:       {total_cplex_time:.2f}s")
        print(f"Total time:       {total_rl_time + total_cplex_time:.2f}s")
    else:
        print(f"CPLEX time:       {total_cplex_time:.2f}s")

    # Per-window summary
    print(f"\nPer-window breakdown:")
    print(f"{'Window':>8} {'Orders':>8} {'Assigned':>10} {'Unfinished':>12} "
          f"{'RL(s)':>8} {'CPLEX(s)':>10} {'Gap':>8}")
    for wm in window_metrics:
        print(f"{wm['window']:>8} {wm['orders']:>8} {wm['assigned']:>10} "
              f"{wm['unfinished']:>12} {wm['rl_time']:>8.2f} "
              f"{wm['cplex_time']:>10.2f} {wm['gap']:>8.4f}")

    print(f"{'='*80}\n")
    return results


def save_outputs(panda_df, power_df, results, mode):
    """Save output files to Output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data representation
    data_rep_path = os.path.join(OUTPUT_DIR, f'data_representation_{mode}.xlsx')
    try:
        panda_df.to_excel(data_rep_path)
        print(f"Data representation saved to {data_rep_path}")
    except Exception as e:
        print(f"Warning: Could not save data representation: {e}")

    # Power consumption
    if not power_df.empty:
        power_path = os.path.join(OUTPUT_DIR, f'power_consumption_{mode}.xlsx')
        try:
            power_df.to_excel(power_path, index=False)
            print(f"Power consumption saved to {power_path}")
        except Exception as e:
            print(f"Warning: Could not save power data: {e}")

    # Results summary
    results_path = os.path.join(OUTPUT_DIR, f'results_{mode}.txt')
    with open(results_path, 'w') as f:
        f.write(f"Mode: {results['mode']}\n")
        f.write(f"Total orders: {results['total_orders']}\n")
        f.write(f"Total unfinished: {results['total_unfinished']}\n")
        f.write(f"Total RL time: {results['total_rl_time']:.2f}s\n")
        f.write(f"Total CPLEX time: {results['total_cplex_time']:.2f}s\n")
        f.write(f"Total runtime: {results['total_runtime']:.2f}s\n")
        f.write(f"Unfinished orders: {results['unfinished_orders']}\n")
    print(f"Results saved to {results_path}")


# =============================================================================
# SECTION 15: ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RL Warm-Start Hybrid Solver for CTO Server Scheduling')

    parser.add_argument('--mode',
                        choices=['parvez_only', 'rl_only', 'hybrid', 'generate_data'],
                        default='hybrid',
                        help='Execution mode')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to order data Excel file')
    parser.add_argument('--orders', type=int, default=400,
                        help='Number of orders for data generation')
    parser.add_argument('--arrival', choices=['back', 'front', 'uniform'],
                        default='back',
                        help='Arrival pattern for data generation')
    parser.add_argument('--two_tc_pct', type=float, default=0.2,
                        help='Fraction of 2TC servers (0.1 or 0.3)')
    parser.add_argument('--window_size', type=int, default=DIVIDE_TIME,
                        help='Window size in time blocks')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained LHAC model')
    parser.add_argument('--cplex_time_limit', type=int, default=3600,
                        help='CPLEX time limit per window (seconds)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data generation')

    args = parser.parse_args()

    # Create facility
    facility = FacilityConfiguration()
    summary = facility.get_summary()
    print(f"Facility: {summary['total_banks']} banks, "
          f"{summary['total_cells']} total cells, "
          f"{summary['ful_cells']} FUL cells")

    # Ensure directories exist
    for d in [INPUT_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)

    # Generate CELL_BANK.xlsx if needed
    cell_bank_path = os.path.join(INPUT_DIR, 'CELL_BANK.xlsx')
    if not os.path.exists(cell_bank_path):
        save_cell_bank(facility, cell_bank_path)

    if args.mode == 'generate_data':
        # Generate synthetic dataset
        two_tc_int = int(args.two_tc_pct * 100)
        filename = f"{args.orders}_orders_{args.arrival}_{two_tc_int}%.xlsx"
        output_path = os.path.join(DATA_DIR, filename)

        df = generate_orders(args.orders, args.arrival,
                            args.two_tc_pct, args.seed)
        df.to_excel(output_path, index=False)
        print(f"Generated {len(df)} orders -> {output_path}")
        print(df.describe())
        return

    # Main solve modes
    if args.dataset_path is None:
        # Look for a default dataset
        default_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')]
        if default_files:
            args.dataset_path = os.path.join(DATA_DIR, sorted(default_files)[0])
            print(f"Using default dataset: {args.dataset_path}")
        else:
            print("ERROR: No dataset specified and no files found in Data/")
            print("Generate data first: python rl_warmstart_hybrid.py --mode generate_data")
            return

    results = main_function(
        mode=args.mode,
        dataset_path=args.dataset_path,
        facility=facility,
        divide_time=args.window_size,
        model_path=args.model_path,
        cplex_time_limit=args.cplex_time_limit,
        block_list_path=os.path.join(INPUT_DIR, 'block_list.xlsx'),
    )

    return results


if __name__ == '__main__':
    main()
