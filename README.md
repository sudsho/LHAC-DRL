# Lexicographic Hierarchical Deep Reinforcement Learning for Production Scheduling in High-End Server Manufacturing

Reference implementation of the LHAC (Lexicographic Hierarchical
Actor-Critic) framework for server-to-cell production scheduling,
accompanying the manuscript

> Srivastava, S., Aqlan, F., Parikh, P. J., and Noor-E-Alam, M.
> Lexicographic Hierarchical Deep Reinforcement Learning for
> Production Scheduling in High-End Server Manufacturing.
> *IISE Transactions* (under review).

LHAC formulates the OD-DBP-LA scheduling problem of Parvez et al. (2024)
as a stochastic, sequential, lexicographic bi-objective MOMDP, and
trains two actor-critic agents jointly with adaptive thresholded
lexicographic ordering. The repository contains the code used to
produce every numerical figure in the manuscript and a deployable
Streamlit dashboard that exposes the trained policy at inference time.

Live demo: <https://lhac-drl-uj9qgbettemzmd3tcnyvq8.streamlit.app>


## Repository layout

```
.
+-- lhac/                LHAC core package
|   +-- env.py           MOMDP environment for OD-DBP-LA
|   +-- networks.py      Actor-critic networks with CASE cross-attention
|   +-- ppo.py           Joint dual-agent PPO trainer
|   +-- tlo.py           Adaptive thresholded lexicographic filter
|   +-- dap.py           Deferred Action Placement helpers
|   +-- fpr.py           Feasibility-preserving reward shaping
|   +-- aggregator.py    Multi-seed validation aggregator
|   +-- data.py          OD-DBP-LA instance loader and helpers
+-- baselines/           Comparison methods used in the paper
|   +-- ga_mip.py        GA-MIP (Parvez et al., 2024)
|   +-- dispatch.py      EDD and slack-based dispatching
|   +-- dqn.py           DQN+DAF
|   +-- morl.py          PPO-Lagrangian, RCPO, Envelope MORL, LPPO
+-- reproduce/           Per-figure reproduction scripts
|   +-- reproduce_fig4_main.py
|   +-- reproduce_fig5_robustness.py
|   +-- reproduce_fig6_ablation.py
|   +-- reproduce_fig7_morl.py
|   +-- reproduce_fig9_arch.py
|   +-- reproduce_all.py
+-- results/             Aggregated multi-seed CSV summaries
+-- Models/              Trained checkpoints
+-- figures/             Output PNGs at 600 DPI
+-- streamlit_app.py     Interactive scheduling dashboard
+-- daf_lhac_core.py     LHAC environment, networks, and PPO trainer
+-- evaluate.py          Evaluation pipeline for the benchmark grid
+-- train_multibank.py   Multi-bank LHAC training driver
+-- hybrid_cplex.py      Hybrid LHAC+CPLEX residual solver
+-- requirements.txt
+-- LICENSE              MIT
```


## Installation

The code targets Python 3.10 or later. CUDA-enabled PyTorch is
recommended for training; inference and the Streamlit dashboard run
on CPU.

```
git clone https://github.com/Aqlanlab/LHAC-DRL.git
cd LHAC-DRL
pip install -r requirements.txt
```

The GA-MIP baseline uses CPLEX 22.1 through `docplex` when available
and falls back to a pure-Python genetic algorithm otherwise. A CPLEX
install is therefore optional but recommended for the paper's runtime
numbers.


## Reproducing the figures

The reproduction scripts run the full benchmark sweep against the
data path supplied through `--data`. The dataset used in the paper
is held by the authors under confidentiality restrictions and is not
redistributed in this repository (see *Data availability* below).

```
# Default: full sweep, seed-averaged
python reproduce/reproduce_fig4_main.py --data data/industrial_servers_2024.csv --seeds 10

# Cache mode (replots from the previously aggregated CSV under results/)
python reproduce/reproduce_fig4_main.py --use-cache
```

The remaining four figures expose the same interface:

| Manuscript | Script |
|------------|--------|
| Figure 4 (main comparison)        | `reproduce/reproduce_fig4_main.py`        |
| Figure 5 (robustness analysis)    | `reproduce/reproduce_fig5_robustness.py`  |
| Figure 6 (4-panel ablation)       | `reproduce/reproduce_fig6_ablation.py`    |
| Figure 7 (MORL Pareto comparison) | `reproduce/reproduce_fig7_morl.py`        |
| Figure 9 (architecture variants)  | `reproduce/reproduce_fig9_arch.py`        |

`python reproduce/reproduce_all.py` runs all five sequentially.


## Training

Hyperparameters follow Table 2 of the manuscript:

```
python train_multibank.py \
    --banks 4 --episodes 5000 \
    --seed 0 --save Models/daf_full_daf.pth
```

Both actor-critics are updated jointly under PPO with separate
advantage signals. Agent 1 optimises the FPR-shaped completion
reward `r_tilde_1` (Eq. 9); Agent 2 optimises the proportional
tardiness reward `r_2` (Eq. 4) restricted to Agent 1's adaptive
threshold candidate set. Checkpoints are written as a single `.pth`
file containing both networks and the current threshold value.


## Evaluation

```
python evaluate.py --model Models/daf_full_daf.pth \
    --dap --fpr --variant full_daf --windows 10d
```

Pass `--hybrid --hybrid_completion_threshold 1.0` to enable the
LHAC+CPLEX hybrid fallback for residual unplaced servers.


## Streamlit dashboard

```
streamlit run streamlit_app.py
```

The dashboard runs LHAC and the selected baselines on a chosen
facility configuration and reports completion rate, tardiness rate,
and runtime side by side. The hosted version at
<https://lhac-drl-uj9qgbettemzmd3tcnyvq8.streamlit.app> uses the same code path.


## Hardware used in the paper

All numerical results in the manuscript were produced on a single
workstation with the following configuration.

- 2 x Intel Xeon Platinum (48 physical cores total, 96 logical threads)
- NVIDIA RTX A6000, 48 GB VRAM
- 256 GB DDR4 ECC memory
- Windows 11 Pro for Workstations
- Python 3.11.2, PyTorch 2.1.0 with CUDA 12.1
- CPLEX 22.1.1


## Data availability

The dataset used to fit the calibration statistics and to evaluate
both LHAC and the GA-MIP baseline is proprietary production data
from an industry partner. The data are governed by a confidentiality
agreement and are not redistributed in this repository. Where
editorial verification of the manuscript is required, the data can
be made available through a secured, password-protected channel
under the IISE Transactions confidentiality protocol. Please refer
to the reproducibility report submitted alongside the manuscript.


## Citation

```bibtex
@article{srivastava2026lhac,
  author  = {Srivastava, Sudhanshu and Aqlan, Faisal and
             Parikh, Pratik J. and Noor-E-Alam, Md},
  title   = {Lexicographic Hierarchical Deep Reinforcement Learning
             for Production Scheduling in High-End Server
             Manufacturing},
  journal = {IISE Transactions},
  year    = {2026},
  note    = {Under review}
}
```


## Licence

Released under the MIT licence; see `LICENSE`.


## Contact

Faisal Aqlan, Department of Industrial & Systems Engineering, University of
Louisville. Email: `faisal.aqlan@louisville.edu`.
