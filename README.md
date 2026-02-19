# Articulated


<h3 align="center">
    Learning Neural Manifold Representations for Articulated Roboticss
</h3>



<div align="center">
    <img src="assets/overview.png" width="600">
</div>

Note: The ECE594 project is currently limited in scope to "Aim 1" above.

## Overview

This project investigates whether state-estimation objectives for articulated bodies induce structured neural representations analogous to grid codes in spatial navigation. We train recurrent networks on path integration for a robotic arm with configuration space Q = SO(3) × SO(3), analyze the learned representations, and evaluate their utility for downstream reinforcement learning.

**Project Stages:**
1. **Body-state estimation** (Team Estimation): Train RNN to perform path integration on joint angular velocities
2. **Representation analysis** (Team Interpretation): Visualize and analyze latent structure
3. **RL evaluation** (Team RL): Use learned representations as embeddings for Reacher-v5

## Getting Started

### 1. Install Poetry

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Verify installation
poetry --version
```

### 2. Clone and Setup

```bash
git clone <repository-url>
cd articulated

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### 3. Verify Installation

```bash
pytest
black --check .
ruff check .
```

## Project Structure

```
articulated/
├── articulated/
│   ├── estimation/          # Team Estimation
│   │   ├── datamodule.py    # Trajectory data generation
│   │   ├── model.py         # RNN architectures
│   │   └── train.py         # Training script
│   │
│   ├── analysis/            # Team Interpretation
│   │   ├── dimensionality.py  # PCA
│   │   ├── tuning.py        # Tuning curve analysis
│   │   └── visualization.py # Plotting utilities
│   │
│   ├── rl/                  # Team RL
│   │   ├── environment.py   # Reacher-v5 wrappers
│   │   ├── agent.py         # RL agents
│   │   └── train.py         # Training script
│   │
│   ├── shared/              # Shared utilities
│   │   └── robot_arm.py     # Kinematics on SO(3) × SO(3)
│   │
│   ├── configs/             # Configuration files
│   │   ├── estimation/      # Team Estimation configs
│   │   └── rl/              # Team RL configs
│   │
│   └── notebooks/           # Analysis notebooks
│
├── tests/
├── docs/
└── pyproject.toml
```

---

## Team Guides

### Team Estimation

**Goal:** Train RNN to perform path integration on SO(3) × SO(3).

**Key files:**
- `articulated/estimation/datamodule.py`: Implement trajectory generation (inputs,targets)
- `articulated/estimation/model.py`: Define RNN architecture
- `articulated/estimation/train.py`: Training script

**Key TODOs:**
1. Implement proper SO(3) × SO(3) trajectory generation in `_generate_single_trajectory()`
2. Implement proper "place cell" targets on SO(3) × SO(3)
3. Experiment with RNN vs LSTM vs GRU architectures

**Run training:**
```bash
python -m articulated.estimation.train --config articulated/configs/estimation/rnn.yaml
```

**Interface with other teams:**
- The `StateEstimationModel.get_embedding()` will be used by Team RL
- The `StateEstimationModel.get_hidden_states()` will be used by Team Interpretation

**Resources:**
- [Banino et al. 2018 - Grid cells in RNNs](https://www.nature.com/articles/s41586-018-0102-6)
- [Path integration in neural networks](https://ieeexplore.ieee.org/abstract/document/8929622)

---

### Team Interpretation

**Goal:** Analyze how the RNN encodes joint position using dimensionality reduction and tuning curves.

**Key files:**
- `articulated/analysis/dimensionality.py`: PCA
- `articulated/analysis/tuning.py`: Tuning curve analysis
- `articulated/analysis/visualization.py`: Plotting

**Key TODOs:**
1. Apply PCA to hidden states from trained models
2. Compute tuning curves w.r.t. configuration variables (joint angles)
3. Check if representations have structure (e.g. grid-like, etc.).

**Getting hidden states:**
```python
from articulated.estimation.model import StateEstimationModel

model = StateEstimationModel.load_for_embedding("checkpoints/estimation/best.ckpt")
hidden_states = model.get_hidden_states(velocity_trajectory)
```

**Resources:**
- [Banino et al. 2018](https://www.nature.com/articles/s41586-018-0102-6)
- [Mechanistic interpretability](https://www.sciencedirect.com/science/article/pii/S0896627322009072)

---

### Team RL

**Goal:** Compare RL performance using raw observations vs learned embeddings on Reacher-v5.

**Key files:**
- `articulated/rl/environment.py` - Reacher wrappers
- `articulated/rl/agent.py` - RL agents (PPO, SAC)
- `articulated/rl/train.py` - Training script

**Key TODOs:**
1. Understand Reacher-v5 observation space
2. Implement `_extract_angular_velocities()` in environment wrapper
3. Run baseline (raw obs) vs embedded experiments

**Reacher-v5 observation space (Gymnasium 1.2.3 / MuJoCo 3.4.0):**
- `observation_space`: `Box(-inf, inf, (10,), float64)`
- `action_space`: `Box(-1.0, 1.0, (2,), float32)`
- Observation vector layout (indices):
  - `0:2` -> `cos(theta1), cos(theta2)`
  - `2:4` -> `sin(theta1), sin(theta2)`
  - `4:6` -> target position in plane (`qpos[2:4]`)
  - `6:8` -> joint angular velocities (`qvel[0:2]`)
  - `8:10` -> fingertip-to-target vector (x, y)

**Run training:**
```bash
# Baseline (raw observations)
python -m articulated.rl.train --config articulated/configs/rl/baseline.yaml

# Tuned baselines (more steps, normalization, TensorBoard logging)
python -m articulated.rl.train --config articulated/configs/rl/baseline_tuned.yaml
python -m articulated.rl.train --config articulated/configs/rl/baseline_tuned2.yaml

# With embeddings (requires trained estimation model)
python -m articulated.rl.train --config articulated/configs/rl/embedded.yaml
```

**Team RL logging & visualization:**
- TensorBoard: set `tensorboard_log` in config and run `tensorboard --logdir logs/rl`
- Render a policy: `python -m articulated.rl.render --checkpoint <path>.zip --out logs/rl/reacher.mp4`

**Resources:**
- [Gymnasium Reacher-v5](https://gymnasium.farama.org/environments/mujoco/reacher/)
- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/)
- [RL tutorial video](https://www.youtube.com/watch?v=zP2MTeqCk5k)

---

## Development

### Code Style

```bash
black .           # Format
ruff check .      # Lint
ruff check --fix  # Auto-fix
mypy articulated/ # Type check
```

### Running Tests

```bash
pytest
pytest -v
pytest tests/test_models.py
```

## Contributors

**Project Leads:**
- Mathilde Papillon (papillon@ucsb.edu)
- Francisco Acosta (facosta@ucsb.edu)
Geometric Intelligence Lab @ UCSB

**ECE594 Project Team:**
- **Team Estimation:** Awsaf Rahman (lead), Siheng Wang
- **Team Interpretation:** Rohan Koshy, Siheng Wang
- **Team RL:** Pushpita Joardar, Hun Tae Kim
