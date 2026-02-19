"""Data module for state estimation training.

Tasks:
1. Generate or load trajectories of joint angular velocities
2. Compute ground-truth joint configurations via integration
3. Create place cell targets tiled over SO(3) x SO(3)
"""

from typing import Optional

import lightning as L
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from articulated.shared.robot_arm import RobotArm2DKinematics, RobotArmKinematics


class EstimationDataModule(L.LightningDataModule):
    """DataModule for state estimation training.

    Generates trajectories on SO(3) x SO(3) configuration space.
    Input: angular velocities (6D per timestep)
    Target: place cell activations encoding current configuration
    """

    def __init__(
        self,
        batch_size: int = 64,
        seq_length: int = 100,
        n_trajectories_train: int = 1000,
        n_trajectories_val: int = 100,
        n_place_cells: int = 64,
        dt: float = 0.01,
        seed: Optional[int] = None,
        velocity_sigma: float = 1.0,
        velocity_theta: float = 2.0,
        provide_init_pos: bool = True,
        place_cell_kappa: float = 5.0,
        data_path: Optional[str] = None,
        num_workers: int = 4,
        manifold: str = "so3",
    ):
        """Initialize the data module.

        Args:
            batch_size: Batch size for data loaders.
            seq_length: Length of each trajectory sequence.
            n_trajectories_train: Number of training trajectories.
            n_trajectories_val: Number of validation trajectories.
            n_place_cells: Total number of place cells (split equally between
                joints, so must be even). Each joint gets n_place_cells // 2
                cells with independent softmax distributions.
            dt: Time step for integration.
            seed: Random seed for reproducibility.
            velocity_sigma: Steady-state std of OU angular velocity process.
            velocity_theta: Mean-reversion rate of OU angular velocity process.
            provide_init_pos: Whether to include initial position place cell
                activations in the dataset (needed for path integration).
            place_cell_kappa: Concentration parameter for von Mises-Fisher kernel.
                Higher = more peaked. kappa=5 gives ~4-5 effective cells per joint.
            data_path: Path to pre-generated .pt data file. If provided, loads
                data from disk instead of generating. Use scripts/generate_data.py
                to create these files.
            num_workers: Number of DataLoader workers.
            manifold: Configuration space manifold. "so3" for SO(3)xSO(3) (6D),
                "so2" for SO(2)xSO(2) (2D torus).
        """
        super().__init__()
        self.save_hyperparameters()

        assert manifold in ("so3", "so2"), f"Unknown manifold: {manifold}"
        assert (
            n_place_cells % 2 == 0
        ), "n_place_cells must be even (split between 2 joints)"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_trajectories_train = n_trajectories_train
        self.n_trajectories_val = n_trajectories_val
        self.n_place_cells = n_place_cells
        self.n_cells_per_joint = n_place_cells // 2
        self.dt = dt
        self.seed = seed
        self.velocity_sigma = velocity_sigma
        self.velocity_theta = velocity_theta
        self.provide_init_pos = provide_init_pos
        self.place_cell_kappa = place_cell_kappa
        self.data_path = data_path
        self.num_workers = num_workers
        self.manifold = manifold

        # Velocity dimension and init_pos dimension depend on manifold
        self.vel_dim = 2 if manifold == "so2" else 6
        # SO(2): raw angles encoded as (cos θ1, sin θ1, cos θ2, sin θ2) → 4D
        # SO(3): place cell activations → output_size
        self.init_pos_dim = 4 if manifold == "so2" else n_place_cells

        if manifold == "so2":
            self.kinematics = RobotArm2DKinematics()
        else:
            self.kinematics = RobotArmKinematics()
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None

        # Place cell centers - to be initialized (separate per joint)
        self.place_cell_centers: Optional[list] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Generate or load trajectory datasets."""
        if stage == "fit" or stage is None:
            if self.data_path is not None:
                self._load_from_disk()
            else:
                self._generate_all_data()

    def _load_from_disk(self) -> None:
        """Load pre-generated data from a .pt file."""
        assert self.data_path is not None
        print(f"Loading data from {self.data_path}...")
        data = torch.load(self.data_path, weights_only=True)

        # Init position key depends on manifold
        init_key_train = (
            "train_init_angles" if self.manifold == "so2" else "train_init_pcs"
        )
        init_key_val = "val_init_angles" if self.manifold == "so2" else "val_init_pcs"

        train_tensors = [data["train_velocities"], data["train_targets"]]
        if self.provide_init_pos and init_key_train in data:
            train_tensors.append(data[init_key_train])
        self.train_dataset = TensorDataset(*train_tensors)

        val_tensors = [data["val_velocities"], data["val_targets"]]
        if self.provide_init_pos and init_key_val in data:
            val_tensors.append(data[init_key_val])
        self.val_dataset = TensorDataset(*val_tensors)

        # Restore place cell centers for analysis
        if self.manifold == "so2":
            if "place_cell_angles1" in data and "place_cell_angles2" in data:
                self._centers_angles1 = data["place_cell_angles1"].numpy()
                self._centers_angles2 = data["place_cell_angles2"].numpy()
        else:
            if "place_cell_quats1" in data and "place_cell_quats2" in data:
                self._centers_R1 = Rotation.from_quat(data["place_cell_quats1"].numpy())
                self._centers_R2 = Rotation.from_quat(data["place_cell_quats2"].numpy())

        n_train = data["train_velocities"].shape[0]
        n_val = data["val_velocities"].shape[0]
        print(f"Loaded {n_train} train + {n_val} val trajectories ({self.manifold})")

    def _generate_all_data(self) -> None:
        """Generate all train/val data from scratch."""
        rng = np.random.default_rng(self.seed)

        # Initialize place cells
        self._initialize_place_cells(rng)

        # Generate training data
        train_velocities, train_targets, train_init = self._generate_trajectories(
            self.n_trajectories_train, rng
        )
        train_tensors = [
            torch.from_numpy(train_velocities).float(),
            torch.from_numpy(train_targets).float(),
        ]
        if self.provide_init_pos:
            train_tensors.append(torch.from_numpy(train_init).float())
        self.train_dataset = TensorDataset(*train_tensors)

        # Generate validation data
        val_velocities, val_targets, val_init = self._generate_trajectories(
            self.n_trajectories_val, rng
        )
        val_tensors = [
            torch.from_numpy(val_velocities).float(),
            torch.from_numpy(val_targets).float(),
        ]
        if self.provide_init_pos:
            val_tensors.append(torch.from_numpy(val_init).float())
        self.val_dataset = TensorDataset(*val_tensors)

    def _initialize_place_cells(self, rng: np.random.Generator) -> None:
        """Initialize place cell centers separately for each joint.

        For SO(3): random rotations on SO(3).
        For SO(2): random angles on [0, 2*pi).
        """
        n = self.n_cells_per_joint
        if self.manifold == "so2":
            self._centers_angles1 = rng.uniform(0, 2 * np.pi, size=n)
            self._centers_angles2 = rng.uniform(0, 2 * np.pi, size=n)
            self.place_cell_centers = [
                (self._centers_angles1[i], self._centers_angles2[i]) for i in range(n)
            ]
        else:
            self._centers_R1 = Rotation.random(
                n, random_state=int(rng.integers(0, 2**31))
            )
            self._centers_R2 = Rotation.random(
                n, random_state=int(rng.integers(0, 2**31))
            )
            self.place_cell_centers = [
                (self._centers_R1[i], self._centers_R2[i]) for i in range(n)
            ]

    def _generate_trajectories(
        self, n_trajectories: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random arm trajectories.

        Args:
            n_trajectories: Number of trajectories to generate.
            rng: Random number generator.

        Returns:
            Tuple of (velocities, place_cell_targets, init_pos).
            velocities: Shape (n_trajectories, seq_length, vel_dim).
            targets: Shape (n_trajectories, seq_length, n_place_cells).
            init_pos: Shape (n_trajectories, init_pos_dim).
        """
        velocities = np.zeros((n_trajectories, self.seq_length, self.vel_dim))
        targets = np.zeros((n_trajectories, self.seq_length, self.n_place_cells))
        init_pos = np.zeros((n_trajectories, self.init_pos_dim))

        for i in tqdm(range(n_trajectories), desc="Generating trajectories"):
            vel, target, ip = self._generate_single_trajectory(rng)
            velocities[i] = vel
            targets[i] = target
            init_pos[i] = ip

        return velocities, targets, init_pos

    def _generate_single_trajectory(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a single trajectory.

        Uses an Ornstein-Uhlenbeck process to generate smooth angular velocities,
        integrates on the configuration manifold, and computes place cell activations.

        Args:
            rng: Random number generator.

        Returns:
            Tuple of (velocities, place_cell_targets, init_pos).
            velocities: Shape (seq_length, vel_dim).
            targets: Shape (seq_length, n_place_cells).
            init_pos: Shape (init_pos_dim,).
        """
        config = self.kinematics.sample_random_configuration(rng)

        # Initial position encoding
        if self.manifold == "so2":
            # Raw angles → (cos θ1, sin θ1, cos θ2, sin θ2)
            theta1, theta2 = config
            init_pos = np.array(
                [
                    np.cos(theta1),
                    np.sin(theta1),
                    np.cos(theta2),
                    np.sin(theta2),
                ]
            )
        else:
            # Place cell activation at the starting position
            init_pos = self._compute_place_cell_activations(config)

        decay = np.exp(-self.velocity_theta * self.dt)
        noise_scale = self.velocity_sigma * np.sqrt(1.0 - decay**2)

        velocities = np.zeros((self.seq_length, self.vel_dim))
        targets = np.zeros((self.seq_length, self.n_place_cells))

        # Initialize angular velocity at zero
        omega = np.zeros(self.vel_dim)

        for t in range(self.seq_length):
            # OU update for angular velocity
            omega = decay * omega + noise_scale * rng.standard_normal(self.vel_dim)
            velocities[t] = omega

            # Integrate to get new configuration
            config = self.kinematics.integrate_velocity(config, omega, self.dt)

            # Compute place cell activations
            targets[t] = self._compute_place_cell_activations(config)

        return velocities, targets, init_pos

    def _compute_place_cell_activations(self, configuration: tuple) -> np.ndarray:
        """Compute place cell activations for a single configuration.

        Uses vMF-style kernel with separate softmax per joint.

        For SO(3): geodesic distance on rotation group.
        For SO(2): circular distance on the circle.

        Args:
            configuration: Current configuration. (Rotation, Rotation) for SO(3),
                or (float, float) for SO(2).

        Returns:
            Place cell activations of shape (n_place_cells,), which is the
            concatenation of two independent distributions of size
            n_cells_per_joint.
        """
        kappa = self.place_cell_kappa

        if self.manifold == "so2":
            theta1, theta2 = configuration
            # Circular distance: min(|delta|, 2pi - |delta|)
            TWO_PI = 2.0 * np.pi
            delta1 = np.abs(theta1 - self._centers_angles1)
            d1 = np.minimum(delta1, TWO_PI - delta1)
            delta2 = np.abs(theta2 - self._centers_angles2)
            d2 = np.minimum(delta2, TWO_PI - delta2)
        else:
            R1_cur, R2_cur = configuration
            d1 = (R1_cur.inv() * self._centers_R1).magnitude()
            d2 = (R2_cur.inv() * self._centers_R2).magnitude()

        # vMF kernel + softmax per joint (numerically stable)
        logits1 = kappa * np.cos(d1)
        logits1 -= logits1.max()
        exp1 = np.exp(logits1)
        pc1 = exp1 / exp1.sum()

        logits2 = kappa * np.cos(d2)
        logits2 -= logits2.max()
        exp2 = np.exp(logits2)
        pc2 = exp2 / exp2.sum()

        return np.concatenate([pc1, pc2])

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
