"""Environment wrappers for RL experiments.

Tasks:
1. Set up Reacher-v5 environment
2. Create wrapper that integrates embeddings from Team Estimation
3. Provide both raw and embedded observation modes
"""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch


class ReacherWithEmbedding(gym.Wrapper):
    """Wrapper for Reacher-v5 that can use learned embeddings.

    This wrapper allows switching between:
    - Raw observations (default Reacher-v5)
    - Embedded observations (using Team Estimation's model)
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        use_embedding: bool = False,
        history_length: int = 10,
        render_mode: Optional[str] = None,
    ):
        """Initialize the environment wrapper.

        Args:
            embedding_model: Trained StateEstimationModel from Team Estimation.
            use_embedding: Whether to use embeddings or raw observations.
            history_length: Number of timesteps of history for embedding.
            render_mode: Gymnasium render mode.
        """
        env = gym.make("Reacher-v5", render_mode=render_mode)
        super().__init__(env)

        self.embedding_model = embedding_model
        self.use_embedding = use_embedding
        self.history_length = history_length

        # Observation history buffer
        self._obs_history: list[np.ndarray] = []

        # Update observation space if using embeddings
        if use_embedding and embedding_model is not None:
            embedding_dim = embedding_model.hidden_size
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(embedding_dim,),
                dtype=np.float32,
            )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs_history = []

        if self.use_embedding:
            obs = self._get_embedded_observation(obs)

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.use_embedding:
            obs = self._get_embedded_observation(obs)

        return obs, reward, terminated, truncated, info

    def _get_embedded_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        """Convert raw observation to embedding.

        TODO: Implement this method:
        1. Extract angular velocities from raw observation
        2. Maintain history buffer
        3. Pass through embedding model
        4. Return embedding

        Args:
            raw_obs: Raw observation from Reacher-v5.

        Returns:
            Embedded observation.
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model is required when use_embedding is True.")

        angular_velocities = self._extract_angular_velocities(raw_obs)
        angular_velocities = angular_velocities.astype(np.float32, copy=False)
        self._obs_history.append(angular_velocities)

        if len(self._obs_history) > self.history_length:
            self._obs_history = self._obs_history[-self.history_length :]

        vel_dim = angular_velocities.shape[0]
        if len(self._obs_history) < self.history_length:
            pad = np.zeros(
                (self.history_length - len(self._obs_history), vel_dim),
                dtype=np.float32,
            )
            history = np.vstack([pad, np.stack(self._obs_history)])
        else:
            history = np.stack(self._obs_history)

        history_tensor = torch.from_numpy(history).unsqueeze(0)
        if hasattr(self.embedding_model, "parameters"):
            try:
                device = next(self.embedding_model.parameters()).device
                history_tensor = history_tensor.to(device)
            except StopIteration:
                pass

        with torch.no_grad():
            embedding = self.embedding_model.get_embedding(history_tensor)

        return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    def _extract_angular_velocities(self, obs: np.ndarray) -> np.ndarray:
        """Extract angular velocities from Reacher observation.

        TODO: Implement based on Reacher-v5 observation structure.
        See: https://gymnasium.farama.org/environments/mujoco/reacher/

        Args:
            obs: Raw observation.

        Returns:
            Angular velocities.
        """
        if obs.shape[0] < 8:
            raise ValueError(
                "Expected Reacher-v5 observation with at least 8 elements."
            )
        return np.asarray(obs[6:8], dtype=np.float32)
