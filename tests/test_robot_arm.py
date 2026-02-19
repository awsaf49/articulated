"""Tests for robot arm kinematics."""

import numpy as np
from scipy.spatial.transform import Rotation

from articulated.shared.robot_arm import RobotArm2DKinematics, RobotArmKinematics


class TestIntegrateVelocity:
    """Tests for integrate_velocity."""

    def test_zero_velocity_preserves_orientation(self):
        """Zero angular velocity should not change orientation."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        config = kin.sample_random_configuration(rng)
        omega = np.zeros(6)

        new_config = kin.integrate_velocity(config, omega, dt=0.01)

        # Relative rotation should be identity
        d1 = (config[0].inv() * new_config[0]).magnitude()
        d2 = (config[1].inv() * new_config[1]).magnitude()
        assert d1 < 1e-10
        assert d2 < 1e-10

    def test_output_is_valid_rotations(self):
        """Output should be a pair of valid Rotation objects."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        config = kin.sample_random_configuration(rng)
        omega = rng.standard_normal(6)

        new_config = kin.integrate_velocity(config, omega, dt=0.01)

        assert isinstance(new_config[0], Rotation)
        assert isinstance(new_config[1], Rotation)
        # Rotation matrices should be orthogonal
        R1 = new_config[0].as_matrix()
        np.testing.assert_allclose(R1 @ R1.T, np.eye(3), atol=1e-10)

    def test_nonzero_velocity_changes_orientation(self):
        """Nonzero velocity should change the orientation."""
        kin = RobotArmKinematics()
        config = (Rotation.identity(), Rotation.identity())
        omega = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        new_config = kin.integrate_velocity(config, omega, dt=0.1)

        d1 = (config[0].inv() * new_config[0]).magnitude()
        d2 = (config[1].inv() * new_config[1]).magnitude()
        assert d1 > 0.05
        assert d2 > 0.05


class TestGeodesicDistance:
    """Tests for geodesic_distance."""

    def test_distance_to_self_is_zero(self):
        """Distance from a config to itself should be zero."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        config = kin.sample_random_configuration(rng)

        d = kin.geodesic_distance(config, config)
        assert abs(d) < 1e-10

    def test_symmetry(self):
        """Distance should be symmetric: d(a, b) == d(b, a)."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        c1 = kin.sample_random_configuration(rng)
        c2 = kin.sample_random_configuration(rng)

        d12 = kin.geodesic_distance(c1, c2)
        d21 = kin.geodesic_distance(c2, c1)
        assert abs(d12 - d21) < 1e-10

    def test_triangle_inequality(self):
        """Triangle inequality: d(a, c) <= d(a, b) + d(b, c)."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        c1 = kin.sample_random_configuration(rng)
        c2 = kin.sample_random_configuration(rng)
        c3 = kin.sample_random_configuration(rng)

        d12 = kin.geodesic_distance(c1, c2)
        d23 = kin.geodesic_distance(c2, c3)
        d13 = kin.geodesic_distance(c1, c3)

        assert d13 <= d12 + d23 + 1e-10

    def test_nonnegative(self):
        """Distance should always be non-negative."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        c1 = kin.sample_random_configuration(rng)
        c2 = kin.sample_random_configuration(rng)

        assert kin.geodesic_distance(c1, c2) >= 0.0


class TestForwardKinematics:
    """Tests for forward_kinematics."""

    def test_identity_rotations(self):
        """Both joints at identity: end-effector at (L1+L2, 0, 0)."""
        kin = RobotArmKinematics(link_lengths=(1.0, 1.0))
        config = (Rotation.identity(), Rotation.identity())

        pos = kin.forward_kinematics(config)
        np.testing.assert_allclose(pos, [2.0, 0.0, 0.0], atol=1e-10)

    def test_custom_link_lengths(self):
        """Different link lengths with identity rotations."""
        kin = RobotArmKinematics(link_lengths=(2.0, 3.0))
        config = (Rotation.identity(), Rotation.identity())

        pos = kin.forward_kinematics(config)
        np.testing.assert_allclose(pos, [5.0, 0.0, 0.0], atol=1e-10)

    def test_output_shape(self):
        """Forward kinematics should return a 3D position."""
        kin = RobotArmKinematics()
        rng = np.random.default_rng(42)
        config = kin.sample_random_configuration(rng)

        pos = kin.forward_kinematics(config)
        assert pos.shape == (3,)

    def test_end_effector_bounded(self):
        """End-effector should be within L1 + L2 of origin."""
        kin = RobotArmKinematics(link_lengths=(1.0, 1.0))
        rng = np.random.default_rng(42)

        for _ in range(10):
            config = kin.sample_random_configuration(rng)
            pos = kin.forward_kinematics(config)
            dist = np.linalg.norm(pos)
            assert dist <= 2.0 + 1e-10


class TestRobotArm2DIntegrateVelocity:
    """Tests for RobotArm2DKinematics.integrate_velocity."""

    def test_zero_velocity_preserves_angles(self):
        """Zero angular velocity should not change angles."""
        kin = RobotArm2DKinematics()
        config = (1.0, 2.0)
        omega = np.zeros(2)

        new_config = kin.integrate_velocity(config, omega, dt=0.01)

        assert abs(new_config[0] - config[0]) < 1e-10
        assert abs(new_config[1] - config[1]) < 1e-10

    def test_wrapping(self):
        """Angles should wrap around [0, 2*pi)."""
        kin = RobotArm2DKinematics()
        config = (6.0, 0.1)  # close to 2*pi
        omega = np.array([10.0, -20.0])  # large velocity

        new_config = kin.integrate_velocity(config, omega, dt=0.1)

        assert 0 <= new_config[0] < 2 * np.pi
        assert 0 <= new_config[1] < 2 * np.pi

    def test_nonzero_velocity_changes_angles(self):
        """Nonzero velocity should change the angles."""
        kin = RobotArm2DKinematics()
        config = (0.0, 0.0)
        omega = np.array([1.0, 2.0])

        new_config = kin.integrate_velocity(config, omega, dt=0.1)

        assert abs(new_config[0] - 0.1) < 1e-10
        assert abs(new_config[1] - 0.2) < 1e-10


class TestRobotArm2DGeodesicDistance:
    """Tests for RobotArm2DKinematics.geodesic_distance."""

    def test_distance_to_self_is_zero(self):
        kin = RobotArm2DKinematics()
        config = (1.5, 3.0)
        assert abs(kin.geodesic_distance(config, config)) < 1e-10

    def test_symmetry(self):
        kin = RobotArm2DKinematics()
        c1 = (0.5, 1.0)
        c2 = (2.0, 4.0)
        d12 = kin.geodesic_distance(c1, c2)
        d21 = kin.geodesic_distance(c2, c1)
        assert abs(d12 - d21) < 1e-10

    def test_circular_distance(self):
        """Distance should use circular metric (shorter arc)."""
        kin = RobotArm2DKinematics()
        # theta1: 0.1 vs 6.1 → distance is ~0.1+0.083 = ~0.183 (wraps around)
        c1 = (0.1, 0.0)
        c2 = (2 * np.pi - 0.1, 0.0)
        d = kin.geodesic_distance(c1, c2)
        # Circular distance should be 0.2, not ~6.08
        assert abs(d - 0.2) < 1e-10

    def test_nonnegative(self):
        kin = RobotArm2DKinematics()
        rng = np.random.default_rng(42)
        c1 = kin.sample_random_configuration(rng)
        c2 = kin.sample_random_configuration(rng)
        assert kin.geodesic_distance(c1, c2) >= 0.0

    def test_triangle_inequality(self):
        kin = RobotArm2DKinematics()
        rng = np.random.default_rng(42)
        c1 = kin.sample_random_configuration(rng)
        c2 = kin.sample_random_configuration(rng)
        c3 = kin.sample_random_configuration(rng)
        d12 = kin.geodesic_distance(c1, c2)
        d23 = kin.geodesic_distance(c2, c3)
        d13 = kin.geodesic_distance(c1, c3)
        assert d13 <= d12 + d23 + 1e-10


class TestRobotArm2DForwardKinematics:
    """Tests for RobotArm2DKinematics.forward_kinematics."""

    def test_zero_angles(self):
        """Both joints at zero: end-effector at (L1+L2, 0)."""
        kin = RobotArm2DKinematics(link_lengths=(1.0, 1.0))
        pos = kin.forward_kinematics((0.0, 0.0))
        np.testing.assert_allclose(pos, [2.0, 0.0], atol=1e-10)

    def test_output_shape(self):
        """Forward kinematics should return a 2D position."""
        kin = RobotArm2DKinematics()
        rng = np.random.default_rng(42)
        config = kin.sample_random_configuration(rng)
        pos = kin.forward_kinematics(config)
        assert pos.shape == (2,)

    def test_end_effector_bounded(self):
        """End-effector should be within L1 + L2 of origin."""
        kin = RobotArm2DKinematics(link_lengths=(1.0, 1.0))
        rng = np.random.default_rng(42)
        for _ in range(10):
            config = kin.sample_random_configuration(rng)
            pos = kin.forward_kinematics(config)
            dist = np.linalg.norm(pos)
            assert dist <= 2.0 + 1e-10

    def test_custom_link_lengths(self):
        """Different link lengths with zero angles."""
        kin = RobotArm2DKinematics(link_lengths=(2.0, 3.0))
        pos = kin.forward_kinematics((0.0, 0.0))
        np.testing.assert_allclose(pos, [5.0, 0.0], atol=1e-10)

    def test_right_angle(self):
        """First joint at pi/2, second at 0: end-effector at (0, L1+L2)."""
        kin = RobotArm2DKinematics(link_lengths=(1.0, 1.0))
        pos = kin.forward_kinematics((np.pi / 2, 0.0))
        np.testing.assert_allclose(pos, [0.0, 2.0], atol=1e-10)


class TestRobotArm2DSampleConfig:
    """Tests for RobotArm2DKinematics.sample_random_configuration."""

    def test_range(self):
        """Random config should be in [0, 2*pi)."""
        kin = RobotArm2DKinematics()
        rng = np.random.default_rng(42)
        for _ in range(20):
            config = kin.sample_random_configuration(rng)
            assert 0 <= config[0] < 2 * np.pi
            assert 0 <= config[1] < 2 * np.pi

    def test_returns_tuple_of_floats(self):
        kin = RobotArm2DKinematics()
        config = kin.sample_random_configuration(np.random.default_rng(42))
        assert isinstance(config[0], float)
        assert isinstance(config[1], float)
