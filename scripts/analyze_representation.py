"""Analyze learned representations from trained state estimation models.

Generates PCA, t-SNE, and tuning curve plots on validation data.
For SO(2) models, generates direct (θ1, θ2) heatmaps — no PCA needed.

Usage:
    python scripts/analyze_representation.py logs/estimation/rnn/checkpoints/last-v3.ckpt
    python scripts/analyze_representation.py path/to/checkpoint.ckpt --manifold so2
    python scripts/analyze_representation.py path/to/checkpoint.ckpt --n_traj 500
    python scripts/analyze_representation.py path/to/checkpoint.ckpt --help
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from articulated.estimation.model import StateEstimationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze learned representations")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to trajectories.pt (auto from manifold if not set)",
    )
    parser.add_argument(
        "--manifold",
        type=str,
        default="so3",
        choices=["so3", "so2"],
        help="Configuration manifold",
    )
    parser.add_argument(
        "--n_traj", type=int, default=200, help="Number of val trajectories to use"
    )
    parser.add_argument(
        "--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path (default: next to ckpt)"
    )
    return parser.parse_args()


def get_joint_angles_so3(data, n_traj, seq_len):
    """Reconstruct true continuous joint rotations by replaying velocity integration."""
    from articulated.shared.robot_arm import RobotArmKinematics

    kin = RobotArmKinematics()
    vel = data["val_velocities"][:n_traj].numpy()
    dt = 0.01

    centers_R1 = Rotation.from_quat(data["place_cell_quats1"].numpy())
    centers_R2 = Rotation.from_quat(data["place_cell_quats2"].numpy())
    n_per_joint = len(centers_R1)

    init_pcs = data["val_init_pcs"][:n_traj].numpy()

    targets = data["val_targets"][:n_traj].numpy()
    j1_tgt = targets[:, :, :n_per_joint].reshape(-1, n_per_joint)
    j2_tgt = targets[:, :, n_per_joint:].reshape(-1, n_per_joint)
    j1_cell = j1_tgt.argmax(axis=1)
    j2_cell = j2_tgt.argmax(axis=1)

    j1_rotvec = np.zeros((n_traj, seq_len, 3))
    j2_rotvec = np.zeros((n_traj, seq_len, 3))

    for i in range(n_traj):
        init_j1 = init_pcs[i, :n_per_joint]
        init_j2 = init_pcs[i, n_per_joint:]
        R1_init = centers_R1[int(init_j1.argmax())]
        R2_init = centers_R2[int(init_j2.argmax())]
        config = (R1_init, R2_init)

        for t in range(seq_len):
            omega = vel[i, t]
            config = kin.integrate_velocity(config, omega, dt)
            j1_rotvec[i, t] = config[0].as_rotvec()
            j2_rotvec[i, t] = config[1].as_rotvec()

    j1_rotvec = j1_rotvec.reshape(-1, 3)
    j2_rotvec = j2_rotvec.reshape(-1, 3)

    return j1_rotvec, j2_rotvec, j1_cell, j2_cell


def get_joint_angles_so2(data, n_traj, seq_len):
    """Reconstruct true joint angles by replaying velocity integration on SO(2)."""
    from articulated.shared.robot_arm import RobotArm2DKinematics

    kin = RobotArm2DKinematics()
    vel = data["val_velocities"][:n_traj].numpy()
    dt = 0.01

    # Recover initial angles from (cos θ, sin θ) encoding
    init_encoded = data["val_init_angles"][:n_traj].numpy()  # (n_traj, 4)

    n_per_joint = len(data["place_cell_angles1"])
    targets = data["val_targets"][:n_traj].numpy()
    j1_tgt = targets[:, :, :n_per_joint].reshape(-1, n_per_joint)
    j2_tgt = targets[:, :, n_per_joint:].reshape(-1, n_per_joint)
    j1_cell = j1_tgt.argmax(axis=1)
    j2_cell = j2_tgt.argmax(axis=1)

    j1_angles = np.zeros((n_traj, seq_len))
    j2_angles = np.zeros((n_traj, seq_len))

    for i in range(n_traj):
        theta1 = np.arctan2(init_encoded[i, 1], init_encoded[i, 0]) % (2 * np.pi)
        theta2 = np.arctan2(init_encoded[i, 3], init_encoded[i, 2]) % (2 * np.pi)
        config = (float(theta1), float(theta2))

        for t in range(seq_len):
            omega = vel[i, t]
            config = kin.integrate_velocity(config, omega, dt)
            j1_angles[i, t] = config[0]
            j2_angles[i, t] = config[1]

    j1_angles = j1_angles.reshape(-1)
    j2_angles = j2_angles.reshape(-1)

    return j1_angles, j2_angles, j1_cell, j2_cell


def find_selective_neurons(hidden_flat, angles, top_k=3):
    """Find neurons most correlated with a joint angle variable."""
    correlations = np.array(
        [
            np.abs(np.corrcoef(hidden_flat[:, i], angles)[0, 1])
            for i in range(hidden_flat.shape[1])
        ]
    )
    correlations = np.nan_to_num(correlations)
    top_idx = np.argsort(correlations)[::-1][:top_k]
    return top_idx, correlations[top_idx]


def _plot_tuning_curve(ax, activations, angles, title, xlabel, color):
    """Plot binned tuning curve with scatter and confidence band."""
    n_bins = 50
    bins = np.linspace(angles.min(), angles.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = np.zeros(n_bins)
    bin_stds = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (angles >= bins[b]) & (angles < bins[b + 1])
        if mask.sum() > 0:
            bin_means[b] = activations[mask].mean()
            bin_stds[b] = activations[mask].std()

    ax.fill_between(
        bin_centers, bin_means - bin_stds, bin_means + bin_stds, alpha=0.2, color=color
    )
    ax.plot(bin_centers, bin_means, "-", linewidth=2, color=color)

    n_show = min(2000, len(angles))
    idx = np.random.default_rng(0).choice(len(angles), n_show, replace=False)
    ax.scatter(angles[idx], activations[idx], s=1, alpha=0.08, color="gray")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Activation")
    ax.set_title(title)


def analyze_so2(args, model, data, n_traj, seq_len, hidden_size):
    """SO(2)-specific analysis with direct (θ1, θ2) heatmaps."""
    val_vel = data["val_velocities"][:n_traj]
    val_init = data["val_init_angles"][:n_traj]

    print(f"Running inference on {n_traj} trajectories...")
    with torch.no_grad():
        h0 = model._encode_init_pos(val_init)
        _, hidden_states = model(val_vel, hidden=h0)

    hidden_flat = hidden_states.numpy().reshape(-1, hidden_size)
    n_points = hidden_flat.shape[0]
    print(f"Hidden states: {n_points} points x {hidden_size} dims")

    j1_angles, j2_angles, j1_cell, j2_cell = get_joint_angles_so2(data, n_traj, seq_len)

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=min(50, hidden_size))
    hidden_pca = pca.fit_transform(hidden_flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)

    # Find most selective neurons for θ1 and θ2
    top_j1, corr_j1 = find_selective_neurons(hidden_flat, j1_angles, top_k=3)
    top_j2, corr_j2 = find_selective_neurons(hidden_flat, j2_angles, top_k=3)
    print(f"Most θ1-selective neurons: {top_j1} (r={corr_j1})")
    print(f"Most θ2-selective neurons: {top_j2} (r={corr_j2})")

    # ===== PLOT =====
    model_type = model.hparams.get("model_type", "unknown").upper()
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle(
        f"{model_type} Representation Analysis — SO(2)×SO(2) "
        f"(vMF, kappa=5, hidden={hidden_size})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # ── Row 1: PCA spectrum + summary ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(
        range(min(20, len(pca.explained_variance_ratio_))),
        pca.explained_variance_ratio_[:20],
        color="steelblue",
    )
    ax.set_xlabel("PC Index")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Spectrum")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(range(1, min(21, len(cumvar) + 1)), cumvar[:20], "o-", color="steelblue")
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95%")
    ax.axvline(x=n95, color="red", linestyle=":", alpha=0.7)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative Variance")
    ax.set_title(f"Cumulative Variance\n{n95} PCs for 95%")
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.text(
        0.5,
        0.5,
        f"Model: {model_type} (hidden={hidden_size})\n"
        f"Manifold: SO(2)×SO(2)\n"
        f"Output: {model.output_size} ({model.cells_per_joint}/joint)\n"
        f"Dropout: {model.hparams.get('dropout', '?')}\n\n"
        f"PCs for 95%: {n95}\n"
        f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%\n"
        f"Top-3: {cumvar[2]*100:.1f}%\n\n"
        f"θ1-selective: {top_j1}\n"
        f"  r = [{', '.join(f'{c:.3f}' for c in corr_j1)}]\n"
        f"θ2-selective: {top_j2}\n"
        f"  r = [{', '.join(f'{c:.3f}' for c in corr_j2)}]",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("Summary")
    ax.axis("off")

    # ── Row 2: PCA colored by θ1, θ2, and cell ID ───────────────────────────
    n_plot = min(3000, n_points)
    plot_idx = np.random.default_rng(0).choice(n_points, n_plot, replace=False)

    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(
        hidden_pca[plot_idx, 0],
        hidden_pca[plot_idx, 1],
        c=j1_angles[plot_idx],
        cmap="hsv",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="θ1")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PC1 vs PC2\nColored by θ1")

    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(
        hidden_pca[plot_idx, 0],
        hidden_pca[plot_idx, 1],
        c=j2_angles[plot_idx],
        cmap="hsv",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="θ2")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PC1 vs PC2\nColored by θ2")

    ax = fig.add_subplot(gs[1, 2])
    sc = ax.scatter(
        hidden_pca[plot_idx, 0],
        hidden_pca[plot_idx, 1],
        c=j1_cell[plot_idx],
        cmap="tab20",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Dominant cell ID")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PC1 vs PC2\nColored by cell ID")

    # ── Row 3: Direct (θ1, θ2) heatmaps — θ1-selective neurons ─────────────
    n_bins = 40
    for i in range(3):
        neuron_idx = top_j1[i]
        ax = fig.add_subplot(gs[2, i])
        _plot_torus_heatmap(
            ax,
            j1_angles,
            j2_angles,
            hidden_flat[:, neuron_idx],
            n_bins,
            f"Neuron {neuron_idx} — θ1-selective (r={corr_j1[i]:.3f})",
        )

    # ── Row 4: Direct (θ1, θ2) heatmaps — θ2-selective neurons ──────────
    for i in range(3):
        neuron_idx = top_j2[i]
        ax = fig.add_subplot(gs[3, i])
        _plot_torus_heatmap(
            ax,
            j1_angles,
            j2_angles,
            hidden_flat[:, neuron_idx],
            n_bins,
            f"Neuron {neuron_idx} — θ2-selective (r={corr_j2[i]:.3f})",
        )

    # Save
    if args.output:
        output_path = args.output
    else:
        output_path = str(
            Path(args.checkpoint).parent.parent / "representation_analysis_so2.png"
        )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


def _plot_torus_heatmap(ax, theta1, theta2, activations, n_bins, title):
    """Plot neuron activation as a heatmap on (θ1, θ2) torus."""
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    heatmap = np.full((n_bins, n_bins), np.nan)

    for i in range(n_bins):
        for j in range(n_bins):
            mask = (
                (theta1 >= bins[i])
                & (theta1 < bins[i + 1])
                & (theta2 >= bins[j])
                & (theta2 < bins[j + 1])
            )
            if mask.sum() > 0:
                heatmap[j, i] = activations[mask].mean()

    im = ax.imshow(
        heatmap,
        extent=[0, 2 * np.pi, 0, 2 * np.pi],
        origin="lower",
        aspect="equal",
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Mean activation")
    ax.set_xlabel("θ1")
    ax.set_ylabel("θ2")
    ax.set_title(title)
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", "π", "2π"])
    ax.set_yticks([0, np.pi, 2 * np.pi])
    ax.set_yticklabels(["0", "π", "2π"])


def analyze_so3(args, model, data, n_traj, seq_len, hidden_size):
    """SO(3)-specific analysis (original code path)."""
    val_vel = data["val_velocities"][:n_traj]
    val_init = data["val_init_pcs"][:n_traj]

    print(f"Running inference on {n_traj} trajectories...")
    with torch.no_grad():
        h0 = model._encode_init_pos(val_init)
        _, hidden_states = model(val_vel, hidden=h0)

    hidden_flat = hidden_states.numpy().reshape(-1, hidden_size)
    n_points = hidden_flat.shape[0]
    print(f"Hidden states: {n_points} points x {hidden_size} dims")

    j1_rotvec, j2_rotvec, j1_cell, j2_cell = get_joint_angles_so3(data, n_traj, seq_len)
    axis_labels = ["x", "y", "z"]

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=min(50, hidden_size))
    hidden_pca = pca.fit_transform(hidden_flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)

    # t-SNE
    n_tsne = min(5000, n_points)
    tsne_idx = np.random.default_rng(42).choice(n_points, n_tsne, replace=False)
    print(f"Running t-SNE on {n_tsne} points (perplexity={args.tsne_perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, random_state=42)
    hidden_tsne = tsne.fit_transform(hidden_flat[tsne_idx])

    # Find most selective neurons
    best_j1_neurons, best_j1_corr, best_j1_comp = [], [], []
    for comp in range(3):
        idx, corr = find_selective_neurons(hidden_flat, j1_rotvec[:, comp], top_k=1)
        best_j1_neurons.append(idx[0])
        best_j1_corr.append(corr[0])
        best_j1_comp.append(comp)
    best_j2_neurons, best_j2_corr, best_j2_comp = [], [], []
    for comp in range(3):
        idx, corr = find_selective_neurons(hidden_flat, j2_rotvec[:, comp], top_k=1)
        best_j2_neurons.append(idx[0])
        best_j2_corr.append(corr[0])
        best_j2_comp.append(comp)
    print(f"Most J1-selective neurons: {best_j1_neurons} (r={best_j1_corr})")
    print(f"Most J2-selective neurons: {best_j2_neurons} (r={best_j2_corr})")

    # ===== PLOT =====
    n_plot = min(3000, n_points)
    plot_idx = np.random.default_rng(0).choice(n_points, n_plot, replace=False)

    fig = plt.figure(figsize=(20, 28))
    gs = GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.3)

    model_type = model.hparams.get("model_type", "unknown").upper()
    fig.suptitle(
        f"{model_type} Representation Analysis "
        f"(vMF, kappa=5, hidden={hidden_size})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Row 1: PCA spectrum
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(
        range(min(20, len(pca.explained_variance_ratio_))),
        pca.explained_variance_ratio_[:20],
        color="steelblue",
    )
    ax.set_xlabel("PC Index")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Spectrum")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(range(1, min(21, len(cumvar) + 1)), cumvar[:20], "o-", color="steelblue")
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95%")
    ax.axvline(x=n95, color="red", linestyle=":", alpha=0.7)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative Variance")
    ax.set_title(f"Cumulative Variance\n{n95} PCs for 95%")
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    ax.text(
        0.5,
        0.5,
        f"Model: {model_type} (hidden={hidden_size})\n"
        f"Output: {model.output_size} ({model.cells_per_joint}/joint)\n"
        f"Dropout: {model.hparams.get('dropout', '?')}\n"
        f"Checkpoint: {Path(args.checkpoint).name}\n\n"
        f"PCs for 95%: {n95}\n"
        f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%\n"
        f"Top-3: {cumvar[2]*100:.1f}%\n"
        f"Top-6: {cumvar[5]*100:.1f}%\n\n"
        f"J1-selective (x,y,z): {best_j1_neurons}\n"
        f"  r = [{', '.join(f'{c:.3f}' for c in best_j1_corr)}]\n"
        f"J2-selective (x,y,z): {best_j2_neurons}\n"
        f"  r = [{', '.join(f'{c:.3f}' for c in best_j2_corr)}]",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("Summary")
    ax.axis("off")

    # Row 2: PCA colored by J1 rotvec x, y, z
    for col, comp in enumerate(range(3)):
        ax = fig.add_subplot(gs[1, col])
        sc = ax.scatter(
            hidden_pca[plot_idx, 0],
            hidden_pca[plot_idx, 1],
            c=j1_rotvec[plot_idx, comp],
            cmap="coolwarm",
            s=2,
            alpha=0.5,
        )
        plt.colorbar(sc, ax=ax, label=f"J1 rotvec {axis_labels[comp]}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PC1 vs PC2\nJ1 rotation {axis_labels[comp]}-component")

    # Row 3: t-SNE
    ax = fig.add_subplot(gs[2, 0])
    sc = ax.scatter(
        hidden_tsne[:, 0],
        hidden_tsne[:, 1],
        c=j1_rotvec[tsne_idx, 0],
        cmap="coolwarm",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="J1 rotvec x")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE\nJ1 rotation x-component")

    ax = fig.add_subplot(gs[2, 1])
    sc = ax.scatter(
        hidden_tsne[:, 0],
        hidden_tsne[:, 1],
        c=j2_rotvec[tsne_idx, 0],
        cmap="coolwarm",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="J2 rotvec x")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE\nJ2 rotation x-component")

    ax = fig.add_subplot(gs[2, 2])
    sc = ax.scatter(
        hidden_tsne[:, 0],
        hidden_tsne[:, 1],
        c=j1_cell[tsne_idx],
        cmap="tab20",
        s=2,
        alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Dominant cell ID")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE\nColored by dominant cell ID")

    # Row 4: Tuning — most J1-selective neurons
    for i in range(3):
        neuron_idx = best_j1_neurons[i]
        corr = best_j1_corr[i]
        comp = best_j1_comp[i]
        ax = fig.add_subplot(gs[3, i])
        _plot_tuning_curve(
            ax,
            hidden_flat[:, neuron_idx],
            j1_rotvec[:, comp],
            f"Neuron {neuron_idx} vs J1-{axis_labels[comp]} (r={corr:.3f})",
            f"Joint 1 rotvec {axis_labels[comp]}",
            "blue",
        )

    # Row 5: Tuning — most J2-selective neurons
    for i in range(3):
        neuron_idx = best_j2_neurons[i]
        corr = best_j2_corr[i]
        comp = best_j2_comp[i]
        ax = fig.add_subplot(gs[4, i])
        _plot_tuning_curve(
            ax,
            hidden_flat[:, neuron_idx],
            j2_rotvec[:, comp],
            f"Neuron {neuron_idx} vs J2-{axis_labels[comp]} (r={corr:.3f})",
            f"Joint 2 rotvec {axis_labels[comp]}",
            "red",
        )

    # Save
    if args.output:
        output_path = args.output
    else:
        output_path = str(
            Path(args.checkpoint).parent.parent / "representation_analysis.png"
        )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


def main():
    args = parse_args()

    if args.data_path is None:
        if args.manifold == "so2":
            args.data_path = "data/estimation/trajectories_so2.pt"
        else:
            args.data_path = "data/estimation/trajectories.pt"

    print(f"Loading checkpoint: {args.checkpoint}")
    model = StateEstimationModel.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    model.eval()

    print(f"Loading data: {args.data_path}")
    data = torch.load(args.data_path, weights_only=True)

    n_traj = min(args.n_traj, data["val_velocities"].shape[0])
    seq_len = (
        data["val_velocities"].shape[2] if data["val_velocities"].ndim > 2 else 100
    )
    seq_len = data["val_velocities"][:n_traj].shape[1]
    hidden_size = model.hidden_size

    if args.manifold == "so2":
        analyze_so2(args, model, data, n_traj, seq_len, hidden_size)
    else:
        analyze_so3(args, model, data, n_traj, seq_len, hidden_size)


if __name__ == "__main__":
    main()
