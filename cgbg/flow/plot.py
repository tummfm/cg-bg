import os
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from pathlib import Path
from chemtrain import quantity
import seaborn as sns

sns.set_style("white")
palette = sns.color_palette("muted", n_colors=10)

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 12,
    "axes.titlesize": 20,
    "legend.fontsize": 5.5,
    "legend.title_fontsize": 5.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.markersize": 3,
    "lines.linewidth": 1.5,
    "figure.dpi": 800,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
palette_paper = {
    'Exact':                 palette[3],  
    'Simulation':            palette[2],  
    'Proposal':              palette[1],  
    'Proposal (reweighted)': palette[0],  
    'Implicit':              palette[4], 
}

def plot_aldp_energy_density(
    U_ref: jnp.ndarray, 
    U_gen: jnp.ndarray, 
    w: jnp.ndarray, 
    es_percent: float,
    js_divergence: float,
    pmf_distance: float,
):
    mask = w > 1e-10
    U_gen_masked = U_gen[mask]
    w_masked = w[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    bins = 100

    ax.hist(
        U_ref,
        bins=bins,
        density=True,
        color=palette_paper['Simulation'],
        edgecolor='none',
        alpha=0.7,
        label='MD Reference',
    )
    ax.hist(
        U_gen_masked,
        bins=bins,
        density=True,
        color=palette_paper['Proposal'],
        edgecolor='none',
        alpha=0.7,
        label='CG-BG (proposal)',
    )
    ax.hist(
        U_gen_masked,
        bins=bins,
        density=True,
        color=palette_paper['Proposal (reweighted)'],
        histtype='step',
        linestyle='-',
        label='CG-BG (reweighted)',
        weights=w_masked
    )
    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    legend_title = f"ESS: {es_percent:.2%}\nJS Divergence: {js_divergence:.4f}\nPMF Distance: {pmf_distance:.4f}"
    ax.legend(title=legend_title, loc="best")

    plt.tight_layout()
    plt.savefig("plots/aldp_energy_distribution.png")

def plot_aldp_density(
    phi_ref: jnp.ndarray,
    phi_gen: jnp.ndarray,
    psi_ref: jnp.ndarray,
    psi_gen: jnp.ndarray,
    w: jnp.ndarray,
):
    fig, (ax_phi, ax_psi) = plt.subplots(1, 2, figsize=(16, 6))
    plot_range = (-jnp.pi, jnp.pi)
    bins = 100

    # Phi density plot
    ax_phi.hist(
        phi_ref,
        bins=bins,
        range=plot_range,
        density=True,
        color= palette_paper['Simulation'],
        edgecolor='none',
        alpha=0.7,
        label='MD Reference',
    )
    ax_phi.hist(
        phi_gen,
        bins=bins,
        range=plot_range,
        density=True,
        color=palette_paper['Proposal'],
        edgecolor='none',
        alpha=0.7,
        label='CG-BG (proposal)',
    )
    ax_phi.hist(
        phi_gen,
        bins=bins,
        range=plot_range,
        density=True,
        color=palette_paper['Proposal (reweighted)'],
        histtype='step',
        linestyle='-',
        label='CG-BG (reweighted)',
        weights=w
    )
    ax_phi.set_xlabel(r"$\phi$")
    ax_phi.set_ylabel("Density")
    ax_phi.legend(loc="best", alignment="left")

    # Psi density plot
    ax_psi.hist(
        psi_ref,
        bins=bins,
        range=plot_range,
        density=True,
        color=palette_paper['Simulation'],
        edgecolor='none',
        alpha=0.7,
        label='MD Reference',
    )
    ax_psi.hist(
        psi_gen,
        bins=bins,
        range=plot_range,
        density=True,
        color=palette_paper['Proposal'],
        edgecolor='none',
        alpha=0.7,
        label='CG-BG (proposal)',
    )
    ax_psi.hist(
        psi_gen,
        bins=bins,
        range=plot_range,
        density=True,
        color=palette_paper['Proposal (reweighted)'],
        histtype='step',
        linestyle='-',
        label='CG-BG (reweighted)',
        weights=w
    )
    ax_psi.set_xlabel(r"$\psi$")
    ax_psi.set_ylabel("Density")
    ax_psi.legend(loc="best", alignment="left")

    plt.tight_layout()
    plt.savefig("plots/aldp_density.png")

def plot_aldp_free_energy(
    kT: float,
    phi_ref: jnp.ndarray,
    phi_gen: jnp.ndarray,
    phi_implicit: jnp.ndarray,
    psi_ref: jnp.ndarray,
    psi_gen: jnp.ndarray,
    psi_implicit: jnp.ndarray,
    w: jnp.ndarray,
):
    x_grid = jnp.linspace(-jnp.pi, jnp.pi, 300)
    energy_cutoff = 22.5

    def compute_F(samples: jnp.ndarray, weights: Optional[jnp.ndarray] = None):
        kde = gaussian_kde(samples, weights=weights)
        P = kde(x_grid)
        F = -kT * jnp.log(P)
        min_val = jnp.nanmin(jnp.where(jnp.isfinite(F), F, jnp.inf))
        F = F - min_val
        F = jnp.where(F > energy_cutoff, jnp.nan, F)
        return F

    F_phi_ref = compute_F(phi_ref)
    F_phi_gen = compute_F(phi_gen)
    F_phi_rew = compute_F(phi_gen, weights=w) # Reweighted
    F_phi_imp = compute_F(phi_implicit)

    F_psi_ref = compute_F(psi_ref)
    F_psi_gen = compute_F(psi_gen)
    F_psi_rew = compute_F(psi_gen, weights=w) # Reweighted
    F_psi_imp = compute_F(psi_implicit)
    
    fig, (ax_phi, ax_psi) = plt.subplots(1, 2, figsize=(6, 3))
    y_pos = 0.95
    x_positions = [0.03, 0.52]
    labels = ["a)", "b)"]
    for x, label in zip(x_positions, labels):
        fig.text(
            x, y_pos, label,
            va='top',
            ha='left'
        )

    # Phi free energy plot
    ax_phi.plot(x_grid, F_phi_ref, color=palette_paper['Simulation'], linestyle='-', label='MD Reference (explicit)')
    ax_phi.plot(x_grid, F_phi_imp, color=palette_paper['Implicit'], linestyle='-', label='MD Reference (implicit)')
    ax_phi.plot(x_grid, F_phi_gen, color=palette_paper['Proposal'], linestyle='--', label='CG-BG (proposal)')
    ax_phi.plot(x_grid, F_phi_rew, color=palette_paper['Proposal (reweighted)'], linestyle='--', label='CG-BG (reweighted)')
    ax_phi.set_xlabel(r"$\phi$")
    ax_phi.set_ylabel(r"Free Energy / $k_B T$")
    ax_phi.set_xlim(-jnp.pi, jnp.pi)
    ax_phi.legend(loc="lower right", alignment="left", fontsize=7)

    # Psi free energy plot
    ax_psi.plot(x_grid, F_psi_ref, color=palette_paper['Simulation'], linestyle='-', label='MD Reference (explicit)')
    ax_psi.plot(x_grid, F_psi_imp, color=palette_paper['Implicit'], linestyle='-', label='MD Reference (implicit)')
    ax_psi.plot(x_grid, F_psi_gen, color=palette_paper['Proposal'], linestyle='--', label='CG-BG (proposal)')
    ax_psi.plot(x_grid, F_psi_rew, color=palette_paper['Proposal (reweighted)'], linestyle='--', label='CG-BG (reweighted)')
    ax_psi.set_xlabel(r"$\psi$")
    ax_psi.set_ylabel(r"Free Energy / $k_B T$")
    ax_psi.set_xlim(-jnp.pi, jnp.pi)
    ax_psi.legend(loc="upper right", alignment="left", fontsize=7)

    plt.tight_layout()
    plt.savefig("plots/aldp_free_energy.png")

def plot_aldp_ramachandran_fes(
    kT: float,
    w: jnp.ndarray,
    phi_gen: jnp.ndarray,
    psi_gen: jnp.ndarray,
    phi_ref: jnp.ndarray,
    psi_ref: jnp.ndarray,
):
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    kT = kT / 4.184
    ticks = [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi]
    tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    cmap = plt.get_cmap('viridis')

    h_ref, x_edges_ref, y_edges_ref = jnp.histogram2d(phi_ref, psi_ref, bins=100, density=True)
    h_ref = -kT * jnp.log(h_ref)
    x_ref, y_ref = jnp.meshgrid(x_edges_ref, y_edges_ref)
    cax_ref = ax[0].pcolormesh(x_ref, y_ref, h_ref.T, cmap=cmap, vmax=5.25)
    ax[0].set_xlabel(r"$\phi$")
    ax[0].set_ylabel(r"$\psi$")
    ax[0].set_box_aspect(1)
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(tick_labels)
    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels(tick_labels)
    ax[0].set_xlim(-jnp.pi, jnp.pi)
    ax[0].set_ylim(-jnp.pi, jnp.pi)
    ax[0].set_title("Reference")

    h, x_edges, y_edges = jnp.histogram2d(phi_gen, psi_gen, bins=100, density=True)
    h = -kT * jnp.log(h)
    x, y = jnp.meshgrid(x_edges, y_edges)
    cax = ax[1].pcolormesh(x, y, h.T, cmap=cmap, vmax=5.25)
    ax[1].set_xlabel(r"$\phi$")
    ax[1].set_ylabel(r"$\psi$")
    ax[1].set_box_aspect(1)
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(tick_labels)
    ax[1].set_yticks(ticks)
    ax[1].set_yticklabels(tick_labels)
    ax[1].set_xlim(-jnp.pi, jnp.pi)
    ax[1].set_ylim(-jnp.pi, jnp.pi)
    ax[1].set_title("Proposal")

    h, x_edges, y_edges = jnp.histogram2d(phi_gen, psi_gen, bins=100, density=True, weights=w)
    h = -kT * jnp.log(h)
    x, y = jnp.meshgrid(x_edges, y_edges)
    cax = ax[2].pcolormesh(x, y, h.T, cmap=cmap, vmax=5.25)
    ax[2].set_xlabel(r"$\phi$")
    ax[2].set_ylabel(r"$\psi$")
    ax[2].set_box_aspect(1)
    ax[2].set_xticks(ticks)
    ax[2].set_xticklabels(tick_labels)
    ax[2].set_yticks(ticks)
    ax[2].set_yticklabels(tick_labels)
    ax[2].set_xlim(-jnp.pi, jnp.pi)
    ax[2].set_ylim(-jnp.pi, jnp.pi)
    ax[2].set_title("Reweighted Proposal")

    plt.tight_layout()
    plt.savefig("plots/aldp_ramachandran_fes.png")

def V_shallow(x, y):
    term1 = -17.3 * jnp.exp(-0.0039 * (x - 48)**2 - 0.0391 * (y - 8)**2)
    term2 = -8.7 * jnp.exp(-0.0039 * (x - 32)**2 - 0.0391 * (y - 16)**2)
    term3 = -14.7 * jnp.exp(-0.0254 * (x - 24)**2 + 0.043 * (x - 24) * (y - 32) - 0.0254 * (y - 32)**2)
    term4 = 1.3 * jnp.exp(0.00273 * (x - 16)**2 + 0.0023 * (x - 16) * (y - 24) + 0.00273 * (y - 24)**2)
    return term1 + term2 + term3 + term4

def plot_mb(kT, x_ref, x_gen, w):

    x_vals = jnp.linspace(0, 50, 300)
    y_vals = jnp.linspace(0, 50, 300)
    def P(xi):
        V_y = jax.vmap(lambda y: V_shallow(xi, y))(y_vals)
        return jnp.trapezoid(jnp.exp(- V_y/kT), y_vals)

    P_x = jax.vmap(P)(x_vals)
    P_x /= jnp.trapezoid(P_x, x_vals)
    F_true = -kT * jnp.log(P_x)
    F_true -= jnp.min(F_true)

    def compute_F(samples: jnp.ndarray, weights: Optional[jnp.ndarray] = None):
        kde = gaussian_kde(samples, weights=weights)
        P = kde(x_vals)
        F = -kT * jnp.log(P)
        min_val = jnp.nanmin(jnp.where(jnp.isfinite(F), F, jnp.inf))
        F = F - min_val
        return F
    
    F_ref = compute_F(x_ref)
    F_gen = compute_F(x_gen)
    F_rew = compute_F(x_gen, weights=w)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].plot(x_vals, P_x, color=palette_paper['Exact'], label="Exact")
    ax[0].hist(
        x_ref,
        bins=100,
        density=True,
        color= palette_paper['Simulation'],
        edgecolor='none',
        alpha=0.7,
        label='MD Reference',
    )
    ax[0].hist(
        x_gen,
        bins=100,
        density=True,
        color=palette_paper['Proposal'],
        edgecolor='none',
        alpha=0.4,
        label='CG-BG (proposal)',
        weights=w
    )
    ax[0].hist(
        x_gen,
        bins=100,
        density=True,
        weights=w,
        color=palette_paper['Proposal (reweighted)'],
        histtype='step',
        linestyle='-',
        label='CG-BG (reweighted)',
    )
    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$P(x)$")
    ax[0].set_xlim(10, 50)
    ax[0].set_ylim(-0.01, 0.2)
    ax[0].legend()

    ax[1].plot(x_vals, F_true, color=palette_paper['Exact'], linestyle='-', label='Exact')
    ax[1].plot(x_vals, F_ref, color=palette_paper['Simulation'], linestyle='-', label='MD Reference')
    ax[1].plot(x_vals, F_gen, color=palette_paper['Proposal'], linestyle='--', label='CG-BG (proposal)')
    ax[1].plot(x_vals, F_rew, color=palette_paper['Proposal (reweighted)'], linestyle='--', label='CG-BG (reweighted)')
    ax[1].set_xlim(10, 50)
    ax[1].set_ylim(-0.5, 15)
    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"Free Energy / $k_B T$")
    ax[1].legend(loc="best")
    ax[1].set_yticks([0, 5, 10, 15])
    
    plt.savefig("plots/mb_plots.png")

@jax.jit
@jax.vmap
def dihedral(p: jnp.ndarray) -> jnp.ndarray:
    """The code is taken and adapted from: http://stackoverflow.com/q/20305272/1128289

    Args:
        p: A set of points in the form of a jax numpy array with shape (batch, 4, 3).
    Returns:
        The dihedral angles in radians.
    """
    b = p[:-1] - p[1:]
    b = b.at[0].set(-b[0])
    v = jnp.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= jnp.sqrt(jnp.einsum("...i,...i", v, v)).reshape(-1, 1)
    b1 = b[1] / jnp.linalg.norm(b[1])
    x = jnp.dot(v[0], v[1])
    m = jnp.cross(v[0], b1)
    y = jnp.dot(m, v[1])
    return jnp.arctan2(y, x)

def compute_metrics_2d(
    target: jnp.ndarray,
    sample: jnp.ndarray,
    bg_weights: Optional[jnp.ndarray] = None,
    limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    bins: int = 64,
    baseline: float = 1e-6
) -> Tuple[float, float, float]:
    
    if limits is None:
        all_data = jnp.vstack([target, sample])
        x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
        y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
        range_arr = [[x_min, x_max], [y_min, y_max]]
    else:
        (x_min, x_max), (y_min, y_max) = limits
        range_arr = [[x_min, x_max], [y_min, y_max]]

    hist_target, _, _ = jnp.histogram2d(target[:, 0], target[:, 1], bins=bins, range=range_arr)
    hist_sampled, _, _ = jnp.histogram2d(sample[:, 0], sample[:, 1], bins=bins, range=range_arr, weights=bg_weights)

    p_target = hist_target / (hist_target.sum() + 1e-10)
    p_sampled = hist_sampled / (hist_sampled.sum() + 1e-10)

    p = p_target.flatten()  # Target / True
    q = p_sampled.flatten() # Sampled / Model

    # ==========================================
    # Metric 1: JS Divergence (Standard)
    # ==========================================
    js_distance = jensenshannon(p, q)
    js_div = js_distance ** 2

    # ==========================================
    # Metric 2: RMS FE Sq Error (Free Energy)
    # ==========================================
    p_filled = jnp.where(p == 0, baseline, p)
    q_filled = jnp.where(q == 0, baseline, q)
    m_filled = (p_filled + q_filled) / 2
    weights_filled = m_filled / m_filled.sum()

    fe_loss = (-jnp.log(q_filled) - (-jnp.log(p_filled))) ** 2
    rms_fe_sq_error = jnp.sum(weights_filled * fe_loss)

    # ==========================================
    # Metric 3: Effective Sample Size (ESS)
    # ==========================================
    ess = 1.0 / jnp.sum(bg_weights ** 2)
    es_percent = ess / len(bg_weights)

    return js_div, rms_fe_sq_error, es_percent

def aldp_plots(reweighted_state: dict, task: str, work_dir: str):

    kT = quantity.kb * 300  # in kJ/mol
    work_dir = Path(work_dir)

    sample_impl_path = work_dir / "cgbg" / "data" / "aldp" / "raw" / "openmm_allatomMD_implicit.npz"
    phi_impl_indices = [4, 6, 7, 8]
    psi_impl_indices = [6, 7, 8, 16]

    if "CB" in task:
        sample_ref_path = work_dir / "cgbg" / "data" / "aldp" / "raw" / "openmm_corebetaMD.npz"
        phi_indices = [0, 1, 2, 4]
        psi_indices = [1, 2, 4, 5]
    elif "HA" in task:
        sample_ref_path = work_dir / "cgbg" / "data" / "aldp" / "raw" / "openmm_heavyatomMD.npz"
        phi_indices = [1, 3, 4, 6]
        psi_indices = [3, 4, 6, 8]

    w = reweighted_state["w"]
    U_gen = reweighted_state["U"]
    x_gen = reweighted_state["R"]
    phi_gen, psi_gen = dihedral(x_gen[:, phi_indices, :]), dihedral(x_gen[:, psi_indices, :])

    data_ref = jnp.load(sample_ref_path, allow_pickle=True)
    U_ref = data_ref["U"]
    x_ref = data_ref["R"]
    phi_ref, psi_ref = dihedral(x_ref[:, phi_indices, :]), dihedral(x_ref[:, psi_indices, :])
    
    data_impl = jnp.load(sample_impl_path, allow_pickle=True)
    x_impl = data_impl["R"]
    phi_implicit, psi_implicit = dihedral(x_impl[:, phi_impl_indices, :]), dihedral(x_impl[:, psi_impl_indices, :])

    js_snis_div, rms_snis_fe, es_percent = compute_metrics_2d(
        target=jnp.stack([phi_ref, psi_ref], axis=1),
        sample=jnp.stack([phi_gen, psi_gen], axis=1),
        bg_weights=w,
        limits=((-jnp.pi, jnp.pi), (-jnp.pi, jnp.pi)),
        bins=100,
        baseline=1e-10,
    )

    os.makedirs("plots", exist_ok=True)
    plot_aldp_energy_density(U_ref, U_gen, w, es_percent, js_snis_div, rms_snis_fe)
    plot_aldp_density(phi_ref, phi_gen, psi_ref, psi_gen, w)
    plot_aldp_free_energy(kT, phi_ref, phi_gen, phi_implicit, psi_ref, psi_gen, psi_implicit, w,)
    plot_aldp_ramachandran_fes(kT, w, phi_gen, psi_gen, phi_ref, psi_ref)

def mb_plots(reweighted_state: dict, work_dir: str):
    kT = 1.0
    work_dir = Path(work_dir)

    w = reweighted_state["w"]
    x_gen = jnp.squeeze(reweighted_state["R"])

    sample_ref_path = work_dir / "cgbg" / "data" / "mb" / "raw" / "mb_unbiased.npz"
    data_ref = jnp.load(sample_ref_path, allow_pickle=True)
    x_ref = jnp.squeeze(data_ref["R"])

    os.makedirs("plots", exist_ok=True)
    plot_mb(kT, x_ref, x_gen, w)