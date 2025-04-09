import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.constants import c
import os
from casatools import vpmanager, quanta

qa = quanta()
vp = vpmanager()

# Speed of light in m/s
C_LIGHT = c


def airy_model(R, s, A):
    """
    Compute the Airy pattern for a uniform disk of radius R (arcsec) at uv distance s (in wavelengths).

    Parameters:
      R : float
          Disk radius in arcseconds.
      s : array_like
          UV distance in wavelengths.
      A : float
          Overall amplitude scaling factor.

    Returns:
      Array of model visibilities.
    """
    # z = 2 * R * pi^2 / (180 * 3600) * s  (per the given code)
    z = 2.0 * R * (np.pi ** 2) / (180.0 * 3600.0) * s
    z = np.where(z == 0, 1e-8, z)  # Avoid division by zero
    return A * (2 * j1(z) / z)


def disk_size_function(v, c1, alpha1, c2, alpha2):
    """
    Analytic model for disk size as a function of frequency (v, in GHz).

    R(v) = c1 * v^(-alpha1) + c2 * v^(-alpha2)

    Returns disk radius in arcseconds.
    """
    return c1 * v ** (-alpha1) + c2 * v ** (-alpha2)


import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def generate_fibonacci_spiral_antenna_positions(n_antennas=20, scale=5, latitude=39.54780):
    """
    Generate antenna positions in a Fibonacci spiral layout.

    Parameters:
      n_antennas : int
          Number of antennas to generate.
      scale : float
          Scaling factor for the radial distance.
      latitude : float
          Latitude (in degrees) to adjust the aspect ratio between east (x) and north (y).

    Returns:
      positions : numpy.ndarray
          Array of shape (n_antennas, 2) containing (x, y) antenna positions in meters.
    """
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~2.39996 radians
    positions = []
    for i in range(n_antennas):
        r = scale * i
        theta = i * golden_angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions.append((x, y))
    positions = np.array(positions)
    # Adjust the north coordinate for latitude aspect ratio.
    aspect = 1 / np.cos(np.deg2rad(latitude))
    positions[:, 1] *= aspect
    return positions


def generate_golden_spiral_antenna_positions(n_antennas=20, r0=5, r_max=100, n_turns=2, latitude=39.54780):
    """
    Generate antenna positions along a golden spiral layout.

    Parameters:
      n_antennas : int
          Number of antennas.
      r0 : float
          Starting radius (m).
      r_max : float
          Maximum radius (m) after n_turns.
      n_turns : float
          Number of spiral turns.
      latitude : float
          Latitude (in degrees) to adjust the aspect ratio.

    Returns:
      positions : numpy.ndarray
          Array of shape (n_antennas, 2) containing (x, y) positions in meters.
    """
    theta_max = n_turns * 2 * np.pi
    beta = np.log(r_max / r0) / theta_max
    positions = []
    for i in range(n_antennas):
        theta = i * theta_max / (n_antennas - 1)
        r = r0 * np.exp(beta * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions.append((x, y))
    positions = np.array(positions)
    aspect = 1 / np.cos(np.deg2rad(latitude))
    positions[:, 1] *= aspect
    return positions


def generate_log_spiral_antenna_positions(n_arms=3, antennas_per_arm=6, n_turn=1.0, r0=5, r_max=100,
                                          alpha=1.0, gamma=1.0, latitude=39.54780):
    """
    Generate antenna positions using a multi-arm logarithmic spiral layout.

    The spiral is defined by:
         r = r0 * exp( beta * theta**gamma )
    where beta is chosen so that r reaches r_max when theta = n_turn * 2π.

    If the first antenna along each arm has a radial distance less than minimum_dist,
    it is removed from the beginning and appended to the tail of that arm.

    Parameters:
      n_arms : int
          Number of spiral arms.
      antennas_per_arm : int
          Number of antennas per arm.
      n_turn : float
          Number of spiral turns (default is 1.0).
      r0 : float
          Starting radius (m).
      r_max : float
          Maximum radius (m) at the outer end of each arm.
      alpha : float
          Angular scaling factor; scales the angular coordinate.
      gamma : float
          Exponent for modifying the spiral curvature.
      latitude : float
          Latitude (in degrees) to adjust the aspect ratio between east (x) and north (y)
          via a factor of 1/cos(latitude).

    Returns:
      positions : numpy.ndarray
          Array of shape ((n_arms*antennas_per_arm + 1), 2) containing the (x, y) antenna positions in meters.
          A central antenna at (0,0) is appended.
    """

    # Compute theta_max (total angle) based on n_turn.
    theta_max = n_turn * 2 * np.pi
    # Compute beta so that when theta=theta_max, r = r_max.
    beta = np.log(r_max / r0) / (theta_max ** gamma)

    # Create an array of indices along a single arm.
    i_vals = np.arange(antennas_per_arm)
    # Compute the array of angles for one arm.
    theta_arm = (theta_max * i_vals) / (antennas_per_arm - 1)  # shape: (antennas_per_arm,)
    # Calculate radial distances for these angles.
    r_vals = r0 * np.exp(beta * (theta_arm ** gamma))  # shape: (antennas_per_arm,)
    # print(r_vals)

    # Create an array for the arm indices.
    arm_indices = np.arange(n_arms)
    # Compute theta offset for each arm.
    theta_offset = (2 * np.pi * arm_indices) / n_arms  # shape: (n_arms,)

    # Broadcast to compute total angle for each antenna on every arm.
    theta_total = -alpha * theta_arm[None, :] + theta_offset[:, None]  # shape: (n_arms, antennas_per_arm)
    # Broadcast radial values.
    r_matrix = r_vals[None, :]  # shape: (1, antennas_per_arm)
    theta_total[:, 0] += np.pi / 4


    # Compute x and y positions.
    x_mat = r_matrix * np.cos(theta_total)  # shape: (n_arms, antennas_per_arm)
    y_mat = r_matrix * np.sin(theta_total)  # shape: (n_arms, antennas_per_arm)

    # Flatten the 2D arrays to a 1D list of positions.
    positions = np.column_stack((x_mat.flatten(), y_mat.flatten()))

    # # Append the central antenna position.
    # positions = np.vstack((positions, np.array([[-1, 0.1]]), np.array([[1.5, 1]]), np.array([[1.2, -2.1]])))
    # positions = np.vstack((positions, np.array([[-1, 0.1]]), np.array([[1.5, 1]]), np.array([[1.2, -2.1]])))

    # Adjust north (y) coordinate by the aspect factor from the latitude.
    aspect = 1 / np.cos(np.deg2rad(latitude))
    positions[:, 1] *= aspect

    # print(f'fig-fasr_Log_Spiral-{len(positions)}_n_arms={n_arms}, antennas_per_arm={antennas_per_arm}, alpha={alpha:.2f}, gamma={gamma:.2f}, r0={r0:.1f}, r_max={r_max:.0f}, n_turn={n_turn:.1f}')
    return positions


def generate_archimedean_spiral_antenna_positions(n_antennas=20, a=1, b=5, theta_max=4 * np.pi, latitude=39.54780):
    """
    Generate antenna positions along an Archimedean spiral.

    The Archimedean spiral is defined as r = a + b * theta.

    Parameters:
      n_antennas : int
          Number of antennas.
      a : float
          Initial radius offset.
      b : float
          Scaling factor (rate of radial growth).
      theta_max : float
          Maximum angle (radians) to span.
      latitude : float
          Latitude (in degrees) to adjust the aspect ratio.

    Returns:
      positions : numpy.ndarray
          Array of (x, y) antenna positions in meters.
    """
    positions = []
    for i in range(n_antennas):
        theta = i * theta_max / (n_antennas - 1)
        r = a + b * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions.append((x, y))
    positions = np.array(positions)
    aspect = 1 / np.cos(np.deg2rad(latitude))
    positions[:, 1] *= aspect
    return positions


def generate_pseudorandom_disk_antenna_positions(n_antennas=20, radius=150, n_edge=5, cluster_antennas=3,
                                                 cluster_radius=1.5,
                                                 latitude=39.54780):
    """
    Generate antenna positions pseudo-randomly within a circular disk.

    This function creates a total of n_antennas positions such that:
      - 'cluster_antennas' form a dense cluster within a circle of radius 'cluster_radius'
        (ensuring they are closely spaced).
      - The remaining antennas are generated pseudo-randomly:
          - 25% of these are placed exactly on the disk's circumference.
          - The remaining are uniformly distributed inside 80% of the disk's radius,
            to yield a denser interior.

    Parameters:
      n_antennas : int
          Total number of antennas (must be >= cluster_antennas).
      radius : float
          Radius of the disk in meters.
      n_edge : int
            Number of antennas in a circular disk.
      cluster_antennas : int
          Number of antennas in the dense cluster.
      cluster_radius : float
          Radius of the dense cluster (in meters).
      latitude : float
          Latitude (in degrees) to adjust the aspect ratio.

    Returns:
      positions : numpy.ndarray
          Array of shape (n_antennas, 2) containing (x, y) positions in meters.
    """
    if n_antennas < cluster_antennas:
        raise ValueError("n_antennas must be at least equal to cluster_antennas.")

    # --- Generate Dense Cluster ---
    r_cluster = cluster_radius * np.sqrt(np.random.rand(cluster_antennas))
    theta_cluster = np.random.uniform(0, 2 * np.pi, cluster_antennas)
    x_cluster = r_cluster * np.cos(theta_cluster)
    y_cluster = r_cluster * np.sin(theta_cluster)
    cluster_positions = np.column_stack((x_cluster, y_cluster))

    # --- Generate Pseudo-random Positions for the Remaining Antennas ---
    other_count = n_antennas - cluster_antennas
    n_inside = other_count - n_edge

    # Edge antennas: Equally spaced on the disk's circumference.
    theta_edge = np.linspace(0, 2 * np.pi, n_edge + 1)[:-1]
    x_edge = radius * np.cos(theta_edge)
    y_edge = radius * np.sin(theta_edge)
    edge_positions = np.column_stack((x_edge, y_edge))

    # Interior antennas: Uniformly distributed within 80% of the disk's radius.
    r_inside = radius * 0.8 * np.sqrt(np.random.rand(n_inside))
    theta_inside = np.random.uniform(0, 2 * np.pi, n_inside)
    x_inside = r_inside * np.cos(theta_inside)
    y_inside = r_inside * np.sin(theta_inside)
    inside_positions = np.column_stack((x_inside, y_inside))

    other_positions = np.vstack((edge_positions, inside_positions))
    positions = np.vstack((cluster_positions, other_positions))
    np.random.shuffle(positions)
    aspect = 1 / np.cos(np.deg2rad(latitude))
    positions[:, 1] *= aspect
    return positions


def generate_concentric_rings_antenna_positions(n_rings=3, antennas_per_ring=6, inner_radius=10, outer_radius=150,
                                                add_center=True, latitude=39.54780):
    """
    Generate antenna positions arranged on concentric rings.

    Parameters:
      n_rings : int
          Number of concentric rings.
      antennas_per_ring : int
          Number of antennas to place on each ring.
      inner_radius : float
          Radius of the innermost ring (meters).
      outer_radius : float
          Radius of the outermost ring (meters).
      add_center : bool
          Whether to include an antenna at the center.
      latitude : float
          Latitude (in degrees) to adjust the aspect ratio.

    Returns:
      positions : numpy.ndarray
          Array of (x, y) antenna positions in meters.
    """
    positions = []
    # Linearly spaced radii for the rings.
    if n_rings > 1:
        radii = np.linspace(inner_radius, outer_radius, n_rings)
    else:
        radii = [inner_radius]
    # Place antennas evenly on each ring.
    for r in radii:
        for j in range(antennas_per_ring):
            theta = 2 * np.pi * j / antennas_per_ring
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append((x, y))
    if add_center:
        positions.append((0, 0))
    positions = np.array(positions)
    aspect = 1 / np.cos(np.deg2rad(latitude))
    positions[:, 1] *= aspect
    return positions


def compute_uv_coverage(positions):
    uv_points = []
    for (x1, y1), (x2, y2) in combinations(positions, 2):
        u = x2 - x1
        v = y2 - y1
        uv_points.append((u, v))
        uv_points.append((-u, -v))  # include the conjugate
    return np.array(uv_points)


def plot_all_panels0(positions, title):
    '''
    Plot all panels for the given antenna positions in a 2x2 layout.
    '''
    uv = compute_uv_coverage(positions)

    # Create a 2x2 layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Antenna Layout
    ax1 = axs[0, 0]
    ax1.set_title(f"{title} - Antenna Layout")
    ax1.plot(positions[:, 0], positions[:, 1], 'o')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.axis('equal')
    # ax1.grid(True)

    # Panel 2: UV Coverage
    ax2 = axs[0, 1]
    ax2.set_title(f"{title} - UV Coverage")
    ax2.plot(uv[:, 0], uv[:, 1], '.', markersize=1)
    ax2.set_xlabel("u (m)")
    ax2.set_ylabel("v (m)")
    ax2.axis('equal')
    #     ax2.grid(True)

    # Panel 3: UV Density vs. UV Distance
    ax3 = axs[1, 0]
    # Compute radial distances of the uv points
    uv_dist = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
    bins = np.arange(0, np.max(uv_dist) + 10, 10)
    counts, bin_edges = np.histogram(uv_dist, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax3.step(bin_centers, counts, where='mid')
    ax3.set_title("UV Sampling Density")
    ax3.set_xlabel("UV Distance [m]")
    ax3.set_ylabel("Density (counts)")
    ax3.axhline(10, ls='--', color='gray', label='Nyquist sampling rate (20 GHz)')
    ax3.axhline(5.26, ls=':', color='gray', label='Nyquist sampling rate (10 GHz)')
    ax3.set_ylim(0, 100)
    # ax3.grid(True)
    ax3.legend()

    # Panel 4: PSF of the UV Sampling
    ax4 = axs[1, 1]
    # Define grid parameters for 2D histogram
    grid_size = 256
    u_min, u_max = -120, 120
    v_min, v_max = -120, 120
    H, xedges, yedges = np.histogram2d(uv[:, 0], uv[:, 1], bins=grid_size, range=[[u_min, u_max], [v_min, v_max]])
    # Compute the PSF via FFT of the sampling grid
    psf = np.abs(np.fft.fftshift(np.fft.fft2(H)))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax4.imshow(psf, extent=extent, origin='lower', aspect='equal', cmap='viridis')
    ax4.set_title("PSF of UV Sampling")
    ax4.set_xlabel("Image Plane X (arbitrary)")
    ax4.set_ylabel("Image Plane Y (arbitrary)")
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    fig.tight_layout()



def plot_all_panels(positions, title='', labels = [], frequency=2, nyq_sample=None, figname=None, array_config_str=None):
    """
    Plot antenna positions, UV coverage, and (optionally) PSF or UV density.

    If the input is a single 2D NumPy array (shape (N,2)), it is interpreted as one
    antenna configuration, and a 2x3 layout (including the PSF panel) is used.

    If the input is a list (or tuple) of 2D NumPy arrays, each array is treated as a separate
    configuration and a 2x2 layout (without the PSF panel) is produced, with each configuration
    overplotted in the panels.

    Parameters:
      positions : numpy.ndarray or list/tuple of numpy.ndarray
          Either a single array of shape (N,2) or a list of such arrays.
      title : str, optional
          Title text for the plots.
      frequency : float, optional
          Frequency in GHz (used only when a single array is provided for the PSF panel).
      nyq_sample : dict or None, optional
          Dictionary of Nyquist sampling rates to plot horizontal reference lines in the density plot.

    Returns:
      fig, axes : tuple
          The Matplotlib figure and a tuple of axis objects.
    """
    import matplotlib.gridspec as gridspec
    # Ensure the input is a list of 2D arrays.
    if isinstance(positions, np.ndarray):
        # Check that it is 2D.
        if positions.ndim == 2:
            pos_list = [positions]
        else:
            raise ValueError("If 'positions' is a numpy array, it must be 2D with shape (N,2).")
    elif isinstance(positions, (list, tuple)):
        # Check that each element is a 2D numpy array.
        pos_list = []
        for pos in positions:
            if not (isinstance(pos, np.ndarray) and pos.ndim == 2):
                raise ValueError("Every element in 'positions' must be a 2D numpy array with shape (N,2).")
            pos_list.append(pos)
    else:
        raise ValueError("'positions' must be either a 2D numpy array or a list/tuple of 2D numpy arrays.")


    # If only one set is provided, use a 2x3 layout (with PSF panel).
    if len(pos_list) == 1:
        pos = pos_list[0]
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])

        uv = compute_uv_coverage(pos)
        # Panel 1 (top left): Antenna Layout
        ax_ant = fig.add_subplot(gs[0, 0])
        ax_ant.set_title(f"{title} - {len(pos[:, 0])} Antenna Layout")
        ax_ant.plot(pos[:, 0], pos[:, 1], 'o')
        ax_ant.set_xlabel("X [m]")
        ax_ant.set_ylabel("Y [m]")
        ax_ant.set_aspect('equal')
        # ax_ant.grid(True)

        # Add equation text based on the title keyword
        eq_text = ""
        if "Fibonacci" in title:
            eq_text = r"$r = \varphi^{\frac{2\theta}{\pi}},\quad \varphi = \frac{1+\sqrt{5}}{2}$"
        elif "Golden" in title:
            eq_text = r"$r = r_0\,e^{\beta \theta},\quad \beta = \frac{\ln\left(\frac{r_{max}}{r_0}\right)}{\theta_{max}},\quad \theta_{max}=n_{turns}\cdot2\pi$"
        elif "Log" in title:
            eq_text = r"$r = r_0\,\exp\left(\beta\,\theta^\gamma\right),\quad \beta = \frac{\ln\left({r_{max}}/{r_0}\right)}{\theta_{max}^\gamma}$"
        elif "Archimedean" in title:
            eq_text = r"$r = a + b\,\theta$"
        else:
            eq_text = r"Equation not specified."
        ax_ant.text(0.02, 0.98, eq_text, transform=ax_ant.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

        # Panel 2 (top center): UV Coverage
        ax_uvcov = fig.add_subplot(gs[0, 1])
        ax_uvcov.set_title(f"{title} - UV Coverage")
        ax_uvcov.plot(uv[:, 0], uv[:, 1], '.', markersize=1)
        ax_uvcov.set_xlabel("u [m]")
        ax_uvcov.set_ylabel("v [m]")
        ax_uvcov.set_aspect('equal')
        #     ax_uvcov.grid(True)

        # Panel 3 (top right): PSF of UV Sampling
        ax_psf = fig.add_subplot(gs[0, 2])
        ax_psf.set_title(f"PSF ({frequency:.1f} GHz)")
        # Define grid parameters for 2D histogram
        grid_size = 128
        padded_size = grid_size * 8
        u_min, u_max = np.min(uv[:, 0]), np.max(uv[:, 0])
        v_min, v_max = np.min(uv[:, 1]), np.max(uv[:, 1])
        H, xedges, yedges = np.histogram2d(uv[:, 0], uv[:, 1], bins=grid_size, range=[[u_min, u_max], [v_min, v_max]])
        # Compute the PSF via FFT of the sampling grid
        psf = np.abs(np.fft.fftshift(np.fft.fft2(H, s=(padded_size, padded_size))))
        # Convert the uv grid (meters) to image plane coordinates in arcsec
        # First, compute the wavelength (m) from the input frequency (GHz)
        lambda_m = C_LIGHT / (frequency * 1e9)
        # The uv grid spans (u_max - u_min) in meters, which corresponds to (u_max - u_min)/λ in wavelengths.
        # In Fourier transform, the pixel scale in radians is ~ 1/(grid_size*Δu) with Δu in wavelengths.
        # Here, Δu = (u_max - u_min)/grid_size in meters, so in wavelengths it is Δu/λ.
        # Thus, the image pixel scale is:
        pixel_scale_rad = (grid_size * lambda_m) / ((u_max - u_min) * padded_size)
        pixel_scale_arcsec = pixel_scale_rad * 206265
        # Total field-of-view (FOV) in arcsec:
        fov_arcsec = padded_size * pixel_scale_arcsec
        extent = [-fov_arcsec / 2, fov_arcsec / 2, -fov_arcsec / 2, fov_arcsec / 2]
        im = ax_psf.imshow(psf, extent=extent, origin='lower', aspect='equal', cmap='viridis')
        ax_psf.set_xlim(-1000, 1000)
        ax_psf.set_ylim(-1000, 1000)
        ax_psf.set_xlabel("RA [arcsec]")
        ax_psf.set_ylabel("DEC [arcsec])")
        # fig.colorbar(im, ax=ax_psf, fraction=0.046, pad=0.04)

        # Panel 4 (bottom row, spanning all columns): UV Sampling Density vs. UV Distance
        ax_uvdensity = fig.add_subplot(gs[1, :])
        ax_uvdensity.set_title("UV Sampling Density")
        # Compute radial distances of the uv points
        uv_dist = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        binwidth = 10
        bins = np.arange(0, np.max(uv_dist) + binwidth, binwidth)
        counts, bin_edges = np.histogram(uv_dist, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax_uvdensity.step(bin_centers, counts, where='mid')
        ax_uvdensity.set_xlabel("UV Distance [m]")
        ax_uvdensity.set_ylabel(f"Density (counts/per {binwidth:d} m)")
        if nyq_sample is not None:
            ls = ['--', ':', '-.', '-']
            for i, (k, v) in enumerate(nyq_sample.items()):
                ax_uvdensity.axhline(v, ls=ls[i], color='gray', label=f'Nyquist sampling rate ({k})')

        # ax_uvdensity.set_ylim(0, 100)
        ax_uvdensity.set_yscale('log')
        ax_uvdensity.legend()
        #     ax_uvdensity.grid(True)

        gs.tight_layout(fig)
        gs.update(hspace=0.0)
        # return fig, (ax_ant, ax_uvcov, ax_psf, ax_uvdensity)
        # If more than one positions array is provided, use new 2x2 layout (no PSF panel).
        if array_config_str is not None:
            fig.text(0.5, 0.93, array_config_str, ha='center', va='top', fontsize=12)
        if figname is None:
            figname = f'fig-layout_{title.replace(" ","_")}-{len(pos_list[0])}.jpg'
        fig.savefig(figname, dpi=300)
    else:
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Top left: Antenna Layout (overplot each positions array).
        ax_ant = fig.add_subplot(gs[0, 0])
        ax_ant.set_title(f"{title} - Antenna Layout")
        for idx, pos in enumerate(pos_list):
            if len(labels)==len(pos_list):
                label = labels[idx]
            else:
                label = f"Set {idx+1}"
            ax_ant.plot(pos[:, 0], pos[:, 1], 'o', label=label, color=colors[idx % len(colors)])
        ax_ant.set_xlabel("X [m]")
        ax_ant.set_ylabel("Y [m]")
        ax_ant.set_aspect('equal')
        ax_ant.legend()

        # Top right: UV Coverage.
        ax_uvcov = fig.add_subplot(gs[0, 1])
        ax_uvcov.set_title(f"{title} - UV Coverage")
        for idx, pos in enumerate(pos_list):
            if len(labels)==len(pos_list):
                label = labels[idx]
            else:
                label = f"Set {idx+1}"
            uv = compute_uv_coverage(pos)
            ax_uvcov.plot(uv[:, 0], uv[:, 1], '.', markersize=1, label = label, color=colors[idx % len(colors)])
        ax_uvcov.set_xlabel("u [m]")
        ax_uvcov.set_ylabel("v [m]")
        ax_uvcov.set_aspect('equal')
        ax_uvcov.legend()

        # Bottom (spanning both columns): UV Sampling Density.
        ax_uvdensity = fig.add_subplot(gs[1, :])
        ax_uvdensity.set_title("UV Sampling Density")
        for idx, pos in enumerate(pos_list):
            if len(labels)==len(pos_list):
                label = labels[idx]
            else:
                label = f"Set {idx+1}"
            uv = compute_uv_coverage(pos)
            uv_dist = np.sqrt(uv[:,0]**2 + uv[:,1]**2)
            binwidth = 10
            bins = np.arange(0, np.max(uv_dist) + binwidth, binwidth)
            counts, bin_edges = np.histogram(uv_dist, bins=bins)
            bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
            ax_uvdensity.step(bin_centers, counts, where='mid', label=label, color=colors[idx % len(colors)])
        ax_uvdensity.set_xlabel("UV Distance [m]")
        ax_uvdensity.set_ylabel(f"Density (counts/per {binwidth} m)")
        if nyq_sample is not None:
            ls = ['--', ':', '-.', '-']
            for j, (k, v) in enumerate(nyq_sample.items()):
                ax_uvdensity.axhline(v, ls=ls[j % len(ls)], color='gray', label=f'Nyquist rate ({k})')
        ax_uvdensity.set_yscale('log')
        ax_uvdensity.legend()

        plt.tight_layout()
        if figname is None:
            figname = f'fig-layout_{"_".join(labels)}.jpg'
        fig.savefig(figname, dpi=300)
        # return fig, (ax_ant, ax_uvcov, ax_uvdensity)


def geodetic_to_ecef(lon, lat, h):
    """
    Convert geodetic coordinates (lon, lat, h) to ECEF (ITRF) coordinates.

    Parameters:
      lon : float
          Longitude in degrees.
      lat : float
          Latitude in degrees.
      h : float
          Altitude in meters.

    Returns:
      np.ndarray: ECEF coordinate (X, Y, Z) in meters.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis [m]
    f = 1 / 298.257223563  # flattening
    e2 = 2 * f - f ** 2  # eccentricity squared
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
    X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - e2) + h) * np.sin(lat_rad)
    return np.array([X, Y, Z])


def local_to_ecef_offsets(positions, cofa_lon, cofa_lat):
    """
    Convert local ENU offsets to ECEF offsets given a reference point (COFA).

    Assumes local coordinates: x is east offset, y is north offset, and up=0.

    Parameters:
      positions : numpy.ndarray
          Array of shape (N, 2) with local (east, north) offsets in meters.
      cofa_lon : float
          Longitude of the reference point (degrees).
      cofa_lat : float
          Latitude of the reference point (degrees).

    Returns:
      numpy.ndarray: Array of shape (N, 3) with ECEF offsets in meters.
    """
    lat0 = np.deg2rad(cofa_lat)
    lon0 = np.deg2rad(cofa_lon)
    # Extract east and north components; assume zero altitude offset (up)
    E = positions[:, 0]
    N = positions[:, 1]
    U = np.zeros_like(E)
    # ENU to ECEF conversion (for small offsets)
    dX = -np.sin(lon0) * E - np.sin(lat0) * np.cos(lon0) * N
    dY = np.cos(lon0) * E - np.sin(lat0) * np.sin(lon0) * N
    dZ = np.cos(lat0) * N
    return np.column_stack((dX, dY, dZ))


def read_casa_antenna_list(cfg_filename):
    """
    Read a CASA antenna list file and return local (east, north) offsets relative to the COFA.

    The file is expected to have header lines starting with '#' that include a line like:
        #COFA=-114.4258, 39.5478
    Optionally, the altitude may also be given (e.g., "#COFA=-114.4258, 39.5478, 0.0").
    The data lines are expected to have 5 columns:
         x, y, z, diam, pad
    where x,y,z are geocentric (ECEF) coordinates in meters.

    The function computes the COFA ECEF coordinate and then subtracts it from each antenna's
    ECEF coordinate. It then converts the ECEF offsets to local ENU offsets using the inverse
    transformation (assuming small offsets).

    Returns:
        positions : numpy.ndarray of shape (N, 2)
            Array of local (east, north) offsets in meters.
        cofa : tuple (cofa_lon, cofa_lat, cofa_alt)
    """
    # Initialize default values.
    cofa_lon = None
    cofa_lat = None
    cofa_alt = 0.0  # default if not found

    # Read header lines to extract COFA.
    with open(cfg_filename, 'r') as f:
        header_lines = []
        for line in f:
            if line.startswith('#'):
                header_lines.append(line.strip())
            else:
                break
    for line in header_lines:
        if line.startswith('#COFA='):
            # Remove the prefix and any spaces.
            cofa_str = line[len('#COFA='):].strip()
            # Split by comma.
            parts = [p.strip() for p in cofa_str.split(',')]
            if len(parts) >= 2:
                cofa_lon = float(parts[0])
                cofa_lat = float(parts[1])
            if len(parts) >= 3:
                cofa_alt = float(parts[2])
            break

    if cofa_lon is None or cofa_lat is None:
        raise ValueError("COFA information not found in header of the config file.")

    # Read the antenna data (skip header lines).
    data = np.genfromtxt(cfg_filename, comments='#')
    # data columns: x, y, z, diam, pad  (we only need columns 0,1,2)
    antenna_xyz = data[:, 0:3]

    # Compute the COFA ECEF coordinate.
    cofa_xyz = geodetic_to_ecef(cofa_lon, cofa_lat, cofa_alt)

    # Compute ECEF offsets for each antenna.
    dX, dY, dZ = (antenna_xyz - cofa_xyz).T

    # Compute local offsets using the pseudo-inverse (A^T) of the forward ENU-to-ECEF matrix.
    # First convert cofa_lon, cofa_lat to radians.
    lon0 = np.deg2rad(cofa_lon)
    lat0 = np.deg2rad(cofa_lat)
    # Inverse transformation:
    # E = - sin(lon0) * dX + cos(lon0) * dY
    # N = - sin(lat0)*cos(lon0) * dX - sin(lat0)*sin(lon0) * dY + cos(lat0) * dZ
    E = - np.sin(lon0) * dX + np.cos(lon0) * dY
    N = - np.sin(lat0) * np.cos(lon0) * dX - np.sin(lat0) * np.sin(lon0) * dY + np.cos(lat0) * dZ

    positions = np.column_stack((E, N))
    return positions, (cofa_lon, cofa_lat, cofa_alt)


def write_casa_antenna_list(filename, positions, cofa_lon=-114.42580, cofa_lat=39.54780, cofa_alt=0.0, diam=2.0,
                            prefix="A"):
    """
    Write antenna positions (given as local east/north offsets) to a CASA-compatible antenna list.

    The function converts the local (east, north) offsets (with zero altitude)
    to ITRF/XYZ Earth-centered coordinates by first converting the reference (COFA)
    to ECEF coordinates, then applying the ENU-to-ECEF transformation.

    Parameters:
      filename : str
          Output file name.
      positions : numpy.ndarray
          Array of shape (N, 2) containing local (east, north) offsets in meters.
      cofa_lon : float
          Longitude of the reference (COFA) in degrees.
      cofa_lat : float
          Latitude of the reference (COFA) in degrees.
      cofa_alt : float
          Altitude of the reference (COFA) in meters.
      diam : float
          Antenna diameter in meters (will be written as-is).
      prefix : str
          Prefix for antenna names (e.g., "A" to produce names like A00, A01, ...).
    """
    # Get the COFA ECEF coordinates.
    cofa_xyz = geodetic_to_ecef(cofa_lon, cofa_lat, cofa_alt)

    # Convert local (east, north) positions to ECEF offsets.
    offsets = local_to_ecef_offsets(positions, cofa_lon, cofa_lat)

    # Compute final ECEF coordinates for each antenna.
    antenna_xyz = cofa_xyz + offsets

    # Write the CASA antenna list file.
    with open(filename, 'w') as f:
        f.write("#observatory=FASR\n")
        f.write(f"#COFA={cofa_lon:.5f}, {cofa_lat:.5f}\n")
        f.write("#coordsys=XYZ\n")
        f.write("# x y z diam pad\n")
        for i, (x, y, z) in enumerate(antenna_xyz):
            name = f"{prefix}{i:02d}"
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {diam:.1f} {name}\n")
    print(f'Wrote {filename}')


def make_vp_table_airy(vptab='my_beams.vp'):
    # Remove existing beam table if present
    os.system('rm -rf ' + vptab)

    # Reset the vpmanager
    vp.reset()

    # Construct an Airy primary beam for a dish with 2 m diameter.
    # The 'dishdiam' parameter takes a list of qa quantities.
    vp.setpbairy(telescope='FASR_CA', dishdiam=[qa.quantity('2m')])

    # Summarize and save the voltage pattern table to a CASA-compatible file.
    vp.summarizevps()
    vp.saveastable(vptab)


"""
CASA Simulator Script Using FITS Solar Model

This script:
  - Reads the antenna configuration file (cfg) which includes x, y, z, dish diameter, and antenna names.
  - Uses Astropy to open a solar model FITS file and extract:
      * Frequency from the CRVAL3 header keyword (in Hz, converted to GHz)
      * Source RA and DEC from CRVAL1 and CRVAL2 (in degrees, converted to radians)
      * Cell size from the CDELT1 header keyword (in arcsec)
      * Flux data from the primary HDU (assumed to be the model image)
  - Sets up the CASA simulator (casatools.simulator) with the antenna configuration, field, feed, times, and spectral window.
  - Simulates an observation and predicts visibilities using the FITS file as the sky model.
"""

import os
import numpy as np
from astropy.io import fits
from casatools import simulator, measures, vpmanager, image
from datetime import datetime


def update_fits_header(fits_file, freq_GHz):
    """
    Update the FITS header of the given file:
      - Set CRVAL3 to the frequency in Hz.
      - Set RESTFRQ to the same value.

    Parameters:
      fits_file : str
          Path to the FITS file to update.
      freq_GHz : str
          Frequency string in the format 'XGHz' (e.g., '2.0GHz').
    """
    # Remove the 'GHz' suffix and convert the remaining string to a float.
    freq_value = float(freq_GHz.rstrip('GHz')) * 1e9  # Convert to Hz.
    # Open the FITS file in update mode.
    hdul = fits.open(fits_file, mode='update')
    # Update the CRVAL3 and RESTFRQ keywords.
    hdul[0].header['CRVAL3'] = freq_value
    hdul[0].header['RESTFRQ'] = freq_value
    # Flush changes to disk and close the file.
    hdul.flush()
    hdul.close()
    print(f"Updated {fits_file}: CRVAL3 and RESTFRQ set to {freq_value} Hz")


def generate_ms(config_file, solar_model, reftime, freqghz=None,
                integration_time=60, msname='fasr.ms', duration=None):
    """
    Generate a Measurement Set (MS) using CASA's simulator tool with the solar model read from a FITS file.

    Parameters:
      config_file : str
          Path to the antenna configuration file (columns: x, y, z, diam, pad).
      solar_model : str
          Path to the solar model FITS file.
      reftime : str
          Reference time (UTC) in CASA format.
      freqghz: str
      integration_time : int
          Integration time in seconds.
      msname : str
          Output MS name.
      duration : int or None
          Total observation duration in seconds (if None, equals integration_time).

    Returns:
      None; the MS is generated and saved under msname.
    """
    # Create CASA simulator and measures tools.
    sm = simulator()
    me = measures()
    vp = vpmanager()
    ia = image()

    # Remove any existing MS with the same name.
    os.system('rm -rf ' + msname)
    sm.open(msname)

    # Read the antenna configuration file.
    # The file is assumed to have comment lines (starting with "#") and columns:
    # x, y, z, dish diameter, antenna name.
    antenna_params = np.genfromtxt(config_file, comments='#')
    x = antenna_params[:, 0]
    y = antenna_params[:, 1]
    z = antenna_params[:, 2]
    dish_dia = antenna_params[:, 3]  # Use dish diameters from the file.
    try:
        ant_names = np.genfromtxt(config_file, comments='#', usecols=(4,), dtype=str)
    except Exception:
        ant_names = ['A' + "{0:02d}".format(i) for i in range(len(x))]

    # Read the solar model FITS file using Astropy.
    hdul = fits.open(solar_model)
    header = hdul[0].header
    flux = hdul[0].data  # Assume the model flux is in the primary HDU.

    # Read frequency from CRVAL3 (Hz) and convert to GHz.
    if freqghz is None:
        freq_Hz = header.get('CRVAL3')
        if freq_Hz is None:
            raise ValueError("Frequency (CRVAL3) not found in FITS header.")
        freq_GHz = f'{freq_Hz / 1e9}GHz'
    else:
        freq_GHz = freqghz

    # Read source RA and DEC from CRVAL1 and CRVAL2 (in degrees) and convert to radians.
    ra_deg = header.get('CRVAL1')
    dec_deg = header.get('CRVAL2')
    if ra_deg is None or dec_deg is None:
        raise ValueError("RA/DEC (CRVAL1/CRVAL2) not found in FITS header.")
    source_ra = np.deg2rad(ra_deg)
    source_dec = np.deg2rad(dec_deg)

    # Read the cell size (in arcsec) from CDELT1.
    cell = header.get('CDELT1')
    hdul.close()

    print("Extracted from FITS header:")
    print(f"Frequency = {freq_GHz}")
    print("Source RA = {:.6f} rad, DEC = {:.6f} rad".format(source_ra, source_dec))

    # Set the antenna configuration.
    sm.setconfig(telescopename="FASR",
                 x=x, y=y, z=z,
                 dishdiameter=dish_dia[0],
                 mount='alt-az',
                 antname=ant_names,
                 padname="FASR",
                 coordsystem='global')

    # Configure the spectral window using the frequency from the FITS header.
    chosen_index = 0
    sm.setspwindow(spwname='Band0',
                   freq=freq_GHz,
                   deltafreq='1MHz',
                   freqresolution='1MHz',
                   nchannels=1,
                   stokes='RR LL')

    # Set the feed configuration.
    sm.setfeed('perfect R L')

    # Set the field using the extracted source RA/DEC.
    sm.setfield(sourcename='Sun',
                sourcedirection=['J2000', f"{source_ra:.22f}rad", f"{source_dec:.22f}rad"])

    # sm.setauto(autocorrwt=0.0)

    sm.settimes(integrationtime=f"{integration_time}s",
                referencetime=me.epoch('UTC', reftime),
                usehourangle=False)

    if duration is None:
        duration = integration_time

    # Define observation start and stop times (centered about the reference time).
    starttime = f"{-duration / 2}s"
    endtime = f"{duration / 2}s"
    sm.observe("Sun", "Band0",
               starttime=starttime,
               stoptime=endtime,
               project="FASR",
               state_obs_mode="")

    sm.setdata(spwid=chosen_index)
    # Use the solar model FITS image as the sky model for prediction.
    dishdiam = np.min(dish_dia)
    vprec = vp.setpbairy(telescope='FASR', dishdiam='{0:.1f}m'.format(dishdiam),
                         blockagediam='0.5m', maxrad='{0:.3f}deg'.format(np.degrees(1.22 * 3e8 / (1e9 * dishdiam))),
                         reffreq=freq_GHz, dopb=True)
    sm.setvp(dovp=True, usedefaultvp=False)
    solar_model_copy = os.path.join(os.path.dirname(msname), os.path.basename(solar_model))
    solar_model_im = os.path.join(os.path.dirname(msname), os.path.basename(solar_model.replace('.fits', '.im')))
    os.system(f'cp -r {solar_model} {solar_model_copy}')
    # if not os.path.exists(solar_model_im):
    update_fits_header(solar_model_copy, freq_GHz)
    ia.fromfits(outfile=solar_model_im, infile=solar_model_copy, overwrite=True)
    ia.close()
    # ia.open(solar_model_im)
    # mycs = ia.coordsys()
    # mycs.setreferencevalue(freq_GHz, 'spectral')
    # mycs.setreferencepixel([0],'spectral')
    # mycs.setincrement('0.01GHz','spectral')
    # ia.setcoordsys(mycs.torecord())
    # print(mycs.torecord())
    # ia.close()
    sm.predict(imagename=solar_model_im)
    sm.setnoise(mode='tsys-atm', trx=500)

    sm.close()
    print("Simulation complete. Measurement set generated:", msname)


def plot_casa_image(image_filename, crop_fraction=(0.0, 1.0), figsize=(10, 7), title='', norm='linear', cmap='viridis'):
    """
    Open a CASA image file, extract pixel data and coordinate system,
    and plot a cropped region of the image with WCS projection.

    Parameters:
      image_filename : str
          Path to the CASA image file.
      crop_fraction : tuple (start, end), optional
          Fraction of the image to plot along each axis (default is (0.25, 0.75)).
      figsize : tuple, optional
          Size of the figure in inches (default is (10, 7)).

    Returns:
      fig, ax : tuple
          Matplotlib figure and axis objects.
    """
    # Open the CASA image and extract data and coordinate system.
    from casatools import image as IA
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    import numpy as np
    ia = IA()
    ia.open(image_filename)
    # Get the image pixel data (assume image has shape [nx, ny, 1, 1])
    pix = ia.getchunk()[:, :, 0, 0]
    csys = ia.coordsys()
    ia.close()

    # Build an Astropy WCS object using the CASA coordinate system.
    rad_to_deg = 180 / np.pi
    w = WCS(naxis=2)
    w.wcs.crpix = csys.referencepixel()['numeric'][0:2]
    w.wcs.cdelt = csys.increment()['numeric'][0:2] * rad_to_deg
    w.wcs.crval = csys.referencevalue()['numeric'][0:2] * rad_to_deg
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']

    # Determine the cropping indices.
    p1 = int(pix.shape[0] * crop_fraction[0])
    p2 = int(pix.shape[0] * crop_fraction[1])
    cropped = pix[p1:p2, p1:p2].transpose()  # transpose for correct orientation

    # Plot the cropped image using the WCS projection.
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': w})
    im = ax.imshow(cropped, origin='lower', cmap=plt.get_cmap(cmap), norm=norm, vmax=np.nanpercentile(cropped, 99.99))
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_title(title)

    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
from casatools import image as IA
from astropy.wcs import WCS


def plot_two_casa_images_with_convolution(image1_filename, image2_filename,
                                          crop_fraction=(0.0, 1.0),
                                          figsize=(15, 4),
                                          title1='First Image',
                                          title2='Second (Convolved) Image',
                                          compare_two=False,
                                          contour_levels=None, cmap='viridis',
                                          overwrite_conv=True, vmax=99.9, vmin=0,
                                          vmax2=None, vmin2=None, ):
    """
    Open two CASA images using casatools.image (IA), convolve the second image
    with the restoring beam from the first image, and plot them side-by-side.

    The left panel shows the (optionally cropped) first image. The right panel shows
    the second image after convolution with the restoring beam from the first image,
    with contours from the first image overlaid.

    Parameters:
      image1_filename : str
          Path to the first CASA image file.
      image2_filename : str
          Path to the second CASA image file.
      crop_fraction : tuple of float, optional
          Fractional start and end indices (e.g., (0.0, 1.0) uses the full image).
      figsize : tuple, optional
          The figure size in inches.
      title1 : str, optional
          Title for the left panel.
      title2 : str, optional
          Title for the right panel.
      contour_levels : array-like or None, optional
          Contour levels to overlay on the second panel.
          If None, levels are set to default percentiles of the first image.

    Returns:
      fig, axs : tuple
          Matplotlib figure and axes objects.
    """
    if not compare_two:
        figsize = (figsize[0] / 3 * 2, figsize[1])
    ia = IA()
    # --- Open the first image and extract data, coordinate system, and restoring beam ---
    ia.open(image1_filename)
    pix1 = ia.getchunk()[:, :, 0, 0]  # assume image shape [nx, ny, 1, 1]
    csys1 = ia.coordsys()
    beam = ia.restoringbeam()  # e.g., returns {'major': {'value': 6.0, 'unit': 'arcsec'},
    #                   'minor': {'value': 6.0, 'unit': 'arcsec'},
    #                   'positionangle': {'value': 0.0, 'unit': 'deg'}}
    ia.close()

    # Build an Astropy WCS object using the CASA coordinate system from image1.
    rad_to_deg = 180.0 / np.pi
    w = WCS(naxis=2)
    w.wcs.crpix = csys1.referencepixel()['numeric'][0:2]
    w.wcs.cdelt = np.array(csys1.increment()['numeric'][0:2]) * rad_to_deg
    w.wcs.crval = np.array(csys1.referencevalue()['numeric'][0:2]) * rad_to_deg
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']

    # Generate an output filename for the convolved image.
    output_filename = image2_filename.replace('.im', '.im.convolved')

    # --- Convolve the second image with the restoring beam from image1 using IA.convolve2d ---
    # Format beam parameters as strings.
    major = f"{beam['major']['value']}{beam['major']['unit']}"
    minor = f"{beam['minor']['value']}{beam['minor']['unit']}"
    pa = f"{beam['positionangle']['value']}{beam['positionangle']['unit']}"

    if overwrite_conv or not os.path.exists(output_filename):
        # Open the second image and apply convolution.
        ia.open(image2_filename)
        ia.convolve2d(outfile=output_filename, axes=[0, 1], type='gauss',
                      major=major, minor=minor, pa=pa, overwrite=True)
        ia.close()

    # --- Read the convolved image to extract its pixel data ---
    ia.open(output_filename)
    pix2 = ia.getchunk()[:, :, 0, 0]
    ia.close()

    # --- Crop both images using the same crop_fraction ---
    shape = pix1.shape[0]  # assume square images.
    if isinstance(crop_fraction[0], float):
        p1 = int(shape * crop_fraction[0])
        p2 = int(shape * crop_fraction[1])
        cropped1 = pix1[p1:p2, p1:p2]
        cropped2 = pix2[p1:p2, p1:p2]
    elif isinstance(crop_fraction[0], tuple) or isinstance(crop_fraction[0], list):
        px1 = int(shape * crop_fraction[0][0])
        px2 = int(shape * crop_fraction[0][1])
        py1 = int(shape * crop_fraction[1][0])
        py2 = int(shape * crop_fraction[1][1])
        cropped1 = pix1[px1:px2, py1:py2]
        cropped2 = pix2[px1:px2, py1:py2]
    else:
        cropped1 = pix1
        cropped2 = pix2

    # --- Plotting: Create a figure with two panels using the WCS projection from image1 ---
    if compare_two:
        fig, axs = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': w})
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': w})

    vmax_val = np.nanmax(cropped2) * float(vmax) / 100
    vmin_val = np.nanmax(cropped2) * float(vmin) / 100
    if vmax2 is None:
        vmax2 = vmax
    if vmin2 is None:
        vmin2 = vmin
    vmax_val2 = np.nanmax(cropped2) * float(vmax2) / 100
    vmin_val2 = np.nanmax(cropped2) * float(vmin2) / 100
    # Left panel: Image1 (original)
    ax1 = axs[0]
    im1 = ax1.imshow(cropped1.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                     vmax=vmax_val, vmin=vmin_val)
    ax1.set_xlabel('Right Ascension')
    ax1.set_ylabel('Declination')
    ax1.set_title(title1)
    plt.colorbar(im1, ax=ax1)

    # Right panel: Convolved Image2 as background.
    ax2 = axs[-1]
    im2 = ax2.imshow(cropped2.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                     vmax=vmax_val2, vmin=vmin_val2)
    ax2.set_xlabel('Right Ascension')
    ax2.set_ylabel('Declination')
    ax2.set_title(title2)
    plt.colorbar(im2, ax=ax2)

    # Overlay contours from image1 onto the right panel.
    if compare_two:
        ax_comp = axs[1]
        im2 = ax_comp.imshow(cropped2.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                             vmax=vmax_val2, vmin=vmin_val2)
        ax_comp.set_xlabel('Right Ascension')
        ax_comp.set_ylabel('Declination')

        plt.colorbar(im2, ax=ax_comp)
        if contour_levels is None:
            contour_levels = np.linspace(0.1, 0.9, 5) * np.nanmax(cropped1)
        else:
            contour_levels = np.array(contour_levels) * np.nanmax(cropped1)
        cs = axs[1].contour(cropped1.transpose(), levels=contour_levels, colors='tab:cyan', origin='lower',
                            linewidths=0.5)
        ax_comp.set_title(f'Contour: Left Panel, Background: Right Panel')

    plt.tight_layout()
    return fig, axs


def plot_two_casa_images(image1_filename, image2_filename,
                         crop_fraction=(0.0, 1.0),
                         figsize=(15, 4),
                         title1='First Image',
                         title2='Second Image',
                         compare_two=False,
                         contour_levels=None, cmap='viridis',
                         vmax=99.9, vmin=0,
                         uni_vmaxmin=False):
    """
    Open two CASA images using casatools.image (IA), convolve the second image
    with the restoring beam from the first image, and plot them side-by-side.

    The left panel shows the (optionally cropped) first image. The right panel shows
    the second image after convolution with the restoring beam from the first image,
    with contours from the first image overlaid.

    Parameters:
      image1_filename : str
          Path to the first CASA image file.
      image2_filename : str
          Path to the second CASA image file.
      crop_fraction : tuple of float, optional
          Fractional start and end indices (e.g., (0.0, 1.0) uses the full image).
      figsize : tuple, optional
          The figure size in inches.
      title1 : str, optional
          Title for the left panel.
      title2 : str, optional
          Title for the right panel.
      contour_levels : array-like or None, optional
          Contour levels to overlay on the second panel.
          If None, levels are set to default percentiles of the first image.

    Returns:
      fig, axs : tuple
          Matplotlib figure and axes objects.
    """
    if not compare_two:
        figsize = (figsize[0] / 3 * 2, figsize[1])
    ia = IA()
    # --- Open the first image and extract data, coordinate system, and restoring beam ---
    ia.open(image1_filename)
    pix1 = ia.getchunk()[:, :, 0, 0]  # assume image shape [nx, ny, 1, 1]
    csys1 = ia.coordsys()
    # beam = ia.restoringbeam()  # e.g., returns {'major': {'value': 6.0, 'unit': 'arcsec'},
    #                   'minor': {'value': 6.0, 'unit': 'arcsec'},
    #                   'positionangle': {'value': 0.0, 'unit': 'deg'}}
    ia.close()

    # Build an Astropy WCS object using the CASA coordinate system from image1.
    rad_to_deg = 180.0 / np.pi
    w = WCS(naxis=2)
    w.wcs.crpix = csys1.referencepixel()['numeric'][0:2]
    w.wcs.cdelt = np.array(csys1.increment()['numeric'][0:2]) * rad_to_deg
    w.wcs.crval = np.array(csys1.referencevalue()['numeric'][0:2]) * rad_to_deg
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']

    # # Generate an output filename for the convolved image.
    # output_filename = image2_filename.replace('.im', '.im.convolved')
    #
    # # --- Convolve the second image with the restoring beam from image1 using IA.convolve2d ---
    # # Format beam parameters as strings.
    # major = f"{beam['major']['value']}{beam['major']['unit']}"
    # minor = f"{beam['minor']['value']}{beam['minor']['unit']}"
    # pa = f"{beam['positionangle']['value']}{beam['positionangle']['unit']}"

    # if overwrite_conv or not os.path.exists(output_filename):
    #     # Open the second image and apply convolution.
    #     ia.open(image2_filename)
    #     ia.convolve2d(outfile=output_filename, axes=[0, 1], type='gauss',
    #                   major=major, minor=minor, pa=pa, overwrite=True)
    #     ia.close()

    # --- Read the convolved image to extract its pixel data ---
    ia.open(image2_filename)
    pix2 = ia.getchunk()[:, :, 0, 0]
    ia.close()

    # --- Crop both images using the same crop_fraction ---
    shape = pix1.shape[0]  # assume square images.
    if isinstance(crop_fraction[0], float):
        p1 = int(shape * crop_fraction[0])
        p2 = int(shape * crop_fraction[1])
        cropped1 = pix1[p1:p2, p1:p2]
        cropped2 = pix2[p1:p2, p1:p2]
    elif isinstance(crop_fraction[0], tuple) or isinstance(crop_fraction[0], list):
        px1 = int(shape * crop_fraction[0][0])
        px2 = int(shape * crop_fraction[0][1])
        py1 = int(shape * crop_fraction[1][0])
        py2 = int(shape * crop_fraction[1][1])
        cropped1 = pix1[px1:px2, py1:py2]
        cropped2 = pix2[px1:px2, py1:py2]
    else:
        cropped1 = pix1
        cropped2 = pix2

    # --- Plotting: Create a figure with two panels using the WCS projection from image1 ---
    if compare_two:
        fig, axs = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': w})
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': w})

    vmax_val1 = np.nanmax(cropped1) * float(vmax) / 100
    vmin_val1 = np.nanmax(cropped1) * float(vmin) / 100
    if uni_vmaxmin:
        vmax_val2 = vmax_val1
        vmin_val2 = vmin_val1
    else:
        vmax_val2 = np.nanmax(cropped2) * float(vmax) / 100
        vmin_val2 = np.nanmax(cropped2) * float(vmin) / 100
    # Left panel: Image1 (original)
    ax1 = axs[0]
    im1 = ax1.imshow(cropped1.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                     vmax=vmax_val1, vmin=vmin_val1)
    ax1.set_xlabel('Right Ascension')
    ax1.set_ylabel('Declination')
    ax1.set_title(title1)
    plt.colorbar(im1, ax=ax1)

    # Right panel: Convolved Image2 as background.
    ax2 = axs[-1]
    im2 = ax2.imshow(cropped2.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                     vmax=vmax_val2, vmin=vmin_val2)
    ax2.set_xlabel('Right Ascension')
    ax2.set_ylabel('Declination')
    ax2.set_title(title2)
    plt.colorbar(im2, ax=ax2)

    # Overlay contours from image1 onto the right panel.
    if compare_two:
        ax_comp = axs[1]
        im2 = ax_comp.imshow(cropped2.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                             vmax=vmax_val2, vmin=vmin_val2)
        ax_comp.set_xlabel('Right Ascension')
        ax_comp.set_ylabel('Declination')

        plt.colorbar(im2, ax=ax_comp)
        if contour_levels is None:
            contour_levels = np.linspace(0.1, 0.9, 5) * np.nanmax(cropped1)
        else:
            contour_levels = np.array(contour_levels) * np.nanmax(cropped1)
        cs = axs[1].contour(cropped1.transpose(), levels=contour_levels, colors='tab:cyan', origin='lower',
                            linewidths=0.5)
        ax_comp.set_title(f'Contour: Left Panel, Background: Right Panel')

    plt.tight_layout()
    return fig, axs
