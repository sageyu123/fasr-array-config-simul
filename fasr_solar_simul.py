from __future__ import annotations
from astropy.wcs import WCS
from casatools import image as IA
from datetime import datetime
from casatools import simulator, measures, vpmanager, image
from astropy.io import fits
from itertools import combinations
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.constants import c
import os
import glob
from casatools import vpmanager, quanta
from scipy.stats import binned_statistic
from scipy.signal import fftconvolve
from scipy.ndimage import shift as ndimage_shift
import sunpy.map
from scipy.spatial.distance import pdist
import re
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
import astropy.units as u
import time
import shutil
from functools import wraps
from matplotlib.patches import Ellipse, Circle
from scipy.spatial import distance


def make_msname(project: str,
                target: str,
                freq: str,
                reftime: datetime,
                duration: int,
                integration: int,
                config: str,
                noise: str = None,
                ) -> str:
    msfilepath = os.path.join(project, 'msfiles', target)
    if not os.path.exists(msfilepath):
        os.makedirs(msfilepath)
    return os.path.join(msfilepath,
                        f'fasr_{os.path.basename(config.rstrip(".cfg"))}_{target}_{freq}_{reftime.strftime("%Y%m%dT%HUT")}_dur{duration:.0f}s_int{integration:.0f}s_noise{noise:.0f}K')


def make_imname(msname: str,
                deconvolver: str = 'hogbom',
                phaerr: float = None,
                amperr: float = None
                ) -> str:
    parts = [msname]
    if phaerr is not None:
        phaerr_deg = np.degrees(phaerr)
        parts.append(f'phaerr{phaerr_deg:.1f}deg')
    if amperr is not None:
        parts.append(f'amperr{np.int_(amperr * 100)}pct')
    parts.append(deconvolver)
    return '_'.join(parts)


def format_duration(seconds):
    """Format seconds into appropriate unit string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} minutes"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.2f} hours"
    days = hours / 24
    return f"{days:.2f} days"


def runtime_report(func):
    """Decorator to report runtime of a function and log completion time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"'{func.__name__}' completed at {now}; "
            f"runtime: {format_duration(duration)}"
        )
        return result

    return wrapper


qa = quanta()
vp = vpmanager()

# Speed of light in m/s
C_LIGHT = c


def jy2mk(I, nu, res_arcsec):
    '''
    Convert intensity in Jy to brightness temperature in MK.

    Parameters:
        I: intensity in Jy/beam
        nu: frequency in GHz
        res_arcsec: angular resolution in arcseconds

    Returns:
        Tb: brightness temperature in MK
    '''
    Tb = 1.22e6 * I / (nu * res_arcsec) ** 2 * 1e-6
    return Tb


def mk2jy(Tb, nu, res_arcsec):
    '''
    Convert brightness temperature in MK to intensity in Jy/beam.

    Parameters:
        Tb: brightness temperature in MK
        nu: frequency in GHz
        res_arcsec: angular resolution in arcseconds

    Returns:
        I: intensity in Jy/beam
    '''
    I = Tb * (nu * res_arcsec) ** 2 * 1e6 / 1.22e6
    return I


def sfu2tb(freq, flux, size, square=True, reverse=False):
    '''
    freq: single element or array, in Hz
        flux: single element or array of flux, in sfu; if reverse, it is brightness temperature in K
        size: width of the radio source, [major,minor], in arcsec
        reverse: if True, convert brightness temperature in K to flux in sfu integrated uniformly withing the size
    '''
    sfu2cgs = 1e-19
    vc = 2.998e10
    kb = 1.38065e-16
    if square:
        sr = 4. * (size[0] / 206265. / 2.) * (size[1] / 206265. / 2.)
    else:
        sr = np.pi * (size[0] / 206265. / 2.) * (size[1] / 206265. / 2.)  # circular area
    factor = sfu2cgs * vc ** 2. / (2. * kb * freq ** 2. * sr)
    if reverse:
        # returned value is flux in sfu
        return flux / factor
    else:
        # returned value is brightness temperature in K
        return flux * factor


def tb_quietsun(fghz=1.):
    """
    Inputs:
    fghz: frequency in GHz
    Return:
    Tb_sun: brightness temperature of the Sun following Zirin 1991
    """
    ## brightness temperature spectrum, taken from Zirin 1991
    # Valid for ~1 GHz and up
    return (140077. * fghz ** (-2.1) + 10880.)


def tant_quiet_sun_calc(dish_diameter=1.5, fghz=1., eta_a=0.6):
    """
    This is a crude estimate for antenna temperature on the Sun by simply comparing
    the primary beam size with the size of the Sun.
    Better to use rstn2ant() instead
    Inputs:
    dishd: dish diameter in m
    fghz: frequency in GHz
    eta_a: antenna aperture efficiency
    Return:
    Tant: antenna temperature at the given frequency
    """
    R_sun = 6.96e10
    AU = 1.496e13
    clight = 2.998e10
    Tb_sun = tb_quietsun(fghz)
    Omega_sun = np.pi * (R_sun / AU) ** 2.
    Omega_ant = np.pi * ((clight / (fghz * 1e9)) / (dish_diameter * 1e2) / 2.) ** 2.
    # Beam ratio is defined as the ratio between the solid angle of the source to the primary beam
    beam_ratio_ = Omega_sun / Omega_ant
    # The beam ratio becomes unity when the sun fills the primary beam
    beam_ratio = np.minimum(beam_ratio_, np.ones_like(beam_ratio_))
    Tant = eta_a * beam_ratio * Tb_sun
    return Tant


def total_flux_to_tant(total_flux, eta_a=0.6, dish_diameter=2.):
    '''
    Convert total flux (in sfu) to antenna temperature (in K)
    Inputs:
    total_flux: total power in sfu
    eta_a: antenna aperture efficiency
    dish_diameter: dish diameter in m
    Return:
    Tant: antenna temperature in K
    '''
    k_b = 1.38065e-16  # Boltzmann constant in erg/K
    A_e = eta_a * np.pi * ((dish_diameter * 1e2) / 2.) ** 2.  # effective area in cm^2
    Tant = total_flux * 1e-19 * A_e / (2 * k_b)
    return Tant


def get_baseline_lengths(filename):
    '''
    Computes the baseline length given a configuration file. Assumes source is at zenith
    '''
    data = np.loadtxt(filename, usecols=(0, 1))
    return pdist(data)


def get_max_resolution(freq, max_baseline):
    '''
    freq in GHz, max_baseline in m
    '''
    wavelength = 299792458 / (freq * 1e9)
    res = wavelength / max_baseline * 180 / 3.14159 * 3600
    return res  # in arcsec


def resolve_casa_image_path(path_hint):
    """Resolve a CASA image path from a base name or `.image` path."""
    if path_hint is None:
        return None
    candidates = [
        path_hint,
        path_hint + '.image',
        path_hint[:-6] if str(path_hint).endswith('.image') else None,
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0]


def humanize_array_config_label(cfg_file):
    """Return a compact display label for a FASR config filename."""
    base = os.path.basename(str(cfg_file)).replace('.cfg', '')
    parts = base.split('_')
    lower = [p.lower() for p in parts]
    if 'hybrid' in lower and '120' in lower:
        remote_count = next((p[6:] for p in lower if p.startswith('remote') and len(p) > 6), None)
        range_token = next((p for p in lower if p.startswith('r') and p.endswith('km') and '-' in p), None)
        if remote_count and range_token:
            range_label = range_token[1:].replace('km', ' km').replace('-', '–')
            return f'Hybrid 120 + {remote_count} remote ({range_label})'
        return 'Hybrid 120'
    return base


def simple_array_config_label(cfg_file):
    """Return a short display label for a FASR config filename.

    :param cfg_file: Config filename or path.
    :type cfg_file: str
    :returns: Short config label without remote-range details.
    :rtype: str
    """
    base = os.path.basename(str(cfg_file)).replace('.cfg', '')
    parts = base.split('_')
    lower = [p.lower() for p in parts]
    if 'hybrid' in lower and '120' in lower:
        remote_count = next((p[6:] for p in lower if p.startswith('remote') and len(p) > 6), None)
        if remote_count:
            return f'Hybrid 120 + {remote_count} remote'
        return 'Hybrid 120'
    return base


def base_tag_from_path(path):
    """Return a stable basename tag for CASA image or FITS-like paths."""
    return os.path.basename(str(path)).replace('.image', '').replace('.im', '')


def crop_fraction_tag(crop_fraction):
    """Return a filename-safe tag describing a crop tuple."""
    if isinstance(crop_fraction[0], float):
        return f"crop{int(crop_fraction[0] * 100):02d}-{int(crop_fraction[1] * 100):02d}"
    if isinstance(crop_fraction[0], (tuple, list)):
        return (
            f"cropx{int(crop_fraction[0][0] * 100):02d}-{int(crop_fraction[0][1] * 100):02d}"
            f"_y{int(crop_fraction[1][0] * 100):02d}-{int(crop_fraction[1][1] * 100):02d}"
        )
    return 'cropfull'


def freqstr_to_hz(freqstr):
    """Parse a frequency string like `05.15GHz` into Hz."""
    s = str(freqstr).strip().lower()
    if s.endswith('ghz'):
        return float(s[:-3]) * 1e9
    if s.endswith('mhz'):
        return float(s[:-3]) * 1e6
    if s.endswith('khz'):
        return float(s[:-3]) * 1e3
    if s.endswith('hz'):
        return float(s[:-2])
    raise ValueError(f'Unrecognized frequency string: {freqstr}')


def remove_path(path):
    """Remove a file or directory if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def clear_casa_imagename(imagename_base):
    """Remove CASA image sidecar products for a given base name."""
    for path in glob.glob(imagename_base + '.*'):
        remove_path(path)


def clear_casa_imagename_if_shape_mismatch(imagename_base, imsize):
    """Remove CASA image sidecars when the existing image shape does not match.

    :param imagename_base: CASA imagename base without extension.
    :type imagename_base: str
    :param imsize: Requested CASA image size.
    :type imsize: int or list or tuple
    :returns: True if stale products were removed, else False.
    :rtype: bool
    """
    probe_paths = [
        imagename_base + ext
        for ext in ('.image', '.psf', '.residual', '.model', '.pb', '.mask')
    ]
    existing_probe = next((p for p in probe_paths if os.path.exists(p)), None)
    if existing_probe is None:
        return False

    if isinstance(imsize, (list, tuple, np.ndarray)):
        if len(imsize) == 0:
            return False
        expected_shape = tuple(int(v) for v in np.atleast_1d(imsize)[:2])
        if len(expected_shape) == 1:
            expected_shape = (expected_shape[0], expected_shape[0])
    else:
        expected_shape = (int(imsize), int(imsize))

    ia_local = IA()
    try:
        ia_local.open(existing_probe)
        actual_shape = tuple(int(v) for v in ia_local.shape()[:2])
    except Exception:
        return False
    finally:
        try:
            ia_local.close()
        except Exception:
            pass

    if actual_shape != expected_shape:
        clear_casa_imagename(imagename_base)
        return True
    return False


def casa_safe_scales(scales):
    """Return CASA-safe integer multiscale list."""
    try:
        values = np.atleast_1d(scales).tolist()
    except Exception:
        values = [0, 6, 20]
    out = []
    for value in values:
        try:
            out.append(int(round(float(value))))
        except Exception:
            continue
    return out if out else [0, 6, 20]


def calc_noise(tsys, array_config_file, dish_diameter=None, total_flux=None, duration=10., integration_time=1.,
               channel_width_mhz=10., nchannel=1, eta_q=0.93, eta_a=0.6, freqghz='1GHz', uv_cell=None, verbose=False):
    """
    Calculate the noise level for a given array configuration.

    Parameters:
      tsys : float
          System noise temperature in K (e.g., '300'). This should include both the system temperature
            and the source-induced noise temperature. See https://ui.adsabs.harvard.edu/abs/2025SoPh..300...91B/abstract
      array_config_file : str
          Path to the antenna configuration file.
      duration : float
          Total observation duration in seconds.
      integration_time : float
          Single integration time in seconds.
      channel_width_mhz : float
            Channel width in MHz.
      nchannel: int
            Number of channels (default is 1).
      eta_q: float
            Digitizer quantization efficiency (default is 0.93 for 8-bit sampling).
      eta_a: float
            Antenna efficiency (default is 0.7).
      freqghz: float or str
            Observing frequency in GHz (e.g., '1GHz' or 1.0).
      uv_cell: float
            UV cell size in number of wavelengths. It is recommended to use the CLEAN parameters to calculate it as
            uv_cell = 1 / (imsize * cellsize in radians).
            If None, defaults to twice the dish diameter expressed in wavelengths at the observing frequency.

    Reference: https://casaguides.nrao.edu/index.php/Simulating_ngVLA_Data-CASA5.4.1#Estimating_the_Scaling_Parameter_for_Adding_Thermal_Noise

    Returns:
      noisejy  : float
        Calculated noise level per baseline per chanel per polarization per integration in Jy.
      sigma_na : float
        Naturally weighted point source sensitivity in Jy/beam.
      sigma_un : float
        Uniformly weighted point source sensitivity in Jy/beam.
    """
    # Get baseline lengths from the antenna configuration file.
    baseline_lengths = get_baseline_lengths(array_config_file)
    positions, _ = read_casa_antenna_list(array_config_file)
    if dish_diameter is None:
        antenna_params = np.genfromtxt(array_config_file, comments='#')
        dish_diameter = antenna_params[0, 3]
    n_ant = len(positions)
    n_bl = len(baseline_lengths)
    n_vis = n_ant * (n_ant - 1)

    c = 2.998e8  # speed of light in m/s
    if isinstance(freqghz, str):
        freq_hz = float(re.findall(r"[-+]?\d*\.\d+|\d+", freqghz)[0]) * 1e9
    elif isinstance(freqghz, (float, int)):
        freq_hz = float(freqghz) * 1e9
    else:
        raise ValueError("freqghz must be a string or a float/int representing frequency in GHz")
    lam = c / freq_hz  # Wavelength in meters
    if uv_cell is None:
        uv_cell = dish_diameter * 2.0 / lam  # Default UV cell size in meters

    # 2. Generate UV Coverage (Snapshot at Zenith)
    # We only need the relative distribution, so Zenith snapshot is a good proxy
    # for the instantaneous density.
    u_list = []
    v_list = []

    for i in range(n_ant):
        for j in range(n_ant):
            if i != j:
                u = (positions[i, 0] - positions[j, 0]) / lam
                v = (positions[i, 1] - positions[j, 1]) / lam
                u_list.append(u)
                v_list.append(v)

    u = np.array(u_list)
    v = np.array(v_list)

    # Create a 2D Histogram (The UV Grid)
    # We define the range to cover the max baseline
    uv_max = np.max(np.abs(u))
    bins = int((2 * uv_max) / uv_cell)
    #print(f"UV Grid: {bins} x {bins} cells covering +/- {uv_max:.1f} wavelengths")

    # Calculate density map
    counts, _, _ = np.histogram2d(u, v, bins=bins, range=[[-uv_max, uv_max], [-uv_max, uv_max]])

    # 4. Assign Weights
    # For every visibility, find which bin it falls into and get the density
    # NOTE: A faster way for estimation is to sum over the GRID, not the visibilities.
    # Sum(w_i) is roughly Sum(1/density * density) = Number of occupied cells.

    # Only consider cells that have data
    valid_cells = counts[counts > 0]

    # Natural Weighting Simulation
    # Sum of weights = Total Visibilities

    # Uniform Weighting Simulation
    # Weight for points in a cell = 1 / count
    # Sum of weights over all visibilities = Sum ( (1/count) * count ) over all filled cells
    #                                     = Number of filled cells
    sum_w_uni = len(valid_cells)

    # Sum of weights squared over all visibilities:
    # For a cell with 'k' hits, we have 'k' visibilities each with weight (1/k).
    # Contribution = k * (1/k)^2 = 1/k.
    sum_w2_uni = np.sum(1.0 / valid_cells)

    # 5. Calculate Factors
    # SEFD_term cancels out. We look at the ratio of the coefficients.

    # Natural Noise Factor (Reference = 1.0)
    # sigma_nat ~ sqrt( sum(1^2) / sum(1)^2 ) = sqrt( N_VIS / N_VIS^2 ) = 1/sqrt(N_VIS)
    raw_sens_nat = np.sqrt(n_vis / (n_vis ** 2))

    # Uniform Noise Factor
    raw_sens_uni = np.sqrt(sum_w2_uni / (sum_w_uni ** 2))

    # The Ratio (How much worse is Uniform?)
    amplification_factor = raw_sens_uni / raw_sens_nat
    if verbose:
        print(f"Number of antennas: {n_ant}")
        print(f"Total Baselines: {n_bl}")
        print(f"Total Visibilities: {n_vis}")
        print(f"Occupied UV Cells: {len(valid_cells)}")
        print(f"Number of visibilities in cells: {np.sum(valid_cells)}")
        print(f"Mean Density: {np.mean(valid_cells):.1f} points per cell")
        print(f"-" * 30)
        print(f"Noise Amplification Factor: {amplification_factor:.2f}x")
        print(f"Sensitivity Loss: {(1 - 1 / amplification_factor) * 100:.1f}%")

    # Estimate antenna temperature
    if total_flux is None:
        print('Use tant_quiet_sun_calc to estimate antenna temperature on the quiet Sun')
        tant = tant_quiet_sun_calc(fghz=freqghz, dish_diameter=dish_diameter, eta_a=eta_a)
    else:
        print(total_flux)
        print(f'Calculate antenna temperature from total flux {total_flux} sfu incident on the dish')
        tant = total_flux_to_tant(total_flux, eta_a=eta_a, dish_diameter=dish_diameter)

    t_total = tsys + tant  # Total noise temperature in K
    print(f'Total noise temperature (K): {t_total:.3e} K')

    # Calculate System Equivalent Flux Density (SEFD) using the provided noise temperature.
    sefd = 2 * 1.38e-16 * t_total / 1e-23 / eta_a / eta_q / (np.pi * (dish_diameter / 2 * 1e2) ** 2)
    print(f'Estimated SEFD: {sefd:.3e} Jy')  # in Jy

    # Calculate Stokes I naturally weighted point source sensitivity for the entire duration and bandwidth
    sigma_na = sefd / np.sqrt(2 * n_vis * duration * (nchannel * channel_width_mhz * 1e6))  # in Jy/beam
    print(f"Estimated natural weighting point source sensitivity sigma_na: {sigma_na:.3e} Jy/beam")
    sigma_un = amplification_factor * sigma_na
    print(f"Estimated uniform weighting point source sensitivity sigma_un: {sigma_un:.3e} Jy/beam")

    # Calculate noise per baseline per channel per polarization per integration, which will be used to corrupt the visibilities
    n_int = duration / integration_time
    noisejy = sigma_na * np.sqrt(nchannel * 2 * len(baseline_lengths) * n_int)
    print(
        f"Estimated noise per baseline per channel per polarization per integration: {noisejy:.3e} Jy for {n_bl} baselines")
    return noisejy, sigma_na, sigma_un


@runtime_report
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


@runtime_report
def generate_log_spiral_antenna_positions(n_arms=3, antennas_per_arm=24, r0=35, r_max=2000,
                                          k=0.43, gamma=1.0, n_turn=None,
                                          latitude=39.54780, clockwise=True,
                                          add_perturb=True, theta_perturb=0.02, r_perturb=0.02):
    """
    Generate antenna positions using a multi-arm logarithmic spiral layout.

    The spiral is defined by:
         r = r0 * exp( k * theta**gamma )
    where k is chosen so that r reaches r_max when theta = n_turn * 2π.

    If the first antenna along each arm has a radial distance less than minimum_dist,
    it is removed from the beginning and appended to the tail of that arm.

    Parameters:
      n_arms : int
          Number of spiral arms.
      antennas_per_arm : int
          Number of antennas per arm.
      r0 : float
          Starting radius (m).
      r_max : float
          Maximum radius (m) at the outer end of each arm.
      k  : float
          Scaling factor for the spiral growth.
      gamma : float
          Exponent for modifying the spiral curvature.
      n_turn : float
          Number of spiral turns (default is 1.0). If defined, it overides k by calculating it.
      latitude : float
          Latitude (in degrees) to adjust the aspect ratio between east (x) and north (y)
          via a factor of 1/cos(latitude).
      clockwise : bool
            If True, spiral winds clockwise; otherwise counter-clockwise.
      add_perturb : bool
            If True, add random perturbations to antenna positions.
      theta_perturb : float
            Standard deviation of angular perturbation (radians).
      r_perturb : float
            Standard deviation of radial perturbation (fractional).

    Returns:
      positions : numpy.ndarray
          Array of shape ((n_arms*antennas_per_arm + 1), 2) containing the (x, y) antenna positions in meters.
          A central antenna at (0,0) is appended.

    """

    if n_turn:
        # Compute theta_max (total angle) based on n_turn.
        print(f'n_turn is defined to be {n_turn}, overriding k')
        theta_max = n_turn * 2 * np.pi
        # Compute k so that when theta=theta_max, r = r_max.
        k = np.log(r_max / r0) / (theta_max ** gamma)
        print('spiral winding parameter k', k)
    else:
        # Compute theta_max based on k and r_max.
        theta_max = (np.log(r_max / r0) / k) ** (1 / gamma)
        print('Number of turns', theta_max / (2 * np.pi))

    # Create an array of indices along a single arm.
    i_vals = np.arange(antennas_per_arm)
    # Compute the array of angles for one arm.
    # shape: (antennas_per_arm,)
    theta_arm = (theta_max * i_vals) / (antennas_per_arm - 1)
    # Calculate radial distances for these angles.
    # shape: (antennas_per_arm,)
    r_vals = r0 * np.exp(k * (theta_arm ** gamma))
    # print(r_vals)

    # Create an array for the arm indices.
    arm_indices = np.arange(n_arms)
    # Compute theta offset for each arm.
    theta_offset = (2 * np.pi * arm_indices) / n_arms  # shape: (n_arms,)

    # Broadcast to compute total angle for each antenna on every arm.
    # shape: (n_arms, antennas_per_arm)
    theta_total = -theta_arm[None, :] + theta_offset[:, None]
    if not clockwise:
        theta_total = -theta_total
    # Broadcast radial values.
    r_matrix = r_vals[None, :]  # shape: (1, antennas_per_arm)
    # BC: why the innermost antenna on each arm is offset by pi/4?
    #theta_total[:, 0] += np.pi / 4
    if add_perturb:
        delta_theta = np.random.normal(0., theta_perturb, antennas_per_arm)
        delta_r = np.random.normal(0., r_perturb, antennas_per_arm) * r_matrix
        r_matrix += delta_r
        theta_total += delta_theta

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

    bl = distance.pdist(np.transpose(np.array([positions[:, 0], positions[:, 1]])), 'euclidean')
    print('Minimum baseline length: {0:.1f} m'.format(np.min(bl)))
    print('Maximum baseline length: {0:.1f} m'.format(np.max(bl)))

    # print(f'fig-fasr_Log_Spiral-{len(positions)}_n_arms={n_arms}, antennas_per_arm={antennas_per_arm}, alpha={alpha:.2f}, gamma={gamma:.2f}, r0={r0:.1f}, r_max={r_max:.0f}, n_turn={n_turn:.1f}')
    return positions


def generate_hybrid_rand_spiral_array(n_antennas=120, r_max=2000, min_dist=1.8, n_core=48,
                                      r_start=35.0, n_turns=1.5, rotation_offset=0., sigma_g=15.2,
                                      latitude=39.5846):
    """
    Generates a hybrid Gaussian Core + 3-Arm Log Spiral array.

    Parameters:
        n_antennas: Total number of antennas
        r_max: Maximum radius of the array (meters)
        min_dist: Minimum physical distance between antennas (collision avoidance)
        core_fraction: Fraction of antennas to place in the Gaussian core
    """

    n_spiral0 = n_antennas - n_core

    antennas = []

    # --- 1. Generate Spiral Arms (Outer Resolution) ---
    # We generate this first to determine the "handoff" radius
    n_arms = 3
    ants_per_arm = n_spiral0 // n_arms

    print('Generating spiral with {0} arms, {1} antennas per arm'.format(n_arms, ants_per_arm))
    print('Total spiral antennas: {0}'.format(ants_per_arm * n_arms))
    n_core = n_antennas - (ants_per_arm * n_arms)
    print('Adjusted core antennas to: {0}'.format(n_core))

    # Apply rotation offset to the phase
    phase_shift = np.deg2rad(rotation_offset)

    # Spiral parameters
    # We want the spiral to start where the core creates a "shelf"
    # Transition radius ~ 10% of total size is a good rule of thumb for solar
    r_end = r_max

    # Logarithmic spacing constant
    # r = r_0 * e^(k * theta)
    # theta_max chosen to allow arms to wrap somewhat (e.g., 2 full turns) or just expand
    theta_start = 0
    theta_total = n_turns * 2. * np.pi  # Wrap amount

    # b calculation for r = r_start * e^(k * theta)
    # r_end = r_start * e^(b * theta_total) -> b = ln(r_end/r_start) / theta_total
    b = np.log(r_end / r_start) / theta_total

    for i in range(n_arms):
        arm_phase = (2 * np.pi / n_arms) * i + phase_shift
        for j in range(ants_per_arm):
            # Logarithmic distribution of angles for sampling
            # We want denser sampling at inner radii, so we space theta linearly
            t = j / (ants_per_arm - 1)
            theta = t * theta_total

            r = r_start * np.exp(b * theta)
            x = r * np.cos(theta + arm_phase)
            y = r * np.sin(theta + arm_phase)
            antennas.append([x, y])

    # --- 2. Generate Gaussian Core (Inner Surface Brightness) ---
    # We use rejection sampling to fill the center without colliding
    core_count = 0
    attempts = 0
    while core_count < n_core and attempts < 10000:
        attempts += 1
        # Gaussian cloud
        rx, ry = np.random.normal(0, sigma_g, 2)

        # Hard clip at transition radius (optional, but keeps core distinct)
        if np.sqrt(rx ** 2 + ry ** 2) > r_start * 1.2:
            continue

        # Check collision with ALL existing antennas (spiral + core so far)
        pos = np.array([rx, ry])
        if len(antennas) > 0:
            dists = np.linalg.norm(np.array(antennas) - pos, axis=1)
            if np.min(dists) < min_dist:
                continue

        antennas.append([rx, ry])
        core_count += 1

    print('Generated {0} core antennas'.format(core_count))

    # turn the antennas list into a numpy array with shape (n_antenna, 2)
    positions = np.array(antennas)

    # Adjust north (y) coordinate by the aspect factor from the latitude.
    aspect = 1 / np.cos(np.deg2rad(latitude))
    positions[:, 1] *= aspect

    bl = distance.pdist(np.transpose(np.array([positions[:, 0], positions[:, 1]])), 'euclidean')
    print('Minimum baseline length: {0:.1f} m'.format(np.min(bl)))
    print('Maximum baseline length: {0:.1f} m'.format(np.max(bl)))

    return positions


def extend_log_spiral_select_outriggers(inner_ants, n_new_ants=26, r_min=400, r_max=5000):
    """
    Selects outriggers from a random field using Logarithmic Bridge logic.
    """
    from scipy.spatial.distance import cdist
    # 1. Generate a field of "Existing Candidates" (Pseudo-random pads)
    n_candidates = 500
    candidates = []
    while len(candidates) < n_candidates:
        # Uniform distribution in Area (r^2) implies sqrt for radius
        r = np.sqrt(np.random.uniform(r_min ** 2, r_max ** 2))
        theta = np.random.uniform(0, 2 * np.pi)
        candidates.append([r * np.cos(theta), r * np.sin(theta)])
    candidates = np.array(candidates)

    selected_outriggers = []

    # 2. Define Logarithmic Rings (The Bridge)
    # We split the range 400m -> 5000m into rings where radius doubles
    # e.g., 400-800, 800-1600, 1600-3200, 3200-5000
    edges = [400, 800, 1600, 3200, 5000]

    # Target allocation per ring (Weighted slightly towards inner rings for bridging)
    # Total = 26. Example: [8, 8, 6, 4]
    allocation = [8, 8, 6, 4]

    current_pool = inner_ants.tolist()

    for i in range(len(edges) - 1):
        r_inner, r_outer = edges[i], edges[i + 1]
        n_needed = allocation[i]

        # Filter candidates in this ring
        dists = np.linalg.norm(candidates, axis=1)
        in_ring_idx = np.where((dists >= r_inner) & (dists < r_outer))[0]
        ring_candidates = candidates[in_ring_idx]

        # Greedy Selection: Pick candidates that are furthest from ALL currently selected antennas
        # This maximizes Azimuthal spread naturally
        for _ in range(n_needed):
            if len(ring_candidates) == 0: break

            # Calculate distance from every candidate to nearest existing antenna
            # Shape: (n_candidates, n_current_pool)
            d_matrix = cdist(ring_candidates, np.array(current_pool))

            # Distance to NEAREST neighbor for each candidate
            min_dists = np.min(d_matrix, axis=1)

            # Pick the candidate with the LARGEST distance to its nearest neighbor (Max-Min)
            best_idx = np.argmax(min_dists)

            # Add to selected
            chosen = ring_candidates[best_idx]
            selected_outriggers.append(chosen)
            current_pool.append(chosen)

            # Remove from candidates so we don't pick it again
            ring_candidates = np.delete(ring_candidates, best_idx, axis=0)

    return np.array(selected_outriggers), np.array(candidates)


def generate_candidates(n_candidates=147, width=4000, height=5000):
    """Generates 147 pseudo-random candidate locations in a 4x5 km box."""
    # Using Halton sequence or stratified sampling is better than pure random
    # to avoid accidental clumping in the candidates themselves, but uniform is fine for this demo.
    x = np.random.uniform(-width / 2, width / 2, n_candidates)
    y = np.random.uniform(-height / 2, height / 2, n_candidates)
    return np.column_stack((x, y))


def generate_ideal_outrigger_template(n_points=30, width=4000, height=5000, rotation_offset=60.):
    """
    Generates an Ideal Hollow Spiral stretched to the desired footprint.
    """
    template = []
    n_arms = 3
    points_per_arm = n_points // n_arms

    # Apply rotation offset to the phase
    # rotation_offset should be roughly pi/3 (60 degrees) if the core is at 0
    phase_shift = np.deg2rad(rotation_offset)

    # Inner/Outer Radii (The "Hollow" Logic)
    # We use a normalized radius 0.0 to 1.0, then scale by width/height
    # Min radius 5% (to get ~200m baselines), Max radius 100% (edge)
    r_norm_min = 0.05
    r_norm_max = 0.5  # 0.5 corresponds to full width/height

    theta_total = 2.0 * np.pi
    b = np.log(r_norm_max / r_norm_min) / theta_total

    for i in range(n_arms):
        phase = (2 * np.pi / n_arms) * i + phase_shift
        for j in range(points_per_arm):
            t = j / (points_per_arm - 1)
            theta = t * theta_total
            r_norm = r_norm_min * np.exp(b * theta)

            # Elliptical Projection
            x = r_norm * width * np.cos(theta + phase)
            y = r_norm * height * np.sin(theta + phase)
            template.append([x, y])

    return np.array(template)


def match_template(candidates, template):
    """
    Matches template points to the nearest available candidates.
    Greedy approach: Find global closest pair, lock it, repeat.
    """
    selected_indices = []
    selected_coords = []

    # Cost matrix: Distance between every template point and every candidate
    dists = cdist(template, candidates)

    # We have 30 template points to fill
    # We iterate 30 times finding the best remaining match

    # Mask for used candidates
    candidate_mask = np.ones(len(candidates), dtype=bool)

    # To optimize simply: For each template point, find nearest neighbor.
    # But two template points might claim the same candidate.
    # Better greedy approach: Sort ALL potential pairings by distance.

    # Create list of all (template_idx, candidate_idx, distance)
    pairings = []
    for t_i in range(len(template)):
        for c_i in range(len(candidates)):
            pairings.append((t_i, c_i, dists[t_i, c_i]))

    # Sort by distance (ascending)
    pairings.sort(key=lambda x: x[2])

    used_template = set()
    used_candidate = set()

    final_selection = []

    for t_i, c_i, d in pairings:
        if len(final_selection) == len(template):
            break
        if t_i not in used_template and c_i not in used_candidate:
            final_selection.append(candidates[c_i])
            used_template.add(t_i)
            used_candidate.add(c_i)

    return np.array(final_selection)


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
        raise ValueError(
            "n_antennas must be at least equal to cluster_antennas.")

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


def radial_profile(psf, bin_size=1):
    """
    Compute the averaged radial profile of a 2D image (psf).

    Parameters:
      psf : 2D numpy.ndarray
          Input image.
      bin_size : int, optional
          Bin size in pixels (default is 1).

    Returns:
      bin_centers : numpy.ndarray
          Centers of the radial bins (in pixels).
      radial_mean : numpy.ndarray
          Mean value in each radial bin.
    """
    y_idx, x_idx = np.indices(psf.shape)
    center = (psf.shape[0] // 2, psf.shape[1] // 2)
    r = np.sqrt((x_idx - center[1]) ** 2 + (y_idx - center[0]) ** 2)
    max_r = int(np.ceil(r.max()))
    bins = np.arange(0, max_r + bin_size, bin_size)
    radial_mean, bin_edges, _ = binned_statistic(
        r.ravel(), psf.ravel(), statistic='mean', bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, radial_mean


def fit_beam(psf, uv_max, dtheta, nsigma=2.):
    """
    Fit the main beam with a 2D Gaussian to extract FWHM major and minor axes.
    :param psf:
    :param uv_max: maximum uv distance (in meters)
    :param dtheta: pixel scale in the input psf image in arcsec
    :return:
    """
    from astropy.modeling import models, fitting
    fit_w = fitting.LevMarLSQFitter()
    # Find out the grid size of psf
    ngrid = psf.shape[0]
    # Find out the beam size in pixels
    sigma = 206265. / uv_max / dtheta / 2.
    # select only the center part to fit
    ymin = round(max([ngrid / 2 - nsigma / 2. * sigma, 0]))
    ymax = round(min([ngrid / 2 + nsigma / 2. * sigma, ngrid]))
    xmin = round(max([ngrid / 2 - nsigma / 2. * sigma, 0]))
    xmax = round(min([ngrid / 2 + nsigma / 2. * sigma, ngrid]))
    psf_sub = psf[ymin:ymax, xmin:xmax]
    y0, x0 = np.unravel_index(np.argmax(psf_sub), psf_sub.shape)
    amp = np.max(psf_sub)
    w = models.Gaussian2D(amp, x0, y0, sigma, sigma)
    yi_, xi_ = np.indices(psf_sub.shape)
    g_par = fit_w(w, xi_, yi_, psf_sub)
    # print(g)
    fwhm_major = g_par.x_fwhm * dtheta
    fwhm_minor = g_par.y_fwhm * dtheta
    fwhm_mean = (fwhm_major + fwhm_minor) / 2.
    return g_par, fwhm_major, fwhm_minor, fwhm_mean


@runtime_report
def plot_all_panels(positions, title='', labels=[], frequency=5, nyq_sample=None,
                    image_fits=None, pad_factor=4, crop_uv_bins=10,
                    figname=None, array_config_str=None, psf_mode='profile', rprofile_method='mean',
                    plot_psf_fit=True, legend_fontsize=10, no_psf_legend=False):
    """
    Plot antenna positions, UV coverage, PSF, and UV sampling density in a 2x3 layout.

    The function accepts either a single antenna configuration (a 2D numpy array of shape (N,2))
    or a list/tuple of such arrays. In either case the layout is 2x3:
      - Top-left: Antenna Layout (overplotted if multiple sets).
      - Top-center: UV Coverage (overplotted for each set).
      - Top-right: PSF panel. For a single configuration:
            If psf_mode=='image', displays the 2D PSF;
            If psf_mode=='rprofile' (or 'profile'), displays the averaged radial profile;
            If psf_mode=='uprofile', displays the profile along the central row;
            If psf_mode=='vprofile', displays the profile along the central column.
      - Bottom (spanning all columns): UV Sampling Density (overplotted for each set).

    Global text (array_config_str) is placed at the top-center of the figure.

    Parameters:
      positions : numpy.ndarray or list/tuple of numpy.ndarray
          Either a single 2D array (shape (N,2)) or a list/tuple of such arrays.
      title : str, optional
          Title text for the plots.
      labels : list, optional
          Labels for each configuration (if multiple).
      frequency : float, optional
          Frequency in GHz (used for PSF calculation).
      nyq_sample : dict or None, optional
          Dictionary of Nyquist sampling rates for the density plot.
      figname : str or None, optional
          If provided, the figure is saved with this filename.
      array_config_str : str or None, optional
          Global configuration text to display at the top-center of the figure.
      psf_mode : str, optional
          Either 'image', 'rprofile' (alias 'profile'), 'uprofile', or 'vprofile'.
          In multiple-set mode the PSF mode is forced to be 'uprofile'.

    Returns:
      fig, axes : tuple
          The Matplotlib figure and a tuple of axis objects.
    """
    import matplotlib.gridspec as gridspec
    # Convert the input to a list of 2D arrays.
    if isinstance(positions, np.ndarray):
        if positions.ndim == 2:
            pos_list = [positions]
        else:
            raise ValueError(
                "If 'positions' is a numpy array, it must be 2D with shape (N,2).")
    elif isinstance(positions, (list, tuple)):
        pos_list = []
        for pos in positions:
            if not (isinstance(pos, np.ndarray) and pos.ndim == 2):
                raise ValueError(
                    "Each element in 'positions' must be a 2D numpy array with shape (N,2).")
            pos_list.append(pos)
    else:
        raise ValueError(
            "'positions' must be either a 2D numpy array or a list/tuple of such arrays.")

    # If there is more than one configuration, force psf_mode to 'profile'
    npos = len(pos_list)
    if npos > 1:
        psf_mode = 'profile'

    # Create a 2x3 layout using GridSpec.
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0], wspace=0.3)
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1], wspace=0.3)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Panel 1 (top-left): Antenna Layout.
    ax_ant = fig.add_subplot(gs1[0, 0])
    ax_ant.set_title(f"{title} - Antenna Layout")
    ax_ant_zoom = fig.add_subplot(gs1[1, 0])
    #ax_ant_zoom.set_title(f"{title} - Antenna Layout")
    for idx, pos in enumerate(pos_list):
        lab = labels[idx] if (labels and len(labels) == len(pos_list)) else f"Set {idx + 1}"
        ax_ant.plot(pos[:, 0], pos[:, 1], 'o', label=lab,
                    color=colors[idx % len(colors)])
        ax_ant_zoom.plot(pos[:, 0], pos[:, 1], 'o', label=lab,
                         color=colors[idx % len(colors)])
    ax_ant.set_xlabel("X [m]")
    ax_ant.set_ylabel("Y [m]")
    ax_ant.set_aspect('equal')
    ax_ant_zoom.set_xlim(-100, 100)
    ax_ant_zoom.set_ylim(-100, 100)
    ax_ant_zoom.set_xlabel("X [m]")
    ax_ant_zoom.set_ylabel("Y [m]")
    ax_ant_zoom.set_aspect('equal')
    if npos > 1:
        ax_ant.legend(fontsize=legend_fontsize)
        ax_ant_zoom.legend(fontsize=legend_fontsize)

    # Global figure text at the top center.
    if array_config_str is not None:
        fig.text(0.5, 0.98, array_config_str, ha='center', va='top', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

    # Panel 2 (top-middle): UV Coverage.
    ax_uvcov = fig.add_subplot(gs1[0, 1])
    ax_uvcov.set_title(f"{title} - UV Coverage")
    ax_uvcov_zoom = fig.add_subplot(gs1[1, 1])
    #ax_uvcov_zoom.set_title(f"{title} - UV Coverage (Zoomed)")
    for idx, pos in enumerate(pos_list):
        lab = labels[idx] if (labels and len(labels) == len(pos_list)) else f"Set {idx + 1}"
        uv = compute_uv_coverage(pos)
        ax_uvcov.plot(uv[:, 0], uv[:, 1], '.', markersize=1,
                      label=lab, color=colors[idx % len(colors)])
        ax_uvcov_zoom.plot(uv[:, 0], uv[:, 1], '.', markersize=1,
                           label=lab, color=colors[idx % len(colors)])
    ax_uvcov.set_xlabel("u [m]")
    ax_uvcov.set_ylabel("v [m]")
    ax_uvcov.set_aspect('equal')
    ax_uvcov_zoom.set_xlim(-100, 100)
    ax_uvcov_zoom.set_ylim(-100, 100)
    ax_uvcov_zoom.set_xlabel("u [m]")
    ax_uvcov_zoom.set_ylabel("v [m]")
    ax_uvcov_zoom.set_aspect('equal')
    if npos > 1:
        ax_uvcov.legend(fontsize=legend_fontsize)
        ax_uvcov_zoom.legend(fontsize=legend_fontsize)

    # Panel 3 and 4 (top-right): Natural and Uniform PSF.
    ax_psf_im_natural = fig.add_subplot(gs2[0, 0])
    ax_psf_im_natural.set_title(f"Natural PSF ({frequency:.1f} GHz)")
    ax_psf_im_uniform = fig.add_subplot(gs2[0, 1])
    ax_psf_im_uniform.set_title(f"Uniform PSF ({frequency:.1f} GHz)")

    ax_psf_prof_natural = fig.add_subplot(gs2[1, 0])
    #ax_psf_prof_natural.set_title(f"Natural PSF ({frequency:.1f} GHz)")
    ax_psf_prof_uniform = fig.add_subplot(gs2[1, 1])
    #ax_psf_prof_uniform.set_title(f"Uniform PSF ({frequency:.1f} GHz)")
    # For each configuration, compute the PSF from its UV coverage.
    # Use the following procedure for each set:
    #  - Compute uv = compute_uv_coverage(pos)
    #  - Define grid parameters for a 2D histogram: grid_size and padded_size.
    #  - Compute H, then the PSF via FFT.
    #  - Compute pixel scale from frequency.
    #  - Convert the PSF to either a 2D image (if psf_mode=='image') or compute its averaged radial profile (if psf_mode=='profile').
    grid_size = 128
    padded_size = grid_size * 8
    psf_profiles = []
    for idx, pos in enumerate(pos_list):
        uv = compute_uv_coverage(pos)
        # Use the min/max of uv for each configuration.
        u_min, u_max = np.min(uv[:, 0]), np.max(uv[:, 0])
        v_min, v_max = np.min(uv[:, 1]), np.max(uv[:, 1])
        H_natural, xedges, yedges = np.histogram2d(uv[:, 0], uv[:, 1], bins=grid_size,
                                                   range=[[u_min, u_max], [v_min, v_max]])

        # Assign each bin with a nonzero value to 1 (uniform weighting).
        H_uniform = H_natural.copy()
        H_uniform[H_natural > 0] = 1.0

        psf_natural = np.abs(np.fft.fftshift(
            np.fft.fft2(H_natural, s=(padded_size, padded_size))))

        psf_uniform = np.abs(np.fft.fftshift(
            np.fft.fft2(H_uniform, s=(padded_size, padded_size))))

        # Compute pixel scale in arcsec:
        C_LIGHT = 3e8
        lambda_m = C_LIGHT / (frequency * 1e9)
        pixel_scale_rad = (grid_size * lambda_m) / \
                          ((u_max - u_min) * padded_size)
        pixel_scale_arcsec = pixel_scale_rad * 206265
        fov_arcsec = padded_size * pixel_scale_arcsec
        lab = labels[idx] if (labels and len(labels) == len(pos_list)) else f"Set {idx + 1}"

        g_par_n, fwhm_major_n, fwhm_minor_n, fwhm_mean_n = fit_beam(psf_natural, np.max([u_max, v_max]),
                                                                    pixel_scale_arcsec)
        print(f'Set {idx + 1}: Beam (Natural) FWHM Major: {fwhm_major_n:.2f} arcsec, Minor: {fwhm_minor_n:.2f}"')
        g_par_u, fwhm_major_u, fwhm_minor_u, fwhm_mean_u = fit_beam(psf_uniform, np.max([u_max, v_max]),
                                                                    pixel_scale_arcsec)
        print(f'Set {idx + 1}: Beam (Uniform) FWHM Major: {fwhm_major_u:.2f} arcsec, Minor: {fwhm_minor_u:.2f}"')
        g_par_n.x_mean = padded_size / 2
        g_par_n.y_mean = padded_size / 2
        g_par_u.x_mean = padded_size / 2
        g_par_u.y_mean = padded_size / 2

        # Compute sidelobe levels, defined as the rms in the annulus between 2x and 5x FWHM
        y_idx, x_idx = np.indices(psf_natural.shape)
        center = (psf_natural.shape[0] // 2, psf_natural.shape[1] // 2)
        r = np.sqrt((x_idx - center[1]) ** 2 + (y_idx - center[0]) ** 2)
        fwhm_pixels_n = fwhm_mean_n / pixel_scale_arcsec
        annulus_mask_n = (r >= 1.5 * fwhm_pixels_n) & (r <= 5 * fwhm_pixels_n)
        sidelobe_rms_n = np.std(psf_natural[annulus_mask_n]) / np.nanmax(psf_natural)
        print(f'Set {idx + 1}: Sidelobe RMS (Natural): {sidelobe_rms_n * 100:.1f}%')

        fwhm_pixels_u = fwhm_mean_u / pixel_scale_arcsec
        annulus_mask_u = (r >= 1.5 * fwhm_pixels_u) & (r <= 5 * fwhm_pixels_u)
        sidelobe_rms_u = np.std(psf_uniform[annulus_mask_u]) / np.nanmax(psf_uniform)
        print(f'Set {idx + 1}: Sidelobe RMS (Uniform): {sidelobe_rms_u * 100:.1f}%')

        # Plot psf image
        im_psf_natural = ax_psf_im_natural.imshow(psf_natural, extent=[-fov_arcsec / 2, fov_arcsec / 2, -fov_arcsec / 2,
                                                                       fov_arcsec / 2],
                                                  origin='lower', aspect='equal', cmap='viridis', alpha=1)
        yi, xi = np.indices(psf_natural.shape)
        model_gauss_n = g_par_n(xi, yi)
        xs = (np.arange(padded_size) - padded_size / 2. + 0.5) * pixel_scale_arcsec
        ys = (np.arange(padded_size) - padded_size / 2. + 0.5) * pixel_scale_arcsec
        if plot_psf_fit:
            ax_psf_im_natural.contour(xs, ys, model_gauss_n, levels=np.array([0.5]) * np.max(model_gauss_n),
                                      colors='w', linestyles=':')
        ax_psf_im_natural.set_xlabel("RA [arcsec]")
        ax_psf_im_natural.set_ylabel("DEC [arcsec]")
        #plt.colorbar(im_psf_natural, ax=ax_psf_natural)
        im_psf_uniform = ax_psf_im_uniform.imshow(psf_uniform, extent=[-fov_arcsec / 2, fov_arcsec / 2, -fov_arcsec / 2,
                                                                       fov_arcsec / 2],
                                                  origin='lower', aspect='equal', cmap='viridis', alpha=1)
        model_gauss_u = g_par_u(xi, yi)
        if plot_psf_fit:
            ax_psf_im_uniform.contour(xs, ys, model_gauss_u, levels=np.array([0.5]) * np.max(model_gauss_u),
                                      colors='w', linestyles=':')
        ax_psf_im_uniform.set_xlabel("RA [arcsec]")
        ax_psf_im_uniform.set_ylabel("DEC [arcsec]")
        #plt.colorbar(im_psf_uniform, ax=ax_psf_uniform)
        if psf_mode == 'rprofile' or psf_mode == 'profile':
            # Compute the averaged radial profile.
            y_idx, x_idx = np.indices(psf_natural.shape)
            center = (psf_natural.shape[0] // 2, psf_natural.shape[1] // 2)
            r = np.sqrt((x_idx - center[1]) ** 2 + (y_idx - center[0]) ** 2)
            bin_size = 1  # pixel bin size
            max_r = int(np.ceil(r.max()))
            bins_r = np.arange(0, max_r + bin_size, bin_size)
            radial_mean_natural, bin_edges, _ = binned_statistic(
                r.ravel(), psf_natural.ravel(), statistic='mean', bins=bins_r)
            radial_max_natural, _, _ = binned_statistic(
                r.ravel(), psf_natural.ravel(), statistic='max', bins=bins_r)

            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            r_arcsec = bin_centers * pixel_scale_arcsec

            # Find out the maximum sidelobe level between 1.5 to 5 FWHM
            idxr, = np.where((r_arcsec >= 1.5 * fwhm_mean_n) & (r_arcsec <= 5 * fwhm_mean_n))
            sidelobe_rms_n_max = np.nanmax(radial_max_natural[idxr]) / np.nanmax(radial_max_natural)

            ax_psf_prof_natural.plot(r_arcsec, radial_mean_natural / np.nanmax(radial_mean_natural),
                                     '--', color=colors[idx % len(colors)], label=lab + ' (mean)')
            ax_psf_prof_natural.plot(r_arcsec, radial_max_natural / np.nanmax(radial_max_natural),
                                     '-', color=colors[idx % len(colors)], label=lab + ' (max)')
            ax_psf_prof_natural.axvline(fwhm_mean_n, color=colors[idx % len(colors)],
                                        ls=':', label=lab + f' FWHM: {fwhm_mean_n:.1f}"')
            ax_psf_prof_natural.plot(np.array([1.5, 5.]) * fwhm_mean_n, [sidelobe_rms_n_max] * 2,
                                     color=colors[idx % len(colors)], ls='-',
                                     label=lab + f' Sidelobe Max: {sidelobe_rms_n_max * 100:.1f}%')
            ax_psf_prof_natural.plot(np.array([1.5, 5.]) * fwhm_mean_n, [sidelobe_rms_n] * 2,
                                     color=colors[idx % len(colors)], ls='-',
                                     label=lab + f' Sidelobe RMS: {sidelobe_rms_n * 100:.1f}%')
            ax_psf_prof_natural.set_xscale('log')
            ax_psf_prof_natural.set_yscale('log')
            ax_psf_prof_natural.set_xlabel("Radius [arcsec]")
            ax_psf_prof_natural.set_ylabel("Normalized PSF intensity")
            if not no_psf_legend:
                ax_psf_prof_natural.legend(fontsize=legend_fontsize)

            # Do the same for uniform PSF
            radial_mean_uniform, _, _ = binned_statistic(
                r.ravel(), psf_uniform.ravel(), statistic='mean', bins=bins_r)
            radial_max_uniform, _, _ = binned_statistic(
                r.ravel(), psf_uniform.ravel(), statistic='max', bins=bins_r)
            # Find out the maximum sidelobe level between 1.5 to 5 FWHM
            idxr, = np.where((r_arcsec >= 1.5 * fwhm_mean_u) & (r_arcsec <= 5 * fwhm_mean_u))
            sidelobe_rms_u_max = np.nanmax(radial_max_uniform[idxr]) / np.nanmax(radial_max_uniform)
            ax_psf_prof_uniform.plot(r_arcsec, radial_mean_uniform / np.nanmax(radial_mean_uniform),
                                     '--', color=colors[idx % len(colors)], label=lab + ' (mean)')
            ax_psf_prof_uniform.plot(r_arcsec, radial_max_uniform / np.nanmax(radial_max_uniform),
                                     '-', color=colors[idx % len(colors)], label=lab + ' (max)')
            ax_psf_prof_uniform.axvline(fwhm_mean_u, color=colors[idx % len(colors)],
                                        ls=':', label=lab + f' FWHM: {fwhm_mean_u:.1f}"')
            ax_psf_prof_uniform.plot(np.array([1.5, 5.]) * fwhm_mean_u, [sidelobe_rms_u_max] * 2,
                                     color=colors[idx % len(colors)], ls='-',
                                     label=lab + f' Sidelobe Max: {sidelobe_rms_u_max * 100:.1f}%')
            ax_psf_prof_uniform.plot(np.array([1.5, 5.]) * fwhm_mean_u, [sidelobe_rms_u] * 2,
                                     color=colors[idx % len(colors)], ls='--',
                                     label=lab + f' Sidelobe RMS: {sidelobe_rms_u * 100:.1f}%')

            ax_psf_prof_uniform.set_xscale('log')
            ax_psf_prof_uniform.set_yscale('log')
            ax_psf_prof_uniform.set_xlabel("Radius [arcsec]")
            ax_psf_prof_uniform.set_ylabel("Normalized PSF intensity")
            if not no_psf_legend:
                ax_psf_prof_uniform.legend(fontsize=legend_fontsize)
        elif psf_mode == 'uprofile':
            center_y = psf_natural.shape[0] // 2
            profile = psf_natural[center_y, :]
            x_axis = np.linspace(-fov_arcsec / 2, fov_arcsec / 2, psf_natural.shape[1])
            ax_psf_prof_natural.plot(x_axis, profile / np.nanmax(profile), '-',
                                     color=colors[idx % len(colors)], label=lab)
            ax_psf_prof_natural.set_xscale('log')
            ax_psf_prof_natural.set_yscale('log')
            ax_psf_prof_natural.set_xlabel("U [arcsec]")
            ax_psf_prof_natural.set_ylabel("Normalized PSF intensity")
            ax_psf_prof_natural.legend(fontsize=legend_fontsize)
            # Do the same for uniform PSF
            profile_uniform = psf_uniform[center_y, :]
            ax_psf_prof_uniform.plot(x_axis, profile_uniform / np.nanmax(profile_uniform), '-',
                                     color=colors[idx % len(colors)], label=lab)
            ax_psf_prof_uniform.set_xscale('log')
            ax_psf_prof_uniform.set_yscale('log')
            ax_psf_prof_uniform.set_xlabel("U [arcsec]")
            ax_psf_prof_uniform.set_ylabel("Normalized PSF intensity")
            ax_psf_prof_uniform.legend(fontsize=legend_fontsize)
        elif psf_mode == 'vprofile':
            center_x = psf_natural.shape[1] // 2
            profile = psf_natural[:, center_x]
            y_axis = np.linspace(-fov_arcsec / 2, fov_arcsec / 2, psf_natural.shape[0])
            ax_psf_prof_natural.plot(y_axis, profile / np.nanmax(profile), '-',
                                     color=colors[idx % len(colors)], label=lab)
            ax_psf_prof_natural.set_xscale('log')
            ax_psf_prof_natural.set_yscale('log')
            ax_psf_prof_natural.set_xlabel("V [arcsec]")
            ax_psf_prof_natural.set_ylabel("Normalized PSF intensity")
            ax_psf_prof_natural.legend(fontsize=legend_fontsize)
            # Do the same for uniform PSF
            profile_uniform = psf_uniform[:, center_x]
            ax_psf_prof_uniform.plot(y_axis, profile_uniform / np.nanmax(profile_uniform), '-',
                                     color=colors[idx % len(colors)], label=lab)
            ax_psf_prof_uniform.set_xscale('log')
            ax_psf_prof_uniform.set_yscale('log')
            ax_psf_prof_uniform.set_xlabel("V [arcsec]")
            ax_psf_prof_uniform.set_ylabel("Normalized PSF intensity")
            ax_psf_prof_uniform.legend(fontsize=legend_fontsize)
        else:
            raise ValueError(
                "psf_mode must be 'image', 'rprofile' (or 'profile'), 'uprofile', or 'vprofile'")

    # Panel 4 (bottom row, spanning all columns): UV Sampling Density.
    ax_uvdensity = fig.add_subplot(gs[1, 0])
    ax_uvdensity.set_title("UV Sampling Density")
    for idx, pos in enumerate(pos_list):
        lab = labels[idx] if (labels and len(labels) == len(pos_list)) else f"Set {idx + 1}"
        uv = compute_uv_coverage(pos)
        uv_dist = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        binwidth = 10
        bins_uv = np.arange(0, np.max(uv_dist) + binwidth, binwidth)
        counts, bin_edges = np.histogram(uv_dist, bins=bins_uv)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax_uvdensity.step(bin_centers, counts, where='mid',
                          label=lab, color=colors[idx % len(colors)])

    ax_uvdensity.set_xlabel("UV Distance [m]")
    ax_uvdensity.set_ylabel(f"Density (counts/per {binwidth:d} m)")
    if nyq_sample is not None:
        ls = ['--', ':', '-.', '-']
        for j, (k, v) in enumerate(nyq_sample.items()):
            ax_uvdensity.axhline(v, ls=ls[j % len(ls)], color='gray', label=f'Nyquist rate ({k})')
    ax_uvdensity.set_yscale('log')
    if npos > 1:
        ax_uvdensity.legend(fontsize=legend_fontsize)

    # overlay FFT radial profile if requested
    if image_fits:
        # read image, compute FFT amplitude
        smap = sunpy.map.Map(image_fits)
        Z = smap.data
        padded = np.fft.fftshift(np.fft.fft2(
            Z, s=(Z.shape[0] * pad_factor, Z.shape[1] * pad_factor)))
        amp = np.abs(padded)
        amin, amax = amp.min(), amp.max()
        amp_norm = (amp - amin) / (amax - amin) if amax > amin else amp
        # radial profile of amp_norm
        cen_pix, mean_amp = radial_profile(amp_norm, bin_size=1)
        # compute uv_extent in metres
        λ = C_LIGHT / (frequency * 1e9)
        fft_extent = (3600 * 180 / np.pi) / (pad_factor * Z.shape[0] * smap.scale.axis2.value * 3600) * λ * \
                     amp_norm.shape[0]
        uv_scale = fft_extent / amp_norm.shape[0]
        uv_r = cen_pix * uv_scale
        # twin-y to show FFT profile
        ax_uvdensity_tw = ax_uvdensity.twinx()
        ax_uvdensity_tw.plot(uv_r, mean_amp / np.nanmax(mean_amp),
                             color='k', label='Sun viz amp', alpha=0.5)
        ax_uvdensity_tw.set_ylabel("Norm Viz amp")
        ax_uvdensity_tw.set_yscale('log')
        ax_uvdensity_tw.legend(loc='upper right', fontsize=legend_fontsize)
        ax_uvdensity_tw.axhline(3e-4, ls='--', color='gray')

    # add another plot to the bottom row for the cumulative number of antennas as a function of distance to the center.
    ax_antcount = fig.add_subplot(gs[1, 1])
    ax_antcount.set_title("Cumulative Antenna Count vs Distance from Center")
    for idx, pos in enumerate(pos_list):
        lab = labels[idx] if (labels and len(labels) == len(pos_list)) else f"Set {idx + 1}"
        # find the center of the array
        center_x, center_y = np.mean(pos[:, 0]), np.mean(pos[:, 1])
        pos_centered = pos - np.array([center_x, center_y])
        distances = np.sqrt(pos_centered[:, 0] ** 2 + pos_centered[:, 1] ** 2)
        binwidth = 5
        #bins_dist = np.arange(0, np.max(distances) + binwidth, binwidth)
        #counts, bin_edges = np.histogram(distances, bins=bins_dist, density=False)
        #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax_antcount.hist(distances, bins=np.arange(0, np.max(distances) + binwidth, binwidth), density=False,
                         cumulative=True, histtype='step', label=lab,
                         color=colors[idx % len(colors)])
        ax_antcount.set_xscale('log')
        ax_antcount.set_xlabel('Distance from Center [m]')
        ax_antcount.set_ylabel('Accumulative Antenna Count')

    ax_antcount.legend(fontsize=legend_fontsize)

    gs.tight_layout(fig)
    # gs.update(hspace=0.0)

    # Save the figure if a filename is provided.
    if figname is not None:
        fig.savefig(figname, dpi=300)


def hadec_to_azel(ha_deg, dec_deg, lat_deg):
    """
    Converts Hour Angle (HA) and Declination (Dec) to Azimuth and Elevation.
    Returns: Az (radians), El (radians)
    """
    ha = np.radians(ha_deg)
    dec = np.radians(dec_deg)
    lat = np.radians(lat_deg)

    sin_el = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    el = np.arcsin(np.clip(sin_el, -1, 1))

    # Azimuth formula
    # sin(Az) = - sin(HA) * cos(Dec) / cos(El)
    # cos(Az) = (sin(Dec) - sin(El)*sin(Lat)) / (cos(El)*cos(Lat))

    # Using arctan2 for robustness
    # y_term = sin(Azimuth) * cos(Elevation)
    y = -np.sin(ha) * np.cos(dec)

    # x_term = cos(Azimuth) * cos(Elevation)
    # Derivation: cos(A) * cos(E) = (sin(d) - sin(phi)sin(E)) / cos(phi)
    x = (np.sin(dec) - np.sin(el) * np.sin(lat)) / np.cos(lat)

    # Now both are scaled by cos(E), so the ratio y/x is correct
    az = np.arctan2(y, x)

    return az, el


def calculate_shadowing(positions, dish_diameter=1.5, lat_deg=40.0, ngrid_ha=100, ngrid_dec=60):
    """
    Calculates number of shadowed antennas for a grid of HA and Dec.

    positions: (N, 2) numpy array of antenna coordinates (x=East, y=North)
    dish_diameter: Diameter in meters (shadowing threshold)
    lat_deg: Observatory latitude
    """
    # Add z-coordinate (assume flat array)
    n_ant = positions.shape[0]
    ant_pos = np.hstack((positions, np.zeros((n_ant, 1))))  # (N, 3)

    # Simulation Grid
    ha_range = np.linspace(-90, 90, ngrid_ha)  # -6h to +6h
    dec_range = np.linspace(-23.5, 23.5, ngrid_dec)  # Winter to Summer Solstice

    shadow_grid = np.zeros((len(dec_range), len(ha_range)))
    el_grid = np.zeros((len(dec_range), len(ha_range)))

    print(f"Simulating shadowing for {n_ant} antennas...")

    for i, dec in enumerate(dec_range):
        for j, ha in enumerate(ha_range):

            # 1. Get Sun Position vector
            az, el = hadec_to_azel(ha, dec, lat_deg)
            el_deg = np.degrees(el)
            el_grid[i, j] = el_deg  # Store for plotting

            # If sun is below horizon, ignore (or mark as invalid)
            if el <= 0:
                shadow_grid[i, j] = np.nan
                continue

            # Vector pointing TO the source (Sun)
            # Standard conversion: x=East, y=North, z=Up
            s_vec = np.array([
                np.sin(az) * np.cos(el),
                np.cos(az) * np.cos(el),
                np.sin(el)
            ])

            # 2. Project Antennas onto plane perpendicular to Sun
            # We construct a basis (u, v) perpendicular to s_vec (w)
            # w = s_vec
            # v = z_axis x s_vec (horizontal vector) -> normalized
            # u = v x w

            # Robust basis construction
            # If s_vec is vertical (zenith), handle gracefully
            if abs(s_vec[2]) > 0.99:
                u_vec = np.array([1, 0, 0])
                v_vec = np.array([0, 1, 0])
            else:
                up = np.array([0, 0, 1])
                v_vec = np.cross(s_vec, up)
                v_vec /= np.linalg.norm(v_vec)
                u_vec = np.cross(v_vec, s_vec)

            # Project positions: u = r . u_vec, v = r . v_vec
            u_coords = ant_pos @ u_vec
            v_coords = ant_pos @ v_vec
            w_coords = ant_pos @ s_vec  # Distance along Line of Sight

            # 3. Check Shadows
            # An antenna is shadowed if another antenna is:
            # a) Within distance D in (u,v) plane
            # b) Has a LARGER w coordinate (is closer to the sun)

            # We sort antennas by w (closest to sun first)
            # This makes checking easier: we only check if current ant is shadowed by previous ones

            indices = np.argsort(w_coords)[::-1]  # Descending w (Source -> Ground)
            sorted_u = u_coords[indices]
            sorted_v = v_coords[indices]

            shadowed_count = 0

            # Simple N^2 check (fast enough for N=120)
            for k in range(n_ant):
                current_u = sorted_u[k]
                current_v = sorted_v[k]

                # Check against all antennas "upstream" (indices 0 to k-1)
                if k > 0:
                    dist_sq = (sorted_u[:k] - current_u) ** 2 + (sorted_v[:k] - current_v) ** 2
                    if np.any(dist_sq < dish_diameter ** 2):
                        shadowed_count += 1

            shadow_grid[i, j] = shadowed_count

    return ha_range, dec_range, el_grid, shadow_grid


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
        raise ValueError(
            "COFA information not found in header of the config file.")

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
    N = - np.sin(lat0) * np.cos(lon0) * dX - np.sin(lat0) * \
        np.sin(lon0) * dY + np.cos(lat0) * dZ

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

from astropy.coordinates import SkyCoord, EarthLocation
from sunpy.coordinates import frames, sun


def make_diskmodel_from_mhd(modeldir='../fasr_sim/skymodels/quiet_sun/',
                            npzfile='psi_mhd_20201126_0.2-2GHz_emission_maps.npz',
                            tref='2020-11-26T20:45:47', outfitspre='psi_solar_disk_model_20201126', dorotate=False):
    from scipy.ndimage import rotate
    from suncasa.utils import helioimage2fits as helio
    import sunpy
    from casatools import image
    from astropy.time import Time


    ia = image()
    f = np.load(modeldir + '/' + npzfile)
    freqsghz = f['frequencies_Hz'] / 1e9
    tb_array = np.array(f['emission_cube'])  # Stokes I in K
    # find apparent solar radius at the reference time
    rsun_arcsec = sunpy.coordinates.sun.angular_radius(Time(tref)).value
    rsun_m = sunpy.sun.constants.radius.value

    dx = np.median(np.diff(f['x_coords'])) / rsun_m * rsun_arcsec
    dy = np.median(np.diff(f['y_coords'])) / rsun_m * rsun_arcsec
    flx_array = sfu2tb(freqsghz[None, None, :] * 1e9, tb_array, [dy, dx], square=True, reverse=True)
    (ny, nx, nf) = flx_array.shape

    # Open a CASA image as a template to make the coordinate system.
    # Need to change to make it more generic
    ephem = helio.read_horizons(Time(tref), observatory='OVRO')
    ra0 = ephem['ra'][0]
    dec0 = ephem['dec'][0]
    p0 = ephem['p0'][0]

    ia.open(modeldir + 'solar_disk_model_20201126.1GHz.fits')
    cs0 = ia.coordsys()
    cs0_rec = cs0.torecord()
    cs0_rec['direction0']['cdelt'] = np.array([-dx, dy]) / 3600. / 180. * np.pi
    cs0_rec['direction0']['crpix'] = np.array([nx / 2. - 0.5, ny / 2. - 0.5])
    cs0_rec['direction0']['crval'] = np.array([ra0, dec0])
    cs0_rec['spectral2']['wcs']['cdelt'] = np.median(np.diff(freqsghz)) * 1e9
    ia.done()

    # CASA wants axes to be in the order of (x, y, freq), so need to transpose
    imdata0 = np.transpose(flx_array, axes=(1, 0, 2))

    for i in range(nf):
        cs0_rec['spectral2']['wcs']['crval'] = freqsghz[i] * 1e9
        outfits = modeldir + '/' + outfitspre + f'.{freqsghz[i]:.2f}GHz.fits'
        imdata_ = imdata0[:, :, i]
        # if desired, rotate the image by p angle (to allign with GEO NS)
        if dorotate:
            imdata_ = rotate(imdata_, p0, reshape=False, mode='constant', cval=0.0)
        imdata = np.reshape(imdata_, (nx, ny, 1, 1)) * 1e4  # convert to Jy/pixel

        ia.fromarray(pixels=imdata, csys=cs0_rec)
        print(cs0_rec['spectral2']['wcs'])
        ia.setbrightnessunit("Jy/pixel")
        ia.setrestoringbeam(major='20arcsec', minor='20arcsec', pa='0deg')
        ia.tofits(outfits, overwrite=True)
        print('Wrote ' + outfits)
        ia.done()


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


def calc_model_radial_profile(solar_model, dr=30., tb_min=100., bkg_radius_range=[1.4, 1.6],
                              snr_min=1.2, reftime='2020-11-26T20:45:47', apply_mask=True, solar_model_out=None):
    '''
    Scale the intensity of the solar model by the off-limb profile.
    '''
    from sunpy import coordinates
    from matplotlib import colors as mcolors
    solar_radius_asec = coordinates.sun.angular_radius(reftime).value
    #calculate the mean intensity profile by averaging the intensity in a circular annulus around the solar disk.
    hdul = fits.open(solar_model)
    flux = hdul[0].data[0, 0]
    header = hdul[0].header
    pixscale_x = np.abs(header['CDELT1']) * 3600.  # in arcsec
    pixscale_y = np.abs(header['CDELT2']) * 3600.  # in arcsec
    freqhz = header['CRVAL3']  # in Hz
    print(f"Pixel scale: {pixscale_x:.2f} x {pixscale_y:.2f} arcsec")
    jypx2k = 1.222e6 / (freqhz / 1e9) ** 2 / ((pixscale_x * pixscale_y) / (np.pi / (4 * np.log(2))))
    flux_min = tb_min / jypx2k  # convert to Jy/pixel
    print(f"Minimum flux threshold: {flux_min:.4f} Jy/pixel at {freqhz / 1e9:.2f} GHz")
    dr_pix = int(dr / ((pixscale_x + pixscale_y) / 2.))
    nx = flux.shape[1]
    ny = flux.shape[0]
    hdul.close()
    ics = int(nx / 2)
    radius_max = np.sqrt(nx ** 2 + ny ** 2) / 2
    mean_intensity_profile = []
    radius_list_pix = []
    for radius in np.arange(0., radius_max, dr_pix):
        # get pixels within radius to radius + dr
        y, x = np.ogrid[-ics:ny - ics, -ics:nx - ics]
        mask_in = x * x + y * y <= radius * radius
        mask_out = x * x + y * y > (radius + dr_pix) * (radius + dr_pix)
        mask = mask_in | mask_out
        flux_ring = np.copy(flux)
        flux_ring[mask] = np.nan
        # set all negative values to nan
        flux_ring[flux_ring < flux_min] = np.nan
        mean_intensity_profile.append(np.nanmean(flux_ring))
        radius_list_pix.append(radius)

    mean_intensity_profile = np.array(mean_intensity_profile)

    radius_list = np.array(radius_list_pix) * pixscale_x  # convert to arcsec
    idx_bkg = np.where((radius_list >= bkg_radius_range[0] * solar_radius_asec) &
                       (radius_list <= bkg_radius_range[1] * solar_radius_asec))[0]
    background_level = np.nanmean(mean_intensity_profile[idx_bkg])
    print(f'Background level at {freqhz / 1e9:.2f} GHz: {background_level:.4f} Jy/pixel')
    plt.plot(radius_list, mean_intensity_profile, label=f'{freqhz / 1e9:.2f} GHz')
    idx = np.where(mean_intensity_profile[:idx_bkg[0]] > background_level * snr_min)[0][-1]
    radius_cutoff = radius_list[idx]
    plt.plot(radius_cutoff, mean_intensity_profile[idx], marker='o', fillstyle='full', markersize=8)
    plt.plot(np.array(bkg_radius_range) * solar_radius_asec, [background_level] * 2, ls='--')
    plt.axvline(solar_radius_asec, ls='--', color='k')
    plt.yscale('log')
    plt.legend()
    plt.ylim(bottom=np.min(np.array(flux_min)))
    plt.xlabel('Radial Distance (arcsec)')
    plt.ylabel('Average Intensity (Jy/pixel)')
    plt.show()

    if apply_mask:
        if solar_model_out is None:
            solar_model_out = solar_model.replace('.fits', '.masked.fits')
        # Create a masked copy of the solar model, zeroing emission beyond the cutoff radius.
        os.system(f'cp {solar_model} {solar_model_out}')
        hdul_mask = fits.open(solar_model_out)
        data_mask = hdul_mask[0].data  # expect shape (stokes, freq, ny, nx) or similar
        header_mask = hdul_mask[0].header
        ny_mask, nx_mask = data_mask.shape[-2], data_mask.shape[-1]
        ics_mask = int(nx_mask / 2)
        # radial coordinate in pixels relative to image centre
        y_mask, x_mask = np.ogrid[-ics_mask:ny_mask - ics_mask, -ics_mask:nx_mask - ics_mask]
        r_pix = np.sqrt(x_mask * x_mask + y_mask * y_mask)
        # convert to arcsec using the same pixel scale as above (assume square pixels)
        r_arcsec = r_pix * pixscale_x
        mask_beyond = r_arcsec > radius_cutoff
        # apply mask across all leading axes (stokes/freq) by broadcasting on the last two axes
        data_mask[..., mask_beyond] = 0.0
        # Also replace all NaNs and negative values with zeros
        data_mask = np.nan_to_num(data_mask, nan=0.0)
        data_mask[data_mask < 0] = 0.0
        hdul_mask[0].data = data_mask
        # Store the cutoff radius in the output header (arcsec)
        header_mask['RADCUT'] = (float(radius_cutoff), 'Radius cutoff [arcsec] used for masking')
        hdul_mask.writeto(solar_model_out, overwrite=True)
        hdul_mask.close()
        print(f"Masked solar model written to {solar_model_out} (RADCUT={radius_cutoff:.2f} arcsec)")

        # Plot the masked image (2D slice) for inspection
        img = data_mask
        if img.ndim == 4:
            img2d = img[0, 0]
        elif img.ndim == 3:
            img2d = img[0]
        else:
            img2d = img

        extent = (-nx_mask / 2 * pixscale_x, nx_mask / 2 * pixscale_x,
                  -ny_mask / 2 * pixscale_y, ny_mask / 2 * pixscale_y)
        plt.figure()
        plt.imshow(img2d, origin='lower', extent=extent, norm=mcolors.LogNorm())
        plt.colorbar(label='Jy/pixel')
        plt.title(f'Masked solar model at {freqhz / 1e9:.2f} GHz')
        plt.xlabel('X (arcsec)')
        plt.ylabel('Y (arcsec)')
        plt.show()

    # fit an exponential to the mean intensity profile at radii > solar radius
    return radius_list, mean_intensity_profile, background_level, radius_cutoff


def equivalent_amp_pha_error(phaerr: float, unit: str = 'deg') -> tuple[float, float]:
    """
    Compute the equivalent amplitude error (%) and the phase error (%) 
    (relative to a full cycle) for a given phase error.
    ## Bryan Butler: A phase error of x radians has the same effects as an amplitude error of 100 x %

    Parameters
    ----------
    phaerr : float
        The phase error value.
    unit : {'deg', 'rad'}
        Unit of `phase_error`:
        - 'deg': phase_error is in degrees
        - 'rad': phase_error is in radians

    Returns
    -------
    phase_pct : float
        Fractional Phase error of a full cycle (360° or 2π).
    amp_pct : float
        Equivalent fractional amplitude error.

    Raises
    ------
    ValueError
        If `unit` is not 'deg' or 'rad'.

    Examples
    --------
    >>> equivalent_amp_and_phase_error(5.7, unit='deg')
    (1.5833333333333333, 9.95)
    >>> # 0.1 rad phase error:
    >>> equivalent_amp_and_phase_error(0.1, unit='rad')
    (1.5915494309189535, 10.0)
    """
    if unit == 'deg':
        # Phase error in degrees → percent of 360°
        phase_pct = (phaerr / 360.0) * 100.0
        # Convert to radians for amplitude error
        phase_rad = np.deg2rad(phaerr)
    elif unit == 'rad':
        # Phase error in radians → percent of 2π
        phase_pct = (phaerr / (2 * np.pi)) * 100.0
        phase_rad = phaerr
    else:
        raise ValueError("unit must be 'deg' or 'rad'")

    # Equivalent amplitude error (%) = phase error in radians × 100
    amp_pct = phase_rad * 100.0

    return float(phase_pct) / 100, float(amp_pct) / 100


def generate_caltb(msfile, caltype=['ph', 'amp', 'mbd'], calerr=[0.05, 0.05, 0.01], caltbdir='./', pol='R,L'):
    '''
    Generate CASA calibration tables (caltb) to corrupt the visibilities in a Measurement Set (MS) by assuming potential errors in the calibration.
    Default to generate phase (ph), amplitude (amp), and multi-band delay (mhd) calibration tables.
    The default errors are 5% of a radian, 5% for amplitude, and 0.01 ns for multi-band delay.
    Parameters:
        msfile : str
            Path to the Measurement Set file.
        caltype : list of str
            List of calibration types to generate. Options are 'ph' (phase), 'amp' (amplitude), 'mbd' (multi-band delay).
        calerr : list of float
            List of errors for each calibration type in radians for phase, fractional for amplitude, and nanoseconds for multi-band delay.
    '''
    from casatasks import gencal
    from casatools import ms

    ms_tool = ms()
    ms_tool.open(msfile)
    mdata = ms_tool.metadata()
    nant = mdata.nantennas()
    ant_names = mdata.antennanames()
    antlist = ','.join(ant_names)
    ms_tool.close()

    gaintable = []
    for cal in caltype:
        if cal == 'ph':
            # Generate phase calibration table
            err = calerr[caltype.index(cal)]
            # write in degrees.
            err_deg = np.rad2deg(err)
            caltable = f'{caltbdir}/caltb_FASR_corrupt_{err_deg:.0f}deg.ph'
            os.system(f'rm -rf {caltable}')
            if pol == '':
                pha = np.degrees(np.random.normal(0, err, nant))
            elif pol == 'R,L':
                pha = np.degrees(np.random.normal(0, err, (nant, 2)))
            else:
                raise ValueError("pol must be '' or 'R,L'")
            gencal(vis=msfile, caltable=caltable, caltype='ph', antenna=antlist,
                   pol=pol, parameter=pha.flatten().tolist())
            print(f"Generated phase calibration table: {caltable}")
        elif cal == 'amp':
            # Generate amplitude calibration table
            err = calerr[caltype.index(cal)]
            caltable = f'{caltbdir}/caltb_FASR_corrupt_{np.int_(err * 100)}pct.amp'
            os.system(f'rm -rf {caltable}')
            if pol == '':
                amp = np.random.normal(1, err, nant)
            elif pol == 'R,L':
                amp = np.random.normal(1, err, (nant, 2))
            else:
                raise ValueError("pol must be '' or 'R,L'")
            gencal(vis=msfile, caltable=caltable, caltype='amp', antenna=antlist,
                   pol=pol, parameter=amp.flatten().tolist())
            print(f"Generated amplitude calibration table: {caltable}")
        elif cal == 'mbd':
            # Generate multi-band delay calibration table
            err = calerr[caltype.index(cal)]
            caltable = f'{caltbdir}/caltb_FASR_corrupt_{err}ns.mbd'
            os.system(f'rm -rf {caltable}')
            if pol == '':
                mbd = np.random.normal(0, err, nant)
            elif pol == 'R,L':
                mbd = np.random.normal(0, err, (nant, 2))
            else:
                raise ValueError("pol must be '' or 'R,L'")
            gencal(vis=msfile, caltable=caltable, caltype='mbd', antenna=antlist, pol=pol,
                   parameter=mbd.flatten().tolist())
            print(f"Generated multi-band delay calibration table: {caltable}")
        gaintable.append(caltable)

    return gaintable


def boost_long_baseline_weights(ms_in, ms_out, freqstr, uv0_klambda=30, power=2.0, max_boost=50.0,
                                uvrange_min_klambda=0.0):
    """Create a copy of an MS and reweight rows by UV distance."""
    from casatools import table as tbtool

    if os.path.exists(ms_out):
        print('MS already exists, skipping copy:', ms_out)
    else:
        print('Copying MS ->', ms_out)
        shutil.copytree(ms_in, ms_out)

    lam_m = 3e8 / freqstr_to_hz(freqstr)

    tb = tbtool()
    tb.open(ms_out, nomodify=False)
    try:
        uvw = tb.getcol('UVW')
        u_m, v_m = uvw[0, :], uvw[1, :]
        uv_kl = (np.sqrt(u_m * u_m + v_m * v_m) / lam_m) / 1e3

        boost = (np.maximum(uv_kl, uvrange_min_klambda) / float(uv0_klambda)) ** float(power)
        boost = np.where(uv_kl >= uvrange_min_klambda, boost, 1.0)
        boost = np.clip(boost, 1.0, float(max_boost))

        for col, op in [('WEIGHT', 'mul'), ('SIGMA', 'divsqrt')]:
            if col not in tb.colnames():
                continue
            arr = tb.getcol(col)
            if arr.ndim == 1:
                if op == 'mul':
                    arr = arr * boost
                else:
                    arr = arr / np.sqrt(boost)
            elif arr.ndim == 2:
                if arr.shape[1] == boost.size:
                    if op == 'mul':
                        arr = arr * boost[None, :]
                    else:
                        arr = arr / np.sqrt(boost[None, :])
                elif arr.shape[0] == boost.size:
                    if op == 'mul':
                        arr = arr * boost[:, None]
                    else:
                        arr = arr / np.sqrt(boost[:, None])
                else:
                    raise ValueError(f'Unexpected {col} shape {arr.shape} for nrow={boost.size}')
            else:
                raise ValueError(f'Unexpected {col} ndim={arr.ndim}')
            tb.putcol(col, arr)
            print('Updated', col)

        if 'WEIGHT_SPECTRUM' in tb.colnames():
            wspec = tb.getcol('WEIGHT_SPECTRUM')
            if wspec.ndim == 3 and wspec.shape[2] == boost.size:
                wspec = wspec * boost[None, None, :]
                tb.putcol('WEIGHT_SPECTRUM', wspec)
                print('Updated WEIGHT_SPECTRUM')
    finally:
        tb.close()

    return ms_out


def uv_distance_m_from_ms(msfile):
    """Return UV distance in meters for each visibility row in an MS."""
    from casatools import table as tbtool

    tb = tbtool()
    tb.open(msfile)
    uvw = tb.getcol('UVW')
    tb.close()
    u = uvw[0, :]
    v = uvw[1, :]
    return np.sqrt(u * u + v * v)


def plot_uv_density_splits(msfile, cfg_label_text, thr_list, figdir, freqstr, bin_m=10.0):
    """Plot UV-density histograms split by threshold in klambda."""
    uvd = uv_distance_m_from_ms(msfile)
    lam = 3e8 / freqstr_to_hz(freqstr)
    uv_kl = uvd / lam / 1e3

    max_uv = np.nanmax(uvd) if np.isfinite(np.nanmax(uvd)) else 10000.0
    edges = np.arange(0.0, max_uv + bin_m, bin_m)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axs = plt.subplots(len(thr_list), 1, figsize=(5, 1.5 * len(thr_list)), sharex=True, sharey=True)
    if len(thr_list) == 1:
        axs = [axs]

    for ax, thr in zip(axs, thr_list):
        mask_short = uv_kl < thr
        mask_long = uv_kl >= thr
        h_short, _ = np.histogram(uvd[mask_short], bins=edges)
        h_long, _ = np.histogram(uvd[mask_long], bins=edges)
        ax.step(centers, h_short, where='mid', label=f'short (<{thr} klambda)', lw=1.5)
        ax.step(centers, h_long, where='mid', label=f'long (>{thr} klambda)', lw=1.5)
        ax.set_ylabel('Density\n(counts/10 m)')
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_yscale('log')

    axs[-1].set_xlabel('UV distance [m]')
    fig.suptitle(f'UV sampling density split comparison ({cfg_label_text})', y=0.995)
    fig.tight_layout()
    out = os.path.join(figdir, f'fig-uv_density_split_compare_{os.path.basename(msfile).replace(".ms", "")}.jpg')
    fig.savefig(out, dpi=200)
    print('Saved figure:', out)
    return out


def plot_boost_weighting_curves(msfile, cfg_label_text, boost_grid, figdir, freqstr, max_boost,
                                uvrange_min_klambda=0.0):
    """Plot manual weight-boost curves against UV distance."""
    uvd = uv_distance_m_from_ms(msfile)
    max_uv = np.nanmax(uvd) if np.isfinite(np.nanmax(uvd)) else 10000.0
    uvd_grid = np.linspace(0.0, max_uv, 800)

    lam = 3e8 / freqstr_to_hz(freqstr)
    uv_kl_grid = uvd_grid / lam / 1e3

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for uv0, pwr in boost_grid:
        w = (np.maximum(uv_kl_grid, uvrange_min_klambda) / float(uv0)) ** float(pwr)
        w = np.clip(w, 1.0, float(max_boost))
        ax.plot(uvd_grid, w, lw=1.8, label=f'uv0={uv0}k, p={pwr}')

    ax.set_xlabel('UV distance [m]')
    ax.set_ylabel('Weight factor')
    ax.set_title(f'Manual boost weighting vs UV distance ({cfg_label_text})')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    out = os.path.join(figdir, f'fig-boost_weighting_curves_{os.path.basename(msfile).replace(".ms", "")}.jpg')
    fig.savefig(out, dpi=200)
    print('Saved figure:', out)
    return out


def ensure_default_calibration(msfile, phaerr, amperr, caltbdir='./', calibrated_msfile=None,
                               overwrite=False, in_place=False):
    """Generate and apply default phase/amplitude calibration corruption.

    :param msfile: Input measurement set.
    :type msfile: str
    :param phaerr: Phase error in radians.
    :type phaerr: float
    :param amperr: Fractional amplitude error.
    :type amperr: float
    :param caltbdir: Directory holding calibration tables.
    :type caltbdir: str
    :param calibrated_msfile: Output calibrated MS path when ``in_place`` is False.
    :type calibrated_msfile: str or None
    :param overwrite: If True, rebuild the calibrated MS copy.
    :type overwrite: bool
    :param in_place: If True, apply calibration directly to ``msfile``.
    :type in_place: bool
    :returns: Path to the calibrated MS.
    :rtype: str
    """
    from casatasks import applycal, clearcal

    phaerr_deg = np.rad2deg(phaerr)
    gaintable = [
        f'{caltbdir}/caltb_FASR_corrupt_{phaerr_deg:.0f}deg.ph',
        f'{caltbdir}/caltb_FASR_corrupt_{np.int_(amperr * 100)}pct.amp',
    ]
    if not os.path.exists(gaintable[0]) or not os.path.exists(gaintable[1]):
        gaintable = generate_caltb(
            msfile,
            caltype=['ph', 'amp'],
            calerr=[phaerr, amperr],
            caltbdir=caltbdir,
        )
    if in_place:
        calibrated_ms = msfile
    else:
        calibrated_ms = calibrated_msfile or msfile.replace('.ms', '_defaultcal.ms')
        if overwrite and os.path.exists(calibrated_ms):
            remove_path(calibrated_ms)
        if not os.path.exists(calibrated_ms):
            shutil.copytree(msfile, calibrated_ms)

    clearcal(vis=calibrated_ms)
    applycal(vis=calibrated_ms, gaintable=gaintable, applymode='calonly', calwt=False)
    return calibrated_ms


def calc_noise_terms_for_cfg(config_file, solar_model, freqstr, imsize, cell, tsys, duration,
                             integration_time, channel_width_mhz, eta_a=0.6):
    """Return cached-style noise terms matching notebook sensitivity calculations."""
    antenna_params = np.genfromtxt(config_file, comments='#')
    dish_diameter = antenna_params[0, 3]
    uv_cell = 1. / (imsize * float(str(cell).rstrip('arcsec')) / 206265.)
    total_flux = calc_total_flux_on_dish(solar_model, dish_diameter=dish_diameter, freqghz=freqstr)
    tant = total_flux_to_tant(total_flux, eta_a=eta_a, dish_diameter=dish_diameter)
    noisejy, sigma_na, sigma_un = calc_noise(
        tsys, config_file, dish_diameter=dish_diameter,
        total_flux=total_flux, duration=duration,
        integration_time=integration_time,
        channel_width_mhz=channel_width_mhz,
        freqghz=freqstr, uv_cell=uv_cell,
    )
    sigma_jy = sigma_un / np.sqrt((duration / integration_time))
    return {
        'dish_diameter': dish_diameter,
        'uv_cell': uv_cell,
        'total_flux': total_flux,
        'tant': tant,
        'noisejy': noisejy,
        'sigma_na': sigma_na,
        'sigma_un': sigma_un,
        'sigma_jy': sigma_jy,
    }


def boost_restoringbeam_arcsec(uv0_klambda):
    """Compute boosted restoring beam in arcsec using 3e4/uv0 scaling."""
    uv0_lambda = float(uv0_klambda) * 1e3
    if uv0_lambda <= 0:
        raise ValueError(f'uv0_klambda must be > 0, got {uv0_klambda}')
    return (3.0 * 10e3) / uv0_lambda


def ensure_split_feather(msfile, msname, uvthr_klambda, cell, deconvolver, scales, imsize, phasecenter,
                         niter, overwrite_images=False, weighting='superuniform'):
    """Create short-UV, long-UV, and feathered images for a given MS.

    :param msfile: Input measurement set.
    :type msfile: str
    :param msname: Base product name for output images.
    :type msname: str
    :param uvthr_klambda: UV split threshold in klambda.
    :type uvthr_klambda: float
    :param cell: CASA cell size string.
    :type cell: str
    :param deconvolver: CASA deconvolver name.
    :type deconvolver: str
    :param scales: Multiscale list used by CASA.
    :type scales: list
    :param imsize: Image size.
    :type imsize: int or list
    :param phasecenter: CASA phasecenter string.
    :type phasecenter: str
    :param niter: CLEAN iteration limit.
    :type niter: int
    :param overwrite_images: If True, remove existing image products first.
    :type overwrite_images: bool
    :param weighting: CASA weighting passed to both split-image ``tclean`` runs.
    :type weighting: str
    :returns: Dict with ``short``, ``long``, and ``feather`` image paths.
    :rtype: dict
    """
    from casatasks import feather, tclean

    short_base = f'{msname}_uvlt{uvthr_klambda}k_{cell}_{weighting}_{deconvolver}'
    long_base = f'{msname}_uvgt{uvthr_klambda}k_{cell}_{weighting}_{deconvolver}'
    feather_base = f'{msname}_feather_uvthr{uvthr_klambda}k_{weighting}_{deconvolver}'

    if overwrite_images:
        for imagename in [short_base, long_base, feather_base]:
            clear_casa_imagename(imagename)
    else:
        clear_casa_imagename_if_shape_mismatch(short_base, imsize)
        clear_casa_imagename_if_shape_mismatch(long_base, imsize)
        clear_casa_imagename_if_shape_mismatch(feather_base, imsize)

    if not os.path.exists(short_base + '.image'):
        tclean(
            vis=msfile, imagename=short_base, datacolumn='data', specmode='mfs',
            deconvolver=deconvolver, scales=casa_safe_scales(scales),
            imsize=imsize, cell=cell, phasecenter=phasecenter,
            weighting=weighting, uvrange=f'<{uvthr_klambda}klambda',
            niter=niter, interactive=False,
        )

    if not os.path.exists(long_base + '.image'):
        tclean(
            vis=msfile, imagename=long_base, datacolumn='data', specmode='mfs',
            deconvolver=deconvolver, scales=casa_safe_scales(scales),
            imsize=imsize, cell=cell, phasecenter=phasecenter,
            weighting=weighting, uvrange=f'>{uvthr_klambda}klambda',
            niter=niter, interactive=False,
        )

    short_path = short_base + '.image'
    long_path = long_base + '.image'
    if not os.path.exists(short_path):
        raise FileNotFoundError(f'Short-UV image was not created: {short_path}')
    if not os.path.exists(long_path):
        raise FileNotFoundError(f'Long-UV image was not created: {long_path}')

    feather_path = feather_base + '.image'
    feather_needs_refresh = False
    if os.path.exists(feather_path):
        try:
            feather_needs_refresh = (
                os.path.getmtime(short_path) > os.path.getmtime(feather_path)
                or os.path.getmtime(long_path) > os.path.getmtime(feather_path)
            )
        except OSError:
            feather_needs_refresh = False
    if feather_needs_refresh:
        clear_casa_imagename(feather_base)
    if not os.path.exists(feather_path):
        feather(imagename=feather_path, lowres=short_path, highres=long_path, sdfactor=2.0)
    if not os.path.exists(feather_path):
        raise RuntimeError(f'Feather output was not created: {feather_path}. Inputs were lowres={short_path}, highres={long_path}')

    return {'short': short_path, 'long': long_path, 'feather': feather_path}


def ensure_boost_image(msfile, msname, freqstr, uv0_klambda, power, max_boost, cell, deconvolver, scales,
                       imsize, phasecenter, niter, overwrite_ms=False, overwrite_image=False,
                       uvrange_min_klambda=0.0, use_boost_restoringbeam=True, weighting='superuniform'):
    """Create a boosted-weight MS/image product for a given configuration.

    :param msfile: Input measurement set.
    :type msfile: str
    :param msname: Base product name for output images.
    :type msname: str
    :param freqstr: Frequency string used for UV reweighting.
    :type freqstr: str
    :param uv0_klambda: Characteristic UV scale in klambda.
    :type uv0_klambda: float
    :param power: Power-law index for baseline boosting.
    :type power: float
    :param max_boost: Maximum applied baseline boost.
    :type max_boost: float
    :param cell: CASA cell size string.
    :type cell: str
    :param deconvolver: CASA deconvolver name.
    :type deconvolver: str
    :param scales: Multiscale list used by CASA.
    :type scales: list
    :param imsize: Image size.
    :type imsize: int or list
    :param phasecenter: CASA phasecenter string.
    :type phasecenter: str
    :param niter: CLEAN iteration limit.
    :type niter: int
    :param overwrite_ms: If True, rebuild the boosted MS.
    :type overwrite_ms: bool
    :param overwrite_image: If True, rebuild the boosted image.
    :type overwrite_image: bool
    :param uvrange_min_klambda: Minimum UV range included in boost weighting.
    :type uvrange_min_klambda: float
    :param use_boost_restoringbeam: If True, set a custom restoring beam.
    :type use_boost_restoringbeam: bool
    :param weighting: CASA weighting passed to the boosted-image ``tclean`` run.
    :type weighting: str
    :returns: Boosted CASA image path.
    :rtype: str
    """
    from casatasks import tclean

    ms_boost = msfile.replace('.ms', f'_wboost_uv0{uv0_klambda}k_p{power}.ms')
    if overwrite_ms and os.path.exists(ms_boost):
        remove_path(ms_boost)

    if not os.path.exists(ms_boost):
        boost_long_baseline_weights(
            ms_in=msfile, ms_out=ms_boost, freqstr=freqstr,
            uv0_klambda=uv0_klambda, power=power, max_boost=max_boost,
            uvrange_min_klambda=uvrange_min_klambda,
        )

    boost_base = ms_boost.replace('.ms', '') + f'_image_{weighting}_{deconvolver}'
    if overwrite_image:
        clear_casa_imagename(boost_base)
    else:
        clear_casa_imagename_if_shape_mismatch(boost_base, imsize)

    if not os.path.exists(boost_base + '.image'):
        tclean_kwargs = dict(
            vis=ms_boost, imagename=boost_base, datacolumn='data', specmode='mfs',
            deconvolver=deconvolver, scales=casa_safe_scales(scales),
            imsize=imsize, cell=cell, phasecenter=phasecenter,
            weighting=weighting, niter=niter, interactive=False,
        )
        if use_boost_restoringbeam:
            rb_arcsec = boost_restoringbeam_arcsec(uv0_klambda)
            tclean_kwargs['restoringbeam'] = f'{rb_arcsec:.3f}arcsec'
        tclean(**tclean_kwargs)

    return boost_base + '.image'


def ensure_weighting_reference(msfile, msname, weighting, deconvolver, scales, imsize, cell, phasecenter,
                               niter, overwrite_image=False):
    """Ensure a reference image exists for a given CASA weighting mode."""
    from casatasks import tclean

    imbase = f'{msname}_{weighting}_{deconvolver}'
    if overwrite_image:
        clear_casa_imagename(imbase)
    else:
        clear_casa_imagename_if_shape_mismatch(imbase, imsize)

    if not os.path.exists(imbase + '.image'):
        tclean(
            vis=msfile, imagename=imbase, datacolumn='data', specmode='mfs',
            deconvolver=deconvolver, scales=casa_safe_scales(scales),
            imsize=imsize, cell=cell, phasecenter=phasecenter,
            weighting=weighting, niter=niter, interactive=False,
        )
    return imbase + '.image'


def plot_compare_casa_vs_model(image_path, model_path, cfg_label_text, noise_terms, image_meta_common, reftime,
                               crop_fraction, figsize, cmap, vmax, conv_tag, image_rms_corner_size, suffix,
                               figdir, overwrite_plot=False, extra_meta=None, show_cal_error_text=False):
    """Plot a CASA image against a model and save the comparison figure."""
    meta = {
        **image_meta_common,
        'title': [image_meta_common.get('title_left', 'Cleaned image'), 'Model (convolved)'],
        'array_config': cfg_label_text,
        'tant': f"{noise_terms['tant']:.0f}K",
        'sigma_jy': f"{noise_terms['sigma_jy']:.1f}Jy/beam",
    }
    meta.pop('title_left', None)
    if extra_meta:
        meta.update(extra_meta)

    fig, _ = plot_two_casa_images_with_convolution_sunpy(
        image_path, model_path, crop_fraction=crop_fraction, reftime=reftime,
        figsize=figsize, image_meta=meta, cmap=cmap, vmax=vmax, vmin=0.1,
        conv_tag=conv_tag, overwrite_conv=True, rms_mask_radius=1.2,
        image_rms_corner_size=image_rms_corner_size, coord_frame='radec',
        match_fov=True, align_model_time_to_image=True, north_up=False,
        apply_hpc_p_angle=False, anchor_model_wcs_to_image=True,
        use_reftime_for_hpc=True, show_cal_error_text=show_cal_error_text,
    )
    plt.show()

    figname = os.path.join(
        figdir,
        f"fig-{base_tag_from_path(image_path)}_vs_{base_tag_from_path(model_path)}_{suffix}_{crop_fraction_tag(crop_fraction)}.jpg"
    )
    if os.path.exists(figname) and overwrite_plot:
        remove_path(figname)
    if not os.path.exists(figname):
        fig.savefig(figname, dpi=200)
        print('Saved figure:', figname)
    return fig, figname


@runtime_report
def get_local_noon_utc(cfg_path: str, date: datetime = None) -> Time:
    """
    Read longitude and latitude from a config file and compute local solar noon in UTC.

    :param cfg_path: Path to the .cfg file containing a line like '#COFA=-114.42610, 39.47780'
    :type cfg_path: str
    :param date: Date for which to compute local noon. If None, uses the current date/time.
    :type date: datetime.datetime, optional
    :return: Time of local solar noon in UTC
    :rtype: astropy.time.Time
    :raises ValueError: If the COFA line cannot be found or parsed
    :raises TypeError: If `date` is not a datetime object
    """
    # Parse the COFA line
    pattern = re.compile(r"#COFA\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)")
    lon = lat = None
    with open(cfg_path, 'r') as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                lon, lat = map(float, m.groups())
                break
    if lon is None or lat is None:
        raise ValueError("Could not find or parse '#COFA=' line in config")

    # Determine the central time
    if date is None:
        central_time = Time.now()
    else:
        if not isinstance(date, datetime):
            raise TypeError("`date` must be a datetime.datetime instance")
        central_time = Time(date)

    # Set up observer location
    location = EarthLocation(lat=lat * u.deg,
                             lon=lon * u.deg,
                             height=0 * u.m)

    # Create a time grid spanning ±12 hours around the central time
    times = central_time + np.linspace(-0.5, 0.5, 1001) * u.day
    frame = AltAz(obstime=times, location=location)

    # Compute Sun altitudes and find the maximum (local noon)
    sun_altitudes = get_sun(times).transform_to(frame).alt
    idx_noon = np.argmax(sun_altitudes)
    return times[idx_noon]


# Example usage:
# noon_utc = get_local_noon_utc("observatory.cfg")
# print("Local noon (UTC):", noon_utc.iso)

def calc_total_flux_on_dish(solar_model, dish_diameter=1.5, freqghz=None):
    # Read the solar model FITS file using Astropy.
    hdul = fits.open(solar_model)
    header = hdul[0].header
    flux = hdul[0].data[0, 0]  # Assume the model flux is in the primary HDU.

    # Read or normalize frequency to GHz as a float.
    if freqghz is None:
        freq_Hz = header.get('CRVAL3')
        if freq_Hz is None:
            raise ValueError("Frequency (CRVAL3) not found in FITS header.")
        freq_ghz = float(freq_Hz) / 1e9
    else:
        # Allow freqghz as a float (GHz) or as a string like "1.4GHz".
        if isinstance(freqghz, str):
            match = re.match(r"\s*([0-9.+eE-]+)", freqghz)
            if not match:
                raise ValueError(f"Could not parse frequency from string: {freqghz!r}")
            freq_ghz = float(match.group(1))
        else:
            freq_ghz = float(freqghz)

    # Calculate total flux from the model
    # Create a mask that corresponds to lambda / dish_diameter
    dx_model = header.get('CDELT1') * 3600.  # in arcsec per pixel
    dy_model = header.get('CDELT2') * 3600.  # in arcsec per pixel
    nx = header.get('NAXIS1')
    ny = header.get('NAXIS2')
    primary_beam = 1.22 * (3e10 / (freq_ghz * 1e9)) / (dish_diameter * 100) * (206265.)  # in arcsec
    # Mask out pixels outside the primary beam
    cx = int(nx / 2)
    cy = int(ny / 2)
    radius_pix = int(primary_beam / 2 * dx_model)  # masking out beyond half power point
    y_model, x_model = np.ogrid[-cy:ny - cy, -cx:nx - cx]
    mask = x_model * x_model + y_model * y_model >= radius_pix * radius_pix
    flux_masked = np.copy(flux)
    flux_masked[mask] = np.nan
    if header['bunit'].lower() in ['jy/px', 'jy/pixel']:
        total_flux = np.nansum(flux_masked) / 1e4  # Total flux in sfu units.
        print(f'Total flux within primary beam used to calcuate the antenna temperature: {total_flux:.1f} sfu')
    else:
        raise ValueError(f"Unsupported brightness unit in FITS header: {header['bunit']}")
    hdul.close()
    return total_flux


@runtime_report
def generate_ms(config_file, solar_model, reftime, freqghz=None, channel_width_mhz=10,
                integration_time=1., msname='fasr.ms', duration=None, tsys=300.,
                usehourangle=True, ra_deg=None, dec_deg=None, uv_cell=None):
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
      tsys : float
            System temperature in Kelvin.
      usehourangle : bool
            Whether to use hour angle for time specification.
      ra_deg: float or None
            Right Ascension of the source in degrees (if None, read from FITS header).
      dec_deg: float or None
            Declination of the source in degrees (if None, read from FITS header).
      uv_cell : float or None
            UV cell size in wavelengths passed through to ``calc_noise`` so the
            MS corruption noise uses the same weighting-grid assumption as any
            precomputed sensitivity estimate in the notebook.

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
    nant = len(dish_dia)
    N_bl = nant * (nant - 1) // 2
    try:
        ant_names = np.genfromtxt(
            config_file, comments='#', usecols=(4,), dtype=str)
    except Exception:
        ant_names = ['A' + "{0:03d}".format(i) for i in range(len(x))]

    # Read the solar model FITS file using Astropy.
    hdul = fits.open(solar_model)
    header = hdul[0].header

    # Read frequency from CRVAL3 (Hz) and convert to GHz.
    if freqghz is None:
        freq_Hz = header.get('CRVAL3')
        if freq_Hz is None:
            raise ValueError("Frequency (CRVAL3) not found in FITS header.")
        freq_GHz = f'{freq_Hz / 1e9}GHz'
    else:
        freq_GHz = freqghz

    dish_diameter = float(np.min(dish_dia))
    total_flux = calc_total_flux_on_dish(solar_model, dish_diameter=dish_diameter, freqghz=freq_GHz)

    # Read source RA and DEC from CRVAL1 and CRVAL2 (in degrees) and convert to radians.
    if ra_deg is None:
        ra_deg = header.get('CRVAL1')
    if dec_deg is None:
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
    print("Source RA = {:.6f} rad, DEC = {:.6f} rad".format(
        source_ra, source_dec))
    if usehourangle:
        print("I am using hour angle for time specification.")

    # Set the antenna configuration.
    sm.setconfig(telescopename="FASR",
                 x=x, y=y, z=z,
                 dishdiameter=dish_dia[0],
                 mount='alt-az',
                 antname=list(ant_names),  ## must be a list of strings. np.array will not work.
                 padname="FASR",
                 coordsystem='global')

    # Configure the spectral window using the frequency from the FITS header.
    chosen_index = 0
    sm.setspwindow(spwname='Band0',
                   freq=freq_GHz,
                   deltafreq=f'{channel_width_mhz}MHz',
                   freqresolution=f'{channel_width_mhz}MHz',
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
                usehourangle=usehourangle)

    if duration is None:
        duration = integration_time

    # Define observation start and stop times.
    starttime = f"0s"
    endtime = f"{duration}s"
    sm.observe("Sun", "Band0",
               starttime=starttime,
               stoptime=endtime,
               project="FASR",
               state_obs_mode="")

    sm.setdata(spwid=chosen_index)
    # Use the solar model FITS image as the sky model for prediction.
    dishdiam = dish_diameter
    vprec = vp.setpbairy(telescope='FASR', dishdiam='{0:.1f}m'.format(dishdiam),
                         blockagediam='0.5m', maxrad='{0:.3f}deg'.format(np.degrees(1.22 * 3e8 / (1e9 * dishdiam))),
                         reffreq=freq_GHz, dopb=True)
    sm.setvp(dovp=True, usedefaultvp=False)
    solar_model_copy = os.path.join(os.path.dirname(
        msname), os.path.basename(solar_model))
    solar_model_im = os.path.join(os.path.dirname(
        msname), os.path.basename(solar_model.replace('.fits', '.im')))
    shutil.copy2(solar_model, solar_model_copy)
    # if not os.path.exists(solar_model_im):
    update_fits_header(solar_model_copy, freq_GHz)
    ia.fromfits(outfile=solar_model_im,
                infile=solar_model_copy, overwrite=True)
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

    sm.close()

    sm.openfromms(msname)
    if tsys is None:
        print("No Tsys provided, skipping noise corruption.")
        pass
        # sm.setnoise(mode='tsys-atm', trx=500)
    else:
        noisejy, sigma_na, sigma_un = calc_noise(tsys, config_file, dish_diameter=dish_diameter,
                                                 total_flux=total_flux, duration=duration,
                                                 integration_time=integration_time, channel_width_mhz=channel_width_mhz,
                                                 freqghz=freq_GHz, uv_cell=uv_cell)
        sm.setnoise(mode='simplenoise', simplenoise=f'{noisejy:.2f}Jy')
        print(f'Corrupting the visibilities with a noise level of {noisejy:.2f} Jy (Tsys={tsys} K)')
        sm.corrupt()
    sm.done()
    print("Simulation complete. Measurement set generated:", msname)
    return msname


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
    # transpose for correct orientation
    cropped = pix[p1:p2, p1:p2].transpose()

    # Plot the cropped image using the WCS projection.
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': w})
    im = ax.imshow(cropped, origin='lower', cmap=plt.get_cmap(
        cmap), norm=norm, vmax=np.nanpercentile(cropped, 99.99))
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_title(title)

    return fig, ax

def calc_image_rms_snr(pix, rms_mask_radius, pixscale, solar_radius_arcsec=960.):
    """Compute RMS noise outside a circular mask, plus peak, min, SNR, and the mask.

    Parameters:
        pix (ndarray): 2D image array. Need to be centered on the solar disk.
        rms_mask_radius (float): Mask radius in units of solar radii.
        pixscale (float): Pixel scale in arcsec.
        solar_radius_arcsec (float): Solar radius in arcsec.

    Returns:
        (rms, datamax, datamin, snr, mask_outside):
            rms: standard-deviation noise outside the mask.
            datamax: maximum pixel value.
            datamin: minimum pixel value.
            snr: peak-to-noise ratio.
            mask_outside: boolean mask, True for pixels **outside** the mask radius.
    """
    shape = pix.shape[0]
    ics = int(shape / 2)
    radius_pix = int(solar_radius_arcsec * rms_mask_radius / pixscale)
    y, x = np.ogrid[-ics:shape - ics, -ics:shape - ics]
    mask_inside = x * x + y * y <= radius_pix * radius_pix
    mask_outside = ~mask_inside
    rms = np.std(pix[mask_outside])
    datamax = np.nanmax(pix)
    datamin = np.nanmin(pix)
    snr = datamax / rms if rms > 0 else np.inf
    return rms, datamax, datamin, snr, mask_outside


def calculate_rms_from_image(input_image, rms_mask_radius=1.2, solar_radius_arcsec=960.):
    """Convenience wrapper: read a CASA image and compute RMS outside a circular mask.

    Parameters:
        input_image (str): Path to the CASA image file.
        rms_mask_radius (float): Mask radius in units of solar radii.
        solar_radius_arcsec (float): Solar radius in arcsec.

    Returns:
        (rms, mask_outside): RMS noise and boolean mask (True = outside the mask radius).
    """
    img = read_casa_image_data(input_image)
    pixscale = (img['pixscale_x'] + img['pixscale_y']) / 2.0
    rms, _, _, _, mask_outside = calc_image_rms_snr(
        img['pix'], rms_mask_radius, pixscale, solar_radius_arcsec)
    return rms, mask_outside

def convolve_model_image_with_template(model_image, template_image, output_model=None, overwrite=True):
    """
    Convolve the model_image with the restoring beam from the template_image.
    Parameters:
        model_image (str): Path to the model image to be convolved
        template_image (str): Path to the template image from which to extract the restoring beam
        output_model (str or None): Path to save the convolved model image. If None, appends '.convolved' to model_image.
        overwrite (bool): Whether to overwrite the output_model if it already exists
    Returns:
        output_model (str): Path to the convolved model image
    """
    from casatools import image as IA
    ia = IA()
    if output_model is None:
        output_model = model_image.replace('.im', '.im.convolved')
    # --- Open the first image and extract data, coordinate system, and restoring beam ---
    ia.open(template_image)
    pix1 = ia.getchunk()[:, :, 0, 0]  # assume image shape [nx, ny, 1, 1]
    csys1 = ia.coordsys()
    # e.g., returns {'major': {'value': 6.0, 'unit': 'arcsec'},
    beam = ia.restoringbeam()
    #                   'minor': {'value': 6.0, 'unit': 'arcsec'},
    #                   'positionangle': {'value': 0.0, 'unit': 'deg'}}
    s = ia.summary()
    bunit = ia.summary()['unit']
    freqhz = s['refval'][3]
    ia.close()

    # --- Convolve the second image with the restoring beam from image1 using IA.convolve2d ---
    # Format beam parameters as strings.
    major = f"{beam['major']['value']}{beam['major']['unit']}"
    minor = f"{beam['minor']['value']}{beam['minor']['unit']}"
    pa = f"{beam['positionangle']['value']}{beam['positionangle']['unit']}"

    if overwrite or not os.path.exists(output_model):
        # Open the second image and apply convolution.
        ia.open(model_image)
        ia.convolve2d(outfile=output_model, axes=[0, 1], type='gauss',
                      major=major, minor=minor, pa=pa, overwrite=True)
        ia.close()

    # --- Read the convolved image to extract its pixel data ---
    ia.open(output_model)
    pix2 = ia.getchunk()[:, :, 0, 0]
    csys2 = ia.coordsys()
    s = ia.summary()
    bunit = ia.summary()['unit']
    ia.close()
    return output_model

def fidelity_evaluation(restored_image, model_image, fidelity_image=None,
                        doconvolve=True, rms_mask_radius=1.2, n_bins=50, sigma_jy=None,
                        solar_radius_arcsec=960., do_plot=True, figsize=(12, 5), fontsize=8,
                        xlim=None, ylim=None, 
                        image_arcsec_shift=(0.0, 0.0), image_pixel_shift=(0.0, 0.0)):
    """
    Performs the convolution and calculates the Fidelity Map.

    Parameters:
        restored_image (str): Path to the restored/cleaned image
        model_image (str): Path to the model image
        doconvolve (bool): Whether to convolve the model image to match the restored image's beam;
                if False, assumes the model image is already convolved.
        image_arcsec_shift (tuple): Manual sky shift ``(dx, dy)`` in arcsec
                applied to the restored image on the regridded comparison grid.
        image_pixel_shift (tuple): Manual pixel shift ``(dx, dy)`` applied to
                the restored image on the comparison grid.

    Returns:
        fidelity_map (ndarray): The spatial fidelity values (model / (restored - model))
        diff_map (ndarray): The difference (Artifacts) map
        convolved_model (ndarray): The resolution-matched truth
    """
    from casatools import image as IA
    from casatasks import imregrid
    import os
    from matplotlib.patches import Circle
    from matplotlib.colors import LogNorm
    plt.rcParams.update({'font.size': fontsize})

    ia = IA()

    if doconvolve:
        # 1. Convolve the Model Image to match the restored image's beam
        convolved_model_path = convolve_model_image_with_template(
            model_image=model_image,
            template_image=restored_image,
            output_model=model_image.replace('.im', '.im.convolved'),
            overwrite=True
        )
    else:
        convolved_model_path = model_image

    # 2. Regrid the convolved model to match the restored image's geometry
    regridded_model_path = convolved_model_path.replace('.convolved', '.convolved.regrid')
    if os.path.exists(regridded_model_path):
        os.system(f'rm -rf {regridded_model_path}')

    # 1. Regrid the model image to match the restored image geometry (pixels, direction)
    imregrid(imagename=convolved_model_path, template=restored_image, output=regridded_model_path, overwrite=True)

    # 3. Extract pixel data from both images
    ia.open(restored_image)
    restored_data = ia.getchunk()
    csys0 = ia.coordsys()
    cs0_rec = csys0.torecord()
    pixscale_x = -np.degrees(csys0.increment()['numeric'][0]) * 3600.  # arcsec
    pixscale_y = np.degrees(csys0.increment()['numeric'][1]) * 3600.  # arcsec
    s = ia.summary()
    bunit = ia.summary()['unit']
    freqhz = s['refval'][3]
    beam = ia.restoringbeam()
    jybm2k = 1.222e6 / (freqhz / 1e9) ** 2 / (beam['major']['value'] * beam['minor']['value'])
    ia.close()

    dx_pix = float(image_pixel_shift[0]) + float(image_arcsec_shift[0]) / abs(pixscale_x)
    dy_pix = float(image_pixel_shift[1]) + float(image_arcsec_shift[1]) / abs(pixscale_y)
    if not (np.isclose(dx_pix, 0.0) and np.isclose(dy_pix, 0.0)):
        restored_data[:, :, 0, 0] = ndimage_shift(
            restored_data[:, :, 0, 0],
            shift=(dy_pix, dx_pix),
            order=1,
            mode='constant',
            cval=0.0,
            prefilter=False,
        )
    peak = np.nanmax(restored_data)

    ia.open(regridded_model_path)
    model_data = ia.getchunk()
    ia.close()

    # 4. Calculate Fidelity and Difference Maps
    diff_map = restored_data - model_data
    noise_floor, mask = calculate_rms_from_image(restored_image, rms_mask_radius=rms_mask_radius,
                                                 solar_radius_arcsec=solar_radius_arcsec)
    if sigma_jy is None:
        sigma_jy = noise_floor  # in Jy/beam
        print(f'Estimated noise sigma from the restored image: {sigma_jy:.2f} Jy/beam')
    sigma_k = sigma_jy * jybm2k
    rms_radius_arcsec = rms_mask_radius * solar_radius_arcsec
    print(f'RMS noise in the restored image: {noise_floor:.1f}')
    fidelity_map = np.abs(model_data) / np.maximum(np.abs(diff_map), noise_floor)

    # Mask the region outside the solar disk to avoid artifacts in the fidelity map due to division by small numbers.
    fidelity_map[mask] = 0.0

    # Convert the fidelity map into CASA image
    if fidelity_image is None:
        fidelity_image = restored_image.replace('.im', '.fidelity.im')
    if os.path.exists(fidelity_image):
        os.system(f'rm -rf {fidelity_image}')
    ia.open(restored_image)
    # set the coordinate system and pixel data for the fidelity image the same as the restored image
    ia.fromarray(outfile=fidelity_image, pixels=fidelity_map.astype(np.float32), overwrite=True, csys=cs0_rec)
    ia.close()

    # Calculate statistics of the fidelity image
    # exclude zero values
    fidelity_valid = fidelity_map[fidelity_map > 0]
    model_valid = model_data[fidelity_map > 0]
    # convert the unit from jy/beam to k
    model_valid *= jybm2k
    fidelity_peak = np.max(fidelity_valid)
    fidelity_mean = np.mean(fidelity_valid)
    fidelity_median = np.median(fidelity_valid)
    fidelity_std = np.std(fidelity_valid)
    print(
        f'Fidelity Image Statistics - Peak: {fidelity_peak:.2f}, Mean: {fidelity_mean:.2f}, Median: {fidelity_median:.2f}, Std Dev: {fidelity_std:.2f}, SNR: {peak / noise_floor:.2f}')

    if do_plot:
        # Plot the fidelity map and the model image side by side for visual comparison
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax0 = axs[0]
        fidelity_data = fidelity_map[:, :, 0, 0] # assume shape [nx, ny, 1, 1]
        nx = fidelity_data.shape[1]
        ny = fidelity_data.shape[0]
        extent = (-nx / 2 * pixscale_x, nx / 2 * pixscale_x,
                  -ny / 2 * pixscale_y, ny / 2 * pixscale_y)
        im0 = ax0.imshow(fidelity_data.transpose(), origin='lower', cmap='viridis', vmin=0, vmax=10, extent=extent)

        # Dotted open circle indicating the RMS mask radius.
        rms_circle1 = Circle((0.0, 0.0), rms_radius_arcsec,
                             edgecolor='white', facecolor='none',
                             linestyle=':', linewidth=1.0)
        ax0.add_patch(rms_circle1)

        ax0.text(0.98, 0.02, f'Peak Fidelity: {fidelity_peak:.2f}\nMedian Fidelity: {fidelity_median:.2f}\nSNR: {peak / noise_floor:.2f}',
                 color='white', fontsize=8, ha='right', va='bottom', transform=ax0.transAxes, bbox=dict(facecolor='black', alpha=0.5, pad=5))
        ax0.text(0.98, 0.98, f'{freqhz / 1e9:.1f} GHz', color='white', fontsize=fontsize,
                 ha='right', va='top', transform=ax0.transAxes)
        ax0.set_title('Fidelity Image')
        ax0.set_xlabel('Solar X (arcsec)')
        ax0.set_ylabel('Solar Y (arcsec)')
        if xlim is not None:
            ax0.set_xlim(xlim)
        if ylim is not None:
            ax0.set_ylim(ylim)
        plt.colorbar(im0, ax=ax0, label='Fidelity')

        ax1 = axs[1]
        model_data_2d = model_data[:, :, 0, 0] * jybm2k# assume shape [nx, ny, 1, 1]
        if vmax is None:
            vmax = 90.
        if vmin is None:
            snr = peak / noise_floor
            vmin = -1000 / snr # set the minimum to -10 * rms (expressed in percentage)
            print(f'Estimated noise sigma from the restored image: {vmin:.2f} K')
        vmax_val = np.nanmax(model_data_2d) * float(vmax) / 100
        vmin_val = np.nanmax(model_data_2d) * float(vmin) / 100
        im1 = ax1.imshow(model_data_2d.transpose(), origin='lower', cmap='hinodexrt',
                         vmin=vmin_val, vmax=vmax_val, extent=extent)
        ax1.text(0.98, 0.98, f'{freqhz / 1e9:.1f} GHz', color='white', fontsize=fontsize,
                 ha='right', va='top', transform=ax1.transAxes)
        rms_circle2 = Circle((0.0, 0.0), rms_radius_arcsec,
                             edgecolor='white', facecolor='none',
                             linestyle=':', linewidth=1.0)
        ax1.add_patch(rms_circle2)
        ax1.set_title('Convolved Model Image')
        ax1.set_xlabel('Solar X (arcsec)')
        ax1.set_ylabel('Solar Y (arcsec)')
        if xlim is not None:
            ax1.set_xlim(xlim)
        if ylim is not None:
            ax1.set_ylim(ylim)
        cbar = plt.colorbar(im1, ax=ax1, label=r'T$_B$ [K]')
        cticks = cbar.ax.get_yticks()
        ctick_labels = [f'{tick:.0e}' for tick in cticks]
        cbar.ax.set_yticklabels(ctick_labels)
        plt.tight_layout()
        plt.show()

        # We divide the brightness range into bins and find the Median Fidelity for each bin.
        bins = np.linspace(np.min(model_valid), np.max(model_valid), n_bins)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        median_fidelity = []

        # Loop through bins to calculate median
        for i in range(len(bins) - 1):
            # Find pixels that fall into this brightness range
            mask_in_bin = (model_valid >= bins[i]) & (model_valid < bins[i + 1])
            if np.any(mask_in_bin):
                median_fidelity.append(np.median(fidelity_valid[mask_in_bin]))
            else:
                median_fidelity.append(np.nan)  # Handle empty bins

        # 4. Generate the Plot
        fig_stat = plt.figure(figsize=(10, 6))

        # A. The 2D Histogram (Heatmap)
        # Use LogNorm because there are way more Quiet Sun pixels than Active Region pixels
        h = plt.hist2d(model_valid / 1e6, fidelity_valid, bins=n_bins, cmap='inferno', norm=LogNorm())
        plt.colorbar(h[3], label='Number of Pixels (Log Scale)')

        # B. The Trend Line (Median Performance)
        plt.plot(bin_centers / 1e6, median_fidelity, 'c-', linewidth=3, marker='o',
                 markeredgecolor='k', label='Median Fidelity')

        # C. Formatting
        plt.title(f'Fidelity Analysis: Recovery by Source Brightness at {freqhz / 1e9:.1f} GHz')
        plt.xlabel(f'Model Brightness (MK)')
        plt.ylabel('Fidelity (Signal / Error)')
        plt.grid(True, linestyle='--', alpha=0.5)

        # Optional: Draw a "Good Science" line at Fidelity=3 or 5
        plt.axhline(y=5.0, color='r', linestyle='--', linewidth=2, label='"Good Science" Threshold (F=5)')
        plt.axvline(x=sigma_k * 5 / 1e6, color='k', linestyle='--', linewidth=2, label=r'$5\sigma_T$ ' + f'({sigma_k / 1e6:.4f} MK)')
        plt.legend()

        plt.tight_layout()
        plt.show()
        return fidelity_image, noise_floor, peak, fig, axs, fig_stat
    else:
        return fidelity_image, noise_floor, peak


def read_casa_image_data(image_filename):
    """Open a CASA image and return pixel data, coordinate metadata, and beam info.

    Parameters:
        image_filename (str): Path to the CASA image file.

    Returns:
        dict with keys: pix, beam, bunit, freqhz, pixscale_x, pixscale_y, nx, ny, extent
    """
    from casatools import image as IA
    ia = IA()
    ia.open(image_filename)
    pix = ia.getchunk()[:, :, 0, 0]  # assume shape [nx, ny, 1, 1]
    csys = ia.coordsys()
    try:
        beam = ia.restoringbeam()
    except Exception:
        beam = None
    s = ia.summary()
    bunit = s['unit']
    freqhz = s['refval'][3]
    ia.close()

    rad_to_deg = 180.0 / np.pi
    pixscale_x = abs(csys.increment()['numeric'][0]) * rad_to_deg * 3600.0
    pixscale_y = abs(csys.increment()['numeric'][1]) * rad_to_deg * 3600.0
    nx, ny = pix.shape
    extent = (-nx / 2 * pixscale_x, nx / 2 * pixscale_x,
              -ny / 2 * pixscale_y, ny / 2 * pixscale_y)

    return {
        'pix': pix, 'beam': beam, 'bunit': bunit, 'freqhz': freqhz,
        'pixscale_x': pixscale_x, 'pixscale_y': pixscale_y,
        'nx': nx, 'ny': ny, 'extent': extent
    }


def convert_flux_to_tb(pix, bunit, freqhz, beam, pixscale_x, pixscale_y):
    """Convert pixel flux values to brightness temperature (K).

    Parameters:
        pix (ndarray): 2D pixel data.
        bunit (str): Unit string from the image ('Jy/beam', 'Jy/pixel', or 'K').
        freqhz (float): Observing frequency in Hz.
        beam (dict): Restoring beam dict with 'major' and 'minor' sub-dicts.
        pixscale_x, pixscale_y (float): Pixel scales in arcsec.

    Returns:
        (pix_k, conv_factor) – the converted array and the conversion factor used.
    """
    if bunit.lower() == 'jy/beam':
        factor = 1.222e6 / (freqhz / 1e9) ** 2 / (beam['major']['value'] * beam['minor']['value'])
    elif bunit.lower() == 'jy/pixel':
        factor = 1.222e6 / (freqhz / 1e9) ** 2 / ((pixscale_x * pixscale_y) / (np.pi / (4 * np.log(2))))
    elif bunit.lower() == 'k':
        return pix.copy(), 1.0
    else:
        return pix.copy(), 1.0
    return pix * factor, factor


def crop_image(pix, crop_fraction):
    """Crop a 2D image array according to fractional bounds.

    Parameters:
        pix (ndarray): 2D image array (assumed square).
        crop_fraction: Either a (start, end) float tuple, or a pair of (start, end) tuples
                       for x and y separately, or anything else (returns pix unchanged).

    Returns:
        Cropped 2D array.
    """
    shape = pix.shape[0]
    if isinstance(crop_fraction[0], float):
        p1 = int(shape * crop_fraction[0])
        p2 = int(shape * crop_fraction[1])
        return pix[p1:p2, p1:p2]
    elif isinstance(crop_fraction[0], (tuple, list)):
        px1 = int(shape * crop_fraction[0][0])
        px2 = int(shape * crop_fraction[0][1])
        py1 = int(shape * crop_fraction[1][0])
        py2 = int(shape * crop_fraction[1][1])
        return pix[px1:px2, py1:py2]
    return pix


@runtime_report
def plot_two_casa_images_with_convolution(image1_filename, image2_filename,
                                          crop_fraction=(0.0, 1.0), rms_mask_radius=1.2, reftime=None,
                                          figsize=(12, 5),
                                          image_meta={'freq': '', 'title': ['', ''], 'array_config': ''},
                                          overlay_contours=False,
                                          contour_levels=None, cmap='viridis',
                                          conv_tag='',
                                          overwrite_conv=True, vmax=None, vmin=None,
                                          vmax2=None, vmin2=None, fontsize=8, legend_size=6,
                                          log_scale=False):
                                          xlim=None, ylim=None):
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
      log_scale : bool, optional
          If True, render images with logarithmic color scaling.
    Returns:
      fig, axs : tuple
          Matplotlib figure and axes objects.
    """
    from sunpy import coordinates
    from astropy.time import Time
    if reftime is None:
        reftime = Time.now()
    elif not isinstance(reftime, Time):
        reftime = Time(reftime)
    solar_radius_asec = coordinates.sun.angular_radius(reftime).value

    plt.rcParams.update({'font.size': fontsize})
    titiles = image_meta.get('title', ["", ""])
    title1 = titiles[0]
    title2 = titiles[1]
    freqstr = image_meta.get('freq', '')
    freqstr_plot = freqstr[1:] if isinstance(freqstr, str) and freqstr.startswith('0') else freqstr
    array_config = image_meta.get('array_config', '')
    array_config_text = image_meta.get('array_config_text', None)
    array_config_text_explicit = array_config_text is not None
    if array_config_text is None:
        array_config_text = array_config

    def _compact_array_cfg_label(cfg_text):
        if not isinstance(cfg_text, str) or not cfg_text:
            return cfg_text
        base = os.path.basename(cfg_text.rstrip('/'))
        if base.endswith('.cfg'):
            base = base[:-4]
        parts = base.split('_')
        parts_lower = [p.lower() for p in parts]
        if 'hybrid' in parts_lower and '120' in parts_lower:
            remote_count = next((p[6:] for p in parts_lower if p.startswith('remote') and len(p) > 6), None)
            range_token = next((p for p in parts_lower if p.startswith('r') and p.endswith('km') and '-' in p), None)
            if remote_count and range_token:
                range_label = range_token[1:].replace('km', ' km').replace('-', '–')
                return f'Hybrid 120 + {remote_count} remote ({range_label})'
            return 'Hybrid 120'
        return base

    if not array_config_text_explicit:
        array_config_text = _compact_array_cfg_label(array_config_text)
    if isinstance(array_config_text, str):
        array_config_text = os.path.basename(array_config_text.rstrip('/')) if array_config_text else ''
    weighting_meta = image_meta.get('weighting', None)
    weighting_explicit = image_meta.get('imaging_weighting', None)
    uvrange_text = image_meta.get('uvrange', None)

    def _infer_weighting_from_name(image_name):
        name = os.path.basename(image_name).lower()
        if 'superuniform' in name:
            return 'superuniform'
        if 'uniform' in name:
            return 'uniform'
        if 'natural' in name:
            return 'natural'
        if 'briggs' in name or 'robust' in name:
            return 'briggs'
        return None

    inferred_weighting = _infer_weighting_from_name(image1_filename)
    if weighting_explicit is not None:
        weighting = weighting_explicit
    elif isinstance(weighting_meta, str) and ('<' in weighting_meta or '>' in weighting_meta):
        # Preserve old notebooks where weighting carried a UV-range cut.
        weighting = inferred_weighting if inferred_weighting is not None else weighting_meta
        if uvrange_text is None:
            uvrange_text = weighting_meta
    else:
        weighting = weighting_meta if weighting_meta is not None else inferred_weighting
    tsys = image_meta.get('tsys', None)
    tant = image_meta.get('tant', None)
    sigma_jy = image_meta.get('sigma_jy', None)
    cal_error = image_meta.get('cal_error', None)
    dur = image_meta.get('duration', None)
    bandwidth = image_meta.get('bandwidth', None)
    if not compare_two:
        figsize = (figsize[0] / 3 * 2, figsize[1])
    weighting = image_meta.get('weighting', None)
    # --- Read image1 (restored image) ---
    img1 = read_casa_image_data(image1_filename)
    beam = img1['beam']
    freqhz = img1['freqhz']
    pixscale_x = img1['pixscale_x']
    pixscale_y = img1['pixscale_y']
    extent = img1['extent']

    # Convert image1 to brightness temperature
    pix1, jybm2k = convert_flux_to_tb(
        img1['pix'], img1['bunit'], freqhz, beam, pixscale_x, pixscale_y)
    if sigma_jy is not None:
        sigma_tb = float(sigma_jy.rstrip('Jy/beam')) * jybm2k
    else:
        sigma_tb = 0.0

    major_pix = beam['major']['value'] / pixscale_x
    minor_pix = beam['minor']['value'] / pixscale_y
    pa_deg = beam['positionangle']['value']

    # --- Convolve image2 with image1's restoring beam ---
    output_filename = image2_filename.replace('.im', f'{conv_tag}.im.convolved')
    convolve_model_image_with_template(
        model_image=image2_filename,
        template_image=image1_filename,
        output_model=output_filename,
        overwrite=overwrite_conv)

    # --- Read the convolved image to extract its pixel data ---
    ia.open(output_filename)
    pix2 = ia.getchunk()[:, :, 0, 0]
    csys2 = ia.coordsys()
    s = ia.summary()
    bunit = ia.summary()['unit']
    ia.close()

    w2 = WCS(naxis=2)
    w2.wcs.crpix = csys2.referencepixel()['numeric'][0:2]
    w2.wcs.cdelt = np.array(csys2.increment()['numeric'][0:2]) * rad_to_deg
    w2.wcs.crval = np.array(csys2.referencevalue()['numeric'][0:2]) * rad_to_deg
    w2.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    pixscale_x2 = abs(w2.wcs.cdelt[0]) * 3600.0
    pixscale_y2 = abs(w2.wcs.cdelt[1]) * 3600.0
    nx2 = pix2.shape[0]
    ny2 = pix2.shape[1]
    extent2 = (-nx2 / 2 * pixscale_x2, nx2 / 2 * pixscale_x2,
                  -ny2 / 2 * pixscale_y2, ny2 / 2 * pixscale_y2)

    if bunit.lower() == 'jy/beam':
        jybm2k = 1.222e6 / (freqhz / 1e9) ** 2 / (beam['major']['value'] * beam['minor']['value'])
        pix2 *= jybm2k
    elif bunit.lower() == 'jy/pixel':
        jypx2k = 1.222e6 / (freqhz / 1e9) ** 2 / ((pixscale_x2 * pixscale_y2) / (np.pi / (4 * np.log(2))))
        pix2 *= jypx2k
    elif bunit.lower() == 'k':
        pass

    # --- Crop both images using the same crop_fraction ---
    shape1 = pix1.shape[0]  # assume square images.
    shape2 = pix2.shape[0]

    print(f'Applying a mask of {rms_mask_radius:.1f} solar radii to calculate rms outside of it.')

    def _estimate_rms_outside_disk_or_border(pix, pixscale_arcsec, solar_radius_arcsec, mask_radius):
        ics_loc = int(pix.shape[0] / 2)
        radius_pix_loc = int(float(solar_radius_arcsec) * float(mask_radius) / float(pixscale_arcsec))
        yy, xx = np.ogrid[-ics_loc:pix.shape[0] - ics_loc, -ics_loc:pix.shape[0] - ics_loc]
        disk_mask = (xx * xx + yy * yy) <= (radius_pix_loc * radius_pix_loc)
        outside = np.where(~disk_mask, pix, np.nan)
        rms_loc = np.sqrt(np.nanmean(outside ** 2))
        if np.isfinite(rms_loc):
            return rms_loc
        border = max(2, int(0.1 * pix.shape[0]))
        border_vals = np.concatenate([
            pix[:border, :].ravel(),
            pix[-border:, :].ravel(),
            pix[:, :border].ravel(),
            pix[:, -border:].ravel(),
        ])
        border_vals = border_vals[np.isfinite(border_vals)]
        if border_vals.size == 0:
            return np.nan
        print('Warning: off-disk mask has no valid pixels in this FOV; using border RMS fallback.')
        return np.sqrt(np.nanmean(border_vals ** 2))

    datamax1 = np.nanmax(pix1)
    rms1 = _estimate_rms_outside_disk_or_border(
        pix1, pixscale_x, solar_radius_asec, rms_mask_radius
    )
    print(f'Peak of {os.path.basename(image1_filename)}: {datamax1:.3e} K')
    print(f'rms of {os.path.basename(image1_filename)} (excluding solar disk): {rms1:.3e} K')
    snr1 = datamax1 / rms1 if np.isfinite(rms1) and rms1 > 0 else np.nan
    print(f'SNR of the image: {snr1:.1f}')

    rms2 = _estimate_rms_outside_disk_or_border(
        pix2, pixscale_x2, solar_radius_asec, rms_mask_radius
    )
    datamax2 = np.nanmax(pix2)
    datamin2 = np.nanmin(pix2)
    snr2 = datamax2 / rms2 if np.isfinite(rms2) and rms2 > 0 else np.nan
    print(f'Peak of {os.path.basename(image2_filename)}: {datamax2:.3e} K')
    print(f'rms of {os.path.basename(image2_filename)} (excluding solar disk): {rms2:.3e} K')

    if isinstance(crop_fraction[0], float):
        pbl1 = int(shape1 * crop_fraction[0])
        ptr1 = int(shape1 * crop_fraction[1])
        pbl2 = int(shape2 * crop_fraction[0])
        ptr2 = int(shape2 * crop_fraction[1])
        cropped1 = pix1[pbl1:ptr1, pbl1:ptr1]
        cropped2 = pix2[pbl2:ptr2, pbl2:ptr2]
    elif isinstance(crop_fraction[0], tuple) or isinstance(crop_fraction[0], list):
        px1 = int(shape1 * crop_fraction[0][0])
        px2 = int(shape1 * crop_fraction[0][1])
        py1 = int(shape1 * crop_fraction[1][0])
        py2 = int(shape1 * crop_fraction[1][1])
        cropped1 = pix1[px1:px2, py1:py2]
        px1 = int(shape2 * crop_fraction[0][0])
        px2 = int(shape2 * crop_fraction[0][1])
        py1 = int(shape2 * crop_fraction[1][0])
        py2 = int(shape2 * crop_fraction[1][1])
        cropped2 = pix2[px1:px2, py1:py2]
    else:
        cropped1 = pix1
        cropped2 = pix2

    datamin1 = np.nanmin(cropped1)

    # Radius (in arcsec) of the mask used to compute the RMS.
    rms_radius_arcsec = solar_radius_asec * rms_mask_radius

    # --- Plotting: Create a figure with two panels using the WCS projection from image1 ---

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    if vmax is None:
        vmax = 90  # 90% of the maximum
    if vmin is None:
        vmin = -1000 / snr1  # set the minimum to -10 * rms (expressed in percentage)
    vmax_val = np.nanmax(cropped1) * float(vmax) / 100
    vmin_val = np.nanmax(cropped1) * float(vmin) / 100
    print(f'Plotting image with vmin={vmin_val:.3e} K, vmax={vmax_val:.3e} K')
    if vmax2 is None:
        vmax2 = vmax
    if vmin2 is None:
        vmin2 = vmin

    def _build_color_norm(data, vmin_pct, vmax_pct, use_log):
        from matplotlib.colors import Normalize
        peak = np.nanmax(data)
        vmin_local = peak * float(vmin_pct) / 100
        vmax_local = peak * float(vmax_pct) / 100
        if use_log:
            from matplotlib.colors import LogNorm
            positive = data[np.isfinite(data) & (data > 0)]
            if positive.size > 0:
                vmin_local = max(vmin_local, np.nanmin(positive))
                vmax_local = max(vmax_local, vmin_local * (1.0 + 1e-6))
                return LogNorm(vmin=vmin_local, vmax=vmax_local), vmin_local, vmax_local
            print('log_scale=True but image has no positive pixels; using linear scale instead.')
        if vmax_local <= vmin_local:
            vmax_local = vmin_local + np.finfo(float).eps
        return Normalize(vmin=vmin_local, vmax=vmax_local), vmin_local, vmax_local

    norm1, vmin_val, vmax_val = _build_color_norm(cropped1, vmin, vmax, log_scale)
    norm2, vmin_val2, vmax_val2 = _build_color_norm(cropped2, vmin2, vmax2, log_scale)
    plot_cmap = plt.get_cmap(cmap)
    if log_scale:
        plot_cmap = plot_cmap.copy()
        plot_cmap.set_under(plot_cmap(0.0))
        plot_cmap.set_bad(plot_cmap(0.0))
    print(f'Plotting image with vmin={vmin_val:.3e} K, vmax={vmax_val:.3e} K')
    print(f'Plotting convolved image with vmin={vmin_val2:.3e} K, vmax={vmax_val2:.3e} K')
    # Left panel: Image1 (original)
    ax1 = axs[0]
    im1 = ax1.imshow(cropped1.transpose(), origin='lower', cmap=plot_cmap,
                     norm=norm1, extent=extent1_plot)

    # Beam ellipse.
    ell = Ellipse((ax1.get_xlim()[0]*0.95, ax1.get_ylim()[0]*0.95),
                  width=major_pix*pixscale_x, height=minor_pix*pixscale_x,
                  angle=(-(90-pa_deg)),
                  edgecolor='white', facecolor='none', lw=1.5)
    ax1.add_patch(ell)

    # Dotted open circle indicating the RMS mask radius.
    rms_circle1 = Circle((0.0, 0.0), rms_radius_arcsec,
                         edgecolor='white', facecolor='none',
                         linestyle=':', linewidth=1.0)
    ax1.add_patch(rms_circle1)

    
    ax1.set_xlabel('Solar X (arcsec)')
    ax1.set_ylabel('Solar Y (arcsec)')
    if xlim is not None:
        ax1.set_xlim(xlim)
    if ylim is not None:
        ax1.set_ylim(ylim)
    ax1.set_title(title1)
    ax1.text(0.98, 0.01, r'$\sigma_I$' + f': {sigma_jy}, ' + r'$\sigma_T$' + f': {sigma_tb:.1e}K',
             transform=ax1.transAxes, ha='right', va='bottom', color='white', fontsize=legend_size)
    ax1.text(0.98, 0.04, r'T$_{B}^{min}$' + f': {datamin1:.1e} K', transform=ax1.transAxes, ha='right',
             va='bottom', color='white', fontsize=legend_size)
    ax1.text(0.98, 0.07, r'T$_{B}^{rms}$' + f': {rms1:.1e} K', transform=ax1.transAxes, ha='right',
             va='bottom', color='white', fontsize=legend_size)
    ax1.text(0.98, 0.10, r'T$_{B}^{max}$' + f': {datamax1:.1e} K', transform=ax1.transAxes, ha='right',
             va='bottom', color='white', fontsize=legend_size)
    ax1.text(0.98, 0.13, f'SNR: {snr1:.1f}', transform=ax1.transAxes, ha='right',
             va='bottom', color='white', fontsize=legend_size)
    ax1.text(0.98, 0.16,
             r"B$_{maj}$, B$_{min}$" + f": {bmaj_arcsec:.1f}\"" + f", {bmin_arcsec:.1f}\"",
             transform=ax1.transAxes, ha='right', va='bottom', color='white', fontsize=legend_size)
    ax1.text(0.98, 0.22, f'Weighting: {weighting}',
             transform=ax1.transAxes, ha='right', va='bottom', color='white', fontsize=legend_size)
    if uvrange_text is not None:
        ax1.text(0.98, 0.19, f'UV range: {uvrange_text}',
                 transform=ax1.transAxes, ha='right', va='bottom', color='white', fontsize=legend_size)
    ax1.text(0.98, 0.98, freqstr_plot, transform=ax1.transAxes, ha='right',
             va='top', color='white', fontsize=legend_size)
    ax1.text(0.02, 0.98, f'Array cfg: {array_config_text}', transform=ax1.transAxes, ha='left',
             va='top', color='white', fontsize=legend_size)
    ax1.text(0.02, 0.95, f'Dur: {dur}, Band: {bandwidth}', transform=ax1.transAxes, ha='left',
             va='top', color='white', fontsize=legend_size)
    ax1.text(0.02, 0.92, r'T$_{sys}$' + f': {tsys}, ' + r'T$_{ant}$' + f': {tant}',
             transform=ax1.transAxes, ha='left', va='top', color='white', fontsize=legend_size)
    ax1.text(0.02, 0.89, f'Cal err: {cal_error}', transform=ax1.transAxes, ha='left',
             va='top', color='white', fontsize=legend_size)
    cbar = plt.colorbar(im1, ax=ax1, label=r'T$_B$ [K]')
    cticks = cbar.ax.get_yticks()
    ctick_labels = [f'{tick:.0e}' for tick in cticks]
    cbar.ax.set_yticklabels(ctick_labels)

    # Right panel: Convolved Image2 as background.
    ax2 = axs[-1]
    im2 = ax2.imshow(cropped2.transpose(), origin='lower', cmap=plot_cmap,
                     norm=norm2, extent=extent2_plot)
    major_pix2 = beam['major']['value'] / pixscale_x2
    minor_pix2 = beam['minor']['value'] / pixscale_y2

    ax2.set_xlabel('Solar X (arcsec)')
    ax2.set_ylabel('Solar Y (arcsec)')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_title(title2)

    ell = Ellipse((ax2.get_xlim()[0]*0.95, ax2.get_ylim()[0]*0.95),
                  width=major_pix2*pixscale_x2, height=minor_pix2*pixscale_x2,
                  angle=(-(90-pa_deg)),
                  edgecolor='white', facecolor='none', lw=1.5)
    ax2.add_patch(ell)

    # Same RMS mask circle on the convolved image.
    rms_circle2 = Circle((0.0, 0.0), rms_radius_arcsec,
                         edgecolor='white', facecolor='none',
                         linestyle=':', linewidth=1.0)
    ax2.add_patch(rms_circle2)
    ax2.text(0.98, 0.01, r'T$_{B}^{min}$'+f': {datamin2:.1e} K', transform=ax2.transAxes, ha='right',
             va='bottom', color='white', fontsize=legend_size)
    ax2.text(0.98, 0.04, r'T$_{B}^{rms}$' + f': {rms2:.1e} K', transform=ax2.transAxes, ha='right',
             va='bottom', color='white', fontsize=legend_size)
    ax2.text(0.98, 0.07, r'T$_{B}^{max}$' + f': {datamax2:.1e} K', transform=ax2.transAxes, ha='right',
             va='bottom', color='white', fontsize=legend_size)
    ax2.text(0.98, 0.10, f'SNR: {snr2:.1f}', transform=ax2.transAxes, ha='right',
             va='bottom', color='white', fontsize=legend_size)
    ax2.text(0.98, 0.98, freqstr_plot, transform=ax2.transAxes, ha='right',
             va='top', color='white', fontsize=legend_size)
    if xlim is not None:
        ax2.set_xlim(xlim)
    if ylim is not None:
        ax2.set_ylim(ylim)
    # set ax2's xy limit to the same as ax1
    #ax2.set_xlim(ax1.get_xlim())
    #ax2.set_ylim(ax1.get_ylim())
    # fill the background of ax2 with black color
    ax2.set_facecolor('black')  # fill the background of ax2 with black color

    cbar = plt.colorbar(im2, ax=ax2, label=r'T$_B$ [K]')
    cticks = cbar.ax.get_yticks()
    ctick_labels = [f'{tick:.0e}' for tick in cticks]
    cbar.ax.set_yticklabels(ctick_labels)

    # Overlay contours from image1 onto the right panel.
    if overlay_contours:
        ax_comp = axs[1]
        if contour_levels is None:
            contour_levels = np.linspace(0.1, 0.9, 5) * np.nanmax(cropped1)
        else:
            contour_levels = np.array(contour_levels) * np.nanmax(cropped1)
        ax_comp.contour(cropped1.transpose(), levels=contour_levels, colors='tab:cyan', origin='lower',
                        linewidths=0.5, extent=extent1_plot)
        ax_comp.set_xlim(ax1.get_xlim())
        ax_comp.set_ylim(ax1.get_ylim())
        ax_comp.set_title('Contour: Left Panel, Background: Right Panel')

    plt.tight_layout()
    return fig, axs

@runtime_report
def plot_two_casa_images_with_convolution_sunpy(image1_filename, image2_filename,
                                                 crop_fraction=(0.0, 1.0), reftime=None,
                                                 figsize=(9, 4),
                                                 image_meta={'freq': '', 'title': ['', ''], 'array_config': ''},
                                                 panel_titles=None,
                                                 cmap='hinodexrt',
                                                 conv_tag='',
                                                 overwrite_conv=True,
                                                 convolve_model=True,
                                                 overwrite_fits=True,
                                                 coord_frame='hpc',
                                                 match_fov=True,
                                                 align_model_time_to_image=True,
                                                 north_up=False,
                                                 apply_hpc_p_angle=False,
                                                 anchor_model_wcs_to_image=False,
                                                 use_reftime_for_hpc=True,
                                                 hpc_reference_xy=None,
                                                 force_hpc_origin_at_image_center=False,
                                                 hpc_xy_shift=(0.0, 0.0),
                                                 image_arcsec_shift=(0.0, 0.0),
                                                 image_pixel_shift=(0.0, 0.0),
                                                 rms_mask_radius=1.2,
                                                 image_rms_corner_size=20,
                                                 legend_size=9,
                                                 log_scale=True,
                                                 vmin=None,
                                                 vmax=None,
                                                 norm_vmin_abs=None,
                                                 norm_vmax_abs=None,
                                                 show_cal_error_text=False,
                                                 show_slit_profiles=False,
                                                 slit_locations=None,
                                                 slit_x_fracs=(0.23, 0.49, 0.75),
                                                 slit_y_fracs=None,
                                                 slit_profile_axis='y',
                                                 slit_half_width_px=1,
                                                 slit_profile_lw=0.8,
                                                 overlay_image_contours=False,
                                                 image_contour_levels=(30, 50, 70, 90),
                                                 image_contour_kwargs=None,
                                                 return_prepared=False):
    """Plot two CASA images with SunPy maps after convolving image2 to image1 beam.

    This is a SunPy-map alternative to ``plot_two_casa_images_with_convolution``.
    It exports CASA images to FITS, optionally converts RA/Dec FITS to HPC FITS,
    and then plots with ``sunpy.map.Map`` on WCS axes.

    :param image1_filename: Path to the reference CASA image, or a list/tuple
        of CASA images for multi-image comparison mode.
    :type image1_filename: str or list[str] or tuple[str]
    :param image2_filename: Path to the model CASA image.
    :type image2_filename: str
    :param crop_fraction: Fractional crop definition. Either ``(f0, f1)`` for a
        square crop, or ``((x0, x1), (y0, y1))`` for independent axes.
    :type crop_fraction: tuple
    :param reftime: Observation reference time for HPC conversion.
    :type reftime: astropy.time.Time or str or None
    :param figsize: Figure size in inches.
    :type figsize: tuple
    :param image_meta: Plot metadata dict, including panel titles.
    :type image_meta: dict
    :param panel_titles: Optional list of titles for image panels in multi-image
        mode. If None, basenames are used.
    :type panel_titles: list[str] or None
    :param cmap: Matplotlib colormap name.
    :type cmap: str
    :param conv_tag: Tag appended to the convolved CASA image name.
    :type conv_tag: str
    :param overwrite_conv: If True, recompute convolved image2.
    :type overwrite_conv: bool
    :param convolve_model: If True, convolve ``image2`` to the restoring beam
        of ``image1`` before plotting. If False, plot ``image2`` directly.
    :type convolve_model: bool
    :param overwrite_fits: If True, overwrite exported FITS intermediates.
    :type overwrite_fits: bool
    :param coord_frame: Output plotting frame, either ``'radec'`` or ``'hpc'``.
    :type coord_frame: str
    :param match_fov: If True, apply image1 crop world bounds to image2 map.
    :type match_fov: bool
    :param align_model_time_to_image: If True, force model-map time metadata to
        match image1 map time before plotting. Useful when model FITS has stale
        DATE-OBS values.
    :type align_model_time_to_image: bool
    :param north_up: If True, rotate map data to solar north up before submap/plot.
        Default is False to preserve original pixel data orientation.
    :type north_up: bool
    :param apply_hpc_p_angle: If True, apply ``-sun.P(obstime)`` in the HPC
        header construction. Keep False when simulation images are already
        north-up.
    :type apply_hpc_p_angle: bool
    :param anchor_model_wcs_to_image: If True, replace model-map WCS metadata
        with image-map WCS metadata after conversion. This is useful for
        synthetic model products whose absolute RA/Dec header may be offset.
    :type anchor_model_wcs_to_image: bool
    :param use_reftime_for_hpc: If True, force RA/Dec->HPC conversion obstime to
        use ``reftime`` for both maps instead of FITS DATE-OBS.
    :type use_reftime_for_hpc: bool
    :param hpc_reference_xy: Optional fixed HPC reference center ``(x, y)`` in
        arcsec. When set, this overrides the RA/Dec-derived reference coordinate
        during HPC header construction.
    :type hpc_reference_xy: tuple or None
    :param force_hpc_origin_at_image_center: If True, set the HPC reference pixel
        to image center and assign ``hpc_reference_xy`` (default ``(0,0)``) there.
        This guarantees the plotted map center is the requested HPC origin.
    :type force_hpc_origin_at_image_center: bool
    :param hpc_xy_shift: Common post-conversion shift ``(dx, dy)`` in arcsec
        applied to both HPC maps before FOV matching/cropping.
    :type hpc_xy_shift: tuple
    :param image_arcsec_shift: Manual sky shift ``(dx, dy)`` in arcsec applied
        to the image panel on the prepared comparison grid.
    :type image_arcsec_shift: tuple
    :param image_pixel_shift: Manual pixel shift ``(dx, dy)`` applied to the
        image panel on the prepared comparison grid. Positive values move the
        image toward larger pixel indices in x/y.
    :type image_pixel_shift: tuple
    :param show_cal_error_text: If True, include the calibration-error text
        line in the left-panel annotation block. Default is False.
    :type show_cal_error_text: bool
    :param show_slit_profiles: If True, add slit-profile panels for the image
        and convolved-model comparison. If ``slit_locations`` is provided, slit
        panels are shown regardless of this toggle.
    :type show_slit_profiles: bool
    :param slit_locations: Optional fractional x positions for the slit cuts.
        When provided, these positions are used directly and the slit/profile
        panels are enabled.
    :type slit_locations: tuple or list or None
    :param slit_x_fracs: Fractional x positions used for the slit cuts when
        ``slit_locations`` is not provided. Retained for backward compatibility.
    :type slit_x_fracs: tuple or list
    :param slit_y_fracs: Fractional y positions used for horizontal slit cuts
        when ``slit_profile_axis`` includes ``'x'``. Defaults to
        ``slit_x_fracs`` when omitted.
    :type slit_y_fracs: tuple or list or None
    :param slit_profile_axis: Slit-profile direction to show. Use ``'y'`` for
        vertical cuts, ``'x'`` for horizontal cuts, or ``'both'`` for both.
    :type slit_profile_axis: str
    :param slit_half_width_px: Half-width in pixels used to average each slit.
    :type slit_half_width_px: int
    :param slit_profile_lw: Line width for the slit-profile curves.
    :type slit_profile_lw: float
    :param overlay_image_contours: If True, draw contours from the image panel
        on top of the model panel to make small spatial shifts easier to see.
    :type overlay_image_contours: bool
    :param image_contour_levels: Contour levels expressed as percent of the
        image peak when all values are <= 100, otherwise treated as absolute K.
    :type image_contour_levels: tuple or list
    :param image_contour_kwargs: Optional kwargs forwarded to
        :meth:`matplotlib.axes.Axes.contour` for the overlaid image contours.
    :type image_contour_kwargs: dict or None
    :returns: Figure and axes.
    :rtype: tuple
    :raises ValueError: If ``coord_frame`` is not supported.
    :raises RuntimeError: If HPC conversion requested but conversion fails.
    """
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    from sunpy.coordinates import frames, sun

    if reftime is None:
        reftime = Time.now()
    elif not isinstance(reftime, Time):
        reftime = Time(reftime)

    coord_mode = str(coord_frame).strip().lower()
    if coord_mode not in ('radec', 'hpc'):
        raise ValueError("coord_frame must be 'radec' or 'hpc'")

    titles = image_meta.get('title', ['', ''])
    title1 = titles[0] if len(titles) > 0 else 'Image 1'
    title2 = titles[1] if len(titles) > 1 else 'Image 2 (convolved)'
    active_slit_x_fracs = tuple(slit_locations) if slit_locations is not None else tuple(slit_x_fracs)
    active_slit_y_fracs = tuple(slit_y_fracs) if slit_y_fracs is not None else active_slit_x_fracs
    show_slit_profiles = bool(show_slit_profiles or slit_locations is not None)
    slit_profile_axis = str(slit_profile_axis).strip().lower()
    if slit_profile_axis not in ('x', 'y', 'both'):
        raise ValueError("slit_profile_axis must be 'x', 'y', or 'both'")

    def _crop_pixel_bounds(nx, ny, crop):
        if isinstance(crop[0], float):
            x0 = int(nx * crop[0]); x1 = int(nx * crop[1])
            y0 = int(ny * crop[0]); y1 = int(ny * crop[1])
        elif isinstance(crop[0], (tuple, list)):
            x0 = int(nx * crop[0][0]); x1 = int(nx * crop[0][1])
            y0 = int(ny * crop[1][0]); y1 = int(ny * crop[1][1])
        else:
            x0, x1, y0, y1 = 0, nx, 0, ny
        x0 = max(0, min(nx - 1, x0)); x1 = max(x0 + 1, min(nx, x1))
        y0 = max(0, min(ny - 1, y0)); y1 = max(y0 + 1, min(ny, y1))
        return x0, x1, y0, y1

    def _export_casa_image_to_fits(casa_image, fits_path):
        if overwrite_fits and os.path.exists(fits_path):
            os.system('rm -rf ' + fits_path)
        if overwrite_fits or not os.path.exists(fits_path):
            ia_local = IA()
            ia_local.open(casa_image)
            ia_local.tofits(fits_path, overwrite=True)
            ia_local.close()
        return fits_path

    def _fits_path_for_casa_image(casa_image, fits_tag):
        # Keep names unambiguous and avoid global string replacements like ".im" -> tag.
        return casa_image + fits_tag

    def _replace_nan_with_zero_in_fits(fits_path):
        hdul = fits.open(fits_path, mode='update')
        try:
            arr = hdul[0].data
            if arr is not None:
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                hdul[0].data = arr
                hdul.flush()
        finally:
            hdul.close()

    def _set_map_obstime(map_obj, obstime_target):
        meta = dict(map_obj.meta)
        isot = Time(obstime_target).isot
        meta['date-obs'] = isot
        meta['date_obs'] = isot
        return sunpy.map.Map(map_obj.data, meta)

    def _shift_hpc_map(map_obj, shift_xy):
        dx, dy = float(shift_xy[0]), float(shift_xy[1])
        if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
            return map_obj
        meta = dict(map_obj.meta)
        # In this workflow, HPC map CRVAL units are arcsec.
        meta['crval1'] = float(meta.get('crval1', 0.0)) + dx
        meta['crval2'] = float(meta.get('crval2', 0.0)) + dy
        return sunpy.map.Map(map_obj.data, meta)

    def _to_plot_map(casa_image, fits_tag, obstime_override=None):
        fits_path = _fits_path_for_casa_image(casa_image, fits_tag)
        _export_casa_image_to_fits(casa_image, fits_path)
        if coord_mode == 'radec':
            m = sunpy.map.Map(fits_path)
            if obstime_override is not None:
                m = _set_map_obstime(m, obstime_override)
            return m

        # Build an HPC map directly from the RA/Dec FITS header using SunPy/Astropy.
        # NaN/Inf are zeroed in the HPC branch as requested.
        try:
            hdul = fits.open(fits_path)
            hdr = hdul[0].header
            arr = hdul[0].data
            hdul.close()

            if arr is None:
                raise RuntimeError('FITS data is empty')
            arr = np.asarray(arr)
            if arr.ndim >= 4:
                data2d = arr[0, 0, :, :]
            elif arr.ndim == 3:
                data2d = arr[0, :, :]
            elif arr.ndim == 2:
                data2d = arr
            else:
                raise RuntimeError(f'Unsupported FITS data ndim={arr.ndim}')
            data2d = np.nan_to_num(data2d, nan=0.0, posinf=0.0, neginf=0.0)

            if obstime_override is not None:
                obstime = Time(obstime_override)
            elif use_reftime_for_hpc:
                obstime = Time(reftime)
            else:
                obstime = Time(hdr.get('date-obs', reftime.isot))
            obs_loc = EarthLocation(lat=37.232259 * u.deg, lon=-118.28479 * u.deg)
            obs_gcrs = SkyCoord(obs_loc.get_gcrs(obstime))

            cunit1 = u.Unit(hdr.get('cunit1', 'deg'))
            cunit2 = u.Unit(hdr.get('cunit2', 'deg'))
            reference_coord = SkyCoord(
                hdr['crval1'] * cunit1,
                hdr['crval2'] * cunit2,
                frame='gcrs',
                obstime=obstime,
                obsgeoloc=obs_gcrs.cartesian,
                obsgeovel=obs_gcrs.velocity.to_cartesian(),
                distance=obs_gcrs.hcrs.distance,
            )
            reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=obs_gcrs))
            if hpc_reference_xy is not None:
                reference_coord_arcsec = SkyCoord(
                    hpc_reference_xy[0] * u.arcsec,
                    hpc_reference_xy[1] * u.arcsec,
                    frame=frames.Helioprojective(observer=obs_gcrs, obstime=obstime),
                )

            cdelt1 = (np.abs(hdr['cdelt1']) * u.deg).to(u.arcsec)
            cdelt2 = (np.abs(hdr['cdelt2']) * u.deg).to(u.arcsec)
            p_angle = sun.P(obstime)
            rot_angle = -p_angle if apply_hpc_p_angle else 0.0 * u.deg

            if force_hpc_origin_at_image_center:
                ny, nx = data2d.shape
                refpix = [(nx - 1) / 2.0, (ny - 1) / 2.0]
            else:
                refpix = [hdr['crpix1'] - 1, hdr['crpix2'] - 1]

            mk_kwargs = dict(
                reference_pixel=u.Quantity(refpix * u.pixel),
                scale=u.Quantity([cdelt1, cdelt2] * u.arcsec / u.pix),
                rotation_angle=rot_angle,
                observatory='FASR',
            )
            if 'crval3' in hdr and np.isfinite(hdr['crval3']):
                mk_kwargs['wavelength'] = (hdr['crval3'] * u.Hz).to(u.MHz)
            new_header = sunpy.map.make_fitswcs_header(data2d, reference_coord_arcsec, **mk_kwargs)
            new_header['date-obs'] = obstime.isot
            return sunpy.map.Map(data2d, new_header)
        except Exception as e:
            raise RuntimeError(f'Failed to build HPC map from RA/Dec FITS: {e}')

    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.axes import Axes
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Normalize input so single and multiple images share one entry flow.
    image_list = [str(p) for p in image1_filename] if isinstance(image1_filename, (list, tuple)) else [str(image1_filename)]
    if len(image_list) == 0:
        raise ValueError('image1_filename list is empty')
    if len(image_list) > 1 and convolve_model:
        raise ValueError('Multi-image mode only supports convolve_model=False')

    image_titles = list(panel_titles) if panel_titles is not None else None
    if image_titles is None:
        if len(image_list) == 1:
            image_titles = [title1]
        else:
            image_titles = [os.path.basename(p) for p in image_list]

    def _single_call_meta(idx):
        meta_local = dict(image_meta)
        meta_local_titles = list(meta_local.get('title', [title1, title2]))
        if len(meta_local_titles) < 2:
            meta_local_titles = [title1, title2]
        meta_local_titles[0] = image_titles[idx] if idx < len(image_titles) else f'Image {idx + 1}'
        meta_local['title'] = meta_local_titles
        return meta_local

    def _build_shared_norm(prepared_list):
        common_parts = []
        for prep_item in prepared_list:
            common_parts.append(prep_item['data1'][np.isfinite(prep_item['data1'])].ravel())
        common_parts.append(prepared_list[0]['data2'][np.isfinite(prepared_list[0]['data2'])].ravel())
        common_data = np.concatenate([c for c in common_parts if c.size > 0]) if len(common_parts) > 0 else np.array([1.0])
        if common_data.size == 0:
            common_data = np.array([1.0])

        vmin_pct = (0.1 if vmin is None else vmin)
        vmax_pct = (90 if vmax is None else vmax)
        peak_ref = np.nanmax(common_data)
        if norm_vmin_abs is not None and norm_vmax_abs is not None:
            shared_vmin = float(norm_vmin_abs)
            shared_vmax = float(norm_vmax_abs)
        else:
            shared_vmin = peak_ref * float(vmin_pct) / 100.0
            shared_vmax = peak_ref * float(vmax_pct) / 100.0
        if log_scale:
            pos = common_data[np.isfinite(common_data) & (common_data > 0)]
            if pos.size > 0:
                shared_vmin = max(shared_vmin, np.nanmin(pos))
                shared_vmax = max(shared_vmax, shared_vmin * (1.0 + 1e-6))
            else:
                shared_vmin, shared_vmax = 1e-3, 1.0
            norm_shared = LogNorm(vmin=shared_vmin, vmax=shared_vmax)
        else:
            if shared_vmax <= shared_vmin:
                shared_vmax = shared_vmin + np.finfo(float).eps
            norm_shared = Normalize(vmin=shared_vmin, vmax=shared_vmax)
        return norm_shared

    def _render_prepared(prepared_list):
        norm_shared = _build_shared_norm(prepared_list)
        plot_cmap = plt.get_cmap(cmap)
        profile_axes_to_plot = ()
        if show_slit_profiles:
            if slit_profile_axis == 'both':
                profile_axes_to_plot = ('y', 'x')
            else:
                profile_axes_to_plot = (slit_profile_axis,)
        if log_scale:
            plot_cmap = plot_cmap.copy()
            plot_cmap.set_under(plot_cmap(0.0))
            plot_cmap.set_bad(plot_cmap(0.0))

        ntotal = len(prepared_list) + 1
        if ntotal == 4 and not show_slit_profiles:
            nrows, ncols = 2, 2
        elif ntotal == 2:
            nrows, ncols = 1, 2
        else:
            nrows, ncols = 1, ntotal

        if show_slit_profiles:
            nrows, ncols = 1, ntotal + len(profile_axes_to_plot)
        fig = plt.figure(figsize=figsize,
                         constrained_layout=(ntotal == 2 and not show_slit_profiles))
        gs = fig.add_gridspec(
            nrows, ncols, wspace=0.04 if show_slit_profiles else 0.02, hspace=0.04,
            width_ratios=([2] * ntotal + [1] * len(profile_axes_to_plot)) if show_slit_profiles else None
        )
        axs = []
        map_axes = []
        txt_kw = dict(color='white', fontsize=legend_size)

        def _add_beam(ax, prep_item):
            bmaj_arcsec = float(prep_item['beam']['major']['value'])
            bmin_arcsec = float(prep_item['beam']['minor']['value'])
            bpa_deg = float(prep_item['beam']['positionangle']['value'])
            beam_angle = (-(90.0 - bpa_deg))
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            bw = max(bmaj_arcsec, 0.01 * abs(x1 - x0))
            bh = max(bmin_arcsec, 0.01 * abs(y1 - y0))
            bx = x0 + 0.06 * (x1 - x0) + 0.5 * bw
            by = y0 + 0.06 * (y1 - y0) + 0.5 * bh
            ax.add_patch(Ellipse((bx, by), width=bw, height=bh, angle=beam_angle,
                                 edgecolor='none', facecolor='white', lw=0.0))

        def _annotate_image_panel(ax, prep_item):
            lt1 = [
                f'Array cfg: {prep_item["array_config_text"]}',
                f'Dur: {prep_item["dur"]}, Band: {prep_item["bandwidth"]}',
                r'T$_{sys}$' + f': {prep_item["tsys"]}, ' + r'T$_{ant}$' + f': {prep_item["tant"]}',
            ]
            if show_cal_error_text:
                lt1.append(f'Cal err: {prep_item["cal_error"]}')
            for j, txt in enumerate(lt1):
                ax.text(0.02, 0.98 - j * 0.06, txt, transform=ax.transAxes, ha='left', va='top', **txt_kw)

            rb1 = [
                r'$\sigma_I$' + f': {prep_item["sigma_jy"]}, ' + r'$\sigma_T$' + f': {prep_item["sigma_tb"]:.1e}K',
                r'T$_B^{min}$' + f': {prep_item["datamin1"]:.1e} K',
                r'T$_B^{rms}$' + f': {prep_item["rms1"]:.1e} K',
                r'T$_B^{max}$' + f': {prep_item["datamax1"]:.1e} K',
                f'SNR: {prep_item["snr1"]:.1f}',
                r"B$_{maj}$, B$_{min}$" + f": {float(prep_item['beam']['major']['value']):.1f}\""
                + f", {float(prep_item['beam']['minor']['value']):.1f}\"",
            ]
            if prep_item['uvrange_text'] is not None:
                rb1.append(f'UV range: {prep_item["uvrange_text"]}')
            if prep_item['weighting'] is not None:
                rb1.append(f'Weighting: {prep_item["weighting"]}')
            for j, txt in enumerate(rb1):
                ax.text(0.98, 0.02 + j * 0.055, txt, transform=ax.transAxes, ha='right', va='bottom', **txt_kw)
            ax.text(0.98, 0.98, prep_item['freqstr_plot'], transform=ax.transAxes, ha='right', va='top', **txt_kw)

        def _annotate_model_panel(ax, prep_item):
            if prep_item.get('right_panel_is_image', False):
                rb2 = [
                    r'$\sigma_I$' + f': {prep_item["sigma_jy"]}, ' + r'$\sigma_T$' + f': {prep_item["sigma_tb2"]:.1e}K',
                    r'T$_B^{min}$' + f': {prep_item["datamin2"]:.1e} K',
                    r'T$_B^{rms}$' + f': {prep_item["rms2"]:.1e} K',
                    r'T$_B^{max}$' + f': {prep_item["datamax2"]:.1e} K',
                    f'SNR: {prep_item["snr2"]:.1f}',
                    r"B$_{maj}$, B$_{min}$" + f": {float(prep_item['beam2']['major']['value']):.1f}\""
                    + f", {float(prep_item['beam2']['minor']['value']):.1f}\"",
                ]
                if prep_item['weighting'] is not None:
                    rb2.append(f'Weighting: {prep_item["weighting"]}')
            else:
                rb2 = [
                    r'T$_B^{min}$' + f': {prep_item["datamin2"]:.1e} K',
                    r'T$_B^{rms}$' + f': {prep_item["rms2"]:.1e} K',
                    r'T$_B^{max}$' + f': {prep_item["datamax2"]:.1e} K',
                    f'SNR: {prep_item["snr2"]:.1f}',
                ]
            for j, txt in enumerate(rb2):
                ax.text(0.98, 0.02 + j * 0.055, txt, transform=ax.transAxes, ha='right', va='bottom', **txt_kw)
            ax.text(0.98, 0.98, prep_item['freqstr_plot'], transform=ax.transAxes, ha='right', va='top', **txt_kw)

        def _add_slit_guides(ax, prep_item):
            data_model = np.asarray(prep_item['data2'], dtype=float)
            ny, nx = data_model.shape
            ncolors = max(len(active_slit_x_fracs), len(active_slit_y_fracs), 1)
            slit_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:ncolors]
            pix_transform = ax.get_transform('pixel')
            if slit_profile_axis in ('y', 'both'):
                for xf, color in zip(active_slit_x_fracs, slit_colors):
                    xpix = float(xf) * (nx - 1)
                    ax.plot([xpix, xpix], [0, ny - 1], color=color, lw=1.0, transform=pix_transform, zorder=10)
            if slit_profile_axis in ('x', 'both'):
                for yf, color in zip(active_slit_y_fracs, slit_colors):
                    ypix = float(yf) * (ny - 1)
                    ax.plot([0, nx - 1], [ypix, ypix], color=color, lw=1.0, transform=pix_transform, zorder=10)

        def _profile_coords(prep_item, axis):
            data_img = np.asarray(prep_item['data1'], dtype=float)
            ny, nx = data_img.shape
            try:
                pixscale_y_arcsec = abs(prep_item['sub1'].scale.axis2.to_value(u.arcsec / u.pix))
            except Exception:
                pixscale_y_arcsec = abs(prep_item['sub1'].scale.axis1.to_value(u.arcsec / u.pix))
            try:
                pixscale_x_arcsec = abs(prep_item['sub1'].scale.axis1.to_value(u.arcsec / u.pix))
            except Exception:
                pixscale_x_arcsec = pixscale_y_arcsec
            if axis == 'y':
                coord_arcsec = (np.arange(ny) - 0.5 * (ny - 1)) * pixscale_y_arcsec
                return data_img, ny, nx, coord_arcsec
            coord_arcsec = (np.arange(nx) - 0.5 * (nx - 1)) * pixscale_x_arcsec
            return data_img, ny, nx, coord_arcsec

        def _resolve_contour_levels(data_img):
            levels = np.atleast_1d(np.array(image_contour_levels, dtype=float))
            levels = levels[np.isfinite(levels)]
            if levels.size == 0:
                return np.array([])
            data_pos = data_img[np.isfinite(data_img) & (data_img > 0)]
            if data_pos.size == 0:
                return np.array([])
            peak_img = np.nanmax(data_pos)
            if np.nanmax(levels) <= 100.0:
                levels = peak_img * levels / 100.0
            levels = np.unique(np.sort(levels[(levels > 0) & (levels < peak_img * (1.0 + 1e-9))]))
            return levels

        def _overlay_image_contours_on_model(ax, prep_item):
            if not overlay_image_contours:
                return
            contour_levels = _resolve_contour_levels(np.asarray(prep_item['data1'], dtype=float))
            if contour_levels.size == 0:
                return
            contour_kwargs = dict(colors='tab:blue', linewidths=0.8, alpha=0.85)
            if isinstance(image_contour_kwargs, dict):
                contour_kwargs.update(image_contour_kwargs)
            ax.contour(
                np.asarray(prep_item['data1'], dtype=float),
                levels=contour_levels,
                transform=ax.get_transform(prep_item['sub1'].wcs),
                origin='lower',
                zorder=9,
                **contour_kwargs,
            )

        def _plot_slit_profiles(ax, prepared_items, axis='y', show_ylabel=False, title=None):
            fracs = active_slit_x_fracs if axis == 'y' else active_slit_y_fracs
            slit_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(fracs)]
            image_linestyles = ['-', '--', '-.', (0, (1.2, 1.2))]
            model_plotted = False

            for img_idx, prep_item in enumerate(prepared_items):
                data_img, ny, nx, coord_arcsec = _profile_coords(prep_item, axis)
                data_model = np.asarray(prep_item['data2'], dtype=float)
                image_ls = image_linestyles[min(img_idx, len(image_linestyles) - 1)]
                for frac, color in zip(fracs, slit_colors):
                    if axis == 'y':
                        xpix = int(round(float(frac) * (nx - 1)))
                        x0 = max(0, xpix - int(slit_half_width_px))
                        x1 = min(nx, xpix + int(slit_half_width_px) + 1)
                        prof_img = np.nanmean(data_img[:, x0:x1], axis=1)
                        prof_model = np.nanmean(data_model[:, x0:x1], axis=1)
                    else:
                        ypix = int(round(float(frac) * (ny - 1)))
                        y0 = max(0, ypix - int(slit_half_width_px))
                        y1 = min(ny, ypix + int(slit_half_width_px) + 1)
                        prof_img = np.nanmean(data_img[y0:y1, :], axis=0)
                        prof_model = np.nanmean(data_model[y0:y1, :], axis=0)
                    ax.plot(prof_img, coord_arcsec, color=color, lw=slit_profile_lw, linestyle=image_ls)
                    if not model_plotted:
                        ax.plot(prof_model, coord_arcsec, color=color, lw=slit_profile_lw, linestyle=':')
                model_plotted = True

            ax.set_xscale('log')
            ax.set_xlabel(r'T$_B$ [K]')
            if show_ylabel:
                ax.set_ylabel(f'Relative {axis.upper()} [arcsec]')
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
            if title is not None:
                ax.set_title(title)
            ax.grid(True, alpha=0.3, linestyle=':')
            xlim_prof = ax.get_xlim()
            ax.set_xlim(max(1e-4 * xlim_prof[1], np.finfo(float).tiny), xlim_prof[1])

        for i, prep_item in enumerate(prepared_list):
            r, c = (i // ncols, i % ncols)
            ax = fig.add_subplot(gs[r, c], projection=prep_item['sub1'])
            prep_item['sub1'].plot(axes=ax, cmap=plot_cmap, norm=norm_shared)
            if show_slit_profiles:
                ax.set_anchor('E')
            _add_beam(ax, prep_item)
            ax.set_title(prep_item['title1'])
            _annotate_image_panel(ax, prep_item)
            axs.append(ax)
            map_axes.append(ax)

        prep0 = prepared_list[0]
        i = len(prepared_list)
        r, c = (i // ncols, i % ncols)
        axm = fig.add_subplot(gs[r, c], projection=prep0['sub2'])
        im_model = prep0['sub2'].plot(axes=axm, cmap=plot_cmap, norm=norm_shared)
        if show_slit_profiles:
            axm.set_anchor('W')
        _add_beam(axm, prep0)
        _overlay_image_contours_on_model(axm, prep0)
        if show_slit_profiles:
            _add_slit_guides(axm, prep0)
        axm.set_title(prep0['title2'])
        _annotate_model_panel(axm, prep0)
        try:
            axm.coords[1].set_axislabel('')
        except Exception:
            pass
        axs.append(axm)
        map_axes.append(axm)

        if show_slit_profiles:
            legend_handles = [
                Line2D([0], [0], color='black', lw=slit_profile_lw, linestyle='-', label='Image'),
                Line2D([0], [0], color='black', lw=slit_profile_lw, linestyle=':', label='Model'),
            ]
            for idx_profile, axis_name in enumerate(profile_axes_to_plot):
                axp = fig.add_subplot(gs[0, ntotal + idx_profile])
                axp.set_anchor('W')
                _plot_slit_profiles(
                    axp,
                    prepared_list,
                    axis=axis_name,
                    show_ylabel=True,
                    title=f'{axis_name.upper()}-slit profiles',
                )
                if idx_profile == 0:
                    axp.legend(handles=legend_handles, fontsize=legend_size, ncol=1, loc='upper right')
                axs.append(axp)

        if show_slit_profiles:
            divider = make_axes_locatable(axm)
            cax = divider.append_axes("right", size="4.0%", pad=0.1, axes_class=Axes)
            cbar = fig.colorbar(im_model, cax=cax)
            cbar.ax.set_xlabel('')
            cbar.ax.xaxis.set_visible(False)
            cbar.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.ax.yaxis.set_label_position('right')
            cbar.ax.tick_params(axis='y', which='both', labelsize=legend_size - 1)
        elif ntotal == 2:
            cbar = fig.colorbar(im_model, ax=axs, label=r'T$_B$ [K]', fraction=0.046, pad=0.04)
        else:
            if ntotal == 4:
                cax = fig.add_axes([0.885, 0.16, 0.024, 0.70])
            else:
                cax = fig.add_axes([0.92, 0.14, 0.022, 0.72])
            sm = plt.cm.ScalarMappable(norm=norm_shared, cmap=plt.get_cmap(cmap))
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(r'T$_B$ [K]')
        ticks = cbar.ax.get_yticks()
        def _fmt_sci_tick(val):
            s = f'{val:.0e}'
            mant, exp = s.split('e')
            return f'{mant}e{int(exp)}'
        cbar.ax.set_yticklabels([_fmt_sci_tick(t) for t in ticks])

        if show_slit_profiles:
            fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.12, wspace=0.03)
        elif ntotal == 4:
            fig.subplots_adjust(left=0.03, right=0.86, top=0.97, bottom=0.04, wspace=0.2, hspace=0.2)
        elif ntotal != 2:
            fig.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.05, wspace=0.04)
        return fig, axs

    if not return_prepared:
        prepared = []
        for idx, img_path in enumerate(image_list):
            prepared.append(
                plot_two_casa_images_with_convolution_sunpy(
                    img_path,
                    image2_filename,
                    crop_fraction=crop_fraction,
                    reftime=reftime,
                    figsize=figsize,
                    image_meta=_single_call_meta(idx),
                    panel_titles=None,
                    cmap=cmap,
                    conv_tag=conv_tag,
                    overwrite_conv=overwrite_conv,
                    convolve_model=convolve_model,
                    overwrite_fits=overwrite_fits,
                    coord_frame=coord_frame,
                    match_fov=match_fov,
                    align_model_time_to_image=align_model_time_to_image,
                    north_up=north_up,
                    apply_hpc_p_angle=apply_hpc_p_angle,
                    anchor_model_wcs_to_image=anchor_model_wcs_to_image,
                    use_reftime_for_hpc=use_reftime_for_hpc,
                    hpc_reference_xy=hpc_reference_xy,
                    force_hpc_origin_at_image_center=force_hpc_origin_at_image_center,
                    hpc_xy_shift=hpc_xy_shift,
                    image_arcsec_shift=image_arcsec_shift,
                    image_pixel_shift=image_pixel_shift,
                    rms_mask_radius=rms_mask_radius,
                    image_rms_corner_size=image_rms_corner_size,
                    legend_size=legend_size,
                    log_scale=log_scale,
                    vmin=vmin,
                    vmax=vmax,
                    norm_vmin_abs=norm_vmin_abs,
                    norm_vmax_abs=norm_vmax_abs,
                    show_cal_error_text=show_cal_error_text,
                    show_slit_profiles=show_slit_profiles,
                    slit_locations=active_slit_x_fracs,
                    slit_x_fracs=slit_x_fracs,
                    slit_y_fracs=active_slit_y_fracs,
                    slit_profile_axis=slit_profile_axis,
                    slit_half_width_px=slit_half_width_px,
                    slit_profile_lw=slit_profile_lw,
                    overlay_image_contours=overlay_image_contours,
                    image_contour_levels=image_contour_levels,
                    image_contour_kwargs=image_contour_kwargs,
                    return_prepared=True,
                )
            )
        return _render_prepared(prepared)

    # Continue with the stable single-image path.
    image1_filename = image_list[0]

    # Build optional convolved image2 in CASA image format first.
    ia = IA()
    ia.open(image1_filename)
    beam = ia.restoringbeam()
    s1 = ia.summary()
    bunit1 = str(s1.get('unit', 'K')).lower()
    freqhz = s1['refval'][3]
    ia.close()

    if convolve_model:
        output_filename = image2_filename.replace('.im', f'{conv_tag}.im.convolved')
        major = f"{beam['major']['value']}{beam['major']['unit']}"
        minor = f"{beam['minor']['value']}{beam['minor']['unit']}"
        pa = f"{beam['positionangle']['value']}{beam['positionangle']['unit']}"
        if overwrite_conv or not os.path.exists(output_filename):
            ia.open(image2_filename)
            ia.convolve2d(outfile=output_filename, axes=[0, 1], type='gauss',
                          major=major, minor=minor, pa=pa, overwrite=True)
            ia.close()
    else:
        output_filename = image2_filename

    ia.open(output_filename)
    if convolve_model:
        beam2 = beam
    else:
        try:
            beam2 = ia.restoringbeam()
        except Exception:
            beam2 = beam
    pix2_full = ia.getchunk()[:, :, 0, 0]
    csys2 = ia.coordsys()
    s2 = ia.summary()
    bunit2 = str(s2.get('unit', 'K')).lower()
    inc2 = np.array(csys2.increment()['numeric'][0:2])
    pixscale_x2 = abs(inc2[0]) * (180.0 / np.pi) * 3600.0
    pixscale_y2 = abs(inc2[1]) * (180.0 / np.pi) * 3600.0
    ia.close()

    map1_obstime = Time(reftime) if (coord_mode == 'hpc' and use_reftime_for_hpc) else None
    map1 = _to_plot_map(image1_filename, '.sunpy_map.fits', obstime_override=map1_obstime)
    model_obstime = Time(map1.meta.get('date-obs', reftime.isot)) if align_model_time_to_image else None
    map2 = _to_plot_map(output_filename, '.sunpy_map.fits', obstime_override=model_obstime)
    print(f'Image1 map obstime: {map1.meta.get("date-obs", "N/A")}, Image2 map obstime: {map2.meta.get("date-obs", "N/A")}')

    if coord_mode == 'hpc' and anchor_model_wcs_to_image:
        # For synthetic simulations, enforce the same HPC WCS grid to avoid
        # absolute-coordinate offsets carried in model headers.
        meta2 = dict(map1.meta)
        if align_model_time_to_image and 'date-obs' in map1.meta:
            meta2['date-obs'] = map1.meta['date-obs']
            meta2['date_obs'] = map1.meta['date-obs']
        map2 = sunpy.map.Map(map2.data, meta2)

    if north_up:
        map1 = map1.rotate(recenter=True)
        map2 = map2.rotate(recenter=True)

    # Workflow: convert to HPC first, then apply common XY shift, then FOV match.
    if coord_mode == 'hpc':
        map1 = _shift_hpc_map(map1, hpc_xy_shift)
        map2 = _shift_hpc_map(map2, hpc_xy_shift)
        
    ny1, nx1 = map1.data.shape
    ix0, ix1, iy0, iy1 = _crop_pixel_bounds(nx1, ny1, crop_fraction)
    bl = map1.pixel_to_world(ix0 * u.pix, iy0 * u.pix)
    tr = map1.pixel_to_world((ix1 - 1) * u.pix, (iy1 - 1) * u.pix)
    try:
        sub1 = map1.submap(bl, top_right=tr)
    except Exception as e:
        print(f'Warning: submap failed for image1 ({e}); using full map.')
        sub1 = map1


        
    if match_fov:
        try:
            # Reproject model to the exact image1 submap grid to avoid panel offsets/gaps.
            sub2 = map2.reproject_to(sub1.wcs)
        except Exception as e:
            print(f'Warning: reproject_to failed for image2 ({e}); using submap fallback.')
            try:
                sub2 = map2.submap(bl, top_right=tr)
            except Exception:
                sub2 = map2
    else:
        ny2, nx2 = map2.data.shape
        jx0, jx1, jy0, jy1 = _crop_pixel_bounds(nx2, ny2, crop_fraction)
        bl2 = map2.pixel_to_world(jx0 * u.pix, jy0 * u.pix)
        tr2 = map2.pixel_to_world((jx1 - 1) * u.pix, (jy1 - 1) * u.pix)
        sub2 = map2.submap(bl2, top_right=tr2)

    freqstr = image_meta.get('freq', '')
    freqstr_plot = freqstr[1:] if isinstance(freqstr, str) and freqstr.startswith('0') else freqstr
    array_config = image_meta.get('array_config', '')
    array_config_text = image_meta.get('array_config_text', None)
    if array_config_text is None:
        array_config_text = array_config
    weighting_meta = image_meta.get('weighting', None)
    weighting_explicit = image_meta.get('imaging_weighting', None)
    uvrange_text = image_meta.get('uvrange', None)

    def _infer_weighting_from_name(image_name):
        name = os.path.basename(image_name).lower()
        if 'uniform' in name:
            return 'uniform'
        if 'natural' in name:
            return 'natural'
        if 'briggs' in name or 'robust' in name:
            return 'briggs'
        return None

    inferred_weighting = _infer_weighting_from_name(image1_filename)
    if weighting_explicit is not None:
        weighting = weighting_explicit
    elif isinstance(weighting_meta, str) and ('<' in weighting_meta or '>' in weighting_meta):
        weighting = inferred_weighting if inferred_weighting is not None else weighting_meta
        if uvrange_text is None:
            uvrange_text = weighting_meta
    else:
        weighting = weighting_meta if weighting_meta is not None else inferred_weighting

    tsys = image_meta.get('tsys', None)
    tant = image_meta.get('tant', None)
    sigma_jy = image_meta.get('sigma_jy', None)
    cal_error = image_meta.get('cal_error', None)
    dur = image_meta.get('duration', None)
    bandwidth = image_meta.get('bandwidth', None)

    def _compact_array_cfg_label(cfg_text):
        if not isinstance(cfg_text, str) or not cfg_text:
            return cfg_text
        base = os.path.basename(cfg_text.rstrip('/'))
        if base.endswith('.cfg'):
            base = base[:-4]
        parts = base.split('_')
        parts_lower = [s.lower() for s in parts]
        if 'hybrid' in parts_lower and '120' in parts_lower:
            remote_count = next((s[6:] for s in parts_lower if s.startswith('remote') and len(s) > 6), None)
            range_token = next((s for s in parts_lower if s.startswith('r') and s.endswith('km') and '-' in s), None)
            if remote_count and range_token:
                range_label = range_token[1:].replace('km', ' km').replace('-', '–')
                return f'Hybrid 120 + {remote_count} remote ({range_label})'
            return 'Hybrid 120'
        return base

    array_config_text = _compact_array_cfg_label(array_config_text)

    data1 = np.asarray(sub1.data, dtype=float)
    data2 = np.asarray(sub2.data, dtype=float)

    def _rms_from_corners(arr, box=20):
        ny, nx = arr.shape
        b = max(2, int(box))
        b = min(b, ny // 2, nx // 2)
        if b < 2:
            return np.nan
        corners = np.concatenate([
            arr[:b, :b].ravel(),
            arr[:b, -b:].ravel(),
            arr[-b:, :b].ravel(),
            arr[-b:, -b:].ravel(),
        ])
        corners = corners[np.isfinite(corners)]
        if corners.size == 0:
            return np.nan
        return np.sqrt(np.nanmean(corners ** 2))


    # Convert units to brightness temperature [K] to match the original plotting function.
    bmaj = float(beam['major']['value'])
    bmin = float(beam['minor']['value'])
    jybm2k = 1.222e6 / (freqhz / 1e9) ** 2 / (bmaj * bmin)
    bmaj2 = float(beam2['major']['value'])
    bmin2 = float(beam2['minor']['value'])
    jybm2k2 = 1.222e6 / (freqhz / 1e9) ** 2 / (bmaj2 * bmin2)

    def _pix_jy_to_k_factor(smap):
        try:
            px = abs(smap.scale.axis1.to_value(u.arcsec / u.pix))
            py = abs(smap.scale.axis2.to_value(u.arcsec / u.pix))
        except Exception:
            px = py = 1.0
        return 1.222e6 / (freqhz / 1e9) ** 2 / ((px * py) / (np.pi / (4 * np.log(2))))

    if bunit1 == 'jy/beam':
        data1 = data1 * jybm2k
    elif bunit1 == 'jy/pixel':
        data1 = data1 * _pix_jy_to_k_factor(sub1)

    if bunit2 == 'jy/beam':
        data2 = data2 * jybm2k2
    elif bunit2 == 'jy/pixel':
        data2 = data2 * _pix_jy_to_k_factor(sub2)

    try:
        image_pixscale_x_arcsec = abs(sub1.scale.axis1.to_value(u.arcsec / u.pix))
    except Exception:
        image_pixscale_x_arcsec = 1.0
    try:
        image_pixscale_y_arcsec = abs(sub1.scale.axis2.to_value(u.arcsec / u.pix))
    except Exception:
        image_pixscale_y_arcsec = image_pixscale_x_arcsec

    dx_pix = float(image_pixel_shift[0]) + float(image_arcsec_shift[0]) / image_pixscale_x_arcsec
    dy_pix = float(image_pixel_shift[1]) + float(image_arcsec_shift[1]) / image_pixscale_y_arcsec
    if not (np.isclose(dx_pix, 0.0) and np.isclose(dy_pix, 0.0)):
        # Apply the manual image shift on the final comparison grid so the
        # left-hand panel, contour overlay, and slit profiles stay consistent.
        data1 = ndimage_shift(
            data1,
            shift=(dy_pix, dx_pix),
            order=1,
            mode='constant',
            cval=0.0,
            prefilter=False,
        )

    # Rebuild maps so plotting and colorbars are in K.
    sub1 = sunpy.map.Map(data1, dict(sub1.meta))
    sub2 = sunpy.map.Map(data2, dict(sub2.meta))

    def _estimate_rms_outside_disk_or_border(pix, pixscale_arcsec):
        ny, nx = pix.shape
        ics_y = int(ny / 2)
        ics_x = int(nx / 2)
        rsun_arcsec = sun.angular_radius(reftime).to_value(u.arcsec)
        radius_pix = int(rsun_arcsec * float(rms_mask_radius) / pixscale_arcsec)
        yy, xx = np.ogrid[-ics_y:ny - ics_y, -ics_x:nx - ics_x]
        disk_mask = (xx * xx + yy * yy) <= (radius_pix * radius_pix)
        outside = np.where(~disk_mask, pix, np.nan)
        rms = np.sqrt(np.nanmean(outside ** 2))
        if np.isfinite(rms):
            return rms
        border = max(2, int(0.1 * min(ny, nx)))
        border_vals = np.concatenate([
            pix[:border, :].ravel(),
            pix[-border:, :].ravel(),
            pix[:, :border].ravel(),
            pix[:, -border:].ravel(),
        ])
        border_vals = border_vals[np.isfinite(border_vals)]
        if border_vals.size == 0:
            return np.nan
        print('Warning: off-disk mask has no valid pixels in model full-FOV; using border RMS fallback.')
        return np.sqrt(np.nanmean(border_vals ** 2))

    try:
        pixscale_sub1 = abs(sub1.scale.axis1.to_value(u.arcsec / u.pix))
    except Exception:
        pixscale_sub1 = pixscale_x2
    rms1 = _estimate_rms_outside_disk_or_border(data1, pixscale_sub1)
    if not np.isfinite(rms1):
        rms1 = _rms_from_corners(data1, image_rms_corner_size)
    datamax1, datamin1 = np.nanmax(data1), np.nanmin(data1)
    snr1 = datamax1 / rms1 if np.isfinite(rms1) and rms1 > 0 else np.nan

    # Model stats use the FULL convolved model image (uncropped), following the original method.
    pix2_model = np.array(pix2_full, dtype=float)
    if bunit2 == 'jy/beam':
        pix2_model *= jybm2k2
    elif bunit2 == 'jy/pixel':
        jypx2k2 = 1.222e6 / (freqhz / 1e9) ** 2 / ((pixscale_x2 * pixscale_y2) / (np.pi / (4 * np.log(2))))
        pix2_model *= jypx2k2
    rms2 = _estimate_rms_outside_disk_or_border(pix2_model, pixscale_x2)
    datamax2 = np.nanmax(pix2_model)
    datamin2 = np.nanmin(pix2_model)
    snr2 = datamax2 / rms2 if np.isfinite(rms2) and rms2 > 0 else np.nan

    if vmax is None:
        vmax = 90
    if vmin is None:
        vmin = (-1000.0 / snr1) if (np.isfinite(snr1) and snr1 != 0) else 0.1

    # Sigma conversion for annotation, same convention as original function.
    sigma_tb = np.nan
    sigma_tb2 = np.nan
    if sigma_jy is not None:
        try:
            sigma_tb = float(str(sigma_jy).rstrip('Jy/beam')) * jybm2k
        except Exception:
            sigma_tb = np.nan
        try:
            sigma_tb2 = float(str(sigma_jy).rstrip('Jy/beam')) * jybm2k2
        except Exception:
            sigma_tb2 = np.nan

    print(f'Peak of {os.path.basename(image1_filename)}: {datamax1:.3e} K')
    print(f'rms of {os.path.basename(image1_filename)} (off-disk/border): {rms1:.3e} K')
    print(f'Peak of {os.path.basename(image2_filename)}: {datamax2:.3e} K')
    print(f'rms of {os.path.basename(image2_filename)} (off-disk): {rms2:.3e} K')

    def _build_common_norm(data_ref, vmin_pct, vmax_pct, use_log,
                           vmin_abs=None, vmax_abs=None):
        peak_ref = np.nanmax(data_ref)
        if vmin_abs is not None and vmax_abs is not None:
            vmin_val = float(vmin_abs)
            vmax_val = float(vmax_abs)
        else:
            vmin_val = peak_ref * float(vmin_pct) / 100.0
            vmax_val = peak_ref * float(vmax_pct) / 100.0
        if use_log:
            pos_ref = data_ref[np.isfinite(data_ref) & (data_ref > 0)]
            if pos_ref.size > 0:
                vmin_val = max(vmin_val, np.nanmin(pos_ref))
                vmax_val = max(vmax_val, vmin_val * (1.0 + 1e-6))
                return LogNorm(vmin=vmin_val, vmax=vmax_val), vmin_val, vmax_val
            print('log_scale=True but image has no positive pixels; using linear scale instead.')
        if vmax_val <= vmin_val:
            vmax_val = vmin_val + np.finfo(float).eps
        return Normalize(vmin=vmin_val, vmax=vmax_val), vmin_val, vmax_val

    # Use one shared color scale for image+model, following the same percent-of-peak mechanism.
    common_data = np.concatenate([
        data1[np.isfinite(data1)].ravel(),
        data2[np.isfinite(data2)].ravel(),
    ])
    if common_data.size == 0:
        common_data = np.array([1.0])
    norm_common, vmin_val, vmax_val = _build_common_norm(
        common_data, vmin, vmax, log_scale,
        vmin_abs=norm_vmin_abs, vmax_abs=norm_vmax_abs
    )
    norm1 = norm_common
    norm2 = norm_common
    print(f'Plotting both panels with shared vmin={vmin_val:.3e} K, vmax={vmax_val:.3e} K')

    if return_prepared:
        return {
            'sub1': sub1,
            'sub2': sub2,
            'data1': data1,
            'data2': data2,
            'beam': beam,
            'beam2': beam2,
            'title1': title1,
            'title2': title2,
            'freqstr_plot': freqstr_plot,
            'array_config_text': array_config_text,
            'weighting': weighting,
            'uvrange_text': uvrange_text,
            'tsys': tsys,
            'tant': tant,
            'sigma_jy': sigma_jy,
            'sigma_tb': sigma_tb,
            'sigma_tb2': sigma_tb2,
            'cal_error': cal_error,
            'dur': dur,
            'bandwidth': bandwidth,
            'datamin1': datamin1,
            'datamax1': datamax1,
            'rms1': rms1,
            'snr1': snr1,
            'datamin2': datamin2,
            'datamax2': datamax2,
            'rms2': rms2,
            'snr2': snr2,
            'right_panel_is_image': (not convolve_model),
            'image_arcsec_shift': tuple(float(v) for v in image_arcsec_shift),
            'image_pixel_shift': (dx_pix, dy_pix),
            'norm_vmin': vmin_val,
            'norm_vmax': vmax_val,
        }


def measure_casa_image_model_offset_sunpy(image1_filename, image2_filename,
                                          crop_fraction=(0.0, 1.0), reftime=None,
                                          image_meta=None,
                                          conv_tag='',
                                          overwrite_conv=True,
                                          convolve_model=True,
                                          overwrite_fits=True,
                                          coord_frame='hpc',
                                          match_fov=True,
                                          align_model_time_to_image=True,
                                          north_up=False,
                                          apply_hpc_p_angle=False,
                                          anchor_model_wcs_to_image=False,
                                          use_reftime_for_hpc=True,
                                          hpc_reference_xy=None,
                                          force_hpc_origin_at_image_center=False,
                                          hpc_xy_shift=(0.0, 0.0),
                                          rms_mask_radius=1.2,
                                          image_rms_corner_size=20,
                                          offset_threshold_percent=20.0,
                                          focus_box=None,
                                          peak_box_size_px=21,
                                          return_debug_arrays=False):
    """Measure image-to-model spatial offset on the SunPy comparison grid.

    The returned shifts are the pixel/arcsec offsets that should be applied to
    the model image to align it with the reference image on the prepared grid.
    Positive ``shift_x_pix`` means shifting the model toward larger x-pixel
    index on the displayed array; positive ``shift_y_pix`` means toward larger
    y-pixel index.

    :param image1_filename: Reference CASA image path.
    :type image1_filename: str
    :param image2_filename: Model CASA image path.
    :type image2_filename: str
    :param crop_fraction: Crop definition passed through to the SunPy prep path.
    :type crop_fraction: tuple
    :param reftime: Observation reference time for HPC conversion.
    :type reftime: astropy.time.Time or str or None
    :param image_meta: Optional metadata dict containing at least ``freq`` when
        available.
    :type image_meta: dict or None
    :param conv_tag: Tag appended to the convolved CASA image name.
    :type conv_tag: str
    :param overwrite_conv: If True, recompute convolved image2.
    :type overwrite_conv: bool
    :param convolve_model: If True, convolve ``image2`` to the restoring beam
        of ``image1`` before measurement.
    :type convolve_model: bool
    :param overwrite_fits: If True, overwrite exported FITS intermediates.
    :type overwrite_fits: bool
    :param coord_frame: Output plotting frame, either ``'radec'`` or ``'hpc'``.
    :type coord_frame: str
    :param match_fov: If True, reproject the model to the image grid first.
    :type match_fov: bool
    :param align_model_time_to_image: If True, force model-map time metadata to
        match the image-map time.
    :type align_model_time_to_image: bool
    :param north_up: If True, rotate map data to solar north up.
    :type north_up: bool
    :param apply_hpc_p_angle: If True, apply ``-sun.P(obstime)`` in the HPC
        header construction.
    :type apply_hpc_p_angle: bool
    :param anchor_model_wcs_to_image: If True, replace model-map WCS metadata
        with image-map WCS metadata after conversion.
    :type anchor_model_wcs_to_image: bool
    :param use_reftime_for_hpc: If True, force RA/Dec->HPC conversion obstime to
        use ``reftime`` for both maps instead of FITS DATE-OBS.
    :type use_reftime_for_hpc: bool
    :param hpc_reference_xy: Optional fixed HPC reference center ``(x, y)`` in
        arcsec.
    :type hpc_reference_xy: tuple or None
    :param force_hpc_origin_at_image_center: If True, fix the HPC reference
        pixel at image center.
    :type force_hpc_origin_at_image_center: bool
    :param hpc_xy_shift: Common post-conversion shift ``(dx, dy)`` in arcsec.
    :type hpc_xy_shift: tuple
    :param rms_mask_radius: Radius factor used in off-disk RMS estimation.
    :type rms_mask_radius: float
    :param image_rms_corner_size: Corner-box size used for RMS fallback.
    :type image_rms_corner_size: int
    :param offset_threshold_percent: Keep only pixels above this fraction of
        each image peak before correlation. Set to None to disable thresholding.
    :type offset_threshold_percent: float or None
    :param focus_box: Optional fractional ROI ``((x0, x1), (y0, y1))`` on the
        prepared comparison grid. Use this to isolate the flare core and avoid
        large-scale morphology differences dominating the registration.
    :type focus_box: tuple or None
    :param peak_box_size_px: Size of the square box used for peak-core centroid
        measurement around each image maximum.
    :type peak_box_size_px: int
    :param return_debug_arrays: If True, include the thresholded arrays and
        correlation surface in the returned dict.
    :type return_debug_arrays: bool
    :returns: Offset diagnostics for one image/model pair.
    :rtype: dict
    """
    image_meta = {} if image_meta is None else dict(image_meta)
    prepared = plot_two_casa_images_with_convolution_sunpy(
        image1_filename,
        image2_filename,
        crop_fraction=crop_fraction,
        reftime=reftime,
        image_meta=image_meta,
        conv_tag=conv_tag,
        overwrite_conv=overwrite_conv,
        convolve_model=convolve_model,
        overwrite_fits=overwrite_fits,
        coord_frame=coord_frame,
        match_fov=match_fov,
        align_model_time_to_image=align_model_time_to_image,
        north_up=north_up,
        apply_hpc_p_angle=apply_hpc_p_angle,
        anchor_model_wcs_to_image=anchor_model_wcs_to_image,
        use_reftime_for_hpc=use_reftime_for_hpc,
        hpc_reference_xy=hpc_reference_xy,
        force_hpc_origin_at_image_center=force_hpc_origin_at_image_center,
        hpc_xy_shift=hpc_xy_shift,
        rms_mask_radius=rms_mask_radius,
        image_rms_corner_size=image_rms_corner_size,
        return_prepared=True,
    )

    data_img = np.asarray(prepared['data1'], dtype=float)
    data_model = np.asarray(prepared['data2'], dtype=float)

    def _focus_slices(shape, roi):
        ny, nx = shape
        if roi is None:
            return slice(0, ny), slice(0, nx)
        (x0f, x1f), (y0f, y1f) = roi
        x0 = max(0, min(nx - 1, int(np.floor(nx * float(x0f)))))
        x1 = max(x0 + 1, min(nx, int(np.ceil(nx * float(x1f)))))
        y0 = max(0, min(ny - 1, int(np.floor(ny * float(y0f)))))
        y1 = max(y0 + 1, min(ny, int(np.ceil(ny * float(y1f)))))
        return slice(y0, y1), slice(x0, x1)

    yslice, xslice = _focus_slices(data_img.shape, focus_box)
    data_img_focus = data_img[yslice, xslice]
    data_model_focus = data_model[yslice, xslice]

    def _prepare_for_offset(arr):
        work = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        finite = work[np.isfinite(work)]
        if finite.size == 0:
            raise ValueError('Offset measurement received an empty image')
        work = work - np.nanpercentile(finite, 5.0)
        work = np.clip(work, 0.0, None)
        peak = np.nanmax(work)
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError('Offset measurement image has no positive signal after background subtraction')
        if offset_threshold_percent is not None:
            cutoff = peak * float(offset_threshold_percent) / 100.0
            work = np.where(work >= cutoff, work, 0.0)
        total = np.nansum(work)
        if not np.isfinite(total) or total <= 0:
            raise ValueError('Offset measurement threshold removed all signal; lower offset_threshold_percent')
        return work / total

    def _quadratic_peak_offset(arr, peak_idx):
        offsets = []
        for axis, idx in enumerate(peak_idx):
            if idx <= 0 or idx >= arr.shape[axis] - 1:
                offsets.append(0.0)
                continue
            slicer = list(peak_idx)
            slicer[axis] = idx - 1
            fm1 = float(arr[tuple(slicer)])
            slicer[axis] = idx
            f0 = float(arr[tuple(slicer)])
            slicer[axis] = idx + 1
            fp1 = float(arr[tuple(slicer)])
            denom = fm1 - 2.0 * f0 + fp1
            if np.isclose(denom, 0.0):
                offsets.append(0.0)
            else:
                delta = 0.5 * (fm1 - fp1) / denom
                offsets.append(float(np.clip(delta, -1.0, 1.0)))
        return np.array(offsets, dtype=float)

    def _weighted_centroid(arr):
        yy, xx = np.indices(arr.shape, dtype=float)
        total = np.nansum(arr)
        if not np.isfinite(total) or total <= 0:
            return np.array([np.nan, np.nan], dtype=float)
        cy = np.nansum(yy * arr) / total
        cx = np.nansum(xx * arr) / total
        return np.array([cy, cx], dtype=float)

    def _peak_location(arr):
        peak_idx_int = np.array(np.unravel_index(np.nanargmax(arr), arr.shape), dtype=int)
        return peak_idx_int.astype(float) + _quadratic_peak_offset(arr, peak_idx_int)

    def _local_peak_centroid(arr, peak_yx, box_size):
        half = max(1, int(box_size) // 2)
        py = int(round(float(peak_yx[0])))
        px = int(round(float(peak_yx[1])))
        y0 = max(0, py - half)
        y1 = min(arr.shape[0], py + half + 1)
        x0 = max(0, px - half)
        x1 = min(arr.shape[1], px + half + 1)
        sub = np.array(arr[y0:y1, x0:x1], dtype=float)
        sub = np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0)
        sub = sub - np.nanmin(sub)
        sub = np.clip(sub, 0.0, None)
        centroid_local = _weighted_centroid(sub)
        return np.array([y0, x0], dtype=float) + centroid_local

    img_work = _prepare_for_offset(data_img_focus)
    model_work = _prepare_for_offset(data_model_focus)

    corr = fftconvolve(img_work, model_work[::-1, ::-1], mode='same')
    peak_idx_int = np.array(np.unravel_index(np.nanargmax(corr), corr.shape), dtype=int)
    peak_idx = peak_idx_int.astype(float) + _quadratic_peak_offset(corr, peak_idx_int)
    center_idx = 0.5 * (np.array(corr.shape, dtype=float) - 1.0)
    shift_yx_pix = peak_idx - center_idx

    centroid_img_yx = _weighted_centroid(img_work)
    centroid_model_yx = _weighted_centroid(model_work)
    centroid_shift_yx_pix = centroid_img_yx - centroid_model_yx
    peak_img_yx_pix = _peak_location(data_img_focus)
    peak_model_yx_pix = _peak_location(data_model_focus)
    peak_shift_yx_pix = peak_img_yx_pix - peak_model_yx_pix
    core_img_yx_pix = _local_peak_centroid(data_img_focus, peak_img_yx_pix, peak_box_size_px)
    core_model_yx_pix = _local_peak_centroid(data_model_focus, peak_model_yx_pix, peak_box_size_px)
    core_shift_yx_pix = core_img_yx_pix - core_model_yx_pix

    try:
        pixscale_x_arcsec = abs(prepared['sub1'].scale.axis1.to_value(u.arcsec / u.pix))
    except Exception:
        pixscale_x_arcsec = 1.0
    try:
        pixscale_y_arcsec = abs(prepared['sub1'].scale.axis2.to_value(u.arcsec / u.pix))
    except Exception:
        pixscale_y_arcsec = pixscale_x_arcsec

    result = {
        'freq': image_meta.get('freq', prepared.get('freqstr_plot', '')),
        'image_path': str(image1_filename),
        'model_path': str(image2_filename),
        'shift_x_pix': float(shift_yx_pix[1]),
        'shift_y_pix': float(shift_yx_pix[0]),
        'shift_x_arcsec': float(shift_yx_pix[1] * pixscale_x_arcsec),
        'shift_y_arcsec': float(shift_yx_pix[0] * pixscale_y_arcsec),
        'shift_r_arcsec': float(np.hypot(shift_yx_pix[1] * pixscale_x_arcsec,
                                         shift_yx_pix[0] * pixscale_y_arcsec)),
        'centroid_shift_x_pix': float(centroid_shift_yx_pix[1]),
        'centroid_shift_y_pix': float(centroid_shift_yx_pix[0]),
        'centroid_shift_x_arcsec': float(centroid_shift_yx_pix[1] * pixscale_x_arcsec),
        'centroid_shift_y_arcsec': float(centroid_shift_yx_pix[0] * pixscale_y_arcsec),
        'centroid_shift_r_arcsec': float(np.hypot(centroid_shift_yx_pix[1] * pixscale_x_arcsec,
                                                  centroid_shift_yx_pix[0] * pixscale_y_arcsec)),
        'peak_shift_x_pix': float(peak_shift_yx_pix[1]),
        'peak_shift_y_pix': float(peak_shift_yx_pix[0]),
        'peak_shift_x_arcsec': float(peak_shift_yx_pix[1] * pixscale_x_arcsec),
        'peak_shift_y_arcsec': float(peak_shift_yx_pix[0] * pixscale_y_arcsec),
        'peak_shift_r_arcsec': float(np.hypot(peak_shift_yx_pix[1] * pixscale_x_arcsec,
                                              peak_shift_yx_pix[0] * pixscale_y_arcsec)),
        'core_shift_x_pix': float(core_shift_yx_pix[1]),
        'core_shift_y_pix': float(core_shift_yx_pix[0]),
        'core_shift_x_arcsec': float(core_shift_yx_pix[1] * pixscale_x_arcsec),
        'core_shift_y_arcsec': float(core_shift_yx_pix[0] * pixscale_y_arcsec),
        'core_shift_r_arcsec': float(np.hypot(core_shift_yx_pix[1] * pixscale_x_arcsec,
                                              core_shift_yx_pix[0] * pixscale_y_arcsec)),
        'offset_threshold_percent': offset_threshold_percent,
        'focus_box': focus_box,
        'correlation_peak': float(np.nanmax(corr)),
        'pixscale_x_arcsec': float(pixscale_x_arcsec),
        'pixscale_y_arcsec': float(pixscale_y_arcsec),
    }
    if return_debug_arrays:
        result['image_work'] = img_work
        result['model_work'] = model_work
        result['correlation'] = corr
        result['image_focus'] = data_img_focus
        result['model_focus'] = data_model_focus
    return result


def measure_casa_image_model_offsets_sunpy(image1_filenames, image2_filenames,
                                           freq_labels=None,
                                           image_meta_list=None,
                                           **kwargs):
    """Measure image-to-model offsets for multiple frequency/image pairs.

    :param image1_filenames: Reference CASA image paths.
    :type image1_filenames: list[str] or tuple[str]
    :param image2_filenames: Model CASA image paths, or a single shared path.
    :type image2_filenames: list[str] or tuple[str] or str
    :param freq_labels: Optional frequency labels used in the results.
    :type freq_labels: list[str] or tuple[str] or None
    :param image_meta_list: Optional list of metadata dicts parallel to
        ``image1_filenames``.
    :type image_meta_list: list[dict] or tuple[dict] or None
    :param kwargs: Forwarded to :func:`measure_casa_image_model_offset_sunpy`.
    :type kwargs: dict
    :returns: Offset diagnostics for all pairs.
    :rtype: list[dict]
    """
    image1_list = list(image1_filenames)
    if isinstance(image2_filenames, str):
        image2_list = [image2_filenames] * len(image1_list)
    else:
        image2_list = list(image2_filenames)
    if len(image1_list) != len(image2_list):
        raise ValueError('image1_filenames and image2_filenames must have the same length')
    if freq_labels is not None and len(freq_labels) != len(image1_list):
        raise ValueError('freq_labels must match image1_filenames length')
    if image_meta_list is not None and len(image_meta_list) != len(image1_list):
        raise ValueError('image_meta_list must match image1_filenames length')

    results = []
    for idx, (image_path, model_path) in enumerate(zip(image1_list, image2_list)):
        meta_local = {}
        if image_meta_list is not None and image_meta_list[idx] is not None:
            meta_local.update(dict(image_meta_list[idx]))
        if freq_labels is not None:
            meta_local.setdefault('freq', freq_labels[idx])
        result = measure_casa_image_model_offset_sunpy(
            image_path,
            model_path,
            image_meta=meta_local,
            **kwargs,
        )
        if freq_labels is not None:
            result['freq'] = freq_labels[idx]
        results.append(result)
    return results


def debug_casa_image_model_registration_sunpy(image1_filename, image2_filename,
                                              crop_fraction=(0.0, 1.0), reftime=None,
                                              image_meta=None,
                                              conv_tag='',
                                              overwrite_conv=True,
                                              convolve_model=True,
                                              overwrite_fits=True,
                                              coord_frame='hpc',
                                              match_fov=True,
                                              align_model_time_to_image=True,
                                              north_up=False,
                                              apply_hpc_p_angle=False,
                                              anchor_model_wcs_to_image=False,
                                              use_reftime_for_hpc=True,
                                              hpc_reference_xy=None,
                                              force_hpc_origin_at_image_center=False,
                                              hpc_xy_shift=(0.0, 0.0)):
    """Return peak-location and beam diagnostics for one image/model pair.

    :param image1_filename: Reference CASA image path.
    :type image1_filename: str
    :param image2_filename: Model CASA image path.
    :type image2_filename: str
    :param crop_fraction: Crop definition passed through to the SunPy prep path.
    :type crop_fraction: tuple
    :param reftime: Observation reference time for HPC conversion.
    :type reftime: astropy.time.Time or str or None
    :param image_meta: Optional metadata dict containing at least ``freq``.
    :type image_meta: dict or None
    :returns: Registration diagnostics including peak world coordinates and
        offset-to-beam ratios.
    :rtype: dict
    """
    image_meta = {} if image_meta is None else dict(image_meta)
    prepared = plot_two_casa_images_with_convolution_sunpy(
        image1_filename,
        image2_filename,
        crop_fraction=crop_fraction,
        reftime=reftime,
        image_meta=image_meta,
        conv_tag=conv_tag,
        overwrite_conv=overwrite_conv,
        convolve_model=convolve_model,
        overwrite_fits=overwrite_fits,
        coord_frame=coord_frame,
        match_fov=match_fov,
        align_model_time_to_image=align_model_time_to_image,
        north_up=north_up,
        apply_hpc_p_angle=apply_hpc_p_angle,
        anchor_model_wcs_to_image=anchor_model_wcs_to_image,
        use_reftime_for_hpc=use_reftime_for_hpc,
        hpc_reference_xy=hpc_reference_xy,
        force_hpc_origin_at_image_center=force_hpc_origin_at_image_center,
        hpc_xy_shift=hpc_xy_shift,
        return_prepared=True,
    )

    def _peak_location(arr):
        peak_idx = np.array(np.unravel_index(np.nanargmax(arr), arr.shape), dtype=float)
        return peak_idx

    data1 = np.asarray(prepared['data1'], dtype=float)
    data2 = np.asarray(prepared['data2'], dtype=float)
    peak1_yx = _peak_location(data1)
    peak2_yx = _peak_location(data2)

    world1 = prepared['sub1'].pixel_to_world(peak1_yx[1] * u.pix, peak1_yx[0] * u.pix)
    world2 = prepared['sub2'].pixel_to_world(peak2_yx[1] * u.pix, peak2_yx[0] * u.pix)

    def _world_xy_arcsec(world_a, world_b):
        try:
            xa = world_a.Tx.to_value(u.arcsec)
            ya = world_a.Ty.to_value(u.arcsec)
            xb = world_b.Tx.to_value(u.arcsec)
            yb = world_b.Ty.to_value(u.arcsec)
            dx = xa - xb
            dy = ya - yb
            return xa, ya, xb, yb, dx, dy
        except Exception:
            pass
        try:
            ra_a = world_a.ra.to(u.arcsec)
            dec_a = world_a.dec.to(u.arcsec)
            ra_b = world_b.ra.to(u.arcsec)
            dec_b = world_b.dec.to(u.arcsec)
            dra, ddec = world_b.spherical_offsets_to(world_a)
            dx = dra.to_value(u.arcsec)
            dy = ddec.to_value(u.arcsec)
            return ra_a.value, dec_a.value, ra_b.value, dec_b.value, dx, dy
        except Exception:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    tx1, ty1, tx2, ty2, dx_arcsec, dy_arcsec = _world_xy_arcsec(world1, world2)

    bmaj1 = float(prepared['beam']['major']['value'])
    bmin1 = float(prepared['beam']['minor']['value'])
    bmaj2 = float(prepared['beam2']['major']['value'])
    bmin2 = float(prepared['beam2']['minor']['value'])
    beam_ref_arcsec = np.sqrt(max(bmaj1, 0.0) * max(bmin1, 0.0))
    offset_r_arcsec = float(np.hypot(dx_arcsec, dy_arcsec))

    return {
        'freq': image_meta.get('freq', prepared.get('freqstr_plot', '')),
        'image_path': str(image1_filename),
        'model_path': str(image2_filename),
        'image_peak_x_pix': float(peak1_yx[1]),
        'image_peak_y_pix': float(peak1_yx[0]),
        'model_peak_x_pix': float(peak2_yx[1]),
        'model_peak_y_pix': float(peak2_yx[0]),
        'image_peak_tx_arcsec': float(tx1),
        'image_peak_ty_arcsec': float(ty1),
        'model_peak_tx_arcsec': float(tx2),
        'model_peak_ty_arcsec': float(ty2),
        'peak_dx_arcsec': float(dx_arcsec),
        'peak_dy_arcsec': float(dy_arcsec),
        'peak_r_arcsec': offset_r_arcsec,
        'beam1_bmaj_arcsec': bmaj1,
        'beam1_bmin_arcsec': bmin1,
        'beam2_bmaj_arcsec': bmaj2,
        'beam2_bmin_arcsec': bmin2,
        'offset_over_geom_beam': float(offset_r_arcsec / beam_ref_arcsec) if beam_ref_arcsec > 0 else np.nan,
        'sub1_crval1': float(prepared['sub1'].meta.get('crval1', np.nan)),
        'sub1_crval2': float(prepared['sub1'].meta.get('crval2', np.nan)),
        'sub2_crval1': float(prepared['sub2'].meta.get('crval1', np.nan)),
        'sub2_crval2': float(prepared['sub2'].meta.get('crval2', np.nan)),
    }


def measure_casa_image_model_contour_offset_sunpy(image1_filename, image2_filename,
                                                  crop_fraction=(0.0, 1.0), reftime=None,
                                                  image_meta=None,
                                                  conv_tag='',
                                                  overwrite_conv=True,
                                                  convolve_model=True,
                                                  overwrite_fits=True,
                                                  coord_frame='radec',
                                                  match_fov=True,
                                                  align_model_time_to_image=True,
                                                  north_up=False,
                                                  apply_hpc_p_angle=False,
                                                  anchor_model_wcs_to_image=False,
                                                  use_reftime_for_hpc=True,
                                                  hpc_reference_xy=None,
                                                  force_hpc_origin_at_image_center=False,
                                                  hpc_xy_shift=(0.0, 0.0),
                                                  contour_percent=70.0,
                                                  focus_box=None):
    """Measure offset from contour-mask centroids on the prepared comparison grid.

    This is intended to match what is visually apparent in an overlaid contour
    plot more closely than a pure peak or full-image correlation metric.

    :param image1_filename: Reference CASA image path.
    :type image1_filename: str
    :param image2_filename: Model CASA image path.
    :type image2_filename: str
    :param contour_percent: Mask threshold as a percent of each image peak.
    :type contour_percent: float
    :param focus_box: Optional fractional ROI ``((x0, x1), (y0, y1))`` on the
        prepared comparison grid.
    :type focus_box: tuple or None
    :returns: Contour-centroid offset diagnostics.
    :rtype: dict
    """
    image_meta = {} if image_meta is None else dict(image_meta)
    prepared = plot_two_casa_images_with_convolution_sunpy(
        image1_filename,
        image2_filename,
        crop_fraction=crop_fraction,
        reftime=reftime,
        image_meta=image_meta,
        conv_tag=conv_tag,
        overwrite_conv=overwrite_conv,
        convolve_model=convolve_model,
        overwrite_fits=overwrite_fits,
        coord_frame=coord_frame,
        match_fov=match_fov,
        align_model_time_to_image=align_model_time_to_image,
        north_up=north_up,
        apply_hpc_p_angle=apply_hpc_p_angle,
        anchor_model_wcs_to_image=anchor_model_wcs_to_image,
        use_reftime_for_hpc=use_reftime_for_hpc,
        hpc_reference_xy=hpc_reference_xy,
        force_hpc_origin_at_image_center=force_hpc_origin_at_image_center,
        hpc_xy_shift=hpc_xy_shift,
        return_prepared=True,
    )

    data1 = np.asarray(prepared['data1'], dtype=float)
    data2 = np.asarray(prepared['data2'], dtype=float)

    def _focus_slices(shape, roi):
        ny, nx = shape
        if roi is None:
            return slice(0, ny), slice(0, nx)
        (x0f, x1f), (y0f, y1f) = roi
        x0 = max(0, min(nx - 1, int(np.floor(nx * float(x0f)))))
        x1 = max(x0 + 1, min(nx, int(np.ceil(nx * float(x1f)))))
        y0 = max(0, min(ny - 1, int(np.floor(ny * float(y0f)))))
        y1 = max(y0 + 1, min(ny, int(np.ceil(ny * float(y1f)))))
        return slice(y0, y1), slice(x0, x1)

    yslice, xslice = _focus_slices(data1.shape, focus_box)
    arr1 = data1[yslice, xslice]
    arr2 = data2[yslice, xslice]

    def _mask_centroid(arr, level_pct):
        arr = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        peak = np.nanmax(arr)
        if not np.isfinite(peak) or peak <= 0:
            return np.array([np.nan, np.nan], dtype=float), np.zeros_like(arr, dtype=bool)
        cutoff = peak * float(level_pct) / 100.0
        mask = arr >= cutoff
        if not np.any(mask):
            return np.array([np.nan, np.nan], dtype=float), mask
        yy, xx = np.indices(arr.shape, dtype=float)
        weights = np.where(mask, arr, 0.0)
        total = np.nansum(weights)
        cy = np.nansum(yy * weights) / total
        cx = np.nansum(xx * weights) / total
        return np.array([cy, cx], dtype=float), mask

    ctr1_yx, mask1 = _mask_centroid(arr1, contour_percent)
    ctr2_yx, mask2 = _mask_centroid(arr2, contour_percent)
    shift_yx_pix = ctr1_yx - ctr2_yx

    try:
        pixscale_x_arcsec = abs(prepared['sub1'].scale.axis1.to_value(u.arcsec / u.pix))
    except Exception:
        pixscale_x_arcsec = 1.0
    try:
        pixscale_y_arcsec = abs(prepared['sub1'].scale.axis2.to_value(u.arcsec / u.pix))
    except Exception:
        pixscale_y_arcsec = pixscale_x_arcsec

    return {
        'freq': image_meta.get('freq', prepared.get('freqstr_plot', '')),
        'image_path': str(image1_filename),
        'model_path': str(image2_filename),
        'contour_percent': float(contour_percent),
        'focus_box': focus_box,
        'contour_shift_x_pix': float(shift_yx_pix[1]),
        'contour_shift_y_pix': float(shift_yx_pix[0]),
        'contour_shift_x_arcsec': float(shift_yx_pix[1] * pixscale_x_arcsec),
        'contour_shift_y_arcsec': float(shift_yx_pix[0] * pixscale_y_arcsec),
        'contour_shift_r_arcsec': float(np.hypot(shift_yx_pix[1] * pixscale_x_arcsec,
                                                 shift_yx_pix[0] * pixscale_y_arcsec)),
        'image_mask_npix': int(np.count_nonzero(mask1)),
        'model_mask_npix': int(np.count_nonzero(mask2)),
    }

@runtime_report
def plot_two_casa_images0(image1_filename, image2_filename,
                          crop_fraction=(0.0, 1.0),
                          figsize=(15, 4),
                          title1='First Image',
                          title2='Second Image',
                          compare_two=False,
                          contour_levels=None, cmap='viridis',
                          vmax=99.9, vmin=0,
                          uni_vmaxmin=False,
                          norm='linear',
                          image_model_filename=None):
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

    if image_model_filename is not None:
        ia.open(image_model_filename)
        pix_model = ia.getchunk()[:, :, 0, 0]
        ia.close()
        figsize = (figsize[0], figsize[1] * 2)

    # --- Crop both images using the same crop_fraction ---
    shape = pix1.shape[0]  # assume square images.
    if isinstance(crop_fraction[0], float):
        p1 = int(shape * crop_fraction[0])
        p2 = int(shape * crop_fraction[1])
        cropped1 = pix1[p1:p2, p1:p2]
        cropped2 = pix2[p1:p2, p1:p2]
        if image_model_filename is not None:
            cropped_model = pix_model[p1:p2, p1:p2]
    elif isinstance(crop_fraction[0], tuple) or isinstance(crop_fraction[0], list):
        px1 = int(shape * crop_fraction[0][0])
        px2 = int(shape * crop_fraction[0][1])
        py1 = int(shape * crop_fraction[1][0])
        py2 = int(shape * crop_fraction[1][1])
        cropped1 = pix1[px1:px2, py1:py2]
        cropped2 = pix2[px1:px2, py1:py2]
        if image_model_filename is not None:
            cropped_model = pix_model[px1:px2, py1:py2]
    else:
        cropped1 = pix1
        cropped2 = pix2
        if image_model_filename is not None:
            cropped_model = pix_model

    # --- Plotting: Create a figure with two panels using the WCS projection from image1 ---
    if compare_two:
        fig, axs = plt.subplots(1, 3, figsize=figsize,
                                subplot_kw={'projection': w})
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize,
                                subplot_kw={'projection': w})

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
                     vmax=vmax_val1, vmin=vmin_val1, norm=norm)
    ax1.set_xlabel('Right Ascension')
    ax1.set_ylabel('Declination')
    ax1.set_title(title1)
    plt.colorbar(im1, ax=ax1)

    # Right panel: Convolved Image2 as background.
    ax2 = axs[-1]
    im2 = ax2.imshow(cropped2.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                     vmax=vmax_val2, vmin=vmin_val2, norm=norm)
    ax2.set_xlabel('Right Ascension')
    ax2.set_ylabel('Declination')
    ax2.set_title(title2)
    plt.colorbar(im2, ax=ax2)

    # Overlay contours from image1 onto the right panel.
    if compare_two:
        ax_comp = axs[1]
        im2 = ax_comp.imshow(cropped2.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                             vmax=vmax_val2, vmin=vmin_val2, norm=norm)
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


@runtime_report
def plot_two_casa_images(image1_filename, image2_filename,
                         crop_fraction=(0.0, 1.0),
                         figsize=(15, 4),
                         image_meta={'freq': '', 'title': ['', ''], 'array_config': ['', '']},
                         contour_levels=None, cmap='viridis',
                         cmap_model='turbo',
                         vmax=100.0, vmin=0.0,
                         vmax_percentile=99.9, vmin_percentile=0,
                         uni_vmaxmin=False,
                         image1_model_filename=None,
                         image2_model_filename=None):
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
    titiles = image_meta.get('title', ["", ""])
    title1 = titiles[0]
    title2 = titiles[1]
    freqstr = image_meta.get('freq', '')
    array_configs = image_meta.get('array_config', '')
    array_config1 = array_configs[0]
    array_config2 = array_configs[1]
    noise = image_meta.get('noise', None)
    cal_error = image_meta.get('cal_error', None)
    dur = image_meta.get('duration', None)

    ia = IA()
    # --- Open the first image and extract data, coordinate system, and restoring beam ---
    ia.open(image1_filename)
    pix1 = ia.getchunk()[:, :, 0, 0]  # assume image shape [nx, ny, 1, 1]
    csys1 = ia.coordsys()
    beam1 = ia.restoringbeam()
    ia.close()

    # Build an Astropy WCS object using the CASA coordinate system from image1.
    rad_to_deg = 180.0 / np.pi
    w = WCS(naxis=2)
    w.wcs.crpix = csys1.referencepixel()['numeric'][0:2]
    w.wcs.cdelt = np.array(csys1.increment()['numeric'][0:2]) * rad_to_deg
    w.wcs.crval = np.array(csys1.referencevalue()['numeric'][0:2]) * rad_to_deg
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']

    # --- add restoring‐beam ellipse in pixel units ---
    # convert beam major/minor (arcsec) → pixels using WCS cdelt (deg→arcsec)
    pixscale_x = abs(w.wcs.cdelt[0]) * 3600.0
    pixscale_y = abs(w.wcs.cdelt[1]) * 3600.0

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
    beam2 = ia.restoringbeam()
    ia.close()

    if image1_model_filename is not None:
        ia.open(image1_model_filename)
        pix_model1 = ia.getchunk()[:, :, 0, 0]
        ia.close()
        ia.open(image2_model_filename)
        pix_model2 = ia.getchunk()[:, :, 0, 0]
        ia.close()
        figsize = (figsize[0], figsize[1] * 2)

    # --- Crop both images using the same crop_fraction ---
    shape = pix1.shape[0]  # assume square images.

    crop_fraction_rms = (0.9, 1.0)
    p1_rms = int(shape * crop_fraction_rms[0])
    p2_rms = int(shape * crop_fraction_rms[1])
    datamax1 = np.nanmax(pix1)
    datamax2 = np.nanmax(pix2)

    cropped1_rms = pix1[p1_rms:p2_rms, p1_rms:p2_rms]
    rms1 = np.sqrt(np.nanmean(cropped1_rms ** 2))
    snr1 = datamax1 / rms1
    cropped2_rms = pix2[p1_rms:p2_rms, p1_rms:p2_rms]
    rms2 = np.sqrt(np.nanmean(cropped2_rms ** 2))
    snr2 = datamax2 / rms2

    if isinstance(crop_fraction[0], float):
        p1 = int(shape * crop_fraction[0])
        p2 = int(shape * crop_fraction[1])
        cropped1 = pix1[p1:p2, p1:p2]
        cropped2 = pix2[p1:p2, p1:p2]
        if image1_model_filename is not None:
            cropped1_model = pix_model1[p1:p2, p1:p2]
            cropped2_model = pix_model2[p1:p2, p1:p2]
    elif isinstance(crop_fraction[0], tuple) or isinstance(crop_fraction[0], list):
        px1 = int(shape * crop_fraction[0][0])
        px2 = int(shape * crop_fraction[0][1])
        py1 = int(shape * crop_fraction[1][0])
        py2 = int(shape * crop_fraction[1][1])
        cropped1 = pix1[px1:px2, py1:py2]
        cropped2 = pix2[px1:px2, py1:py2]
        if image1_model_filename is not None:
            cropped1_model = pix_model1[px1:px2, py1:py2]
            cropped2_model = pix_model2[px1:px2, py1:py2]
    else:
        cropped1 = pix1
        cropped2 = pix2
        if image1_model_filename is not None:
            cropped1_model = pix_model1
            cropped2_model = pix_model2
    datamin1 = np.nanmin(cropped1)
    datamin2 = np.nanmin(cropped2)

    if image1_model_filename is not None:
        diff1 = np.abs(cropped1_model - cropped1)
        diff2 = np.abs(cropped2_model - cropped2)
        diff1[diff1 < rms1 * 0.75] = rms1 * 0.75
        diff2[diff2 < rms2 * 0.75] = rms2 * 0.75
        img_fidelity1 = cropped1 / diff1
        img_fidelity2 = cropped2 / diff2
    else:
        img_fidelity1 = cropped1
        img_fidelity2 = cropped2

    # --- Plotting: Create a figure with two panels using the WCS projection from image1 ---
    fig, axs = plt.subplots(2, 2, figsize=figsize,
                            subplot_kw={'projection': w})

    vmax_val1 = np.nanmax(cropped1) * float(vmax) / 100
    vmin_val1 = np.nanmax(cropped1) * float(vmin) / 100
    if uni_vmaxmin:
        vmax_val2 = vmax_val1
        vmin_val2 = vmin_val1
    else:
        vmax_val2 = np.nanmax(cropped2) * float(vmax) / 100
        vmin_val2 = np.nanmax(cropped2) * float(vmin) / 100

    vmax_val1_model = np.nanpercentile(img_fidelity1, vmax_percentile)
    vmin_val1_model = np.nanpercentile(img_fidelity1, vmin_percentile)
    vmax_val2_model = np.nanpercentile(img_fidelity1, vmax_percentile)
    vmin_val2_model = np.nanpercentile(img_fidelity1, vmin_percentile)

    if uni_vmaxmin:
        vmax_val2_model = vmax_val1_model if vmax_val1_model > vmax_val2_model else vmax_val2_model
        vmin_val2_model = vmin_val1_model if vmin_val1_model < vmin_val2_model else vmin_val2_model
    else:
        vmax_val2_model = np.nanpercentile(img_fidelity2, vmax_percentile)
        vmin_val2_model = np.nanpercentile(img_fidelity2, vmin_percentile)

    # Left panel: Image1 (original)
    ax1 = axs[0, 0]
    im1 = ax1.imshow(cropped1.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                     vmax=vmax_val1, vmin=vmin_val1)
    ax1.set_xlabel('Right Ascension')
    ax1.set_ylabel('Declination')
    ax1.set_title(title1)
    ax1.text(0.98, 0.02, r'T$_{Bmin}$' + f': {datamin1:.1e} K', transform=ax1.transAxes, ha='right',
             va='bottom', color='white', )
    ax1.text(0.98, 0.08, r'T$_{Bmax}$' + f': {datamax1:.1e} K', transform=ax1.transAxes, ha='right',
             va='bottom', color='white', )
    ax1.text(0.98, 0.14, f'SNR: {snr1:.1f}', transform=ax1.transAxes, ha='right',
             va='bottom', color='white', )
    ax1.text(0.02, 0.02, freqstr, transform=ax1.transAxes, ha='left',
             va='bottom', color='white')
    ax1.text(0.02, 0.98, f'Array cfg: {array_config1}', transform=ax1.transAxes, ha='left',
             va='top', color='white', fontweight='bold')
    ax1.text(0.02, 0.92, f'Dur: {dur}', transform=ax1.transAxes, ha='left',
             va='top', color='white', )
    ax1.text(0.02, 0.86, f'Themal noise: {noise}', transform=ax1.transAxes, ha='left',
             va='top', color='white', )
    ax1.text(0.02, 0.80, f'Cal err: {cal_error}', transform=ax1.transAxes, ha='left',
             va='top', color='white', )
    plt.colorbar(im1, ax=ax1, label=r'T$_B$ [K]')

    # place ellipse at 10% in from lower-left corner of the panel
    shape_cropped = cropped1.shape
    x0 = shape_cropped[1] * 0.10
    y0 = shape_cropped[0] * 0.10

    major1_pix = beam1['major']['value'] / pixscale_x
    minor1_pix = beam1['minor']['value'] / pixscale_y
    pa1_deg = beam1['positionangle']['value']

    ell = Ellipse((x0, y0),
                  width=major1_pix, height=minor1_pix,
                  angle=pa1_deg,
                  edgecolor='white', facecolor='none', lw=1.5)
    ax1.add_patch(ell)

    # Right panel: Convolved Image2 as background.
    ax2 = axs[0, 1]
    im2 = ax2.imshow(cropped2.transpose(), origin='lower', cmap=plt.get_cmap(cmap),
                     vmax=vmax_val2, vmin=vmin_val2)
    ax2.set_xlabel('Right Ascension')
    ax2.set_ylabel('Declination')
    ax2.set_title(title2)
    ax2.text(0.98, 0.02, r'T$_{Bmin}$' + f': {datamin2:.1e} K', transform=ax2.transAxes, ha='right',
             va='bottom', color='white', )
    ax2.text(0.98, 0.08, r'T$_{Bmax}$' + f': {datamax2:.1e} K', transform=ax2.transAxes, ha='right',
             va='bottom', color='white', )
    ax2.text(0.98, 0.14, f'SNR: {snr2:.1f}', transform=ax2.transAxes, ha='right',
             va='bottom', color='white', )
    ax2.text(0.02, 0.02, freqstr, transform=ax2.transAxes, ha='left',
             va='bottom', color='white')
    ax2.text(0.02, 0.98, f'Array cfg: {array_config2}', transform=ax2.transAxes, ha='left',
             va='top', color='white', fontweight='bold')
    ax2.text(0.02, 0.92, f'Dur: {dur}', transform=ax2.transAxes, ha='left',
             va='top', color='white', )
    ax2.text(0.02, 0.86, f'Themal noise: {noise}', transform=ax2.transAxes, ha='left',
             va='top', color='white', )
    ax2.text(0.02, 0.80, f'Cal err: {cal_error}', transform=ax2.transAxes, ha='left',
             va='top', color='white', )

    plt.colorbar(im2, ax=ax2, label=r'T$_B$ [K]')

    major2_pix = beam2['major']['value'] / pixscale_x
    minor2_pix = beam2['minor']['value'] / pixscale_y
    pa2_deg = beam2['positionangle']['value']

    ell = Ellipse((x0, y0),
                  width=major2_pix, height=minor2_pix,
                  angle=pa2_deg,
                  edgecolor='white', facecolor='none', lw=1.5)
    ax2.add_patch(ell)

    # Left panel: Image1 (original)
    ax3 = axs[1, 0]
    im3 = ax3.imshow(img_fidelity1.transpose(), origin='lower', cmap=plt.get_cmap(cmap_model),
                     vmax=vmax_val1_model, vmin=vmin_val1_model)
    ax3.set_xlabel('Right Ascension')
    ax3.set_ylabel('Declination')
    ax3.set_title('Fidelity image')
    plt.colorbar(im3, ax=ax3, label=r'I/|I-T|')

    # Right panel: Convolved Image2 as background.
    ax4 = axs[1, 1]
    im4 = ax4.imshow(img_fidelity2.transpose(), origin='lower', cmap=plt.get_cmap(cmap_model),
                     vmax=vmax_val2_model, vmin=vmin_val2_model)
    ax4.set_xlabel('Right Ascension')
    ax4.set_ylabel('Declination')
    ax4.set_title('Fidelity image')
    plt.colorbar(im4, ax=ax4, label=r'I/|I-T|')

    plt.tight_layout()
    stats = {'snr': [snr1, snr2]}

    return fig, axs, stats
