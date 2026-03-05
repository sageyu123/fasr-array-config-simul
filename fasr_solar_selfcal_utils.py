import os
import time
from datetime import datetime
from functools import wraps

import numpy as np


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
            f"[{now}] Function '{func.__name__}' completed in {duration:.2f} seconds."
        )
        return result

    return wrapper


def plot_caltb_stats(caltb_files, caltype=None, yrange=None, refant=0):
    """
    Plot statistics of the calibration tables.
    Parameters:
        caltb_files : list of str or str
            List of calibration table files to plot.
    """
    from casatools import table
    import matplotlib.pyplot as plt

    if isinstance(caltb_files, str):
        caltb_files = [caltb_files]

    for caltb in caltb_files:
        tb = table()
        print("Opening calibration table:", caltb)
        tb.open(caltb)
        data = tb.getcol("CPARAM")
        print(f"Shape of the data in {caltb}: {data.shape}")
        tb.close()

        # Check how many polarizations are in the data
        n_pol = data.shape[0]
        print(f"Number of polarizations in {caltb}: {n_pol}")

        # Plot antenna phase/amplitude/multi-band delay vs antenna index for each polarization
        # first, create two subplots: one for antenna-based values, one for histogram
        fig = plt.figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)

        if caltype is None:
            if "ph" in caltb:
                caltype = "ph"
            elif "amp" in caltb:
                caltype = "amp"
            elif "mbd" in caltb:
                caltype = "mbd"
            else:
                raise ValueError("Unknown calibration table type in filename.")
        elif caltype == "ph":
            # references all phases to the first antenna
            phases = np.angle(data[:, 0, :].squeeze())
            phases -= phases[0, refant]
            # ensure phases are between -pi and pi
            phases = (phases + np.pi) % (2 * np.pi) - np.pi
            if n_pol == 1:
                axes[0].plot(
                    phases,
                    marker="o",
                    linestyle="None",
                    color="r",
                    label="Pol 0",
                )
                axes[1].hist(np.angle(data[0, 0, :]), bins=30, alpha=0.3, color="r")
            else:
                axes[0].plot(
                    phases[0, :],
                    marker="o",
                    linestyle="None",
                    color="r",
                    label="Pol 0",
                )
                axes[0].plot(
                    phases[1, :],
                    marker="o",
                    linestyle="None",
                    color="g",
                    label="Pol 1",
                )
                axes[1].hist(np.angle(data[0, 0, :]), bins=30, alpha=0.3, color="r")
                axes[1].hist(np.angle(data[1, 0, :]), bins=30, alpha=0.3, color="b")

            axes[0].legend()
            axes[0].set_ylim(yrange) if yrange is not None else None
            axes[0].set_xlabel("Antenna Index")
            axes[0].set_ylabel("Phase Error (radians)")
            axes[0].set_title(f"Phase Calibration Table: {caltb}")

            axes[1].set_xlim(yrange) if yrange is not None else None
            axes[1].set_title("Phase Error Distribution")
            axes[1].set_xlabel("Phase Error (radians)")
            axes[1].set_ylabel("Number of Antennas")
            axes[1].grid()
        elif caltype == "amp":
            if n_pol == 1:
                axes[0].plot(
                    np.abs(data[0, 0, :]),
                    marker="o",
                    linestyle="None",
                    color="r",
                    label="Pol 0",
                )
                axes[1].hist(np.abs(data[0, 0, :]), bins=30, alpha=0.3, color="r")
            elif n_pol == 2:
                axes[0].plot(
                    np.abs(data[0, 0, :]),
                    marker="o",
                    linestyle="None",
                    color="r",
                    label="Pol 0",
                )
                axes[0].plot(
                    np.abs(data[1, 0, :]),
                    marker="o",
                    linestyle="None",
                    color="g",
                    label="Pol 1",
                )
                axes[1].hist(np.abs(data[0, 0, :]), bins=30, alpha=0.3, color="r")
                axes[1].hist(np.abs(data[1, 0, :]), bins=30, alpha=0.3, color="b")
            else:
                raise ValueError(
                    f"Unknown number of polarizations in {caltb}: {n_pol}"
                )
            axes[0].legend()
            axes[0].set_ylim(yrange) if yrange is not None else None
            axes[0].set_xlabel("Antenna Index")
            axes[0].set_ylabel("Amplitude Gain")
            axes[0].set_title(f"Amplitude Calibration Table: {caltb}")

            axes[1].set_xlim(yrange) if yrange is not None else None
            axes[1].set_title("Amplitude Gain Distribution")
            axes[1].set_xlabel("Amplitude Gain")
            axes[1].set_ylabel("Number of Antennas")
            axes[1].grid()
        elif caltype == "mbd":
            if n_pol == 1:
                axes[0].plot(
                    np.abs(data[0, 0, :]),
                    marker="o",
                    linestyle="None",
                    color="r",
                    label="Pol 0",
                )
                axes[1].hist(np.abs(data[0, 0, :]), bins=30, alpha=0.3, color="r")
            elif n_pol == 2:
                axes[0].plot(
                    np.abs(data[0, 0, :]),
                    marker="o",
                    linestyle="None",
                    color="r",
                    label="Pol 0",
                )
                axes[0].plot(
                    np.abs(data[1, 0, :]),
                    marker="o",
                    linestyle="None",
                    color="g",
                    label="Pol 1",
                )
                axes[1].hist(np.abs(data[0, 0, :]), bins=30, alpha=0.3, color="r")
                axes[1].hist(np.abs(data[1, 0, :]), bins=30, alpha=0.3, color="b")
            else:
                raise ValueError(
                    f"Unknown number of polarizations in {caltb}: {n_pol}"
                )
            axes[0].set_ylim(yrange) if yrange is not None else None
            axes[0].legend()
            axes[0].set_xlabel("Antenna Index")
            axes[0].set_ylabel("Multi-Band Delay (ns)")
            axes[0].set_title(f"Multi-Band Delay Calibration Table: {caltb}")
            axes[1].set_xlim(yrange) if yrange is not None else None
            axes[1].set_title("Multi-Band Delay Distribution")
            axes[1].set_xlabel("Multi-Band Delay (ns)")
        else:
            raise ValueError("Unknown calibration table type.")

        plt.ylabel("Number of Antennas")
        plt.grid()
        plt.show()


def compare_two_gaintables(
    caltb,
    caltb_ref,
    caltype=None,
    yrange=None,
    refant=None,
    invert=False,
    do_plot=True,
):
    """
    Compare two calibration tables by plotting their differences.

    When comparing self-cal solutions to the "ground truth" table used to corrupt
    the data: CASA applycal uses corrected = data / gain, and gaincal solves for
    G such that data / G = model, so the solved gain is the *inverse* of the
    corrupting gain. Use invert_second=True and refant set to the gaincal refant
    so that the comparison is like-for-like (difference should be ~0 if recovered).

    Parameters:
        caltb : str
            Path to the first calibration table file (e.g. ground truth corrupting table).
        caltb_ref : str
            Path to the second calibration table file (e.g. gaincal output).
        caltype : str, optional
            'ph', 'amp', or 'mbd'. Inferred from caltb1 filename if None.
        yrange : list, optional
            [ymin, ymax] for the difference plot and histogram.
        refant : int or str, optional
            Reference antenna index (0-based). If set, the first table is made
            relative to this antenna so comparison matches gaincal's refant convention.
        invert_second : bool, optional
            If True, treat caltb2 as the inverse of the true gains (e.g. from gaincal
            when the data were corrupted by applying caltb1). For phase: compare
            phase(caltb1) - phase(caltb1)[refant] to -phase(caltb2). For amp: compare
            so that ground_truth * solved ≈ 1. Use with refant when comparing
            self-cal to corrupting table.
    """
    from casatools import table
    import matplotlib.pyplot as plt

    tb1 = table()
    tb1.open(caltb)
    data1 = tb1.getcol("CPARAM")
    flag1 = tb1.getcol("FLAG")
    tb1.close()

    if caltb_ref is not None:
        tb2 = table()
        tb2.open(caltb_ref)
        data_ref = tb2.getcol("CPARAM")
        tb2.close()
    else:
        data_ref = np.zeros_like(data1)

    if caltype is None:
        if "ph" in caltb:
            caltype = "ph"
        elif "amp" in caltb:
            caltype = "amp"
        elif "mbd" in caltb:
            caltype = "mbd"
        else:
            raise ValueError("Unknown calibration table type in filename.")

    refant_ix = int(refant) if refant is not None else None

    if caltype == "ph":
        val1 = np.angle(data1)
        val_ref = np.angle(data_ref)
        if refant_ix is not None:
            val1 = val1 - val1[:, :, refant_ix : refant_ix + 1]  # relative to refant
            val_ref = val_ref - val_ref[:, :, refant_ix : refant_ix + 1]  # relative to refant
        if invert:
            val1 = -val1  # gaincal solution is inverse of corrupting gain
        diff = val1 - val_ref
        # Ensure the phase difference falls within [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        ylabel = "Phase Difference (radians)"
    elif caltype == "amp":
        val1 = np.abs(data1)
        val_ref = np.abs(data_ref)
        if invert:
            # ground_truth * solved should be 1; show ratio (should be ~1)
            diff = val1 * val_ref
            ylabel = "Amplitude ratio (ground truth×solved, ≈1)"
        else:
            diff = val1 / val_ref
            ylabel = "Amplitude Difference (ratio)"
    elif caltype == "mbd":
        val1 = np.abs(data1)
        val_ref = np.abs(data_ref)
        if invert:
            diff = val1 + val_ref  # delay: inverse in phase, so difference of delays
        else:
            diff = val1 - val_ref
        ylabel = "Multi-Band Delay Difference (ns)"
    else:
        raise ValueError("Unknown calibration table type.")

    if do_plot:
        fig = plt.figure(figsize=(12, 5))
        axes = fig.subplots(1, 2)

        diff[flag1 == 1] = np.nan

        axes[0].plot(
            diff[0, 0, :],
            marker="o",
            linestyle="None",
            color="r",
            label="Pol 0",
        )
        axes[0].plot(
            diff[1, 0, :],
            marker="o",
            linestyle="None",
            color="g",
            label="Pol 1",
        )
        if yrange is not None:
            axes[0].set_ylim(yrange)
        axes[0].set_xlabel("Antenna Index")
        axes[0].set_ylabel(ylabel)
        title = f"Comparison of {caltype} Calibration Tables"
        if invert and refant_ix is not None:
            title += f" (self-cal vs ground truth, refant={refant_ix})"
        elif invert:
            title += " (invert_second=True)"
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid()

        axes[1].hist(
            diff[0, 0, :][flag1[0, 0, :] == 0],
            bins=30,
            alpha=0.3,
            color="r",
            label="Pol 0",
        )
        axes[1].hist(
            diff[1, 0, :][flag1[1, 0, :] == 0],
            bins=30,
            alpha=0.3,
            color="g",
            label="Pol 1",
        )
        if yrange is not None:
            axes[1].set_xlim(yrange)
        axes[1].set_xlabel(ylabel)
        axes[1].set_ylabel("Number of Antennas")
        axes[1].set_title(f"{caltype} Difference Distribution")
        axes[1].legend()
        axes[1].grid()
        plt.show()
    return diff


def combine_gaintables(caltable_list, output_caltable, overwrite=False):
    """
    Combine multiple CASA calibration tables by element-wise multiplication of CPARAM.

    Useful for self-calibration when each round produces an incremental table
    (e.g. round0.p, round1.ap, ...) and the total correction is the product
    of all gains.

    Parameters
    ----------
    caltable_list : list of str
        Paths to calibration tables, in application order (first applied first).
    output_caltable : str
        Path for the combined output table. Structure is copied from the last table.
    overwrite : bool
        If True, remove output_caltable if it exists before writing.

    Returns
    -------
    str
        Path to the combined calibration table (same as output_caltable).

    Raises
    ------
    ValueError
        If caltable_list is empty.
    """
    from casatools import table
    import shutil

    if not caltable_list:
        raise ValueError("caltable_list must not be empty")

    if overwrite and os.path.exists(output_caltable):
        os.system(f"rm -rf {output_caltable}")

    if os.path.exists(output_caltable):
        return output_caltable

    tb = table()
    combined_data = None
    combined_flag = None
    for ct in caltable_list:
        tb.open(ct, nomodify=True)
        data_j = tb.getcol("CPARAM")
        flag_j = tb.getcol("FLAG")
        tb.close()
        if combined_data is None:
            combined_data = data_j.copy()
            combined_flag = flag_j.copy()
        else:
            combined_data *= data_j
            combined_flag |= flag_j
    shutil.copytree(caltable_list[-1], output_caltable)
    tb.open(output_caltable, nomodify=False)
    tb.putcol("CPARAM", combined_data)
    tb.putcol("FLAG", combined_flag)
    tb.close()

    return output_caltable


def plot_antenna_phase_map(
    config_file,
    phase_caltb,
    ground_truth_caltb=None,
    refant=0,
    invert=False,
    pol=0,
    chan=0,
    cmap="Reds",
    vmin=0,
    vmax=3.14,
    title=None,
    cbar_label=None,
    do_plot_diff=True,
):
    """
    Plot antenna positions colored by self-calibration phase (or phase error).

    Parameters:
        config_file : str
            Path to the array config .cfg file (antenna positions).
        phase_caltb : str
            Path to the phase calibration table (e.g. self-cal output).
        ground_truth_caltb : str, optional
            If set, color by phase error (ground_truth - selfcal with refant/invert_second).
        refant : int
            Reference antenna index (0-based). Used when ground_truth_caltb is set.
        invert_second : bool
            If True and ground_truth_caltb is set, treat phase_caltb as inverse (gaincal) solution.
        pol : int
            Polarization index (0 or 1) for phase value.
        chan : int
            Channel index for phase value.
        cmap : str
            Matplotlib colormap name.
        title : str, optional
            Plot title.
        cbar_label : str, optional
            Colorbar label.
    """
    from casatools import table
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib

    # Avoid circular import at module load; import lazily
    from fasr_solar_simul import read_casa_antenna_list

    positions, _ = read_casa_antenna_list(config_file)
    n_ant = positions.shape[0]

    diff = compare_two_gaintables(
        phase_caltb,
        caltb_ref=ground_truth_caltb,
        caltype="ph",
        refant=refant,
        invert=invert,
        do_plot=do_plot_diff,
    )
    values = np.abs(diff[pol, chan, :])
    default_label = "Phase error (rad)"

    cmap_obj = plt.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(15, 7))

    # Full array scatter
    sc = ax_full.scatter(
        positions[:, 0],
        positions[:, 1],
        c=values,
        cmap=cmap_obj,
        s=80,
        edgecolors="k",
        linewidths=0.5,
        norm=norm,
    )
    for i in range(n_ant):
        ax_full.annotate(
            str(i), (positions[i, 0], positions[i, 1]), fontsize=7, ha="center", va="bottom"
        )
    ax_full.set_xlabel("East (m)")
    ax_full.set_ylabel("North (m)")
    ax_full.set_aspect("equal")
    ax_full.grid(True)
    divider = make_axes_locatable(ax_full)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sc, cax=cax, label="Phase Error (radians)" if title is None else None)

    # Panel title for full array
    if title is not None:
        ax_full.set_title(f"{title} (full array)")
    elif ground_truth_caltb is not None:
        ax_full.set_title(
            "Antenna positions colored by self-cal phase error (full array)"
        )
    else:
        ax_full.set_title("Antenna positions colored by self-cal phase (full array)")

    # Inner 100m zoom-in scatter (always included in second panel)
    radius = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    idx_zoom = radius <= 100

    sc2 = ax_zoom.scatter(
        positions[idx_zoom, 0],
        positions[idx_zoom, 1],
        c=values[idx_zoom],
        cmap=cmap_obj,
        s=80,
        edgecolors="k",
        linewidths=0.5,
        norm=norm,
    )
    for i in range(n_ant):
        if idx_zoom[i]:
            ax_zoom.annotate(
                str(i),
                (positions[i, 0], positions[i, 1]),
                fontsize=7,
                ha="center",
                va="bottom",
            )
    ax_zoom.set_xlabel("East (m)")
    ax_zoom.set_ylabel("North (m)")
    ax_zoom.set_aspect("equal")
    ax_zoom.grid(True)
    ax_zoom.set_xlim(-105, 105)
    ax_zoom.set_ylim(-105, 105)
    ax_zoom.set_title("Zoom-in: inner 100 m antennas")

    divider = make_axes_locatable(ax_zoom)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sc2, cax=cax2, label=cbar_label or default_label)

    plt.tight_layout()
    plt.show()


@runtime_report
def iterative_selfcal(
    ms_selfcal,
    model_image,
    config_file,
    uvrange_steps=None,
    calmodes=None,
    niter_steps=None,
    imsize=1024,
    cell="2.4arcsec",
    weighting="uniform",
    scales=None,
    deconvolver="multiscale",
    refant="0",
    minblperant=4,
    minsnr=3.0,
    caltbdir="./",
    imname_base=None,
    ground_truth_ph_caltb=None,
    ground_truth_amp_caltb=None,
    overwrite=False,
):
    """
    Iterative baseline-bootstrapping self-calibration.

    Progressively extends uvrange so that inner (well-constrained) antennas
    are solved first and act as anchors for subsequent rounds that include
    longer baselines / outer antennas.

    At each round:
      1. ft()  -- put the current model into the MODEL column of ms_selfcal
      2. gaincal() -- solve for gains with all prior tables applied on the fly
      3. applycal() -- apply all accumulated tables to a working copy
      4. tclean()  -- shallow deconvolution to update the model
      5. diagnostics -- print stats, optionally plot phase map vs ground truth
    """
    from casatasks import ft, gaincal, applycal, clearcal, tclean
    import shutil

    # ---- defaults ----
    if uvrange_steps is None:
        uvrange_steps = ["<10klambda", "<30klambda", "<80klambda", ""]
    if calmodes is None:
        calmodes = ["p", "ap", "ap", "ap"]
    if niter_steps is None:
        niter_steps = [200, 400, 800, 1500]
    n_rounds = len(uvrange_steps)
    assert len(calmodes) == n_rounds, "calmodes must have the same length as uvrange_steps"
    assert len(niter_steps) == n_rounds, "niter_steps must have the same length as uvrange_steps"

    if imname_base is None:
        imname_base = ms_selfcal.replace(".ms", "")

    all_gaintables = []
    current_model = model_image

    for i_round in range(n_rounds):
        uvr = uvrange_steps[i_round]
        calmode = calmodes[i_round]
        niter = niter_steps[i_round]
        uvr_label = uvr if uvr else "all"
        print("=" * 70)
        print(f"Self-cal round {i_round}: uvrange={uvr_label}, calmode={calmode}, niter={niter}")
        print("=" * 70)

        # ---- 1. Put current model into MODEL column ----
        # Do NOT call clearcal(ms_selfcal) here: CASA clearcal can reset MODEL_DATA to
        # unity, which would overwrite the model we put in with ft() and break gaincal
        # (phase differences would be wrong). DATA in ms_selfcal stays raw; we only fill MODEL.
        print(f"  [ft] Filling MODEL column of {ms_selfcal} with {current_model}")
        ft(vis=ms_selfcal, spw="0", model=current_model, usescratch=False)

        # ---- 2. gaincal with all prior tables applied on the fly ----
        caltable_i = os.path.join(caltbdir, f"selfcal_round{i_round}.{calmode}")
        if overwrite and os.path.exists(caltable_i):
            os.system(f"rm -rf {caltable_i}")
        if not os.path.exists(caltable_i):
            gaincal_kwargs = dict(
                vis=ms_selfcal,
                caltable=caltable_i,
                solint="inf",
                calmode=calmode,
                gaintype="G",
                refant=refant,
                minblperant=minblperant,
                minsnr=minsnr,
            )
            if uvr:
                gaincal_kwargs["uvrange"] = uvr
            if all_gaintables:
                gaincal_kwargs["gaintable"] = list(all_gaintables)
            print(f"  [gaincal] Solving round {i_round} -> {caltable_i}")
            if all_gaintables:
                print(f"            on-the-fly gaintable: {all_gaintables}")
            gaincal(**gaincal_kwargs)
        else:
            print(f"  [gaincal] {caltable_i} already exists, skipping.")

        all_gaintables.append(caltable_i)

        # ---- 3. Diagnostics on this round's table ----
        output_caltable = os.path.join(caltbdir, f"selfcal_combined_round{i_round}.ap")
        combined_caltb_round = combine_gaintables(
            all_gaintables, output_caltable=output_caltable, overwrite=overwrite
        )
        _print_caltable_diagnostics(
            combined_caltb_round,
            i_round,
            calmode,
            ground_truth_ph_caltb=ground_truth_ph_caltb,
            ground_truth_amp_caltb=ground_truth_amp_caltb,
            config_file=config_file,
            refant=int(refant) if refant else 0,
        )

        # ---- 4. Apply ALL accumulated tables to a working copy ----
        ms_working = ms_selfcal.replace(".ms", f"_iter{i_round}_applied.ms")
        if overwrite and os.path.exists(ms_working):
            os.system(f"rm -rf {ms_working}")
        if not os.path.exists(ms_working):
            print(f"  [applycal] Applying {len(all_gaintables)} table(s) -> {ms_working}")
            shutil.copytree(ms_selfcal, ms_working)
            clearcal(vis=ms_working)
            applycal(
                vis=ms_working,
                gaintable=list(all_gaintables),
                applymode="calonly",
                calwt=False,
            )

        # ---- 5. Shallow tclean to update the model ----
        imname_round = f"{imname_base}_selfcal_iter{i_round}"
        if overwrite:
            for ext in [
                ".image",
                ".model",
                ".residual",
                ".psf",
                ".flux",
                ".mask",
                ".pb",
                ".image.pbcor",
                ".sumwt",
            ]:
                p = f"{imname_round}{ext}"
                if os.path.exists(p):
                    os.system(f"rm -rf {p}")

        if not os.path.exists(f"{imname_round}.image"):
            tclean_kwargs = dict(
                vis=ms_working,
                imagename=imname_round,
                imsize=imsize,
                cell=cell,
                weighting=weighting,
                niter=niter,
                savemodel="modelcolumn",
                interactive=False,
                pblimit=0.01,
            )
            if deconvolver == "multiscale" and scales is not None:
                tclean_kwargs["deconvolver"] = "multiscale"
                tclean_kwargs["scales"] = scales
                tclean_kwargs["gain"] = 0.2
                tclean_kwargs["cycleniter"] = 200
                tclean_kwargs["minpsffraction"] = 0.1
            else:
                tclean_kwargs["deconvolver"] = "hogbom"
            print(f"  [tclean] Shallow clean (niter={niter}) -> {imname_round}")
            tclean(**tclean_kwargs)
            # clean up intermediate files (keep .image)
            for suffix in [
                ".model",
                ".residual",
                ".psf",
                ".flux",
                ".mask",
                ".pb",
                ".image.pbcor",
                ".sumwt",
            ]:
                p = f"{imname_round}{suffix}"
                if os.path.exists(p):
                    os.system(f"rm -rf {p}")
        else:
            print(f"  [tclean] {imname_round}.image already exists, skipping.")

        # Lazy import to avoid circularity at module import time
        from fasr_solar_simul import plot_casa_image

        plot_casa_image(imname_round + ".image", cmap="hinodexrt")
        current_model = f"{imname_round}.image"
        print(f"  Model updated to: {current_model}")
        print()

    # ---- Build combined calibration table (multiply all per-round gains) ----
    print("=" * 70)
    print("Building combined calibration table from all rounds")
    print("=" * 70)
    combined_caltb = os.path.join(caltbdir, "selfcal_combined.ap")
    combine_gaintables(all_gaintables, combined_caltb, overwrite=overwrite)
    print(f"  Combined calibration table: {combined_caltb}")

    # ---- Final diagnostic: compare combined table to ground truth ----
    if ground_truth_ph_caltb is not None:
        print("\n--- Final comparison: combined selfcal vs ground truth (phase) ---")
        compare_two_gaintables(
            ground_truth_ph_caltb,
            combined_caltb,
            caltype="ph",
            yrange=[-3.5, 3.5],
            refant=int(refant) if refant else 0,
            invert_second=True,
        )
        plot_antenna_phase_map(
            config_file,
            combined_caltb,
            ground_truth_caltb=ground_truth_ph_caltb,
            refant=int(refant) if refant else 0,
            invert_second=True,
            do_plot_diff=False,
        )
    if ground_truth_amp_caltb is not None:
        print("--- Final comparison: combined selfcal vs ground truth (amplitude) ---")
        compare_two_gaintables(
            ground_truth_amp_caltb,
            combined_caltb,
            caltype="amp",
            yrange=[0, 5],
            invert_second=True,
        )

    # ---- Final corrected MS ----
    ms_corrected = ms_selfcal.replace(".ms", "_selfcal_final.ms")
    if overwrite and os.path.exists(ms_corrected):
        os.system(f"rm -rf {ms_corrected}")
    if not os.path.exists(ms_corrected):
        print(
            f"\nApplying all {len(all_gaintables)} tables to produce final corrected MS: {ms_corrected}"
        )
        import shutil
        from casatasks import clearcal, applycal

        shutil.copytree(ms_selfcal, ms_corrected)
        clearcal(vis=ms_corrected)
        applycal(
            vis=ms_corrected,
            gaintable=list(all_gaintables),
            applymode="calonly",
            calwt=False,
        )

    print("\n*** Iterative self-calibration complete ***")
    print(f"  Rounds completed:      {n_rounds}")
    print(f"  Calibration tables:    {all_gaintables}")
    print(f"  Combined table:        {combined_caltb}")
    print(f"  Final model image:     {current_model}")
    print(f"  Final corrected MS:    {ms_corrected}")

    return all_gaintables, combined_caltb, current_model, ms_corrected


def _print_caltable_diagnostics(
    caltable,
    i_round,
    calmode,
    ground_truth_ph_caltb=None,
    ground_truth_amp_caltb=None,
    config_file=None,
    refant=0,
):
    """
    Print per-round diagnostics for a calibration table:
    number of antennas with valid solutions, RMS phase/amp error if ground truth available.
    Optionally plot antenna phase map.
    """
    from casatools import table

    tb = table()
    tb.open(caltable, nomodify=True)
    data = tb.getcol("CPARAM")  # (n_pol, n_chan, n_ant)
    flag = tb.getcol("FLAG")  # (n_pol, n_chan, n_ant)
    tb.close()

    n_ant = data.shape[2]
    # Valid solutions: not flagged in at least one pol
    valid_mask = ~flag[0, 0, :] | ~flag[1, 0, :]  # True if at least one pol unflagged
    n_valid = int(np.sum(valid_mask))
    n_flagged = n_ant - n_valid
    flagged_ants = np.where(~valid_mask)[0]

    print(f"  --- Round {i_round} diagnostics ({caltable}) ---")
    print(f"      Antennas with solutions: {n_valid}/{n_ant}")
    if n_flagged > 0:
        print(f"      Flagged antennas ({n_flagged}): {list(flagged_ants)}")

    if "p" in calmode:
        phase_pol0 = np.angle(data[0, 0, valid_mask])
        phase_pol1 = np.angle(data[1, 0, valid_mask])
        rms_ph0 = np.sqrt(np.mean(phase_pol0**2))
        rms_ph1 = np.sqrt(np.mean(phase_pol1**2))
        print(
            f"      RMS solved phase (Pol0/Pol1): "
            f"{np.degrees(rms_ph0):.2f} / {np.degrees(rms_ph1):.2f} deg"
        )

    if calmode in ("a", "ap"):
        amp_pol0 = np.abs(data[0, 0, valid_mask])
        amp_pol1 = np.abs(data[1, 0, valid_mask])
        print(
            f"      Mean solved amp  (Pol0/Pol1): "
            f"{np.mean(amp_pol0):.4f} / {np.mean(amp_pol1):.4f}"
        )

    # Compare to ground truth if available
    if ground_truth_ph_caltb is not None and config_file is not None:
        try:
            diff = compare_two_gaintables(
                ground_truth_ph_caltb,
                caltable,
                caltype="ph",
                refant=refant,
                invert_second=True,
                yrange=[-3.5, 3.5],
                do_plot=False,
            )
            rms_err0 = np.sqrt(np.nanmean(diff[0, 0, valid_mask] ** 2))
            rms_err1 = np.sqrt(np.nanmean(diff[1, 0, valid_mask] ** 2))
            print(
                f"      RMS phase error vs truth (Pol0/Pol1): "
                f"{np.degrees(rms_err0):.2f} / {np.degrees(rms_err1):.2f} deg"
            )
            plot_antenna_phase_map(
                config_file,
                caltable,
                ground_truth_caltb=ground_truth_ph_caltb,
                refant=refant,
                invert_second=True,
                do_plot_diff=False,
            )
        except Exception as e:
            print(f"      [warning] Could not compare to ground truth: {e}")


def generate_kolmogorov_delay_screen(screen_size_m=5000, pixel_scale_m=10, r0_m=200, scaling_factor=10.0):
    """
    Generates a 2D atmospheric delay screen based on Kolmogorov turbulence statistics.
    
    Parameters:
    - size_m: Physical size of the atmospheric delay screen side (meters)
    - pixel_scale_m: Resolution of the grid (meters/pixel)
    - r0_m: Fried parameter (coherence length). Smaller r0 = stronger turbulence.
            At 20 GHz, r0 might be ~100-300m. At 1 GHz (ionosphere), effectively larger.
    - scaling_factor: Scaling factor for the delay screen. Default is 10.0.
    """
    N = int(screen_size_m / pixel_scale_m)
    
    # 1. Coordinate Grid in Frequency Domain
    df = 1.0 / screen_size_m
    fx = np.fft.fftfreq(N, d=pixel_scale_m)
    fy = np.fft.fftfreq(N, d=pixel_scale_m)
    FX, FY = np.meshgrid(fx, fy)
    K = np.sqrt(FX**2 + FY**2)
    
    # Avoid division by zero at DC component (k=0)
    K[0,0] = 1e-10 
    
    # 2. Power Spectrum for Kolmogorov Turbulence
    # P(k) ~ k^(-11/3) -> Amplitude ~ k^(-11/6)
    # The factor 0.023 is a standard scaling constant for r0 definitions
    spectral_amplitude = 0.023 * (r0_m**(-5.0/3.0)) * (K**(-11.0/6.0))
    spectral_amplitude[0,0] = 0 # Remove DC offset (constant phase doesn't matter)
    
    # 3. Generate Random Gaussian Noise (Real + Imaginary)
    noise = np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))
    
    # 4. Filter Noise by the Turbulence Spectrum
    screen_freq = noise * np.sqrt(spectral_amplitude)
    
    # 5. Inverse FFT to get Spatial Domain Screen
    screen = np.fft.ifft2(screen_freq).real
    screen = screen * scaling_factor
    # Scale: The theoretical variance needs scaling by grid size factors from FFT
    # This is an approximation; usually we normalize to the desired variance manually
    # or rely on r0 definition. For this demo, the structure is key.
    
    return screen, pixel_scale_m


def sample_antennas_from_delay_screen(screen, pixel_scale_m, ant_x, ant_y):
    """
    Interpolates the screen delay values at specific antenna locations.
    """
    N = screen.shape[0]
    screen_size_m = N * pixel_scale_m
    # Convert physical coords (centered at 0,0) to array indices
    # Screen spans [-size/2, +size/2]
    idx_x = ((ant_x + screen_size_m/2) / pixel_scale_m).astype(int)
    idx_y = ((ant_y + screen_size_m/2) / pixel_scale_m).astype(int)
    
    # Clip to be safe
    idx_x = np.clip(idx_x, 0, N-1)
    idx_y = np.clip(idx_y, 0, N-1)
    
    return screen[idx_y, idx_x]
