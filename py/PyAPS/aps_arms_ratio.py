def calculate_arms_ratio_overlap_weighted(apsob, overlap_trim=50,
                                          min_snr=5, sigma_clip=3.0,
                                          plot_diagnostics=False):
    """
    Calculate arms ratio using WEIGHTED overlap region (LR mode only).

    Uses inverse variance weighting for robust flux comparison in the overlap.
    This is more accurate than simple median because it properly accounts for
    pixel-by-pixel uncertainties.

    Parameters
    ----------
    apsob : APSOB
        APSOB object with loaded data (must have 2 arms, not yet joined)
    overlap_trim : float, default=50
        Angstroms to trim from each edge of overlap (corrupted regions)
    min_snr : float, default=5
        Minimum SNR threshold for including targets
    sigma_clip : float, default=3.0
        Sigma clipping threshold for outlier rejection
    plot_diagnostics : bool, default=False
        Generate diagnostic plots

    Returns
    -------
    arms_ratio : list
        [1.0, ratio_for_red_arm]
    metadata : dict
        Detailed diagnostics

    Notes
    -----
    For each target in overlap region:
    1. Calculate weighted mean flux in blue overlap: F_blue = Σ(F_i × w_i) / Σ(w_i)
    2. Calculate weighted mean flux in red overlap:  F_red  = Σ(F_i × w_i) / Σ(w_i)
       where w_i = ivar_i (inverse variance weights)
    3. Ratio = F_blue / F_red
    4. Take median across all high-SNR targets

    This accounts for:
    - Different uncertainties per pixel
    - Bad pixels (ivar=0 are automatically excluded)
    - Wavelength-dependent noise
    """

    import numpy as np
    import matplotlib.pyplot as plt

    print("\n" + "="*70)
    print("WEIGHTED OVERLAP METHOD (LR MODE)")
    print("="*70)

    # ========================================================================
    # STEP 0: Validate input
    # ========================================================================

    if apsob.nbands() != 2:
        raise ValueError(f"Requires 2 arms (got {apsob.nbands()})")

    if apsob.join_arms():
        raise ValueError("Arms already joined. Run before join_arms=True")

    # Get wavelength arrays
    wavelist = apsob.wavelist()
    blue_wave = wavelist[0]
    red_wave = wavelist[1]

    # Check for overlap
    blue_max = blue_wave[-1]
    red_min = red_wave[0]

    if red_min >= blue_max:
        raise ValueError(f"No overlap detected. Blue ends at {blue_max:.1f}Å, "
                        f"Red starts at {red_min:.1f}Å. "
                        f"Use calculate_arms_ratio_universal() for HR mode.")

    overlap_start = red_min
    overlap_end = blue_max
    overlap_width = overlap_end - overlap_start

    print(f"\nAPSOB configuration:")
    print(f"  Mode: {apsob.mode()}")
    print(f"  Resolution: {apsob.res_mode()}")
    print(f"  Number of targets: {len(apsob._targetlist)}")

    print(f"\nOverlap region:")
    print(f"  Full overlap: {overlap_start:.1f} - {overlap_end:.1f} Å ({overlap_width:.1f}Å)")

    # Define trimmed overlap (avoid corrupted edges)
    overlap_trim_start = overlap_start + overlap_trim
    overlap_trim_end = overlap_end - overlap_trim
    overlap_trim_width = overlap_trim_end - overlap_trim_start

    if overlap_trim_width < 20:
        raise ValueError(f"Overlap too narrow after trimming ({overlap_trim_width:.1f}Å). "
                        f"Reduce overlap_trim parameter.")

    print(f"  Trimmed overlap: {overlap_trim_start:.1f} - {overlap_trim_end:.1f} Å "
          f"({overlap_trim_width:.1f}Å)")
    print(f"  Edge trim: {overlap_trim:.0f}Å on each side")

    # ========================================================================
    # STEP 1: Extract data from APSOB
    # ========================================================================

    blue_flux_list = []
    blue_ivar_list = []
    red_flux_list = []
    red_ivar_list = []
    snr_list = []
    aps_id_list = []

    for targ in apsob._targetlist:
        if targ.fib_status.upper() != 'A':
            continue

        blue_flux_list.append(targ.spectra[0].flux)
        blue_ivar_list.append(targ.spectra[0].ivar)
        red_flux_list.append(targ.spectra[1].flux)
        red_ivar_list.append(targ.spectra[1].ivar)

        # Get SNR from metadata
        snr_blue = targ.meta[0].get('SNR', np.nan)
        snr_red = targ.meta[1].get('SNR', np.nan)
        combined_snr = np.minimum(snr_blue, snr_red)
        snr_list.append(combined_snr)

        aps_id_list.append(targ.aps_id)

    blue_flux = np.array(blue_flux_list)
    blue_ivar = np.array(blue_ivar_list)
    red_flux = np.array(red_flux_list)
    red_ivar = np.array(red_ivar_list)
    snr_array = np.array(snr_list)
    aps_ids = np.array(aps_id_list)

    n_targets = len(blue_flux)

    # ========================================================================
    # STEP 2: Select good targets (SNR threshold)
    # ========================================================================

    good_snr_mask = snr_array >= min_snr
    n_good = np.sum(good_snr_mask)

    print(f"\nTarget selection:")
    print(f"  Total active targets: {n_targets}")
    print(f"  SNR threshold: {min_snr:.1f}")
    print(f"  Targets with SNR ≥ {min_snr}: {n_good}")

    # if n_good < 3:
    #     raise ValueError(f"Too few high-SNR targets ({n_good}). Lower min_snr threshold.")

    # ========================================================================
    # STEP 3: Get overlap masks
    # ========================================================================

    # Masks for TRIMMED overlap region
    blue_overlap_mask = (blue_wave >= overlap_trim_start) & (blue_wave <= overlap_trim_end)
    red_overlap_mask = (red_wave >= overlap_trim_start) & (red_wave <= overlap_trim_end)

    n_blue_pixels = np.sum(blue_overlap_mask)
    n_red_pixels = np.sum(red_overlap_mask)

    print(f"\nOverlap pixels:")
    print(f"  Blue arm: {n_blue_pixels} pixels in trimmed overlap")
    print(f"  Red arm:  {n_red_pixels} pixels in trimmed overlap")

    # ========================================================================
    # STEP 4: Calculate weighted flux ratios
    # ========================================================================

    print(f"\nCalculating weighted flux ratios...")

    ratios = []
    blue_weighted_flux = []
    red_weighted_flux = []
    effective_pixels_blue = []
    effective_pixels_red = []
    target_indices = []

    for i in range(n_targets):
        if not good_snr_mask[i]:
            continue

        # Get overlap flux and ivar for blue arm
        blue_flux_overlap = blue_flux[i, blue_overlap_mask]
        blue_ivar_overlap = blue_ivar[i, blue_overlap_mask]

        # Get overlap flux and ivar for red arm
        red_flux_overlap = red_flux[i, red_overlap_mask]
        red_ivar_overlap = red_ivar[i, red_overlap_mask]

        # CRITICAL: Only use pixels with good data (ivar > 0)
        blue_good = (blue_ivar_overlap > 0) & np.isfinite(blue_flux_overlap)
        red_good = (red_ivar_overlap > 0) & np.isfinite(red_flux_overlap)

        # Need at least 5 good pixels in each arm
        if np.sum(blue_good) < 5 or np.sum(red_good) < 5:
            continue

        # Calculate WEIGHTED MEAN flux in overlap
        # Formula: <F> = Σ(F_i × w_i) / Σ(w_i)  where w_i = ivar_i

        blue_weights = blue_ivar_overlap[blue_good]
        blue_flux_good = blue_flux_overlap[blue_good]
        blue_weighted_mean = np.sum(blue_flux_good * blue_weights) / np.sum(blue_weights)

        red_weights = red_ivar_overlap[red_good]
        red_flux_good = red_flux_overlap[red_good]
        red_weighted_mean = np.sum(red_flux_good * red_weights) / np.sum(red_weights)

        # Sanity checks
        if blue_weighted_mean <= 0 or red_weighted_mean <= 0:
            continue

        if not np.isfinite(blue_weighted_mean) or not np.isfinite(red_weighted_mean):
            continue

        # Calculate ratio
        ratio = blue_weighted_mean / red_weighted_mean

        # Sanity check on ratio
        if ratio < 0.3 or ratio > 10.0:
            continue

        # Store results
        ratios.append(ratio)
        blue_weighted_flux.append(blue_weighted_mean)
        red_weighted_flux.append(red_weighted_mean)
        effective_pixels_blue.append(np.sum(blue_good))
        effective_pixels_red.append(np.sum(red_good))
        target_indices.append(i)

    ratios = np.array(ratios)
    n_used = len(ratios)

    print(f"  ✓ Successfully calculated ratios for {n_used}/{n_good} targets")

    # if n_used < 3:
    #     raise ValueError(f"Too few successful calculations ({n_used}). "
    #                     f"Check data quality in overlap region.")

    # ========================================================================
    # STEP 5: Sigma clipping to remove outliers
    # ========================================================================

    print(f"\nRatio statistics (before clipping):")
    print(f"  Mean:   {np.mean(ratios):.4f}")
    print(f"  Median: {np.median(ratios):.4f}")
    print(f"  Std:    {np.std(ratios):.4f}")
    print(f"  Range:  {np.min(ratios):.4f} - {np.max(ratios):.4f}")

    # Iterative sigma clipping
    mask = np.ones(len(ratios), dtype=bool)

    for iteration in range(3):
        median_ratio = np.median(ratios[mask])
        std_ratio = np.std(ratios[mask])

        outliers = np.abs(ratios - median_ratio) > sigma_clip * std_ratio
        n_outliers = np.sum(outliers & mask)

        mask &= ~outliers

        if n_outliers == 0:
            break

        print(f"  Iteration {iteration + 1}: Removed {n_outliers} outliers "
              f"(threshold: {median_ratio:.4f} ± {sigma_clip}×{std_ratio:.4f})")

    ratios_clipped = ratios[mask]
    target_indices_clipped = np.array(target_indices)[mask]

    print(f"\nRatio statistics (after clipping):")
    print(f"  Mean:   {np.mean(ratios_clipped):.4f}")
    print(f"  Median: {np.median(ratios_clipped):.4f}")
    print(f"  Std:    {np.std(ratios_clipped):.4f}")
    print(f"  Range:  {np.min(ratios_clipped):.4f} - {np.max(ratios_clipped):.4f}")
    print(f"  N targets used: {len(ratios_clipped)}")

    # ========================================================================
    # STEP 6: Calculate final arms_ratio
    # ========================================================================

    final_ratio = np.median(ratios_clipped)
    final_std = np.std(ratios_clipped)

    # Quality assessment
    if final_std < 0.02:
        quality = "excellent"
    elif final_std < 0.04:
        quality = "good"
    elif final_std < 0.06:
        quality = "fair"
    else:
        quality = "poor"

    print(f"\n" + "="*70)
    print(f"RESULT: arms_ratio = [1.0, {final_ratio:.4f}] ± {final_std:.4f}")
    print(f"Quality: {quality.upper()}")
    print(f"Method: Weighted overlap (inverse variance weights)")
    print(f"="*70 + "\n")

    # ========================================================================
    # STEP 7: Diagnostic plots
    # ========================================================================

    if plot_diagnostics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Ratio distribution
        ax = axes[0, 0]
        ax.hist(ratios, bins=30, alpha=0.5, label='All', color='gray', edgecolor='black')
        ax.hist(ratios_clipped, bins=30, alpha=0.7, label='After clipping',
                color='blue', edgecolor='black')
        ax.axvline(final_ratio, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {final_ratio:.4f}')
        ax.axvline(final_ratio - final_std, color='red', linestyle=':', alpha=0.5)
        ax.axvline(final_ratio + final_std, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Arms Ratio (Blue/Red)', fontsize=12)
        ax.set_ylabel('Number of Targets', fontsize=12)
        ax.set_title(f'Ratio Distribution (N={len(ratios_clipped)}, Quality: {quality})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Plot 2: Ratio vs SNR
        ax = axes[0, 1]
        snr_used = snr_array[target_indices_clipped]
        ax.scatter(snr_used, ratios_clipped, alpha=0.6, s=40, c='blue', edgecolors='black')
        ax.axhline(final_ratio, color='red', linestyle='--', linewidth=2)
        ax.axhline(final_ratio - final_std, color='red', linestyle=':', alpha=0.5)
        ax.axhline(final_ratio + final_std, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Combined SNR', fontsize=12)
        ax.set_ylabel('Arms Ratio', fontsize=12)
        ax.set_title('Ratio vs SNR', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        # Plot 3: Example overlap comparison
        ax = axes[1, 0]
        # Pick median example
        median_idx = np.argmin(np.abs(ratios_clipped - final_ratio))
        example_target_idx = target_indices_clipped[median_idx]
        example_aps_id = aps_ids[example_target_idx]

        # Get overlap data
        blue_ex = blue_flux[example_target_idx, blue_overlap_mask]
        red_ex = red_flux[example_target_idx, red_overlap_mask]
        blue_wave_ex = blue_wave[blue_overlap_mask]
        red_wave_ex = red_wave[red_overlap_mask]

        ax.plot(blue_wave_ex, blue_ex, 'b-', linewidth=2, label='Blue overlap', alpha=0.7)
        ax.plot(red_wave_ex, red_ex, 'r-', linewidth=2, label='Red overlap', alpha=0.7)

        # Mark trimmed region
        ax.axvspan(overlap_trim_start, overlap_trim_end, alpha=0.2, color='green',
                   label='Used region')
        ax.axvspan(overlap_start, overlap_trim_start, alpha=0.1, color='gray',
                   label='Trimmed')
        ax.axvspan(overlap_trim_end, overlap_end, alpha=0.1, color='gray')

        ax.set_xlabel('Wavelength (Å)', fontsize=12)
        ax.set_ylabel('Flux', fontsize=12)
        ax.set_title(f'Example Overlap (APS_ID={example_aps_id}, ratio={ratios_clipped[median_idx]:.4f})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Plot 4: Blue vs Red weighted flux
        ax = axes[1, 1]
        blue_wflux = np.array(blue_weighted_flux)[mask]
        red_wflux = np.array(red_weighted_flux)[mask]

        # Plot data points
        ax.scatter(red_wflux, blue_wflux, alpha=0.6, s=40, c=snr_used,
                   cmap='viridis', edgecolors='black')

        # Plot ideal 1:1 line
        flux_min = min(np.min(red_wflux), np.min(blue_wflux))
        flux_max = max(np.max(red_wflux), np.max(blue_wflux))
        ax.plot([flux_min, flux_max], [flux_min, flux_max], 'k--',
                linewidth=2, alpha=0.5, label='1:1 line')

        # Plot fitted ratio line
        ax.plot([flux_min, flux_max], [flux_min * final_ratio, flux_max * final_ratio],
                'r-', linewidth=2, label=f'Ratio = {final_ratio:.4f}')

        ax.set_xlabel('Red Weighted Flux (overlap)', fontsize=12)
        ax.set_ylabel('Blue Weighted Flux (overlap)', fontsize=12)
        ax.set_title('Blue vs Red Flux in Overlap (colored by SNR)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('SNR', fontsize=10)

        plt.tight_layout()

        plot_filename = 'arms_ratio_overlap_diagnostics.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Diagnostic plot saved: {plot_filename}\n")
        plt.show()

    # ========================================================================
    # STEP 8: Return results
    # ========================================================================

    metadata = {
        'method': 'weighted_overlap',
        'ratio': final_ratio,
        'ratio_std': final_std,
        'n_targets_total': n_targets,
        'n_targets_good_snr': n_good,
        'n_targets_used': len(ratios_clipped),
        'quality': quality,
        'overlap_full': [overlap_start, overlap_end],
        'overlap_trimmed': [overlap_trim_start, overlap_trim_end],
        'overlap_width': overlap_width,
        'overlap_trim_width': overlap_trim_width,
        'min_snr': min_snr,
        'ratios_all': ratios_clipped,
        'aps_ids_used': aps_ids[target_indices_clipped],
        'mode': apsob.mode(),
        'res_mode': apsob.res_mode()[0],
        'setups': apsob.setups(),
    }

    return [1.0, final_ratio], metadata


###########################################################################
# MAIN EXECUTION - UPDATED WITH OVERLAP METHOD
###########################################################################

if __name__ == "__main__":

    print("\n" + "="*70)
    print("ARMS RATIO CALIBRATION - ALL METHODS")
    print("="*70)

    # ========================================================================
    # Example 1: LR mode - WEIGHTED OVERLAP METHOD (BEST FOR LR!)
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 1: LR MODE - WEIGHTED OVERLAP METHOD")
    print("="*70)

    try:
        infiles_lr = [
            '<PYAPS_DATA>/L1/20250810/stack_3105914.fit',
            '<PYAPS_DATA>/L1/20250810/stack_3105913.fit'
        ]
        # infiles_lr = [
        #     '<PYAPS_DATA>/L1/20251116/stack_3124070.fit',
        #     '<PYAPS_DATA>/L1/20251116/stack_3124069.fit'
        # ]
        # infiles_lr = [
        #     '<PYAPS_DATA>/L1/20251022/single_3120060.fit',
        #     '<PYAPS_DATA>/L1/20251022/single_3120059.fit'
        # ]

        # infiles_lr = [
        #     '<PYAPS_DATA>/L1/20251114/stack_3123893.fit',
        #     '<PYAPS_DATA>/L1/20251114/stack_3123892.fit'
        # ]

        # infiles_lr = [
        #     '<PYAPS_DATA>/L1/20251202/stack_3126869.fit',
        #     '<PYAPS_DATA>/L1/20251202/stack_3126868.fit'
        # ]

        # infiles_lr = [
        #     '<PYAPS_DATA>/SOLAR/20250630/solar_3097703_all.fit',
        #     '<PYAPS_DATA>/SOLAR/20250630/solar_3097704_all.fit'
        # ]

        # infiles_lr = [
        #     '<PYAPS_DATA>/SOLAR/20250630/solar_3097055.fit',
        #     '<PYAPS_DATA>/SOLAR/20250630/solar_3097056.fit'
        # ]


        debugdir= '<PYAPS_DATA>/L2/SOLAR/20250630/'
        caldir='<PYAPS_DATA>/CAL'
        catdir='<PYAPS_DATA>/CAT'
        configdir='<PYAPS_DIR>/configs/ExGal_configs'


        from PyAPS.aps_utils import APSOB

        apsob_lr = APSOB(
            infiles_lr,
            wlranges=None,
            join_arms=False,
            sens_corr=True,
            mask_gaps=True,
            vacuum=True,
            catdir=catdir , caldir=caldir, debugdir=debugdir, configdir=configdir, lsftype ='SOLAR', collapse=True
        )

        # Use WEIGHTED OVERLAP method for LR
        arms_ratio_lr, meta_lr = calculate_arms_ratio_overlap_weighted(
            apsob_lr,
            overlap_trim=50,  # Trim 50Å from each edge
            min_snr=2,
            sigma_clip=3.0,
            plot_diagnostics=True
        )

        print(f"\n✓ RESULT LR MODE (Overlap Method):")
        print(f"  Recommended arms_ratio: {arms_ratio_lr}")
        print(f"  Quality: {meta_lr['quality']}")
        print(f"  Method: {meta_lr['method']}")
        print(f"  Overlap used: {meta_lr['overlap_trimmed']} Å")

    except Exception as e:
        print(f"❌ Example 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # # ========================================================================
    # # Example 2: HR mode - UNIVERSAL CONTINUUM METHOD (NO OVERLAP)
    # # ========================================================================
    # print("\n" + "="*70)
    # print("EXAMPLE 2: HR MODE - UNIVERSAL CONTINUUM METHOD")
    # print("="*70)

    # try:
    #     infiles_hr = [
    #         '/path/to/blue_HR.fits',
    #         '/path/to/red_HR.fits'
    #     ]

    #     apsob_hr = APSOB(
    #         infiles_hr,
    #         wlranges=[[3800, 5800], [6000, 9280]],
    #         join_arms=False,
    #         sens_corr=True,
    #         mask_gaps=True,
    #         vacuum=True
    #     )

    #     # Use UNIVERSAL method for HR (no overlap)
    #     arms_ratio_hr, meta_hr = calculate_arms_ratio_universal(
    #         apsob_hr,
    #         min_snr_percentile=85,
    #         edge_buffer=150,
    #         fit_window=200,
    #         continuum_order=2,
    #         plot_diagnostics=True
    #     )

    #     print(f"\n✓ RESULT HR MODE (Universal Method):")
    #     print(f"  Recommended arms_ratio: {arms_ratio_hr}")
    #     print(f"  Quality: {meta_hr['quality']}")
    #     print(f"  Method: {meta_hr['method']}")
    #     print(f"  Gap size: {meta_hr['gap_size']:.1f} Å")

    # except Exception as e:
    #     print(f"❌ Example 2 failed: {e}")
    #     import traceback
    #     traceback.print_exc()

    # # ========================================================================
    # # Example 3: LR mode - COMPARE BOTH METHODS
    # # ========================================================================
    # print("\n" + "="*70)
    # print("EXAMPLE 3: LR MODE - METHOD COMPARISON")
    # print("="*70)

    # try:
    #     infiles_comp = [
    #         '<PYAPS_DATA>/L1/20250810/stack_3105914.fit',
    #         '<PYAPS_DATA>/L1/20250810/stack_3105913.fit'
    #     ]

    #     apsob_comp = APSOB(
    #         infiles_comp,
    #         wlranges=None,
    #         join_arms=False,
    #         sens_corr=True,
    #         mask_gaps=True,
    #         vacuum=True
    #     )

    #     # Method A: Weighted overlap
    #     ratio_overlap, meta_overlap = calculate_arms_ratio_overlap_weighted(
    #         apsob_comp,
    #         overlap_trim=50,
    #         min_snr=5,
    #         plot_diagnostics=False
    #     )

    #     # Method B: Universal continuum
    #     ratio_universal, meta_universal = calculate_arms_ratio_universal(
    #         apsob_comp,
    #         min_snr_percentile=90,
    #         edge_buffer=100,
    #         fit_window=200,
    #         plot_diagnostics=False
    #     )

    #     print(f"\n✓ METHOD COMPARISON:")
    #     print(f"  Overlap method:    {ratio_overlap[1]:.4f} ± {meta_overlap['ratio_std']:.4f}")
    #     print(f"  Universal method:  {ratio_universal[1]:.4f} ± {meta_universal['ratio_std']:.4f}")
    #     print(f"  Difference:        {abs(ratio_overlap[1] - ratio_universal[1]):.4f}")
    #     print(f"  Agreement:         {abs(ratio_overlap[1] - ratio_universal[1]) < 0.05}")

    #     # Recommendation
    #     if meta_overlap['quality'] in ['excellent', 'good']:
    #         print(f"\n  RECOMMENDATION: Use overlap method (quality: {meta_overlap['quality']})")
    #         print(f"  → arms_ratio = {ratio_overlap}")
    #     else:
    #         print(f"\n  RECOMMENDATION: Use universal method (overlap quality: {meta_overlap['quality']})")
    #         print(f"  → arms_ratio = {ratio_universal}")

    # except Exception as e:
    #     print(f"❌ Example 3 failed: {e}")
    #     import traceback
    #     traceback.print_exc()

    # print("\n" + "="*70)
    # print("ALL EXAMPLES COMPLETED")
    # print("="*70 + "\n")
