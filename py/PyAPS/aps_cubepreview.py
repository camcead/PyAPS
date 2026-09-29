"""
aps_cubepreview.py

Diagnostic visualization tool for IFU data processing pipeline.
Shows FLUX and SNR maps at three stages with INTERACTIVE Plotly plots.

Part of PyAPS - can be used for both ExGal and Galactic sources.

Version: 2.0 - Interactive Plotly Edition
Author: APS Team  
Date: October 2025
"""

import numpy as np
import os
import sys
from pathlib import Path
import json
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, ICRS
from astropy_healpix import HEALPix
import astropy.units as u

from PyAPS.aps_utils import APSOB, none_or_str, str2bool, l1_fileinfo, print_args
from PyAPS.aps_common_args import build_common_parser, resolve_common_args
from PyAPS import aps_constants
from PyAPS import ExGalPrepare, ExGalutil, IFUExGalPrepare

# Constants
Clight = 299792.458  # km/s
large_error = aps_constants.large_error


class CubePreview:
    """
    Main class for generating diagnostic previews of IFU cube processing.
    """
    
    def __init__(self, infiles, patch_file, patch_id, config_file, outpath,
                 wlranges=None, aps_ids=None, targsrvy=None, targclass=None,
                 mask_aps_ids=None, sens_corr=True, mask_gaps=True, 
                 safe_mask_gaps=True, vacuum=False, tellurics=False,
                 fill_gap=False, arms_ratio=None, join_arms=True,
                 catdir=None, caldir=None, configdir=None):
        
        self.infiles = infiles
        self.patch_file = patch_file
        self.patch_id = patch_id
        self.config_file = config_file
        self.outpath = Path(outpath)
        self.wlranges = wlranges
        self.aps_ids = aps_ids
        self.targsrvy = targsrvy
        self.targclass = targclass
        self.mask_aps_ids = mask_aps_ids
        self.sens_corr = sens_corr
        self.mask_gaps = mask_gaps
        self.safe_mask_gaps = safe_mask_gaps
        self.vacuum = vacuum
        self.tellurics = tellurics
        self.fill_gap = fill_gap
        self.arms_ratio = arms_ratio
        self.join_arms = join_arms
        self.catdir = catdir
        self.caldir = caldir
        self.configdir = configdir
        
        # Create output directory
        self.figdir = self.outpath / 'preview_figs'
        self.figdir.mkdir(parents=True, exist_ok=True)
        
        # Storage for processing stages
        self.cube = None
        self.spatial_bins = None
        self.voronoi_data = None
        self.configs = None
        self.patch_info = None
        self.headname = None
        self.z = None
        self.zerr = None
        self.patch_class = None
        
    def load_patch_info(self):
        """Load and validate patch information from patch file."""
        print(f"\n{'='*70}")
        print("LOADING PATCH INFORMATION")
        print(f"{'='*70}")
        
        # Check if patch file exists
        if not Path(self.patch_file).exists():
            raise FileNotFoundError(f"Patch file not found: {self.patch_file}")
        
        # Load patch table - same as ifu_worker
        if Path(self.patch_file).suffix == '.fits':
            patch_table = Table.read(self.patch_file)
            
            # Replace missing values with nan (since astropy 5.0.1)
            for ptcl in patch_table.colnames:
                if hasattr(patch_table[ptcl], 'mask'):
                    patch_table[ptcl] = patch_table[ptcl].filled(np.nan)
        else:
            patch_table = read_ascii_patchfile(self.patch_file)
            assert patch_table.colnames == ['id','RA_icrs','DEC_icrs','A_world','B_world','angle','flag','type','Z','ZERR','ZWARN', 'CLASS'], \
                'Error: inconsistent patch_file column names'
        
        print(f"Loaded patch table with {len(patch_table)} entries")
        
        # Find the requested patch
        patch_match = patch_table[patch_table['id'] == self.patch_id]
        
        if len(patch_match) == 0:
            raise ValueError(f"Patch ID {self.patch_id} not found in patch file")
        
        if len(patch_match) > 1:
            print(f"WARNING: Multiple entries found for patch ID {self.patch_id}")
            print(f"Using the first match")
        
        self.patch_info = patch_match[0]
        
        # Validate patch - same checks as ifu_worker
        # Skip if it's a Mask target
        if str(self.patch_info['type']).replace(' ', '').upper() == 'M':
            raise ValueError(f"Patch {self.patch_id} is a Mask target - cannot process")
        
        # Check classification - handle both scalar and array cases
        class_value = str(self.patch_info['CLASS']).strip().upper()
        if class_value in ['', 'NAN'] or len(class_value) == 0:
            raise ValueError(f"Patch {self.patch_id} missing or invalid classification")
        
        # Check redshift - handle array case (multi-class support)
        z_value = self.patch_info['Z']
        zerr_value = self.patch_info['ZERR']
        
        # Handle array case by taking first element
        if isinstance(z_value, np.ndarray):
            if np.all(np.isnan(z_value)):
                raise ValueError(f"Patch {self.patch_id} missing redshift information")
            # Take first non-NaN value
            z_value = z_value[~np.isnan(z_value)][0]
        elif np.isnan(z_value):
            raise ValueError(f"Patch {self.patch_id} missing redshift information")
        
        if isinstance(zerr_value, np.ndarray):
            if np.all(np.isnan(zerr_value)):
                raise ValueError(f"Patch {self.patch_id} missing redshift error")
            # Take first non-NaN value
            zerr_value = zerr_value[~np.isnan(zerr_value)][0]
        elif np.isnan(zerr_value):
            raise ValueError(f"Patch {self.patch_id} missing redshift error")
        
        # Store for easier access
        self.z = float(z_value)
        self.zerr = float(zerr_value)
        self.patch_class = class_value
        
        print(f"\nPatch {self.patch_id} Information:")
        print(f"  Type: {self.patch_info['type']}")
        print(f"  Class: {self.patch_class}")
        print(f"  Redshift: {self.z:.6f} ± {self.zerr:.6f}")
        print(f"  RA: {self.patch_info['RA_icrs']:.6f} deg")
        print(f"  DEC: {self.patch_info['DEC_icrs']:.6f} deg")
        print(f"  Semi-major axis: {self.patch_info['A_world']*3600:.2f} arcsec")
        print(f"  Semi-minor axis: {self.patch_info['B_world']*3600:.2f} arcsec")
        
        return self.patch_info
    
    def load_config(self):
        """Load configuration file."""
        print(f"\n{'='*70}")
        print("LOADING CONFIGURATION")
        print(f"{'='*70}")
        
        if not Path(self.config_file).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        self.configs = json.load(open(self.config_file))
        self.configs['CONFIG_FILE'] = os.path.basename(self.config_file)
        
        # CRITICAL: Add infiles to configs (required by save_table)
        self.configs['infiles'] = self.infiles
        
        # Set default values for critical parameters
        if 'MIN_SNR' not in self.configs:
            self.configs['MIN_SNR'] = 1.e-3
        
        if 'SPBIN_SIZE_EXGAL' not in self.configs:
            self.configs['SPBIN_SIZE_EXGAL'] = -1.0
        
        if 'VORONOI' not in self.configs:
            self.configs['VORONOI'] = 0
        
        if 'TARGET_SNR' not in self.configs or str(self.configs['TARGET_SNR']).lower() in ['none', 'null', '']:
            self.configs['TARGET_SNR'] = 20.0
        
        if 'COVAR_VOR' not in self.configs:
            self.configs['COVAR_VOR'] = 0.0
        
        if 'PIXELSIZE' not in self.configs:
            self.configs['PIXELSIZE'] = 0.0
        
        print(f"Configuration loaded: {self.config_file}")
        print(f"  MIN_SNR: {self.configs['MIN_SNR']}")
        print(f"  SPBIN_SIZE_EXGAL: {self.configs['SPBIN_SIZE_EXGAL']}")
        print(f"  VORONOI: {self.configs['VORONOI']}")
        print(f"  TARGET_SNR: {self.configs['TARGET_SNR']}")
        print(f"  Number of input files: {len(self.configs['infiles'])}")
        
        return self.configs
    

    def prepare_cube_data(self):
        """Prepare cube data structure similar to ifu_ExGal with spherical projection correction."""
        print(f"\n{'='*70}")
        print("PREPARING CUBE DATA")
        print(f"{'='*70}")
        
        # Define patch area - same as ifu_worker/ifu_ExGal
        enhanced_rad_factor = 1.0
        
        # Check if we need larger radius (similar to ifu_worker logic)
        patch_area = [
            self.patch_info['RA_icrs'],
            self.patch_info['DEC_icrs'],
            self.patch_info['A_world'] * 3600.0 * enhanced_rad_factor,
            self.patch_info['B_world'] * 3600.0 * enhanced_rad_factor,
            self.patch_info['angle']
        ]
        
        # For now, no mask areas
        mask_areas = None
        
        # Create APSOBJ - same as ifu_ExGal
        print("Creating APSOB object...")
        APSOBJ = APSOB(
            self.infiles, 
            targsrvy=self.targsrvy, 
            targclass=self.targclass, 
            aps_ids=self.aps_ids,
            mask_aps_ids=self.mask_aps_ids, 
            area=patch_area, 
            mask_areas=mask_areas,
            wlranges=self.wlranges, 
            sens_corr=self.sens_corr, 
            mask_gaps=self.mask_gaps, 
            safe_mask_gaps=self.safe_mask_gaps, 
            vacuum=self.vacuum, 
            tellurics=self.tellurics,
            fill_gap=self.fill_gap, 
            arms_ratio=self.arms_ratio, 
            join_arms=self.join_arms,
            catdir=self.catdir, 
            caldir=self.caldir, 
            configdir=self.configdir
        )
        
        targs = APSOBJ.data()
        targs_id = APSOBJ.id()
        targs_idfx = APSOBJ.idfx()
        targs_origin = APSOBJ.origin()
        
        print(f"Found {len(targs_id)} spaxels in patch area")
        
        if len(targs_id) == 0:
            raise ValueError("No spaxels found in the specified patch area")
        
        # Get wavelength information
        original_wave = targs[0].spectra[0].wave
        nwave = len(original_wave)
        pixel_scale = original_wave[1] - original_wave[0]
        
        # Initialize cube structure - same as ifu_ExGal
        targ_len = len(targs_id)
        self.cube = {
            'aps_id': np.zeros(targ_len, dtype=np.int32),
            'targid': np.empty(targ_len, dtype='U40'),
            'cname': np.empty(targ_len, dtype='U40'),
            'x': np.zeros(targ_len, dtype=np.float64),  # EXPLICIT float64
            'y': np.zeros(targ_len, dtype=np.float64),  # EXPLICIT float64
            'z': np.zeros(targ_len),
            'zerr': np.zeros(targ_len),
            'healpix': np.zeros(targ_len, dtype=np.int64),
            'x_0': np.zeros(targ_len),
            'y_0': np.zeros(targ_len),
            'wave': np.zeros(targ_len),
            'spec': np.zeros((nwave, targ_len)),
            'error': np.zeros((nwave, targ_len)),
            'snr': np.zeros(targ_len),
            'signal': np.zeros(targ_len),
            'noise': np.zeros(targ_len),
            'velscale': 0.0,
            'pixelsize': 0.0,
            'pixel_scale': 0.0
        }
        
        # Initialize HEALPix - same as ifu_ExGal
        hp = HEALPix(nside=1024, order='nested', frame=ICRS())
        
        # ===== SPHERICAL PROJECTION CORRECTION =====
        # Get reference coordinates for tangent plane projection
        ref_ra = targs_origin[0]   # degrees
        ref_dec = targs_origin[1]  # degrees
        
        # Calculate spherical correction factor
        cos_dec_correction = np.cos(np.deg2rad(ref_dec))
        
        print(f"\n{'='*70}")
        print("SPHERICAL PROJECTION CORRECTION")
        print(f"{'='*70}")
        print(f"Reference coordinates:")
        print(f"  RA:  {ref_ra:.6f} deg")
        print(f"  DEC: {ref_dec:.6f} deg")
        print(f"Correction factors:")
        print(f"  cos(DEC) = {cos_dec_correction:.6f}")
        print(f"  X-scale factor: {cos_dec_correction:.6f}")
        print(f"  Y-scale factor: 1.0")
        print(f"\nThis corrects for spherical sky geometry on tangent plane.")
        print(f"Without this, maps would be elongated in X by factor {1.0/cos_dec_correction:.2f}")
        print(f"{'='*70}\n")
        
        # Fill cube structure - same as ifu_ExGal but WITH spherical correction
        print("Filling cube data structure...")
        for ntgs, tgs in enumerate(targs_id):
            tgs_indx = targs_idfx[tgs]
            
            self.cube['aps_id'][ntgs] = targs[tgs_indx].aps_id
            self.cube['targid'][ntgs] = targs[tgs_indx].targid
            self.cube['cname'][ntgs] = targs[tgs_indx].cname
            self.cube['spec'][:, ntgs] = targs[tgs_indx].spectra[0].flux
            
            # Handle inverse variance - same as ifu_ExGal
            ivar_tgs = targs[tgs_indx].spectra[0].ivar
            ivar_mask_value = 1.0 / (large_error**2)
            mask_tgs = (ivar_tgs <= 10 * ivar_mask_value)
            nomask_tgs = (ivar_tgs > 10 * ivar_mask_value)
            ivar_tgs[mask_tgs] = ivar_mask_value
            espec_tgs = 1. / (ivar_tgs**0.5)
            self.cube['error'][:, ntgs] = espec_tgs
            
            # Calculate SNR - same as ifu_ExGal
            self.cube['signal'][ntgs] = np.nanmean(
                targs[tgs_indx].spectra[0].flux[nomask_tgs], axis=0
            )
            self.cube['noise'][ntgs] = np.sqrt(
                np.nanmean(espec_tgs[nomask_tgs]**2, axis=0)
            )
            
            if self.cube['noise'][ntgs] > 0.0:
                self.cube['snr'][ntgs] = self.cube['signal'][ntgs] / self.cube['noise'][ntgs]
            else:
                self.cube['snr'][ntgs] = 0.0
            
            # ===== POSITIONS WITH SPHERICAL PROJECTION CORRECTION =====
            # Calculate angular offsets in degrees
            dx_deg = targs[tgs_indx].targra - ref_ra
            dy_deg = targs[tgs_indx].targdec - ref_dec
            
            # Apply tangent plane projection to convert to physical angular separation
            # X: RA offset needs cos(DEC) correction + flip sign (RA increases West)
            # Y: DEC offset is direct conversion
            self.cube['x'][ntgs] = np.float64(-1.0 * dx_deg * 3600.0 * cos_dec_correction)
            self.cube['y'][ntgs] = np.float64(dy_deg * 3600.0)
            
            # Store reference coordinates
            self.cube['x_0'][ntgs] = ref_ra
            self.cube['y_0'][ntgs] = ref_dec
            
            # Use stored z and zerr values
            self.cube['z'][ntgs] = self.z
            self.cube['zerr'][ntgs] = self.zerr
            
            # HEALPix - same as ifu_ExGal
            coord = SkyCoord(
                f"{targs[tgs_indx].targra}d {targs[tgs_indx].targdec}d"
            )
            self.cube['healpix'][ntgs] = hp.skycoord_to_healpix(coord)
        
        # De-redshift spectra - same as ifu_ExGal
        self.cube['wave'] = targs[0].spectra[0].wave / (1 + self.z)
        
        # Set velscale - same as ifu_ExGal
        if str(self.configs.get('VELSCALE', '')).replace(" ", '').lower() in ['none', 'null', '']:
            s_sampling = 1.0
            lam_range = [self.cube['wave'][0], self.cube['wave'][-1]]
            s_lam = len(self.cube['wave'])
            dlam = (lam_range[1] - lam_range[0]) / (s_lam - 1.)
            lim = lam_range / dlam + [-0.5, 0.5]
            loglim = np.log(lim)
            self.cube['velscale'] = float(np.diff(loglim) / (s_sampling * s_lam) * Clight)
        else:
            self.cube['velscale'] = self.configs['VELSCALE']
        
        self.cube['pixelsize'] = self.configs['PIXELSIZE']
        self.cube['pixel_scale'] = pixel_scale
        
        # ===== COORDINATE DIAGNOSTICS =====
        print(f"\nCube prepared with {targ_len} spaxels")
        print(f"\nSpectroscopic properties:")
        print(f"  SNR range: {np.nanmin(self.cube['snr']):.2f} - {np.nanmax(self.cube['snr']):.2f}")
        print(f"  Mean SNR: {np.nanmean(self.cube['snr']):.2f}")
        print(f"  Median SNR: {np.nanmedian(self.cube['snr']):.2f}")
        
        print(f"\nCoordinate system diagnostics:")
        print(f"  Data types: x={self.cube['x'].dtype}, y={self.cube['y'].dtype}")
        
        # Calculate coordinate ranges
        x_range = np.ptp(self.cube['x'])  # peak-to-peak
        y_range = np.ptp(self.cube['y'])
        x_min, x_max = np.min(self.cube['x']), np.max(self.cube['x'])
        y_min, y_max = np.min(self.cube['y']), np.max(self.cube['y'])
        
        print(f"\n  X coordinates:")
        print(f"    Range: [{x_min:.3f}, {x_max:.3f}] arcsec")
        print(f"    Span:  {x_range:.3f} arcsec")
        print(f"  Y coordinates:")
        print(f"    Range: [{y_min:.3f}, {y_max:.3f}] arcsec")
        print(f"    Span:  {y_range:.3f} arcsec")
        
        # Check aspect ratio
        if y_range > 0:
            aspect_ratio = x_range / y_range
            print(f"\n  Aspect ratio (X/Y): {aspect_ratio:.3f}")
            
            # Expected aspect ratio for elliptical patch
            expected_ratio = self.patch_info['A_world'] / self.patch_info['B_world']
            print(f"  Expected ratio (from patch): {expected_ratio:.3f}")
            
            ratio_diff = abs(aspect_ratio - expected_ratio) / expected_ratio
            
            if ratio_diff < 0.15:  # Within 15%
                print(f"  ✓ Aspect ratio matches patch geometry well ({ratio_diff*100:.1f}% difference)")
            elif ratio_diff < 0.30:  # Within 30%
                print(f"  ⚠️  Aspect ratio somewhat different from patch ({ratio_diff*100:.1f}% difference)")
            else:
                print(f"  ❌ WARNING: Aspect ratio very different from patch ({ratio_diff*100:.1f}% difference)")
                print(f"      This may indicate a spherical projection problem")
            
            # Additional check for circular patches
            if abs(expected_ratio - 1.0) < 0.1:  # Nearly circular patch
                if aspect_ratio > 1.3 or aspect_ratio < 0.7:
                    print(f"  ❌ WARNING: Circular patch has elongated aspect ratio {aspect_ratio:.2f}")
                    print(f"      Expected ~1.0. Check spherical projection correction.")
                else:
                    print(f"  ✓ Circular patch geometry preserved")
        
        print(f"{'='*70}")
        
        return self.cube




    def apply_spatial_binning(self):
        """Apply spatial binning if configured."""
        print(f"\n{'='*70}")
        print("APPLYING SPATIAL BINNING")
        print(f"{'='*70}")
        
        spbin_size = self.configs.get('SPBIN_SIZE_EXGAL', -1.0)
        
        if spbin_size <= 0:
            print("Spatial binning disabled (SPBIN_SIZE_EXGAL <= 0)")
            return None
        
        print(f"Spatial bin size: {spbin_size} arcsec")
        
        # Apply spatial binning
        self.spatial_bins, aps_id_to_spatial_bin = ExGalPrepare.spatial_bin_with_provenance(
            self.cube,
            spbin_size,
            min_snr=self.configs['MIN_SNR'],
            verbose=True
        )
        
        idx_inside = np.where(self.spatial_bins['flag'] == 1)[0]
        n_valid = len(idx_inside)
        n_total = len(self.spatial_bins['snr'])
        
        print(f"\nSpatial binning results:")
        print(f"  Total bins: {n_total}")
        print(f"  Valid bins (SNR >= {self.configs['MIN_SNR']}): {n_valid}")
        print(f"  SNR range: {np.nanmin(self.spatial_bins['snr']):.2f} - {np.nanmax(self.spatial_bins['snr']):.2f}")
        print(f"  Mean SNR: {np.nanmean(self.spatial_bins['snr'][idx_inside]):.2f}")
        
        return self.spatial_bins
    
    def apply_voronoi_binning(self):
        """Apply Voronoi binning if configured."""
        print(f"\n{'='*70}")
        print("APPLYING VORONOI BINNING")
        print(f"{'='*70}")
        
        if self.configs['VORONOI'] != 1:
            print("Voronoi binning disabled (VORONOI != 1)")
            return None
        
        print(f"Target SNR: {self.configs['TARGET_SNR']}")
        
        # Generate headname for this patch
        self.headname = f"preview_P{self.patch_id:04d}"
        
        # Determine which data to use for Voronoi binning
        if self.spatial_bins is not None:
            # Use spatial bins
            idx_inside = np.where(self.spatial_bins['flag'] == 1)[0]
            idx_outside = np.where(self.spatial_bins['flag'] == 0)[0]
            
            binNum = ExGalPrepare.define_voronoi_bins(
                self.configs['VORONOI'],
                self.spatial_bins['aps_ids'],
                self.spatial_bins['targid'],
                self.spatial_bins['cname'],
                self.spatial_bins['x'],
                self.spatial_bins['y'],
                self.spatial_bins['z'],
                self.spatial_bins['zerr'],
                self.spatial_bins['healpix'],
                self.spatial_bins['x_0'],
                self.spatial_bins['y_0'],
                self.spatial_bins['signal'],
                self.spatial_bins['noise'],
                self.cube['pixelsize'],
                self.spatial_bins['snr'],
                self.configs['TARGET_SNR'],
                self.configs['COVAR_VOR'],
                idx_inside,
                idx_outside,
                self.headname,
                str(self.outpath),
                self.configs
            )
        else:
            # Use original cube
            idx_inside = np.where(self.cube['snr'] >= self.configs['MIN_SNR'])[0]
            idx_outside = np.where(self.cube['snr'] < self.configs['MIN_SNR'])[0]
            
            binNum = ExGalPrepare.define_voronoi_bins(
                self.configs['VORONOI'],
                self.cube['aps_id'],
                self.cube['targid'],
                self.cube['cname'],
                self.cube['x'],
                self.cube['y'],
                self.cube['z'],
                self.cube['zerr'],
                self.cube['healpix'],
                self.cube['x_0'],
                self.cube['y_0'],
                self.cube['signal'],
                self.cube['noise'],
                self.cube['pixelsize'],
                self.cube['snr'],
                self.configs['TARGET_SNR'],
                self.configs['COVAR_VOR'],
                idx_inside,
                idx_outside,
                self.headname,
                str(self.outpath),
                self.configs
            )
        
        # Read Voronoi data from table file
        table_file = self.outpath / f"{self.headname}_table.fits"
        if table_file.exists():
            self.voronoi_data = ExGalPrepare.read_voronoi_fits_table(str(table_file))
            
            unique_bins = np.unique(self.voronoi_data['binNum'])
            n_bins = len(unique_bins[unique_bins >= 0])
            
            print(f"\nVoronoi binning results:")
            print(f"  Number of bins: {n_bins}")
            print(f"  Number of spaxels: {len(self.voronoi_data['binNum'])}")
        else:
            print("WARNING: Voronoi table file not found")
            self.voronoi_data = None
        
        return self.voronoi_data

    def create_diagnostic_plot_plotly(self):
        """Create interactive Plotly diagnostic plot with independent axes and proper layout."""
        print(f"\n{'='*70}")
        print("CREATING INTERACTIVE PLOTLY DIAGNOSTIC PLOT")
        print(f"{'='*70}")
        
        # Determine number of stages
        n_stages = 1  # Always have original
        if self.spatial_bins is not None:
            n_stages += 1
        if self.voronoi_data is not None:
            n_stages += 1
        
        # Create figure with independent subplots
        fig = make_subplots(
            rows=2, cols=n_stages,
            subplot_titles=[],  # We'll add custom annotations
            vertical_spacing=0.15,
            horizontal_spacing=0.10,
            specs=[[{"type": "scatter"} for _ in range(n_stages)] for _ in range(2)]
        )
        
        # Define colorbar positions for each column
        colorbar_x_positions = []
        for col in range(n_stages):
            # Position colorbars to the right of each column
            x_pos = (col + 1) / n_stages - 0.02
            colorbar_x_positions.append(x_pos)
        
        stage_col = 0
        
        # ===== STAGE 1: Original Spaxels =====
        print("\nPlotting Stage 1: Original Spaxels")
        
        # FLUX (row 1)
        flux_trace = go.Scatter(
            x=self.cube['x'],
            y=self.cube['y'],
            mode='markers',
            marker=dict(
                size=8,
                color=self.cube['signal'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Flux",
                    x=colorbar_x_positions[stage_col],
                    len=0.35,
                    y=0.77,
                    yanchor='middle'
                ),
                line=dict(width=0.5, color='black')
            ),
            text=[f"APS_ID: {aid}<br>X: {x:.3f}<br>Y: {y:.3f}<br>Flux: {f:.3e}<br>SNR: {s:.2f}" 
                for aid, x, y, f, s in zip(self.cube['aps_id'], self.cube['x'], self.cube['y'],
                                            self.cube['signal'], self.cube['snr'])],
            hovertemplate='%{text}<extra></extra>',
            name='Original Flux',
            legendgroup='stage1',
            showlegend=False
        )
        fig.add_trace(flux_trace, row=1, col=stage_col+1)
        
        # SNR (row 2)
        snr_trace = go.Scatter(
            x=self.cube['x'],
            y=self.cube['y'],
            mode='markers',
            marker=dict(
                size=8,
                color=self.cube['snr'],
                colorscale='RdYlGn',
                showscale=True,
                cmin=0,
                cmax=np.nanpercentile(self.cube['snr'], 95),
                colorbar=dict(
                    title="SNR",
                    x=colorbar_x_positions[stage_col],
                    len=0.35,
                    y=0.23,
                    yanchor='middle'
                ),
                line=dict(width=0.5, color='black')
            ),
            text=[f"APS_ID: {aid}<br>X: {x:.3f}<br>Y: {y:.3f}<br>SNR: {s:.2f}" 
                for aid, x, y, s in zip(self.cube['aps_id'], self.cube['x'], self.cube['y'], self.cube['snr'])],
            hovertemplate='%{text}<extra></extra>',
            name='Original SNR',
            legendgroup='stage1',
            showlegend=False
        )
        fig.add_trace(snr_trace, row=2, col=stage_col+1)
        
        stage_col += 1
        
        # ===== STAGE 2: Spatial Binning =====
        if self.spatial_bins is not None:
            print("\nPlotting Stage 2: Spatial Binning")
            
            idx_valid = self.spatial_bins['flag'] == 1
            
            # FLUX (row 1)
            flux_spat = go.Scatter(
                x=self.spatial_bins['x'][idx_valid],
                y=self.spatial_bins['y'][idx_valid],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.spatial_bins['signal'][idx_valid],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Flux",
                        x=colorbar_x_positions[stage_col],
                        len=0.35,
                        y=0.77,
                        yanchor='middle'
                    ),
                    line=dict(width=0.5, color='black')
                ),
                text=[f"Bin: {i}<br>X: {x:.3f}<br>Y: {y:.3f}<br>Flux: {f:.3e}<br>SNR: {s:.2f}<br>N_spaxels: {len(aids)}" 
                    for i, (x, y, f, s, aids) in enumerate(zip(
                        self.spatial_bins['x'][idx_valid],
                        self.spatial_bins['y'][idx_valid],
                        self.spatial_bins['signal'][idx_valid],
                        self.spatial_bins['snr'][idx_valid],
                        [self.spatial_bins['aps_ids'][j] for j in np.where(idx_valid)[0]]
                    ))],
                hovertemplate='%{text}<extra></extra>',
                name='Spatial Flux',
                legendgroup='stage2',
                showlegend=False
            )
            fig.add_trace(flux_spat, row=1, col=stage_col+1)
            
            # SNR (row 2)
            snr_spat = go.Scatter(
                x=self.spatial_bins['x'][idx_valid],
                y=self.spatial_bins['y'][idx_valid],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.spatial_bins['snr'][idx_valid],
                    colorscale='RdYlGn',
                    showscale=True,
                    cmin=0,
                    cmax=np.nanpercentile(self.spatial_bins['snr'][idx_valid], 95),
                    colorbar=dict(
                        title="SNR",
                        x=colorbar_x_positions[stage_col],
                        len=0.35,
                        y=0.23,
                        yanchor='middle'
                    ),
                    line=dict(width=0.5, color='black')
                ),
                text=[f"Bin: {i}<br>X: {x:.3f}<br>Y: {y:.3f}<br>SNR: {s:.2f}" 
                    for i, (x, y, s) in enumerate(zip(
                        self.spatial_bins['x'][idx_valid],
                        self.spatial_bins['y'][idx_valid],
                        self.spatial_bins['snr'][idx_valid]
                    ))],
                hovertemplate='%{text}<extra></extra>',
                name='Spatial SNR',
                legendgroup='stage2',
                showlegend=False
            )
            fig.add_trace(snr_spat, row=2, col=stage_col+1)
            
            stage_col += 1
        
        # ===== STAGE 3: Voronoi Binning =====
        if self.voronoi_data is not None:
            print("\nPlotting Stage 3: Voronoi Binning")
            
            x = self.voronoi_data['x']
            y = self.voronoi_data['y']
            bins = self.voronoi_data['binNum']
            xNode = self.voronoi_data['xNode']
            yNode = self.voronoi_data['yNode']
            
            unique_bins = np.unique(bins[bins >= 0])
            n_bins = len(unique_bins)
            
            print(f"  Voronoi data: {n_bins} bins, {len(x)} spaxels")
            
            # Calculate bin properties from ORIGINAL cube data
            bin_flux = np.zeros(n_bins)
            bin_snr = np.zeros(n_bins)
            
            for i, bin_id in enumerate(unique_bins):
                mask = bins == bin_id
                bin_spaxel_x = x[mask]
                bin_spaxel_y = y[mask]
                
                flux_vals = []
                snr_vals = []
                
                for sx, sy in zip(bin_spaxel_x, bin_spaxel_y):
                    dist = np.sqrt((self.cube['x'] - sx)**2 + (self.cube['y'] - sy)**2)
                    closest_idx = np.argmin(dist)
                    
                    if dist[closest_idx] < 0.1:
                        flux_vals.append(self.cube['signal'][closest_idx])
                        snr_vals.append(self.cube['snr'][closest_idx])
                
                if len(flux_vals) > 0:
                    bin_flux[i] = np.mean(flux_vals)
                    bin_snr[i] = np.mean(snr_vals)
                else:
                    bin_flux[i] = np.nan
                    bin_snr[i] = np.nan
            
            # FLUX (row 1) - Plot bin centers
            valid_flux = np.isfinite(bin_flux)
            flux_vor = go.Scatter(
                x=xNode[valid_flux],
                y=yNode[valid_flux],
                mode='markers',
                marker=dict(
                    size=15,
                    color=bin_flux[valid_flux],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Flux",
                        x=colorbar_x_positions[stage_col],
                        len=0.35,
                        y=0.77,
                        yanchor='middle'
                    ),
                    symbol='square',
                    line=dict(width=2, color='red')
                ),
                text=[f"Vor Bin: {bid}<br>X: {x:.3f}<br>Y: {y:.3f}<br>Flux: {f:.3e}" 
                    for bid, x, y, f in zip(unique_bins[valid_flux], xNode[valid_flux], 
                                            yNode[valid_flux], bin_flux[valid_flux])],
                hovertemplate='%{text}<extra></extra>',
                name='Voronoi Flux',
                legendgroup='stage3',
                showlegend=False
            )
            fig.add_trace(flux_vor, row=1, col=stage_col+1)
            
            # SNR (row 2) - Plot bin centers
            valid_snr = np.isfinite(bin_snr)
            snr_vor = go.Scatter(
                x=xNode[valid_snr],
                y=yNode[valid_snr],
                mode='markers',
                marker=dict(
                    size=15,
                    color=bin_snr[valid_snr],
                    colorscale='RdYlGn',
                    showscale=True,
                    cmin=0,
                    cmax=np.nanpercentile(bin_snr[valid_snr], 95),
                    colorbar=dict(
                        title="SNR",
                        x=colorbar_x_positions[stage_col],
                        len=0.35,
                        y=0.23,
                        yanchor='middle'
                    ),
                    symbol='square',
                    line=dict(width=2, color='red')
                ),
                text=[f"Vor Bin: {bid}<br>X: {x:.3f}<br>Y: {y:.3f}<br>SNR: {s:.2f}" 
                    for bid, x, y, s in zip(unique_bins[valid_snr], xNode[valid_snr], 
                                            yNode[valid_snr], bin_snr[valid_snr])],
                hovertemplate='%{text}<extra></extra>',
                name='Voronoi SNR',
                legendgroup='stage3',
                showlegend=False
            )
            fig.add_trace(snr_vor, row=2, col=stage_col+1)
        
        # ===== UPDATE AXES - INDEPENDENT FOR EACH SUBPLOT =====
        
        # Get global coordinate ranges for consistent aspect ratio
        all_x = list(self.cube['x'])
        all_y = list(self.cube['y'])
        
        if self.spatial_bins is not None:
            all_x.extend(self.spatial_bins['x'][self.spatial_bins['flag'] == 1])
            all_y.extend(self.spatial_bins['y'][self.spatial_bins['flag'] == 1])
        
        if self.voronoi_data is not None:
            all_x.extend(self.voronoi_data['xNode'])
            all_y.extend(self.voronoi_data['yNode'])
        
        x_range = [np.min(all_x), np.max(all_x)]
        y_range = [np.min(all_y), np.max(all_y)]
        
        # Add margins
        x_margin = (x_range[1] - x_range[0]) * 0.1
        y_margin = (y_range[1] - y_range[0]) * 0.1
        x_range = [x_range[0] - x_margin, x_range[1] + x_margin]
        y_range = [y_range[0] - y_margin, y_range[1] + y_margin]
        
        # Update all axes independently
        for row in range(1, 3):
            for col in range(1, n_stages + 1):
                # X axis
                fig.update_xaxes(
                    title_text="X [arcsec]",
                    scaleanchor=None,  # Independent axes
                    constrain="domain",
                    range=x_range,
                    row=row, col=col
                )
                # Y axis
                fig.update_yaxes(
                    title_text="Y [arcsec]",
                    scaleanchor=f"x{(row-1)*n_stages + col}",  # Lock aspect ratio per subplot
                    scaleratio=1,
                    constrain="domain",
                    range=y_range,
                    row=row, col=col
                )
        
        # ===== ADD CUSTOM ANNOTATIONS FOR TITLES =====
        
        stage_names = []
        stage_names.append(f"Stage 1: Original Spaxels<br>(N={len(self.cube['x'])})")
        
        if self.spatial_bins is not None:
            n_valid_spatial = np.sum(self.spatial_bins['flag'] == 1)
            stage_names.append(f"Stage 2: Spatial Binning {self.configs['SPBIN_SIZE_EXGAL']:.1f}\"<br>(N={n_valid_spatial} bins)")
        
        if self.voronoi_data is not None:
            n_vor_bins = len(np.unique(self.voronoi_data['binNum'][self.voronoi_data['binNum'] >= 0]))
            stage_names.append(f"Stage 3: Voronoi SNR={self.configs['TARGET_SNR']:.0f}<br>(N={n_vor_bins} bins)")
        
        # FIXED: Add annotations with correct xref format
        annotations = []
        
        for col_idx, stage_name in enumerate(stage_names):
            # Calculate subplot position (0 to 1 scale)
            subplot_width = 1.0 / n_stages
            subplot_center_x = (col_idx + 0.5) * subplot_width
            
            # FLUX row title (top row)
            annotations.append(
                dict(
                    text=f"<b>{stage_name}</b><br>FLUX",
                    xref="paper",  # Use paper coordinates
                    yref="paper",
                    x=subplot_center_x,
                    y=0.96,  # Just below the main title
                    xanchor='center',
                    yanchor='bottom',
                    showarrow=False,
                    font=dict(size=11)
                )
            )
            
            # SNR row title (bottom row)
            annotations.append(
                dict(
                    text=f"<b>{stage_name}</b><br>SNR",
                    xref="paper",
                    yref="paper",
                    x=subplot_center_x,
                    y=0.44,  # Middle of figure (between rows)
                    xanchor='center',
                    yanchor='bottom',
                    showarrow=False,
                    font=dict(size=11)
                )
            )
        
        # ===== UPDATE LAYOUT =====
        
        fig.update_layout(
            title=dict(
                text=f"<b>Patch {self.patch_id} - {self.patch_class} (z={self.z:.4f})</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=16),
                y=0.99,
                yanchor='top'
            ),
            height=800,
            showlegend=False,
            hovermode='closest',
            annotations=annotations
        )
        
        # Save HTML
        output_html = self.figdir / f"patch_{self.patch_id:04d}_diagnostic_interactive.html"
        
        # Create a configuration for better interactivity
        config = {
            'scrollZoom': True,  # Enable scroll to zoom
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'patch_{self.patch_id:04d}_diagnostic',
                'height': 1200,
                'width': 1600,
                'scale': 2
            }
        }
        
        fig.write_html(str(output_html), config=config)
        print(f"\nInteractive plot saved: {output_html}")
        
        # Show in browser
        fig.show(config=config)
        
        print("\nInteractive controls:")
        print("  - ZOOM: Scroll wheel or drag box")
        print("  - PAN: Click and drag")
        print("  - RESET: Double-click")
        print("  - SCALE: Click colorbar and drag to adjust limits")
        print("  - Each panel has INDEPENDENT zoom/pan")
        
        return str(output_html)



    def run(self):
        """Run the complete preview pipeline."""
        print(f"\n{'#'*70}")
        print("APS CUBE PREVIEW - INTERACTIVE PLOTLY DIAGNOSTIC TOOL")
        print(f"{'#'*70}\n")
        
        # Step 1: Load patch information
        self.load_patch_info()
        
        # Step 2: Load configuration
        self.load_config()
        
        # Step 3: Prepare cube data
        self.prepare_cube_data()
        
        # Step 4: Apply spatial binning (if configured)
        self.apply_spatial_binning()
        
        # Step 5: Apply Voronoi binning (if configured)
        self.apply_voronoi_binning()
        
        # Step 6: Create diagnostic plot
        output_file = self.create_diagnostic_plot_plotly()
        
        print(f"\n{'#'*70}")
        print("PREVIEW COMPLETE")
        print(f"Interactive HTML plot opened in browser")
        print(f"{'#'*70}\n")
        
        return output_file


def main():
    """Main entry point for command-line usage."""
    
    parser = build_common_parser(
        description="APS Cube Preview - Interactive Plotly diagnostic tool",
        groups=["target_selection", "wavelength", "l1_processing",
                "caldirs", "output"],
        # this tool has no --headname/--overwrite (it's an interactive
        # preview, not a pipeline product) and no --area/--mask_areas
        # (patch-based, like the rest of the IFU family) — no
        # "spatial_selection" group above, and headname/overwrite excluded
        # from "output" below.
        exclude=["headname", "overwrite"],
        overrides={
            # this script's own wording/type differs from canonical,
            # preserved verbatim rather than silently switched
            "infiles": {"type": str, "help": "Input L1 FITS files"},
            "vacuum": {"default": False},
            "join_arms": {"default": True},
        },
        extra_args=[
            (("--patch_file",), dict(type=str, required=True,
                help="Path to patch file (FITS or ASCII)")),
            (("--patch_id",), dict(type=int, required=True,
                help="Patch ID to process")),
            (("--config_file",), dict(type=str, required=True,
                help="Path to configuration JSON file")),
        ],
    )

    args = parser.parse_args()

    # resolve_common_args adds real l1_fileinfo-based normalization,
    # the arms_ratio length assert, and the join_arms<2 correction — all
    # previously entirely absent here (this tool never called
    # l1_fileinfo() despite importing it). Disclosed behavior addition,
    # not a silent change: infiles/wlranges/arms_ratio are now validated
    # and reordered exactly like every other pipeline script.
    resolved = resolve_common_args(args)
    wlranges, arms_ratio = resolved.wlranges, resolved.arms_ratio
    aps_ids, targsrvy, targclass, mask_aps_ids = (
        resolved.aps_ids, resolved.targsrvy, resolved.targclass, resolved.mask_aps_ids)

    # This tool has no --headname, so there's no per-run log file to write
    # to (print_args() silently no-ops without one) — print to screen only.
    # Previously this tool never logged its resolved arguments at all.
    print_args(args, module="CubePreview", version=aps_constants.__aps_version__,
               screen_only=True)

    # Create and run preview
    preview = CubePreview(
        infiles=args.infiles,
        patch_file=args.patch_file,
        patch_id=args.patch_id,
        config_file=args.config_file,
        outpath=args.outpath,
        wlranges=wlranges,
        aps_ids=aps_ids,
        targsrvy=targsrvy,
        targclass=targclass,
        mask_aps_ids=mask_aps_ids,
        sens_corr=args.sens_corr,
        mask_gaps=args.mask_gaps,
        safe_mask_gaps=args.safe_mask_gaps,
        vacuum=args.vacuum,
        tellurics=args.tellurics,
        fill_gap=args.fill_gap,
        arms_ratio=arms_ratio,
        join_arms=args.join_arms,
        catdir=args.catdir,
        caldir=args.caldir,
        configdir=args.configdir
    )
    
    preview.run()


if __name__ == '__main__':
    main()