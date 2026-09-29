#!/usr/bin/env python3
"""
Comprehensive FITS file inspector
Usage: python inspect_fits_detailed.py <path_to_fits_file>
"""

import sys
import numpy as np
from astropy.io import fits

def inspect_fits_comprehensive(filepath):
    """Detailed inspection of FITS file with statistics"""
    
    try:
        with fits.open(filepath) as hdul:
            print(f"\n{'='*100}")
            print(f"FITS File: {filepath}")
            print(f"{'='*100}\n")
            
            print(f"Total extensions: {len(hdul)}\n")
            
            for ext_num, hdu in enumerate(hdul):
                print(f"\n{'='*100}")
                print(f"EXTENSION {ext_num}: {hdu.name}")
                print(f"{'='*100}")
                print(f"Type: {type(hdu).__name__}")
                
                # Header information
                print(f"\nHeader keywords: {len(hdu.header)} entries")
                key_cards = [(k, v, hdu.header.comments[k]) for k, v in hdu.header.items() 
                            if k not in ['COMMENT', 'HISTORY', '']]
                if key_cards and len(key_cards) <= 20:
                    print("\nKey header cards:")
                    for k, v, c in key_cards[:20]:
                        comment = f" / {c}" if c else ""
                        print(f"  {k:15s} = {str(v):30s}{comment}")
                    if len(key_cards) > 20:
                        print(f"  ... and {len(key_cards) - 20} more")
                
                # Data inspection
                if isinstance(hdu, fits.PrimaryHDU):
                    if hdu.data is not None:
                        print(f"\nPrimary HDU contains image data")
                        print(f"  Shape: {hdu.data.shape}")
                        print(f"  Data type: {hdu.data.dtype}")
                        if hdu.data.size > 0:
                            print(f"  Min: {np.nanmin(hdu.data):.6e}")
                            print(f"  Max: {np.nanmax(hdu.data):.6e}")
                            print(f"  Mean: {np.nanmean(hdu.data):.6e}")
                            print(f"  Median: {np.nanmedian(hdu.data):.6e}")
                            finite_count = np.sum(np.isfinite(hdu.data))
                            print(f"  Finite values: {finite_count}/{hdu.data.size} ({100*finite_count/hdu.data.size:.1f}%)")
                    else:
                        print(f"\nPrimary HDU (header only, no data)")
                
                # Table data inspection
                elif isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    columns = hdu.columns
                    nrows = hdu.data.shape[0] if hdu.data is not None else 0
                    
                    print(f"\nTable dimensions:")
                    print(f"  Rows: {nrows:,}")
                    print(f"  Columns: {len(columns)}")
                    
                    # Check for problematic column names
                    problematic = []
                    for col in columns:
                        underscore_count = col.name.count('_')
                        # Check for repeated patterns like "9530.60_9530.60"
                        parts = col.name.split('_')
                        if len(parts) >= 2:
                            # Look for numeric duplicates
                            for i in range(len(parts)-1):
                                try:
                                    if float(parts[i]) == float(parts[i+1]):
                                        problematic.append((col.name, "DUPLICATE_NUMBER"))
                                        break
                                except ValueError:
                                    pass
                        if underscore_count > 3:
                            if col.name not in [p[0] for p in problematic]:
                                problematic.append((col.name, "TOO_MANY_UNDERSCORES"))
                    
                    if problematic:
                        print(f"\n  WARNING: {len(problematic)} potentially problematic column names found!")
                        for name, reason in problematic[:5]:
                            print(f"    - {name} ({reason})")
                        if len(problematic) > 5:
                            print(f"    ... and {len(problematic) - 5} more")
                    
                    print(f"\n{'Column Analysis':^100}")
                    print(f"{'-'*100}")
                    print(f"{'#':<5} {'Column Name':<50} {'Format':<8} {'Unit':<12} {'Elements':<10}")
                    print(f"{'-'*100}")
                    
                    for j, col in enumerate(columns, 1):
                        unit = col.unit if col.unit else ''
                        
                        # Get number of elements
                        if hdu.data is not None and nrows > 0:
                            col_data = hdu.data[col.name]
                            if col_data.ndim > 1:
                                n_elem = f"{nrows}x{col_data.shape[1]}"
                            else:
                                n_elem = str(nrows)
                        else:
                            n_elem = "0"
                        
                        print(f"{j:<5} {col.name:<50} {col.format:<8} {unit:<12} {n_elem:<10}")
                    
                    print(f"{'-'*100}\n")
                    
                    # Detailed statistics for each column
                    if hdu.data is not None and nrows > 0:
                        print(f"\n{'Column Statistics':^100}")
                        print(f"{'-'*100}")
                        
                        for j, col in enumerate(columns, 1):
                            col_data = hdu.data[col.name]
                            print(f"\n{j}. {col.name}")
                            print(f"   Format: {col.format}, Unit: {col.unit if col.unit else 'None'}")
                            
                            # Handle different data types
                            if col_data.dtype.kind in ['i', 'u', 'f']:  # Integer or float
                                if col_data.ndim == 1:
                                    finite_mask = np.isfinite(col_data)
                                    n_finite = np.sum(finite_mask)
                                    n_nan = np.sum(np.isnan(col_data))
                                    n_inf = np.sum(np.isinf(col_data))
                                    
                                    print(f"   Shape: ({nrows},)")
                                    print(f"   Data type: {col_data.dtype}")
                                    print(f"   Finite: {n_finite}/{nrows} ({100*n_finite/nrows:.1f}%)")
                                    if n_nan > 0:
                                        print(f"   NaN: {n_nan} ({100*n_nan/nrows:.1f}%)")
                                    if n_inf > 0:
                                        print(f"   Inf: {n_inf} ({100*n_inf/nrows:.1f}%)")
                                    
                                    if n_finite > 0:
                                        finite_data = col_data[finite_mask]
                                        print(f"   Min: {np.min(finite_data):.6e}")
                                        print(f"   Max: {np.max(finite_data):.6e}")
                                        print(f"   Mean: {np.mean(finite_data):.6e}")
                                        print(f"   Median: {np.median(finite_data):.6e}")
                                        print(f"   Std: {np.std(finite_data):.6e}")
                                else:
                                    print(f"   Shape: {col_data.shape}")
                                    print(f"   Multidimensional array - showing first row stats")
                                    first_row = col_data[0]
                                    finite_mask = np.isfinite(first_row)
                                    if np.any(finite_mask):
                                        print(f"   First row min: {np.min(first_row[finite_mask]):.6e}")
                                        print(f"   First row max: {np.max(first_row[finite_mask]):.6e}")
                            
                            elif col_data.dtype.kind in ['S', 'U', 'O']:  # String or object
                                print(f"   Data type: String/Object")
                                unique_vals = np.unique(col_data)
                                print(f"   Unique values: {len(unique_vals)}")
                                if len(unique_vals) <= 10:
                                    print(f"   Values: {', '.join(str(v) for v in unique_vals)}")
                                else:
                                    print(f"   Sample values: {', '.join(str(v) for v in unique_vals[:10])}...")
                            
                            else:
                                print(f"   Data type: {col_data.dtype} (other)")
                            
                            # Stop after 20 columns to avoid excessive output
                            if j >= 20:
                                remaining = len(columns) - j
                                if remaining > 0:
                                    print(f"\n   ... {remaining} more columns (truncated for brevity)")
                                break
                
                print()
                
    except Exception as e:
        print(f"Error reading FITS file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_fits_detailed.py <path_to_fits_file>")
        sys.exit(1)
    
    inspect_fits_comprehensive(sys.argv[1])