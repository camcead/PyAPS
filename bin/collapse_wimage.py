#!/usr/bin/env python
import sys
from astropy.io import fits

def collapse_wimage(input_file, output_file):

    """
    Merge the primary and second extensions of a FITS file into a single extension.

    Parameters:
        input_file (str): Path to the input FITS file.
        output_file (str): Path to the output FITS file.
    
    example:
    ./collapse_wimage.py input.fits output.fits
    
    """


    # Open the input FITS file
    with fits.open(input_file) as hdul:
        # Check if there are exactly 2 extensions
        if len(hdul) != 2:
            print(f"Warning: Input file has {len(hdul)} extensions.")
            print("we only use the primary header and the first extensions")

        # Combine header from primary and second extension
        combined_header = hdul[0].header.copy()
        combined_header.extend(hdul[1].header.items())

        # Create a new HDU with combined header and image data from second extension
        new_hdu = fits.PrimaryHDU(data=hdul[1].data, header=combined_header)

        # Write the new HDU to the output file
        new_hdu.writeto(output_file, overwrite=True)

    print(f"Merged FITS file created: {output_file}")

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python merge_fits.py input.fits output.fits")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    collapse_wimage(input_file, output_file)