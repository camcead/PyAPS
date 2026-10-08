import numpy as np
import json
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import PyAPS
APSVERS = PyAPS.__version__

##############################################
## To pass None keyword as command line argument
## Update (9 september 2020): It also expand path from variable mode into the expanded mode
def none_or_str(value):
    if value == 'None':
        return None
    return os.path.expanduser(os.path.expandvars(value))

##############################################
def binaryToDecimal(n): 
    return np.int32(int(n, 2))

##############################################

"""
read the aps_flag file (in json format) and return the binary value
input:
    flag_file: a json file contains all aps flags and corresponding values
output:
    a decimal value

Note: Flags are read from top to bottom in the json file (e.g. redrock represents the lowest order bit)
"""

def aps_flags_btd(flag_file):
    with open(flag_file) as json_dict:
        btd_dict = json.load(json_dict)


    val_list = [str(btd_dict[flag]) for flag in btd_dict]
    key_list = [flag for flag in btd_dict]
    decimal = binaryToDecimal(''.join(val_list)[::-1])

    return decimal
##############################################

"""
read the aps_flag file (in json format) as template  and decimal value as input and return the bitmasks
input:
    flag_file: a json file contains all aps flags and corresponding values
    decimal: decimal value, provided by user
output:
    aps_flag in python dictionary. Ready to be feed into APS

"""

def aps_flags_dtb(flag_file, decimal):

    assert isinstance(decimal, int), 'decimal must be integer'

    with open(flag_file) as json_dict:
        btd_dict = json.load(json_dict)

    val_list = [str(btd_dict[flag]) for flag in btd_dict]
    key_list = [flag for flag in btd_dict]


    binary = ("{:0>"+str(len(key_list))+"d}").format(int(bin(decimal)[2:]))[::-1]


    for index, flag in enumerate(btd_dict):
        btd_dict[flag] = binary[index]

    return btd_dict

########################################################################


def aps_flags_worker(options=None):

    parser = argparse.ArgumentParser(description="Handling APS_flags")
    
    parser.add_argument("--flag_file", type=none_or_str, default=None,
        required=False, help="APSFLAGS in json format, as input/template")

    parser.add_argument('--decimal', help='Number of threads for the fits',
        type=none_or_str, default='None')


    ## Check if any command-line argument has been passed to the module. It counts the number of system arguments to check this.
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        print('---------------------------------------------------------------------------------')
        print('No command-line argument has been passed to this module. Running DEMO/DEBUG mode!')
        print('---------------------------------------------------------------------------------')

        args = parser.parse_args(options)

    if args.decimal is None:
        decimal =  aps_flags_btd(args.flag_file)
        print('Reading APS_FLAGS file: %s' %(args.flag_file))
        print('APS-compatable decimal code is: %d' %(decimal))
        return decimal
    else:
        decimal=int(args.decimal)
        ### Convert decimal (input) into the aps_flags (return the aps_flags dict)
        aps_flag_dict =  aps_flags_dtb(args.flag_file, decimal)
        print('Reading APS_FLAGS file (as template): %s' %(args.flag_file))
        print('Returning the aps_flag dictionary (for internal APS use only!!!)')
        return aps_flag_dict

##############################################
if __name__ == '__main__':
    # DEMO settings: edit for your setup. Replace the <PYAPS_DATA>, <PYAPS_DIR>, <night>, <runid>, <obid>
    # markers below with your own locations and identifiers (no machine paths belong in this repository).


    debug_demo= ['--flag_file', '<PYAPS_DIR>/configs/APS_FLAGS.json',
                '--decimal', 'None']

    ## If no command-line argument has been passed to this module, it use the debug list as input and runs in the DEMO/DEBUG mode!
    aps_flags_worker(options=debug_demo)


    #Some useful examples for WEAVE developers

    # flag_file= './APS_FLAGS.json'
    # decimal=1

    # ### Convert aps_flags into decimal
    # decimal= aps_flags_btd(flag_file)

    # ### Convert decimal (input) into the aps_flags (return the aps_flags dict)
    # flags = aps_flags_dtb(flag_file, decimal)


