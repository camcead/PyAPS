import sys
import numbers



# ==============================================================================
#                          P R E T T Y   O U T P U T
# ==============================================================================
""" A collection of functions to generate the pretty output."""

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 80, color = 'g'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        color       - Optional  : color identifier (Str)
    """
    if   color == 'y': color = '\033[43m'
    elif color == 'k': color = '\033[40m'
    elif color == 'r': color = '\033[41m'
    elif color == 'g': color = '\033[42m'
    elif color == 'b': color = '\033[44m'
    elif color == 'm': color = '\033[45m'
    elif color == 'c': color = '\033[46m'

    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = color + ' '*filledLength + '\033[49m' + ' '*(barLength - filledLength -1)
    sys.stdout.write('\r%s |%s| %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()

def prettyOutput_Running(outputlabel):
    sys.stdout.write("\r [ "+'\033[0;37m'+"RUNNING "+'\033[0;39m'+"] "+outputlabel)
    sys.stdout.flush(); print("")

def prettyOutput_Done(outputlabel, progressbar=False):
    if progressbar == True:
        sys.stdout.write("\033[K")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r [ "+'\033[0;32m'+"DONE    "+'\033[0;39m'+"] "+outputlabel)
    sys.stdout.flush(); print("")

def prettyOutput_Warning(outputlabel, progressbar=False):
    if progressbar == True:
        sys.stdout.write("\033[K")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r [ "+'\033[0;33m'+"WARNING "+'\033[0;39m'+"] "+outputlabel)
    sys.stdout.flush(); print("")

def prettyOutput_Info(outputlabel, progressbar=False):
    if progressbar == True:
        sys.stdout.write("\033[K")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r [ "+'\033[0;33m'+"INFO "+'\033[0;39m'+"] "+outputlabel)
    sys.stdout.flush(); print("")



def prettyOutput_Failed(outputlabel, progressbar=False):
    if progressbar == True:
        sys.stdout.write("\033[K")
    sys.stdout.write("\033[F"); sys.stdout.write("\033[K")
    sys.stdout.write("\r\r [ "+'\033[0;31m'+"FAILED  "+'\033[0;39m'+"] "+outputlabel)
    sys.stdout.flush(); print("")

def prettyOutput_Error(outputlabel, progressbar=False):
    """Same as prettyOutput_Failed. IFUExGalEMIPPXF calls this name in its error handlers; it did not exist, so the handler itself raised
    AttributeError (found 2026-11-13 on the serial path of the stellar re-fit, which a pool of 40 workers never takes)."""
    prettyOutput_Failed(outputlabel, progressbar)

def prettyOutput_DonePrefix():
    return(" [ "+'\033[0;32m'+"DONE    "+'\033[0;39m'+"] ")

def prettyOutput_WarningPrefix():
    return(" [ "+'\033[0;33m'+"WARNING "+'\033[0;39m'+"] ")

def prettyOutput_FailedPrefix():
    return(" [ "+'\033[0;31m'+"FAILED  "+'\033[0;39m'+"] ")


# ==============================================================================
#                          p P X F   P E N A L I S A T I O N
# ==============================================================================

def ppxf_bias(configs):
    """
    The penalisation ("bias") of the stellar-kinematics pPXF fit, from the config key ``BIAS_PPXF``.

    pPXF penalises the Gauss-Hermite moments h3, h4, ... towards zero; the bias keyword sets how strongly (Cappellari & Emsellem 2004,
    section 3.4). A large bias pulls noisy h3/h4 to a flat zero map, a small one lets them vary (and become noisy). Absent, ``None``,
    ``"None"``, ``"default"`` or an empty string means pPXF's own default (the value that was always used before this key existed),
    so existing configs give identical results. Otherwise a number >= 0 (``0`` switches the penalisation off).

    Returns None (pPXF default) or a float. Raises ValueError for anything else, so a typo stops the run at the start instead of
    giving NaN results bin by bin. The Monte-Carlo error simulations keep ``bias=0`` as before.
    """
    value = configs.get('BIAS_PPXF', None)
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ('', 'none', 'default', 'null'):
            return None
        try:
            value = float(value)
        except ValueError:
            raise ValueError("BIAS_PPXF must be a number >= 0 or 'None' (pPXF default), got %r" % (configs['BIAS_PPXF'],))
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or value != value or value < 0:
        raise ValueError("BIAS_PPXF must be a number >= 0 or 'None' (pPXF default), got %r" % (configs['BIAS_PPXF'],))
    return float(value)
