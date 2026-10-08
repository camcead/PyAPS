# Enhanced APS IFU MapViewer Package
# This makes the directory a proper Python package and allows imports

__version__ = "3.0-Enhanced"
__author__ = "Alireza Molaeinezhad (Enhanced)"
__email__ = "amolaei@ast.cam.ac.uk"

# Import the main classes for easy access
try:
    from .aps_Mapviewer import EnhancedMapviewer, enhanced_mapviewer_worker
    __all__ = ['EnhancedMapviewer', 'enhanced_mapviewer_worker']
except ImportError:
    # If relative imports fail, we're probably running as a script
    __all__ = []

print(f"Enhanced APS IFU MapViewer v{__version__} package loaded")