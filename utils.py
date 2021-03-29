from pathlib import Path
import coloredlogs
import logging
import pprint

RANDOM_SEED = 2

def create_folder(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)


# set up logging
logger = logging.getLogger(__name__)


def setup_logger(level='INFO'):
    global logger
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt='[%(levelname)s] %(message)s',
        level_styles=coloredlogs.parse_encoded_styles('spam=22;debug=28;verbose=34;info=226;notice=220;warning=202;success=118,bold;error=124;critical=background=red'))

setup_logger()

def enable_debug():
    global logger
    setup_logger(level='DEBUG')

def get_printer(depth=2):
    return pprint.PrettyPrinter(depth=depth)

from colour import Color
from matplotlib.colors import LinearSegmentedColormap

ramp_colors = ['#ff0000','#ffff00','#00ff00']
color_ramp = LinearSegmentedColormap.from_list( 'driveability', [ Color( c1 ).rgb for c1 in ramp_colors ] )
