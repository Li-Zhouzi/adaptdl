"""
Shared plotting configuration for all plotting scripts.
This ensures consistent figure sizes, fonts, and styling across all plots.
"""
import matplotlib.pyplot as plt
from pathlib import Path

# Get the directory where this config file is located
CONFIG_DIR = Path(__file__).parent

# Path to the style file
STYLE_FILE = CONFIG_DIR / "plot_style.mplstyle"


def apply_plot_style():
    """
    Apply the consistent plotting style to matplotlib.
    Call this function at the beginning of your plotting script.
    """
    if STYLE_FILE.exists():
        plt.style.use(str(STYLE_FILE))
    else:
        # Fallback to manual configuration if style file doesn't exist
        plt.rcParams.update({
            'figure.figsize': (10, 8),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.size': 16,
            'axes.labelsize': 24,
            'axes.titlesize': 24,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 24,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'axes.linewidth': 1.5,
            'axes.grid': True,
            'grid.alpha': 0.3,
        })


def create_figure(figsize=None):
    """
    Create a figure with consistent styling.
    
    Args:
        figsize: Optional tuple (width, height) in inches. 
                 If None, uses the default from style (10, 8).
    
    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    if figsize is None:
        figsize = (10, 8)
    
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# Common color schemes for consistency across plots
COLORS = {
    'boa_constrictor': 'blue',
    'boa_constrictor_v2': 'red',
    'pollux': 'orange',
    'pollux_autoscaling': 'green',
}

# Common plotting styles for different policies
POLICY_STYLES = {
    "fiximperfectCapNone": {
        "color": COLORS['boa_constrictor'],
        "label": "BOA Constrictor"
    },
    "fiximperfectversion2CapNone": {
        "color": COLORS['boa_constrictor_v2'],
        "label": "BOA Constrictor: Version 2"
    },
    "sia": {
        "color": COLORS['pollux'],
        "label": "Pollux"
    },
    "polluximperfect": {
        "color": COLORS['pollux_autoscaling'],
        "label": "Pollux w/ autoscaling"
    },
}

