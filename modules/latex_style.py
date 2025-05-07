"""
LaTeX-style plotting configuration for academic publications.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np

def set_latex_style(use_tex=False):
    """
    Configure matplotlib to use a LaTeX-like style suitable for academic publications.
    
    Args:
        use_tex: Whether to use actual LaTeX rendering (requires LaTeX installation).
                 If False, uses LaTeX-like styling without requiring LaTeX.
    """
    # Check if we can use LaTeX
    if use_tex:
        try:
            # Test if LaTeX is available
            plt.rcParams["text.usetex"] = True
            plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
            # Create a small test figure to see if LaTeX works
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, r"$\alpha$")
            plt.close(fig)
            print("LaTeX rendering enabled.")
        except Exception as e:
            print(f"LaTeX rendering failed, falling back to LaTeX-like style: {e}")
            plt.rcParams["text.usetex"] = False
            use_tex = False
    
    # Font settings
    if use_tex:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman"]
    else:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Times", "serif"]
    
    # Figure size and DPI
    plt.rcParams["figure.figsize"] = (6, 4.5)  # Default figure size (4:3 aspect ratio)
    plt.rcParams["figure.dpi"] = 300  # High resolution for publication
    plt.rcParams["savefig.dpi"] = 300  # High resolution for saved figures
    plt.rcParams["savefig.format"] = "pdf"  # PDF is preferred for publications
    
    # Line and marker styles
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["lines.markeredgewidth"] = 0.8
    
    # Font sizes
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 13
    
    # Axes and grid
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"
    
    # Tick marks
    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 3.5
    plt.rcParams["xtick.minor.size"] = 2.0
    plt.rcParams["ytick.minor.size"] = 2.0
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.minor.width"] = 0.6
    plt.rcParams["ytick.minor.width"] = 0.6
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    
    # Legend
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.9
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.fancybox"] = False
    
    # Set a scientific color scheme suitable for publications
    # These colors are colorblind-friendly and print well in grayscale
    colors = [
        "#0173B2",  # blue
        "#DE8F05",  # orange
        "#029E73",  # green
        "#D55E00",  # vermillion
        "#CC78BC",  # purple
        "#CA9161",  # brown
        "#FBAFE4",  # pink
        "#949494",  # gray
    ]
    
    plt.rcParams["axes.prop_cycle"] = cycler("color", colors)
    
    # Figure layout
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.autolayout"] = False
    
    return use_tex

def format_axis_in_latex_style(ax):
    """
    Apply additional LaTeX-style formatting to a specific axis.
    
    Args:
        ax: The matplotlib axis to format
    """
    # Set spines
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    
    # Set tick parameters
    ax.tick_params(which="major", length=3.5, width=0.8, direction="in", top=True, right=True)
    ax.tick_params(which="minor", length=2.0, width=0.6, direction="in", top=True, right=True)
    
    # Enable minor ticks
    ax.minorticks_on()
    
    # Set grid
    ax.grid(True, linestyle="--", alpha=0.3)
    
    return ax

def save_figure_for_publication(fig, filename, width=None, height=None, dpi=300, formats=None):
    """
    Save a figure in formats suitable for publication.
    
    Args:
        fig: The matplotlib figure to save
        filename: Base filename without extension
        width: Width in inches (if None, uses current figure width)
        height: Height in inches (if None, uses current figure height)
        dpi: Resolution in dots per inch
        formats: List of formats to save (default: ['pdf', 'png'])
    """
    if formats is None:
        formats = ['pdf', 'png']
    
    # Adjust figure size if specified
    if width is not None or height is not None:
        current_width, current_height = fig.get_size_inches()
        width = width if width is not None else current_width
        height = height if height is not None else current_height
        fig.set_size_inches(width, height)
    
    # Save in each format
    for fmt in formats:
        output_filename = f"{filename}.{fmt}"
        fig.savefig(output_filename, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure as {output_filename}")