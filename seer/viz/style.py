"""Shared matplotlib/seaborn style for ad hoc exploration plots (distinct from the
output-table-specific TECH_COLORS/tech_color() in seer/viz/plot.py)."""
import seaborn as sns

LINE_PALETTE = sns.color_palette("husl", 8)


def set_style():
    """Call once per script, before plotting."""
    sns.set_theme(style="whitegrid", palette=LINE_PALETTE)


def line_color(i):
    """i-th color in the shared line palette, cycling if i >= 8."""
    return LINE_PALETTE[i % len(LINE_PALETTE)]
