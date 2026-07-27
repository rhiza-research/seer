"""Plotting helpers for a solved seer model instance."""

import pandas as pd
from matplotlib.transforms import blended_transform_factory

from generate_data import DAYS_IN_MONTH

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HOURS_PER_DAY = 24


def _month_start_hours():
    """Hour-of-year (0-indexed) at which each month starts."""
    starts = []
    hour = 0
    for days in DAYS_IN_MONTH:
        starts.append(hour)
        hour += days * HOURS_PER_DAY
    return starts


def _label_months_and_years(ax, years, hours_per_year):
    """Add month tick labels plus a bold year label under each year's block."""
    month_start_hours = _month_start_hours()

    month_ticks = []
    month_tick_labels = []
    offset = 0
    for year in years:
        for month_index, start_hour in enumerate(month_start_hours):
            month_ticks.append(offset + start_hour)
            month_tick_labels.append(MONTH_LABELS[month_index])

        ax.axvline(offset, color="grey", linewidth=0.5, linestyle="--")
        ax.text(
            offset + hours_per_year / 2, -0.12, str(year),
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", va="top", fontsize=9, fontweight="bold",
        )
        offset += hours_per_year

    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_tick_labels, rotation=90, fontsize=7)
    ax.set_xlim(0, offset)


def plot_demand(ax, instance, zone=None):
    """Plot full-resolution demand as a line, labeling months and years only."""
    years = list(instance.years)
    hours = list(instance.hours)
    hours_per_year = len(hours)
    zones = [zone] if zone else list(instance.zones)

    demand = instance.demand.extract_values()
    df = pd.DataFrame(
        [(z, y, h, v) for (z, y, h), v in demand.items() if z in zones],
        columns=["zone", "year", "hour", "value"],
    )
    series = df.groupby(["year", "hour"])["value"].sum().reindex(
        pd.MultiIndex.from_product([years, hours], names=["year", "hour"])
    )

    ax.plot(range(len(series)), series.values, linewidth=0.7)
    _label_months_and_years(ax, years, hours_per_year)
    ax.set_ylabel("Demand (MW)")
    ax.set_title(f"Demand ({zone if zone else 'all zones'})")


def plot_generation(ax, instance, zone=None):
    """Plot a stacked area of generation output, grouped by generator tech."""
    generator_tech = instance.generator_tech.extract_values()
    generator_zone = instance.generator_zone.extract_values()

    years = list(instance.years)
    hours = list(instance.hours)
    hours_per_year = len(hours)

    output = instance.generation_output.extract_values()
    df = pd.DataFrame(
        [(gen, y, h, v) for (gen, y, h), v in output.items()],
        columns=["generator", "year", "hour", "value"],
    )
    df["tech"] = df["generator"].map(generator_tech)
    if zone:
        df = df[df["generator"].map(generator_zone) == zone]

    pivot = df.groupby(["year", "hour", "tech"])["value"].sum().unstack("tech").reindex(
        pd.MultiIndex.from_product([years, hours], names=["year", "hour"])
    ).fillna(0.0)

    techs = list(pivot.columns)
    x = range(len(pivot))
    ax.stackplot(x, *[pivot[tech] for tech in techs], labels=techs)
    _label_months_and_years(ax, years, hours_per_year)
    ax.set_ylabel("Generation (MW)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(f"Generation by tech ({zone if zone else 'all zones'})")


def plot_storage(ax, instance, zone=None):
    """Plot total state of charge as a line, labeling months and years only."""
    storage_zone = instance.storage_zone.extract_values()

    years = list(instance.years)
    hours = list(instance.hours)
    hours_per_year = len(hours)

    soc = instance.soc.extract_values()
    df = pd.DataFrame(
        [(s, y, h, v) for (s, y, h), v in soc.items() if zone is None or storage_zone[s] == zone],
        columns=["storage", "year", "hour", "value"],
    )
    series = df.groupby(["year", "hour"])["value"].sum().reindex(
        pd.MultiIndex.from_product([years, hours], names=["year", "hour"])
    )

    ax.plot(range(len(series)), series.values, linewidth=0.7, color="tab:green")
    _label_months_and_years(ax, years, hours_per_year)
    ax.set_ylabel("State of charge (MWh)")
    ax.set_title(f"Storage SOC ({zone if zone else 'all zones'})")
