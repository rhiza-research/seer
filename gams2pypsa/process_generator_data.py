import pandas as pd
import numpy as np

def process_generator_data(data_path="data/Gambia/"):
    import pdb; pdb.set_trace()
    input_data = load_input_data(data_path)
    all_generators = input_data["all_generators"]
    default_gen_params = input_data["default_gen_params"]
    custom_gen_params = input_data["custom_gen_params"]
    default_gen_capex = input_data["default_gen_capex"]
    custom_gen_capex = input_data["custom_gen_capex"]
    fuel_prices = input_data["fuel_prices"]

    # ----------------------------
    # 1️⃣ Build generator constants
    # ----------------------------
    # Merge default parameters into generators (by tech/fuel)
    gens = all_generators.merge(default_gen_params, on=["zone", "tech", "fuel"], how="left")
    # Merge custom parameters (by generator name) to override defaults
    gens = gens.merge(
        custom_gen_params[["generator", "parameters", "value"]],
        on=["generator", "parameters"],
        how="left",
        suffixes=("_default", "_custom")
    )
    # Use custom if exists
    gens["value_final"] = gens["value_custom"].combine_first(gens["value_default"])
    # Pivot so each parameter is a column
    generator_constants = gens.pivot_table(
        index=["generator", "zone", "tech", "fuel"],
        columns="parameters",
        values="value_final"
    ).reset_index()
    
    # Create standard columns PyPSA expects
    generator_constants["expandable"] = generator_constants.get("UnitSize", 0) > 0
    generator_constants["annual_build_limit"] = generator_constants.get("UnitSize", pd.NA).where(generator_constants["expandable"], pd.NA)
    # Rename ramp rates and heat rate to PyPSA conventions
    generator_constants.rename(columns={
        "RampUpRate": "ramp_up_rate",
        "RampDnRate": "ramp_down_rate",
        "HeatRate": "heat_rate",
        "StYr": "start_year",
        "RetrYr": "retirement_year",
        "UnitSize": "unit_size"
    }, inplace=True)
    # Select relevant columns
    generator_constants = generator_constants[
        ["generator", "zone", "tech", "fuel", "expandable", "annual_build_limit",
         "ramp_up_rate", "ramp_down_rate", "heat_rate", "start_year", "retirement_year", "unit_size"]
    ]

    # ----------------------------
    # 2️⃣ Build generator costs
    # ----------------------------
    years = sorted(default_gen_capex["year"].unique())
    cost_rows = []

    for _, gen in generator_constants.iterrows():
        # CAPEX trajectory
        # start with default by tech/fuel
        df_capex = default_gen_capex[
            (default_gen_capex["zone"] == gen.zone) &
            (default_gen_capex["tech"] == gen.tech) &
            (default_gen_capex["fuel"] == gen.fuel)
        ][["year", "value"]].set_index("year")
        # override with custom trajectories if exist
        df_custom = custom_gen_capex[custom_gen_capex["plant"] == gen.generator][["year", "value"]].set_index("year")
        capex = df_capex["value"].combine_first(df_custom["value"])
        # add row
        cost_rows.append(pd.DataFrame([capex.values], index=["capex"], columns=years, dtype=float).assign(generator=gen.generator))

        # Marginal cost from fuel price × heat rate
        if pd.notna(gen.heat_rate):
            fuel_price = fuel_prices[fuel_prices["fuel"] == gen.fuel][["year", "value"]].set_index("year")
            mc = fuel_price["value"] * gen.heat_rate
            cost_rows.append(pd.DataFrame([mc.values], index=["marginal_cost"], columns=years, dtype=float).assign(generator=gen.generator))

    generator_cost = pd.concat(cost_rows).reset_index().rename(columns={"index": "cost_type"})

    return generator_constants, generator_cost

    import pdb; pdb.set_trace()

    return all_generators, default_gen_params, custom_gen_params, default_gen_capex, custom_gen_capex, fuel_prices


def load_input_data(data_path):
    # list of all generators including name, zone, fuel
    all_generators = pd.read_csv(data_path + "gmap.csv")
    # generator parameters (by zone, tech, fuel)
    default_gen_params = pd.read_csv(data_path + "pGenDataInputDefault.csv")
    custom_gen_params = pd.read_csv(data_path + "pGenDataInputCustom.csv")
    # generator capex (by zone, tech, fuel)
    default_gen_capex = pd.read_csv(data_path + "pCapexTrajectoriesDefault.csv")
    custom_gen_capex = pd.read_csv(data_path + "pCapexTrajectoriesCustom.csv")
    # fuel prices
    fuel_prices = pd.read_csv(data_path + "pFuelPrice.csv")

    return {
        "all_generators": all_generators,
        "default_gen_params": default_gen_params,
        "custom_gen_params": custom_gen_params,
        "default_gen_capex": default_gen_capex,
        "custom_gen_capex": custom_gen_capex,
        "fuel_prices": fuel_prices,
    }

