import pandas as pd
import numpy as np

def process_generator_data(data_path="data/Gambia/"):
    input_data = load_input_data(data_path)
    all_generators = input_data["all_generators"]
    default_gen_params = input_data["default_gen_params"]
    custom_gen_params = input_data["custom_gen_params"]
    default_gen_capex = input_data["default_gen_capex"]
    custom_gen_capex = input_data["custom_gen_capex"]
    fuel_prices = input_data["fuel_prices"]

    """Setup generator constants.
    Use generator list, default parameters, and custom parameters to fill in a single dataframe of generator constants.
    """
    param_list = [
        "Capacity", "Capex", "FOMperMW", "MinLimitShare", "Overloadfactor",
        "RampDnRate", "RampUpRate", "ReserveCost", "ResLimShare", "Life",
        "HeatRate", "VOM", "RETech", "RetrYr", "StYr", "UnitSize",
        "BuildLimitperYear"
    ]
    # Initialize generator_constants dataframe
    generator_constants = pd.DataFrame(columns=["generator", "zone", "tech", "fuel"] + param_list)
    generator_constants[["generator", "zone", "tech", "fuel"]] = all_generators[["generator", "zone", "tech", "fuel"]]

    # Fill in parameters from defaults first
    for idx, gen in generator_constants.iterrows():
        for param in param_list:
            # find default row for this zone/tech/fuel/parameter
            mask = (
                (default_gen_params["zone"] == gen.zone) &
                (default_gen_params["tech"] == gen.tech) &
                (default_gen_params["fuel"] == gen.fuel) &
                (default_gen_params["parameters"] == param)
            )
            if mask.any():
                generator_constants.loc[idx, param] = default_gen_params.loc[mask, "value"].values[0]
            # override with custom parameters where available
            mask = (
                (custom_gen_params["generator"] == gen.generator) &
                (custom_gen_params["parameters"] == param)
            )
            if mask.any():
                generator_constants.loc[idx, param] = custom_gen_params.loc[mask, "value"].values[0]
    
    
    # Create standard columns PyPSA expects
    # this is a build option if buildlimitperyear is not nan
    generator_constants["expandable"] = generator_constants["BuildLimitperYear"].notna()
    # rename columns to PyPSA conventions
    generator_constants.rename(columns={
        "BuildLimitperYear": "annual_build_limit",
        "RampUpRate": "ramp_up_rate",
        "RampDnRate": "ramp_down_rate",
        "HeatRate": "heat_rate",
        "StYr": "start_year",
        "RetrYr": "retirement_year",
        "UnitSize": "unit_size"
    }, inplace=True)


    """Setup generator costs.
    Use generator constants, default capex trajectories, and custom capex trajectories to fill in a single dataframe of generator costs.
    """
    # setup cost dataframe
    years = sorted(default_gen_capex["year"].unique())
    rows = []

    for _, gen in generator_constants.iterrows():

        # capex
        row_capex = {"generator": gen.generator, "cost_type": "capex"}

        # get capex trajectory - custom if it exists, otherwise default
        if gen.generator in custom_gen_capex["plant"].values:
            capex_traj = custom_gen_capex[
                custom_gen_capex["plant"] == gen.generator
            ][["year", "value"]].set_index("year")
        else:
            capex_traj = default_gen_capex[
                (default_gen_capex["zone"] == gen.zone) &
                (default_gen_capex["tech"] == gen.tech) &
                (default_gen_capex["fuel"] == gen.fuel)
            ][["year", "value"]].set_index("year")

        capex_traj = gen.Capex * capex_traj["value"]

        for y in years:
            row_capex[y] = capex_traj.get(y, pd.NA)

        rows.append(row_capex)

        # marginal cost
        if pd.notna(gen.heat_rate):
            row_mc = {"generator": gen.generator, "cost_type": "marginal_cost"}

            fuel_price = fuel_prices[
                fuel_prices["fuel"] == gen.fuel
            ][["year", "value"]].set_index("year")

            mc = fuel_price["value"] * gen.heat_rate

            for y in years:
                row_mc[y] = mc.get(y, pd.NA)

            rows.append(row_mc)

    generator_cost = pd.DataFrame(rows, columns=["generator", "cost_type"] + years)
    return generator_constants, generator_cost


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


if __name__ == "__main__":
    data_path = "gams2pypsa/data/processed/Gambia/"
    generator_constants, generator_cost = process_generator_data(data_path)
    # save to csvs
    generator_constants.to_csv(data_path + "generator_constants.csv", index=False)
    generator_cost.to_csv(data_path + "generator_costs.csv", index=False)