import warnings

from synthseer.build_model.time_utils import get_valid_years

def build_storage(model, storage_constants, storage_costs, modular_expansion=False):
    # carrier set to be returned after processing
    carriers = set("battery")
    # convert csv inputs into indexed dataframes
    storage_constants = storage_constants.set_index("storage")
    storage_costs = storage_costs.set_index(["storage", "cost_type"])
    storage_costs.columns = storage_costs.columns.astype(int)

    nstorage = len(storage_constants)

    if nstorage == 0:
        warnings.warn("No storage units found in the model.")
        return carriers

    for i in range(nstorage):
        sto = storage_constants.iloc[i]

        name, expandable = sto.name, sto.expandable

        if expandable:
            """Build option.
            In the PyPSA framework, we need to create a build option for each year.
            """

            y0, y1 = sto.start_year, sto.retirement_year
            capex = storage_costs.loc[(name, "capex"), :]
            unit_capacity = sto.unit_size_mwh if modular_expansion else None

            for year in get_valid_years(model, y0, y1):
                sto_year_name = f"{name}_{year}"

                model.add("StorageUnit", sto_year_name, bus=sto.zone,
                    carrier="battery",
                    build_year=year,
                    lifetime=sto.life,
                    # energy/power sizing
                    p_nom_extendable=True,
                    p_nom_mod=unit_capacity,
                    p_nom_max=sto.annual_build_limit_mwh,
                    # assume fixed duration
                    max_hours=sto.duration_hours,
                    # efficiencies
                    efficiency_store=sto.efficiency,
                    efficiency_dispatch=sto.efficiency,
                    # costs
                    capital_cost=capex[year],
                )

        else:
            """Existing storage unit.
            Create a single storage unit for the entire lifetime.
            """

            y0, y1 = sto.start_year, sto.retirement_year
            years = get_valid_years(model, y0, y1)

            if len(years) == 0:
                warnings.warn(f"No valid years found for storage {name}.")
                continue

            model.add(
                "StorageUnit", name, bus=sto.zone,
                carrier="battery",
                build_year=years[0],
                lifetime=sto.life,
                # fixed size
                p_nom=sto.capacity_mw,
                p_nom_extendable=False,
                # duration
                max_hours=sto.duration_hours,
                # efficiencies
                efficiency_store=sto.efficiency,
                efficiency_dispatch=sto.efficiency,
            )
    return carriers