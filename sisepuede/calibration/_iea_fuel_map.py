"""
sisepuede/calibration/_iea_fuel_map.py

Single source of truth for the IEA product code -> SISEPUEDE fuel mapping.

Both build_iea_energy_crosswalk.py and build_energy_calibration_plan.py import
from here so the groupings stay in sync automatically.

The canonical form uses bare fuel suffixes (no "fuel_" prefix).
Three helper functions convert to the format each consumer needs:

    fuel_keys(iea_code)     -> ["fuel_coal", "fuel_coke"]
        Used by build_iea_energy_crosswalk._fields_for(), which expects ENFU
        category key_values from the ModelAttributes attribute table.

    fuel_suffixes(iea_code) -> ["coal", "coke"]
        Used by build_energy_calibration_plan to construct field names such as
        frac_inen_energy_{category}_{suffix}.

    agg_type(iea_code)      -> "sum" | "direct"
        Aggregation method for the IEA product in the crosswalk.
        Returns "sum" when the IEA code maps to multiple SISEPUEDE fuels
        (e.g. COAL -> coal + coke, OIL -> crude + diesel + ...), and
        "direct" when there is a one-to-one correspondence (e.g. NATGAS,
        ELECTR). Derived automatically from IEA_FUEL_MAP so it always stays
        in sync with the fuel groupings.

"""

from typing import Dict, List

##  IEA product code -> SISEPUEDE fuel suffixes (no "fuel_" prefix)
IEA_FUEL_MAP: Dict[str, List[str]] = {
    "COAL":      ["coal", "coke"],
    "OIL":       ["crude", "diesel", "gasoline", "kerosene", "oil",
                  "hydrocarbon_gas_liquids", "natural_gas_liquid"],
    "CRUDEOIL":  ["crude"], # crude oil is included in the oil group but it has it's own import export data
    "NATGAS":    ["natural_gas"],
    "NUCLEAR":   ["nuclear"],
    "HYDRO":     ["water"],
    "WINDSOLAR": ["wind", "solar"],
    "BIOWASTE":  ["biomass", "solid_biomass", "biogas", "biofuels", "waste"],
    "GEOTHERM":  ["geothermal"],
    "ELECTR":    ["electricity"],
    "HYDROGEN":  ["hydrogen"],
}

##  Reverse: suffix -> IEA product code
FUEL_SUFFIX_TO_IEA: Dict[str, str] = {
    suffix: iea
    for iea, suffixes in IEA_FUEL_MAP.items()
    for suffix in suffixes
}


def fuel_keys(iea_code: str) -> List[str]:
    """SISEPUEDE ENFU category key_values for a given IEA product code.

    Adds the "fuel_" prefix used in ModelAttributes attribute tables.
    Pass the result to _fields_for() in build_iea_energy_crosswalk.py.

    Example
    -------
        fuel_keys("COAL")  ->  ["fuel_coal", "fuel_coke"]
    """
    return [f"fuel_{s}" for s in IEA_FUEL_MAP.get(iea_code, [])]


def fuel_suffixes(iea_code: str) -> List[str]:
    """Bare fuel suffixes for a given IEA product code.

    Used to construct SISEPUEDE field names such as
    frac_inen_energy_{category}_{suffix}.

    Example
    -------
        fuel_suffixes("OIL")  ->  ["crude", "diesel", "gasoline", ...]
    """
    return list(IEA_FUEL_MAP.get(iea_code, []))


def agg_type(iea_code: str) -> str:
    """Crosswalk aggregation method for a given IEA product code.

    Returns "sum" when the IEA code maps to more than one SISEPUEDE fuel
    (the SSP fields must be summed to match the IEA aggregate), or "direct"
    when there is a one-to-one correspondence.

    Derived automatically from IEA_FUEL_MAP, so adding or removing fuels
    from a group automatically updates the aggregation type.

    Example
    -------
        agg_type("COAL")    ->  "sum"    # fuel_coal + fuel_coke
        agg_type("NATGAS")  ->  "direct" # fuel_natural_gas only
        agg_type("OIL")     ->  "sum"    # crude + diesel + gasoline + ...
    """
    return "sum" if len(IEA_FUEL_MAP.get(iea_code, [])) > 1 else "direct"