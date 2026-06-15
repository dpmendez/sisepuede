"""
sisepuede/calibration/build_energy_calibration_plan.py

build_energy_calibration_plan()
---------------------------------
Factory that builds a CalibrationPlan for SISEPUEDE energy calibration,
matching the grouping structure below (one section per row):

  Row  Input variables               IEA target (crosswalk)
  ───  ────────────────────────────  ────────────────────────────────────────────
  1    consumpinit, scalar           Total Final Energy Consumption (by subsector)
  2    all fuel-mix fracs, imports   Total Energy Supply (by fuel)
  3    frac_enfu_demand_imported,    Energy Imports and Exports (by fuel)
       exports_enfu
  4    frac_trns_fuelmix_{mode}      Transport energy by fuel
  5    nemomod_entc_residual,        Electricity Generation Sources (by fuel)
       nemomod_entc_frac_msp
  6    frac_inen_energy_{cat}        Industry TFC by source (by fuel)
  7    frac_scoe_residential         Residential TFC by fuel
  8    frac_scoe_commercial_muni     Commercial/public TFC by fuel

Groups 1–5 target aggregate IEA pairs (INDPROD x fuel, sector x sector,
ELECTOUT x fuel, COALIMPORTS x IMPORTS, …).  Groups 6–8 target fuel-mix pairs
(INDUSTRY x COAL, RESIDENT x ELECTR, …) and use simplex-constrained fracs.

IEA fuel aggregation
---------------------
Multiple SISEPUEDE fuels map to a single IEA fuel code (see _IEA_FUEL_MAP).
This determines which variables are bundled in the same group.  For example,
diesel + gasoline + kerosene + oil + crude + hydrocarbon_gas_liquids all map
to IEA "OIL", so the INDUSTRY x OIL group contains frac_inen_energy_*_diesel,
frac_inen_energy_*_gasoline, etc. For imports/exports, only COAL, CRUDEOIL,
NATGAS, and ELECTR have dedicated trade balance targets.

Usage
------
    from sisepuede.calibration.build_energy_calibration_plan import (
        build_energy_calibration_plan,
    )

    plan = build_energy_calibration_plan(model_attributes)
    plan.summary()

    # Sector-by-sector
    for sector, sub in plan.by_sector().items():
        specs   = sub.get_specs()
        targets = sub.get_targets()

    # Bulk LHS
    result = runner.run_lhs(plan.get_specs(), n_samples=100)
"""

from __future__ import annotations

from dataclasses import fields as _dc_fields
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from sisepuede.calibration.calibration_group import CalibrationGroup, CalibrationPlan
from sisepuede.calibration.sensitivity import VariableSpec
from sisepuede.calibration._iea_fuel_map import IEA_FUEL_MAP, FUEL_SUFFIX_TO_IEA

#  Fallback defaults when callers pass lb/ub=None — these are the
#  declared defaults on the VariableSpec dataclass ("variable table" defaults).
_VARSPEC_DEFAULTS: Dict[str, float] = {
    f.name: f.default
    for f in _dc_fields(VariableSpec)
    if f.name in ("lb", "ub")
}

#  Local aliases to avoid changing call sites below
_IEA_FUEL_MAP = IEA_FUEL_MAP
_FUEL_TO_IEA  = FUEL_SUFFIX_TO_IEA

#  Import/export balance prefixes used in the crosswalk
#  Maps IEA fuel group -> (import_balance, export_balance)
_IEA_TRADE_BALANCES: Dict[str, Tuple[str, str]] = {
    "COAL":     ("COALIMPORTS",   "COALEXPORTS"),
    "CRUDEOIL": ("CRUDIMPORTS",   "CRUDEXPORTS"),
    "NATGAS":   ("GASIMPORTS",    "GASEXPORTS"),
    "ELECTR":   ("ELIMPORTS",     "ELEXPORTS"),
    #  Total-energy trade (all fuels combined)
    "ALL":      ("IMPORTS",       "EXPORTS"),
}


##########################
#    MAIN FACTORY        #
##########################

def build_energy_calibration_plan(
    model_attributes,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    only_groups: Optional[Union[str, Iterable[str]]] = None,
) -> CalibrationPlan:
    """Build the full energy CalibrationPlan for one SISEPUEDE country run.

    Scans model_attributes for all relevant input variable names and groups
    them into CalibrationGroups following the eight-row structure in the
    module docstring.

    Parameters
    ----------
    model_attributes : ModelAttributes
        SISEPUEDE ModelAttributes object.
    lb : float | None
        Lower-bound scale factor for every VariableSpec.  When None, each
        spec falls back to the VariableSpec dataclass default (the
        "variable table" default in sensitivity.py).
    ub : float | None
        Upper-bound scale factor.  Same None-fallback as `lb`.
    only_groups : str | Iterable[str] | None
        If provided, the returned plan contains only the named groups
        (e.g. "transport__transport").  Useful to isolate a single
        calibration target when debugging.  Names that do not match any
        group are silently ignored; if no group matches, the returned
        plan is empty.

    Returns
    -------
    CalibrationPlan
        One group per calibration target x sector or fuel combination.
        Groups are ordered: TFC -> TES -> Imports/Exports -> Transport fuel ->
        ENTC -> Industry fuel -> Residential fuel -> Commercial fuel.
    """
    fields_in: Set[str] = set(model_attributes.all_variable_fields_input)
    plan = CalibrationPlan()

    ##  Attribute tables
    inen_cats  = model_attributes.get_attribute_table(model_attributes.subsec_name_inen).key_values
    scoe_cats  = model_attributes.get_attribute_table(model_attributes.subsec_name_scoe).key_values
    trns_modes = model_attributes.get_attribute_table(model_attributes.subsec_name_trns).key_values
    enfu_keys  = model_attributes.get_attribute_table(model_attributes.subsec_name_enfu).key_values

    # Simplex group IDs (for fuel-fraction simplex groups in inen/trns/scoe)
    dict_simplex: Dict[str, int] = getattr(
        model_attributes, "dict_field_to_simplex_group", {}
    ) or {}

    # ------------------------------------------------------------------ #
    #  ROW 1 – Total Final Energy Consumption, by subsector              #
    #  Inputs: consumpinit_*, scalar_inen_energy_demand_*                #
    #  One group per sector (inen, trns, scoex3)                         #
    # ------------------------------------------------------------------ #

    ##  1a. INEN total
    ##  Only consumpinit_inen_energy_total_pj_* is used here because it is a direct
    ##  linear multiplier on total per-category INEN energy.  Including
    ##  scalar_inen_energy_demand_* alongside consumpinit_inen_* would
    ##  double-scale the production path (intensity x demscalar) as  
    ##  energy[t] = (intensity[0] / frac_norm[0]) × driver[t] × demscalar[t]
    ##  where intensity[0] = consumpinit_inen_energy_total_pj_[cat]
    inen_scalar = [
        f for f in fields_in
        if f.startswith("consumpinit_inen_energy_total_pj_")
    ]
    plan.add(CalibrationGroup(
        name        = "industry__industry",
        sector      = "inen",
        specs       = _make_specs(inen_scalar, lb, ub),
        iea_targets = [("INDUSTRY", "INDUSTRY")],
        notes       = "Initial consumption scalars for all non-agriculture INEN categories -> "
                      "drives total industry TFC (INDUSTRY x INDUSTRY). "
                      "scalar_inen_energy_demand_* excluded to avoid "
                      "double-scaling the production energy path.",
    ))

    ##  1b. TRNS total
    ##  The model's transport-energy equation (project_transportation, ~L3471 of energy_consumption.py)
    ##  is E_fuel prop vehicle_km x frac_fuelmix x energy_density / fuel_efficiency
    ##  with vehicle_km = demand_pkm/occrate (passenger) or demand_mtkm/avgload
    ##  (freight).  The direct knobs on total transport energy are therefore the demand_pkm and
    ##  demand_mtkm variables, which are included in the transport__transport group; the fuel mix
    ##  and efficiency levers are held out of Phase 1 and grouped separately (see below).
    _TRDE_DEMAND_INITS = {
        "deminit_trde_freight_mt_km",
        "deminit_trde_private_and_public_per_capita_passenger_km",
        "deminit_trde_regional_per_capita_passenger_km",
    }
    trde_demand = [f for f in fields_in if f in _TRDE_DEMAND_INITS]
    plan.add(CalibrationGroup(
        name        = "transport__transport",
        sector      = "trde",
        specs       = _make_specs(trde_demand, lb, ub),
        iea_targets = [("TRANSPORT", "TRANSPORT")],
        notes       = "TRDE initial pkm/mtkm demands -> drive total transport TFC "
                      "(TRANSPORT x TRANSPORT) linearly regardless of target year. "
                      "demscalar_trde_* excluded (renormalized to 1 at t=0); TRNS "
                      "inverse levers (fuelefficiency, avgload, occrate) live in "
                      "transport__fuel_efficiency.",
    ))

    ##  1b'. TRNS fuel efficiency — independent group, not wired to Phase 1
    ##  These variables are inverse levers on total transport energy and should
    ##  be tuned separately, not via the direct 'scalar = IEA/current' rule used in Phase 1.
    ##  The group has no iea_targets so the calibrator will ignore it in both
    ##  phases unless explicitly filtered to it.
    trns_efficiency = [
        f for f in fields_in
        if any(f.startswith(p) for p in [
            "fuelefficiency_trns_", "elecfuelefficiency_trns_",
            "avgload_trns_", "occrate_trns_",
        ])
    ]
    plan.add(CalibrationGroup(
        name        = "transport__fuel_efficiency",
        sector      = "trns",
        specs       = _make_specs(trns_efficiency, lb, ub),
        iea_targets = [],
        notes       = "Transport fuel efficiency, average load, and occupancy rates. "
                      "Inverse levers on TRANSPORTxTRANSPORT — held out of Phase 1 "
                      "on purpose.  Include via only_groups=[...] when tuning "
                      "these independently.",
    ))

    ##  1c–1d. SCOE total for residential and commercial (not agriculture)
    _SCOE_CAT_TO_IEA: Dict[str, Tuple[str, str]] = {
        "residential":         ("RESIDENT", "RESIDENT"),
        "commercial_municipal":("COMMPUB",  "COMMPUB"),
    }
    for scoe_cat, iea_target in _SCOE_CAT_TO_IEA.items():
        scoe_consump = [
            f for f in fields_in
            if f.startswith("consumpinit_scoe_") and scoe_cat in f
        ]
        bal, prod = iea_target
        plan.add(CalibrationGroup(
            name        = f"{bal.lower()}__{prod.lower()}",
            sector      = "scoe",
            specs       = _make_specs(scoe_consump, lb, ub),
            iea_targets = [iea_target],
            notes       = f"Initial per-household / per-GDP consumption for SCOE "
                          f"{scoe_cat} -> drives {iea_target[0]}x{iea_target[1]}.",
        ))

    ##  1e. AGRICULT — driven by INEN agriculture_and_livestock, not SCOE other_se.
    ##  The crosswalk maps (AGRICULT, AGRICULT) to energy_consumption_inen_agriculture_and_livestock,
    ##  so the correct input lever is consumpinit_inen_energy_total_pj_agriculture_and_livestock.
    inen_agr = [
        f for f in fields_in
        if f == "consumpinit_inen_energy_total_pj_agriculture_and_livestock"
    ]
    plan.add(CalibrationGroup(
        name        = "agricult__agricult",
        sector      = "inen",
        specs       = _make_specs(inen_agr, lb, ub),
        iea_targets = [("AGRICULT", "AGRICULT")],
        notes       = "Initial INEN agriculture-and-livestock energy -> drives "
                      "energy_consumption_inen_agriculture_and_livestock, "
                      "which is the SSP comparator for IEA AGRICULT x AGRICULT.",
    ))

    # ------------------------------------------------------------------ #
    #  ROW 2 – Total Energy Supply, by fuel                              #
    #  Inputs: cross-sector fuel fracs + import fraction                 #
    #  One group per IEA fuel code                                       #
    #                                                                    #
    #  Note: SISEPUEDE computes domestic production as a residual of     #
    #  total demand minus imports.  There is no explicit "production"    #
    #  input.  The variables that most directly drive TES-per-fuel are   #
    #  the fuel-fraction inputs from ALL sectors (inen, trns, scoe)      #
    #  together with the import fraction for that fuel.  These overlap   #
    #  with groups in rows 4/6/7/8; de-duplication in get_specs() is     #
    #  applied when using the plan in bulk mode.                         #
    # ------------------------------------------------------------------ #

    for iea_fuel, fuel_suffixes in _IEA_FUEL_MAP.items():
        tes_vars: List[str] = []

        ##  Import fraction for this fuel group
        for suf in fuel_suffixes:
            imp_var = f"frac_enfu_fuel_demand_imported_pj_fuel_{suf}"
            if imp_var in fields_in:
                tes_vars.append(imp_var)

        ##  INEN fuel fracs
        for cat in inen_cats:
            for suf in fuel_suffixes:
                v = f"frac_inen_energy_{cat}_{suf}"
                if v in fields_in:
                    tes_vars.append(v)

        ##  TRNS fuel fracs
        for mode in trns_modes:
            for suf in fuel_suffixes:
                v = f"frac_trns_fuelmix_{mode}_{suf}"
                if v in fields_in:
                    tes_vars.append(v)

        ##  SCOE fuel fracs (residential, commercial, other_se)
        for scoe_cat in scoe_cats:
            for suf in fuel_suffixes:
                v = f"frac_scoe_heat_energy_{scoe_cat}_{suf}"
                if v in fields_in:
                    tes_vars.append(v)

        if not tes_vars:
            continue

        # plan.add(CalibrationGroup(
        #     name        = f"indprod__{iea_fuel.lower()}",
        #     sector      = ["inen", "trns", "scoe", "enfu"],
        #     specs       = _make_specs(tes_vars, lb, ub),
        #     iea_targets = [("INDPROD", iea_fuel)],
        #     notes       = f"All fuel-fraction inputs (inen+trns+scoe) and import "
        #                   f"fraction for {iea_fuel} -> drives TES (INDPRODx{iea_fuel}).  "
        #                   f"Overlaps with rows 4/6/7/8; variables are de-duplicated "
        #                   f"when plan is used in bulk mode.",
        # ))

    # ------------------------------------------------------------------ #
    #  ROW 3 – Energy Imports and Exports, by fuel                       #
    #  Inputs: frac_enfu_fuel_demand_imported, exports_enfu              #
    #  One group per fuel per direction (import / export)                #
    # ------------------------------------------------------------------ #

    for iea_fuel, (bal_imp, bal_exp) in _IEA_TRADE_BALANCES.items():
        fuel_suffixes = (
            _IEA_FUEL_MAP.get(iea_fuel, [])
            if iea_fuel != "ALL"
            else list(_FUEL_TO_IEA.keys())    # all fuels
        )

        ##  Import fraction variables
        imp_vars = [
            f"frac_enfu_fuel_demand_imported_pj_fuel_{suf}"
            for suf in fuel_suffixes
            if f"frac_enfu_fuel_demand_imported_pj_fuel_{suf}" in fields_in
        ]
        if imp_vars:
            plan.add(CalibrationGroup(
                name        = f"{bal_imp.lower()}__imports",
                sector      = "enfu",
                specs       = _make_specs(imp_vars, lb, ub),
                iea_targets = [(bal_imp, "IMPORTS")],
                notes       = f"Fraction of total {iea_fuel} demand met by imports "
                              f"-> {bal_imp}xIMPORTS.",
            ))

        ##  Export volume variables
        exp_vars = [
            f"exports_enfu_pj_fuel_{suf}"
            for suf in fuel_suffixes
            if f"exports_enfu_pj_fuel_{suf}" in fields_in
        ]
        if exp_vars:
            plan.add(CalibrationGroup(
                name        = f"{bal_exp.lower()}__exports",
                sector      = "enfu",
                specs       = _make_specs(exp_vars, lb, ub),
                iea_targets = [(bal_exp, "EXPORTS")],
                notes       = f"Scheduled export volumes for {iea_fuel} fuels "
                              f"-> {bal_exp}xEXPORTS.",
            ))

    # ------------------------------------------------------------------ #
    #  ROW 4 – Transport energy by fuel balance                          #
    #  Inputs: frac_trns_fuelmix_{mode}_{fuel}  [simplex within mode]    #
    #  One group per IEA fuel code                                       #
    # ------------------------------------------------------------------ #

    ##  IEA fuel targets available for transport (from crosswalk)
    _TRNS_IEA_TARGETS: Dict[str, str] = {
        "OIL":      ("TRANSPORT", "OIL"),
        "ELECTR":   ("TRANSPORT", "ELECTR"),
        "NATGAS":   ("TRANSPORT", "NATGAS"),
        "BIOWASTE": ("TRANSPORT", "BIOWASTE"),
        "HYDROGEN": ("TRANSPORT", "HYDROGEN"),
    }

    for iea_fuel, target in _TRNS_IEA_TARGETS.items():
        fuel_suffixes = _IEA_FUEL_MAP.get(iea_fuel, [])
        trns_vars: List[str] = []
        simplex_ids: List[int] = []
        ids_seen: Set[int] = set()

        for mode in trns_modes:
            for suf in fuel_suffixes:
                v = f"frac_trns_fuelmix_{mode}_{suf}"
                if v in fields_in:
                    trns_vars.append(v)
                    gid = dict_simplex.get(v)
                    if gid is not None and gid not in ids_seen:
                        ids_seen.add(gid)
                        simplex_ids.append(gid)

        if not trns_vars:
            continue

        specs = _make_specs(trns_vars, lb, ub, simplex_ids=dict_simplex)
        bal, prod = target
        plan.add(CalibrationGroup(
            name              = f"{bal.lower()}__{prod.lower()}",
            sector            = "trns",
            specs             = specs,
            iea_targets       = [target],
            constraint_type   = "simplex",
            simplex_group_ids = simplex_ids,
            notes             = f"Fuel mix fraction for {iea_fuel} across all transport "
                                f"modes -> {target[0]}x{target[1]}.  Simplex within each "
                                f"mode (fracs sum to 1 per mode).",
        ))

    # ------------------------------------------------------------------ #
    #  ROW 5 – Electricity Generation Sources                            #
    #  Inputs: nemomod_entc_residual_capacity_pp_*,                      #
    #          nemomod_entc_frac_min_share_production_pp_*               #
    #  One group per IEA fuel code (power plant technology)              #
    # ------------------------------------------------------------------ #

    #  Technology suffix -> IEA fuel code
    _ENTC_TECH_TO_IEA: Dict[str, str] = {
        "pp_coal":              "COAL",
        "pp_coal_ccs":          "COAL",
        "pp_gas":               "NATGAS",
        "pp_gas_ccs":           "NATGAS",
        "pp_oil":               "OIL",
        "pp_nuclear":           "NUCLEAR",
        "pp_hydropower":        "HYDRO",
        "pp_wind":              "WIND",
        "pp_solar":             "SOLARPV",
        "pp_biomass":           "BIOFUEL",
        "pp_biogas":            "BIOFUEL",
        "pp_waste_incineration":"WASTE",
        "pp_geothermal":        "GEOTHERM",
        "pp_ocean":             "TIDE",
    }

    ##  Collect entc vars per IEA fuel
    entc_by_iea: Dict[str, List[str]] = {}
    _ENTC_PREFIXES = [
        "nemomod_entc_residual_capacity_",
        "nemomod_entc_frac_min_share_production_",
        "nemomod_entc_total_annual_max_capacity_",
        "nemomod_entc_total_annual_min_capacity_",
        "nemomod_entc_scalar_availability_factor_",
    ]
    for tech, iea_fuel in _ENTC_TECH_TO_IEA.items():
        for prefix in _ENTC_PREFIXES:
            # Variables may be named {prefix}{tech} or {prefix}{tech}_gw etc.
            matching = [
                f for f in fields_in
                if f.startswith(f"{prefix}{tech}")
            ]
            for v in matching:
                entc_by_iea.setdefault(iea_fuel, []).append(v)

    for iea_fuel, entc_vars in sorted(entc_by_iea.items()):
        plan.add(CalibrationGroup(
            name        = f"electout__{iea_fuel.lower()}",
            sector      = "entc",
            specs       = _make_specs(list(dict.fromkeys(entc_vars)), lb, ub),
            iea_targets = [("ELECTOUT", iea_fuel)],
            notes       = f"Residual capacity, MSP, and capacity bounds for "
                          f"{iea_fuel} power plant technologies -> ELECTOUTx{iea_fuel}.",
        ))

    # ------------------------------------------------------------------ #
    #  ROW 6 – Industry TFC by source                                    #
    #  Inputs: frac_inen_energy_{cat}_{fuel}  [simplex within each cat]  #
    #  One group per IEA fuel code                                       #
    # ------------------------------------------------------------------ #

    ##  IEA fuel targets for industry from crosswalk
    _INEN_IEA_TARGETS: Dict[str, Tuple[str, str]] = {
        "COAL":     ("INDUSTRY", "COAL"),
        "OIL":      ("INDUSTRY", "OIL"),
        "NATGAS":   ("INDUSTRY", "NATGAS"),
        "ELECTR":   ("INDUSTRY", "ELECTR"),
        "BIOWASTE": ("INDUSTRY", "BIOWASTE"),
        "GEOTHERM": ("INDUSTRY", "GEOTHERM"),
        "WINDSOLAR":("INDUSTRY", "WINDSOLAR"),
    }

    ##  Exclude agriculture_and_livestock: it is its own IEA target
    ##  (AGRICULT, AGRICULT) handled by the agricult__agricult Phase-1 group,
    ##  and the IEA (INDUSTRY, *) crosswalk explicitly excludes agriculture.
    ##  Including frac_inen_energy_agriculture_and_livestock_{fuel} here would
    ##  let Phase 2 drag agriculture's fuel mix to match an INDUSTRY target
    ##  that doesn't include agriculture, distorting agriculture's frac
    ##  trajectory via Aitchison renormalisation. IEA has no
    ##  (AGRICULT, FUEL) breakdown to calibrate against, so agriculture's
    ##  fuel mix is intentionally left at the model defaults.
    inen_cats_excl_ag = [c for c in inen_cats if c != "agriculture_and_livestock"]

    for iea_fuel, target in _INEN_IEA_TARGETS.items():
        fuel_suffixes = _IEA_FUEL_MAP.get(iea_fuel, [])
        inen_vars: List[str] = []
        simplex_ids: List[int] = []
        ids_seen: Set[int] = set()

        for cat in inen_cats_excl_ag:
            for suf in fuel_suffixes:
                v = f"frac_inen_energy_{cat}_{suf}"
                if v in fields_in:
                    inen_vars.append(v)
                    gid = dict_simplex.get(v)
                    if gid is not None and gid not in ids_seen:
                        ids_seen.add(gid)
                        simplex_ids.append(gid)

        if not inen_vars:
            continue

        specs = _make_specs(inen_vars, lb, ub, simplex_ids=dict_simplex)
        bal, prod = target
        plan.add(CalibrationGroup(
            name              = f"{bal.lower()}__{prod.lower()}",
            sector            = "inen",
            specs             = specs,
            iea_targets       = [target],
            constraint_type   = "simplex",
            simplex_group_ids = simplex_ids,
            notes             = f"Fuel fraction for {iea_fuel} across all INEN "
                                f"categories except agriculture_and_livestock "
                                f"-> {target[0]}x{target[1]}.  Simplex within "
                                f"each industry category (fracs sum to 1 per cat).",
        ))

    # ------------------------------------------------------------------ #
    #  ROW 7 – Residential TFC by fuel                                   #
    #  Inputs: frac_scoe_heat_energy_residential_{fuel}  [simplex]       #
    # ------------------------------------------------------------------ #

    _SCOE_IEA_TARGETS: Dict[str, Tuple[str, str]] = {
        "COAL":     ("RESIDENT", "COAL"),
        "OIL":      ("RESIDENT", "OIL"),
        "NATGAS":   ("RESIDENT", "NATGAS"),
        "ELECTR":   ("RESIDENT", "ELECTR"),
        "BIOWASTE": ("RESIDENT", "BIOWASTE"),
    }

    for iea_fuel, target in _SCOE_IEA_TARGETS.items():
        fuel_suffixes = _IEA_FUEL_MAP.get(iea_fuel, [])
        res_vars: List[str] = []
        simplex_ids: List[int] = []
        ids_seen: Set[int] = set()

        for suf in fuel_suffixes:
            v = f"frac_scoe_heat_energy_residential_{suf}"
            if v in fields_in:
                res_vars.append(v)
                gid = dict_simplex.get(v)
                if gid is not None and gid not in ids_seen:
                    ids_seen.add(gid)
                    simplex_ids.append(gid)

        if not res_vars:
            continue

        specs = _make_specs(res_vars, lb, ub, simplex_ids=dict_simplex)
        bal, prod = target
        plan.add(CalibrationGroup(
            name              = f"{bal.lower()}__{prod.lower()}",
            sector            = "scoe",
            specs             = specs,
            iea_targets       = [target],
            constraint_type   = "simplex",
            simplex_group_ids = simplex_ids,
            notes             = f"Residential heat-energy fuel fraction for {iea_fuel} "
                                f"-> {target[0]}x{target[1]}.  Simplex (all residential "
                                f"fuel fracs sum to 1).",
        ))

    # ------------------------------------------------------------------ #
    #  ROW 8 – Commercial/public TFC by fuel                             #
    #  Inputs: frac_scoe_heat_energy_commercial_municipal_{fuel}         #
    # ------------------------------------------------------------------ #

    _SCOE_COMM_IEA_TARGETS: Dict[str, Tuple[str, str]] = {
        "COAL":     ("COMMPUB", "COAL"),
        "OIL":      ("COMMPUB", "OIL"),
        "NATGAS":   ("COMMPUB", "NATGAS"),
        "ELECTR":   ("COMMPUB", "ELECTR"),
        "BIOWASTE": ("COMMPUB", "BIOWASTE"),
    }

    for iea_fuel, target in _SCOE_COMM_IEA_TARGETS.items():
        fuel_suffixes = _IEA_FUEL_MAP.get(iea_fuel, [])
        comm_vars: List[str] = []
        simplex_ids: List[int] = []
        ids_seen: Set[int] = set()

        for suf in fuel_suffixes:
            v = f"frac_scoe_heat_energy_commercial_municipal_{suf}"
            if v in fields_in:
                comm_vars.append(v)
                gid = dict_simplex.get(v)
                if gid is not None and gid not in ids_seen:
                    ids_seen.add(gid)
                    simplex_ids.append(gid)

        if not comm_vars:
            continue

        specs = _make_specs(comm_vars, lb, ub, simplex_ids=dict_simplex)
        bal, prod = target
        plan.add(CalibrationGroup(
            name              = f"{bal.lower()}__{prod.lower()}",
            sector            = "scoe",
            specs             = specs,
            iea_targets       = [target],
            constraint_type   = "simplex",
            simplex_group_ids = simplex_ids,
            notes             = f"Commercial/public heat-energy fuel fraction for "
                                f"{iea_fuel} -> {target[0]}x{target[1]}.  Simplex.",
        ))

    ##  Optional: restrict the plan to a caller-specified subset of group names.
    ##  Handy for debugging: build_energy_calibration_plan(ma, only_groups="transport__transport")
    ##  returns a plan with a single group so the calibrator touches nothing else.
    if only_groups is not None:
        wanted: Set[str] = (
            {only_groups} if isinstance(only_groups, str) else set(only_groups)
        )
        plan = CalibrationPlan([g for g in plan.groups if g.name in wanted])

    return plan


##########################
#    INTERNAL HELPERS    #
##########################

def _make_specs(
    columns: List[str],
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    simplex_ids: Optional[Dict[str, int]] = None,
) -> List[VariableSpec]:
    """Build a VariableSpec per column, optionally marking simplex members.

    When `lb` / `ub` is None, the VariableSpec dataclass default for that
    field is used (the "variable table" default).  This lets callers leave
    bounds unspecified without having to duplicate the canonical defaults.
    """
    lb_final = _VARSPEC_DEFAULTS["lb"] if lb is None else lb
    ub_final = _VARSPEC_DEFAULTS["ub"] if ub is None else ub

    specs = []
    for col in columns:
        gid = simplex_ids.get(col) if simplex_ids else None
        specs.append(VariableSpec(
            column           = col,
            lb               = lb_final,
            ub               = ub_final,
            is_simplex_group = gid is not None,
            simplex_group_id = gid,
        ))
    return specs