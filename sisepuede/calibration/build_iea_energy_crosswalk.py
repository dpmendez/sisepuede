"""
sisepuede/calibration/build_iea_energy_crosswalk.py

IEACrosswalkBuilder
-------------------
Programmatically build the IEA <-> SISEPUEDE energy crosswalk CSV.

Each crosswalk row maps one IEA (Balance x Product) pair to one or more
SISEPUEDE output column names. Field names are verified against
model_attributes at build time using ModelVariable.build_fields(), so the
output CSV always reflects the actual fields present in the installed version
of SISEPUEDE.

Typical usage
-------------
    from sisepuede.manager.sisepuede_file_structure import SISEPUEDEFileStructure
    from sisepuede.calibration.build_iea_energy_crosswalk import IEACrosswalkBuilder

    file_struct      = SISEPUEDEFileStructure()
    model_attributes = file_struct.model_attributes

    builder = IEACrosswalkBuilder(model_attributes)
    df_xw   = builder.build(write_csv = True)
"""

import os
import pandas as pd
from typing import *

from sisepuede.calibration._iea_fuel_map import fuel_keys, agg_type



####################
#    CONSTANTS     #
####################

_CROSSWALK_REL_PATH = os.path.join(
    "data_crosswalks",
    "sisepuede_iea_energy_crosswalk.csv",
)

_VAR_SEP = ":"      # separator between field names inside a CSV cell

_COLS = [
    "iea_balance_code",
    "iea_balance_name",
    "iea_product_code",
    "iea_product_name",
    "sisepuede_subsector",
    "sisepuede_output_variables",
    "aggregation",
    "unit_sisepuede",
    "unit_conversion_to_tj",
    "mapping_quality",
    "notes",
]



############################
#    PRIMARY CLASS         #
############################

class IEACrosswalkBuilder:
    """
    Build the IEA <-> SISEPUEDE energy crosswalk CSV programmatically.

    Field names are looked up via model_attributes.get_variable(name)
    .build_fields(category_restrictions = [...]) so that the output CSV
    is always consistent with the installed version of SISEPUEDE.

    Public interface
    ----------------
        build(write_csv = True) -> pd.DataFrame
            Build and optionally write the crosswalk CSV.
    """

    def __init__(self,
        model_attributes,
        path_out: Union[str, None] = None,
    ) -> None:
        """
        Function Arguments
        ------------------
        model_attributes : ModelAttributes
            SISEPUEDE model attributes object.

        Keyword Arguments
        -----------------
        path_out : Union[str, None]
            Full path for the output CSV.  If None, resolved automatically
            from model_attributes.dir_ref.
        """

        self.model_attributes = model_attributes
        self._path_out = path_out

        self._initialize_variable_registry()

        return None


    ##############################
    #    INITIALISATION METHODS  #
    ##############################

    def _initialize_variable_registry(self,
    ) -> None:
        """
        Verify that every SISEPUEDE variable name used by the builder is
        recognised by model_attributes.  Logs a warning for any missing name
        but does not raise, so partial builds are still possible.

        Sets self._missing_vars (list of unrecognised names).
        """

        ##  VARIABLES NEEDED BY THE BUILDER

        _VARS_NEEDED: List[str] = [
            # ENFU — supply / trade
            "Fuel Production",
            "Fuel Imports",
            "Adjusted Fuel Exports",
            "Electrical Transmission Loss",
            # ENFU — demand by fuel (recorded as ENFU outputs)
            "Energy Demand by Fuel in Industrial Energy",
            "Energy Demand by Fuel in SCOE",
            # ENTC — electricity generation by technology
            "NemoMod Production by Technology",
            # INEN — consumption
            "Total Energy Consumption from Industrial Energy",
            "Energy Consumption from Industrial Energy",
            # SCOE — consumption
            "Total Energy Consumption from SCOE",
            "Energy Consumption from SCOE",
            # TRNS — consumption totals
            "Total Energy Consumption from Transportation",
            # TRNS — consumption by fuel (across all modes)
            "Transportation Modal Energy Consumption from Biofuels",
            "Transportation Modal Energy Consumption from Diesel",
            "Transportation Modal Energy Consumption from Electricity",
            "Transportation Modal Energy Consumption from Gasoline",
            "Transportation Modal Energy Consumption from Hydrocarbon Gas Liquids",
            "Transportation Modal Energy Consumption from Hydrogen",
            "Transportation Modal Energy Consumption from Kerosene",
            "Transportation Modal Energy Consumption from Natural Gas",
            # CCSQ
            "Total Energy Consumption from CCSQ",
        ]

        missing = []
        for name in _VARS_NEEDED:
            if self.model_attributes.get_variable(name) is None:
                missing.append(name)

        if missing:
            import warnings
            warnings.warn(
                "IEACrosswalkBuilder: the following SISEPUEDE variable names "
                f"were not found in model_attributes and will produce empty "
                f"field lists:\n  " + "\n  ".join(missing),
                stacklevel = 2,
            )

        self._missing_vars = missing

        return None


    ##############################
    #    INTERNAL HELPERS        #
    ##############################

    def _fields_for(self,
        var_name: str,
        categories: Union[List[str], str, None] = None,
    ) -> List[str]:
        """
        Return SISEPUEDE field names for a model variable, optionally
        filtered to a subset of categories.

        Uses ModelVariable.build_fields(category_restrictions = ...) so the
        returned names are guaranteed to exist in the model.

        Function Arguments
        ------------------
        var_name : str
            Human-readable SISEPUEDE variable name
            (e.g. "Transportation Modal Energy Consumption from Diesel").

        Keyword Arguments
        -----------------
        categories : Union[List[str], str, None]
            Category values to restrict to. Passed directly to
            build_fields(category_restrictions = ...).
            * None  -> all categories (returns full list of fields)
            * list  -> filtered to the listed categories
            * str   -> treated as a single-element list
        """

        mv = self.model_attributes.get_variable(var_name)
        if mv is None:
            return []

        ##  NORMALISE categories TO list OR None

        if isinstance(categories, str):
            categories = [categories]

        result = mv.build_fields(category_restrictions = categories)

        if result is None:
            return []
        if isinstance(result, str):
            return [result]

        return list(result)


    def _join(self,
        fields: List[str],
    ) -> str:
        """Join field names with _VAR_SEP into a single cell string."""

        return _VAR_SEP.join(f for f in fields if f)


    def _row(self,
        balance_code: str,
        balance_name: str,
        product_code: str,
        product_name: str,
        subsector: str,
        fields: List[str],
        aggregation: str,
        unit: str,
        conversion: Union[int, float],
        quality: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Return a single crosswalk data row as a dict."""

        return dict(zip(
            _COLS,
            [
                balance_code, balance_name,
                product_code, product_name,
                subsector,
                self._join(fields),
                aggregation, unit, conversion,
                quality, notes,
            ],
        ))


    def _section(self,
        label: str,
    ) -> List[Dict[str, Any]]:
        """Return a blank-row + section-header pair used as a CSV divider."""

        blank = {k: None for k in _COLS}
        header = {**blank, "iea_balance_name": f"--- {label} ---"}

        return [blank, header, blank]


    def _note_row(self,
        text: str,
    ) -> Dict[str, Any]:
        """Return a single comment row (all None except iea_balance_name)."""

        return {**{k: None for k in _COLS}, "iea_balance_name": text}


    ##############################
    #    ROW-BUILDING SECTIONS   #
    ##############################

    def _rows_supply(self) -> List[Dict[str, Any]]:
        """
        Build crosswalk rows for the Energy Supply section.
        IEA balance: INDPROD (Indigenous Production / Total Energy Supply).
        # SISEPUEDE variable: Fuel Production.
        SISEPUEDE variable: Total Energy Demand by Fuel.
        """

        rows = self._section("ENERGY SUPPLY")

        # prod_var = "Fuel Production"
        prod_var = "Total Energy Demand by Fuel"

        for product_code, product_name, fuel_cats, agg, quality, notes in [
            (
                "COAL",     "Coal",
                fuel_keys("COAL"),
                agg_type("COAL"), "approximate",
                "IEA TES Coal aggregate; includes fuel_coal + fuel_coke; "
                "SISEPUEDE does not distinguish hard vs. brown coal",
            ),
            (
                "NATGAS",   "Natural gas",
                fuel_keys("NATGAS"),
                agg_type("NATGAS"), "exact", "",
            ),
            (
                "OIL",      "Oil",
                fuel_keys("OIL"),
                agg_type("OIL"), "approximate",
                "IEA Oil includes crude and NGL; SISEPUEDE sum of fuel_{crude, diesel, ...} "
                "and fuel_natural_gas_liquid is the closest analog. "
                "NGLs (propane, butane, ethane, etc.) are extracted during natural gas "
                "processing and classified under Oil in IEA's TES",
            ),
            (
                "NUCLEAR",  "Nuclear",
                fuel_keys("NUCLEAR"),
                agg_type("NUCLEAR"), "exact", "",
            ),
            (
                "HYDRO",    "Hydro",
                fuel_keys("HYDRO"),
                agg_type("HYDRO"), "exact", "",
            ),
            (
                "WINDSOLAR", "Wind solar etc.",
                fuel_keys("WINDSOLAR"),
                agg_type("WINDSOLAR"), "approximate",
                "IEA aggregates wind+solar+other renewables; "
                "sum SISEPUEDE wind and solar",
            ),
            (
                "BIOWASTE",  "Biofuels and waste",
                fuel_keys("BIOWASTE"),
                agg_type("BIOWASTE"), "approximate",
                "IEA aggregates all biofuels and waste; "
                "sum SISEPUEDE biomass+solid_biomass+biogas+biofuels+waste",
            ),
            (
                "GEOTHERM",  "Geothermal",
                fuel_keys("GEOTHERM"),
                agg_type("GEOTHERM"), "exact", "",
            ),
            (
                "ELECTR",  "Electricity",
                fuel_keys("ELECTR"),
                agg_type("ELECTR"), "exact", "",
            ),
            (
                "HYDROGEN",  "Hydrogen",
                fuel_keys("HYDROGEN"),
                agg_type("HYDROGEN"), "no_match",
                "IEA free-access TES has no hydrogen row; "
                "tracked only in premium datasets post-2022",
            ),
            # (
            #     "AMMONIA",  "Ammonia",
            #     ["fuel_ammonia"],
            #     "direct", "no_match",
            #     "Ammonia not tracked as energy carrier in IEA free-access TES data",
            # ),
        ]:
            rows.append(self._row(
                "INDPROD", "Total energy supply",
                product_code, product_name,
                "enfu",
                self._fields_for(prod_var, fuel_cats),
                agg, "PJ", 1000, quality, notes,
            ))

        return rows


    def _rows_imports_exports(self) -> List[Dict[str, Any]]:
        """
        Build crosswalk rows for the Imports and Exports section.
        SISEPUEDE variables: Fuel Imports, Adjusted Fuel Exports.
        """

        rows = self._section("IMPORTS AND EXPORTS")

        imp_var = "Fuel Imports"
        exp_var = "Adjusted Fuel Exports"

        ##  TOTAL IMPORTS AND EXPORTS

        ## Do not calibrate for total imports and exports
        ## Avoid inducing extra errors
        # rows.append(self._row(
        #     "IMPORTS", "Energy imports and exports",
        #     "IMPORTS", "Imports",
        #     "enfu",
        #     self._fields_for(imp_var),
        #     "sum", "PJ", 1000, "approximate",
        #     "Total energy imports; sum across all SISEPUEDE fuel imports",
        # ))
        # rows.append(self._row(
        #     "EXPORTS", "Energy imports and exports",
        #     "EXPORTS", "Exports",
        #     "enfu",
        #     self._fields_for(exp_var),
        #     "sum", "PJ", 1000, "approximate",
        #     "Total energy exports; sum across tracked SISEPUEDE fuel exports",
        # ))

        ##  PER-FUEL IMPORTS AND EXPORTS

        for balance_prefix, balance_name, imp_cats, exp_cats, iea_code, quality, imp_notes, exp_notes in [
            (
                "COAL", "Coal imports and exports",
                fuel_keys("COAL"),
                fuel_keys("COAL"),
                "COAL", "exact",
                "",
                "exportsadj tracks surplus supply; may undercount scheduled exports",
            ),
            (
                "CRUD", "Crude oil imports and exports",
                fuel_keys("CRUDEOIL"),
                fuel_keys("CRUDEOIL"),
                "CRUDEOIL", "exact",
                "",
                "exportsadj tracks surplus supply; may undercount scheduled exports",
            ),
            (
                "GAS", "Natural gas imports and exports",
                fuel_keys("NATGAS"),
                fuel_keys("NATGAS"),
                "NATGAS", "exact",
                "",
                "exportsadj tracks surplus supply; may undercount scheduled exports",
            ),
            (
                "EL", "Electricity imports and exports",
                fuel_keys("ELECTR"),
                fuel_keys("ELECTR"),
                "ELECTR", "exact",
                "",
                "exportsadj tracks surplus supply; may undercount scheduled exports",
            ),
        ]:
            rows.append(self._row(
                f"{balance_prefix}IMPORTS", balance_name,
                "IMPORTS", "Imports",
                "enfu",
                self._fields_for(imp_var, imp_cats),
                agg_type(iea_code), "PJ", 1000, quality, imp_notes,
            ))
            rows.append(self._row(
                f"{balance_prefix}EXPORTS", balance_name,
                "EXPORTS", "Exports",
                "enfu",
                self._fields_for(exp_var, exp_cats),
                agg_type(iea_code), "PJ", 1000, quality, exp_notes,
            ))


        return rows


    def _rows_electricity_generation(self) -> List[Dict[str, Any]]:
        """
        Build crosswalk rows for the Electricity Generation by Source section.
        IEA balance: ELECTOUT. IEA unit is GWh; SISEPUEDE is PJ.
        SISEPUEDE variable: NemoMod Production by Technology.
        """

        rows = self._section("ELECTRICITY GENERATION BY SOURCE")

        rows.append(self._note_row(
            "-- IEA unit is GWh; SISEPUEDE is PJ. "
            "Conversion in prepare_iea: 1 GWh = 0.0036 PJ (or 3.6 TJ) --"
        ))
        rows.append({k: None for k in _COLS})

        entc_var = "NemoMod Production by Technology"

        for product_code, product_name, tech_cats, agg, quality, notes in [
            (
                "COAL",     "Coal",
                ["pp_coal", "pp_coal_ccs"],
                "sum", "exact",
                "IEA unit GWh; conversion handled in prepare_iea",
            ),
            (
                "OIL",      "Oil",
                ["pp_oil"],
                "direct", "approximate", "",
            ),
            (
                "NATGAS",   "Natural gas",
                ["pp_gas", "pp_gas_ccs"],
                "sum", "exact", "",
            ),
            (
                "NUCLEAR",  "Nuclear",
                ["pp_nuclear"],
                "direct", "exact", "",
            ),
            (
                "HYDRO",    "Hydro",
                ["pp_hydropower"],
                "direct", "exact", "",
            ),
            (
                "WIND",     "Wind",
                ["pp_wind"],
                "direct", "exact", "",
            ),
            (
                "SOLARPV",  "Solar PV",
                ["pp_solar"],
                "direct", "approximate",
                "pp_solar covers both PV and thermal; "
                "sum IEA Solar PV + Solar thermal for best comparison",
            ),
            (
                "SOLARTH",  "Solar thermal",
                ["pp_solar"],
                "direct", "approximate",
                "mapped to same pp_solar as Solar PV; do not double-count",
            ),
            (
                "BIOFUEL",  "Biofuels",
                ["pp_biomass", "pp_biogas"],
                "sum", "approximate", "",
            ),
            (
                "WASTE",    "Waste",
                ["pp_waste_incineration"],
                "direct", "approximate", "",
            ),
            (
                "GEOTHERM", "Geothermal",
                ["pp_geothermal"],
                "direct", "exact", "",
            ),
            (
                "TIDE",     "Tide",
                ["pp_ocean"],
                "direct", "exact", "",
            ),
        ]:
            rows.append(self._row(
                "ELECTOUT", "Electricity generation sources",
                product_code, product_name,
                "entc",
                self._fields_for(entc_var, tech_cats),
                agg, "PJ", 1000, quality, notes,
            ))

        ##  TOTAL ELECTRICITY GENERATION

        rows.append(self._row(
            "ELECTOUT", "Electricity generation sources",
            "TOTAL", "Total",
            "entc",
            self._fields_for(entc_var),
            "sum", "PJ", 1000, "approximate",
            "Sum all power plant technologies",
        ))

        ##  TRANSMISSION LOSSES

        rows.append(self._row(
            "LOSSES", "Electricity generation sources",
            "LOSSES", "Losses",
            "enfu",
            self._fields_for("Electrical Transmission Loss", ["fuel_electricity"]),
            "direct", "PJ", 1000, "exact",
            "SISEPUEDE explicitly tracks T&D losses",
        ))

        return rows


    def _rows_tfc_by_sector(self) -> List[Dict[str, Any]]:
        """
        Build crosswalk rows for Total Final Consumption by Sector (all fuels).
        Uses energy_consumption_* variables (point-of-use consumption).
        """

        rows = self._section("TOTAL FINAL ENERGY CONSUMPTION BY SECTOR (all fuels)")

        rows.append(self._note_row(
            "-- Uses energy_consumption_* variables (point-of-use consumption) --"
        ))
        rows.append({k: None for k in _COLS})

        ##  SCOE sub-category fields — filter by sub-category suffix

        scoe_cat_fields = self._fields_for("Energy Consumption from SCOE")

        rows.append(self._row(
            "INDUSTRY", "Total final energy consumption",
            "INDUSTRY", "Industry",
            "inen",
            self._fields_for("Total Energy Consumption from Industrial Energy"),
            "direct", "PJ", 1000, "approximate",
            "Consumption variable (point-of-use); IEA Industry includes "
            "some agriculture-adjacent manufacturing",
        ))
        rows.append(self._row(
            "TRANSPORT", "Total final energy consumption",
            "TRANSPORT", "Transport",
            "trns",
            self._fields_for("Total Energy Consumption from Transportation"),
            "direct", "PJ", 1000, "exact",
            "Consumption variable (point-of-use)",
        ))
        rows.append(self._row(
            "RESIDENT", "Total final energy consumption",
            "RESIDENT", "Residential",
            "scoe",
            # [f for f in scoe_cat_fields if "residential" in f],
            ["energy_consumption_scoe_residential"],
            "direct", "PJ", 1000, "exact",
            "Consumption variable (point-of-use)",
        ))
        rows.append(self._row(
            "COMMPUB", "Total final energy consumption",
            "COMMPUB", "Commercial and public services",
            "scoe",
            # [f for f in scoe_cat_fields if "commercial_municipal" in f],
            ["energy_consumption_scoe_commercial_municipal"],
            "direct", "PJ", 1000, "exact",
            "Consumption variable (point-of-use)",
        ))
        rows.append(self._row(
            "AGRICULT", "Total final energy consumption",
            "AGRICULT", "Agriculture / forestry",
            "scoe",
            # [f for f in scoe_cat_fields if "other_se" in f],
            ["energy_consumption_scoe_other_se"],
            "direct", "PJ", 1000, "partial",
            "SISEPUEDE other_se is a residual; partial match only",
        ))

        return rows


    def _rows_industry_by_source(self) -> List[Dict[str, Any]]:
        """
        Build crosswalk rows for Industry Total Final Consumption by Source.

        IEA provides fuel-level breakdowns for industry. SISEPUEDE INEN
        outputs consumption by industrial sub-category, not by fuel. The
        only per-fuel output for INEN is energy_demand_enfu_subsector_total_pj
        _inen_$CAT-FUEL$ (recorded in the ENFU subsector). For non-electric
        fuels demand equals consumption. For electricity both IEA TFC and
        SISEPUEDE demand measure at point of delivery before T&D losses, so
        they are directly comparable.
        """

        rows = self._section("INDUSTRY TOTAL FINAL CONSUMPTION BY SOURCE")

        for line in [
            "-- IEA provides consumption by fuel. SISEPUEDE INEN outputs consumption "
            "by sub-industry --",
            "-- (energy_consumption_inen_$CAT-INDUSTRY$) and a total "
            "(energy_consumption_inen_total) --",
            "-- but NOT by fuel. The only per-fuel breakdown is "
            "energy_demand_enfu_subsector_total_pj_inen_$CAT-FUEL$ --",
            "-- (fuel demanded by INEN from the ENFU subsector, recorded as a "
            "model output). --",
            "-- For non-electric fuels demand = consumption: the fuel is burned "
            "at the point of demand. --",
            "-- For electricity: both IEA TFC and SISEPUEDE demand measure at "
            "point of delivery to industry --",
            "-- (before T&D losses), so they are directly comparable. --",
        ]:
            rows.append(self._note_row(line))
        rows.append({k: None for k in _COLS})

        # # sum all industrial sources in IEA and equal to energy_consumption_inen_{...s}
        # inen_var  = "Energy Consumption from Industrial Energy"
        inen_var  = "Energy Demand by Fuel in Industrial Energy"
        note_base = "No per-fuel consumption output in INEN; using ENFU demand output"

        for product_code, product_name, fuel_cats, agg, quality, extra in [
            (
                "COAL",    "Coal",
                fuel_keys("COAL"),
                agg_type("COAL"), "approximate",
                "demand = consumption for solid fuels; includes coke",
            ),
            (
                "OIL",     "Oil",
                fuel_keys("OIL"),
                agg_type("OIL"), "approximate",
                "IEA Oil = all liquid petroleum products",
            ),
            (
                "NATGAS",  "Natural gas",
                fuel_keys("NATGAS"),
                agg_type("NATGAS"), "approximate",
                "demand = consumption for gas",
            ),
            (
                "ELECTR",  "Electricity",
                fuel_keys("ELECTR"),
                agg_type("ELECTR"), "approximate",
                "both IEA and SISEPUEDE measure electricity at point of "
                "delivery (comparable); energy_consumption_electricity_inen_total "
                "is a true consumption var but carries no fuel label",
            ),
            (
                "BIOWASTE", "Biofuels and waste",
                fuel_keys("BIOWASTE"),
                agg_type("BIOWASTE"), "approximate",
                "sum all biogenic fuels",
            ),
            (
                "GEOTHERM", "Geothermal",
                fuel_keys("GEOTHERM"),
                agg_type("GEOTHERM"), "approximate", "",
            ),
            (
                "WINDSOLAR", "Wind solar etc.",
                fuel_keys("WINDSOLAR"),
                agg_type("WINDSOLAR"), "approximate",
                "rare in industry but included for completeness",
            ),
        ]:
            note = f"{note_base}; {extra}" if extra else note_base
            rows.append(self._row(
                "INDUSTRY",
                "Industry total final consumption by source",
                product_code, product_name,
                "inen",
                self._fields_for(inen_var, fuel_cats),
                agg, "PJ", 1000, quality, note,
            ))

        return rows


    def _rows_residential_by_source(self) -> List[Dict[str, Any]]:
        """
        Build crosswalk rows for Residential Total Final Consumption by Source.

        Same structural gap as INEN: there is no per-fuel consumption output for
        SCOE.  The ENFU demand variable energy_demand_enfu_subsector_total_pj
        _scoe_$CAT-FUEL$ aggregates ALL scoe sub-categories (residential +
        commercial_municipal + other_se) and cannot be split by sub-category.
        Mapping quality is therefore partial.
        """

        rows = self._section("RESIDENTIAL TOTAL FINAL CONSUMPTION BY SOURCE")

        for line in [
            "-- IEA provides consumption by fuel for residential only. SISEPUEDE "
            "has the same structural --",
            "-- gap as INEN: no per-fuel consumption output for SCOE. The ENFU "
            "demand variables for scoe --",
            "-- (energy_demand_enfu_subsector_total_pj_scoe_$CAT-FUEL$) aggregate "
            "ALL scoe sub-categories --",
            "-- (residential + commercial_municipal + other_se) and cannot be "
            "split by sub-category. --",
            "-- Mapping quality is partial: numbers will be larger than IEA "
            "residential alone. --",
            "-- For sector totals use energy_consumption_scoe_residential "
            "(true consumption by sub-category). --",
        ]:
            rows.append(self._note_row(line))
        rows.append({k: None for k in _COLS})

        # # sum all residential sources in IEA and equal to energy_consumption_scoe_residential
        # scoe_var  = "Energy Consumption from SCOE" 
        scoe_var  = "Energy Demand by Fuel in SCOE"
        note_scoe = (
            "No per-fuel consumption output in SCOE; ENFU demand aggregates "
            "all scoe sub-categories (residential+commercial+other) "
            "— cannot isolate residential"
        )

        for product_code, product_name, fuel_cats in [
            ("COAL",     "Coal",              fuel_keys("COAL")),
            ("OIL",      "Oil products",      fuel_keys("OIL")),
            ("NATGAS",   "Natural gas",       fuel_keys("NATGAS")),
            ("ELECTR",   "Electricity",       fuel_keys("ELECTR")),
            ("GEOTHERM", "Geothermal",        fuel_keys("GEOTHERM")),
            ("BIOWASTE", "Biofuels and waste",fuel_keys("BIOWASTE")),
        ]:
            rows.append(self._row(
                "RESIDENT",
                "Residential total final consumption by source",
                product_code, product_name,
                "scoe",
                self._fields_for(scoe_var, fuel_cats),
                agg_type(product_code), "PJ", 1000, "partial", note_scoe,
            ))

        return rows


    def _rows_commercial_by_source(self) -> List[Dict[str, Any]]:
        """
        Build crosswalk rows for Commercial and Public Services Total Final
        Consumption by Source.

        Same ENFU-only structural gap as residential. The rows point to the
        same scoe-aggregate ENFU demand variables. Do not sum residential +
        commercial rows to avoid double-counting.
        """

        rows = self._section(
            "COMMERCIAL AND PUBLIC SERVICES TOTAL FINAL CONSUMPTION BY SOURCE"
        )

        for line in [
            "-- Same structural gap as residential: ENFU scoe demand is the "
            "only per-fuel output --",
            "-- available and it covers all scoe sub-categories combined. "
            "The SISEPUEDE variable --",
            "-- energy_consumption_scoe_commercial_municipal is a true "
            "consumption output but has --",
            "-- no fuel breakdown. These rows map the same scoe-aggregate "
            "ENFU demand as the --",
            "-- residential rows above; do not sum residential + commercial "
            "rows to avoid double-counting. --",
        ]:
            rows.append(self._note_row(line))
        rows.append({k: None for k in _COLS})

        # # sum all commercial sources in IEA and equal to energy_consumption_scoe_commercial_municipal
        # scoe_var  = "Energy Consumption from SCOE" 
        scoe_var  = "Energy Demand by Fuel in SCOE"
        note_comm = (
            "No per-fuel consumption output in SCOE; same scoe-wide ENFU "
            "demand as residential rows — do not sum with residential"
        )

        for product_code, product_name, fuel_cats in [
            ("COAL",     "Coal",              fuel_keys("COAL")),
            ("OIL",      "Oil products",      fuel_keys("OIL")),
            ("NATGAS",   "Natural gas",       fuel_keys("NATGAS")),
            ("ELECTR",   "Electricity",       fuel_keys("ELECTR")),
            ("GEOTHERM", "Geothermal",        fuel_keys("GEOTHERM")),
            ("BIOWASTE", "Biofuels and waste",fuel_keys("BIOWASTE")),
        ]:
            rows.append(self._row(
                "COMMPUB",
                "Commercial and public services total final consumption by source",
                product_code, product_name,
                "scoe",
                self._fields_for(scoe_var, fuel_cats),
                agg_type(product_code), "PJ", 1000, "partial", note_comm,
            ))

        return rows


    def _rows_transport_by_source(self) -> List[Dict[str, Any]]:
        """
        Build crosswalk rows for Transport Total Final Consumption by Source.

        Transport IS different from INEN/SCOE: SISEPUEDE outputs true
        energy_consumption_trns_{mode}_{fuel} variables for each mode x fuel
        combination.  These are summed across all modes to match IEA's
        fuel-level total for the transport sector.
        """

        rows = self._section("TRANSPORT TOTAL FINAL CONSUMPTION BY SOURCE")

        for line in [
            "-- Transport IS different: SISEPUEDE has "
            "energy_consumption_trns_{mode}_{fuel} --",
            "-- which are true consumption variables. "
            "These are summed across all modes here. --",
        ]:
            rows.append(self._note_row(line))
        rows.append({k: None for k in _COLS})

        for product_code, product_name, trns_var_names, agg, quality, notes in [
            (
                "OIL", "Oil products",
                [
                    "Transportation Modal Energy Consumption from Diesel",
                    "Transportation Modal Energy Consumption from Gasoline",
                    "Transportation Modal Energy Consumption from Kerosene",
                    "Transportation Modal Energy Consumption from Hydrocarbon Gas Liquids",
                ],
                "sum", "approximate",
                "Consumption variables summed across all modes; "
                "IEA Oil products = diesel+gasoline+kerosene+hgl aggregate",
            ),
            (
                "ELECTR", "Electricity",
                ["Transportation Modal Energy Consumption from Electricity"],
                "sum", "exact",
                "Consumption variables summed across all modes",
            ),
            (
                "NATGAS", "Natural gas",
                ["Transportation Modal Energy Consumption from Natural Gas"],
                "sum", "exact",
                "Consumption variables summed across all modes",
            ),
            (
                "BIOWASTE", "Biofuels and waste",
                ["Transportation Modal Energy Consumption from Biofuels"],
                "sum", "approximate",
                "Consumption from biofuel variables summed across all modes; SSP does not output waste.",
            ),
            (
                "HYDROGEN", "Hydrogen",
                ["Transportation Modal Energy Consumption from Hydrogen"],
                "sum", "exact",
                "Consumption variables summed across all modes",
            ),
        ]:
            fields: List[str] = []
            for vname in trns_var_names:
                fields += self._fields_for(vname)

            rows.append(self._row(
                "TRANSPORT",
                "Transport total final consumption by source",
                product_code, product_name,
                "trns",
                fields,
                agg, "PJ", 1000, quality, notes,
            ))

        return rows


    def _rows_industry_subsectors(self) -> List[Dict[str, Any]]:
        """
        Build crosswalk rows for Industry Sub-Sector Consumption.

        Maps IEA industry sub-sector codes to SISEPUEDE INEN industrial
        category fields (energy_consumption_inen_$CAT-INDUSTRY$).
        Uses true consumption output variables (not demand proxies).
        """

        rows = self._section(
            "INDUSTRY SUB-SECTOR CONSUMPTION (where IEA sub-sector data available)"
        )

        ##  Build a mapping: cat_name -> field_name for all INEN categories

        inen_cat_fields = self._fields_for("Energy Consumption from Industrial Energy")
        inen_cat_map: Dict[str, str] = {}
        for field in inen_cat_fields:
            # field pattern: energy_consumption_inen_{cat}
            parts = field.split("energy_consumption_inen_")
            if len(parts) == 2:
                inen_cat_map[parts[1]] = field

        def cat_fields(cats: List[str]) -> List[str]:
            return [inen_cat_map[c] for c in cats if c in inen_cat_map]

        for iea_code, iea_name, inen_cats, agg, quality, notes in [
            (
                "IRONSTL",  "Iron and steel",
                ["metals"],
                "direct", "partial",
                "fuel_metals covers all metals; no iron/steel split in SISEPUEDE",
            ),
            (
                "CHEMICAL", "Chemical and petrochemical",
                ["chemicals"],
                "direct", "approximate", "",
            ),
            (
                "NONMET",   "Non-metallic minerals",
                ["cement", "glass", "lime_and_carbonite"],
                "sum", "approximate",
                "IEA non-metallic minerals ~ "
                "SISEPUEDE cement + glass + lime_and_carbonite",
            ),
            (
                "MINING", "Mining and quarrying",
                ["mining"],
                "direct", "exact", "",
            ),
            (
                "PAPERPRO", "Paper pulp and printing",
                ["paper"],
                "direct", "approximate", "",
            ),
            (
                "WOODPRO", "Wood and wood products",
                ["wood"],
                "direct", "approximate", "",
            ),
            (
                "TEXTILES", "Textile and leather",
                ["textiles", "rubber_and_leather"],
                "sum", "approximate", "",
            ),
            (
                "CONSTRUC", "Construction",
                [],
                "none", "no_match",
                "No construction sub-sector in SISEPUEDE inen",
            ),
            (
                "INONSPEC", "Non-specified industry",
                ["other_product_manufacturing"],
                "direct", "partial",
                "other_product_manufacturing is a residual; partial match only",
            ),
        ]:
            rows.append(self._row(
                iea_code,
                "Industry total final consumption by source",
                iea_code, iea_name,
                "inen",
                cat_fields(inen_cats),
                agg, "PJ", 1000, quality, notes,
            ))

        return rows


    ##############################
    #    PUBLIC INTERFACE        #
    ##############################

    def build(self,
        write_csv: bool = True,
    ) -> pd.DataFrame:
        """
        Build the IEA <-> SISEPUEDE crosswalk DataFrame.

        Assembles all crosswalk sections in order, verifies that all SISEPUEDE
        field names exist, and optionally writes the result to CSV.

        Keyword Arguments
        -----------------
        write_csv : bool
            If True, write the crosswalk to the path resolved from
            model_attributes.dir_ref (or self._path_out if set).

        Returns
        -------
        pd.DataFrame
            The complete crosswalk, including section-header rows.
        """

        ##  ASSEMBLE ALL SECTIONS

        all_rows: List[Dict[str, Any]] = []
        # all_rows += self._rows_supply() # keep commented out until clarifying what variables this exactly maps to
        all_rows += self._rows_imports_exports()
        all_rows += self._rows_electricity_generation()
        all_rows += self._rows_tfc_by_sector()
        all_rows += self._rows_industry_by_source()
        all_rows += self._rows_residential_by_source()
        all_rows += self._rows_commercial_by_source()
        all_rows += self._rows_transport_by_source()
        # all_rows += self._rows_industry_subsectors() # we dont have this level of info (yet)

        df = pd.DataFrame(all_rows, columns = _COLS)

        ##  OPTIONALLY WRITE

        if write_csv:
            path = self._resolve_out_path()
            df.to_csv(path, index = False)
            print(f"IEA crosswalk written to: {path}")

        return df


    def _resolve_out_path(self,
    ) -> str:
        """Return the output CSV path, auto-locating from model_attributes."""

        if self._path_out is not None:
            return self._path_out

        dir_ref = getattr(self.model_attributes, "dir_ref", None)
        if dir_ref is None:
            raise FileNotFoundError(
                "Could not locate dir_ref from model_attributes. "
                "Pass path_out explicitly to IEACrosswalkBuilder()."
            )

        return os.path.join(dir_ref, _CROSSWALK_REL_PATH)
