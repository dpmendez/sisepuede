"""
sisepuede/calibration/iea_data_loader.py

IEADataLoader
-------------
Load per-country IEA energy data from the team's local data repository
(a collection of individual CSV files, one file per country per topic).

The loader reads each relevant folder, parses the country's file (handling
the two CSV formats found in the repo), maps row labels to the
(iea_balance_code, iea_product_code) codes used in the crosswalk, converts
all values to TJ, and returns a single long DataFrame ready to be passed
directly to IEACrosswalk.build_comparison().

Typical usage
-------------
    from sisepuede.manager.sisepuede_file_structure import SISEPUEDEFileStructure
    from sisepuede.calibration.iea_data_loader import IEADataLoader
    from sisepuede.calibration.iea_crosswalk import IEACrosswalk

    file_struct      = SISEPUEDEFileStructure()
    model_attributes = file_struct.model_attributes

    loader = IEADataLoader("/path/to/data_collection_temporary", model_attributes)
    xw     = IEACrosswalk(model_attributes)

    df_iea           = loader.load_country("LBY")
    df_ssp           = xw.aggregate_sisepuede(df_out_energycon, col_year="year")
    df_comparison    = xw.build_comparison(df_ssp, df_iea)

Data format notes
-----------------
The repository contains two CSV layouts:

  Format A (most folders):
      "<topic> in <Country>", Value, Year, Units    <- header row
      "<Row label>",          12345, 2000, TJ       <- data rows
      ...

  Format B (production folders):
      Year, "<Metric, Country>", Units    <- header row
      2000, 12345,               TJ       <- data rows

Detection: if the first column header (stripped) == "Year" -> Format B.

Unit handling:
  TJ   -> pass through (x1)
  GWh  -> x3.6
  ktoe -> x41.868

Exports sign convention:
  IEA stores exports as negative numbers. Values are returned as-is.
  Take abs() if you need magnitudes.
"""

import os
import re
import numpy as np
import pandas as pd
import sisepuede.core.support_classes as sc
from typing import *

####################
#    CONSTANTS     #
####################

_GWH_TO_TJ  = 3.6       # 1 GWh  = 3.6  TJ
_KTOE_TO_TJ = 41.868    # 1 ktoe = 41.868 TJ

##  Row label -> iea_product_code
_ROW_LABEL_TO_PRODUCT: Dict[str, str] = {
    # Energy products (used in supply / sectoral-by-source / electricity folders)
    "coal":                                 "COAL",
    "oil":                                  "OIL",
    "oil products":                         "OIL",
    "natural gas":                          "NATGAS",
    "nuclear":                              "NUCLEAR",
    "hydro":                                "HYDRO",
    "wind solar etc.":                      "WINDSOLAR",
    "wind":                                 "WIND",
    "solar pv":                             "SOLARPV",
    "solar thermal":                        "SOLARTH",
    "biofuels and waste":                   "BIOWASTE",
    "biofuels":                             "BIOFUEL",
    "waste":                                "WASTE",
    "geothermal":                           "GEOTHERM",
    "tide":                                 "TIDE",
    "electricity":                          "ELECTR",
    "electricity (outdated, before dc/ac)": "ELECTR", # e.g. Peru
    "total":                                "TOTAL",
    "losses":                               "LOSSES",
    "hydrogen":                             "HYDROGEN",
    # Imports / exports rows
    "imports":                              "IMPORTS",
    "exports":                              "EXPORTS",
}

##  Row label -> iea_balance_code (only used when mode == "balance_from_row",
##  i.e. the total_final_energy_consumption folder where each row IS a sector)
##  IEA row labels vary by country (e.g. "Industry" vs "Industry Sector"),
##  so multiple variants are listed for each sector.
_ROW_LABEL_TO_BALANCE_TFC: Dict[str, str] = {
    "industry":                                 "INDUSTRY",
    "industry sector":                          "INDUSTRY",   # e.g. Peru
    "transport":                                "TRANSPORT",
    "transport sector":                         "TRANSPORT",  # e.g. Peru
    "residential":                              "RESIDENT",
    "commercial and public services":           "COMMPUB",
    "agriculture / forestry":                   "AGRICULT",
    "agriculture/forestry":                     "AGRICULT",   # e.g. Peru (no spaces)
    "agriculture":                              "AGRICULT",
}

##  Corresponding product codes for the TFC-sector rows
_ROW_LABEL_TO_PRODUCT_TFC: Dict[str, str] = {
    "industry":                                 "INDUSTRY",
    "industry sector":                          "INDUSTRY",
    "transport":                                "TRANSPORT",
    "transport sector":                         "TRANSPORT",
    "residential":                              "RESIDENT",
    "commercial and public services":           "COMMPUB",
    "agriculture / forestry":                   "AGRICULT",
    "agriculture/forestry":                     "AGRICULT",
    "agriculture":                              "AGRICULT",
}

##  ISO-3 -> IEA filename country name
##  IEA filenames use short common names (e.g. "Tanzania", "Korea") while the
##  SISEPUEDE region names use long official forms ("united_republic_of_tanzania")
##  so region.replace("_", " ").title() search string can miss files.
##  Add an entry here whenever an ISO maps to a SISEPUEDE region name that does
##  not appear as a substring in the IEA filename.
_ISO_TO_IEA_FILENAME: Dict[str, str] = {
    "TZA": "Tanzania",
    "IRN": "Iran",
    "RUS": "Russia",
    "BOL": "Bolivia",
    "VEN": "Venezuela",
    "VNM": "Viet Nam",
    "KOR": "Korea",
    "PRK": "Korea, Democratic People's Republic of",
    "MDA": "Moldova",
    "USA": "United States",
    "COD": "Democratic Republic of the Congo",
    "COG": "Congo",
    "SYR": "Syrian Arab Republic",
}

##  Folder -> parsing configuration
##    format         : "A" (row-label format) or "B" (year-leading format)
##    balance_fixed  : iea_balance_code shared by every row (None if determined per-row)
##    product_fixed  : iea_product_code shared by every row (Format B only)
##    mode           : how to determine (balance, product) from each row:
##                       "product_from_row"  — balance_fixed + row_label -> product
##                       "balance_from_row"  — row_label -> balance AND product (TFC)
##                       "imports_exports"   — row_label ("Imports"/"Exports") +
##                                            balance_prefix -> balance and product
##                       "fixed"            — both balance_fixed and product_fixed set (B)
##    balance_prefix : prefix for imports_exports mode (e.g. "COAL" -> COALIMPORTS)
_FOLDER_CONFIG: Dict[str, Dict] = {
    # "total_energy_supply": {
    #     "format":         "A",
    #     "balance_fixed":  "INDPROD",
    #     "mode":           "product_from_row",
    # },
    "total_final_energy_consumption": {
        "format":         "A",
        "balance_fixed":  None,
        "mode":           "balance_from_row",
    },
    "electricity_generation_sources": {
        "format":         "A",
        "balance_fixed":  "ELECTOUT",
        "mode":           "product_from_row",
    },
    "industry_total_final_consumption_by_source": {
        "format":         "A",
        "balance_fixed":  "INDUSTRY",
        "mode":           "product_from_row",
    },
    "residential_total_final_consumption_by_source": {
        "format":         "A",
        "balance_fixed":  "RESIDENT",
        "mode":           "product_from_row",
    },
    "commercial_and_public_services_total_final_consumption_by_source": {
        "format":         "A",
        "balance_fixed":  "COMMPUB",
        "mode":           "product_from_row",
    },
    "transport_total_final_consumption_by_source": {
        "format":         "A",
        "balance_fixed":  "TRANSPORT",
        "mode":           "product_from_row",
    },
    "coal_imports_and_exports": {
        "format":         "A",
        "balance_fixed":  None,
        "mode":           "imports_exports",
        "balance_prefix": "COAL",
    },
    "crude_oil_imports_and_exports": {
        "format":         "A",
        "balance_fixed":  None,
        "mode":           "imports_exports",
        "balance_prefix": "OIL",
    },
    "natural_gas_imports_and_exports": {
        "format":         "A",
        "balance_fixed":  None,
        "mode":           "imports_exports",
        "balance_prefix": "GAS",
    },
    "electricity_imports_and_exports": {
        "format":         "A",
        "balance_fixed":  None,
        "mode":           "imports_exports",
        "balance_prefix": "EL",
    },
    "energy_imports_and_exports": {
        "format":         "A",
        "balance_fixed":  None,
        "mode":           "imports_exports",
        "balance_prefix": "",          # -> IMPORTS / EXPORTS (no prefix)
    },
    # Production folders (coal_production, crude_oil_production,
    # natural_gas_production) are omitted here.  Each gives only domestic
    # production for a single fuel; the total_energy_supply folder already
    # covers the same (INDPROD, product) pairs at the TES level.  Including
    # both would create duplicate rows for (INDPROD, COAL/OIL/NATGAS).
    #
    # Folders not in the crosswalk — also skipped:
    #   total_oil_products_refined, electricity_final_consumption_by_sector
}



####################
#    PRIMARY CLASS #
####################

class IEADataLoader:
    """
    Load per-country IEA energy data from the local CSV repository.

    Reads each folder in the repository, finds the file for the requested
    country, parses it (Format A or B), converts values to TJ, and returns
    a long DataFrame with columns:

        iso_alpha_3, iea_balance_code, iea_product_code, year, value_iea_tj

    This DataFrame can be passed directly to IEACrosswalk.build_comparison().
    """

    def __init__(self,
        path_data_dir: str,
        model_attributes,
        folders_to_skip: Union[List[str], None] = None,
        ) -> None:
        """
        Function Arguments
        ------------------
        path_data_dir : str
        Path to the root of the IEA data repository
        (the directory containing subfolders like total_energy_supply/, etc.).

        model_attributes : ModelAttributes
        SISEPUEDE model attributes object (used for ISO <-> region mapping).

        Keyword Arguments
        -----------------
        folders_to_skip : Union[List[str], None]
            Additional folder names to exclude from loading.
            Always skips total_oil_products_refined and
            electricity_final_consumption_by_sector.
        """

        self.model_attributes = model_attributes

        self._initialize_regions()
        self._initialize_paths(path_data_dir, folders_to_skip)

        return None

    ##################################
    #    INITIALISATION METHODS      #
    ##################################
    
    def _initialize_regions(self,
    ) -> None:
        """
        Set self.regions and build a reverse ISO -> region-name lookup.
        """
        
        self.regions = sc.Regions(self.model_attributes)

        ##  Build reverse map: ISO-3 (upper) -> SISEPUEDE region name
        ##  e.g. "LBY" -> "libya"
        self.dict_iso_to_region = {
            k.upper(): v
            for k, v in self.regions.dict_iso_to_region.items()
        }
        # self.dict_iso_to_region = self.regions.dict_iso_to_region

        return None
    
    def _initialize_paths(self,
        path_data_dir: str,
        folders_to_skip: Union[List[str], None],
    ) -> None:
        """
        Validate the data directory and build the set of active folder configs.

        Function Arguments
        ------------------
        path_data_dir : str
            Root directory of the IEA data repository.
        folders_to_skip : Union[List[str], None]
            Extra folders to exclude.
        """
        
        if not os.path.isdir(path_data_dir):
            raise FileNotFoundError(
                f"IEA data directory not found: {path_data_dir}"
            )
        
        self.path_data_dir = path_data_dir

        # Build the active subset of _FOLDER_CONFIG
        _always_skip = {
            "total_oil_products_refined",
            "electricity_final_consumption_by_sector",
        }
        skip_set = _always_skip | set(folders_to_skip or [])

        self.folder_config = {
            folder: cfg
            for folder, cfg in _FOLDER_CONFIG.items()
            if folder not in skip_set
        }

        return None
    

    ##################################
    #    INTERNAL HELPERS            #
    ##################################

    @staticmethod
    def _unit_to_tj(value: float,
        unit: str,
    ) -> float:

        """Convert a single IEA value to TJ.

        Function Arguments
        ------------------
        value : float
            Raw IEA value.
        unit : str
            IEA unit string (e.g. "TJ", " TJ", "GWh", "ktoe").
        """

        if pd.isna(value):
            return np.nan

        unit_lc = str(unit).strip().lower()
        if "gwh" in unit_lc:
            return float(value) * _GWH_TO_TJ
        if "ktoe" in unit_lc:
            return float(value) * _KTOE_TO_TJ
        
        return float(value) # assume TJ
    
    def _region_name_from_iso(self,
        iso: str,
    ) -> Union[str, None]:
        
        """Return the SISEPUEDE region name (e.g. 'libya') for an ISO-3 code.

        Function Arguments
        ------------------
        iso : str
            ISO-3 country code (e.g. 'LBY').
        """

        return self.dict_iso_to_region.get(iso.upper())
    
    def _find_file_in_folder(self,
        folder: str,
        country_str: str,
    ) -> Union[str, None]:
        """Search a folder for a file whose name contains country_str
        (case-insensitive).  Returns the full path, or None if not found.

        If multiple files match, returns the first match and emits a warning.

        Function Arguments
        ------------------
        folder : str
            Folder name within self.path_data_dir.
        country_str : str
            Country name to search for (e.g. 'Libya').
        """

        folder_path = os.path.join(self.path_data_dir, folder)
        if not os.path.isdir(folder_path):
            return None
        
        pattern = re.compile(re.escape(country_str), re.IGNORECASE)
        matches = [
            f for f in os.listdir(folder_path)
            if f.endswith(".csv") and pattern.search(f)
        ]

        if not matches:
            return None
        
        if len(matches) > 1:
            import warnings

            warnings.warn(
                f"Multiple files match '{country_str}' in {folder}/: "
                f"{matches}. Using the first: {matches[0]}",
                stacklevel = 3,
            )

        return os.path.join(folder_path, matches[0])


    def _detect_format(self,
        path: str,
    ) -> str:
        """Return 'A' or 'B' based on the first column header of the CSV.

        Format B: first column header is 'Year' (case-insensitive, stripped).
        Format A: everything else.

        Function Arguments
        ------------------
        path : str
            Path to the CSV file.
        """

        with open(path, "r", encoding = "utf-8", errors = "replace") as fh:
            header_line = fh.readline()
        
        first_col = header_line.split(",")[0].strip().strip("'")
        if first_col.lower() == "year":
            return "B"
        
        return "A"
    
    def _parse_format_a(self,
        path: str,
        cfg: Dict,
    ) -> pd.DataFrame:
        """Parse a Format-A file (row-label, value, year, units).

        Returns a DataFrame with columns:
            iea_balance_code, iea_product_code, year, value_iea_tj

        Function Arguments
        ------------------
        path : str
            Path to the CSV file.
        cfg : Dict
            Folder configuration entry from self.folder_config.
        """

        df = pd.read_csv(path, header = 0)
        if df.shape[1] < 4:
            return pd.DataFrame()
        
        # Standardise column names
        df.columns = ["row_label", "value", "year", "units"] + list(df.columns[4:])

        df["row_label"] = (
            df["row_label"]
            .astype(str)
            .str.strip()
            .str.strip("'")
            .str.strip()
        )
        df["units"] = df["units"].astype(str).str.strip()
        df["value"]  = pd.to_numeric(df["value"],  errors = "coerce")
        df["year"]   = pd.to_numeric(df["year"],   errors = "coerce").astype("Int64")
        df = df.dropna(subset = ["value", "year"])

        ##  Map row labels to (balance_code, product_code) based on mode
        mode   = cfg["mode"]
        prefix = cfg.get("balance_prefix", "")

        rows = []
        for _, row in df.iterrows():
            label_lc = row["row_label"].lower()

            if mode == "product_from_row":
                product_code = _ROW_LABEL_TO_PRODUCT.get(label_lc)
                if product_code is None:
                    continue
                balance_code = cfg["balance_fixed"]

            elif mode == "balance_from_row":
                balance_code = _ROW_LABEL_TO_BALANCE_TFC.get(label_lc)
                product_code = _ROW_LABEL_TO_PRODUCT_TFC.get(label_lc)
                if balance_code is None or product_code is None:
                    continue

            elif mode == "imports_exports":
                if label_lc == "imports":
                    balance_code = f"{prefix}IMPORTS" if prefix else "IMPORTS"
                    product_code = "IMPORTS"
                elif label_lc == "exports":
                    balance_code = f"{prefix}EXPORTS" if prefix else "EXPORTS"
                    product_code = "EXPORTS"
                else:
                    continue

            else:
                continue   # unknown mode

            value_tj = self._unit_to_tj(row["value"], row["units"])

            rows.append({
                "iea_balance_code":  balance_code,
                "iea_product_code":  product_code,
                "year":              int(row["year"]),
                "value_iea_tj":      value_tj,
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)


    def _parse_format_b(self,
        path: str,
        cfg: Dict,
    ) -> pd.DataFrame:
        """Parse a Format-B file (year-leading, single metric column).

        Returns a DataFrame with columns:
            iea_balance_code, iea_product_code, year, value_iea_tj

        Function Arguments
        ------------------
        path : str
            Path to the CSV file.
        cfg : Dict
            Folder configuration entry from self.folder_config.
        """

        df = pd.read_csv(path, header = 0)

        if df.shape[1] < 3:
            return pd.DataFrame()

        ##  Column layout: Year | value | Units
        df.columns = ["year", "value", "units"] + list(df.columns[3:])

        df["year"]  = pd.to_numeric(df["year"],  errors = "coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors = "coerce")
        df["units"] = df["units"].astype(str).str.strip()
        df = df.dropna(subset = ["value", "year"])

        balance_code = cfg["balance_fixed"]
        product_code = cfg["product_fixed"]

        df["iea_balance_code"] = balance_code
        df["iea_product_code"] = product_code
        df["value_iea_tj"] = df.apply(
            lambda r: self._unit_to_tj(r["value"], r["units"]),
            axis = 1,
        )
        df["year"] = df["year"].astype(int)

        return df[["iea_balance_code", "iea_product_code", "year", "value_iea_tj"]]


    ##################################
    #    PUBLIC METHODS              #
    ##################################

    def load_country(self,
        iso: str,
        country_name: Union[str, None] = None,
    ) -> pd.DataFrame:
        """
        Load all available IEA energy data for one country.

        Scans every configured folder, finds the country's file, parses it,
        converts to TJ, and concatenates into a single long DataFrame.

        Function Arguments
        ------------------
        iso : str
            ISO-3 country code (e.g. 'LBY').

        Keyword Arguments
        -----------------
        country_name : Union[str, None]
            Country name as it appears in IEA filenames (e.g. 'Libya').
            If None, derived from iso using the SISEPUEDE regions mapping
            (capitalised).  Pass explicitly when the automatic derivation
            fails (e.g. 'United Arab Emirates' vs the region name
            'united_arab_emirates').
        """

        iso = iso.upper()

        ##  Resolve country name for filename search
        if country_name is None:
            ##  Prefer the explicit IEA-filename alias when one is registered:
            ##  some ISO codes map to SISEPUEDE region names (e.g.
            ##  "united_republic_of_tanzania") that don't appear in the IEA
            ##  filenames (which use "Tanzania").
            country_name = _ISO_TO_IEA_FILENAME.get(iso)

            if country_name is None:
                region = self._region_name_from_iso(iso)
                if region is None:
                    raise ValueError(
                        f"ISO '{iso}' not found in SISEPUEDE regions. "
                        "Pass country_name explicitly."
                    )
                ##  Convert region name to title case for filename matching
                ##  e.g. "united_arab_emirates" -> "United Arab Emirates"
                country_name = region.replace("_", " ").title()

        ##  Load and parse each folder
        frames = []
        for folder, cfg in self.folder_config.items():

            path = self._find_file_in_folder(folder, country_name)
            if path is None:
                continue   # country absent from this folder — normal

            fmt = cfg.get("format") or self._detect_format(path)

            try:
                if fmt == "B":
                    df_parsed = self._parse_format_b(path, cfg)
                else:
                    df_parsed = self._parse_format_a(path, cfg)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Failed to parse {path}: {exc}",
                    stacklevel = 2,
                )
                continue

            if len(df_parsed) > 0:
                df_parsed["source_folder"] = folder
                frames.append(df_parsed)

        if not frames:
            return pd.DataFrame(columns = [
                "iso_alpha_3", "iea_balance_code",
                "iea_product_code", "year", "value_iea_tj", "source_folder",
            ])

        df_all = pd.concat(frames, ignore_index = True)
        df_all.insert(0, "iso_alpha_3", iso)

        return (
            df_all
            .sort_values(["iea_balance_code", "iea_product_code", "year"])
            .reset_index(drop = True)
        )


    def available_folders(self,
    ) -> List[str]:
        """Return the list of folders that will be searched when loading a country.
        """

        return sorted(self.folder_config.keys())