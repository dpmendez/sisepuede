"""
sisepuede/calibration/calibration_group.py

CalibrationGroup / CalibrationPlan
------------------------------------
Data structures that link sets of SISEPUEDE input parameters to the IEA
calibration targets they control.

Conceptually, a CalibrationGroup is a named "knob cluster":
  - specs            : the input variables that can be scaled
  - iea_targets      : the (iea_balance_code, iea_product_code) pairs to observe
  - sector           : which SISEPUEDE subsector code(s) this group belongs to
  - constraint_type  : "scalar"  — variables are independently scaled (default)
                       "simplex" — variables are fuel fractions summing to 1;
                                   shifting one requires compensating changes in
                                   the others (use shift_fuels_based_on_single_point
                                   from _lib.py when applying this group)

A CalibrationPlan holds an ordered collection of groups and provides helpers
for three calibration workflow modes:

  Bulk mode (all sectors at once):
      plan   = CalibrationPlan(groups)
      specs  = plan.get_specs()          # pass to SensitivityRunner.run_lhs()
      runner.run_lhs(specs, n_samples=50)

  Sector-by-sector mode:
      for sector, sub in plan.by_sector().items():
          specs   = sub.get_specs()
          targets = sub.get_targets()    # observe only this sector's IEA pairs

  Linear / direct-scale mode (one group at a time):
      for group in plan.scalar_groups():
          df_in = scale_inputs_single_value(
              df_in, group.columns, xw_output_fields, iea_value, models
          )
      for group in plan.simplex_groups():
          df_in = shift_fuels_based_on_single_point(
              df_in, sector, fuel_targ, fuels_shift_out, ...
          )

Integration with existing infrastructure
-----------------------------------------
  - VariableSpec (sensitivity.py)    : specs in each group use this dataclass
  - scale_inputs_single_value (_lib.py) : use group.columns as fields_input
  - shift_fuels_based_on_single_point (_lib.py) : for simplex groups
  - IEACrosswalk.build_comparison()  : filter output with group.filter_comparison()
  - ModelAttributes.dict_field_to_simplex_group : auto-populate simplex_group_ids
    via CalibrationPlan.from_specs_dict(spec_dict, model_attributes)
"""

from __future__ import annotations

import warnings
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from sisepuede.calibration.sensitivity import VariableSpec
from sisepuede.calibration._simplex_registry import SimplexRegistry


##########################
#    CONSTANTS           #
##########################

_CONSTRAINT_TYPES: Set[str] = {"scalar", "simplex"}

#  (iea_balance_code, iea_product_code)
IEATarget = Tuple[str, str]


##########################
#    CALIBRATION GROUP   #
##########################

@dataclass
class CalibrationGroup:
    """A named cluster of input parameters that collectively influence one or
    more IEA calibration targets.

    Parameters
    ----------
    name : str
        Short identifier for this group.
        Examples: "inen_total_demand", "trns_fuel_mix", "entc_coal_share".
    sector : str | List[str]
        SISEPUEDE subsector code(s) this group belongs to.
        Examples: "inen", "trns", "scoe", "enfu", "entc".
        A list is used for cross-sector groups (rare).
    specs : List[VariableSpec]
        Input variable specifications (column name + bounds + optional simplex
        metadata).  These are the variables that will be scaled during
        calibration.
    iea_targets : List[Tuple[str, str]]
        List of (iea_balance_code, iea_product_code) pairs that this group
        is expected to influence.  Used to slice df_comparison when evaluating
        calibration error for this group.
        Example: [("INDUSTRY", "COAL"), ("INDUSTRY", "OIL")]
    constraint_type : str
        "scalar"  (default) — each variable is scaled independently within
                              [spec.lb, spec.ub].  Compatible with
                              scale_inputs_single_value() and LHS/OAT sampling.
        "simplex"           — variables are fuel fractions that must sum to 1
                              within each industry/transport category.  Cannot
                              be freely scaled; use shift_fuels_based_on_single_point()
                              from _lib.py.  The is_simplex_group and
                              simplex_group_id fields on each VariableSpec
                              carry the group metadata from ModelAttributes.
    simplex_group_ids : List[int]
        For simplex groups: the integer group IDs from
        ModelAttributes.dict_field_to_simplex_group.  One ID per simplex
        sub-group if specs span multiple industry/transport categories.
        Populated automatically by CalibrationPlan.from_specs_dict() when
        model_attributes is provided; otherwise set manually.
        Empty list for scalar groups.
    notes : str
        Free-text description of what this group represents and any caveats.
    """

    name:              str
    sector:            Union[str, List[str]]
    specs:             List[VariableSpec]
    iea_targets:       List[IEATarget]
    constraint_type:   str = "scalar"
    simplex_group_ids: List[int] = field(default_factory=list)
    notes:             str = ""

    def __post_init__(self) -> None:
        ##  Normalise sector to list
        if isinstance(self.sector, str):
            self.sector = [self.sector]

        ##  Validate constraint_type
        if self.constraint_type not in _CONSTRAINT_TYPES:
            raise ValueError(
                f"CalibrationGroup '{self.name}': constraint_type must be one of "
                f"{sorted(_CONSTRAINT_TYPES)}, got '{self.constraint_type}'."
            )

        ##  Normalise iea_targets to list of tuples
        self.iea_targets = [tuple(t) for t in self.iea_targets]

        ##  Warn if simplex group has no IDs (metadata incomplete)
        if self.constraint_type == "simplex" and not self.simplex_group_ids:
            warnings.warn(
                f"CalibrationGroup '{self.name}' has constraint_type='simplex' "
                "but simplex_group_ids is empty.  Populate from "
                "ModelAttributes.dict_field_to_simplex_group or set manually.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    #    Convenience properties                                          #
    # ------------------------------------------------------------------ #

    @property
    def columns(self) -> List[str]:
        """Column names of all specs in this group (compatible with
        scale_inputs_single_value fields_input argument)."""
        return [s.column for s in self.specs]

    @property
    def is_simplex(self) -> bool:
        """True if this group requires simplex-aware perturbation."""
        return self.constraint_type == "simplex"

    @property
    def n_specs(self) -> int:
        return len(self.specs)

    @property
    def n_targets(self) -> int:
        return len(self.iea_targets)

    # ------------------------------------------------------------------ #
    #    Filtering helpers                                               #
    # ------------------------------------------------------------------ #

    def targets_for_balance(self, iea_balance_code: str) -> List[IEATarget]:
        """Return iea_targets whose balance code matches."""
        return [t for t in self.iea_targets if t[0] == iea_balance_code]

    def filter_comparison(
        self,
        df_comparison: pd.DataFrame,
        col_balance: str = "iea_balance_code",
        col_product: str = "iea_product_code",
    ) -> pd.DataFrame:
        """Return rows of df_comparison that correspond to this group's IEA targets.

        Useful for evaluating calibration error for this specific group:

            df_group = group.filter_comparison(df_comparison)
            mean_rel_error = df_group["rel_error_iea"].abs().mean()

        Parameters
        ----------
        df_comparison : pd.DataFrame
            Output of IEACrosswalk.build_comparison().
        col_balance, col_product : str
            Column names for balance and product codes.
        """
        if not self.iea_targets:
            return df_comparison.iloc[0:0]   # empty with same schema

        mask = pd.Series(False, index=df_comparison.index)
        for bal, prod in self.iea_targets:
            mask = mask | (
                (df_comparison[col_balance] == bal)
                & (df_comparison[col_product] == prod)
            )
        return df_comparison[mask]

    # ------------------------------------------------------------------ #
    #    Display                                                           #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        sectors = "/".join(self.sector)
        return (
            f"CalibrationGroup(name={self.name!r}, sector={sectors!r}, "
            f"n_specs={self.n_specs}, n_targets={self.n_targets}, "
            f"constraint={self.constraint_type!r})"
        )


##########################
#    CALIBRATION PLAN    #
##########################

class CalibrationPlan:
    """An ordered collection of CalibrationGroups with aggregation utilities.

    A CalibrationPlan captures which input parameters to vary (specs) and what
    IEA targets to observe (iea_targets) for one calibration exercise.  Groups
    can be combined for bulk runs or split by sector for targeted tuning.

    Parameters
    ----------
    groups : List[CalibrationGroup] | None
        Initial groups.  Add more with .add() or .extend().

    Examples
    --------
    Build from a config dictionary (recommended):

        plan = CalibrationPlan.from_specs_dict(
            {
                "inen_total_demand": {
                    "sector": "inen",
                    "specs": [
                        VariableSpec("consumpinit_inen_energy_tj_per_tonne_production_metals", 0.5, 2.0),
                        VariableSpec("consumpinit_inen_energy_tj_per_tonne_production_chemicals", 0.5, 2.0),
                    ],
                    "iea_targets": [("INDUSTRY", "INDUSTRY")],
                    "notes": "Scale initial demand for each industrial category",
                },
                "inen_coal_fraction": {
                    "sector": "inen",
                    "constraint_type": "simplex",
                    "specs": [
                        VariableSpec("frac_inen_energy_cement_coal", 0.5, 1.5,
                                     is_simplex_group=True),
                        VariableSpec("frac_inen_energy_cement_oil",  0.5, 1.5,
                                     is_simplex_group=True),
                    ],
                    "iea_targets": [("INDUSTRY", "COAL"), ("INDUSTRY", "OIL")],
                    "notes": "Fuel mix fractions for cement; must sum to 1",
                },
            },
            model_attributes = model_attributes,   # auto-fills simplex_group_ids
        )

    Bulk run (all sectors):
        specs = plan.get_specs()
        result = runner.run_lhs(specs, n_samples=100)

    Sector-by-sector run:
        for sector, sub in plan.by_sector().items():
            result = runner.run_lhs(sub.get_specs(), n_samples=50)
            # evaluate against sub.get_targets() only

    Direct-scale run (one group at a time):
        for group in plan.scalar_groups():
            df_input = scale_inputs_single_value(
                df_input, group.columns, xw_output_fields, iea_val, models
            )
    """

    def __init__(
        self,
        groups: Optional[List[CalibrationGroup]] = None,
    ) -> None:
        self._groups: List[CalibrationGroup] = list(groups or [])

    # ------------------------------------------------------------------ #
    #    Mutation — fluent interface                                     #
    # ------------------------------------------------------------------ #

    def add(self, group: CalibrationGroup) -> "CalibrationPlan":
        """Append one group and return self."""
        self._groups.append(group)
        return self

    def extend(self, groups: List[CalibrationGroup]) -> "CalibrationPlan":
        """Append multiple groups and return self."""
        self._groups.extend(groups)
        return self

    # ------------------------------------------------------------------ #
    #    Access                                                          #
    # ------------------------------------------------------------------ #

    @property
    def groups(self) -> List[CalibrationGroup]:
        """Ordered list of all groups (copy)."""
        return list(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __iter__(self):
        return iter(self._groups)

    def __repr__(self) -> str:
        return f"CalibrationPlan({len(self._groups)} groups)"

    # ------------------------------------------------------------------ #
    #    Flattening                                                      #
    # ------------------------------------------------------------------ #

    def get_specs(
        self,
        sectors: Optional[Union[str, List[str]]] = None,
        constraint_type: Optional[str] = None,
    ) -> List[VariableSpec]:
        """Flat, de-duplicated list of VariableSpecs across selected groups.

        Pass the result directly to SensitivityRunner.run_lhs() or run_oat().

        Parameters
        ----------
        sectors : str | List[str] | None
            Filter to groups whose sector list overlaps these codes.
            None -> include all.
        constraint_type : "scalar" | "simplex" | None
            Filter by constraint type.  None -> include all.

        Notes
        -----
        De-duplication is by column name.  If the same column appears in two
        groups with different bounds, the first occurrence wins.
        """
        sector_set = _normalise_sector_filter(sectors)
        seen: Set[str] = set()
        out: List[VariableSpec] = []
        for group in self._filter_groups(sector_set, constraint_type):
            for spec in group.specs:
                if spec.column not in seen:
                    seen.add(spec.column)
                    out.append(spec)
        return out

    def get_targets(
        self,
        sectors: Optional[Union[str, List[str]]] = None,
    ) -> List[IEATarget]:
        """Flat, de-duplicated list of IEA (balance, product) target pairs.

        Parameters
        ----------
        sectors : str | List[str] | None
            Filter to groups whose sector list overlaps these codes.
        """
        sector_set = _normalise_sector_filter(sectors)
        seen: Set[IEATarget] = set()
        out: List[IEATarget] = []
        for group in self._filter_groups(sector_set):
            for target in group.iea_targets:
                if target not in seen:
                    seen.add(target)
                    out.append(target)
        return out

    def scalar_groups(
        self,
        sectors: Optional[Union[str, List[str]]] = None,
    ) -> List[CalibrationGroup]:
        """All groups with constraint_type == "scalar" (optionally filtered
        by sector).  These are compatible with scale_inputs_single_value()."""
        return self._filter_groups(_normalise_sector_filter(sectors), "scalar")

    def simplex_groups(
        self,
        sectors: Optional[Union[str, List[str]]] = None,
    ) -> List[CalibrationGroup]:
        """All groups with constraint_type == "simplex" (optionally filtered
        by sector).  These require shift_fuels_based_on_single_point()."""
        return self._filter_groups(_normalise_sector_filter(sectors), "simplex")

    # ------------------------------------------------------------------ #
    #    Sector decomposition                                            #
    # ------------------------------------------------------------------ #

    def by_sector(self) -> Dict[str, "CalibrationPlan"]:
        """Split the plan into one CalibrationPlan per sector code.

        A group that spans multiple sectors appears in all of them.

        Returns
        -------
        dict
            {sector_code: CalibrationPlan}, sorted by sector code.
        """
        sector_map: Dict[str, List[CalibrationGroup]] = {}
        for group in self._groups:
            for sec in group.sector:
                sector_map.setdefault(sec, []).append(group)
        return {sec: CalibrationPlan(sector_map[sec]) for sec in sorted(sector_map)}

    # ------------------------------------------------------------------ #
    #    Targeted filtering                                              #
    # ------------------------------------------------------------------ #

    def filter_by_target(
        self,
        iea_balance_code: str,
        iea_product_code: str,
    ) -> "CalibrationPlan":
        """CalibrationPlan containing only groups that include the given
        (balance, product) pair in their iea_targets."""
        pair: IEATarget = (iea_balance_code, iea_product_code)
        return CalibrationPlan([g for g in self._groups if pair in g.iea_targets])

    def filter_by_name(
        self,
        names: Union[str, List[str], Set[str]],
    ) -> "CalibrationPlan":
        """CalibrationPlan containing only groups whose `name` matches.

        Useful for isolating a single calibration target when debugging —
        e.g. plan.filter_by_name("transport__transport") returns a plan
        with just that one group so every other target is untouched.
        """
        wanted: Set[str] = {names} if isinstance(names, str) else set(names)
        return CalibrationPlan([g for g in self._groups if g.name in wanted])

    def filter_by_constraint(self, constraint_type: str) -> "CalibrationPlan":
        """CalibrationPlan with only scalar or only simplex groups."""
        if constraint_type not in _CONSTRAINT_TYPES:
            raise ValueError(
                f"constraint_type must be one of {sorted(_CONSTRAINT_TYPES)}, "
                f"got '{constraint_type}'."
            )
        return CalibrationPlan([g for g in self._groups
                                if g.constraint_type == constraint_type])

    # ------------------------------------------------------------------ #
    #    Summary and coverage                                            #
    # ------------------------------------------------------------------ #

    def summary(self) -> pd.DataFrame:
        """Human-readable table: one row per CalibrationGroup.

        Columns
        -------
        name, sectors, n_specs, n_targets, constraint_type,
        simplex_group_ids, iea_targets, notes
        """
        rows = []
        for g in self._groups:
            rows.append({
                "name":              g.name,
                "sectors":           ", ".join(g.sector),
                "n_specs":           g.n_specs,
                "n_targets":         g.n_targets,
                "constraint_type":   g.constraint_type,
                "simplex_group_ids": g.simplex_group_ids or "",
                "iea_targets":       "; ".join(f"{b}x{p}" for b, p in g.iea_targets),
                "notes":             g.notes,
            })
        if not rows:
            return pd.DataFrame(columns=[
                "name", "sectors", "n_specs", "n_targets",
                "constraint_type", "simplex_group_ids", "iea_targets", "notes",
            ])
        return pd.DataFrame(rows)

    def coverage_report(
        self,
        df_comparison: pd.DataFrame,
        col_balance: str = "iea_balance_code",
        col_product: str = "iea_product_code",
    ) -> pd.DataFrame:
        """Check which group targets have IEA and SISEPUEDE data in df_comparison.

        Returns a DataFrame with one row per (group_name x iea_target) pair,
        tagged with has_iea_data and has_ssp_data.

        Parameters
        ----------
        df_comparison : pd.DataFrame
            Output of IEACrosswalk.build_comparison() — must contain
            value_iea_tj and value_sisepuede_tj columns.
        """
        pairs_iea = set(zip(
            df_comparison.loc[df_comparison["value_iea_tj"].notna(), col_balance],
            df_comparison.loc[df_comparison["value_iea_tj"].notna(), col_product],
        ))
        pairs_ssp = set(zip(
            df_comparison.loc[df_comparison["value_sisepuede_tj"].notna(), col_balance],
            df_comparison.loc[df_comparison["value_sisepuede_tj"].notna(), col_product],
        ))

        rows = []
        for g in self._groups:
            for bal, prod in g.iea_targets:
                rows.append({
                    "group_name":        g.name,
                    "sectors":           ", ".join(g.sector),
                    "iea_balance_code":  bal,
                    "iea_product_code":  prod,
                    "constraint_type":   g.constraint_type,
                    "has_iea_data":      (bal, prod) in pairs_iea,
                    "has_ssp_data":      (bal, prod) in pairs_ssp,
                    "has_both":          (bal, prod) in pairs_iea and (bal, prod) in pairs_ssp,
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------ #
    #    Constructor — dict config                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_specs_dict(
        cls,
        spec_dict: Dict[str, Dict],
        model_attributes=None,
        simplex_registry: Optional[SimplexRegistry] = None,
    ) -> "CalibrationPlan":
        """Build a CalibrationPlan from a plain configuration dictionary.

        This is the recommended way to define a plan in a notebook or script.

        Parameters
        ----------
        spec_dict : Dict[str, Dict]
            Ordered dict; keys are group names.  Each value supports:
              "sector"          : str | List[str]          (required)
              "specs"           : List[VariableSpec]       (required)
              "iea_targets"     : List[Tuple[str, str]]    (required)
              "constraint_type" : "scalar" | "simplex"     (default "scalar")
              "notes"           : str                       (optional "")
        model_attributes : ModelAttributes | None
            When provided (and `simplex_registry` is None), a SimplexRegistry
            is built from it and used to auto-populate simplex metadata on
            the specs.  Ignored if `simplex_registry` is supplied directly.
        simplex_registry : SimplexRegistry | None
            Authoritative registry of simplex-group membership.  When a group
            has constraint_type="simplex", each spec's is_simplex_group /
            simplex_group_id are populated from this registry and the group's
            simplex_group_ids list is derived from it.  A simplex group whose
            columns are not in the registry raises KeyError — we refuse to
            silently produce a plan whose simplex bookkeeping would be wrong.

        Returns
        -------
        CalibrationPlan

        Example
        -------
            plan = CalibrationPlan.from_specs_dict(
                {
                    "inen_total_demand": {
                        "sector": "inen",
                        "specs": [
                            VariableSpec("consumpinit_inen_energy_cement", 0.5, 2.0),
                        ],
                        "iea_targets": [("INDUSTRY", "INDUSTRY")],
                    },
                    "inen_coal_fraction": {
                        "sector": "inen",
                        "constraint_type": "simplex",
                        "specs": [
                            VariableSpec("frac_inen_energy_cement_coal", 0.5, 1.5),
                            VariableSpec("frac_inen_energy_cement_oil",  0.5, 1.5),
                        ],
                        "iea_targets": [("INDUSTRY", "COAL"), ("INDUSTRY", "OIL")],
                    },
                },
                model_attributes = model_attributes,
            )
        """
        ##  Resolve the simplex registry once
        if simplex_registry is None and model_attributes is not None:
            simplex_registry = SimplexRegistry.from_model_attributes(model_attributes)

        groups: List[CalibrationGroup] = []

        for name, cfg in spec_dict.items():
            constraint = cfg.get("constraint_type", "scalar")
            specs: List[VariableSpec] = cfg["specs"]

            ##  For simplex groups: auto-annotate specs and collect group IDs
            simplex_ids: List[int] = []
            if constraint == "simplex":
                if simplex_registry is None:
                    raise ValueError(
                        f"CalibrationGroup '{name}' has constraint_type='simplex' "
                        "but no simplex_registry (or model_attributes) was provided. "
                        "Simplex metadata cannot be populated without the registry."
                    )
                simplex_registry.validate_columns(
                    [s.column for s in specs],
                    context=f"CalibrationGroup '{name}'",
                )
                ids_seen: Set[int] = set()
                for spec in specs:
                    gid = simplex_registry.group_id(spec.column)
                    # validate_columns above guarantees gid is not None
                    spec.is_simplex_group = True
                    spec.simplex_group_id = gid
                    if gid not in ids_seen:
                        ids_seen.add(gid)
                        simplex_ids.append(gid)

            groups.append(CalibrationGroup(
                name              = name,
                sector            = cfg["sector"],
                specs             = specs,
                iea_targets       = cfg["iea_targets"],
                constraint_type   = constraint,
                simplex_group_ids = simplex_ids,
                notes             = cfg.get("notes", ""),
            ))

        return cls(groups)

    # ------------------------------------------------------------------ #
    #    Internal helpers                                                #
    # ------------------------------------------------------------------ #

    def _filter_groups(
        self,
        sector_set: Optional[Set[str]] = None,
        constraint_type: Optional[str] = None,
    ) -> List[CalibrationGroup]:
        out = self._groups
        if sector_set is not None:
            out = [g for g in out if set(g.sector) & sector_set]
        if constraint_type is not None:
            out = [g for g in out if g.constraint_type == constraint_type]
        return out


##########################
#    MODULE HELPERS      #
##########################

def _normalise_sector_filter(
    sectors: Optional[Union[str, List[str]]],
) -> Optional[Set[str]]:
    """Convert a sector argument to a set, or None for 'all'."""
    if sectors is None:
        return None
    if isinstance(sectors, str):
        return {sectors}
    return set(sectors)
