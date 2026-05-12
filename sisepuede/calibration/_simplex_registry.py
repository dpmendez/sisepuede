"""
sisepuede/calibration/_simplex_registry.py

SimplexRegistry
----------------
Single source of truth for "which input columns are co-constrained by a
simplex (sum-to-1) constraint".

Why this exists
---------------
Several pieces of the calibration pipeline need to know which set of columns
must be renormalized together when one of them is perturbed:

    - apply_perturbations (sensitivity.py)  — Aitchison renormalization
    - Calibrator._phase2_fuel_mix           — computing and applying fuel-
                                              fraction scales
    - LHS / Dirichlet sampling              — drawing feasible samples
    - Optimizers                            — ILR / SLSQP constraint setup

Historically the calibration code inferred the co-constraint set from the
column name (everything sharing `column.rsplit("_", 1)[0] + "_"`). That
works today because variable names happen to encode their grouping, but:

    - It is an implicit contract on naming conventions
    - It silently breaks if variables are renamed
    - It cannot distinguish two unrelated groups that happen to share a prefix

The authoritative definition lives in the SISEPUEDE attribute-table CSVs and
is exposed on ModelAttributes as:

    model_attributes.dict_field_to_simplex_group   : {field: group_id}
    model_attributes.dict_simplex_group_to_fields  : {group_id: [fields, ...]}

SimplexRegistry is a thin, read-only facade over those two dictionaries.
Passing a registry around (rather than `model_attributes` directly, or
rather than relying on naming conventions) gives us one clean place to:

    - Look up "what group is this column in?"
    - Look up "what columns are co-constrained with this one?"
    - Validate that a CalibrationGroup's simplex metadata is consistent with
      the underlying model definition
    - Serialize the grouping for offline / surrogate use

Construction
------------
Typical use — wrap a ModelAttributes:

    from sisepuede.calibration import SimplexRegistry

    registry = SimplexRegistry.from_model_attributes(model_attributes)

Standalone use (e.g. surrogate context) — build directly from a dict:

    registry = SimplexRegistry(
        field_to_group = {"frac_inen_energy_cement_coal": 1, ...},
    )

The two directions are kept in sync internally: you can pass either
`field_to_group` or `group_to_fields`; the other is derived.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


class SimplexRegistry:
    """Read-only lookup for simplex-group membership of input columns.

    Parameters
    ----------
    field_to_group : Dict[str, int] | None
        Mapping {column_name: simplex_group_id}.  Columns not in this map
        are treated as unconstrained (not part of any simplex group).
    group_to_fields : Dict[int, List[str]] | None
        Reverse mapping {simplex_group_id: [column_name, ...]}.  If given
        together with field_to_group, both must be consistent; if only one
        is given, the other is derived.

    Raises
    ------
    ValueError
        If both field_to_group and group_to_fields are given and they are
        not consistent with each other.
    """

    def __init__(
        self,
        field_to_group:  Optional[Dict[str, int]]        = None,
        group_to_fields: Optional[Dict[int, List[str]]] = None,
    ) -> None:

        if field_to_group is None and group_to_fields is None:
            field_to_group  = {}
            group_to_fields = {}

        if field_to_group is None:
            field_to_group = {
                field: gid
                for gid, fields in group_to_fields.items()
                for field in fields
            }

        if group_to_fields is None:
            group_to_fields = {}
            for field, gid in field_to_group.items():
                group_to_fields.setdefault(gid, []).append(field)

        # Consistency check if both were supplied externally
        derived_f2g = {
            f: gid for gid, fields in group_to_fields.items() for f in fields
        }
        if field_to_group != derived_f2g:
            # Find the divergences for a helpful error
            diffs = []
            for f, gid in field_to_group.items():
                if derived_f2g.get(f) != gid:
                    diffs.append(
                        f"  {f}: field_to_group says {gid}, "
                        f"group_to_fields says {derived_f2g.get(f)}"
                    )
            raise ValueError(
                "SimplexRegistry: field_to_group and group_to_fields are "
                "inconsistent:\n" + "\n".join(diffs[:10])
            )

        # Store sorted copies so iteration order is deterministic
        self._field_to_group: Dict[str, int] = dict(field_to_group)
        self._group_to_fields: Dict[int, List[str]] = {
            gid: sorted(fields) for gid, fields in group_to_fields.items()
        }

    # ------------------------------------------------------------------
    #   Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_model_attributes(cls, model_attributes) -> "SimplexRegistry":
        """Build a registry from a ModelAttributes instance.

        Reads `dict_field_to_simplex_group` and `dict_simplex_group_to_fields`
        (both populated at ModelAttributes initialization time from the
        attribute-table CSVs).

        Parameters
        ----------
        model_attributes : ModelAttributes
            SISEPUEDE ModelAttributes object.

        Returns
        -------
        SimplexRegistry
        """
        f2g = getattr(model_attributes, "dict_field_to_simplex_group", None) or {}
        g2f = getattr(model_attributes, "dict_simplex_group_to_fields", None) or {}
        # Coerce values of g2f to lists (some callers store tuples/sets)
        g2f = {int(k): list(v) for k, v in g2f.items()}
        f2g = {str(k): int(v) for k, v in f2g.items()}
        return cls(field_to_group=f2g, group_to_fields=g2f)

    @classmethod
    def empty(cls) -> "SimplexRegistry":
        """Registry with no simplex groups (every column is unconstrained)."""
        return cls(field_to_group={}, group_to_fields={})

    # ------------------------------------------------------------------
    #   Core lookups
    # ------------------------------------------------------------------

    def group_id(self, column: str) -> Optional[int]:
        """Return the simplex group ID for `column`, or None if not simplex-constrained."""
        return self._field_to_group.get(column)

    def is_simplex(self, column: str) -> bool:
        """True iff `column` is part of some simplex group."""
        return column in self._field_to_group

    def columns_in_group(self, group_id: int) -> List[str]:
        """Return every column belonging to `group_id`.

        Returns an empty list if the group ID is unknown.
        """
        return list(self._group_to_fields.get(group_id, []))

    def co_constrained_with(self, column: str) -> List[str]:
        """Return every column co-constrained with `column`, including itself.

        If `column` is not in any simplex group, returns `[column]` — the
        caller can treat this as "this column has no simplex siblings" and
        perturb it independently.

        This is the replacement for the old prefix-matching helper.
        """
        gid = self._field_to_group.get(column)
        if gid is None:
            return [column]
        return list(self._group_to_fields.get(gid, [column]))

    # ------------------------------------------------------------------
    #   Bulk access
    # ------------------------------------------------------------------

    @property
    def field_to_group(self) -> Dict[str, int]:
        """A copy of the underlying {field: group_id} dict."""
        return dict(self._field_to_group)

    @property
    def group_to_fields(self) -> Dict[int, List[str]]:
        """A copy of the underlying {group_id: [fields]} dict."""
        return {gid: list(fields) for gid, fields in self._group_to_fields.items()}

    @property
    def group_ids(self) -> List[int]:
        """Sorted list of all simplex group IDs present in the registry."""
        return sorted(self._group_to_fields.keys())

    def __len__(self) -> int:
        """Number of distinct simplex groups."""
        return len(self._group_to_fields)

    def __contains__(self, column: str) -> bool:
        return column in self._field_to_group

    def __repr__(self) -> str:
        n_groups = len(self._group_to_fields)
        n_fields = len(self._field_to_group)
        return f"SimplexRegistry(groups={n_groups}, fields={n_fields})"

    # ------------------------------------------------------------------
    #   Validation
    # ------------------------------------------------------------------

    def validate_columns(
        self,
        columns: Iterable[str],
        context: str = "",
    ) -> None:
        """Raise if any column in `columns` is not known to the registry.

        Use this when a caller claims a set of columns is simplex-constrained
        (e.g. `VariableSpec.is_simplex_group = True`) — we want to fail fast
        if they are not actually registered as simplex-group members, rather
        than silently fall through to a no-op.

        Parameters
        ----------
        columns : Iterable[str]
            Column names claimed to be in some simplex group.
        context : str
            Free-text prefix for the error message (e.g. the group name).

        Raises
        ------
        KeyError
            If any column is not in `field_to_group`.
        """
        missing = [c for c in columns if c not in self._field_to_group]
        if missing:
            head = f"{context}: " if context else ""
            raise KeyError(
                f"{head}the following columns are marked as simplex-constrained "
                f"but are not registered in any simplex group "
                f"(check ModelAttributes / attribute-table CSVs): {missing}"
            )

    def check_sums(
        self,
        df: pd.DataFrame,
        columns_subset: Optional[Iterable[str]] = None,
        tol_ok:     float = 1e-6,
        tol_warn:   float = 1e-3,
        tol_severe: float = 1e-2,
    ) -> List[Dict[str, Any]]:
        """Verify per-simplex-group sum-to-1 on a SISEPUEDE input DataFrame.

        For each simplex group whose columns appear in ``df``, computes the
        per-row sum and the worst absolute deviation from 1.0. Pure
        function: returns findings, never warns or raises. The caller
        decides what to do with each severity band.

        Severity bands (on max absolute deviation across rows):

            |dev| < tol_ok                      -> "ok"
            tol_ok <= |dev| < tol_warn          -> "warn"
            tol_warn <= |dev| < tol_severe      -> "strong_warn"
            |dev| >= tol_severe                 -> "severe"

        Parameters
        ----------
        df : pd.DataFrame
            Input frame containing simplex columns (and typically
            ``time_period`` for diagnostics).
        columns_subset : Iterable[str] | None
            Restrict the check to simplex groups that include at least one
            of these columns. Use this to focus on groups that calibration
            actually touched; groups left alone are not interesting.
            ``None`` checks every simplex group whose columns appear in
            ``df``.
        tol_ok, tol_warn, tol_severe : float
            Cumulative thresholds. Must satisfy 0 <= tol_ok < tol_warn <
            tol_severe.

        Returns
        -------
        List[dict] sorted by ``max_abs_dev`` descending. Each entry:

            {"group_id":          int,
             "n_columns":         int,            # cols actually summed
             "n_cols_missing":    int,            # registered but absent from df
             "max_abs_dev":       float,
             "time_period_worst": int | None,
             "severity":          "ok" | "warn" | "strong_warn" | "severe"}
        """
        if not (0 <= tol_ok < tol_warn < tol_severe):
            raise ValueError(
                "Thresholds must satisfy 0 <= tol_ok < tol_warn < tol_severe; "
                f"got tol_ok={tol_ok}, tol_warn={tol_warn}, tol_severe={tol_severe}."
            )

        # Decide which simplex groups to inspect.
        if columns_subset is None:
            gids_to_check = [
                gid for gid, cols in self._group_to_fields.items()
                if any(c in df.columns for c in cols)
            ]
        else:
            seen: set = set()
            gids_to_check: List[int] = []
            for c in columns_subset:
                gid = self._field_to_group.get(c)
                if gid is not None and gid not in seen:
                    seen.add(gid)
                    gids_to_check.append(gid)

        if not gids_to_check:
            return []

        has_tp = "time_period" in df.columns
        findings: List[Dict[str, Any]] = []

        for gid in gids_to_check:
            registered = self._group_to_fields.get(gid, [])
            present = [c for c in registered if c in df.columns]
            if not present:
                continue
            missing = len(registered) - len(present)

            row_sums = df[present].sum(axis=1).to_numpy(dtype=float)
            abs_dev = np.abs(row_sums - 1.0)
            if abs_dev.size == 0:
                continue

            worst_idx = int(np.argmax(abs_dev))
            max_dev = float(abs_dev[worst_idx])

            if max_dev < tol_ok:
                severity = "ok"
            elif max_dev < tol_warn:
                severity = "warn"
            elif max_dev < tol_severe:
                severity = "strong_warn"
            else:
                severity = "severe"

            tp_worst: Optional[int] = None
            if has_tp:
                try:
                    tp_worst = int(df["time_period"].iloc[worst_idx])
                except (ValueError, TypeError):
                    tp_worst = None

            findings.append({
                "group_id":          gid,
                "n_columns":         len(present),
                "n_cols_missing":    missing,
                "max_abs_dev":       max_dev,
                "time_period_worst": tp_worst,
                "severity":          severity,
            })

        findings.sort(key=lambda r: r["max_abs_dev"], reverse=True)
        return findings

    def group_partition(self, columns: Iterable[str]) -> Dict[int, List[str]]:
        """Partition an iterable of columns by their simplex group ID.

        Columns not in any simplex group are returned under key `-1`.
        Use this to process a mixed set of columns one simplex constraint
        at a time:

            for gid, cols in registry.group_partition(spec_columns).items():
                if gid == -1:
                    # unconstrained columns — scale independently
                    ...
                else:
                    all_cols = registry.columns_in_group(gid)
                    # renormalize `all_cols` together

        Returns
        -------
        Dict[int, List[str]]
            Keys are simplex group IDs (or `-1` for unconstrained).
            Values preserve input order.
        """
        out: Dict[int, List[str]] = {}
        for col in columns:
            gid = self._field_to_group.get(col, -1)
            out.setdefault(gid, []).append(col)
        return out
