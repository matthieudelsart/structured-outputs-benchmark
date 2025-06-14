import json
import math
import re
from typing import Any, Dict, List, Set, Tuple, Union
import numbers
import logging

import argparse
import json_repair
import numpy as np
import pycountry
from dateutil import parser as dt
from jsonschema import Draft7Validator, ValidationError, validators
from rapidfuzz import distance, process
from unidecode import unidecode
from pathlib import Path

logger = logging.getLogger(__name__)

######### Helpers


def assess_json_valid(json_string: str) -> Tuple[float, Any]:
    """
    Assess if the given JSON string is valid.

    Args:
        json_string (str): The JSON string to validate.

    Returns:
        Tuple[int, Dict]: A tuple containing a status code (1 for valid, 0 for invalid) and the parsed record.
    """
    try:
        record = json.loads(json_string)
        return 1, record
    except json.JSONDecodeError:
        try:
            record = json.loads(json_string.replace("```json", "").replace("```", ""))
            return 0.9, record
        except json.JSONDecodeError:
            try:
                record = json.loads(json_string.replace("```json", "").replace("```", "").replace("'",'"'))
                return 0.7, record
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON string: {json_string} - Fixing")
                record = json_repair.loads(json_string)
                return 0, record
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}. Problem with {json_string}")
        return 0, json_string


def f1(tp: int, fp: int, fn: int) -> float:
    """
    Calculate the F1 score.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        float: The F1 score.
    """
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def gmean(x: float, y: float) -> float:
    """
    Calculate the geometric mean of two numbers.

    Args:
        x (float): First number.
        y (float): Second number.

    Returns:
        float: The geometric mean.
    """
    return math.sqrt(x * y) if x and y else 0


def _norm_name(name: str) -> str:
    """
    Normalize a name by lower-casing, collapsing whitespace, and dropping periods.

    Args:
        name (str): The name to normalize.

    Returns:
        str: The normalized name.
    """
    return " ".join(name.lower().replace(".", "").split())


def _numeric_equal(a: Any, b: Any, tol: int | float = 0) -> bool:
    """
    Check if two numeric values are equal within a tolerance.

    Args:
        a (Any): First value.
        b (Any): Second value.
        tol (int | float, optional): Tolerance for comparison. Defaults to 0.

    Returns:
        bool: True if the values are equal within the tolerance, False otherwise.
    """
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return a == b


# ────── 1.  Normalisation helpers ───────────────────────────────────────────

DATE_PAREN_RE = re.compile(r"\(.*?\)")  # strip "(aged 67)"


def _norm_date(val: str) -> str:
    """
    Normalize a date string to ISO 8601 format.

    Args:
        val (str): The date string to normalize.

    Returns:
        str: The normalized date string in ISO 8601 format.
    """
    val = DATE_PAREN_RE.sub("", val).strip()
    try:
        d = dt.parse(val, dayfirst=False, fuzzy=True)
    except Exception as e:
        logger.error(f"Error parsing date: {val}, error: {e}")
        return val.lower().strip()
    iso = d.date().isoformat()
    return iso if re.search(r"\b([12]?\d|3[01])\b", val) else iso[:7]


def _digits_only(val: str) -> str:
    """
    Extract only digits from a string.

    Args:
        val (str): The string to process.

    Returns:
        str: A string containing only digits.
    """
    return "".join(re.findall(r"\d+", val))


ALIASES = {
    "us": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "american": "United States",
    "england": "United Kingdom",
    "scottish": "United Kingdom",
    "welsh": "United Kingdom",
    "czech": "Czechia",
    "cze": "Czechia",
    "états-unis": "United States",
}


def _norm_nationality(val: str) -> str:
    """
    Normalize a nationality string.

    Args:
        val (str): The nationality string to normalize.

    Returns:
        str: The normalized nationality string.
    """
    v = unidecode(val).lower().strip().replace(".", "")
    if v in ALIASES:
        return ALIASES[v]
    try:  # pycountry catch-all
        return pycountry.countries.lookup(v).name
    except LookupError as e:
        logger.warning(f"Lookup error for nationality: {val}, error: {e}")
        return v


REDUNDANT = {
    "u.s.",
    "usa",
    "united states",
    "united kingdom",
    "uk",
    "england",
    "scotland",
    "wales",
    "northern ireland",
}


def _norm_place(val: str) -> str:
    """
    Normalize a place string.

    Args:
        val (str): The place string to normalize.

    Returns:
        str: The normalized place string.
    """
    parts = [p.strip() for p in val.split(",")]
    clean = [
        unidecode(p).lower() for p in parts if unidecode(p).lower() not in REDUNDANT
    ]
    return ", ".join(clean) if clean else unidecode(val).lower()


MULTI_SEP_RE = re.compile(r"[;,/•]+")


def _split_set(val: str) -> List[str]:
    """
    Split a string into a list of strings based on multiple separators.

    Args:
        val (str): The string to split.

    Returns:
        List[str]: A list of strings.
    """
    return [s.strip() for s in MULTI_SEP_RE.split(val) if s.strip()]


def _normalise_value(key: str, val: Any) -> Any:
    """
    Normalize a value based on its key.

    Args:
        key (str): The key associated with the value.
        val (Any): The value to normalize.

    Returns:
        Any: The normalized value.
    """
    if val is None:
        return val
    if isinstance(val, (int, float)):
        return val
    val = str(val).strip()

    if "date" in key:
        return _norm_date(val)
    if key in {
        "height_ft",
        "height_in",
        "weight_lbs",
        "career_start",
        "career_end",
        "debutyear",
        "finalyear",
        "serviceyears",
        "number",
        "years_active",
    }:
        return _digits_only(val)
    if key == "nationality":
        return _norm_nationality(val)
    if key.endswith("_place") or key in {"birth_place", "death_place"}:
        return _norm_place(val)
    return unidecode(val).lower()


def _values_equal(a: Any, b: Any) -> bool:
    """
    Check if two values are equal, treating multi-valued strings as sets.

    Args:
        a (Any): First value.
        b (Any): Second value.

    Returns:
        bool: True if the values are equal, False otherwise.
    """
    if isinstance(a, str) and isinstance(b, str):
        # treat multi-valued strings as sets
        a_set, b_set = map(_split_set, (a, b))
        if a_set or b_set:  # at least one looks like multi-value
            return set(a_set) == set(b_set)
    return a == b


########################################""
### General schema compliance evaluation


class GeneralJsonSchemaEvaluator:
    """
    Validate data against a JSON Schema and return a compliance percentage
    based on the number of primitive checks implied by the schema.
    """

    def __init__(self, schema: Dict):
        self.schema = schema

        # Pick the correct validator implementation for the draft in $schema
        ValidatorCls = validators.validator_for(schema)
        ValidatorCls.check_schema(schema)  # fail fast if schema bogus
        self._validator = ValidatorCls(schema)

    def _leaf_count(self, schema: Dict) -> int:
        """
        Recursively count 'primitive checks' in *schema*.

        • Every primitive type check (string, number, boolean, integer, null)
          counts as 1.
        • Arrays count 1 + items-checks.
        • Objects add up their property schemas (incl. patternProperties,
          propertyNames, additionalProperties when it is a schema).
        • For combinators (anyOf, oneOf, allOf, if/then/else) pick the
          *minimum* branch so denominator ≈ shortest path to validity.
        """

        def _norm_type(t):
            if isinstance(t, list):
                return (
                    "object"
                    if "object" in t
                    else "array"
                    if "array" in t
                    else "primitive"
                )
            return t

        t = _norm_type(schema.get("type"))

        # ---------- object -------------------------------------------------
        if t == "object":
            total = 0
            for sub in schema.get("properties", {}).values():
                total += self._leaf_count(sub)
            for sub in schema.get("patternProperties", {}).values():
                total += self._leaf_count(sub)
            if isinstance(schema.get("additionalProperties"), dict):
                total += self._leaf_count(schema["additionalProperties"])
            if "propertyNames" in schema and isinstance(schema["propertyNames"], dict):
                total += self._leaf_count(schema["propertyNames"])
            return total

        # ---------- array --------------------------------------------------
        if t == "array":
            return 1 + self._leaf_count(schema.get("items", {}))

        # ---------- combinators / conditionals -----------------------------
        for comb in ("anyOf", "oneOf", "allOf"):
            if comb in schema:
                return min(self._leaf_count(sub) for sub in schema[comb])
        if "if" in schema:  # consider shortest of if/then(/else)
            branches = [schema["if"]]
            if "then" in schema:
                branches.append(schema["then"])
            if "else" in schema:
                branches.append(schema["else"])
            return min(self._leaf_count(b) for b in branches)

        # ---------- primitive / fall-through -------------------------------
        return 1

    def score_against_schema(self, data: Union[Dict, List]) -> Dict:
        """
        Validate *data* and return a dict with:
            • percentage  – float 0-1
            • failures    – list[ValidationError]  (for debugging / UX)
        """
        errors: List[ValidationError] = list(self._validator.iter_errors(data))

        failed_count = len(errors)
        total_checks = self._leaf_count(self.schema)  # ③ how many independent checks exist
        passed = total_checks - failed_count

        percentage = max(0.0, round(passed / total_checks, 3)) if total_checks else 1
        return {"compliance": percentage, "compliance_errors": errors}


########## Evaluators


class RotowireEvaluator:
    def __init__(self, schema_loc: str = "data/clean/1-rotowire/schema.json"):
        with open(schema_loc, "r") as f:
            self.schema = json.load(f)
        self.validator = Draft7Validator(self.schema)

    #### Format evaluation

    def _ok(self, value: Any, allowed_types) -> bool:
        """Helper function to check key-wise type."""
        if not isinstance(allowed_types, list):
            allowed_types = [allowed_types]
        return (
            (value is None and "null" in allowed_types)
            or (isinstance(value, int) and "integer" in allowed_types)
            or (isinstance(value, str) and "string" in allowed_types)
        )

    def _score_section(
        self,
        data: List[Dict[str, Any]] | None,
        definition: Dict[str, Any],
        min_items: int | None,
        max_items: int | None,
    ) -> Tuple[int, Any]:
        """
        Returns (valid, total) for the section.
        If data is None (missing) or length violates min/max, returns (0, 1) → 0 %.
        Here, definition can be either a subdefinition in a schema with different parts, or the whole schema.
        """
        if data is None:
            return 0, 1  # missing section → 0 %
        if (min_items and len(data) < min_items) or (
            max_items and len(data) > max_items
        ):
            return 0, 1  # length violation → 0 %

        props = definition["properties"]
        required = set(definition.get("required", []))
        allow_add = definition.get("additionalProperties", False)

        total = valid = 0
        for obj in data:
            # Required keys first
            for key in required:
                total += 1
                if key in obj and self._ok(obj[key], props[key]["type"]):
                    valid += 1

            # Then looking at the other ones
            for key, val in obj.items():
                if key in required:
                    continue
                total += 1
                if key in props:
                    if self._ok(val, props[key]["type"]):
                        valid += 1
                elif allow_add:
                    valid += 1  # Allowed extras
                # else: extra - invalid (counts as 0)

        return valid, total if total else (0, 1)  # avoid div-by-zero

    def compliance_breakdown(self, record: Dict) -> Dict[str, Any]:
        """Percentage compliance + real Draft-07 errors, handles all edge-cases."""

        errors = list(self.validator.iter_errors(record))

        # Pull defs & array length rules from the schema
        team_def = self.schema["definitions"]["team"]
        player_def = self.schema["definitions"]["player"]

        teams_schema = self.schema["items"]["properties"]["teams"]
        players_schema = self.schema["items"]["properties"]["players"]

        t_valid, t_total = self._score_section(
            record.get("teams"),
            team_def,
            teams_schema.get("minItems"),
            teams_schema.get("maxItems"),
        )
        p_valid, p_total = self._score_section(
            record.get("players"),
            player_def,
            players_schema.get("minItems"),
            players_schema.get("maxItems"),
        )

        team_compliance = t_valid / t_total
        player_compliance = p_valid / p_total

        return {
            "team_compliance": team_compliance,
            "player_compliance": player_compliance,
            "overall_compliance": float(np.mean([team_compliance, player_compliance])),
            "errors": errors,
        }
    
    def compute_compliance(self, output: Dict) -> Dict[str, Any]:
        full_compliance = self.compliance_breakdown(output)
        return {"compliance": full_compliance["overall_compliance"], "compliance_errors": full_compliance["errors"]}

    ##### Correctness evaluation

    def match_by_key(
        self,
        ref_objs: List[Dict],
        pred_objs: List[Dict],
        name_key: str,
        edit_threshold: int = 1,
    ) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
        """
        Greedy 1-to-1 matching: exact names first, then fuzzy with Levenshtein ≤ threshold.
        Returns (matched_pairs, unmatched_ref, unmatched_pred).
        """
        ref_map = {_norm_name(o[name_key]): o for o in ref_objs}
        pred_map = {_norm_name(o[name_key]): o for o in pred_objs}

        pairs = []

        # exact matches
        for n in list(ref_map):
            if n in pred_map:
                pairs.append((ref_map.pop(n), pred_map.pop(n)))

        # fuzzy pass
        for r_name, r_obj in list(ref_map.items()):
            if not pred_map:
                break
            best, score, _ = process.extractOne(
                r_name, list(pred_map.keys()), scorer=distance.Levenshtein.distance
            )
            if score <= edit_threshold:
                pairs.append((r_obj, pred_map.pop(best)))
                ref_map.pop(r_name)

        return pairs, list(ref_map.values()), list(pred_map.values())

    def compare_objects(
        self,
        ref_obj: Dict,
        pred_obj: Dict,
        value_tolerance: Dict[str, float] | None = None,
    ) -> Tuple[int, int, int, int, int]:
        """
        Returns (TP_keys, FP_keys, FN_keys, correct_values, total_compared_values)
        """
        if value_tolerance is None:
            value_tolerance = {}

        tp = fp = fn = val_ok = val_tot = 0

        keys_union = ref_obj.keys() | pred_obj.keys()
        for k in keys_union:
            in_ref = k in ref_obj
            in_pred = k in pred_obj
            if in_ref and in_pred:
                tp += 1
                tol = value_tolerance.get(k, 0)
                if _numeric_equal(ref_obj[k], pred_obj[k], tol):
                    val_ok += 1
                val_tot += 1
            elif in_pred:
                fp += 1
            else:
                fn += 1
        return tp, fp, fn, val_ok, val_tot

    def evaluate_game(
        self,
        reference: Dict,
        prediction: Dict,
        value_tolerance: Dict[str, float] | None = None,
        edit_threshold: int = 1,
    ) -> Dict[str, Any]:
        """
        Returns a dict with detection and attribute scores for one game.
        """
        # ---------------- Stage A
        t_pairs, t_miss_ref, t_miss_pred = self.match_by_key(
            reference["teams"], prediction.get("teams", []), "team", edit_threshold
        )
        p_pairs, p_miss_ref, p_miss_pred = self.match_by_key(
            reference["players"],
            prediction.get("players", []),
            "player",
            edit_threshold,
        )

        team_det_tp = len(t_pairs)
        player_det_tp = len(p_pairs)

        # First metric : identification of the right objects
        team_detect_F1 = f1(team_det_tp, len(t_miss_pred), len(t_miss_ref))
        player_detect_F1 = f1(player_det_tp, len(p_miss_pred), len(p_miss_ref))

        # ---------------- Stage B
        def score_pairs(pairs):
            tp = fp = fn = v_ok = v_tot = 0
            for ref_o, pred_o in pairs:
                a, b, c, d, e = self.compare_objects(ref_o, pred_o, value_tolerance)
                tp += a
                fp += b
                fn += c
                v_ok += d
                v_tot += e
            return tp, fp, fn, v_ok, v_tot

        t_tp, t_fp, t_fn, t_v_ok, t_v_tot = score_pairs(t_pairs)
        p_tp, p_fp, p_fn, p_v_ok, p_v_tot = score_pairs(p_pairs)

        # How many attributes were correctly identified
        team_attr_f1 = f1(t_tp, t_fp + len(t_miss_pred), t_fn + len(t_miss_ref))
        player_attr_f1 = f1(p_tp, p_fp + len(p_miss_pred), p_fn + len(p_miss_ref))

        # The accuracy of values for each of them
        team_val_acc = t_v_ok / t_v_tot if t_v_tot else 0.0
        player_val_acc = p_v_ok / p_v_tot if p_v_tot else 0.0

        # Overall score for a given team / player
        team_attr_score = gmean(team_attr_f1, team_val_acc)
        player_attr_score = gmean(player_attr_f1, player_val_acc)

        # Overall score linking identification and key/values correctness
        overall_team_score = gmean(team_detect_F1, team_attr_score)
        overall_player_score = gmean(player_detect_F1, player_attr_score)

        return {
            "ident_f1": (team_detect_F1 + player_detect_F1) / 2,
            "object_attr_F1": (team_attr_f1 + player_attr_f1) / 2,
            "object_value_acc": (team_val_acc + player_val_acc) / 2,
            "object_attribute_score": (team_attr_score + player_attr_score) / 2,
            "correctness": (overall_team_score + overall_player_score) / 2,
            "errors": [], # too complex for that one
        }
    
    def compute_correctness(self, reference: Dict, prediction: Dict) -> Dict[str, Any]:
        """
        Computes correctness of the prediction against the reference.
        Returns a dict with correctness percentage and errors.
        """
        full_score = self.evaluate_game(reference, prediction)
        return {"correctness": full_score["correctness"], "correctness_errors": full_score["errors"]}
    
    def score_record(self, record: Dict) -> Dict[str, Any]:
        """
        Scores a single record against the reference.
        Returns a dict with compliance and correctness scores.
        """

        # --- Syntax validity
        validity_score, pred_dict = assess_json_valid(record["generated_output"])

        if not isinstance(pred_dict, Dict):
            return {
                "is_valid": validity_score,
                "compliance": 0,
                "correctness": 0,
            }

        compliance = self.compute_compliance(pred_dict)
        correctness = self.compute_correctness(record["output"], pred_dict)

        return {"is_valid": validity_score, **compliance, **correctness}


class WikiBioEvaluator:
    def __init__(self):
        pass
    
    def compute_compliance(self, output: Dict, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes compliance of the JSON string against the schema.
        Returns a dict with compliance percentage and errors.
        """
        validator = GeneralJsonSchemaEvaluator(schema)
        return validator.score_against_schema(output)

    def compute_correctness(
        self,
        reference: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Strict key set already guaranteed by compliance → compare *values only*.
        Returns percentage and list of mismatches.
        """
        errors = []
        keys = reference.keys()
        total = len(keys)
        correct = 0

        for k in keys:
            ref_val = _normalise_value(k, reference.get(k))
            pred_val = _normalise_value(k, prediction.get(k))

            if _values_equal(ref_val, pred_val):
                correct += 1
            else:
                errors.append(
                    f"Key '{k}': expected '{reference.get(k)}'  |  got '{prediction.get(k)}'"
                )

        return {
            "correctness": correct / total,
            "correctness_errors": errors,
        }
    
    def score_record(self, record: Dict) -> Dict[str, Any]:
        """
        Scores a single record against the reference.
        Returns a dict with compliance and correctness scores.
        """

        # --- Syntax validity
        validity_score, pred_dict = assess_json_valid(record["generated_output"])

        if not isinstance(pred_dict, Dict):
            return {
                "is_valid": validity_score,
                "compliance": 0,
                "correctness": 0,
            }

        compliance = self.compute_compliance(pred_dict, schema=record["schema"])
        correctness = self.compute_correctness(record["output"], pred_dict)

        return {"is_valid": validity_score, **compliance, **correctness}


######## Need to handle that one takes the json str a input and the other the real dict; see with jsonl


class FewNerdsEvaluator:
    def __init__(self, schema_loc: str = "data/clean/3-few_nerd/schema.json"):
        self.schema = json.load(open(schema_loc))
        self.validator = GeneralJsonSchemaEvaluator(self.schema)

    def compute_compliance(self, output: Dict) -> Dict[str, Any]:
        """
        Computes compliance of the JSON string against the schema.
        Returns a dict with compliance percentage and errors.
        """
        return self.validator.score_against_schema(output)

    @staticmethod
    def _norm(item: Any) -> str:
        """
        Robust normaliser:
        • Strings  → use project-level _norm_name().
        • Other primitives → convert to str and lower-case so the scorer
          never crashes. You could also choose to return "" and mark as error.
        """
        if isinstance(item, str):
            return _norm_name(item)
        return str(item).lower().strip()

    @staticmethod
    def _to_list(val: Any) -> list:
        """None → []; list → list; scalar → [scalar]"""
        if val is None:
            return []
        return val if isinstance(val, list) else [val]

    def compute_correctness(
        self,
        reference: Dict[str, Any],
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Micro-averaged entity accuracy:
            correctness = TP / (TP + FP + FN)
        where a true positive (TP) is an exact match of both entity string and type
        after normalisation with _norm_name.
        """

        tp = fp = fn = 0
        errors: List[str] = []

        for ent_type in reference.keys():  # 8 mandatory slots
            ref_raw = self._to_list(reference.get(ent_type))
            pred_raw = self._to_list(prediction.get(ent_type))

            ref_set: Set[str] = {self._norm(e) for e in ref_raw}
            pred_set: Set[str] = {self._norm(e) for e in pred_raw}

            inter = ref_set & pred_set
            missed = ref_set - pred_set
            extra = pred_set - ref_set

            tp += len(inter)
            fn += len(missed)
            fp += len(extra)

            if missed:
                errors.append(f"Missing {ent_type}: {sorted(missed)}")
            if extra:
                errors.append(f"Spurious {ent_type}: {sorted(extra)}")

        total = tp + fp + fn
        overall = round(tp / total, 3) if total else 1.0

        return {
            "correctness": overall,
            "correctness_errors": errors,
        }

    def score_record(self, record: Dict) -> Dict[str, Any]:
        """
        Scores a single record against the reference.
        Returns a dict with compliance and correctness scores.
        """

        # --- Syntax validity
        validity_score, pred_dict = assess_json_valid(record["generated_output"])

        if not isinstance(pred_dict, Dict):
            return {
                "is_valid": validity_score,
                "compliance": 0,
                "correctness": 0,
            }

        compliance = self.compute_compliance(pred_dict)
        correctness = self.compute_correctness(record["output"], pred_dict)

        return {"is_valid": validity_score, **compliance, **correctness}


class TopV1Evaluator:
    def __init__(self, schema_loc: str = "data/clean/4-TOPv1/schema.json"):
        self.schema = json.load(open(schema_loc))
        self.validator = GeneralJsonSchemaEvaluator(self.schema)  # caching once

    def compute_compliance(self, output: Dict) -> Dict[str, Any]:
        """
        Computes compliance of the JSON string against the schema.
        Returns a dict with compliance percentage and errors.
        """
        return self.validator.score_against_schema(output)

    def compute_correctness(
        self, reference: Dict[str, Any], prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute correctness of the prediction against the reference, using the official TOPv1 metrics:
        | component      | metric                                          | weight |
        | -------------- | ----------------------------------------------- | ------ |
        | **Top-level**  | root-intent accuracy (0 / 1)                    | 0.25   |
        |                | slot-label micro-F1 (all depths, ignore values) | 0.25   |
        | **Whole tree** | TOP triplet F1 (exact-structure match)          | 0.50   |
        | **Overall**    | 0.5 × (top-level) + 0.5 × (tree-F1)             | 1.00   |
        """

        errors: List[str] = []

        # ----- intent accuracy ---------------------------------------------
        intent_correct = reference.get("intent") == prediction.get("intent")
        if not intent_correct:
            errors.append(
                f"root-intent mismatch: ref={reference['intent']!s} "
                f"pred={prediction.get('intent')!s}"
            )

        # ----- slot-label F1 (all depths, ignore values) -------------------
        gold_slots = self._flatten_slot_labels(reference)
        pred_slots = self._flatten_slot_labels(prediction)
        tp_slots = len(gold_slots & pred_slots)
        slot_f1 = f1(
            tp_slots, fp=len(pred_slots - gold_slots), fn=len(gold_slots - pred_slots)
        )
        if slot_f1 < 1.0:
            errors.append(
                f"slot-label F1={slot_f1:.2f} "
                f"(gold={sorted(gold_slots)}, pred={sorted(pred_slots)})"
            )

        # ----- tree-match F1 (triplets) ------------------------------------
        gold_trips = self._collect_triplets(reference)
        pred_trips = self._collect_triplets(prediction)
        tp_trips = len(gold_trips & pred_trips)
        tree_f1 = f1(
            tp_trips, fp=len(pred_trips - gold_trips), fn=len(gold_trips - pred_trips)
        )
        if tree_f1 < 1.0:
            errors.append(f"tree-triplet F1={tree_f1:.2f}")

        # ----- aggregate ----------------------------------------------------
        top_level_score = 0.5 * intent_correct + 0.5 * slot_f1
        overall = 0.5 * top_level_score + 0.5 * tree_f1

        return {
            "correctness": round(overall, 3),
            "correctness_errors": errors,
        }

    @classmethod
    def _flatten_slot_labels(cls, frame: Dict[str, Any]) -> Set[str]:
        """Return the set of *all* slot names that appear anywhere in the frame."""
        labels = set(frame.get("slots", {}).keys())
        for val in frame.get("slots", {}).values():
            if isinstance(val, dict) and "intent" in val and "slots" in val:
                labels |= cls._flatten_slot_labels(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "intent" in item and "slots" in item:
                        labels |= cls._flatten_slot_labels(item)
        return labels

    @classmethod
    def _collect_triplets(
        cls, frame: Dict[str, Any], parent_path: Tuple[str, ...] = ()
    ) -> Set[Tuple[Tuple[str, ...], str, str]]:
        """
        Produce the official TOP-style triplets:
            (path-to-parent, slot_name, child_intent_or_STRING)
        where *path* is the sequence of slot labels from the root down to the parent.
        """
        trips = set()
        for slot_name, val in frame.get("slots", {}).items():
            if isinstance(val, dict) and "intent" in val and "slots" in val:
                # child is a frame
                trips.add((parent_path, slot_name, val["intent"]))
                trips |= cls._collect_triplets(val, parent_path + (slot_name,))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict) and "intent" in item and "slots" in item:
                        trips.add((parent_path, slot_name, item["intent"]))
                        trips |= cls._collect_triplets(item, parent_path + (slot_name,))
                    else:
                        # primitive inside list → mark as STRING
                        trips.add((parent_path, slot_name, "STRING"))
            else:
                trips.add((parent_path, slot_name, "STRING"))
        return trips

    def score_record(self, record: Dict) -> Dict[str, Any]:
        """
        Scores a single record against the reference.
        Returns a dict with compliance and correctness scores.
        """

        # --- Syntax validity
        validity_score, pred_dict = assess_json_valid(record["generated_output"])

        if not isinstance(pred_dict, Dict):
            return {
                "is_valid": validity_score,
                "compliance": 0,
                "correctness": 0,
            }

        compliance = self.compute_compliance(pred_dict)
        correctness = self.compute_correctness(record["output"], pred_dict)

        return {"is_valid": validity_score, **compliance, **correctness}
    
    
class ApiBankEvaluator:
    def __init__(self):
        pass

    def compute_compliance(self, output: Dict, schema: Dict) -> Dict[str, Any]:

        """
        The useful compliance calculator, but with a twist to focus on the right API:
        Compute the compliance score of an output against a schema.
        This function evaluates how well the output complies with the schema by:
        1. Finding the branch in the schema that matches the output's 'api_name'.
        2. Using a general JSON schema evaluator to score the output against the matched branch.
        Parameters:
        ----------
        output : Dict
            The output dictionary to be evaluated.
        schema : Dict
            The JSON schema dictionary to evaluate against, which may contain multiple branches.
        Returns:
        -------
        Dict[str, Any]
            A dictionary containing at least:
            - 'compliance': float between 0.0 and 1.0, where 1.0 means fully compliant
            - 'compliance_errors': list of strings describing compliance errors (if any)
        """
        

        # 1️⃣  choose the branch that matches api_name (if any)
        branches = schema.get("oneOf", []) or [schema]     # cope with no oneOf
        chosen = next(
            (b for b in branches
            if b.get("properties", {}).get("api_name", {}).get("const")
                == output.get("api_name")),
            None
        )

        if chosen is None:       # api_name not allowed at all
            return {"compliance": 0.0,
                    "compliance_errors": ["api_name not in allowed set"]}

        # 2️⃣  run the generic evaluator on that branch
        validator = GeneralJsonSchemaEvaluator(chosen)
        return validator.score_against_schema(output)

    @staticmethod
    def _equivalent(a, b) -> bool:
        """
        Return True if two JSON scalars / arrays / dicts should be treated
        as the same *value* for the purpose of correctness.
        – lists compare order-insensitively
        – "970420" vs 970420 is OK
        """
        # numeric ↔ string-of-digits coercion
        if isinstance(a, numbers.Number) and isinstance(b, str) and b.isdigit():
            b = type(a)(b)
        if isinstance(b, numbers.Number) and isinstance(a, str) and a.isdigit():
            a = type(b)(a)

        # list ↔ list: ignore ordering of simple scalars
        if isinstance(a, list) and isinstance(b, list):
            try:
                return sorted(a) == sorted(b)
            except TypeError:
                # list of dicts etc.: fall back to exact
                return a == b
        return a == b   

    def compute_correctness(
        self, reference: Dict[str, Any], prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        1.0  = perfect match
        0.75 = right API + half the parameters right
        0.5  = right API only
        etc.
        """
        errors: List[str] = []
        score: float = 0.0

        # ---------- API name ---------- #
        if prediction.get("api_name") == reference.get("api_name"):
            score += 0.5
        else:
            errors.append(
                f"api_name mismatch: expected '{reference.get('api_name')}', got "
                f"'{prediction.get('api_name')}'"
            )

        # ---------- parameters ---------- #
        ref_params = reference.get("parameters", {}) or {}
        pred_params = prediction.get("parameters", {}) or {}

        if not isinstance(pred_params, dict):
            errors.append("`parameters` field missing or not an object")
        else:
            # per-key credit
            per_key_weight = 0.5 / max(len(ref_params), 1)
            for k, v in ref_params.items():
                if k not in pred_params:
                    errors.append(f"parameter '{k}' missing")
                elif self._equivalent(v, pred_params[k]):
                    score += per_key_weight
                else:
                    errors.append(
                        f"parameter '{k}' wrong value: expected {v!r}, got {pred_params[k]!r}"
                    )

            # unexpected keys
            extra = set(pred_params) - set(ref_params)
            if extra:
                errors.append(f"unexpected parameters: {', '.join(sorted(extra))}")

            # special case: no parameters expected at all
            if not ref_params:
                score += 0.5   # full parameter slice

        # round for readability
        return {
            "correctness": round(score, 3),
            "correctness_errors": errors,
        }

    
    def score_record(self, record: Dict) -> Dict[str, Any]:
        """
        Scores a single record against the reference.
        Returns a dict with compliance and correctness scores.
        
        :param reference_record: The reference record containing the schema and output.
        :param prediction_record: The prediction record as a JSON string.
        """

        # --- Syntax validity
        validity_score, pred_dict = assess_json_valid(record["generated_output"])

        if not isinstance(pred_dict, Dict):
            return {
                "is_valid": validity_score,
                "compliance": 0,
                "correctness": 0,
            }

        compliance = self.compute_compliance(pred_dict, record["json_schema"])
        correctness = self.compute_correctness(record["output"], pred_dict)

        return {"is_valid": validity_score, **compliance, **correctness}
    
    
class ReasoningEvaluator:
    def __init__(self, subtask="GSM8K"):
        self.subtask = subtask
        self.schema = json.load(open(f"data/clean/6-reasoning/{subtask}/schema.json"))
        self.validator = GeneralJsonSchemaEvaluator(self.schema)

    def compute_compliance(self, output: Dict) -> Dict[str, Any]:
        """
        Computes compliance of the JSON string against the schema.
        Returns a dict with compliance percentage and errors.
        """
        return self.validator.score_against_schema(output)
    
    def compute_correctness(
        self, reference: Dict[str, Any], prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """"Returns correctness score and errors for reasoning tasks.
        Here, correctness is defined as an exact match of the 'answer' field (reasoning is not evaluated).
        Normalisation is applied to compare strings in a robust way and disregard type issues (assessed in compliance).
        """

        score = _norm_name(str(prediction.get("answer", ""))) == _norm_name(
            str(reference.get("answer", ""))
        )
        errors: List[str] = []
        if not score:
            errors.append(
                f"answer mismatch: expected '{reference.get('answer')}', got "
                f"'{prediction.get('answer')}'"
            )
        
        return {
            "correctness": round(score, 3),
            "correctness_errors": errors,
        }
    
    def score_record(self, record:Dict) -> Dict[str, Any]:
        """
        Scores a single record against the reference.
        Returns a dict with compliance and correctness scores.
        
        :param reference_record: The reference record.
        :param prediction_record: The prediction record as a JSON string.
        """

        # --- Syntax validity
        validity_score, pred_dict = assess_json_valid(record["generated_output"])

        if not isinstance(pred_dict, Dict):
            return {
                "is_valid": validity_score,
                "compliance": 0,
                "correctness": 0,
            }

        compliance = self.compute_compliance(pred_dict)
        correctness = self.compute_correctness(record["output"], pred_dict)

        return {"is_valid": validity_score, **compliance, **correctness}

def evaluate_task(evaluator: Any, bench_path: Path) -> List[Dict[str, Any]]:
    """
    Evaluates all records for a given task.
    """
    with open(bench_path, 'r') as f:
        bench_data = json.load(f)
    scores = []
    for record in bench_data:
        try:
            scores.append(evaluator.score_record(record))
        except Exception as e:
            logger.error(f"Error evaluating record: {e}")
            scores.append({"is_valid": 0, "compliance": 0, "correctness": 0})
    return scores

def evaluate_repo(repo_path: str | Path, output_file: str) -> None:
    """Evaluate all benchmark tasks for a single repository.

    Parameters
    ----------
    repo_path : str | Path
        Path to the repository directory (e.g. ``results/gemma-3-1b-it``).
    output_file : str
        Path to the JSON file where consolidated results should be stored.
    """
    repo_path = Path(repo_path)
    repo_name = repo_path.name

    # ------------- logging -------------
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"evaluate_{repo_name}.log"

    # Reset root handlers to avoid duplicate logs when evaluating multiple repos
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )

    logging.info(f"Starting evaluation for repository: {repo_name}")

    # ------------- evaluators -------------
    evaluators: Dict[str, Any] = {
        "1-rotowire": RotowireEvaluator(),
        "2-wiki_bio": WikiBioEvaluator(),
        "3-few_nerd": FewNerdsEvaluator(),
        "4-TOPv1": TopV1Evaluator(),
        "5-api_bank": ApiBankEvaluator(),
        "6-reasoning/GSM8K": ReasoningEvaluator(subtask="GSM8K"),
        "6-reasoning/last_letter": ReasoningEvaluator(subtask="last_letter"),
    }

    all_results: Dict[str, Dict[str, float]] = {}

    for task_name, evaluator in evaluators.items():
        logging.info(f"Evaluating task: {task_name}")
        bench_path = repo_path / task_name / "generated.json"

        if not bench_path.exists():
            logging.warning(f"Benchmark file not found, skipping: {bench_path}")
            continue

        task_scores = evaluate_task(evaluator, bench_path)
        logging.info(f"Computed scores for {task_name}")

        if not task_scores:
            continue

        numeric_keys: Set[str] = {
            key
            for s in task_scores
            for key, value in s.items()
            if isinstance(value, numbers.Number)
        }

        agg_scores: Dict[str, float] = {
            key: round(
                float(
                    np.mean(
                        [s[key] for s in task_scores if key in s and isinstance(s[key], numbers.Number)]
                    )
                ),
                3,
            )
            for key in numeric_keys
        }

        all_results[task_name] = agg_scores
        logging.info(f"Aggregated scores for {task_name}: {agg_scores}")

    # ------------- write output -------------
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "r") as f:
            current_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        current_results = {}

    current_results[repo_name] = all_results

    with open(output_path, "w") as f:
        json.dump(current_results, f, indent=4)

    logging.info("Evaluation complete for repository: %s", repo_name)


def main():
    """
    Main function to run the evaluation for all benchmark tasks.
    """
    parser = argparse.ArgumentParser(description="Evaluate benchmark results.")
    parser.add_argument(
        "--bench_repo",
        type=str,
        default="all",
        help="Path to the directory containing the generated benchmark files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/bench_results.json",
        help="Path to save the consolidated evaluation results."
    )
    args = parser.parse_args()

    if args.bench_repo.lower() == "all":
        results_dir = Path("results")
        bench_repos = [d for d in results_dir.iterdir() if d.is_dir()]

        for repo in bench_repos:
            evaluate_repo(repo, output_file=args.output_file)
    else:
        evaluate_repo(Path(args.bench_repo), output_file=args.output_file)

if __name__ == '__main__':
    main()
    # uv run python -m  src.evaluate --bench_repo results/gemini-2.5-flash-preview-05-20 --output_file results/bench_results.json
    # uv run python -m  src.evaluate --bench_repo results/gemma-3-4b-it     