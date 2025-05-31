import json
import math
import re
from typing import Any, Dict, List, Set, Tuple, Union

import json_repair
import numpy as np
import pycountry
from dateutil import parser as dt
from jsonschema import Draft7Validator, ValidationError, validators
from rapidfuzz import distance, process
from unidecode import unidecode

######### Helpers


def assess_json_valid(json_string: str):
    """
    Returns (1, record) if json_string is valid JSON,
    otherwise returns (0, repaired_record) using json_repair.
    """
    try:
        record = json.loads(json_string)
        return 1, record
    except json.JSONDecodeError:
        record = json_repair.loads(json_string)
        return 0, record


def f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def gmean(x, y):
    return math.sqrt(x * y) if x and y else 0


def _norm_name(name: str) -> str:
    """Lower-case, collapse whitespace, drop periods."""
    return " ".join(name.lower().replace(".", "").split())


def _numeric_equal(a, b, tol: int | float = 0) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return a == b


# ────── 1.  Normalisation helpers ───────────────────────────────────────────

DATE_PAREN_RE = re.compile(r"\(.*?\)")  # strip “(aged 67)”


def _norm_date(val: str) -> str:
    """ISO 8601 (`YYYY-MM-DD` or `YYYY-MM` if day unknown)."""
    val = DATE_PAREN_RE.sub("", val).strip()
    try:
        d = dt.parse(val, dayfirst=False, fuzzy=True)
    except Exception:
        return val.lower().strip()
    iso = d.date().isoformat()
    return iso if re.search(r"\b([12]?\d|3[01])\b", val) else iso[:7]


def _digits_only(val: str) -> str:
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
    v = unidecode(val).lower().strip().replace(".", "")
    if v in ALIASES:
        return ALIASES[v]
    try:  # pycountry catch-all
        return pycountry.countries.lookup(v).name
    except LookupError:
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
    parts = [p.strip() for p in val.split(",")]
    clean = [
        unidecode(p).lower() for p in parts if unidecode(p).lower() not in REDUNDANT
    ]
    return ", ".join(clean) if clean else unidecode(val).lower()


MULTI_SEP_RE = re.compile(r"[;,/•]+")


def _split_set(val: str) -> List[str]:
    return [s.strip() for s in MULTI_SEP_RE.split(val) if s.strip()]


def _normalise_value(key: str, val: Any) -> Any:
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
    """Set-aware equality."""
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

    def _leaf_count(self, schema: Dict[str, Any]) -> int:
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
        total_checks = self._leaf_count(
            self.schema
        )  # ③ how many independent checks exist
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
    ) -> Tuple[int, int]:
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

    def compliance_breakdown(self, json_string: str) -> Dict[str, float]:
        """Percentage compliance + real Draft-07 errors, handles all edge-cases."""

        # Check json valid
        validity_score, record = assess_json_valid(json_string)
        if not isinstance(record, Dict):
            return {
                "is_valid": validity_score,
                "team_compliance": 0,
                "player_compliance": 0,
                "overall_compliance": 0,
                "errors": None,
            }

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
            "is_valid": validity_score,
            "team_compliance": team_compliance,
            "player_compliance": player_compliance,
            "overall_compliance": float(np.mean([team_compliance, player_compliance])),
            "errors": errors,
        }

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
    ) -> Dict[str, float]:
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
        }


class WikiBioEvaluator:
    def __init__(self):
        pass

    def compute_compliance(
        self, json_string: str, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        • Parses *json_string* (repairing if needed) and flags JSON validity.
        • Runs Draft-07 compliance against *schema*.
        • Returns one merged dict:
            {
            "is_valid": 0|1,          # JSON well-formedness
            "compliance": 0-100,      # % of schema checks passed
            "errors": [...]           # jsonschema.ValidationError objects
            }
        """

        # --- Syntax validity
        validity_score, record = assess_json_valid(json_string)

        if not isinstance(record, Dict):
            return {
                "is_valid": validity_score,
                "compliance": 0,
                "errors": None,
            }

        # ---- Schema compliance
        validator = GeneralJsonSchemaEvaluator(schema)
        schema_result = validator.score_against_schema(record)

        return {"is_valid": validity_score, **schema_result}

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
        keys = reference["output"].keys()
        total = len(keys)
        correct = 0

        for k in keys:
            ref_val = _normalise_value(k, reference.get(k))
            try:
                pred_val = _normalise_value(k, prediction.get(k))
            except KeyError:
                errors.append(f"Key '{k}': Missing key)")

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

    def compute_score_all_records(
        self, outputs: str, reference_json: str = "data/clean/wiki_bio/bench.json"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Computes compliance and correctness for all records in the outputs file against the reference JSON.
        :param outputs: Path to the outputs JSON file.
        :param reference_json: Path to the reference JSON file.
        :return: A dictionary with record IDs as keys and their compliance and correctness results.
        """
        with open(reference_json, "r") as rf, open(outputs, "r") as o:
            reference_records = json.load(rf)
            output_records = json.load(o)

        results = {}
        for reference, output in zip(reference_records, output_records):
            compliance_results = self.compute_compliance(
                json_string=output["output"], schema=reference["schema"]
            )
            correctness_results = self.compute_correctness(
                reference=reference["output"],
                prediction=output["output"],
            )
            results[reference["id"]] = {
                "compliance": compliance_results,
                "correctness": correctness_results,
            }
        return results


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

    def score_record(self, reference_record, prediction_record: str) -> Dict[str, Any]:
        """
        Scores a single record against the reference.
        Returns a dict with compliance and correctness scores.
        """

        # --- Syntax validity
        validity_score, pred_dict = assess_json_valid(prediction_record)

        if not isinstance(pred_dict, Dict):
            return {
                "is_valid": validity_score,
                "compliance": 0,
                "correctness": 0,
            }

        compliance = self.compute_compliance(pred_dict)
        correctness = self.compute_correctness(reference_record, pred_dict)

        return {"is_valid": validity_score, **compliance, **correctness}

    def score_all_records(
        self,
    ):
        pass


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

    def score_record(self, reference_record, prediction_record: str) -> Dict[str, Any]:
        """
        Scores a single record against the reference.
        Returns a dict with compliance and correctness scores.
        """

        # --- Syntax validity
        validity_score, pred_dict = assess_json_valid(prediction_record)

        if not isinstance(pred_dict, Dict):
            return {
                "is_valid": validity_score,
                "compliance": 0,
                "correctness": 0,
            }

        compliance = self.compute_compliance(pred_dict)
        correctness = self.compute_correctness(reference_record["output"], pred_dict)

        return {"is_valid": validity_score, **compliance, **correctness}

    def score_all_records(
        self,
    ):
        pass
