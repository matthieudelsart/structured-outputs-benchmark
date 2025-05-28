import json
import numpy as np
from jsonschema import Draft7Validator
from typing import Any, Dict, List, Tuple
from rapidfuzz import process, distance
import math

import json_repair

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
    
########## Evaluators 

class RotowireEvaluator:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
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
            return  {
            "is_valid": validity_score,
            "team_compliance_%": 0,
            "player_compliance_%": 0,
            "overall_compliance_%": 0,
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
            "team_compliance_%": team_compliance,
            "player_compliance_%": player_compliance,
            "overall_compliance_%": float(np.mean([team_compliance, player_compliance])),
            "errors": errors,
        }

    ##### Correctness evaluation

    def match_by_key(self,
        ref_objs: List[Dict],
        pred_objs: List[Dict],
        name_key: str,
        edit_threshold: int = 1
    ) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
        """
        Greedy 1-to-1 matching: exact names first, then fuzzy with Levenshtein ≤ threshold.
        Returns (matched_pairs, unmatched_ref, unmatched_pred).
        """
        ref_map  = {_norm_name(o[name_key]): o for o in ref_objs}
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
                r_name, list(pred_map.keys()),
                scorer=distance.Levenshtein.distance
            )
            if score <= edit_threshold:
                pairs.append((r_obj, pred_map.pop(best)))
                ref_map.pop(r_name)

        return pairs, list(ref_map.values()), list(pred_map.values())

    def compare_objects(self,
        ref_obj: Dict,
        pred_obj: Dict,
        value_tolerance: Dict[str, float] | None = None
    ) -> Tuple[int, int, int, int, int]:
        """
        Returns (TP_keys, FP_keys, FN_keys, correct_values, total_compared_values)
        """
        if value_tolerance is None:
            value_tolerance = {}

        tp = fp = fn = val_ok = val_tot = 0

        keys_union = ref_obj.keys() | pred_obj.keys()
        for k in keys_union:
            in_ref  = k in ref_obj
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

    def evaluate_game(self,
        reference: Dict,
        prediction: Dict,
        value_tolerance: Dict[str, float] | None = None,
        edit_threshold: int = 1
    ) -> Dict[str, float]:
        """
        Returns a dict with detection and attribute scores for one game.
        """
        # ---------------- Stage A
        t_pairs, t_miss_ref, t_miss_pred = self.match_by_key(
            reference["teams"], prediction.get("teams", []), "team", edit_threshold
        )
        p_pairs, p_miss_ref, p_miss_pred = self.match_by_key(
            reference["players"], prediction.get("players", []), "player", edit_threshold
        )

        team_det_tp = len(t_pairs)
        player_det_tp = len(p_pairs)
        
        # First metric : identification of the right objects 
        team_detect_F1 = f1(team_det_tp, len(t_miss_pred), len(t_miss_ref))
        player_detect_F1 =  f1(player_det_tp, len(p_miss_pred), len(p_miss_ref))

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
        team_val_acc   = t_v_ok / t_v_tot if t_v_tot else 0.0
        player_val_acc = p_v_ok / p_v_tot if p_v_tot else 0.0

        # Overall score for a given team / player
        team_attr_score   = gmean(team_attr_f1,   team_val_acc)
        player_attr_score = gmean(player_attr_f1, player_val_acc)

        # Overall score linking identification and key/values correctness
        overall_team_score = gmean(team_detect_F1,   team_attr_score)
        overall_player_score = gmean(player_detect_F1, player_attr_score)

        return {
            "ident_f1": (team_detect_F1 + player_detect_F1) / 2,
            "object_attr_F1": (team_attr_f1 + player_attr_f1) / 2,
            "object_value_acc": (team_val_acc + player_val_acc) / 2,
            "object_attribute_score": (team_attr_score + player_attr_score) / 2,
            "overall_score": (overall_team_score + overall_player_score) / 2
        }

class WikiBioEvaluator():
    def __init__(self):
        pass
    