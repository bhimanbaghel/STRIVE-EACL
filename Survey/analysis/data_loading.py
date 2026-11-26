"""
Data loading: Qualtrics CSV, stimuli CSV, LLM evaluation CSV.
"""

import re
import pandas as pd
import numpy as np
from .config import (
    COND_ORDER, CATEGORY_TO_RATING, CATEGORY_TO_BINARY,
    N_ITEMS, Q_START, QS_PER_ITEM,
    PLAUS_TEXT_MAP, DIFF_TEXT_MAP, IDENT_TEXT_MAP,
)


def load_qualtrics(filepath):
    """Load and parse Qualtrics export CSV, skipping metadata rows."""
    raw = pd.read_csv(filepath, header=None, dtype=str)
    data_start = 1
    for i in range(1, min(5, len(raw))):
        first_val = str(raw.iloc[i, 0]) if pd.notna(raw.iloc[i, 0]) else ""
        if first_val.startswith("{") or "plausible" in first_val.lower() or "ImportId" in first_val:
            data_start = i + 1
        elif first_val.startswith("20"):
            data_start = i
            break
    df = raw.iloc[data_start:].copy()
    df.columns = raw.iloc[0]
    df = df.reset_index(drop=True)
    q_text_row = dict(zip(raw.iloc[0], raw.iloc[1])) if data_start > 1 else None
    print(f"Loaded {len(df)} responses")
    return df, q_text_row


def load_stimuli(filepath):
    """Load stimulus set metadata from sampled_stimuli.csv."""
    sdf = pd.read_csv(filepath)
    items = {}
    for _, row in sdf.iterrows():
        item = int(row["item_num"])
        if item not in items:
            items[item] = {
                "tier": row["tier"], "model": row["model"],
                "source_run": row["source_run"], "source_type": row["source_type"],
                "verb": row["verb"], "conditions": {}
            }
        items[item]["conditions"][row["condition"]] = {
            "agent": row["agent"], "sentence": row["sentence"],
            "presentation_order": int(row["presentation_order"])
        }
    print(f"Loaded {len(items)} stimulus sets")
    return items


def load_llm_data(eval_filepath, items, eval_prefix="evalv3"):
    """
    Load LLM evaluator judgments from the merged evaluation CSV.

    For each stimulus item, finds the matching evaluation row and extracts:
    - The LLM's independent 4-point category (from actual_* columns)
    - The LLM's verdict (CORRECT/INCORRECT/BORDERLINE)
    - Derived binary judgment

    The eval_prefix controls which evaluator version to load. The source_run
    in sampled_stimuli.csv uses "evalv3_" prefix; this is swapped to the
    requested prefix before matching. This lets the same stimuli CSV work
    for any evaluator version (evalv3, evalv4, evalv4_qwen4b, etc.).
    """
    evals = pd.read_csv(eval_filepath, dtype=str)
    llm = {}
    matched_eval_runs = set()

    for item_num, item in items.items():
        # Swap prefix: evalv3_xxx -> evalv4_xxx (or whatever prefix is requested)
        source_run = item["source_run"]
        if eval_prefix != "evalv3" and source_run.startswith("evalv3_"):
            target_run = eval_prefix + "_" + source_run[len("evalv3_"):]
        else:
            target_run = source_run

        match = evals[
            (evals["eval_run"].str.strip() == target_run) &
            (evals["verb"].str.strip() == item["verb"])
        ]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        matched_eval_runs.add(str(row["eval_run"]).strip())

        judgments = {}
        for cond in COND_ORDER:
            actual_col = f"actual_{cond}"
            verdict_col = f"verdict_{cond}"
            actual_val = str(row.get(actual_col, "")).strip() if pd.notna(row.get(actual_col)) else ""
            verdict_val = str(row.get(verdict_col, "")).strip().upper() if pd.notna(row.get(verdict_col)) else ""

            if actual_val in CATEGORY_TO_RATING:
                llm_rating = CATEGORY_TO_RATING[actual_val]
                llm_category = actual_val
            elif verdict_val in ("CORRECT", "BORDERLINE"):
                llm_rating = CATEGORY_TO_RATING[cond]
                llm_category = cond
            elif verdict_val == "INCORRECT":
                llm_rating = None
                llm_category = None
            else:
                llm_rating = None
                llm_category = None

            llm_binary = CATEGORY_TO_BINARY.get(llm_category) if llm_category else None

            judgments[cond] = {
                "llm_category": llm_category,
                "llm_rating": llm_rating,
                "llm_binary": llm_binary,
                "verdict": verdict_val,
                "actual_raw": actual_val,
            }

        llm[item_num] = {
            "tier": item["tier"],
            "score_strict": int(row.get("score_strict", 0)) if pd.notna(row.get("score_strict")) else 0,
            "scene_verdict": str(row.get("scene_verdict", "")).strip(),
            "visual_verdict": str(row.get("visual_verdict", "")).strip(),
            "judgments": judgments,
        }

    print(f"Loaded LLM data for {len(llm)} items")
    print(f"  Matched eval_run values: {sorted(matched_eval_runs)}")
    return llm


def build_question_map(items, q_text_row=None):
    """Map Qualtrics question IDs to item/condition/type."""
    qmap = {}
    for item_num in range(1, N_ITEMS + 1):
        base = Q_START + (item_num - 1) * QS_PER_ITEM
        item = items[item_num]
        order_to_cond = {
            info["presentation_order"]: cond
            for cond, info in item["conditions"].items()
        }
        for p in range(1, 5):
            cond = order_to_cond[p]
            qmap[f"Q{base + (p-1)*2}"] = {"item": item_num, "type": "plausibility", "condition": cond}
            qmap[f"Q{base + (p-1)*2 + 1}"] = {"item": item_num, "type": "difficulty", "condition": cond}
        for off in range(4):
            ident_q = f"Q{base + 8 + off}"
            agent = None
            if q_text_row and ident_q in q_text_row:
                text = str(q_text_row[ident_q])
                # Try <b> tags first (some exports use them)
                m = re.search(r'<b>(.*?)</b>', text)
                if m:
                    agent = m.group(1)
                else:
                    # Fallback: extract from 'photograph of {agent} in their typical'
                    m = re.search(r'photograph of (.*?) in their typical', text, re.IGNORECASE)
                    if m:
                        agent = m.group(1)
            qmap[ident_q] = {"item": item_num, "type": "identifiability", "agent": agent}
        qmap[f"Q{base + 12}"] = {"item": item_num, "type": "pair_distinguish"}
    return qmap


def _parse_value(val, text_map):
    if pd.isna(val) or str(val).strip() == "":
        return None
    val = str(val).strip()
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return int(float(val))
    except ValueError:
        pass
    return text_map.get(val)


def extract_responses(df, qmap, items):
    """Extract structured responses from the Qualtrics dataframe."""
    annotator_ids = [f"A{i+1}" for i in range(len(df))]
    plaus, diff, ident, pair = [], [], [], []

    for a_idx, (_, row) in enumerate(df.iterrows()):
        aid = annotator_ids[a_idx]
        for qcol, info in qmap.items():
            if qcol not in df.columns:
                continue
            val = row.get(qcol)
            if info["type"] == "plausibility":
                r = _parse_value(val, PLAUS_TEXT_MAP)
                if r is not None:
                    plaus.append({
                        "annotator": aid, "item": info["item"],
                        "condition": info["condition"],
                        "tier": items[info["item"]]["tier"],
                        "model": items[info["item"]]["model"],
                        "verb": items[info["item"]]["verb"],
                        "rating": r,
                    })
            elif info["type"] == "difficulty":
                r = _parse_value(val, DIFF_TEXT_MAP)
                if r is not None:
                    diff.append({
                        "annotator": aid, "item": info["item"],
                        "condition": info["condition"], "rating": r,
                    })
            elif info["type"] == "identifiability":
                parsed = None
                if pd.notna(val) and str(val).strip():
                    v = str(val).strip()
                    parsed = IDENT_TEXT_MAP.get(v)
                    if parsed is None and v in ("1", "2", "3"):
                        parsed = {"1": "Yes", "2": "Maybe", "3": "No"}.get(v)
                if parsed:
                    ident.append({
                        "annotator": aid, "item": info["item"],
                        "agent": info.get("agent", "unknown"), "rating": parsed,
                    })
            elif info["type"] == "pair_distinguish":
                if pd.notna(val) and str(val).strip():
                    pair.append({
                        "annotator": aid, "item": info["item"],
                        "response": str(val).strip(),
                    })

    return (
        pd.DataFrame(plaus), pd.DataFrame(diff),
        pd.DataFrame(ident),
        pd.DataFrame(pair) if pair else pd.DataFrame(),
        annotator_ids,
    )