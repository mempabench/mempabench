"""
coverage.py — show per-setting coverage across persona YAML files.

Usage:
    python coverage.py user_a_preferences.yaml user_b_preferences.yaml ...
"""

import sys
import yaml
from collections import defaultdict

# Full taxonomy: attribute → ordered list of canonical settings
TAXONOMY = {
    # Dim 1: Expression Style
    "tone_formality":        ["Casual", "Consultative", "Formal"],
    "verbosity":             ["Terse", "Detailed"],
    "emotional_engagement":  ["Task-focused", "Relationship-focused"],
    "guidance_level":        ["Assumed", "Guided"],
    # Dim 2: Disclosure
    "reasoning_visibility":  ["Show", "Hide"],
    "uncertainty_expression":["Express", "Hide"],
    "process_visibility":    ["Silent", "Bookend", "Full narration"],
    # Dim 3: Initiative & Autonomy
    "autonomy_level":        ["Reactive", "Self-directed", "Autonomous"],
    "proactive_outreach":    ["Low", "High"],
    "task_expansion":        ["Low", "High"],
    "solution_breadth":      ["Low", "High"],
    "capability_boundary":   ["Suggest alternatives", "Find and hand off"],
    # Dim 4: Information Flow
    "information_elicitation":["Infer", "Structured", "Iterative"],
    "topic_management":      ["Organize", "One-at-a-time"],
}

DIMENSIONS = [
    ("Dim 1  Expression Style",   ["tone_formality", "verbosity", "emotional_engagement", "guidance_level"]),
    ("Dim 2  Disclosure",         ["reasoning_visibility", "uncertainty_expression", "process_visibility"]),
    ("Dim 3  Initiative & Autonomy", ["autonomy_level", "proactive_outreach", "task_expansion", "solution_breadth", "capability_boundary"]),
    ("Dim 4  Information Flow",   ["information_elicitation", "topic_management"]),
]


def load_persona(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    name = data.get("character", path)
    matrix = data.get("matrix", {})
    # collect (attr, setting) per context — unit is (character × context)
    slots = []  # list of (ctx, attr, setting)
    by_source = defaultdict(list)  # source → ["ctx.attr", ...]
    for ctx, ctx_data in matrix.items():
        for attr, entry in ctx_data.items():
            by_source[entry.get("source")].append(f"{ctx}.{attr}")
            if entry.get("source") == "no_preference":
                continue
            s = entry.get("setting")
            if s:
                slots.append((ctx, attr, s))
    return name, slots, by_source


def main(paths):
    loaded = [load_persona(p) for p in paths]
    personas = [(n, slots) for n, slots, _ in loaded]
    names = [n for n, _ in personas]

    # (attr, setting) → list of "Character (ctx)" labels
    coverage = defaultdict(list)
    for name, slots in personas:
        for ctx, attr, setting in slots:
            coverage[(attr, setting)].append(f"{name} ({ctx})")

    # ── print header ──────────────────────────────────────────────────────────
    col_attr    = 24
    col_setting = 22
    col_count   = 7
    col_chars   = 45

    sep = "─" * (col_attr + col_setting + col_count + col_chars + 7)

    header = (f"{'Attribute':<{col_attr}}  {'Setting':<{col_setting}}  "
              f"{'n':>{col_count}}  {'Slots (character × context)':<{col_chars}}")
    print()
    print(f"  Personas: {', '.join(names)}")
    print()
    print(header)

    for dim_label, attrs in DIMENSIONS:
        print(sep)
        print(f"  {dim_label}")
        print(sep)
        for attr in attrs:
            canonical = TAXONOMY[attr]
            for i, setting in enumerate(canonical):
                slots = coverage.get((attr, setting), [])
                count = len(slots)
                flag = "  ❌" if count == 0 else ("  ⚠" if count == 1 else "")
                attr_col = attr if i == 0 else ""
                slots_str = ", ".join(slots) if slots else "—"
                print(f"  {attr_col:<{col_attr}}  {setting:<{col_setting}}  "
                      f"{count:>{col_count}}{flag}  {slots_str}")
    print(sep)
    print()

    # ── summary ───────────────────────────────────────────────────────────────
    all_settings = [(a, s) for a, ss in TAXONOMY.items() for s in ss]
    zero = [(a, s) for a, s in all_settings if not coverage.get((a, s))]
    solo = [(a, s) for a, s in all_settings if len(coverage.get((a, s), [])) == 1]

    total_slots = sum(len(slots) for _, slots in personas)
    print(f"  Total settings in taxonomy : {len(all_settings)}")
    print(f"  Covered (≥1 slot)          : {len(all_settings) - len(zero)}")
    print(f"  Missing (0 slots)          : {len(zero)}")
    if zero:
        for a, s in zero:
            print(f"    ❌  {a} → {s}")
    print(f"  Single-slot coverage       : {len(solo)}")
    print(f"  Total slots across personas: {total_slots}")
    print()

    # ── provenance (computed from cells; never stored in the YAML) ───────────
    print("  Cells by source (seed = reviewer authored a setting without evidence;")
    print("  no_preference = reviewer decided not to assign one):")
    for name, _, by_source in loaded:
        counts = ", ".join(f"{src}={len(cells)}" for src, cells in sorted(by_source.items()))
        print(f"    {name}: {counts}")
        for src in ("seed", "no_preference"):
            if by_source.get(src):
                print(f"      {src}: {', '.join(sorted(by_source[src]))}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: coverage.py <yaml1> [yaml2 ...]")
        sys.exit(1)
    main(sys.argv[1:])
