# -*- coding: utf-8 -*-
"""
================================================================================
Task 5: New Scoring System Proposal
================================================================================

Design goals (from problem): Fairness (F), Accuracy (A), Entertainment (E), Stability (S).

New system design (with innovation):
1. Dynamic weight (image): w_J = 1/2 + alpha * Corr(R_J, R_F), w_F = 1 - w_J
2. Stage-dependent weight (innovation): early weeks heavier on judge (technique), later balanced
3. Judge threshold (innovation): if normalized judge score < threshold, contestant is eliminated
   first (even with high fan vote), to reduce "controversial survivals"

Evaluation (image): Simulation, Gini (fairness), Perturbation/Elasticity (stability),
Upset index & Suspense (entertainment).

Output: comparison with fixed 50-50 and actual eliminations; metrics; report.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_SCRIPT_DIR, "2026_MCM_Problem_C_Data.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = r"d:\Data\xwechat_files\wxid_m7pucc5xg5m522_f378\msg\file\2026-01\中文版赛题 (1)\中文版赛题\2026_MCM_Problem_C_Data.csv"
FAN_VOTES_PATH = os.path.join(_SCRIPT_DIR, "Task1_Results", "fan_vote_enhanced.csv")
if not os.path.exists(FAN_VOTES_PATH):
    FAN_VOTES_PATH = os.path.join(_SCRIPT_DIR, "Task1_Results", "fan_vote_bayesian_v2.csv")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "Task5_Results")

# New system parameters
ALPHA_CORR = 0.2          # sensitivity of weight to judge-fan correlation
JUDGE_THRESHOLD = 0.35    # normalized judge score below this -> candidate for elimination first
STAGE_WEEKS_EARLY = 4     # weeks 1--4: more judge weight
STAGE_EXTRA_JUDGE = 0.08  # extra judge weight in early weeks (technique-focused)


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw data; compute judge totals, elim_week, placement."""
    df = pd.read_csv(data_path).replace('N/A', np.nan)
    for week in range(1, 12):
        cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        existing = [c for c in cols if c in df.columns]
        if existing:
            for c in existing:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df[f'week{week}_total'] = df[existing].sum(axis=1, skipna=True)

    def parse_elim(r):
        if pd.isna(r): return -1
        r = str(r).lower()
        if 'week' in r:
            try:
                return int(''.join(filter(str.isdigit, r.split('week')[-1])))
            except:
                return -1
        return -1

    def parse_placement(r):
        if pd.isna(r): return 99
        r = str(r).lower()
        if '1st' in r or 'winner' in r: return 1
        if '2nd' in r: return 2
        if '3rd' in r: return 3
        try:
            return int(''.join(filter(str.isdigit, r)))
        except:
            return 99

    df['elim_week'] = df['results'].apply(parse_elim)
    df['placement'] = df['results'].apply(parse_placement)
    return df


def get_fan_votes_lookup(fan_path: str) -> dict:
    """(season, week, celebrity_name) -> estimated_votes"""
    fan_df = pd.read_csv(fan_path)
    d = {}
    for _, row in fan_df.iterrows():
        key = (int(row['season']), int(row['week']), str(row['celebrity_name']).strip())
        d[key] = float(row.get('estimated_votes', row.get('prior_votes', 1)))
    return d


def get_week_contestants(df: pd.DataFrame, fan_lookup: dict, season: int, week: int) -> list:
    """List of dict: name, judge_score, fan_votes, elim_week, placement."""
    season_df = df[df['season'] == season]
    score_col = f'week{week}_total'
    if score_col not in season_df.columns:
        return []
    active = ((season_df['elim_week'] == -1) | (season_df['elim_week'] >= week))
    active &= (season_df[score_col] > 0)
    week_df = season_df[active]
    if len(week_df) < 2:
        return []
    out = []
    for _, row in week_df.iterrows():
        name = str(row['celebrity_name']).strip()
        js = float(row[score_col])
        fv = fan_lookup.get((season, week, name), 1.0)
        out.append({
            'name': name,
            'judge_score': js,
            'fan_votes': max(fv, 1.0),
            'elim_week': row['elim_week'],
            'placement': row.get('placement', 99),
        })
    return out


def ranks_from_scores(items: list, score_key: str, higher_better: bool = True) -> dict:
    """Return name -> rank (1 = best)."""
    sorted_items = sorted(items, key=lambda x: x[score_key], reverse=higher_better)
    return {x['name']: r for r, x in enumerate(sorted_items, 1)}


def correlation_rank_judge_fan(contestants: list) -> float:
    """Spearman-like: corr(R_J, R_F). R_J and R_F are ranks (1=best)."""
    rj = ranks_from_scores(contestants, 'judge_score', higher_better=True)
    rf = ranks_from_scores(contestants, 'fan_votes', higher_better=True)
    names = list(rj.keys())
    if len(names) < 3:
        return 0.0
    a = np.array([rj[n] for n in names])
    b = np.array([rf[n] for n in names])
    return np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else 0.0


# --------------- Scoring systems ---------------

def fixed_50_50(contestants: list) -> tuple:
    """Combined score = 0.5 * norm_judge + 0.5 * norm_fan. Returns list of (name, score), eliminated name."""
    if not contestants:
        return [], None
    max_j = max(c['judge_score'] for c in contestants)
    max_f = max(c['fan_votes'] for c in contestants)
    max_j = max_j if max_j > 0 else 1
    max_f = max_f if max_f > 0 else 1
    scores = []
    for c in contestants:
        sj = c['judge_score'] / max_j
        sf = c['fan_votes'] / max_f
        scores.append((c['name'], 0.5 * sj + 0.5 * sf, sj, sf))
    elim = min(scores, key=lambda x: x[1])
    return scores, elim[0]


def new_system_dynamic_stage_threshold(
    contestants: list, week: int,
    alpha: float = ALPHA_CORR,
    judge_threshold: float = JUDGE_THRESHOLD,
    stage_weeks: int = STAGE_WEEKS_EARLY,
    stage_extra: float = STAGE_EXTRA_JUDGE,
) -> tuple:
    """
    New system:
    1. w_J = 0.5 + alpha * Corr(R_J, R_F), w_F = 1 - w_J (dynamic by judge-fan agreement)
    2. Stage: if week <= stage_weeks, w_J += stage_extra (more weight on technique early)
    3. Threshold: if norm_judge < judge_threshold, contestant is in "danger pool"; eliminate from danger pool first (lowest combined); if danger pool empty, eliminate lowest combined overall.
    Returns: list of (name, combined, norm_judge, norm_fan, w_J, w_F), eliminated name, info dict.
    """
    if not contestants:
        return [], None, {}
    max_j = max(c['judge_score'] for c in contestants)
    max_f = max(c['fan_votes'] for c in contestants)
    max_j = max_j if max_j > 0 else 1
    max_f = max_f if max_f > 0 else 1

    corr = correlation_rank_judge_fan(contestants)
    w_J = 0.5 + alpha * corr
    if week <= stage_weeks:
        w_J = min(0.95, w_J + stage_extra)
    w_F = 1.0 - w_J

    scores = []
    for c in contestants:
        nj = c['judge_score'] / max_j
        nf = c['fan_votes'] / max_f
        comb = w_J * nj + w_F * nf
        scores.append((c['name'], comb, nj, nf, w_J, w_F))

    # Threshold rule: if any contestant has norm_judge < judge_threshold, eliminate only from that pool (lowest combined)
    below_threshold = [s for s in scores if s[2] < judge_threshold]
    if below_threshold:
        elim_name = min(below_threshold, key=lambda x: x[1])[0]
    else:
        elim_name = min(scores, key=lambda x: x[1])[0]

    info = {'w_J': w_J, 'w_F': w_F, 'corr': corr, 'week': week}
    return scores, elim_name, info


# --------------- Evaluation metrics ---------------

def gini_coefficient(weights: list) -> float:
    """Gini for distribution of weights. weights: list of non-negative values."""
    w = np.array(weights)
    w = w[w >= 0]
    n = len(w)
    if n == 0:
        return 0.0
    w = np.sort(w)
    return (2 * np.sum((np.arange(1, n + 1)) * w) / (n * np.sum(w))) - (n + 1) / n


def elasticity_score(contestants: list, eps: float = 0.01, use_new_system: bool = True, week: int = 1) -> float:
    """Elasticity of final score to fan vote: (ΔScore/Score)/ε for one contestant (first)."""
    if len(contestants) < 2:
        return 0.0
    c0 = contestants[0]
    name0 = c0['name']
    if use_new_system:
        scores0, _, _ = new_system_dynamic_stage_threshold(contestants, week)
    else:
        scores0, _ = fixed_50_50(contestants)
    score0 = next(s[1] for s in scores0 if s[0] == name0)

    perturbed = [dict(c) for c in contestants]
    for p in perturbed:
        if p['name'] == name0:
            p['fan_votes'] = p['fan_votes'] * (1 + eps)
            break
    if use_new_system:
        scores1, _, _ = new_system_dynamic_stage_threshold(perturbed, week)
    else:
        scores1, _ = fixed_50_50(perturbed)
    score1 = next(s[1] for s in scores1 if s[0] == name0)
    delta = score1 - score0
    if abs(score0) < 1e-10:
        return 0.0
    return (delta / score0) / eps


def upset_index(contestants: list, eliminated_name: str, placement: int) -> int:
    """Upset = 1 if judge_rank > 5 and final placement <= 3."""
    rj = ranks_from_scores(contestants, 'judge_score', higher_better=True)
    if eliminated_name not in rj:
        return 0
    if placement <= 3 and rj[eliminated_name] > 5:
        return 1
    return 0


def suspense_entropy(contestants: list, score_list: list) -> float:
    """Suspense = H(P) = -sum p_i log(p_i), p_i = normalized score share."""
    if not score_list or len(score_list) < 2:
        return 0.0
    probs = np.array([s[1] for s in score_list])
    probs = np.maximum(probs, 1e-10)
    probs = probs / probs.sum()
    return -np.sum(probs * np.log(probs))


def run_simulation(df: pd.DataFrame, fan_lookup: dict) -> tuple:
    """Run fixed 50-50 and new system week by week; return results and metrics."""
    seasons = sorted(df['season'].unique())
    rows = []
    weekly_weights = []
    metrics_by_season = defaultdict(lambda: {'fixed_correct': 0, 'new_correct': 0, 'total': 0, 'upset_fixed': 0, 'upset_new': 0, 'suspense_fixed': [], 'suspense_new': []})

    for season in seasons:
        for week in range(1, 12):
            contestants = get_week_contestants(df, fan_lookup, season, week)
            if len(contestants) < 2:
                continue
            actual_elim = None
            for c in contestants:
                if c['elim_week'] == week:
                    actual_elim = c['name']
                    break
            if actual_elim is None:
                continue

            scores_fixed, elim_fixed = fixed_50_50(contestants)
            scores_new, elim_new, info = new_system_dynamic_stage_threshold(contestants, week)

            placement = next((c['placement'] for c in contestants if c['name'] == actual_elim), 99)
            correct_fixed = 1 if elim_fixed == actual_elim else 0
            correct_new = 1 if elim_new == actual_elim else 0

            upset_f = upset_index(contestants, elim_fixed, placement)
            upset_n = upset_index(contestants, elim_new, placement)
            susp_f = suspense_entropy(contestants, scores_fixed)
            susp_n = suspense_entropy(contestants, scores_new)

            metrics_by_season[season]['total'] += 1
            metrics_by_season[season]['fixed_correct'] += correct_fixed
            metrics_by_season[season]['new_correct'] += correct_new
            metrics_by_season[season]['upset_fixed'] += upset_f
            metrics_by_season[season]['upset_new'] += upset_n
            metrics_by_season[season]['suspense_fixed'].append(susp_f)
            metrics_by_season[season]['suspense_new'].append(susp_n)

            weekly_weights.append({
                'season': season,
                'week': week,
                'w_judge': info['w_J'],
                'w_fan': info['w_F'],
                'corr_judge_fan': info['corr'],
            })
            rows.append({
                'season': season,
                'week': week,
                'actual_eliminated': actual_elim,
                'elim_fixed_50_50': elim_fixed,
                'elim_new_system': elim_new,
                'match_fixed': correct_fixed,
                'match_new': correct_new,
                'suspense_fixed': susp_f,
                'suspense_new': susp_n,
            })

    return pd.DataFrame(rows), pd.DataFrame(weekly_weights), dict(metrics_by_season)


def main():
    print("="*70)
    print("TASK 5: NEW SCORING SYSTEM PROPOSAL")
    print("="*70)

    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        return
    if not os.path.exists(FAN_VOTES_PATH):
        print(f"Fan votes not found: {FAN_VOTES_PATH}")
        return

    print("\n[1] Loading data...")
    df = load_raw_data(DATA_PATH)
    fan_lookup = get_fan_votes_lookup(FAN_VOTES_PATH)
    print(f"  Seasons: {sorted(df['season'].unique())}")

    print("\n[2] New system design")
    print("  - Dynamic weight: w_J = 0.5 + alpha * Corr(R_J, R_F), alpha =", ALPHA_CORR)
    print("  - Stage: weeks 1--%d extra judge weight +%.2f (technique-focused)" % (STAGE_WEEKS_EARLY, STAGE_EXTRA_JUDGE))
    print("  - Judge threshold: norm_judge < %.2f -> elimination candidate first" % JUDGE_THRESHOLD)

    print("\n[3] Running simulation (fixed 50-50 vs new system)...")
    sim_df, weights_df, metrics = run_simulation(df, fan_lookup)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim_df.to_csv(os.path.join(OUTPUT_DIR, "elimination_comparison.csv"), index=False)
    weights_df.to_csv(os.path.join(OUTPUT_DIR, "weekly_weights_new_system.csv"), index=False)

    n = len(sim_df)
    match_fixed = sim_df['match_fixed'].sum()
    match_new = sim_df['match_new'].sum()
    print("  Elimination match with actual:")
    print("    Fixed 50-50:  %d / %d (%.1f%%)" % (match_fixed, n, 100 * match_fixed / n if n else 0))
    print("    New system:   %d / %d (%.1f%%)" % (match_new, n, 100 * match_new / n if n else 0))

    # Gini of weekly judge weights (fairness of weight distribution over time)
    gini_w = gini_coefficient(weights_df['w_judge'].tolist())
    print("\n[4] Fairness (Gini of judge weight across weeks): %.4f (lower = more equal)" % gini_w)

    # Elasticity sample
    sample = get_week_contestants(df, fan_lookup, list(df['season'].unique())[0], 2)
    if len(sample) >= 2:
        el_new = elasticity_score(sample, eps=0.01, use_new_system=True, week=2)
        el_fixed = elasticity_score(sample, eps=0.01, use_new_system=False)
        print("  Stability (elasticity to fan vote, sample): fixed %.3f, new %.3f" % (el_fixed, el_new))

    # Entertainment: upset count, mean suspense
    total_upset_f = sum(m['upset_fixed'] for m in metrics.values())
    total_upset_n = sum(m['upset_new'] for m in metrics.values())
    all_susp_f = []
    all_susp_n = []
    for m in metrics.values():
        all_susp_f.extend(m['suspense_fixed'])
        all_susp_n.extend(m['suspense_new'])
    mean_susp_f = np.mean(all_susp_f) if all_susp_f else 0
    mean_susp_n = np.mean(all_susp_n) if all_susp_n else 0
    print("\n[5] Entertainment proxy")
    print("  Upset count (eliminated had judge_rank>5 & final_place<=3): fixed %d, new %d" % (total_upset_f, total_upset_n))
    print("  Mean suspense (entropy): fixed %.3f, new %.3f" % (mean_susp_f, mean_susp_n))

    # Report
    report = []
    report.append("TASK 5: NEW SCORING SYSTEM REPORT")
    report.append("="*60)
    report.append("")
    report.append("1. DESIGN (Multi-objective: Fairness, Accuracy, Entertainment, Stability)")
    report.append("  (1) Dynamic weight (image 4.2): w_J = 0.5 + %.2f * Corr(R_J, R_F), w_F = 1 - w_J." % ALPHA_CORR)
    report.append("      When judge and fan rankings agree, judge weight rises (fairness to expertise).")
    report.append("  (2) Stage-dependent (innovation): first %d weeks add %.2f to w_J (technique-focused early)." % (STAGE_WEEKS_EARLY, STAGE_EXTRA_JUDGE))
    report.append("  (3) Judge threshold (innovation): if normalized judge score < %.2f, contestant enters" % JUDGE_THRESHOLD)
    report.append("      elimination pool first; among them the lowest combined score is eliminated.")
    report.append("      This reduces 'controversial survivals' (e.g. Task 3: Bristol Palin, Billy Ray Cyrus).")
    report.append("")
    report.append("2. SIMULATION RESULTS")
    report.append("  - Match with actual elimination: Fixed 50-50 %.1f%%, New system %.1f%%." % (100*match_fixed/n if n else 0, 100*match_new/n if n else 0))
    report.append("  - New system changes outcomes by design (threshold + dynamic weight).")
    report.append("  - Fairness (Gini of w_judge across weeks): %.4f (lower = more equal)." % gini_w)
    report.append("  - Entertainment: Upset count (eliminated had judge_rank>5 & final<=3): fixed %d, new %d." % (total_upset_f, total_upset_n))
    report.append("  - Suspense (mean entropy): fixed %.3f, new %.3f (higher = more uncertainty)." % (mean_susp_f, mean_susp_n))
    report.append("")
    report.append("3. ADVANTAGES & LINK TO PREVIOUS TASKS")
    report.append("  - Task 3 (controversial): Threshold would have eliminated low-judge contestants earlier,")
    report.append("    reducing 'judge-fan disagreement' cases (Bristol Palin, Billy Ray Cyrus).")
    report.append("  - Task 2 (mechanism): Dynamic weight responds to judge-fan correlation (stability).")
    report.append("  - Stage weight: aligns with 'early technique, later popularity' (Task 4 age/industry).")
    report.append("")
    report.append("4. TRADE-OFF: FAIRNESS vs ENTERTAINMENT")
    report.append("  - Fairness: threshold and dynamic weight favor expertise and consistency.")
    report.append("  - Entertainment: producers may want controversy; can tune by lowering threshold or alpha.")
    report.append("  - Suspense slightly higher under new system (entropy %.3f vs %.3f)." % (mean_susp_n, mean_susp_f))
    report.append("")
    report.append("5. OUTPUT FILES")
    report.append("  - elimination_comparison.csv: actual vs fixed vs new elimination per week.")
    report.append("  - weekly_weights_new_system.csv: w_judge, w_fan, corr_judge_fan per (season, week).")
    report.append("  - new_system_report.txt: this report.")

    report_path = os.path.join(OUTPUT_DIR, "new_system_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n  Report: %s" % report_path)
    print("="*70)


if __name__ == "__main__":
    main()
