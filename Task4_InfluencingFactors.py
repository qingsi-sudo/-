# -*- coding: utf-8 -*-
"""
================================================================================
Task 4: Influencing Factors Analysis Model
================================================================================

Based on the framework from the problem:
3.1 Research Questions: professional dancers, celebrity characteristics, judge vs fan
3.2 Model 1: Judge score  S_i,t = α0 + α1*Age + α2*Industry + α3*Partner + α4*Week + ε
3.2 Model 2: Audience vote log(V_i,t) = β0 + β1*S_i,t + β2*Age + β3*Industry + β4*Partner + β5*Week + u
3.3 Model 3: Final placement (log) = γ0 + γ1*S_bar + γ2*Age + γ3*Industry + γ4*Partner + η
3.3 Dancer effect: Y = X'β + α_partner + ε, rank(α̂_partner), ρ_partner = Var(α_partner)/Var(Y)
3.4 Star feature: age linear/quadratic, optimal age; industry dummies; interaction Age×Industry

Outputs: regression tables, partner ranking, variance decomposition, judge vs fan comparison.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# Paths (adjust DATA_PATH if your CSV is elsewhere)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_SCRIPT_DIR, "2026_MCM_Problem_C_Data.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = r"d:\Data\xwechat_files\wxid_m7pucc5xg5m522_f378\msg\file\2026-01\中文版赛题 (1)\中文版赛题\2026_MCM_Problem_C_Data.csv"
FAN_VOTES_PATH = os.path.join(_SCRIPT_DIR, "Task1_Results", "fan_vote_enhanced.csv")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "Task4_Results")

# Note for .txt alignment: Notepad uses proportional font by default; use monospace or open CSV in Excel
TXT_HEADER = (
    "Note: For aligned columns, open this file with a monospace font (e.g., Consolas, Courier New) in Notepad, "
    "or open the corresponding _coefficients.csv in Excel.\n\n"
)


def save_regression_outputs(result: dict, txt_path: str, csv_path: str) -> None:
    """
    Save regression result as: (1) .txt with header note + summary for monospace viewing;
    (2) .csv coefficient table for Excel (aligned columns).
    """
    if "error" in result or "summary" not in result:
        return
    summary = result["summary"]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(TXT_HEADER)
        f.write(summary)
    model = result.get("model")
    if model is not None:
        try:
            ci = model.conf_int()
            ci.columns = ["ci_lower", "ci_upper"]
            tbl = pd.DataFrame({
                "variable": model.params.index,
                "coef": model.params.values,
                "std_err": model.bse.values,
                "t": model.tvalues.values,
                "p_value": model.pvalues.values,
                "ci_lower": ci["ci_lower"].values,
                "ci_upper": ci["ci_upper"].values,
            })
            tbl.to_csv(csv_path, index=False, encoding="utf-8-sig")
        except Exception:
            pass


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw competition data and compute judge totals, placement, elim_week."""
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
    if 'placement' not in df.columns and 'final_placement' in df.columns:
        df['placement'] = df['final_placement']
    
    return df


def build_panel(df: pd.DataFrame, fan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build panel: one row per (season, week, celebrity) with judge score, fan votes, age, industry, partner.
    """
    fan_lookup = {}
    for _, row in fan_df.iterrows():
        key = (int(row['season']), int(row['week']), row['celebrity_name'].strip())
        fan_lookup[key] = float(row.get('estimated_votes', row.get('prior_votes', 0)))
    
    rows = []
    for _, row in df.iterrows():
        season = row['season']
        celeb = row.get('celebrity_name', row.get('Celebrity', ''))
        if pd.isna(celeb):
            continue
        celeb = str(celeb).strip()
        age = row.get('celebrity_age_during_season', row.get('age', 35))
        if pd.isna(age):
            age = 35
        try:
            age = float(age)
        except:
            age = 35
        industry = row.get('celebrity_industry', row.get('occupation', 'Unknown'))
        if pd.isna(industry):
            industry = 'Unknown'
        industry = str(industry).strip()
        partner = row.get('ballroom_partner', row.get('partner', 'Unknown'))
        if pd.isna(partner):
            partner = 'Unknown'
        partner = str(partner).strip()
        placement = row.get('placement', row.get('final_placement', 99))
        if pd.isna(placement) or placement >= 99:
            placement = 10
        
        for week in range(1, 12):
            col = f'week{week}_total'
            if col not in row.index:
                continue
            judge_score = row[col]
            if pd.isna(judge_score) or judge_score <= 0:
                continue
            active = (row['elim_week'] == -1) or (row['elim_week'] >= week)
            if not active:
                continue
            
            fan_votes = fan_lookup.get((season, week, celeb), np.nan)
            if pd.isna(fan_votes) or fan_votes <= 0:
                fan_votes = 1.0  # avoid log(0)
            
            rows.append({
                'season': season,
                'week': week,
                'celebrity_name': celeb,
                'judge_score': float(judge_score),
                'fan_votes': float(fan_votes),
                'log_fan_votes': np.log(float(fan_votes) + 1),
                'age': age,
                'industry_raw': industry,
                'partner': partner,
                'placement': placement,
            })
    
    panel = pd.DataFrame(rows)
    return panel


def industry_dummy_column(industry: str) -> str:
    """Map industry to broad category for dummy: Athlete, Actor, Singer, Other."""
    s = str(industry).lower()
    if 'athlete' in s or 'olympian' in s or 'sport' in s:
        return 'Athlete'
    if 'actor' in s or 'actress' in s:
        return 'Actor'
    if 'singer' in s or 'rapper' in s or 'music' in s:
        return 'Singer'
    if 'tv' in s or 'personality' in s or 'comedian' in s or 'model' in s or 'news' in s or 'you tube' in s or 'tiktok' in s:
        return 'Other'
    return 'Other'


def prepare_regression_data(panel: pd.DataFrame) -> pd.DataFrame:
    """Add dummy variables and average judge score per celebrity."""
    p = panel.copy()
    p['industry_cat'] = p['industry_raw'].apply(industry_dummy_column)
    
    # Average judge score per celebrity (for Model 3)
    avg_judge = p.groupby(['season', 'celebrity_name'])['judge_score'].transform('mean')
    p['judge_score_avg'] = avg_judge
    
    # Dummies (leave one out: Other as base)
    dummies = pd.get_dummies(p['industry_cat'], prefix='I_', drop_first=False)
    if 'I_Other' in dummies.columns:
        dummies = dummies.drop(columns=['I_Other'])
    for c in dummies.columns:
        p[c] = dummies[c]
    
    return p


def run_model1_judge_score(panel: pd.DataFrame) -> dict:
    """
    Model 1: S_i,t = α0 + α1*Age + α2*I_Industry + α3*Partner_dummy + α4*Week + ε
    Use partner as categorical (many levels) -> use top partners or hash to reduce levels.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {'error': 'statsmodels not installed', 'summary': None}
    
    p = panel.copy()
    p['industry_cat'] = p['industry_raw'].apply(industry_dummy_column)
    dummies = pd.get_dummies(p['industry_cat'], prefix='I_', drop_first=True)
    for c in dummies.columns:
        p[c] = dummies[c]
    
    # Partner: use top 20 partners by frequency, rest "Other"
    top_partners = p['partner'].value_counts().head(20).index.tolist()
    p['partner_cat'] = p['partner'].apply(lambda x: x if x in top_partners else 'Other')
    partner_dummies = pd.get_dummies(p['partner_cat'], prefix='P_', drop_first=True)
    if partner_dummies.shape[1] > 50:
        partner_dummies = partner_dummies.iloc[:, :30]
    for c in partner_dummies.columns:
        p[c] = partner_dummies[c]
    
    y = p['judge_score']
    ind_cols = ['age', 'week'] + [c for c in p.columns if c.startswith('I_')] + [c for c in p.columns if c.startswith('P_')]
    ind_cols = [c for c in ind_cols if c in p.columns]
    X = p[ind_cols]
    X = sm.add_constant(X)
    X = X.loc[:, ~X.columns.duplicated()]
    
    model = sm.OLS(y, X.astype(float)).fit()
    return {'model': model, 'summary': model.summary().as_text(), 'rsquared': model.rsquared, 'params': model.params.to_dict()}


def run_model2_audience_vote(panel: pd.DataFrame) -> dict:
    """
    Model 2: log(V_i,t) = β0 + β1*S_i,t + β2*Age + β3*I_Industry + β4*Partner + β5*Week + u
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {'error': 'statsmodels not installed', 'summary': None}
    
    p = panel.copy()
    p['industry_cat'] = p['industry_raw'].apply(industry_dummy_column)
    dummies = pd.get_dummies(p['industry_cat'], prefix='I_', drop_first=True)
    for c in dummies.columns:
        p[c] = dummies[c]
    
    top_partners = p['partner'].value_counts().head(20).index.tolist()
    p['partner_cat'] = p['partner'].apply(lambda x: x if x in top_partners else 'Other')
    partner_dummies = pd.get_dummies(p['partner_cat'], prefix='P_', drop_first=True)
    if partner_dummies.shape[1] > 50:
        partner_dummies = partner_dummies.iloc[:, :30]
    for c in partner_dummies.columns:
        p[c] = partner_dummies[c]
    
    y = p['log_fan_votes']
    ind_cols = ['judge_score', 'age', 'week'] + [c for c in p.columns if c.startswith('I_')] + [c for c in p.columns if c.startswith('P_')]
    ind_cols = [c for c in ind_cols if c in p.columns]
    X = p[ind_cols]
    X = sm.add_constant(X)
    X = X.loc[:, ~X.columns.duplicated()]
    
    model = sm.OLS(y, X.astype(float)).fit()
    return {'model': model, 'summary': model.summary().as_text(), 'rsquared': model.rsquared, 'params': model.params.to_dict()}


def run_model3_final_placement(panel: pd.DataFrame) -> dict:
    """
    Model 3: Placement_i (or log) = γ0 + γ1*S_bar_i + γ2*Age + γ3*Industry + γ4*Partner + η
    One row per celebrity (season level), S_bar = average judge score over weeks.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {'error': 'statsmodels not installed', 'summary': None}
    
    # One row per (season, celebrity)
    agg = panel.groupby(['season', 'celebrity_name']).agg({
        'judge_score': 'mean',
        'age': 'first',
        'industry_raw': 'first',
        'partner': 'first',
        'placement': 'min'
    }).reset_index()
    agg.rename(columns={'judge_score': 'judge_score_avg'}, inplace=True)
    agg['industry_cat'] = agg['industry_raw'].apply(industry_dummy_column)
    agg['log_placement'] = np.log(agg['placement'].clip(lower=1) + 1)
    
    dummies = pd.get_dummies(agg['industry_cat'], prefix='I_', drop_first=True)
    for c in dummies.columns:
        agg[c] = dummies[c]
    
    top_partners = agg['partner'].value_counts().head(25).index.tolist()
    agg['partner_cat'] = agg['partner'].apply(lambda x: x if x in top_partners else 'Other')
    partner_dummies = pd.get_dummies(agg['partner_cat'], prefix='P_', drop_first=True)
    if partner_dummies.shape[1] > 40:
        partner_dummies = partner_dummies.iloc[:, :25]
    for c in partner_dummies.columns:
        agg[c] = partner_dummies[c]
    
    y = agg['log_placement']
    ind_cols = ['judge_score_avg', 'age'] + [c for c in agg.columns if c.startswith('I_')] + [c for c in agg.columns if c.startswith('P_')]
    ind_cols = [c for c in ind_cols if c in agg.columns]
    X = agg[ind_cols]
    X = sm.add_constant(X)
    X = X.loc[:, ~X.columns.duplicated()]
    
    model = sm.OLS(y, X.astype(float)).fit()
    return {'model': model, 'summary': model.summary().as_text(), 'rsquared': model.rsquared, 'params': model.params.to_dict(), 'data': agg}


def partner_fixed_effects_ranking(panel: pd.DataFrame) -> dict:
    """
    Y_i,t = X'β + α_partner_i + ε.
    Estimate via regression with partner dummies; α̂_partner = coefficient for each partner.
    Partner rank = rank(α̂_partner). Variance decomposition: Var(Y) = Var(Xβ) + Var(α_partner) + Var(ε).
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {'error': 'statsmodels not installed'}
    
    p = panel.copy()
    top_partners = p['partner'].value_counts().head(50).index.tolist()
    p['partner_cat'] = p['partner'].apply(lambda x: x if x in top_partners else 'Other')
    partner_dummies = pd.get_dummies(p['partner_cat'], prefix='P_', drop_first=True)
    
    y = p['judge_score']
    X = partner_dummies.copy()
    X = sm.add_constant(X)
    X = X.loc[:, ~X.columns.duplicated()]
    
    model = sm.OLS(y, X.astype(float)).fit()
    params = model.params
    partner_effects = {k.replace('P_', ''): v for k, v in params.items() if k.startswith('P_')}
    ranked = sorted(partner_effects.items(), key=lambda x: -x[1])
    partner_rank = {name: rank + 1 for rank, (name, _) in enumerate(ranked)}
    
    y_pred = model.predict(X)
    var_y = np.var(y)
    var_pred = np.var(y_pred)
    var_resid = np.var(model.resid)
    rho_partner = var_pred / (var_y + 1e-10)
    
    return {
        'partner_effects': partner_effects,
        'partner_rank': partner_rank,
        'ranked_list': ranked,
        'var_y': var_y,
        'var_explained': var_pred,
        'rho_partner': rho_partner,
        'summary': model.summary().as_text()
    }


def star_feature_analysis(panel: pd.DataFrame) -> dict:
    """
    Age: linear and quadratic; optimal age Age* = -β1/(2β2).
    Industry: dummy encoding. Interaction: Age × 1_Athlete.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {'error': 'statsmodels not installed'}
    
    p = panel.copy()
    p['age2'] = p['age'] ** 2
    p['industry_cat'] = p['industry_raw'].apply(industry_dummy_column)
    dummies = pd.get_dummies(p['industry_cat'], prefix='I_', drop_first=True)
    p['I_Athlete'] = (p['industry_cat'] == 'Athlete').astype(int)
    p['age_x_athlete'] = p['age'] * p['I_Athlete']
    
    for c in dummies.columns:
        if c not in p.columns:
            p[c] = dummies[c]
    
    y = p['judge_score']
    use_cols = ['age', 'age2', 'I_Athlete', 'age_x_athlete']
    X = p[[c for c in use_cols if c in p.columns]].copy()
    X = sm.add_constant(X)
    X = X.loc[:, ~X.columns.duplicated()]
    
    model = sm.OLS(y, X.astype(float)).fit()
    params = model.params
    beta1 = params.get('age', 0)
    beta2 = params.get('age2', 0)
    optimal_age = -beta1 / (2 * beta2) if abs(beta2) > 1e-6 else np.nan
    
    return {
        'model': model,
        'summary': model.summary().as_text(),
        'optimal_age': optimal_age,
        'params': params.to_dict(),
        'rsquared': model.rsquared
    }


def main():
    print("="*70)
    print("TASK 4: INFLUENCING FACTORS ANALYSIS")
    print("="*70)
    
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        print("Please set DATA_PATH in the script to your 2026_MCM_Problem_C_Data.csv path.")
        return
    
    print("\n[1] Loading data and building panel...")
    df = load_raw_data(DATA_PATH)
    fan_df = pd.read_csv(FAN_VOTES_PATH)
    panel = build_panel(df, fan_df)
    panel = prepare_regression_data(panel)
    print(f"  Panel rows: {len(panel)}, celebrities: {panel['celebrity_name'].nunique()}, partners: {panel['partner'].nunique()}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n[2] Model 1: Judge score regression (S = α0 + α1*Age + α2*Industry + α3*Partner + α4*Week)")
    res1 = run_model1_judge_score(panel)
    if 'error' in res1:
        print(f"  {res1['error']}")
    else:
        print(f"  R-squared: {res1['rsquared']:.4f}")
        save_regression_outputs(res1,
            os.path.join(OUTPUT_DIR, "model1_judge_score_summary.txt"),
            os.path.join(OUTPUT_DIR, "model1_judge_score_coefficients.csv"))
    
    print("\n[3] Model 2: Audience vote regression (log(V) = β0 + β1*S + β2*Age + ...)")
    res2 = run_model2_audience_vote(panel)
    if 'error' in res2:
        print(f"  {res2['error']}")
    else:
        print(f"  R-squared: {res2['rsquared']:.4f}")
        save_regression_outputs(res2,
            os.path.join(OUTPUT_DIR, "model2_audience_vote_summary.txt"),
            os.path.join(OUTPUT_DIR, "model2_audience_vote_coefficients.csv"))
    
    print("\n[4] Model 3: Final placement regression (log(Placement) = γ0 + γ1*S_bar + ...)")
    res3 = run_model3_final_placement(panel)
    if 'error' in res3:
        print(f"  {res3['error']}")
    else:
        print(f"  R-squared: {res3['rsquared']:.4f}")
        save_regression_outputs(res3,
            os.path.join(OUTPUT_DIR, "model3_final_placement_summary.txt"),
            os.path.join(OUTPUT_DIR, "model3_final_placement_coefficients.csv"))
    
    print("\n[5] Partner (dancer) fixed effects & ranking")
    pe = partner_fixed_effects_ranking(panel)
    if 'error' in pe:
        print(f"  {pe['error']}")
    else:
        print(f"  Partner effect share of variance (rho): {pe['rho_partner']:.4f}")
        rank_df = pd.DataFrame([
            {'partner': name, 'effect': eff, 'rank': pe['partner_rank'][name]}
            for name, eff in pe['ranked_list'][:30]
        ])
        rank_df.to_csv(os.path.join(OUTPUT_DIR, "partner_effect_ranking.csv"), index=False)
        print(f"  Top 5 partners by effect: {[x[0] for x in pe['ranked_list'][:5]]}")
    
    print("\n[6] Star feature analysis (age linear/quadratic, optimal age, industry, interaction)")
    star = star_feature_analysis(panel)
    if 'error' in star:
        print(f"  {star['error']}")
    else:
        print(f"  Optimal age (from quadratic): {star['optimal_age']:.1f}")
        save_regression_outputs(star,
            os.path.join(OUTPUT_DIR, "star_feature_summary.txt"),
            os.path.join(OUTPUT_DIR, "star_feature_coefficients.csv"))
    
    # Judge vs Fan comparison: which factors matter more for judge vs fan
    print("\n[7] Judge vs Fan: coefficient comparison")
    judge_vs_fan = []
    if 'params' in res1 and 'params' in res2:
        common = set(res1['params']) & set(res2['params'])
        for coef in common:
            if coef == 'const':
                continue
            judge_vs_fan.append({
                'variable': coef,
                'coefficient_judge_model': res1['params'].get(coef, np.nan),
                'coefficient_fan_model': res2['params'].get(coef, np.nan),
            })
        comp_df = pd.DataFrame(judge_vs_fan)
        comp_df.to_csv(os.path.join(OUTPUT_DIR, "judge_vs_fan_coefficients.csv"), index=False)
        print(f"  Saved judge vs fan coefficients to judge_vs_fan_coefficients.csv")
    
    # Panel used for regression (sample)
    panel.head(500).to_csv(os.path.join(OUTPUT_DIR, "panel_sample.csv"), index=False)
    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
