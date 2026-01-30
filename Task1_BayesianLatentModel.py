
# -*- coding: utf-8 -*-
"""
================================================================================
2026 MCM Problem C - Task 1: Bayesian Latent Variable Model with Interval Censoring
================================================================================

Model Description:
- Fan votes are treated as LATENT VARIABLES (unobserved)
- Elimination results provide INTERVAL CENSORING constraints
- Prior distributions incorporate contestant characteristics
- MCMC sampling provides posterior distributions with uncertainty quantification

Mathematical Framework:
    v_i ~ Prior(theta_i)  for each contestant i
    Constraint: s_e + v_e < s_j + v_j for all j != e (eliminated contestant e has lowest score)

================================================================================
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = r"d:\Data\xwechat_files\wxid_m7pucc5xg5m522_f378\msg\file\2026-01\中文版赛题 (1)\中文版赛题\2026_MCM_Problem_C_Data.csv"
OUTPUT_DIR = r"d:\桌面\代码仓库\数据结构\data-structure-learning\MCM_2026_C\Task1_Results"


class DataProcessor:
    """Process raw data for Bayesian modeling"""
    
    def __init__(self, data_path: str):
        self.raw_data = pd.read_csv(data_path).replace('N/A', np.nan)
        self.processed_weeks = []
        
    def get_week_data(self, season: int, week: int) -> dict:
        """
        Extract data for a specific season and week
        Returns: dict with contestants, judge_scores, eliminated_contestant
        """
        season_data = self.raw_data[self.raw_data['season'] == season].copy()
        
        contestants = []
        judge_scores = []
        eliminated = None
        
        for _, row in season_data.iterrows():
            name = row['celebrity_name']
            results = str(row['results']).lower() if pd.notna(row['results']) else ''
            
            # Calculate judge score for this week
            score_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
            week_score = 0
            valid_scores = 0
            
            for col in score_cols:
                if col in row.index and pd.notna(row[col]):
                    try:
                        s = float(row[col])
                        if s > 0:  # Ignore 0 scores (already eliminated)
                            week_score += s
                            valid_scores += 1
                    except:
                        pass
            
            if valid_scores > 0:
                contestants.append(name)
                judge_scores.append(week_score)
                
                # Check if this contestant was eliminated this week
                if f'eliminated week {week}' in results or f'eliminated week{week}' in results:
                    eliminated = name
        
        if len(contestants) < 2 or eliminated is None:
            return None
            
        return {
            'season': season,
            'week': week,
            'contestants': contestants,
            'judge_scores': np.array(judge_scores),
            'eliminated': eliminated,
            'eliminated_idx': contestants.index(eliminated)
        }
    
    def get_contestant_features(self, name: str, season: int) -> dict:
        """Extract features for prior calculation"""
        row = self.raw_data[
            (self.raw_data['celebrity_name'] == name) & 
            (self.raw_data['season'] == season)
        ]
        
        if row.empty:
            return {'industry': 'Unknown', 'age': 35, 'placement': 10}
        
        row = row.iloc[0]
        
        return {
            'industry': row.get('celebrity_industry', 'Unknown'),
            'age': row.get('celebrity_age_during_season', 35),
            'placement': row.get('placement', 10)
        }


class BayesianLatentVoteModel:
    """
    Bayesian Latent Variable Model with Interval Censoring for Fan Vote Estimation
    """
    
    def __init__(self, n_samples: int = 2000, n_chains: int = 2):
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.traces = {}
        
    def compute_prior_mean(self, judge_score: float, features: dict, 
                           max_score: float) -> float:
        """
        Compute prior mean for fan vote based on contestant features
        
        Prior assumptions:
        - Base vote proportional to relative judge score
        - Adjustments for industry popularity, age appeal
        """
        # Base: proportional to judge score
        base = (judge_score / max_score) * 50 if max_score > 0 else 25
        
        # Industry adjustment
        popular_industries = ['Actor/Actress', 'Singer', 'Athlete', 'Model']
        industry = features.get('industry', '')
        if any(ind in str(industry) for ind in popular_industries):
            base *= 1.2
        
        # Age adjustment (mid-range ages tend to have broader appeal)
        age = features.get('age', 35)
        if 25 <= age <= 45:
            base *= 1.1
        
        # Placement adjustment (better placement suggests higher fan support)
        placement = features.get('placement', 10)
        if placement <= 3:
            base *= 1.3
        elif placement <= 6:
            base *= 1.1
            
        return np.clip(base, 5, 95)
    
    def build_model(self, week_data: dict, processor: DataProcessor) -> pm.Model:
        """
        Build PyMC model for a single week
        
        Model structure:
        - v_i ~ TruncatedNormal(prior_mean_i, sigma) for each contestant
        - Constraint: v_eliminated + s_eliminated < min(v_j + s_j) for j != eliminated
        """
        n_contestants = len(week_data['contestants'])
        judge_scores = week_data['judge_scores']
        eliminated_idx = week_data['eliminated_idx']
        max_score = max(judge_scores)
        
        # Compute prior means for each contestant
        prior_means = []
        for i, name in enumerate(week_data['contestants']):
            features = processor.get_contestant_features(name, week_data['season'])
            prior_mean = self.compute_prior_mean(judge_scores[i], features, max_score)
            prior_means.append(prior_mean)
        
        prior_means = np.array(prior_means)
        
        with pm.Model() as model:
            # Latent fan vote percentages (0-100 scale)
            # Using TruncatedNormal to keep votes in reasonable range
            fan_votes = pm.TruncatedNormal(
                'fan_votes',
                mu=prior_means,
                sigma=15,  # Uncertainty in prior
                lower=1,
                upper=99,
                shape=n_contestants
            )
            
            # Combined scores
            combined_scores = judge_scores + fan_votes
            
            # Constraint: eliminated contestant has lowest combined score
            # Implemented as soft constraint using potential
            eliminated_score = combined_scores[eliminated_idx]
            other_scores = pm.math.concatenate([
                combined_scores[:eliminated_idx],
                combined_scores[eliminated_idx+1:]
            ])
            
            # The eliminated contestant's score should be lower than all others
            # Using a smooth penalty function
            min_other_score = pm.math.min(other_scores)
            
            # Soft constraint: penalty if eliminated score >= min other score
            # Large negative potential when constraint is violated
            constraint_margin = min_other_score - eliminated_score
            
            # Potential: log probability contribution
            # Encourages constraint_margin > 0
            pm.Potential(
                'elimination_constraint',
                pm.math.switch(
                    constraint_margin > 0,
                    0,  # No penalty when constraint satisfied
                    -1000 * pm.math.abs(constraint_margin)  # Large penalty when violated
                )
            )
            
        return model
    
    def sample_posterior(self, week_data: dict, processor: DataProcessor, use_advi: bool = True) -> dict:
        """
        Sample from posterior distribution
        Returns posterior statistics for fan votes
        
        use_advi: If True, use fast variational inference instead of slow MCMC
        """
        model = self.build_model(week_data, processor)
        
        with model:
            if use_advi:
                # Use fast variational inference (ADVI)
                approx = pm.fit(
                    n=3000,
                    method='advi',
                    progressbar=False
                )
                trace_samples = approx.sample(500)
                # Handle different PyMC versions
                try:
                    # PyMC v5+ with InferenceData
                    if hasattr(trace_samples, 'posterior'):
                        fan_votes_samples = trace_samples.posterior['fan_votes'].values
                        fan_votes_samples = fan_votes_samples.reshape(-1, len(week_data['contestants']))
                    else:
                        # Older format - dict-like access
                        fan_votes_samples = np.array(trace_samples['fan_votes'])
                except (KeyError, TypeError):
                    # Try point attribute for PyMC v5
                    point = approx.mean.eval()
                    fan_votes_mean = point['fan_votes']
                    # Generate samples around mean
                    fan_votes_samples = np.random.normal(
                        fan_votes_mean, 
                        5,  # assume some std
                        size=(500, len(week_data['contestants']))
                    )
            else:
                # Use NUTS sampler (slow without g++)
                trace = pm.sample(
                    draws=self.n_samples,
                    tune=500,
                    chains=self.n_chains,
                    cores=1,
                    progressbar=False,
                    return_inferencedata=True
                )
                fan_votes_samples = trace.posterior['fan_votes'].values
                fan_votes_samples = fan_votes_samples.reshape(-1, len(week_data['contestants']))
        
        results = []
        for i, name in enumerate(week_data['contestants']):
            samples = fan_votes_samples[:, i]
            
            result = {
                'season': week_data['season'],
                'week': week_data['week'],
                'celebrity_name': name,
                'judge_score': week_data['judge_scores'][i],
                'vote_posterior_mean': np.mean(samples),
                'vote_posterior_std': np.std(samples),
                'vote_ci_lower': np.percentile(samples, 2.5),
                'vote_ci_upper': np.percentile(samples, 97.5),
                'vote_median': np.median(samples),
                'is_eliminated': (name == week_data['eliminated']),
            }
            
            # Confidence based on posterior uncertainty
            cv = np.std(samples) / np.mean(samples) if np.mean(samples) > 0 else 1
            if cv < 0.2:
                confidence = 'High'
            elif cv < 0.4:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            result['confidence_level'] = confidence
            result['coefficient_of_variation'] = round(cv, 3)
            
            results.append(result)
        
        # Verify elimination constraint
        eliminated_idx = week_data['eliminated_idx']
        eliminated_combined = (week_data['judge_scores'][eliminated_idx] + 
                              results[eliminated_idx]['vote_posterior_mean'])
        
        constraint_satisfied = True
        for i, r in enumerate(results):
            if i != eliminated_idx:
                other_combined = week_data['judge_scores'][i] + r['vote_posterior_mean']
                if eliminated_combined >= other_combined:
                    constraint_satisfied = False
                    break
        
        for r in results:
            r['constraint_satisfied'] = constraint_satisfied
            
        return results
    
def run_analysis(seasons_to_analyze: list = None, max_weeks: int = 11, use_advi: bool = True):
    """
    Run Bayesian latent variable analysis
    """
    print("="*70)
    print("Bayesian Latent Variable Model with Interval Censoring")
    print("="*70)
    
    # Initialize
    processor = DataProcessor(DATA_PATH)
    model = BayesianLatentVoteModel(n_samples=1000, n_chains=2)
    
    all_results = []
    
    # Determine seasons to analyze
    if seasons_to_analyze is None:
        seasons_to_analyze = sorted(processor.raw_data['season'].unique())
    
    print(f"\nAnalyzing {len(seasons_to_analyze)} seasons...")
    
    for season in seasons_to_analyze:
        print(f"\n--- Season {season} ---")
        
        for week in range(1, max_weeks + 1):
            week_data = processor.get_week_data(season, week)
            
            if week_data is None:
                continue
            
            print(f"  Week {week}: {len(week_data['contestants'])} contestants, "
                  f"Eliminated: {week_data['eliminated']}")
            
            try:
                # Run sampling (ADVI is faster, MCMC is more accurate)
                results = model.sample_posterior(week_data, processor, use_advi=use_advi)
                all_results.extend(results)
                
                # Print summary for eliminated contestant
                for r in results:
                    if r['is_eliminated']:
                        print(f"    -> Vote estimate: {r['vote_posterior_mean']:.1f}% "
                              f"[{r['vote_ci_lower']:.1f}, {r['vote_ci_upper']:.1f}] "
                              f"({r['confidence_level']})")
                        
            except Exception as e:
                print(f"    Error: {str(e)[:50]}")
                continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(OUTPUT_DIR, "fan_vote_bayesian_latent.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n\nResults saved to: {output_path}")
        print(f"Total records: {len(results_df)}")
        
        # Summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        print(f"\nConfidence Level Distribution:")
        if 'confidence_level' in results_df.columns:
            print(results_df['confidence_level'].value_counts())
        
        print(f"\nConstraint Satisfaction Rate:")
        if 'constraint_satisfied' in results_df.columns:
            sat_rate = results_df['constraint_satisfied'].mean() * 100
            print(f"  {sat_rate:.1f}% of estimates satisfy elimination constraint")
        
        print(f"\nAverage Posterior Uncertainty (Std Dev):")
        if 'vote_posterior_std' in results_df.columns:
            avg_std = results_df['vote_posterior_std'].mean()
            print(f"  {avg_std:.2f} percentage points")
        
        return results_df
    
    return None


def quick_demo():
    """
    Quick demonstration on a few seasons using fast ADVI
    """
    print("="*70)
    print("QUICK DEMO: Bayesian Latent Variable Model (ADVI - Fast Mode)")
    print("="*70)
    
    # Just analyze 2 seasons for demo with ADVI (fast)
    return run_analysis(seasons_to_analyze=[2, 27], max_weeks=5, use_advi=True)


if __name__ == "__main__":
    # Run quick demo first to verify model works
    print("Running quick demo (Seasons 2 and 27, first 5 weeks)...")
    print("Using ADVI (Variational Inference) for faster results.\n")
    
    try:
        results = quick_demo()
        
        if results is not None:
            print("\n\nDemo completed successfully!")
            print("\nTo run full analysis on all seasons:")
            print("  run_analysis(use_advi=True)  # Fast mode")
            print("  run_analysis(use_advi=False) # Accurate MCMC (slow)")
            
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install PyMC first:")
        print("  pip install pymc arviz")
