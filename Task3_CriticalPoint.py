# -*- coding: utf-8 -*-
"""
================================================================================
Task 3: Critical Point Analysis Model for Controversial Contestants
================================================================================

Mathematical Model:
-------------------
For each week, define:
- J_i = Judge score for contestant i
- F_i = Fan votes for contestant i  
- n = Number of contestants

Percentage System Score:
    S_i^pct = 0.5 * (J_i / max(J)) + 0.5 * (F_i / max(F))
    
Ranking System Score:
    S_i^rank = R_J(i) + R_F(i)  (lower is better)
    
Critical Point Definition:
    F_c^critical = minimum fan votes needed for contestant c to NOT be eliminated
    
Safety Margin:
    M_c = S_c - S_min  (positive = safe, negative = eliminated)
    
Sensitivity:
    ΔF_c = F_c^critical - F_c^current
================================================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = r"d:\Data\xwechat_files\wxid_m7pucc5xg5m522_f378\msg\file\2026-01\中文版赛题 (1)\中文版赛题\2026_MCM_Problem_C_Data.csv"
FAN_VOTES_PATH = r"d:\桌面\代码仓库\数据结构\data-structure-learning\MCM_2026_C\Task1_Results\fan_vote_enhanced.csv"
OUTPUT_DIR = r"d:\桌面\代码仓库\数据结构\data-structure-learning\MCM_2026_C\Task3_Results"

CONTROVERSIAL_CASES = [
    {'name': 'Jerry Rice', 'season': 2, 'actual_placement': 2},
    {'name': 'Billy Ray Cyrus', 'season': 4, 'actual_placement': 5},
    {'name': 'Bristol Palin', 'season': 11, 'actual_placement': 3},
    {'name': 'Bobby Bones', 'season': 27, 'actual_placement': 1}
]


class DataLoader:
    """Load competition data and Task1 fan vote estimates"""
    
    def __init__(self, data_path: str, fan_votes_path: str):
        self.raw_data = pd.read_csv(data_path).replace('N/A', np.nan)
        self.weekly_data = self._process_to_weekly()
        
        # Load Task1 fan vote estimates
        try:
            self.fan_votes_df = pd.read_csv(fan_votes_path)
            print(f"  Loaded {len(self.fan_votes_df)} fan vote estimates from Task1")
        except:
            self.fan_votes_df = None
            print("  Warning: Could not load Task1 fan votes, will use simulation")
    
    def _process_to_weekly(self) -> pd.DataFrame:
        """Convert contestant-level data to week-level"""
        records = []
        
        for _, row in self.raw_data.iterrows():
            season = row['season']
            celebrity = row['celebrity_name']
            placement = row['placement']
            
            for week in range(1, 12):
                scores = []
                for j in range(1, 5):
                    col = f'week{week}_judge{j}_score'
                    if col in row.index and pd.notna(row[col]):
                        try:
                            s = float(row[col])
                            if s > 0:
                                scores.append(s)
                        except:
                            pass
                
                if scores:
                    records.append({
                        'season': season,
                        'week': week,
                        'celebrity': celebrity,
                        'judge_total': sum(scores),
                        'placement': placement
                    })
        
        return pd.DataFrame(records)
    
    def get_week_data(self, season: int, week: int) -> pd.DataFrame:
        """Get all contestants for a specific week"""
        return self.weekly_data[
            (self.weekly_data['season'] == season) & 
            (self.weekly_data['week'] == week)
        ].copy()
    
    def get_fan_votes(self, season: int, week: int, celebrity: str) -> float:
        """Get estimated fan votes from Task1"""
        if self.fan_votes_df is None:
            return None
        
        # Try to match celebrity name
        mask = (self.fan_votes_df['season'] == season) & \
               (self.fan_votes_df['week'] == week)
        
        week_votes = self.fan_votes_df[mask]
        
        for _, row in week_votes.iterrows():
            name_col = 'celebrity_name' if 'celebrity_name' in row.index else 'celebrity'
            if celebrity.split()[0].lower() in str(row.get(name_col, '')).lower():
                return row.get('estimated_votes', row.get('prior_votes', None))
        
        return None


class CriticalPointModel:
    """
    Mathematical model for critical point analysis
    
    Key equations:
    1. Percentage System: S_pct = 0.5 * J_norm + 0.5 * F_norm
    2. Safety Margin: M = S_contestant - S_eliminated
    3. Critical Fan Votes: F_critical such that M = 0
    """
    
    @staticmethod
    def compute_percentage_score(judge_scores: dict, fan_votes: dict) -> dict:
        """
        S_i^pct = 0.5 * (J_i / max(J)) * 100 + 0.5 * (F_i / max(F)) * 100
        """
        max_J = max(judge_scores.values()) if judge_scores else 1
        max_F = max(fan_votes.values()) if fan_votes else 1
        
        scores = {}
        for celeb in judge_scores:
            J_norm = (judge_scores[celeb] / max_J) * 100
            F_norm = (fan_votes.get(celeb, 0) / max_F) * 100
            scores[celeb] = 0.5 * J_norm + 0.5 * F_norm
        
        return scores
    
    @staticmethod
    def compute_ranking_score(judge_scores: dict, fan_votes: dict) -> dict:
        """
        S_i^rank = R_J(i) + R_F(i)  (lower is better)
        """
        # Judge ranking (higher score = lower rank number)
        judge_sorted = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
        judge_ranks = {c: i+1 for i, (c, _) in enumerate(judge_sorted)}
        
        # Fan ranking
        fan_sorted = sorted(fan_votes.items(), key=lambda x: x[1], reverse=True)
        fan_ranks = {c: i+1 for i, (c, _) in enumerate(fan_sorted)}
        
        scores = {}
        for celeb in judge_scores:
            scores[celeb] = judge_ranks.get(celeb, 99) + fan_ranks.get(celeb, 99)
        
        return scores
    
    @staticmethod
    def compute_safety_margin(scores: dict, contestant: str, is_ranking: bool = False) -> dict:
        """
        Compute safety margin: M = S_contestant - S_eliminated
        
        For percentage: positive margin = safe
        For ranking: negative margin = safe (lower rank is better)
        """
        if contestant not in scores:
            return None
        
        S_c = scores[contestant]
        
        if is_ranking:
            # Ranking: highest combined rank gets eliminated
            S_elim = max(scores.values())
            # Margin: how far from being eliminated (positive = safe)
            margin = S_elim - S_c
            margin_pct = (margin / S_elim) * 100 if S_elim > 0 else 0
        else:
            # Percentage: lowest score gets eliminated
            S_elim = min(scores.values())
            # Margin: how far from being eliminated (positive = safe)
            margin = S_c - S_elim
            margin_pct = (margin / S_c) * 100 if S_c > 0 else 0
        
        return {
            'contestant_score': S_c,
            'elimination_threshold': S_elim,
            'margin': margin,
            'margin_percent': margin_pct,
            'is_safe': margin > 0,
            'would_be_eliminated': S_c == S_elim if not is_ranking else S_c == S_elim
        }
    
    @staticmethod
    def compute_bottom_two_and_judge_tiebreaker(pct_scores: dict, judge_scores: dict, 
                                                 contestant: str) -> dict:
        """
        Bottom-two rule: among the two lowest combined-score contestants,
        judges eliminate the one with LOWER judge score.
        
        Returns: in_bottom2, would_eliminated_by_tiebreaker, would_saved_by_tiebreaker
        """
        if len(pct_scores) < 2:
            return {'in_bottom2': False, 'would_eliminated_by_tiebreaker': False, 
                    'would_saved_by_tiebreaker': False, 'tiebreaker_eliminated': None}
        
        # Bottom 2 by combined (percentage) score
        sorted_by_pct = sorted(pct_scores.items(), key=lambda x: x[1])
        bottom2_names = [sorted_by_pct[0][0], sorted_by_pct[1][0]]
        
        in_bottom2 = contestant in bottom2_names
        
        if not in_bottom2:
            j1, j2 = judge_scores.get(bottom2_names[0], 0), judge_scores.get(bottom2_names[1], 0)
            tiebreaker_eliminated = bottom2_names[0] if j1 <= j2 else bottom2_names[1]
            return {'in_bottom2': False, 'would_eliminated_by_tiebreaker': False,
                    'would_saved_by_tiebreaker': False, 'tiebreaker_eliminated': tiebreaker_eliminated}
        
        # Among bottom 2, the one with LOWER judge score is eliminated by judges
        j1, j2 = judge_scores.get(bottom2_names[0], 0), judge_scores.get(bottom2_names[1], 0)
        tiebreaker_eliminated = bottom2_names[0] if j1 <= j2 else bottom2_names[1]
        
        would_eliminated_by_tiebreaker = (tiebreaker_eliminated == contestant)
        
        # Saved by tiebreaker: normal rule would eliminate contestant, but tiebreaker eliminates the other
        normal_eliminated = sorted_by_pct[0][0]  # lowest combined score
        would_saved_by_tiebreaker = (normal_eliminated == contestant) and (tiebreaker_eliminated != contestant)
        
        return {
            'in_bottom2': True,
            'bottom2_contestants': bottom2_names,
            'tiebreaker_eliminated': tiebreaker_eliminated,
            'would_eliminated_by_tiebreaker': would_eliminated_by_tiebreaker,
            'would_saved_by_tiebreaker': would_saved_by_tiebreaker,
            'contestant_judge_in_bottom2': judge_scores.get(contestant),
            'other_judge_in_bottom2': j2 if contestant == bottom2_names[0] else j1
        }
    
    @staticmethod
    def compute_critical_fan_votes(judge_scores: dict, fan_votes: dict, 
                                    contestant: str, target_contestant: str = None) -> dict:
        """
        Compute the critical fan vote change needed to change elimination outcome.
        
        For percentage system:
            S_c = 0.5 * (J_c/max_J) + 0.5 * (F_c/max_F)
            
        We solve for F_c such that S_c > S_min (survive) or S_c < S_next (get eliminated)
        """
        if contestant not in judge_scores:
            return None
        
        max_J = max(judge_scores.values())
        max_F = max(fan_votes.values())
        current_F = fan_votes.get(contestant, 0)
        J_c = judge_scores[contestant]
        
        # Current scores
        current_scores = CriticalPointModel.compute_percentage_score(judge_scores, fan_votes)
        current_S_c = current_scores[contestant]
        
        # Find the contestant with next lowest score (the one we need to beat)
        sorted_scores = sorted(current_scores.items(), key=lambda x: x[1])
        
        # If contestant is currently eliminated (lowest score)
        if current_S_c == min(current_scores.values()):
            # Need to find F_c such that S_c > S_next
            next_lowest = sorted_scores[1][1] if len(sorted_scores) > 1 else current_S_c
            
            # S_c = 0.5 * (J_c/max_J)*100 + 0.5 * (F_c/max_F)*100 > S_next
            # 0.5 * (F_c/max_F)*100 > S_next - 0.5 * (J_c/max_J)*100
            # F_c > max_F * (2*S_next/100 - J_c/max_J)
            
            J_term = (J_c / max_J) * 100 * 0.5
            F_critical = max_F * (2 * next_lowest / 100 - J_c / max_J) + 1  # +1 for margin
            
            delta_F = F_critical - current_F
            delta_F_pct = (delta_F / current_F * 100) if current_F > 0 else float('inf')
            
            return {
                'current_fan_votes': current_F,
                'critical_fan_votes': max(0, F_critical),
                'delta_votes': delta_F,
                'delta_percent': delta_F_pct,
                'action': 'INCREASE_TO_SURVIVE',
                'target_score': next_lowest
            }
        else:
            # Currently safe, find how much votes can decrease before elimination
            lowest_score = sorted_scores[0][1]
            
            # S_c = 0.5 * (J_c/max_J)*100 + 0.5 * (F_c/max_F)*100 = S_lowest
            # F_c = max_F * (2*S_lowest/100 - J_c/max_J)
            
            F_critical = max_F * (2 * lowest_score / 100 - J_c / max_J)
            
            delta_F = F_critical - current_F
            delta_F_pct = (delta_F / current_F * 100) if current_F > 0 else 0
            
            return {
                'current_fan_votes': current_F,
                'critical_fan_votes': max(0, F_critical),
                'delta_votes': delta_F,
                'delta_percent': delta_F_pct,
                'action': 'DECREASE_BEFORE_ELIMINATED',
                'buffer_votes': -delta_F,
                'buffer_percent': -delta_F_pct
            }


class ControversyAnalyzer:
    """Analyze controversial contestants using critical point model"""
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.model = CriticalPointModel()
        self.results = []
    
    def _get_fan_votes_for_week(self, season: int, week: int, 
                                 judge_scores: dict, controversial_name: str) -> dict:
        """Get fan votes from Task1 or simulate"""
        fan_votes = {}
        
        for celeb in judge_scores:
            # Try to get from Task1
            fv = self.loader.get_fan_votes(season, week, celeb)
            
            if fv is not None:
                fan_votes[celeb] = fv
            else:
                # Simulate based on judge score with boost for controversial contestant
                base = 1000000 * (judge_scores[celeb] / 40)
                if controversial_name.split()[0].lower() in celeb.lower():
                    base *= 1.5  # Fan favorite boost
                fan_votes[celeb] = base * np.random.uniform(0.9, 1.1)
        
        return fan_votes
    
    def analyze_contestant(self, case: dict) -> dict:
        """Complete critical point analysis for a contestant"""
        name = case['name']
        season = case['season']
        
        print(f"\n{'='*75}")
        print(f"CRITICAL POINT ANALYSIS: {name} (Season {season})")
        print(f"{'='*75}")
        
        # Get contestant's weeks
        contestant_data = self.loader.weekly_data[
            (self.loader.weekly_data['season'] == season) &
            (self.loader.weekly_data['celebrity'].str.contains(name.split()[0], case=False, na=False))
        ]
        
        if contestant_data.empty:
            print(f"  ERROR: No data found for {name}")
            return None
        
        weeks = sorted(contestant_data['week'].unique())
        contestant_celeb = contestant_data.iloc[0]['celebrity']
        
        weekly_analysis = []
        critical_weeks = []  # Weeks where margin is small
        
        for week in weeks:
            week_data = self.loader.get_week_data(season, week)
            if len(week_data) < 2:
                continue
            
            judge_scores = dict(zip(week_data['celebrity'], week_data['judge_total']))
            fan_votes = self._get_fan_votes_for_week(season, week, judge_scores, name)
            
            # Find contestant in this week's data
            celeb_match = [c for c in judge_scores if name.split()[0].lower() in c.lower()]
            if not celeb_match:
                continue
            celeb = celeb_match[0]
            
            # Compute scores under both systems
            pct_scores = self.model.compute_percentage_score(judge_scores, fan_votes)
            rank_scores = self.model.compute_ranking_score(judge_scores, fan_votes)
            
            # Compute safety margins
            pct_margin = self.model.compute_safety_margin(pct_scores, celeb, is_ranking=False)
            rank_margin = self.model.compute_safety_margin(rank_scores, celeb, is_ranking=True)
            
            # Compute critical fan votes
            critical_analysis = self.model.compute_critical_fan_votes(judge_scores, fan_votes, celeb)
            
            # Bottom-two rule: judges choose from bottom 2 to eliminate one (lower judge score)
            tiebreaker = self.model.compute_bottom_two_and_judge_tiebreaker(pct_scores, judge_scores, celeb)
            
            # Judge rank
            sorted_judges = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
            judge_rank = next((i+1 for i, (c,_) in enumerate(sorted_judges) if c == celeb), None)
            
            # Is this a critical week?
            is_critical = pct_margin and pct_margin['margin_percent'] < 10
            if is_critical:
                critical_weeks.append(int(week))
            
            week_result = {
                'week': week,
                'n_contestants': len(judge_scores),
                'judge_score': judge_scores[celeb],
                'judge_rank': judge_rank,
                'is_lowest_judge': judge_rank == len(judge_scores),
                'fan_votes': fan_votes[celeb],
                
                # Percentage system
                'pct_score': pct_margin['contestant_score'] if pct_margin else None,
                'pct_elim_threshold': pct_margin['elimination_threshold'] if pct_margin else None,
                'pct_margin': pct_margin['margin'] if pct_margin else None,
                'pct_margin_percent': pct_margin['margin_percent'] if pct_margin else None,
                'pct_would_eliminate': pct_margin['would_be_eliminated'] if pct_margin else None,
                
                # Ranking system
                'rank_score': rank_margin['contestant_score'] if rank_margin else None,
                'rank_margin': rank_margin['margin'] if rank_margin else None,
                'rank_would_eliminate': rank_margin['would_be_eliminated'] if rank_margin else None,
                
                # Bottom-two & judge tiebreaker
                'in_bottom2': tiebreaker['in_bottom2'],
                'would_eliminated_by_tiebreaker': tiebreaker['would_eliminated_by_tiebreaker'],
                'would_saved_by_tiebreaker': tiebreaker['would_saved_by_tiebreaker'],
                
                # Critical point analysis
                'critical_action': critical_analysis['action'] if critical_analysis else None,
                'critical_delta_votes': critical_analysis['delta_votes'] if critical_analysis else None,
                'critical_delta_pct': critical_analysis['delta_percent'] if critical_analysis else None,
                
                'is_critical_week': is_critical
            }
            
            weekly_analysis.append(week_result)
            
            # Print weekly summary
            status = "SAFE" if pct_margin and pct_margin['margin'] > 0 else "DANGER"
            margin_str = f"{pct_margin['margin_percent']:.1f}%" if pct_margin else "N/A"
            
            print(f"\n  Week {week}: {len(judge_scores)} contestants")
            print(f"    Judge: {judge_scores[celeb]:.0f} (rank {judge_rank}/{len(judge_scores)})"
                  f" {'*LOWEST*' if judge_rank == len(judge_scores) else ''}")
            print(f"    Pct Score: {pct_margin['contestant_score']:.2f} | "
                  f"Elim Threshold: {pct_margin['elimination_threshold']:.2f}")
            print(f"    Safety Margin: {margin_str} [{status}]")
            
            if tiebreaker['in_bottom2']:
                print(f"    [BOTTOM 2] Judge tiebreaker: "
                      f"{'WOULD BE ELIMINATED (lower judge in bottom 2)' if tiebreaker['would_eliminated_by_tiebreaker'] else 'SAVED (higher judge in bottom 2)'}")
            
            if critical_analysis:
                if 'buffer_percent' in critical_analysis:
                    print(f"    Buffer: Can lose {critical_analysis['buffer_percent']:.1f}% fan votes before danger")
                else:
                    print(f"    Need: {critical_analysis['delta_percent']:+.1f}% fan votes to survive")
        
        # Summary statistics (including bottom-two / judge tiebreaker)
        result = {
            'contestant': name,
            'season': season,
            'actual_placement': case['actual_placement'],
            'weeks_participated': len(weekly_analysis),
            'times_lowest_judge': sum(1 for w in weekly_analysis if w['is_lowest_judge']),
            'times_would_elim_pct': sum(1 for w in weekly_analysis if w['pct_would_eliminate']),
            'times_would_elim_rank': sum(1 for w in weekly_analysis if w['rank_would_eliminate']),
            'times_in_bottom2': sum(1 for w in weekly_analysis if w['in_bottom2']),
            'times_eliminated_by_tiebreaker': sum(1 for w in weekly_analysis if w['would_eliminated_by_tiebreaker']),
            'times_saved_by_tiebreaker': sum(1 for w in weekly_analysis if w['would_saved_by_tiebreaker']),
            'critical_weeks': critical_weeks,
            'num_critical_weeks': len(critical_weeks),
            'min_safety_margin': min((w['pct_margin_percent'] for w in weekly_analysis if w['pct_margin_percent']), default=None),
            'avg_safety_margin': np.mean([w['pct_margin_percent'] for w in weekly_analysis if w['pct_margin_percent']]),
            'weekly_analysis': weekly_analysis
        }
        
        # Print summary
        print(f"\n  {'='*60}")
        print(f"  SUMMARY FOR {name}")
        print(f"  {'='*60}")
        print(f"  Weeks Participated: {result['weeks_participated']}")
        print(f"  Times with Lowest Judge Score: {result['times_lowest_judge']}")
        print(f"  Times Would Be Eliminated (Pct): {result['times_would_elim_pct']}")
        print(f"  Times Would Be Eliminated (Rank): {result['times_would_elim_rank']}")
        print(f"  Times in Bottom 2: {result['times_in_bottom2']}")
        print(f"  Judge Tiebreaker: {result['times_saved_by_tiebreaker']} saves, {result['times_eliminated_by_tiebreaker']} eliminations")
        print(f"  Critical Weeks (margin < 10%): {critical_weeks}")
        print(f"  Minimum Safety Margin: {result['min_safety_margin']:.1f}%")
        print(f"  Average Safety Margin: {result['avg_safety_margin']:.1f}%")
        
        self.results.append(result)
        return result
    
    def compare_mechanisms(self) -> pd.DataFrame:
        """Compare how different mechanisms would affect outcomes"""
        comparison = []
        
        for r in self.results:
            if r is None:
                continue
            
            first_elim_pct = next((w['week'] for w in r['weekly_analysis'] 
                                   if w['pct_would_eliminate']), None)
            first_elim_rank = next((w['week'] for w in r['weekly_analysis'] 
                                    if w['rank_would_eliminate']), None)
            
            comparison.append({
                'contestant': r['contestant'],
                'season': r['season'],
                'actual_placement': r['actual_placement'],
                'first_elim_week_pct': first_elim_pct,
                'first_elim_week_rank': first_elim_rank,
                'same_outcome': first_elim_pct == first_elim_rank,
                'times_in_bottom2': r['times_in_bottom2'],
                'judge_tiebreaker_saves': r['times_saved_by_tiebreaker'],
                'judge_tiebreaker_eliminations': r['times_eliminated_by_tiebreaker'],
                'judge_tiebreaker_net_effect': r['times_saved_by_tiebreaker'] - r['times_eliminated_by_tiebreaker'],
                'min_safety_margin_pct': r['min_safety_margin'],
                'avg_safety_margin_pct': r['avg_safety_margin'],
                'num_critical_weeks': r['num_critical_weeks'],
                'critical_weeks': ','.join(map(str, r['critical_weeks']))
            })
        
        return pd.DataFrame(comparison)
    
    def sensitivity_summary(self) -> pd.DataFrame:
        """Generate sensitivity analysis summary"""
        sensitivity = []
        
        for r in self.results:
            if r is None:
                continue
            
            # Find the most critical week (lowest margin)
            weekly = r['weekly_analysis']
            if not weekly:
                continue
            
            most_critical = min(weekly, key=lambda x: x['pct_margin_percent'] or float('inf'))
            
            # Calculate average vote buffer
            buffers = [w['critical_delta_pct'] for w in weekly 
                       if w['critical_delta_pct'] and w['critical_delta_pct'] < 0]
            avg_buffer = np.mean(buffers) if buffers else None
            
            # Calculate average vote needed when in danger
            needs = [w['critical_delta_pct'] for w in weekly 
                     if w['critical_delta_pct'] and w['critical_delta_pct'] > 0]
            avg_need = np.mean(needs) if needs else None
            
            sensitivity.append({
                'contestant': r['contestant'],
                'season': r['season'],
                'most_critical_week': most_critical['week'],
                'most_critical_margin_pct': most_critical['pct_margin_percent'],
                'avg_vote_buffer_pct': avg_buffer,
                'avg_vote_need_when_danger_pct': avg_need,
                'interpretation': self._interpret_sensitivity(r)
            })
        
        return pd.DataFrame(sensitivity)
    
    def _interpret_sensitivity(self, result: dict) -> str:
        """Generate natural language interpretation"""
        name = result['contestant']
        margin = result['avg_safety_margin']
        critical = result['num_critical_weeks']
        
        if margin > 20:
            return f"{name} was generally safe with high fan support (avg margin {margin:.1f}%)"
        elif margin > 10:
            return f"{name} had moderate safety, some close calls ({critical} critical weeks)"
        else:
            return f"{name} survived on thin margins, highly dependent on fan votes"


def main():
    """Main analysis with critical point model"""
    print("="*75)
    print("TASK 3: CRITICAL POINT ANALYSIS MODEL")
    print("="*75)
    print("\nMathematical Framework:")
    print("  S_pct = 0.5 * (J/max_J) + 0.5 * (F/max_F)")
    print("  Safety Margin M = S_contestant - S_elimination_threshold")
    print("  Critical F = votes needed to change outcome")
    
    # Load data
    print("\n[1] Loading data...")
    loader = DataLoader(DATA_PATH, FAN_VOTES_PATH)
    print(f"  Weekly records: {len(loader.weekly_data)}")
    
    # Analyze each case
    print("\n[2] Analyzing controversial contestants...")
    analyzer = ControversyAnalyzer(loader)
    
    for case in CONTROVERSIAL_CASES:
        analyzer.analyze_contestant(case)
    
    # Generate reports
    print("\n[3] Generating reports...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Comparison report
    comparison_df = analyzer.compare_mechanisms()
    comparison_path = os.path.join(OUTPUT_DIR, "critical_point_comparison_v2.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  Comparison: {comparison_path}")
    
    # Sensitivity report
    sensitivity_df = analyzer.sensitivity_summary()
    sensitivity_path = os.path.join(OUTPUT_DIR, "critical_point_sensitivity_v2.csv")
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"  Sensitivity: {sensitivity_path}")
    
    # Detailed weekly report
    weekly_data = []
    for r in analyzer.results:
        if r is None:
            continue
        for w in r['weekly_analysis']:
            w['contestant'] = r['contestant']
            w['season'] = r['season']
            weekly_data.append(w)
    
    weekly_df = pd.DataFrame(weekly_data)
    weekly_path = os.path.join(OUTPUT_DIR, "critical_point_weekly_v2.csv")
    weekly_df.to_csv(weekly_path, index=False)
    print(f"  Weekly: {weekly_path}")
    
    # Final conclusions
    print("\n" + "="*75)
    print("CONCLUSIONS FROM CRITICAL POINT MODEL")
    print("="*75)
    
    print("\n[Q1] Would different voting methods lead to different outcomes?")
    print("-"*70)
    for _, row in comparison_df.iterrows():
        print(f"\n{row['contestant']}:")
        print(f"  Pct System: First eliminated Week {row['first_elim_week_pct'] or 'Never'}")
        print(f"  Rank System: First eliminated Week {row['first_elim_week_rank'] or 'Never'}")
        print(f"  Same outcome: {'YES' if row['same_outcome'] else 'NO'}")
    
    print("\n\n[Q2] Impact of 'Judges choose from bottom 2 to eliminate one' rule")
    print("-"*70)
    for _, row in comparison_df.iterrows():
        b2 = row['times_in_bottom2']
        saves = row['judge_tiebreaker_saves']
        elims = row['judge_tiebreaker_eliminations']
        net = row['judge_tiebreaker_net_effect']
        print(f"\n{row['contestant']}:")
        print(f"  Times in bottom 2: {b2}")
        print(f"  Times SAVED by tiebreaker (higher judge in bottom 2): {saves}")
        print(f"  Times ELIMINATED by tiebreaker (lower judge in bottom 2): {elims}")
        print(f"  Net effect: {'BENEFICIAL' if net > 0 else ('DETRIMENTAL' if net < 0 else 'NEUTRAL')}")
    
    print("\n\n[Q3] How sensitive are outcomes to fan vote changes?")
    print("-"*70)
    for _, row in sensitivity_df.iterrows():
        print(f"\n{row['contestant']}:")
        print(f"  Most critical week: Week {row['most_critical_week']} "
              f"(margin only {row['most_critical_margin_pct']:.1f}%)")
        if row['avg_vote_buffer_pct']:
            print(f"  When safe: Could lose {-row['avg_vote_buffer_pct']:.1f}% votes on average")
        if row['avg_vote_need_when_danger_pct']:
            print(f"  When in danger: Needed +{row['avg_vote_need_when_danger_pct']:.1f}% more votes")
        print(f"  Interpretation: {row['interpretation']}")
    
    print("\n" + "="*75)
    print("Analysis Complete!")
    print("="*75)
    
    return comparison_df, sensitivity_df, weekly_df


if __name__ == "__main__":
    main()
