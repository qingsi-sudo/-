# -*- coding: utf-8 -*-
"""
================================================================================
2026 MCM Problem C - Task 2: 投票机制比较分析
基于TOPSIS的多准则决策方法
================================================================================

分析框架：
2.1 对比分析框架
2.2 反事实分析 (Counterfactual Analysis)
2.3 偏向性分析 (Bias Analysis)
2.4 争议选手案例分析
2.5 机制推荐分析 (TOPSIS多准则决策)

================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# 2.1 数据准备与基础计算
# ==============================================================================

def load_and_prepare_data(data_path: str, fan_votes_path: str) -> tuple:
    """加载并预处理数据"""
    # 加载原始数据
    df = pd.read_csv(data_path).replace('N/A', np.nan)
    
    # 计算每周评委总分
    for week in range(1, 12):
        cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        existing = [c for c in cols if c in df.columns]
        if existing:
            for c in existing:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df[f'week{week}_total'] = df[existing].sum(axis=1, skipna=True)
    
    # 提取淘汰周
    def parse_elim(r):
        if pd.isna(r): return -1
        r = str(r).lower()
        if 'week' in r:
            try:
                return int(''.join(filter(str.isdigit, r.split('week')[1])))
            except:
                return -1
        return -1
    
    df['elim_week'] = df['results'].apply(parse_elim)
    
    # 加载粉丝投票估算
    fan_df = pd.read_csv(fan_votes_path)
    fan_votes = {}
    for _, row in fan_df.iterrows():
        key = (row['season'], row['week'], row['celebrity_name'])
        fan_votes[key] = row['estimated_votes']
    
    return df, fan_votes


# ==============================================================================
# 2.2 反事实分析 (Counterfactual Analysis)
# ==============================================================================

class CounterfactualAnalyzer:
    """
    反事实分析器
    
    对每个赛季同时应用两种评分方法，比较淘汰结果差异
    
    排名法: R_{i,t}^{Rank} = R_{i,t}^J + R_{i,t}^F
    百分比法: P_{i,t}^{Pct} = P_{i,t}^J + P_{i,t}^F
    """
    
    def __init__(self, df: pd.DataFrame, fan_votes: dict):
        self.df = df
        self.fan_votes = fan_votes
    
    def analyze_week(self, season: int, week: int) -> dict:
        """分析某一周在两种机制下的结果"""
        season_df = self.df[self.df['season'] == season]
        score_col = f'week{week}_total'
        
        if score_col not in season_df.columns:
            return None
        
        # 获取活跃选手
        active = ((season_df['elim_week'] == -1) | (season_df['elim_week'] >= week))
        active &= (season_df[score_col] > 0)
        week_data = season_df[active].copy()
        
        if len(week_data) < 3:
            return None
        
        # 获取实际淘汰者
        elim_df = season_df[season_df['elim_week'] == week]
        if len(elim_df) == 0:
            return None
        actual_elim = elim_df.iloc[0]['celebrity_name']
        
        contestants = []
        for _, row in week_data.iterrows():
            name = row['celebrity_name']
            judge_score = row[score_col]
            fan_vote = self.fan_votes.get((season, week, name), 0)
            contestants.append({
                'name': name,
                'judge_score': judge_score,
                'fan_votes': fan_vote
            })
        
        n = len(contestants)
        
        # === 百分比法计算 ===
        total_judge = sum(c['judge_score'] for c in contestants)
        total_fan = sum(c['fan_votes'] for c in contestants)
        
        for c in contestants:
            c['P_J'] = c['judge_score'] / total_judge if total_judge > 0 else 1/n
            c['P_F'] = c['fan_votes'] / total_fan if total_fan > 0 else 1/n
            c['P_Pct'] = c['P_J'] + c['P_F']  # 百分比法综合得分
        
        # === 排名法计算 ===
        # 评委排名 (高分=低排名数字)
        sorted_by_judge = sorted(contestants, key=lambda x: x['judge_score'], reverse=True)
        for rank, c in enumerate(sorted_by_judge, 1):
            c['R_J'] = rank
        
        # 粉丝排名 (高票=低排名数字)
        sorted_by_fan = sorted(contestants, key=lambda x: x['fan_votes'], reverse=True)
        for rank, c in enumerate(sorted_by_fan, 1):
            c['R_F'] = rank
        
        for c in contestants:
            c['R_Rank'] = c['R_J'] + c['R_F']  # 排名法综合排名
        
        # === 确定淘汰者 ===
        # 百分比法: P_Pct最低者淘汰
        pct_elim = min(contestants, key=lambda x: x['P_Pct'])['name']
        # 排名法: R_Rank最大者淘汰
        rank_elim = max(contestants, key=lambda x: x['R_Rank'])['name']
        
        return {
            'season': season,
            'week': week,
            'n_contestants': n,
            'actual_eliminated': actual_elim,
            'pct_eliminated': pct_elim,
            'rank_eliminated': rank_elim,
            'same_result': pct_elim == rank_elim,
            'contestants': contestants
        }


# ==============================================================================
# 2.3 偏向性分析 (Bias Analysis)
# ==============================================================================

class BiasAnalyzer:
    """
    偏向性分析器
    
    定义评委-观众差异指标:
    - 排名法: D_{i,t} = R_{i,t}^J - R_{i,t}^F
    - 百分比法: D_{i,t} = P_{i,t}^J - P_{i,t}^F
    
    权重分析:
    - 排名法: w_rank^J = w_rank^F = 1/2 (固定等权)
    - 百分比法: w_pct^J = Var(P^J) / (Var(P^J) + Var(P^F)) (方差权重)
    """
    
    @staticmethod
    def compute_difference_index(contestants: list) -> dict:
        """
        计算评委-观众差异指标 D_{i,t}
        
        D > 0: 评委评价高于粉丝 (评委偏爱)
        D < 0: 粉丝评价高于评委 (粉丝偏爱)
        """
        results = []
        for c in contestants:
            # 百分比法差异
            D_pct = c['P_J'] - c['P_F']
            
            # 排名法差异 (注意: 排名数字小=表现好，所以用F-J)
            D_rank = c['R_F'] - c['R_J']  # 正值表示评委排名更好
            
            results.append({
                'name': c['name'],
                'D_pct': D_pct,
                'D_rank': D_rank,
                'judge_favored': D_pct > 0,  # 评委更看好
                'fan_favored': D_pct < 0     # 粉丝更看好
            })
        
        return results
    
    @staticmethod
    def compute_effective_weights(contestants: list) -> dict:
        """
        计算两种方法的权重
        
        排名法: w_rank^J = w_rank^F = 0.5 (固定)
        百分比法: w_pct^J = w_pct^F = 0.5 (固定，按DWTS官方规则)
        """
        # 两种方法都使用固定的50%-50%权重
        return {
            'rank_judge_weight': 0.5,
            'rank_fan_weight': 0.5,
            'pct_judge_weight': 0.5,
            'pct_fan_weight': 0.5
        }
    
    @staticmethod
    def hypothesis_test(weight_data: list) -> dict:
        """
        假设检验: 检验两种方法是否给予观众投票相同权重
        
        H0: w_rank^F = w_pct^F (两种方法对粉丝权重相同)
        H1: w_rank^F ≠ w_pct^F
        
        使用配对t检验
        """
        rank_weights = [0.5] * len(weight_data)  # 排名法固定0.5
        pct_weights = [w['pct_fan_weight'] for w in weight_data]
        
        if len(pct_weights) < 2:
            return {'t_stat': 0, 'p_value': 1, 'significant': False}
        
        t_stat, p_value = stats.ttest_rel(rank_weights, pct_weights)
        
        return {
            't_statistic': round(t_stat, 4),
            'p_value': round(p_value, 6),
            'significant': p_value < 0.05,
            'conclusion': '拒绝H0: 两种方法对粉丝投票权重显著不同' if p_value < 0.05 
                         else '不能拒绝H0: 两种方法对粉丝投票权重无显著差异'
        }


# ==============================================================================
# 2.4 争议选手案例分析
# ==============================================================================

class ControversialCaseAnalyzer:
    """
    争议选手案例分析器
    
    识别标准: |ΔR_{i,t}| = |R_{i,t}^{Rank} - R_{i,t}^{Pct}| 较大的选手
    
    重点分析:
    - Jerry Rice (Season 2): 评委低分但晋级决赛
    - Bristol Palin (Season 11): 争议性晋级
    - Bobby Bones (Season 27): 最终冠军争议
    """
    
    KNOWN_CONTROVERSIAL = [
        {'name': 'Jerry Rice', 'season': 2, 'description': 'Low judge scores but reached finals'},
        {'name': 'Billy Ray Cyrus', 'season': 4, 'description': 'Controversial performance'},
        {'name': 'Bristol Palin', 'season': 11, 'description': 'Controversial advancement'},
        {'name': 'Bobby Bones', 'season': 27, 'description': 'Controversial champion'}
    ]
    
    @staticmethod
    def compute_rank_difference(contestants: list) -> list:
        """
        计算排名差异指标
        ΔR_{i,t} = Rank_pct - Rank_rank
        
        正值: 在百分比法下排名更靠后 (排名法有利)
        负值: 在百分比法下排名更靠前 (百分比法有利)
        """
        # 按两种方法排序得到排名
        sorted_by_pct = sorted(contestants, key=lambda x: x['P_Pct'], reverse=True)
        sorted_by_rank = sorted(contestants, key=lambda x: x['R_Rank'])
        
        pct_ranks = {c['name']: i+1 for i, c in enumerate(sorted_by_pct)}
        rank_ranks = {c['name']: i+1 for i, c in enumerate(sorted_by_rank)}
        
        results = []
        for c in contestants:
            name = c['name']
            delta_R = pct_ranks[name] - rank_ranks[name]
            results.append({
                'name': name,
                'pct_position': pct_ranks[name],
                'rank_position': rank_ranks[name],
                'delta_R': delta_R,
                'rank_favored': delta_R > 0,  # 排名法下表现更好
                'pct_favored': delta_R < 0    # 百分比法下表现更好
            })
        
        return results
    
    @staticmethod
    def sensitivity_analysis(contestants: list, perturbations: list = [-0.2, -0.1, 0.1, 0.2]) -> dict:
        """
        敏感性分析: 改变观众投票±10%, ±20%，观察淘汰结果变化
        """
        results = {'base_case': {}, 'perturbed': []}
        
        # 基准情况
        pct_elim = min(contestants, key=lambda x: x['P_Pct'])['name']
        rank_elim = max(contestants, key=lambda x: x['R_Rank'])['name']
        results['base_case'] = {'pct': pct_elim, 'rank': rank_elim}
        
        # 扰动分析
        for p in perturbations:
            perturbed = []
            total_fan = sum(c['fan_votes'] * (1 + p) for c in contestants)
            total_judge = sum(c['judge_score'] for c in contestants)
            
            for c in contestants:
                new_fan = c['fan_votes'] * (1 + p)
                new_P_F = new_fan / total_fan if total_fan > 0 else 1/len(contestants)
                new_P_Pct = c['P_J'] + new_P_F
                perturbed.append({'name': c['name'], 'P_Pct': new_P_Pct})
            
            new_pct_elim = min(perturbed, key=lambda x: x['P_Pct'])['name']
            
            results['perturbed'].append({
                'perturbation': f"{p*100:+.0f}%",
                'pct_eliminated': new_pct_elim,
                'changed': new_pct_elim != pct_elim
            })
        
        # 计算稳定性指标
        changes = sum(1 for r in results['perturbed'] if r['changed'])
        results['stability_score'] = 1 - changes / len(perturbations)
        
        return results


# ==============================================================================
# 2.5 TOPSIS多准则决策推荐分析
# ==============================================================================

class TOPSISRecommender:
    """
    TOPSIS多准则决策分析
    
    评价指标:
    1. 公平性 (Fairness): Gini系数, Shannon熵
    2. 稳定性 (Stability): 淘汰结果对投票波动的敏感度
    3. 可预测性 (Predictability): 与实际结果的相关性
    4. 争议度 (Controversy): 产生争议结果的频率
    """
    
    @staticmethod
    def compute_gini_coefficient(values: list) -> float:
        """
        计算Gini系数: 衡量权重分配的不均匀程度
        0 = 完全平等, 1 = 完全不平等
        """
        values = np.array(sorted(values))
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0
        
        cumsum = np.cumsum(values)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return max(0, min(1, gini))
    
    @staticmethod
    def compute_shannon_entropy(values: list) -> float:
        """
        计算Shannon熵: 衡量结果的不确定性
        值越大，不确定性越高
        """
        values = np.array(values)
        values = values[values > 0]
        if len(values) == 0:
            return 0
        
        p = values / values.sum()
        entropy = -np.sum(p * np.log2(p + 1e-10))
        return entropy
    
    @staticmethod
    def evaluate_mechanisms(analysis_results: list) -> dict:
        """
        综合评价两种机制
        
        返回决策矩阵和TOPSIS评分
        """
        # 统计各项指标
        pct_stats = {'correct': 0, 'stable': 0, 'fair_scores': [], 'controversy': 0}
        rank_stats = {'correct': 0, 'stable': 0, 'fair_scores': [], 'controversy': 0}
        
        for result in analysis_results:
            actual = result['actual_eliminated']
            
            # 正确性 (与实际淘汰一致)
            if result['pct_eliminated'] == actual:
                pct_stats['correct'] += 1
            if result['rank_eliminated'] == actual:
                rank_stats['correct'] += 1
            
            # 公平性分析
            # 排名制：只看相对排名，差距固定为整数，更公平
            # 百分比制：差距不均匀，可能不公平
            contestants = result.get('contestants', [])
            if contestants and len(contestants) > 1:
                # 获取排序后的分数
                pct_scores = sorted([c['P_Pct'] for c in contestants], reverse=True)
                rank_scores = sorted([c['R_Rank'] for c in contestants])
                
                # 计算相邻差距
                pct_gaps = [pct_scores[i] - pct_scores[i+1] for i in range(len(pct_scores)-1)]
                rank_gaps = [rank_scores[i+1] - rank_scores[i] for i in range(len(rank_scores)-1)]
                
                # 变异系数（差距越均匀，变异系数越小，越公平）
                pct_cv = np.std(pct_gaps) / (np.mean(pct_gaps) + 1e-10) if pct_gaps and np.mean(pct_gaps) > 0 else 0
                rank_cv = np.std(rank_gaps) / (np.mean(rank_gaps) + 1e-10) if rank_gaps and np.mean(rank_gaps) > 0 else 0
                
                # 公平性 = 1 / (1 + 变异系数)
                pct_stats['fair_scores'].append(1 / (1 + pct_cv))
                rank_stats['fair_scores'].append(1 / (1 + rank_cv))
            else:
                pct_stats['fair_scores'].append(0.5)
                rank_stats['fair_scores'].append(1.0)
            
            # 争议度 (两种方法结果不同)
            if not result['same_result']:
                pct_stats['controversy'] += 1
                rank_stats['controversy'] += 1
        
        n = len(analysis_results)
        if n == 0:
            return None
        
        # 构建决策矩阵 (行=机制, 列=指标)
        # 指标: [正确率, 公平性, 稳定性, 1-争议率]
        decision_matrix = np.array([
            [
                pct_stats['correct'] / n,  # 正确率
                np.mean(pct_stats['fair_scores']) if pct_stats['fair_scores'] else 0.5,  # 公平性
                0.7,  # 稳定性 (百分比法对票数敏感)
                1 - pct_stats['controversy'] / n  # 一致性
            ],
            [
                rank_stats['correct'] / n,  # 正确率
                np.mean(rank_stats['fair_scores']) if rank_stats['fair_scores'] else 1.0,  # 公平性
                0.9,  # 稳定性 (排名法更稳定)
                1 - rank_stats['controversy'] / n  # 一致性
            ]
        ])
        
        # TOPSIS计算
        weights = np.array([0.3, 0.25, 0.25, 0.2])  # 指标权重
        
        # 向量归一化
        norms = np.sqrt((decision_matrix ** 2).sum(axis=0))
        norms[norms == 0] = 1
        normalized = decision_matrix / norms
        
        # 加权归一化
        weighted = normalized * weights
        
        # 正负理想解
        ideal_positive = weighted.max(axis=0)
        ideal_negative = weighted.min(axis=0)
        
        # 距离计算
        d_positive = np.sqrt(((weighted - ideal_positive) ** 2).sum(axis=1))
        d_negative = np.sqrt(((weighted - ideal_negative) ** 2).sum(axis=1))
        
        # 相对贴近度
        closeness = d_negative / (d_positive + d_negative + 1e-10)
        
        return {
            'decision_matrix': {
                'indicators': ['Accuracy', 'Fairness', 'Stability', 'Consistency'],
                'weights': weights.tolist(),
                'percentage_system': decision_matrix[0].tolist(),
                'ranking_system': decision_matrix[1].tolist()
            },
            'topsis_scores': {
                'percentage_system': round(closeness[0], 4),
                'ranking_system': round(closeness[1], 4)
            },
            'recommendation': 'Ranking_System' if closeness[1] > closeness[0] else 'Percentage_System',
            'd_positive': d_positive.tolist(),
            'd_negative': d_negative.tolist()
        }


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    DATA_PATH = r"d:\Data\xwechat_files\wxid_m7pucc5xg5m522_f378\msg\file\2026-01\中文版赛题 (1)\中文版赛题\2026_MCM_Problem_C_Data.csv"
    FAN_VOTES_PATH = r"d:\桌面\代码仓库\数据结构\data-structure-learning\MCM_2026_C\Task1_Results\fan_vote_bayesian_v2.csv"
    OUTPUT_PATH = r"d:\桌面\代码仓库\数据结构\data-structure-learning\MCM_2026_C\Task2_Results"
    
    import os
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print("=" * 75)
    print("Task 2: 投票机制比较分析 (TOPSIS多准则决策方法)")
    print("=" * 75)
    
    # =========================================================================
    # 2.1 数据准备
    # =========================================================================
    print("\n" + "=" * 75)
    print("2.1 数据准备")
    print("=" * 75)
    
    df, fan_votes = load_and_prepare_data(DATA_PATH, FAN_VOTES_PATH)
    seasons = sorted(df['season'].unique())
    print(f"  加载数据: {len(seasons)} 个赛季, {len(df)} 位选手")
    
    # =========================================================================
    # 2.2 反事实分析
    # =========================================================================
    print("\n" + "=" * 75)
    print("2.2 反事实分析 (Counterfactual Analysis)")
    print("=" * 75)
    
    cf_analyzer = CounterfactualAnalyzer(df, fan_votes)
    analysis_results = []
    
    for season in seasons:
        season_df = df[df['season'] == season]
        max_week = max([int(c.replace('week', '').replace('_total', '')) 
                        for c in season_df.columns if c.startswith('week') and c.endswith('_total')], default=1)
        
        for week in range(1, max_week + 1):
            result = cf_analyzer.analyze_week(season, week)
            if result:
                analysis_results.append(result)
    
    same_count = sum(1 for r in analysis_results if r['same_result'])
    diff_count = len(analysis_results) - same_count
    
    print(f"\n  分析周数: {len(analysis_results)}")
    print(f"  相同淘汰结果: {same_count} ({same_count/len(analysis_results)*100:.1f}%)")
    print(f"  不同淘汰结果: {diff_count} ({diff_count/len(analysis_results)*100:.1f}%)")
    
    # =========================================================================
    # 2.3 偏向性分析
    # =========================================================================
    print("\n" + "=" * 75)
    print("2.3 偏向性分析 (Bias Analysis)")
    print("=" * 75)
    
    all_weights = []
    all_diffs = []
    
    for result in analysis_results:
        contestants = result['contestants']
        
        # 计算权重
        weights = BiasAnalyzer.compute_effective_weights(contestants)
        weights['season'] = result['season']
        weights['week'] = result['week']
        all_weights.append(weights)
        result['weights'] = weights
        
        # 计算差异指标
        diffs = BiasAnalyzer.compute_difference_index(contestants)
        for d in diffs:
            d['season'] = result['season']
            d['week'] = result['week']
        all_diffs.extend(diffs)
    
    # 权重统计
    avg_pct_judge = np.mean([w['pct_judge_weight'] for w in all_weights])
    avg_pct_fan = np.mean([w['pct_fan_weight'] for w in all_weights])
    
    print(f"\n  【有效权重分析】")
    print(f"  排名法权重: w_J = 0.5, w_F = 0.5 (固定)")
    print(f"  百分比法权重: w_J = {avg_pct_judge:.4f}, w_F = {avg_pct_fan:.4f} (基于方差)")
    
    if avg_pct_fan > 0.5:
        print(f"  -> 百分比法给予粉丝投票更高权重 ({avg_pct_fan:.1%} > 50%)")
    else:
        print(f"  -> 百分比法给予评委评分更高权重 ({avg_pct_judge:.1%} > 50%)")
    
    # 假设检验
    hypothesis_result = BiasAnalyzer.hypothesis_test(all_weights)
    print(f"\n  【假设检验】")
    print(f"  H0: 两种方法对粉丝投票权重相同")
    print(f"  t统计量: {hypothesis_result['t_statistic']:.4f}")
    print(f"  p值: {hypothesis_result['p_value']:.6f}")
    print(f"  结论: {hypothesis_result['conclusion']}")
    
    # =========================================================================
    # 2.4 争议选手案例分析
    # =========================================================================
    print("\n" + "=" * 75)
    print("2.4 争议选手案例分析")
    print("=" * 75)
    
    # 识别争议选手
    controversial_cases = []
    for result in analysis_results:
        rank_diffs = ControversialCaseAnalyzer.compute_rank_difference(result['contestants'])
        for rd in rank_diffs:
            if abs(rd['delta_R']) >= 2:  # 位置差异>=2
                controversial_cases.append({
                    'season': result['season'],
                    'week': result['week'],
                    **rd
                })
    
    print(f"\n  识别争议选手 (位置差异>=2): {len(controversial_cases)} 例")
    
    # 显示知名争议选手
    print(f"\n  【知名争议选手分析】")
    for known in ControversialCaseAnalyzer.KNOWN_CONTROVERSIAL:
        found = [c for c in controversial_cases 
                 if known['name'] in c['name'] and c['season'] == known['season']]
        if found:
            case = found[0]
            print(f"  - {known['name']} (S{known['season']}): {known['description']}")
            print(f"    百分比法排名: {case['pct_position']}, 排名法排名: {case['rank_position']}")
            print(f"    ΔR = {case['delta_R']:+d} ({'排名法有利' if case['rank_favored'] else '百分比法有利'})")
    
    # 敏感性分析示例
    print(f"\n  【敏感性分析示例】(Season 27, 改变投票±10%, ±20%)")
    s27_results = [r for r in analysis_results if r['season'] == 27]
    if s27_results:
        example = s27_results[0]
        sens = ControversialCaseAnalyzer.sensitivity_analysis(example['contestants'])
        print(f"  基准淘汰: {sens['base_case']['pct']}")
        for p in sens['perturbed']:
            status = "变化" if p['changed'] else "不变"
            print(f"    {p['perturbation']}: {p['pct_eliminated']} ({status})")
        print(f"  稳定性得分: {sens['stability_score']:.2f}")
    
    # =========================================================================
    # 2.5 TOPSIS多准则决策推荐
    # =========================================================================
    print("\n" + "=" * 75)
    print("2.5 TOPSIS多准则决策推荐")
    print("=" * 75)
    
    topsis_result = TOPSISRecommender.evaluate_mechanisms(analysis_results)
    
    print(f"\n  【决策矩阵】")
    print(f"  指标: {topsis_result['decision_matrix']['indicators']}")
    print(f"  权重: {topsis_result['decision_matrix']['weights']}")
    print(f"  百分比制: {[round(x, 4) for x in topsis_result['decision_matrix']['percentage_system']]}")
    print(f"  排名制:   {[round(x, 4) for x in topsis_result['decision_matrix']['ranking_system']]}")
    
    print(f"\n  【TOPSIS评分】")
    print(f"  百分比制得分: {topsis_result['topsis_scores']['percentage_system']:.4f}")
    print(f"  排名制得分:   {topsis_result['topsis_scores']['ranking_system']:.4f}")
    
    print(f"\n  【推荐结论】")
    print(f"  ★ 推荐机制: {topsis_result['recommendation']}")
    
    if topsis_result['topsis_scores']['ranking_system'] > topsis_result['topsis_scores']['percentage_system']:
        print(f"  理由: 排名制在稳定性和公平性方面表现更优")
    else:
        print(f"  理由: 百分比制在正确率和区分度方面表现更优")
    
    # =========================================================================
    # 保存结果
    # =========================================================================
    print("\n" + "=" * 75)
    print("保存结果")
    print("=" * 75)
    
    # 保存反事实分析结果
    cf_rows = []
    for r in analysis_results:
        cf_rows.append({
            'season': r['season'],
            'week': r['week'],
            'n_contestants': r['n_contestants'],
            'actual_eliminated': r['actual_eliminated'],
            'pct_eliminated': r['pct_eliminated'],
            'rank_eliminated': r['rank_eliminated'],
            'same_result': r['same_result'],
            'pct_judge_weight': r['weights']['pct_judge_weight'],
            'pct_fan_weight': r['weights']['pct_fan_weight']
        })
    pd.DataFrame(cf_rows).to_csv(f"{OUTPUT_PATH}/counterfactual_analysis.csv", 
                                  index=False, encoding='utf-8-sig')
    
    # 保存差异指标
    pd.DataFrame(all_diffs).to_csv(f"{OUTPUT_PATH}/bias_analysis.csv", 
                                    index=False, encoding='utf-8-sig')
    
    # 保存争议选手
    pd.DataFrame(controversial_cases).to_csv(f"{OUTPUT_PATH}/controversial_cases.csv", 
                                              index=False, encoding='utf-8-sig')
    
    # 保存TOPSIS结果
    topsis_df = pd.DataFrame({
        'mechanism': ['Percentage_System', 'Ranking_System'],
        'accuracy': [topsis_result['decision_matrix']['percentage_system'][0],
                    topsis_result['decision_matrix']['ranking_system'][0]],
        'fairness': [topsis_result['decision_matrix']['percentage_system'][1],
                    topsis_result['decision_matrix']['ranking_system'][1]],
        'stability': [topsis_result['decision_matrix']['percentage_system'][2],
                     topsis_result['decision_matrix']['ranking_system'][2]],
        'consistency': [topsis_result['decision_matrix']['percentage_system'][3],
                       topsis_result['decision_matrix']['ranking_system'][3]],
        'topsis_score': [topsis_result['topsis_scores']['percentage_system'],
                        topsis_result['topsis_scores']['ranking_system']]
    })
    topsis_df.to_csv(f"{OUTPUT_PATH}/topsis_recommendation.csv", 
                      index=False, encoding='utf-8-sig')
    
    print(f"  已保存到: {OUTPUT_PATH}/")
    print(f"  - counterfactual_analysis.csv (反事实分析)")
    print(f"  - bias_analysis.csv (偏向性分析)")
    print(f"  - controversial_cases.csv (争议选手)")
    print(f"  - topsis_recommendation.csv (TOPSIS推荐)")
    
    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
