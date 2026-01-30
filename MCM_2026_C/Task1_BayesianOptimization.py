# -*- coding: utf-8 -*-
"""
================================================================================
2026 MCM Problem C - Task 1: 粉丝投票估算模型（贝叶斯+约束优化融合版）
================================================================================

方法：
1. 贝叶斯先验：基于选手特征（行业、舞伴、年龄、赛季进度）计算先验投票倾向
2. 约束优化：满足淘汰约束的条件下，最小化与先验的KL散度
3. 后验融合：结合先验信息和淘汰事件，得到更可靠的估算

输出：一个CSV文件，包含：
1. 每位选手每周的估算投票数（贝叶斯+约束优化融合）
2. 先验投票倾向（纯贝叶斯）
3. 一致性验证结果
4. 置信度指标

================================================================================
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# 数据加载
# ==============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """加载并预处理数据"""
    df = pd.read_csv(csv_path).replace('N/A', np.nan)
    
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
    return df


# ==============================================================================
# 贝叶斯先验计算
# ==============================================================================

class BayesianPrior:
    """
    计算贝叶斯先验投票倾向
    
    先验因素：
    1. 行业热度：不同行业的名人有不同的粉丝基础
    2. 舞伴历史：舞伴过往成绩影响观众期待
    3. 年龄因素：年龄与受欢迎程度的关系
    4. 赛季进度：后期选手获得更多关注
    5. 评委分数趋势：连续高分选手更受青睐
    """
    
    # 行业热度权重（基于历史数据分析）
    INDUSTRY_WEIGHTS = {
        'Actor/Actress': 1.2,
        'Singer/Rapper': 1.3,
        'Athlete': 1.15,
        'TV Personality': 1.1,
        'Model': 1.05,
        'News Anchor': 0.95,
        'Comedian': 1.1,
        'Dancer': 1.0,
        'Olympian': 1.2,
        'Reality TV': 1.15,
        'YouTube/TikTok': 1.4,  # 社交媒体明星粉丝活跃
        'default': 1.0
    }
    
    def __init__(self, full_df: pd.DataFrame):
        """初始化，计算全局统计信息"""
        self.full_df = full_df
        self.partner_stats = self._compute_partner_stats()
        self.industry_stats = self._compute_industry_stats()
    
    def _compute_partner_stats(self) -> dict:
        """计算舞伴历史成绩"""
        stats = defaultdict(lambda: {'wins': 0, 'top3': 0, 'total': 0, 'avg_placement': 5})
        
        for _, row in self.full_df.iterrows():
            partner = row.get('ballroom_partner', 'Unknown')
            if pd.isna(partner):
                partner = 'Unknown'
            
            placement = row.get('placement', 10)
            if pd.isna(placement):
                placement = 10
            else:
                try:
                    placement = int(placement)
                except:
                    placement = 10
            
            stats[partner]['total'] += 1
            if placement == 1:
                stats[partner]['wins'] += 1
            if placement <= 3:
                stats[partner]['top3'] += 1
            
            # 更新平均名次
            n = stats[partner]['total']
            old_avg = stats[partner]['avg_placement']
            stats[partner]['avg_placement'] = old_avg + (placement - old_avg) / n
        
        return dict(stats)
    
    def _compute_industry_stats(self) -> dict:
        """计算行业历史表现"""
        stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'avg_placement': 5})
        
        for _, row in self.full_df.iterrows():
            industry = row.get('celebrity_industry', 'Unknown')
            if pd.isna(industry):
                industry = 'Unknown'
            
            placement = row.get('placement', 10)
            if pd.isna(placement):
                placement = 10
            else:
                try:
                    placement = int(placement)
                except:
                    placement = 10
            
            stats[industry]['total'] += 1
            if placement == 1:
                stats[industry]['wins'] += 1
            
            n = stats[industry]['total']
            old_avg = stats[industry]['avg_placement']
            stats[industry]['avg_placement'] = old_avg + (placement - old_avg) / n
        
        return dict(stats)
    
    def compute_prior(self, week_data: pd.DataFrame, week: int, max_week: int) -> np.ndarray:
        """
        计算该周每位选手的先验投票倾向
        
        返回：归一化的先验概率分布
        """
        n = len(week_data)
        priors = np.ones(n)
        
        for i, (_, row) in enumerate(week_data.iterrows()):
            prior_score = 1.0
            
            # 1. 行业热度
            industry = row.get('celebrity_industry', 'Unknown')
            if pd.isna(industry):
                industry = 'Unknown'
            
            # 检查行业关键词
            industry_weight = self.INDUSTRY_WEIGHTS.get('default', 1.0)
            for key, weight in self.INDUSTRY_WEIGHTS.items():
                if key != 'default' and key.lower() in str(industry).lower():
                    industry_weight = weight
                    break
            prior_score *= industry_weight
            
            # 行业历史表现
            if industry in self.industry_stats:
                ind_stat = self.industry_stats[industry]
                win_rate = ind_stat['wins'] / max(ind_stat['total'], 1)
                prior_score *= (1 + win_rate * 0.3)
            
            # 2. 舞伴历史
            partner = row.get('ballroom_partner', 'Unknown')
            if pd.isna(partner):
                partner = 'Unknown'
            
            if partner in self.partner_stats:
                p_stat = self.partner_stats[partner]
                win_rate = p_stat['wins'] / max(p_stat['total'], 1)
                top3_rate = p_stat['top3'] / max(p_stat['total'], 1)
                prior_score *= (1 + win_rate * 0.5 + top3_rate * 0.3)
            
            # 3. 年龄因素（30-45岁区间最受欢迎）
            age = row.get('celebrity_age_during_season', 35)
            if pd.isna(age):
                age = 35
            else:
                try:
                    age = float(age)
                except:
                    age = 35
            
            age_factor = 1.0 - 0.01 * abs(age - 37)  # 37岁为最佳
            age_factor = max(0.7, min(1.1, age_factor))
            prior_score *= age_factor
            
            # 4. 赛季进度（后期选手更受关注）
            progress_factor = 1 + 0.05 * (week - 1)
            prior_score *= progress_factor
            
            # 5. 评委分数趋势（如果有历史周数据）
            score_trend = 0
            score_count = 0
            for prev_week in range(1, week):
                col = f'week{prev_week}_total'
                if col in row.index and not pd.isna(row[col]) and row[col] > 0:
                    score_trend += row[col]
                    score_count += 1
            
            if score_count > 0:
                avg_past_score = score_trend / score_count
                current_col = f'week{week}_total'
                if current_col in row.index and not pd.isna(row[current_col]):
                    current_score = row[current_col]
                    # 如果当前分数高于历史平均，增加先验
                    if current_score > avg_past_score:
                        prior_score *= 1.1
                    elif current_score < avg_past_score * 0.8:
                        prior_score *= 0.9
            
            priors[i] = prior_score
        
        # 归一化
        priors = priors / priors.sum()
        return priors


# ==============================================================================
# 贝叶斯+约束优化融合估算
# ==============================================================================

def estimate_votes_bayesian(week_data: pd.DataFrame, eliminated: str, week: int,
                            bayesian_prior: BayesianPrior, max_week: int,
                            total_votes: float = 10_000_000,
                            prior_weight: float = 0.3) -> list:
    """
    贝叶斯+约束优化融合估算
    
    参数：
    - prior_weight: 先验权重 (0-1)，越高越依赖先验，越低越依赖约束
    
    方法：
    1. 计算贝叶斯先验
    2. 约束优化：目标函数 = 最大熵 + λ * KL散度(估计||先验)
    3. 满足淘汰约束
    """
    n = len(week_data)
    names = week_data['celebrity_name'].values
    J = week_data[f'week{week}_total'].values
    J_sum = J.sum()
    J_pct = J / J_sum if J_sum > 0 else np.ones(n) / n
    
    # 找被淘汰者
    e_idx = np.where(names == eliminated)[0]
    if len(e_idx) == 0:
        return None
    e_idx = e_idx[0]
    
    # 计算贝叶斯先验
    prior = bayesian_prior.compute_prior(week_data, week, max_week)
    prior_votes = prior * total_votes
    
    # 目标函数：最大熵 + KL散度惩罚
    def objective(v):
        v_pct = v / v.sum()
        v_pct = np.clip(v_pct, 1e-10, 1)
        prior_clip = np.clip(prior, 1e-10, 1)
        
        # 熵项（希望最大化，所以取负）
        entropy = np.sum(v_pct * np.log(v_pct))
        
        # KL散度项（希望接近先验）
        kl_div = np.sum(v_pct * np.log(v_pct / prior_clip))
        
        # 组合目标（prior_weight 控制先验影响程度）
        return entropy + prior_weight * kl_div
    
    def constraint_sum(v):
        return v.sum() - total_votes
    
    def constraint_elim(v):
        v_pct = v / v.sum()
        S = J_pct + v_pct
        min_gap = min(S[i] - S[e_idx] for i in range(n) if i != e_idx)
        return min_gap - 0.001
    
    # 用先验作为初始猜测
    v0 = prior_votes.copy()
    v0[e_idx] *= 0.3  # 被淘汰者降低
    v0 = v0 / v0.sum() * total_votes
    
    # 求解
    result = minimize(objective, v0, method='SLSQP',
                     bounds=[(0, total_votes) for _ in range(n)],
                     constraints=[{'type': 'eq', 'fun': constraint_sum},
                                 {'type': 'ineq', 'fun': constraint_elim}],
                     options={'maxiter': 1000})
    
    v_est = result.x if result.success else v0
    v_est = np.maximum(v_est, 0)
    v_est = v_est / v_est.sum() * total_votes
    
    # 计算综合得分
    v_pct = v_est / total_votes
    S = J_pct + v_pct
    
    # 计算投票范围
    ranges = []
    for i in range(n):
        if i == e_idx:
            max_v = total_votes
            for j in range(n):
                if j != e_idx:
                    upper = total_votes/n*0.3 + total_votes*(J_pct[j] - J_pct[e_idx])
                    max_v = min(max_v, max(0, upper))
            ranges.append((0, max_v))
        else:
            lower = max(0, total_votes/n*0.3 + total_votes*(J_pct[e_idx] - J_pct[i]))
            ranges.append((lower, total_votes))
    
    # 构造结果
    results = []
    
    # 计算排名和边际信息
    sorted_indices = sorted(range(n), key=lambda x: S[x])
    rank_map = {idx: rank for rank, idx in enumerate(sorted_indices)}
    
    for i in range(n):
        is_elim = (i == e_idx)
        rank = rank_map[i]
        
        # === 置信度计算（合理版）===
        
        # 1. 范围置信度：范围越窄越确定（更严格的计算）
        vote_range = ranges[i][1] - ranges[i][0]
        range_ratio = vote_range / total_votes
        # 范围占总票数比例：0-10% 高置信，10-50% 中置信，>50% 低置信
        if range_ratio < 0.1:
            range_conf = 0.8 + 0.2 * (1 - range_ratio / 0.1)
        elif range_ratio < 0.5:
            range_conf = 0.3 + 0.5 * (0.5 - range_ratio) / 0.4
        else:
            range_conf = 0.3 * (1 - min(range_ratio, 1))
        
        # 2. 先验一致性：估计值与先验的接近程度
        # 使用相对误差，更严格
        relative_diff = abs(v_pct[i] - prior[i]) / max(prior[i], 0.01)
        prior_consistency = max(0, 1 - relative_diff)
        prior_consistency = min(1, prior_consistency)
        
        # 3. 排名边际置信度（更保守）
        if is_elim:
            # 被淘汰者：与第二低分的差距
            if len(sorted_indices) > 1:
                second_lowest = S[sorted_indices[1]]
                margin = second_lowest - S[i]
                # 差距需要足够大才有高置信度
                margin_conf = min(0.8, margin * 5)
            else:
                margin_conf = 0.3
        else:
            # 未淘汰者：与最低分的差距
            lowest = S[sorted_indices[0]]
            margin = S[i] - lowest
            margin_conf = min(0.7, margin * 3)
        
        # 4. 选手数量因素：选手越多越难确定
        contestant_factor = max(0.3, 1 - (n - 2) * 0.08)
        
        # 综合置信度（加权平均，更保守）
        overall_conf = (
            0.30 * range_conf +
            0.25 * prior_consistency +
            0.25 * margin_conf +
            0.20 * contestant_factor
        )
        
        # 被淘汰者有约束，略微提升（但不过分）
        if is_elim:
            overall_conf = min(0.85, overall_conf * 1.15)
        
        # 确保在合理范围
        overall_conf = max(0.1, min(0.9, overall_conf))
        
        # 调整阈值：High > 0.55, Medium 0.35-0.55, Low < 0.35
        conf_level = 'High' if overall_conf > 0.55 else ('Medium' if overall_conf > 0.35 else 'Low')
        
        results.append({
            'celebrity_name': names[i],
            'judge_score': J[i],
            'judge_percentage': round(J_pct[i] * 100, 2),
            'prior_votes': int(prior_votes[i]),  # 纯贝叶斯先验
            'prior_percentage': round(prior[i] * 100, 2),
            'estimated_votes': int(v_est[i]),  # 融合后估计
            'vote_percentage': round(v_pct[i] * 100, 2),
            'combined_score': round(S[i] * 100, 2),
            'min_votes': int(ranges[i][0]),
            'max_votes': int(ranges[i][1]),
            'prior_consistency': round(prior_consistency * 100, 1),
            'margin_to_elim': round((S[i] - S[e_idx]) * 100, 2),  # 与被淘汰者的差距
            'confidence': round(overall_conf * 100, 1),
            'confidence_level': conf_level,
            'is_eliminated': is_elim
        })
    
    return results


# ==============================================================================
# 一致性验证
# ==============================================================================

def validate_consistency(results: list) -> dict:
    """验证估算结果的一致性"""
    sorted_by_score = sorted(results, key=lambda x: x['combined_score'])
    predicted_elim = sorted_by_score[0]['celebrity_name']
    actual_elim = next((r['celebrity_name'] for r in results if r['is_eliminated']), None)
    
    is_consistent = (predicted_elim == actual_elim)
    margin = sorted_by_score[1]['combined_score'] - sorted_by_score[0]['combined_score'] if len(sorted_by_score) > 1 else 0
    
    return {
        'is_consistent': is_consistent,
        'actual_eliminated': actual_elim,
        'predicted_eliminated': predicted_elim,
        'margin': round(margin, 2)
    }


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    DATA_PATH = r"d:\Data\xwechat_files\wxid_m7pucc5xg5m522_f378\msg\file\2026-01\中文版赛题 (1)\中文版赛题\2026_MCM_Problem_C_Data.csv"
    OUTPUT_PATH = r"d:\桌面\代码仓库\数据结构\data-structure-learning\MCM_2026_C\Task1_Results\fan_vote_bayesian_v2.csv"
    
    print("=" * 70)
    print("Task 1: 粉丝投票估算模型（贝叶斯+约束优化融合版）")
    print("=" * 70)
    
    # 加载数据
    print("\n[1] 加载数据...")
    df = load_data(DATA_PATH)
    seasons = sorted(df['season'].unique())
    print(f"    共 {len(seasons)} 个赛季, {len(df)} 位选手")
    
    # 初始化贝叶斯先验计算器
    print("\n[2] 计算贝叶斯先验...")
    bayesian_prior = BayesianPrior(df)
    print(f"    舞伴统计: {len(bayesian_prior.partner_stats)} 位舞伴")
    print(f"    行业统计: {len(bayesian_prior.industry_stats)} 个行业")
    
    # 估算所有赛季
    print("\n[3] 估算粉丝投票（融合方法）...")
    all_rows = []
    total_weeks = 0
    consistent_weeks = 0
    conf_stats = {'High': 0, 'Medium': 0, 'Low': 0}
    skipped_weeks = []
    
    for season in seasons:
        season_df = df[df['season'] == season]
        
        # 确定周数
        max_week = 1
        for w in range(11, 0, -1):
            if f'week{w}_total' in season_df.columns and (season_df[f'week{w}_total'] > 0).any():
                max_week = w
                break
        
        season_week_count = 0
        
        for week in range(1, max_week + 1):
            # 获取该周在场选手
            active = ((season_df['elim_week'] == -1) | (season_df['elim_week'] >= week))
            score_col = f'week{week}_total'
            
            if score_col not in season_df.columns:
                skipped_weeks.append({'season': season, 'week': week, 'reason': '无评分数据'})
                continue
            
            active &= (season_df[score_col] > 0)
            week_data = season_df[active].copy()
            
            if len(week_data) < 2:
                skipped_weeks.append({'season': season, 'week': week, 'reason': '选手不足'})
                continue
            
            # 获取被淘汰者
            elim_df = season_df[season_df['elim_week'] == week]
            eliminated = elim_df.iloc[0]['celebrity_name'] if len(elim_df) > 0 else None
            
            if eliminated is None:
                skipped_weeks.append({'season': season, 'week': week, 'reason': '无淘汰'})
                continue
            
            if eliminated not in week_data['celebrity_name'].values:
                skipped_weeks.append({'season': season, 'week': week, 'reason': '淘汰者无评分'})
                continue
            
            # 贝叶斯+约束优化融合估算
            results = estimate_votes_bayesian(week_data, eliminated, week, 
                                              bayesian_prior, max_week)
            if not results:
                skipped_weeks.append({'season': season, 'week': week, 'reason': '估算失败'})
                continue
            
            # 验证一致性
            validation = validate_consistency(results)
            total_weeks += 1
            season_week_count += 1
            if validation['is_consistent']:
                consistent_weeks += 1
            
            # 添加到总结果
            for r in results:
                r['season'] = season
                r['week'] = week
                r['consistency_check'] = validation['is_consistent']
                conf_stats[r['confidence_level']] += 1
                all_rows.append(r)
        
        print(f"    Season {season:2d}: {season_week_count} 周已估算")
    
    # 保存到CSV
    print(f"\n[4] 保存结果...")
    result_df = pd.DataFrame(all_rows)
    
    # 调整列顺序
    cols = ['season', 'week', 'celebrity_name', 'judge_score', 'judge_percentage',
            'prior_votes', 'prior_percentage',  # 贝叶斯先验
            'estimated_votes', 'vote_percentage', 'combined_score',  # 融合估计
            'min_votes', 'max_votes', 'margin_to_elim', 'prior_consistency',
            'confidence', 'confidence_level',
            'is_eliminated', 'consistency_check']
    result_df = result_df[cols]
    result_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    # 输出统计
    print("\n" + "=" * 70)
    print("结果统计")
    print("=" * 70)
    
    consistency_rate = consistent_weeks / total_weeks * 100 if total_weeks > 0 else 0
    print(f"\n一致性验证:")
    print(f"  总周数: {total_weeks}")
    print(f"  正确解释: {consistent_weeks}")
    print(f"  一致性: {consistency_rate:.2f}%")
    
    total_est = sum(conf_stats.values())
    print(f"\n置信度分布:")
    for level in ['High', 'Medium', 'Low']:
        pct = conf_stats[level] / total_est * 100 if total_est > 0 else 0
        print(f"  {level:8s}: {conf_stats[level]:4d} ({pct:5.1f}%)")
    
    # 先验一致性统计
    if all_rows:
        prior_cons = [r['prior_consistency'] for r in all_rows]
        print(f"\n先验一致性:")
        print(f"  平均: {np.mean(prior_cons):.1f}%")
        print(f"  中位: {np.median(prior_cons):.1f}%")
    
    if skipped_weeks:
        print(f"\n跳过的周 (共 {len(skipped_weeks)} 个):")
        reason_counts = {}
        for s in skipped_weeks:
            reason = s['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count} 周")
    
    print(f"\n结果已保存: {OUTPUT_PATH}")
    print("=" * 70)
    
    # 打印列说明
    print("\n" + "=" * 70)
    print("各列指标说明")
    print("=" * 70)
    print("""
| 列名               | 含义                                                |
|--------------------|-----------------------------------------------------|
| season             | 赛季编号 (1-34)                                      |
| week               | 比赛周数                                             |
| celebrity_name     | 参赛明星姓名                                         |
| judge_score        | 该周评委总分                                         |
| judge_percentage   | 评委得分百分比                                       |
| prior_votes        | 【贝叶斯先验】纯先验估算的投票数                       |
| prior_percentage   | 先验投票百分比                                       |
| estimated_votes    | 【核心】融合估算的粉丝投票数（贝叶斯+约束优化）         |
| vote_percentage    | 融合后的投票百分比                                    |
| combined_score     | 综合得分 = judge_percentage + vote_percentage         |
| min_votes          | 投票数下界                                           |
| max_votes          | 投票数上界                                           |
| margin_to_elim     | 与被淘汰者综合得分的差距                              |
| prior_consistency  | 先验一致性：估算值与贝叶斯先验的吻合程度 (0-100%)       |
| confidence         | 综合置信度分数                                        |
| confidence_level   | 置信度等级: High(>60%) / Medium(40-60%) / Low(<40%)   |
| is_eliminated      | 该选手本周是否被淘汰                                  |
| consistency_check  | 一致性验证: 模型是否正确预测了本周的淘汰结果            |

贝叶斯先验因素：
1. 行业热度：歌手/演员/运动员等不同行业的粉丝活跃度不同
2. 舞伴历史：舞伴过往冠军/前三名次数影响观众预期
3. 年龄因素：30-45岁区间选手通常更受欢迎
4. 赛季进度：后期选手获得更多关注和投票
5. 评委分数趋势：连续高分选手更受青睐
""")


if __name__ == "__main__":
    main()
