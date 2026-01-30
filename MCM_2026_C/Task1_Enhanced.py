# -*- coding: utf-8 -*-
"""
================================================================================
2026 MCM Problem C - Task 1: 粉丝投票估算模型（增强版）
================================================================================

方法融合：
1. K-means聚类：根据选手特征将选手分为不同人气等级
2. 贝叶斯先验：基于聚类结果和选手特征计算先验
3. 约束优化：满足淘汰约束的条件下估算投票

创新点：
- 使用K-means对选手进行无监督分类
- 聚类结果作为先验分布的依据
- 结合多种信息源提高估算准确性

================================================================================
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# 1. 数据加载与预处理
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
    
    # 提取最终名次
    def parse_placement(r):
        if pd.isna(r): return 99
        r = str(r).lower()
        if '1st' in r: return 1
        if '2nd' in r: return 2
        if '3rd' in r: return 3
        try:
            return int(''.join(filter(str.isdigit, r)))
        except:
            return 99
    
    df['final_placement'] = df['results'].apply(parse_placement)
    
    return df


# ==============================================================================
# 2. K-means选手聚类
# ==============================================================================

class ContestantClusterer:
    """
    K-means选手聚类器
    
    特征：
    1. 行业类型（编码）
    2. 舞伴历史成绩
    3. 年龄
    4. 最终名次（作为人气代理）
    
    输出：选手人气等级 (高/中/低)
    """
    
    # 行业编码
    INDUSTRY_ENCODING = {
        'Singer/Rapper': 5,
        'Actor/Actress': 4,
        'Athlete': 4,
        'TV Personality': 3,
        'Olympian': 4,
        'Model': 3,
        'Comedian': 3,
        'YouTube/TikTok': 5,
        'News Anchor': 2,
        'default': 2
    }
    
    def __init__(self, df: pd.DataFrame, n_clusters: int = 3):
        self.df = df
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_labels = None
        self.cluster_stats = None
        
    def extract_features(self) -> np.ndarray:
        """提取聚类特征"""
        features = []
        
        # 计算舞伴历史成绩
        partner_stats = defaultdict(lambda: {'avg_placement': 10, 'count': 0})
        for _, row in self.df.iterrows():
            partner = row.get('ballroom_partner', 'Unknown')
            if pd.isna(partner):
                partner = 'Unknown'
            placement = row.get('final_placement', 10)
            if placement < 99:
                n = partner_stats[partner]['count']
                old_avg = partner_stats[partner]['avg_placement']
                partner_stats[partner]['avg_placement'] = (old_avg * n + placement) / (n + 1)
                partner_stats[partner]['count'] = n + 1
        
        for _, row in self.df.iterrows():
            # 特征1: 行业编码
            industry = row.get('celebrity_industry', 'Unknown')
            if pd.isna(industry):
                industry = 'Unknown'
            industry_score = self.INDUSTRY_ENCODING.get('default', 2)
            for key, score in self.INDUSTRY_ENCODING.items():
                if key != 'default' and key.lower() in str(industry).lower():
                    industry_score = score
                    break
            
            # 特征2: 舞伴历史成绩
            partner = row.get('ballroom_partner', 'Unknown')
            if pd.isna(partner):
                partner = 'Unknown'
            partner_score = 10 - partner_stats[partner]['avg_placement']  # 转换为正向
            
            # 特征3: 年龄（转换为适合度分数）
            age = row.get('celebrity_age_during_season', 35)
            if pd.isna(age):
                age = 35
            else:
                try:
                    age = float(age)
                except:
                    age = 35
            age_score = 10 - abs(age - 35) * 0.2  # 35岁最优
            age_score = max(0, min(10, age_score))
            
            # 特征4: 最终名次（转换为人气分数）
            placement = row.get('final_placement', 10)
            if placement >= 99:
                placement = 10
            popularity_score = max(0, 11 - placement)  # 第1名=10分，第10名=1分
            
            features.append([industry_score, partner_score, age_score, popularity_score])
        
        return np.array(features)
    
    def fit(self):
        """执行K-means聚类"""
        features = self.extract_features()
        
        # 标准化
        features_scaled = self.scaler.fit_transform(features)
        
        # K-means聚类
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # 计算每个聚类的统计信息
        self.cluster_stats = {}
        for c in range(self.n_clusters):
            mask = self.cluster_labels == c
            cluster_features = features[mask]
            avg_score = cluster_features.mean()
            
            # 根据平均分确定人气等级
            self.cluster_stats[c] = {
                'avg_score': avg_score,
                'count': mask.sum(),
                'avg_placement': self.df.loc[mask, 'final_placement'].mean()
            }
        
        # 按平均名次排序确定人气等级
        sorted_clusters = sorted(self.cluster_stats.items(), 
                                  key=lambda x: x[1]['avg_placement'])
        
        self.popularity_mapping = {}
        labels = ['High', 'Medium', 'Low']
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            if i < len(labels):
                self.popularity_mapping[cluster_id] = labels[i]
            else:
                self.popularity_mapping[cluster_id] = 'Medium'
        
        # 为每个选手分配人气等级
        self.df['cluster'] = self.cluster_labels
        self.df['popularity_level'] = self.df['cluster'].map(self.popularity_mapping)
        
        return self
    
    def get_popularity_prior(self, name: str, season: int) -> float:
        """
        获取选手的人气先验系数
        
        High: 1.3
        Medium: 1.0
        Low: 0.7
        """
        mask = (self.df['celebrity_name'] == name) & (self.df['season'] == season)
        if mask.sum() == 0:
            return 1.0
        
        pop_level = self.df.loc[mask, 'popularity_level'].values[0]
        
        priors = {'High': 1.3, 'Medium': 1.0, 'Low': 0.7}
        return priors.get(pop_level, 1.0)


# ==============================================================================
# 3. 增强贝叶斯先验
# ==============================================================================

class EnhancedBayesianPrior:
    """
    增强贝叶斯先验
    
    结合：
    1. K-means聚类结果（人气等级）
    2. 行业特征
    3. 舞伴历史
    4. 评委分数趋势
    """
    
    def __init__(self, df: pd.DataFrame, clusterer: ContestantClusterer):
        self.df = df
        self.clusterer = clusterer
    
    def compute_prior(self, week_data: pd.DataFrame, week: int, season: int) -> np.ndarray:
        """计算增强先验"""
        n = len(week_data)
        priors = np.ones(n)
        
        for i, (_, row) in enumerate(week_data.iterrows()):
            prior_score = 1.0
            name = row['celebrity_name']
            
            # 1. K-means人气先验（核心创新）
            popularity_prior = self.clusterer.get_popularity_prior(name, season)
            prior_score *= popularity_prior
            
            # 2. 当前评委分数（相对于平均）
            score_col = f'week{week}_total'
            if score_col in row.index:
                current_score = row[score_col]
                avg_score = week_data[score_col].mean()
                if avg_score > 0:
                    score_ratio = current_score / avg_score
                    prior_score *= (0.7 + 0.3 * score_ratio)  # 调整幅度
            
            # 3. 历史趋势
            trend_scores = []
            for prev_week in range(1, week):
                col = f'week{prev_week}_total'
                if col in row.index and not pd.isna(row[col]) and row[col] > 0:
                    trend_scores.append(row[col])
            
            if len(trend_scores) >= 2:
                # 分数趋势：上升=加分，下降=减分
                trend = (trend_scores[-1] - trend_scores[0]) / len(trend_scores)
                prior_score *= (1 + 0.02 * trend)
            
            # 4. 赛季进度加成
            progress_factor = 1 + 0.03 * (week - 1)
            prior_score *= progress_factor
            
            priors[i] = max(0.1, prior_score)  # 确保正值
        
        # 归一化
        priors = priors / priors.sum()
        return priors


# ==============================================================================
# 4. 约束优化估算
# ==============================================================================

def estimate_votes_enhanced(week_data: pd.DataFrame, eliminated: str, week: int,
                            bayesian_prior: EnhancedBayesianPrior, season: int,
                            total_votes: float = 10_000_000,
                            prior_weight: float = 0.4) -> list:
    """
    增强版投票估算
    
    结合K-means聚类先验和约束优化
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
    
    # 计算增强贝叶斯先验
    prior = bayesian_prior.compute_prior(week_data, week, season)
    prior_votes = prior * total_votes
    
    # 目标函数：最大熵 + KL散度惩罚
    def objective(v):
        v_pct = v / v.sum()
        v_pct = np.clip(v_pct, 1e-10, 1)
        prior_clip = np.clip(prior, 1e-10, 1)
        
        entropy = np.sum(v_pct * np.log(v_pct))
        kl_div = np.sum(v_pct * np.log(v_pct / prior_clip))
        
        return entropy + prior_weight * kl_div
    
    def constraint_sum(v):
        return v.sum() - total_votes
    
    def constraint_elim(v):
        v_pct = v / v.sum()
        S = J_pct + v_pct
        min_gap = min(S[i] - S[e_idx] for i in range(n) if i != e_idx)
        return min_gap - 0.001
    
    # 初始猜测
    v0 = prior_votes.copy()
    v0[e_idx] *= 0.3
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
    
    # 计算结果
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
    sorted_indices = sorted(range(n), key=lambda x: S[x])
    
    for i in range(n):
        is_elim = (i == e_idx)
        
        # 置信度计算
        vote_range = ranges[i][1] - ranges[i][0]
        range_ratio = vote_range / total_votes
        
        if range_ratio < 0.1:
            range_conf = 0.8 + 0.2 * (1 - range_ratio / 0.1)
        elif range_ratio < 0.5:
            range_conf = 0.3 + 0.5 * (0.5 - range_ratio) / 0.4
        else:
            range_conf = 0.3 * (1 - min(range_ratio, 1))
        
        prior_consistency = 1 - abs(v_pct[i] - prior[i]) / max(prior[i], 0.01)
        prior_consistency = max(0, min(1, prior_consistency))
        
        if is_elim:
            margin_conf = 0.6
        else:
            lowest = S[sorted_indices[0]]
            margin = S[i] - lowest
            margin_conf = min(0.7, margin * 3)
        
        contestant_factor = max(0.3, 1 - (n - 2) * 0.08)
        
        overall_conf = (
            0.25 * range_conf +
            0.30 * prior_consistency +
            0.25 * margin_conf +
            0.20 * contestant_factor
        )
        
        if is_elim:
            overall_conf = min(0.85, overall_conf * 1.15)
        
        overall_conf = max(0.1, min(0.9, overall_conf))
        conf_level = 'High' if overall_conf > 0.55 else ('Medium' if overall_conf > 0.35 else 'Low')
        
        # 获取人气等级
        pop_level = 'Unknown'
        mask = (bayesian_prior.df['celebrity_name'] == names[i]) & \
               (bayesian_prior.df['season'] == season)
        if mask.sum() > 0:
            pop_level = bayesian_prior.df.loc[mask, 'popularity_level'].values[0]
        
        results.append({
            'celebrity_name': names[i],
            'judge_score': J[i],
            'judge_percentage': round(J_pct[i] * 100, 2),
            'popularity_level': pop_level,  # K-means聚类结果
            'prior_votes': int(prior_votes[i]),
            'prior_percentage': round(prior[i] * 100, 2),
            'estimated_votes': int(v_est[i]),
            'vote_percentage': round(v_pct[i] * 100, 2),
            'combined_score': round(S[i] * 100, 2),
            'min_votes': int(ranges[i][0]),
            'max_votes': int(ranges[i][1]),
            'prior_consistency': round(prior_consistency * 100, 1),
            'confidence': round(overall_conf * 100, 1),
            'confidence_level': conf_level,
            'is_eliminated': is_elim
        })
    
    return results


# ==============================================================================
# 5. 一致性验证
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
# 6. 主程序
# ==============================================================================

def main():
    DATA_PATH = r"d:\Data\xwechat_files\wxid_m7pucc5xg5m522_f378\msg\file\2026-01\中文版赛题 (1)\中文版赛题\2026_MCM_Problem_C_Data.csv"
    OUTPUT_PATH = r"d:\桌面\代码仓库\数据结构\data-structure-learning\MCM_2026_C\Task1_Results\fan_vote_enhanced.csv"
    
    print("=" * 75)
    print("Task 1: 粉丝投票估算模型（K-means + 贝叶斯 + 约束优化 增强版）")
    print("=" * 75)
    
    # 加载数据
    print("\n[1] 加载数据...")
    df = load_data(DATA_PATH)
    seasons = sorted(df['season'].unique())
    print(f"    共 {len(seasons)} 个赛季, {len(df)} 位选手")
    
    # K-means聚类
    print("\n[2] K-means选手聚类...")
    clusterer = ContestantClusterer(df, n_clusters=3)
    clusterer.fit()
    
    # 输出聚类统计
    print("\n    【聚类结果统计】")
    for cluster_id, stats in clusterer.cluster_stats.items():
        pop_level = clusterer.popularity_mapping[cluster_id]
        print(f"    {pop_level} (簇{cluster_id}): {stats['count']}人, 平均名次={stats['avg_placement']:.2f}")
    
    # 初始化增强贝叶斯先验
    print("\n[3] 初始化增强贝叶斯先验...")
    bayesian_prior = EnhancedBayesianPrior(df, clusterer)
    
    # 估算所有赛季
    print("\n[4] 估算粉丝投票...")
    all_rows = []
    total_weeks = 0
    consistent_weeks = 0
    conf_stats = {'High': 0, 'Medium': 0, 'Low': 0}
    pop_stats = {'High': 0, 'Medium': 0, 'Low': 0}
    
    for season in seasons:
        season_df = df[df['season'] == season]
        
        max_week = 1
        for w in range(11, 0, -1):
            if f'week{w}_total' in season_df.columns and (season_df[f'week{w}_total'] > 0).any():
                max_week = w
                break
        
        season_week_count = 0
        
        for week in range(1, max_week + 1):
            active = ((season_df['elim_week'] == -1) | (season_df['elim_week'] >= week))
            score_col = f'week{week}_total'
            
            if score_col not in season_df.columns:
                continue
            
            active &= (season_df[score_col] > 0)
            week_data = season_df[active].copy()
            
            if len(week_data) < 2:
                continue
            
            elim_df = season_df[season_df['elim_week'] == week]
            eliminated = elim_df.iloc[0]['celebrity_name'] if len(elim_df) > 0 else None
            
            if eliminated is None:
                continue
            
            if eliminated not in week_data['celebrity_name'].values:
                continue
            
            # 增强版估算
            results = estimate_votes_enhanced(week_data, eliminated, week, 
                                               bayesian_prior, season)
            if not results:
                continue
            
            validation = validate_consistency(results)
            total_weeks += 1
            season_week_count += 1
            if validation['is_consistent']:
                consistent_weeks += 1
            
            for r in results:
                r['season'] = season
                r['week'] = week
                r['consistency_check'] = validation['is_consistent']
                conf_stats[r['confidence_level']] += 1
                
                pop = r.get('popularity_level', '未知')
                if pop in pop_stats:
                    pop_stats[pop] += 1
                
                all_rows.append(r)
        
        print(f"    Season {season:2d}: {season_week_count} 周已估算")
    
    # 保存结果
    print(f"\n[5] 保存结果...")
    result_df = pd.DataFrame(all_rows)
    
    cols = ['season', 'week', 'celebrity_name', 'popularity_level',
            'judge_score', 'judge_percentage',
            'prior_votes', 'prior_percentage',
            'estimated_votes', 'vote_percentage', 'combined_score',
            'min_votes', 'max_votes', 'prior_consistency',
            'confidence', 'confidence_level',
            'is_eliminated', 'consistency_check']
    result_df = result_df[cols]
    result_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    # 输出统计
    print("\n" + "=" * 75)
    print("结果统计")
    print("=" * 75)
    
    consistency_rate = consistent_weeks / total_weeks * 100 if total_weeks > 0 else 0
    print(f"\n【一致性验证】")
    print(f"  总周数: {total_weeks}")
    print(f"  正确解释: {consistent_weeks}")
    print(f"  一致性: {consistency_rate:.2f}%")
    
    total_est = sum(conf_stats.values())
    print(f"\n【置信度分布】")
    for level in ['High', 'Medium', 'Low']:
        pct = conf_stats[level] / total_est * 100 if total_est > 0 else 0
        print(f"  {level:8s}: {conf_stats[level]:4d} ({pct:5.1f}%)")
    
    total_pop = sum(pop_stats.values())
    print(f"\n[K-means Popularity Distribution]")
    for level in ['High', 'Medium', 'Low']:
        pct = pop_stats[level] / total_pop * 100 if total_pop > 0 else 0
        print(f"  {level}: {pop_stats[level]:4d} ({pct:5.1f}%)")
    
    print(f"\n结果已保存: {OUTPUT_PATH}")
    print("=" * 75)
    
    # 方法说明
    print("\n" + "=" * 75)
    print("方法说明")
    print("=" * 75)
    print("""
【K-means聚类】
特征向量: [行业编码, 舞伴历史成绩, 年龄适合度, 最终名次]
聚类数: 3 (高人气/中人气/低人气)
作用: 为选手分配人气等级，作为先验分布的基础

【增强贝叶斯先验】
先验系数 = K-means人气系数 × 评委分数相对值 × 历史趋势 × 赛季进度
- 高人气选手: 先验系数 × 1.3
- 中人气选手: 先验系数 × 1.0
- 低人气选手: 先验系数 × 0.7

【约束优化】
目标: 最大熵 + λ × KL散度(估计||先验)
约束: 被淘汰者综合得分必须最低

【创新点】
1. 无监督学习(K-means)与有约束优化的融合
2. 聚类结果直接影响先验分布
3. 多源信息融合提高估算可靠性
""")


if __name__ == "__main__":
    main()
