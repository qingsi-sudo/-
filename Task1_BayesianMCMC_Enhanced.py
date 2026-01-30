# -*- coding: utf-8 -*-
"""
================================================================================
2026 MCM Problem C - Task 1: 融合模型
================================================================================

方法融合：
1. K-means聚类 → 人气等级先验
2. 贝叶斯隐变量模型 → 投票比例建模
3. 区间截尾 → 淘汰约束处理
4. MCMC采样 → 后验分布估计

创新点：
- K-means聚类结果作为Dirichlet先验的超参数
- 区间截尾自然处理"被淘汰者得分最低"约束
- MCMC提供完整的不确定性量化
- 与确定性优化结果对比验证

================================================================================
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import dirichlet, truncnorm, norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# 1. 数据加载与预处理（复用）
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
# 2. K-means选手聚类（复用并增强）
# ==============================================================================

class ContestantClusterer:
    """K-means选手聚类器"""
    
    INDUSTRY_ENCODING = {
        'Singer/Rapper': 5, 'Actor/Actress': 4, 'Athlete': 4,
        'TV Personality': 3, 'Olympian': 4, 'Model': 3,
        'Comedian': 3, 'YouTube/TikTok': 5, 'News Anchor': 2, 'default': 2
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
        
        partner_stats = defaultdict(lambda: {'avg_placement': 10, 'count': 0})
        for _, row in self.df.iterrows():
            partner = row.get('ballroom_partner', 'Unknown')
            if pd.isna(partner): partner = 'Unknown'
            placement = row.get('final_placement', 10)
            if placement < 99:
                n = partner_stats[partner]['count']
                old_avg = partner_stats[partner]['avg_placement']
                partner_stats[partner]['avg_placement'] = (old_avg * n + placement) / (n + 1)
                partner_stats[partner]['count'] = n + 1
        
        for _, row in self.df.iterrows():
            industry = row.get('celebrity_industry', 'Unknown')
            if pd.isna(industry): industry = 'Unknown'
            industry_score = self.INDUSTRY_ENCODING.get('default', 2)
            for key, score in self.INDUSTRY_ENCODING.items():
                if key != 'default' and key.lower() in str(industry).lower():
                    industry_score = score
                    break
            
            partner = row.get('ballroom_partner', 'Unknown')
            if pd.isna(partner): partner = 'Unknown'
            partner_score = 10 - partner_stats[partner]['avg_placement']
            
            age = row.get('celebrity_age_during_season', 35)
            if pd.isna(age): age = 35
            else:
                try: age = float(age)
                except: age = 35
            age_score = max(0, min(10, 10 - abs(age - 35) * 0.2))
            
            placement = row.get('final_placement', 10)
            if placement >= 99: placement = 10
            popularity_score = max(0, 11 - placement)
            
            features.append([industry_score, partner_score, age_score, popularity_score])
        
        return np.array(features)
    
    def fit(self):
        """执行K-means聚类"""
        features = self.extract_features()
        features_scaled = self.scaler.fit_transform(features)
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        self.cluster_stats = {}
        for c in range(self.n_clusters):
            mask = self.cluster_labels == c
            cluster_features = features[mask]
            self.cluster_stats[c] = {
                'avg_score': cluster_features.mean(),
                'count': mask.sum(),
                'avg_placement': self.df.loc[mask, 'final_placement'].mean()
            }
        
        sorted_clusters = sorted(self.cluster_stats.items(), 
                                  key=lambda x: x[1]['avg_placement'])
        
        self.popularity_mapping = {}
        labels = ['High', 'Medium', 'Low']
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            self.popularity_mapping[cluster_id] = labels[min(i, len(labels)-1)]
        
        self.df['cluster'] = self.cluster_labels
        self.df['popularity_level'] = self.df['cluster'].map(self.popularity_mapping)
        
        return self
    
    def get_popularity_prior(self, name: str, season: int) -> float:
        """获取选手的人气先验系数"""
        mask = (self.df['celebrity_name'] == name) & (self.df['season'] == season)
        if mask.sum() == 0:
            return 1.0
        
        pop_level = self.df.loc[mask, 'popularity_level'].values[0]
        priors = {'High': 1.5, 'Medium': 1.0, 'Low': 0.6}
        return priors.get(pop_level, 1.0)
    
    def get_dirichlet_alpha(self, names: list, season: int, base_alpha: float = 2.0) -> np.ndarray:
        """
        根据K-means聚类结果生成Dirichlet分布的alpha参数
        
        这是融合的关键：聚类结果 → 先验超参数
        """
        alphas = []
        for name in names:
            prior = self.get_popularity_prior(name, season)
            alphas.append(base_alpha * prior)
        return np.array(alphas)


# ==============================================================================
# 3. 区间截尾贝叶斯隐变量模型
# ==============================================================================

class TruncatedBayesianVoteModel:
    """
    带区间截尾的贝叶斯隐变量模型
    
    模型结构：
    - 隐变量 θ: 真实投票比例 (Dirichlet先验)
    - 观测约束: 被淘汰者综合得分最低
    - 区间截尾: 投票比例必须满足淘汰约束
    
    θ ~ Dirichlet(α)  其中 α 由K-means聚类结果确定
    约束: J_pct[elim] + θ[elim] < J_pct[i] + θ[i] for all i ≠ elim
    """
    
    def __init__(self, judge_pct: np.ndarray, elim_idx: int, 
                 alpha: np.ndarray, total_votes: float = 10_000_000):
        """
        参数:
            judge_pct: 评委分数百分比
            elim_idx: 被淘汰者索引
            alpha: Dirichlet先验参数（来自K-means）
            total_votes: 总投票数
        """
        self.J = judge_pct
        self.e_idx = elim_idx
        self.alpha = alpha
        self.total_votes = total_votes
        self.n = len(judge_pct)
        
        # 存储MCMC样本
        self.samples = None
        self.acceptance_rate = 0
        
    def is_valid(self, theta: np.ndarray) -> bool:
        """检查投票比例是否满足淘汰约束"""
        S = self.J + theta  # 综合得分
        elim_score = S[self.e_idx]
        
        # 被淘汰者得分必须最低
        for i in range(self.n):
            if i != self.e_idx and S[i] <= elim_score:
                return False
        return True
    
    def log_prior(self, theta: np.ndarray) -> float:
        """Dirichlet先验的对数概率"""
        if np.any(theta <= 0) or np.abs(theta.sum() - 1) > 1e-6:
            return -np.inf
        
        # Dirichlet log-pdf
        log_p = np.sum((self.alpha - 1) * np.log(theta))
        return log_p
    
    def log_likelihood(self, theta: np.ndarray) -> float:
        """
        似然函数（区间截尾）
        
        如果满足约束返回0（均匀），否则返回-inf
        """
        if self.is_valid(theta):
            return 0.0
        return -np.inf
    
    def log_posterior(self, theta: np.ndarray) -> float:
        """后验对数概率"""
        ll = self.log_likelihood(theta)
        if ll == -np.inf:
            return -np.inf
        return self.log_prior(theta) + ll
    
    def propose(self, theta: np.ndarray, step_size: float = 0.02) -> np.ndarray:
        """
        Metropolis-Hastings提议分布
        
        使用对数正态随机游走，保证正值和归一化
        """
        # 对数空间随机游走
        log_theta = np.log(theta)
        log_theta_new = log_theta + np.random.normal(0, step_size, self.n)
        theta_new = np.exp(log_theta_new)
        theta_new = theta_new / theta_new.sum()  # 归一化
        return theta_new
    
    def run_mcmc(self, n_samples: int = 5000, burn_in: int = 1000, 
                 thin: int = 2, step_size: float = 0.02) -> np.ndarray:
        """
        运行MCMC采样
        
        参数:
            n_samples: 采样数
            burn_in: 预热期
            thin: 稀疏化间隔
            step_size: 提议步长
        """
        # 初始化：从先验采样，直到找到有效样本
        max_init_tries = 10000
        theta_current = None
        
        for _ in range(max_init_tries):
            # 从Dirichlet先验采样
            theta_try = np.random.dirichlet(self.alpha)
            if self.is_valid(theta_try):
                theta_current = theta_try
                break
        
        if theta_current is None:
            # 如果无法从先验找到有效样本，构造一个
            theta_current = self.alpha / self.alpha.sum()
            # 调整使其满足约束
            theta_current[self.e_idx] *= 0.3
            for i in range(self.n):
                if i != self.e_idx:
                    theta_current[i] *= 1.1
            theta_current = theta_current / theta_current.sum()
        
        log_p_current = self.log_posterior(theta_current)
        
        # MCMC采样
        samples = []
        accepted = 0
        total_iter = burn_in + n_samples * thin
        
        for i in range(total_iter):
            # 提议新样本
            theta_proposed = self.propose(theta_current, step_size)
            log_p_proposed = self.log_posterior(theta_proposed)
            
            # Metropolis-Hastings接受准则
            log_accept_ratio = log_p_proposed - log_p_current
            
            if np.log(np.random.random()) < log_accept_ratio:
                theta_current = theta_proposed
                log_p_current = log_p_proposed
                accepted += 1
            
            # 存储样本（跳过burn-in和thin）
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(theta_current.copy())
        
        self.samples = np.array(samples)
        self.acceptance_rate = accepted / total_iter
        
        return self.samples
    
    def get_posterior_stats(self) -> dict:
        """计算后验统计量"""
        if self.samples is None:
            raise ValueError("请先运行MCMC采样")
        
        mean = self.samples.mean(axis=0)
        std = self.samples.std(axis=0)
        
        # 95%可信区间
        ci_lower = np.percentile(self.samples, 2.5, axis=0)
        ci_upper = np.percentile(self.samples, 97.5, axis=0)
        
        # 转换为投票数
        votes_mean = mean * self.total_votes
        votes_std = std * self.total_votes
        votes_ci_lower = ci_lower * self.total_votes
        votes_ci_upper = ci_upper * self.total_votes
        
        return {
            'mean_pct': mean,
            'std_pct': std,
            'ci_lower_pct': ci_lower,
            'ci_upper_pct': ci_upper,
            'mean_votes': votes_mean,
            'std_votes': votes_std,
            'ci_lower_votes': votes_ci_lower,
            'ci_upper_votes': votes_ci_upper,
            'acceptance_rate': self.acceptance_rate
        }


# ==============================================================================
# 4. 确定性优化估算（对比用）
# ==============================================================================

def estimate_votes_deterministic(week_data: pd.DataFrame, eliminated: str, week: int,
                                  clusterer: ContestantClusterer, season: int,
                                  total_votes: float = 10_000_000) -> np.ndarray:
    """确定性优化估算（原方法）"""
    n = len(week_data)
    names = week_data['celebrity_name'].values
    J = week_data[f'week{week}_total'].values
    J_sum = J.sum()
    J_pct = J / J_sum if J_sum > 0 else np.ones(n) / n
    
    e_idx = np.where(names == eliminated)[0]
    if len(e_idx) == 0:
        return None
    e_idx = e_idx[0]
    
    # K-means先验
    alpha = clusterer.get_dirichlet_alpha(list(names), season)
    prior = alpha / alpha.sum()
    prior_votes = prior * total_votes
    
    # 目标函数
    def objective(v):
        v_pct = v / v.sum()
        v_pct = np.clip(v_pct, 1e-10, 1)
        prior_clip = np.clip(prior, 1e-10, 1)
        entropy = np.sum(v_pct * np.log(v_pct))
        kl_div = np.sum(v_pct * np.log(v_pct / prior_clip))
        return entropy + 0.4 * kl_div
    
    def constraint_sum(v):
        return v.sum() - total_votes
    
    def constraint_elim(v):
        v_pct = v / v.sum()
        S = J_pct + v_pct
        min_gap = min(S[i] - S[e_idx] for i in range(n) if i != e_idx)
        return min_gap - 0.001
    
    v0 = prior_votes.copy()
    v0[e_idx] *= 0.3
    v0 = v0 / v0.sum() * total_votes
    
    result = minimize(objective, v0, method='SLSQP',
                     bounds=[(0, total_votes) for _ in range(n)],
                     constraints=[{'type': 'eq', 'fun': constraint_sum},
                                 {'type': 'ineq', 'fun': constraint_elim}],
                     options={'maxiter': 1000})
    
    v_est = result.x if result.success else v0
    v_est = np.maximum(v_est, 0)
    v_est = v_est / v_est.sum() * total_votes
    
    return v_est


# ==============================================================================
# 5. 融合估算器
# ==============================================================================

def estimate_votes_hybrid(week_data: pd.DataFrame, eliminated: str, week: int,
                          clusterer: ContestantClusterer, season: int,
                          total_votes: float = 10_000_000,
                          n_mcmc_samples: int = 3000,
                          mcmc_burn_in: int = 500) -> list:
    """
    融合估算：K-means + MCMC + 确定性优化
    
    返回包含两种方法结果的列表
    """
    n = len(week_data)
    names = week_data['celebrity_name'].values
    J = week_data[f'week{week}_total'].values
    J_sum = J.sum()
    J_pct = J / J_sum if J_sum > 0 else np.ones(n) / n
    
    e_idx = np.where(names == eliminated)[0]
    if len(e_idx) == 0:
        return None
    e_idx = e_idx[0]
    
    # 1. K-means生成Dirichlet先验
    alpha = clusterer.get_dirichlet_alpha(list(names), season, base_alpha=3.0)
    
    # 2. MCMC采样
    mcmc_model = TruncatedBayesianVoteModel(J_pct, e_idx, alpha, total_votes)
    mcmc_model.run_mcmc(n_samples=n_mcmc_samples, burn_in=mcmc_burn_in, 
                        thin=2, step_size=0.03)
    mcmc_stats = mcmc_model.get_posterior_stats()
    
    # 3. 确定性优化
    det_votes = estimate_votes_deterministic(week_data, eliminated, week, 
                                              clusterer, season, total_votes)
    
    # 4. 融合结果
    # 加权平均：MCMC后验均值 + 确定性优化结果
    # 权重可根据置信度调整
    mcmc_votes = mcmc_stats['mean_votes']
    
    # 简单加权融合
    hybrid_votes = 0.6 * mcmc_votes + 0.4 * det_votes
    hybrid_votes = hybrid_votes / hybrid_votes.sum() * total_votes
    
    # 计算综合得分
    hybrid_pct = hybrid_votes / total_votes
    S_hybrid = J_pct + hybrid_pct
    
    mcmc_pct = mcmc_votes / total_votes
    S_mcmc = J_pct + mcmc_pct
    
    det_pct = det_votes / total_votes
    S_det = J_pct + det_pct
    
    # 构造结果
    results = []
    sorted_indices = sorted(range(n), key=lambda x: S_hybrid[x])
    
    for i in range(n):
        is_elim = (i == e_idx)
        
        # 获取人气等级
        pop_level = 'Unknown'
        mask = (clusterer.df['celebrity_name'] == names[i]) & \
               (clusterer.df['season'] == season)
        if mask.sum() > 0:
            pop_level = clusterer.df.loc[mask, 'popularity_level'].values[0]
        
        # 置信度：基于MCMC后验标准差
        # 标准差越小，置信度越高
        relative_std = mcmc_stats['std_pct'][i] / mcmc_stats['mean_pct'][i] if mcmc_stats['mean_pct'][i] > 0 else 1
        mcmc_confidence = max(0, 1 - relative_std)
        
        # 两种方法一致性
        method_agreement = 1 - abs(mcmc_pct[i] - det_pct[i]) / max(mcmc_pct[i], det_pct[i], 0.01)
        
        # 综合置信度
        overall_conf = 0.5 * mcmc_confidence + 0.3 * method_agreement + 0.2 * (0.7 if is_elim else 0.5)
        overall_conf = max(0.1, min(0.95, overall_conf))
        
        conf_level = 'High' if overall_conf > 0.6 else ('Medium' if overall_conf > 0.4 else 'Low')
        
        results.append({
            'celebrity_name': names[i],
            'judge_score': J[i],
            'judge_percentage': round(J_pct[i] * 100, 2),
            'popularity_level': pop_level,
            
            # MCMC结果
            'mcmc_votes': int(mcmc_votes[i]),
            'mcmc_percentage': round(mcmc_pct[i] * 100, 2),
            'mcmc_ci_lower': int(mcmc_stats['ci_lower_votes'][i]),
            'mcmc_ci_upper': int(mcmc_stats['ci_upper_votes'][i]),
            'mcmc_std': int(mcmc_stats['std_votes'][i]),
            
            # 确定性优化结果
            'det_votes': int(det_votes[i]),
            'det_percentage': round(det_pct[i] * 100, 2),
            
            # 融合结果
            'hybrid_votes': int(hybrid_votes[i]),
            'hybrid_percentage': round(hybrid_pct[i] * 100, 2),
            'combined_score': round(S_hybrid[i] * 100, 2),
            
            # 置信度
            'mcmc_confidence': round(mcmc_confidence * 100, 1),
            'method_agreement': round(method_agreement * 100, 1),
            'overall_confidence': round(overall_conf * 100, 1),
            'confidence_level': conf_level,
            
            'is_eliminated': is_elim
        })
    
    # 添加MCMC诊断信息
    results[0]['mcmc_acceptance_rate'] = round(mcmc_stats['acceptance_rate'] * 100, 1)
    
    return results


# ==============================================================================
# 6. 一致性验证
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
# 7. 主程序
# ==============================================================================

def main():
    # 数据路径
    DATA_PATH = "/Users/fangyu/Documents/com/ms/code/-/Data.csv"
    OUTPUT_PATH = "/Users/fangyu/Documents/com/ms/code/-/Task1_Results/fan_vote_hybrid_mcmc.csv"
    
    print("=" * 80)
    print("Task 1: 融合模型 (K-means + 区间截尾贝叶斯 + MCMC + 确定性优化)")
    print("=" * 80)
    
    # 加载数据
    print("\n[1] 加载数据...")
    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        print(f"    错误: 找不到数据文件 {DATA_PATH}")
        print("    请修改 DATA_PATH 为正确的路径")
        return
    
    seasons = sorted(df['season'].unique())
    print(f"    共 {len(seasons)} 个赛季, {len(df)} 位选手")
    
    # K-means聚类
    print("\n[2] K-means选手聚类...")
    clusterer = ContestantClusterer(df, n_clusters=3)
    clusterer.fit()
    
    print("\n    【聚类结果统计】")
    for cluster_id, stats in clusterer.cluster_stats.items():
        pop_level = clusterer.popularity_mapping[cluster_id]
        print(f"    {pop_level} (簇{cluster_id}): {stats['count']}人, 平均名次={stats['avg_placement']:.2f}")
    
    # 估算所有赛季
    print("\n[3] 融合估算 (MCMC + 确定性优化)...")
    print("    注意: MCMC采样需要一些时间，请耐心等待...")
    
    all_rows = []
    total_weeks = 0
    consistent_weeks = 0
    conf_stats = {'High': 0, 'Medium': 0, 'Low': 0}
    method_agreements = []
    
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
            
            # 融合估算
            results = estimate_votes_hybrid(week_data, eliminated, week, 
                                            clusterer, season,
                                            n_mcmc_samples=2000, mcmc_burn_in=300)
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
                method_agreements.append(r['method_agreement'])
                all_rows.append(r)
        
        print(f"    Season {season:2d}: {season_week_count} 周已估算")
    
    # 保存结果
    print(f"\n[4] 保存结果...")
    result_df = pd.DataFrame(all_rows)
    
    cols = ['season', 'week', 'celebrity_name', 'popularity_level',
            'judge_score', 'judge_percentage',
            'mcmc_votes', 'mcmc_percentage', 'mcmc_ci_lower', 'mcmc_ci_upper', 'mcmc_std',
            'det_votes', 'det_percentage',
            'hybrid_votes', 'hybrid_percentage', 'combined_score',
            'mcmc_confidence', 'method_agreement', 'overall_confidence', 'confidence_level',
            'is_eliminated', 'consistency_check']
    
    # 只保留存在的列
    cols = [c for c in cols if c in result_df.columns]
    result_df = result_df[cols]
    result_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    # 输出统计
    print("\n" + "=" * 80)
    print("结果统计")
    print("=" * 80)
    
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
    
    avg_agreement = np.mean(method_agreements) if method_agreements else 0
    print(f"\n【方法一致性】")
    print(f"  MCMC vs 确定性优化 平均一致度: {avg_agreement:.1f}%")
    
    print(f"\n结果已保存: {OUTPUT_PATH}")
    
    # 方法说明
    print("\n" + "=" * 80)
    print("方法说明")
    print("=" * 80)
    print("""
【融合模型架构】

┌─────────────────┐
│   K-means聚类   │ ──→ 人气等级 (High/Medium/Low)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dirichlet先验 α │ ──→ α_i = base_α × popularity_prior_i
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌─────────────┐
│ MCMC  │  │ 确定性优化  │
│ 采样  │  │ (最大熵)    │
└───┬───┘  └──────┬──────┘
    │             │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  融合结果   │ ──→ 0.6×MCMC + 0.4×确定性
    └─────────────┘

【区间截尾贝叶斯模型】
θ ~ Dirichlet(α)
约束: S[elim] < S[i] for all i ≠ elim
其中 S[i] = J_pct[i] + θ[i]

【MCMC采样】
- 提议分布: 对数正态随机游走
- 接受准则: Metropolis-Hastings
- 输出: 后验均值、标准差、95%可信区间

【优势】
1. K-means提供数据驱动的先验
2. MCMC提供完整的不确定性量化
3. 区间截尾自然处理淘汰约束
4. 融合提高鲁棒性
5. 方法一致性作为额外验证
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
