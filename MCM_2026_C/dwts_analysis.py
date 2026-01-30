# -*- coding: utf-8 -*-
"""
2026 MCM Problem C: Dancing with the Stars 数据分析
===============================================

本代码实现：
1. 数据读取与预处理
2. 粉丝投票估算模型（基于约束优化）
3. 投票机制对比分析
4. 可视化图表

作者: MCM Team
日期: 2026-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 第一部分：数据读取与预处理
# ============================================================================

class DWTSDataProcessor:
    """Dancing with the Stars 数据处理器"""
    
    def __init__(self, csv_path: str):
        """
        初始化数据处理器
        
        Parameters:
        -----------
        csv_path : str
            CSV数据文件路径
        """
        self.raw_data = pd.read_csv(csv_path)
        self.processed_data = None
        self.seasons_data = {}
        
    def preprocess(self) -> pd.DataFrame:
        """
        数据预处理主函数
        
        Returns:
        --------
        pd.DataFrame : 预处理后的数据
        """
        df = self.raw_data.copy()
        
        # 1. 标准化列名
        df.columns = df.columns.str.strip().str.lower()
        
        # 2. 处理缺失值和N/A
        df = df.replace('N/A', np.nan)
        
        # 3. 计算每周评委总分
        df = self._calculate_weekly_scores(df)
        
        # 4. 提取淘汰周信息
        df = self._extract_elimination_week(df)
        
        # 5. 数据类型转换
        df['season'] = df['season'].astype(int)
        df['placement'] = df['placement'].astype(int)
        
        self.processed_data = df
        return df
    
    def _calculate_weekly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算每周评委总分"""
        for week in range(1, 12):  # 最多11周
            judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
            existing_cols = [col for col in judge_cols if col in df.columns]
            
            if existing_cols:
                # 将字符串转换为数值
                for col in existing_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 计算该周总分（忽略N/A）
                df[f'week{week}_total'] = df[existing_cols].sum(axis=1, skipna=True)
                
                # 计算该周平均分
                df[f'week{week}_avg'] = df[existing_cols].mean(axis=1, skipna=True)
                
                # 统计该周有效评委数
                df[f'week{week}_judge_count'] = df[existing_cols].notna().sum(axis=1)
        
        return df
    
    def _extract_elimination_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """从results列提取淘汰周数"""
        def parse_elimination(result: str) -> int:
            """解析淘汰周数，返回-1表示进入决赛或退赛"""
            if pd.isna(result):
                return -1
            result = str(result).lower()
            if 'week' in result:
                try:
                    # 提取 "Eliminated Week X" 中的数字
                    parts = result.split('week')
                    if len(parts) > 1:
                        num = ''.join(filter(str.isdigit, parts[1]))
                        return int(num) if num else -1
                except:
                    return -1
            elif 'place' in result or 'withdrew' in result:
                return -1  # 进入决赛或主动退赛
            return -1
        
        df['elimination_week'] = df['results'].apply(parse_elimination)
        return df
    
    def get_season_data(self, season: int) -> pd.DataFrame:
        """获取指定赛季的数据"""
        if self.processed_data is None:
            self.preprocess()
        return self.processed_data[self.processed_data['season'] == season].copy()
    
    def get_week_data(self, season: int, week: int) -> pd.DataFrame:
        """
        获取指定赛季、指定周的在场选手数据
        
        Parameters:
        -----------
        season : int
            赛季号
        week : int
            周数
            
        Returns:
        --------
        pd.DataFrame : 该周仍在场的选手数据
        """
        season_df = self.get_season_data(season)
        
        # 筛选该周仍在场的选手（未被淘汰或淘汰周>=当前周）
        active_mask = (
            (season_df['elimination_week'] == -1) |  # 进入决赛
            (season_df['elimination_week'] >= week)   # 还未被淘汰
        )
        
        # 同时要求该周有有效评分
        score_col = f'week{week}_total'
        if score_col in season_df.columns:
            active_mask &= (season_df[score_col] > 0)
        
        return season_df[active_mask].copy()
    
    def get_eliminated_contestant(self, season: int, week: int) -> Optional[str]:
        """获取指定赛季、指定周被淘汰的选手"""
        season_df = self.get_season_data(season)
        eliminated = season_df[season_df['elimination_week'] == week]
        if len(eliminated) > 0:
            return eliminated.iloc[0]['celebrity_name']
        return None
    
    def get_all_seasons(self) -> List[int]:
        """获取所有赛季列表"""
        if self.processed_data is None:
            self.preprocess()
        return sorted(self.processed_data['season'].unique().tolist())
    
    def summary(self) -> Dict:
        """输出数据摘要"""
        if self.processed_data is None:
            self.preprocess()
        
        df = self.processed_data
        summary = {
            'total_contestants': len(df),
            'total_seasons': df['season'].nunique(),
            'industries': df['celebrity_industry'].value_counts().to_dict(),
            'avg_age': df['celebrity_age_during_season'].mean(),
            'top_partners': df['ballroom_partner'].value_counts().head(10).to_dict()
        }
        return summary


# ============================================================================
# 第二部分：粉丝投票估算模型
# ============================================================================

class FanVoteEstimator:
    """
    粉丝投票估算器
    
    核心思想：
    - 粉丝投票是未知的，但我们知道每周淘汰的选手
    - 淘汰规则：综合得分最低的选手被淘汰
    - 通过逆向推理，找出满足淘汰结果的粉丝投票分布
    """
    
    def __init__(self, processor: DWTSDataProcessor):
        self.processor = processor
        self.estimated_votes = {}  # 存储估算结果
        
    def estimate_season(self, season: int, method: str = 'percentage') -> Dict:
        """
        估算一个赛季的粉丝投票
        
        Parameters:
        -----------
        season : int
            赛季号
        method : str
            投票机制 ('ranking' 或 'percentage')
            
        Returns:
        --------
        Dict : 每周每位选手的估算投票数
        """
        season_df = self.processor.get_season_data(season)
        results = {}
        
        # 确定该赛季的周数
        max_week = self._get_max_week(season_df)
        
        for week in range(1, max_week + 1):
            week_data = self.processor.get_week_data(season, week)
            eliminated = self.processor.get_eliminated_contestant(season, week)
            
            if len(week_data) < 2 or eliminated is None:
                continue
            
            # 估算该周的粉丝投票
            if method == 'percentage':
                votes = self._estimate_percentage_method(week_data, eliminated, week)
            else:
                votes = self._estimate_ranking_method(week_data, eliminated, week)
            
            results[week] = votes
        
        self.estimated_votes[season] = results
        return results
    
    def _get_max_week(self, season_df: pd.DataFrame) -> int:
        """获取赛季的最大周数"""
        for week in range(11, 0, -1):
            col = f'week{week}_total'
            if col in season_df.columns:
                if (season_df[col] > 0).any():
                    return week
        return 6  # 默认值
    
    def _estimate_percentage_method(self, week_data: pd.DataFrame, 
                                     eliminated: str, week: int) -> Dict:
        """
        百分比制下估算粉丝投票
        
        规则：综合百分比 = 评委得分百分比 + 粉丝投票百分比
        被淘汰者的综合百分比最低
        """
        n = len(week_data)
        score_col = f'week{week}_total'
        
        # 获取评委总分
        judge_scores = week_data[score_col].values
        total_judge = judge_scores.sum()
        
        if total_judge == 0:
            return {row['celebrity_name']: 0 for _, row in week_data.iterrows()}
        
        # 评委得分百分比
        judge_pct = judge_scores / total_judge
        
        # 找到被淘汰选手的索引
        names = week_data['celebrity_name'].values
        elim_idx = np.where(names == eliminated)[0]
        
        if len(elim_idx) == 0:
            # 找不到被淘汰选手，返回均匀分布
            return {name: 1.0 / n for name in names}
        
        elim_idx = elim_idx[0]
        
        # 构造满足约束的粉丝投票
        # 约束：被淘汰者的综合百分比 < 其他所有人
        # 即：judge_pct[elim] + fan_pct[elim] < judge_pct[i] + fan_pct[i] for all i
        
        # 使用简化假设：根据评委得分的差距来分配粉丝票
        # 被淘汰者获得最少的粉丝票
        
        fan_votes = np.zeros(n)
        base_votes = 1_000_000  # 基准投票数 100万
        
        for i in range(n):
            if i == elim_idx:
                # 被淘汰者获得最低票数
                fan_votes[i] = base_votes * 0.5
            else:
                # 其他人根据评委得分比例获得投票，但确保高于被淘汰者
                fan_votes[i] = base_votes * (1.0 + (judge_pct[i] - judge_pct[elim_idx]))
        
        # 归一化使总票数合理
        fan_votes = fan_votes / fan_votes.sum() * base_votes * n
        
        return {names[i]: fan_votes[i] for i in range(n)}
    
    def _estimate_ranking_method(self, week_data: pd.DataFrame,
                                  eliminated: str, week: int) -> Dict:
        """
        排名制下估算粉丝投票
        
        规则：评委排名 + 粉丝排名，排名和最大者淘汰
        """
        n = len(week_data)
        score_col = f'week{week}_total'
        
        # 获取评委排名（得分越高排名越靠前）
        judge_scores = week_data[score_col].values
        names = week_data['celebrity_name'].values
        
        # 评委排名（1=最高分）
        judge_rank = n - np.argsort(np.argsort(judge_scores))
        
        # 找到被淘汰选手
        elim_idx = np.where(names == eliminated)[0]
        if len(elim_idx) == 0:
            return {name: 1.0 / n for name in names}
        elim_idx = elim_idx[0]
        
        # 被淘汰者的排名和必须最大
        elim_judge_rank = judge_rank[elim_idx]
        
        # 估算粉丝排名：让被淘汰者的粉丝排名也较低
        fan_rank = np.zeros(n)
        fan_votes = np.zeros(n)
        base_votes = 1_000_000
        
        for i in range(n):
            if i == elim_idx:
                fan_rank[i] = n  # 粉丝排名最后
                fan_votes[i] = base_votes * 0.3
            else:
                # 其他人获得更好的粉丝排名
                fan_rank[i] = judge_rank[i]  # 假设粉丝投票与评委类似
                fan_votes[i] = base_votes * (1.2 - 0.1 * judge_rank[i])
        
        return {names[i]: fan_votes[i] for i in range(n)}
    
    def validate_estimation(self, season: int, method: str = 'percentage') -> Dict:
        """
        验证估算结果是否与实际淘汰结果一致
        
        Returns:
        --------
        Dict : 验证结果，包括一致性分数
        """
        if season not in self.estimated_votes:
            self.estimate_season(season, method)
        
        season_df = self.processor.get_season_data(season)
        estimates = self.estimated_votes[season]
        
        correct = 0
        total = 0
        details = []
        
        for week, votes in estimates.items():
            actual_eliminated = self.processor.get_eliminated_contestant(season, week)
            if actual_eliminated is None:
                continue
            
            # 根据估算投票计算谁应该被淘汰
            week_data = self.processor.get_week_data(season, week)
            predicted_eliminated = self._predict_elimination(
                week_data, votes, week, method
            )
            
            is_correct = (predicted_eliminated == actual_eliminated)
            correct += int(is_correct)
            total += 1
            
            details.append({
                'week': week,
                'actual': actual_eliminated,
                'predicted': predicted_eliminated,
                'correct': is_correct
            })
        
        return {
            'season': season,
            'accuracy': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total,
            'details': details
        }
    
    def _predict_elimination(self, week_data: pd.DataFrame, votes: Dict,
                             week: int, method: str) -> str:
        """根据估算投票预测谁被淘汰"""
        score_col = f'week{week}_total'
        
        combined_scores = {}
        total_judge = week_data[score_col].sum()
        total_votes = sum(votes.values())
        
        for _, row in week_data.iterrows():
            name = row['celebrity_name']
            judge_score = row[score_col]
            fan_vote = votes.get(name, 0)
            
            if method == 'percentage':
                judge_pct = judge_score / total_judge if total_judge > 0 else 0
                fan_pct = fan_vote / total_votes if total_votes > 0 else 0
                combined_scores[name] = judge_pct + fan_pct
            else:
                combined_scores[name] = judge_score + fan_vote
        
        # 返回得分最低的选手
        return min(combined_scores, key=combined_scores.get)


# ============================================================================
# 第三部分：投票机制对比分析
# ============================================================================

class VotingMechanismAnalyzer:
    """投票机制对比分析器"""
    
    def __init__(self, processor: DWTSDataProcessor, estimator: FanVoteEstimator):
        self.processor = processor
        self.estimator = estimator
        
    def compare_mechanisms(self, season: int) -> pd.DataFrame:
        """
        比较排名制和百分比制在同一赛季的不同结果
        
        Returns:
        --------
        pd.DataFrame : 对比结果表
        """
        # 估算两种机制下的投票
        votes_pct = self.estimator.estimate_season(season, 'percentage')
        votes_rank = self.estimator.estimate_season(season, 'ranking')
        
        results = []
        season_df = self.processor.get_season_data(season)
        max_week = self.estimator._get_max_week(season_df)
        
        for week in range(1, max_week + 1):
            week_data = self.processor.get_week_data(season, week)
            actual_eliminated = self.processor.get_eliminated_contestant(season, week)
            
            if actual_eliminated is None or len(week_data) < 2:
                continue
            
            # 百分比制预测
            if week in votes_pct:
                pred_pct = self.estimator._predict_elimination(
                    week_data, votes_pct[week], week, 'percentage'
                )
            else:
                pred_pct = None
            
            # 排名制预测
            if week in votes_rank:
                pred_rank = self.estimator._predict_elimination(
                    week_data, votes_rank[week], week, 'ranking'
                )
            else:
                pred_rank = None
            
            results.append({
                'week': week,
                'actual_eliminated': actual_eliminated,
                'percentage_method_pred': pred_pct,
                'ranking_method_pred': pred_rank,
                'methods_agree': pred_pct == pred_rank,
                'pct_correct': pred_pct == actual_eliminated,
                'rank_correct': pred_rank == actual_eliminated
            })
        
        return pd.DataFrame(results)
    
    def analyze_controversial_cases(self) -> pd.DataFrame:
        """
        分析四个争议案例
        
        争议案例：
        1. 第2季 Jerry Rice
        2. 第4季 Billy Ray Cyrus  
        3. 第11季 Bristol Palin
        4. 第27季 Bobby Bones
        """
        cases = [
            {'season': 2, 'name': 'Jerry Rice', 'description': '评委垫底5周仍获亚军'},
            {'season': 4, 'name': 'Billy Ray Cyrus', 'description': '评委垫底6周仍获第5'},
            {'season': 11, 'name': 'Bristol Palin', 'description': '12次评委最低仍获第3'},
            {'season': 27, 'name': 'Bobby Bones', 'description': '评委一直最低仍夺冠'},
        ]
        
        results = []
        for case in cases:
            season = case['season']
            name = case['name']
            
            season_df = self.processor.get_season_data(season)
            contestant = season_df[season_df['celebrity_name'] == name]
            
            if len(contestant) == 0:
                continue
            
            contestant = contestant.iloc[0]
            
            # 计算该选手每周的评委排名
            max_week = self.estimator._get_max_week(season_df)
            weekly_ranks = []
            
            for week in range(1, max_week + 1):
                week_data = self.processor.get_week_data(season, week)
                score_col = f'week{week}_total'
                
                if score_col not in week_data.columns:
                    continue
                
                if name not in week_data['celebrity_name'].values:
                    break
                
                scores = week_data[score_col].values
                names = week_data['celebrity_name'].values
                
                # 计算排名
                sorted_idx = np.argsort(scores)[::-1]
                rank = np.where(names[sorted_idx] == name)[0]
                if len(rank) > 0:
                    weekly_ranks.append(rank[0] + 1)  # 1-indexed
            
            results.append({
                'season': season,
                'celebrity': name,
                'description': case['description'],
                'final_placement': contestant['placement'],
                'avg_judge_rank': np.mean(weekly_ranks) if weekly_ranks else None,
                'times_ranked_last': sum(1 for r in weekly_ranks if r == max(weekly_ranks)),
                'weeks_survived': len(weekly_ranks)
            })
        
        return pd.DataFrame(results)


# ============================================================================
# 第四部分：可视化
# ============================================================================

class DWTSVisualizer:
    """可视化工具类"""
    
    def __init__(self, processor: DWTSDataProcessor):
        self.processor = processor
        
    def plot_season_scores(self, season: int, save_path: str = None):
        """
        绘制某赛季各选手的评委得分趋势
        """
        season_df = self.processor.get_season_data(season)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for _, row in season_df.iterrows():
            name = row['celebrity_name']
            scores = []
            weeks = []
            
            for week in range(1, 12):
                col = f'week{week}_total'
                if col in row and pd.notna(row[col]) and row[col] > 0:
                    scores.append(row[col])
                    weeks.append(week)
            
            if scores:
                ax.plot(weeks, scores, 'o-', label=name, alpha=0.7)
        
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Total Judge Score', fontsize=12)
        ax.set_title(f'Season {season}: Weekly Judge Scores by Contestant', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_industry_distribution(self, save_path: str = None):
        """
        绘制选手行业分布饼图
        """
        df = self.processor.processed_data
        industry_counts = df['celebrity_industry'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(industry_counts)))
        wedges, texts, autotexts = ax.pie(
            industry_counts.values,
            labels=industry_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        ax.set_title('Distribution of Celebrity Industries', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_partner_success(self, save_path: str = None):
        """
        绘制专业舞伴的成功率（平均最终排名）
        """
        df = self.processor.processed_data
        
        partner_stats = df.groupby('ballroom_partner').agg({
            'placement': ['mean', 'count', 'min']
        }).reset_index()
        partner_stats.columns = ['partner', 'avg_placement', 'appearances', 'best_placement']
        
        # 只显示出场次数>=3的舞伴
        partner_stats = partner_stats[partner_stats['appearances'] >= 3]
        partner_stats = partner_stats.sort_values('avg_placement')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(partner_stats['partner'], partner_stats['avg_placement'], 
                       color='steelblue', alpha=0.7)
        
        # 添加冠军次数标注
        for i, (_, row) in enumerate(partner_stats.iterrows()):
            if row['best_placement'] == 1:
                ax.annotate('🏆', (row['avg_placement'] + 0.1, i), fontsize=12)
        
        ax.set_xlabel('Average Final Placement (lower is better)', fontsize=12)
        ax.set_ylabel('Professional Partner', fontsize=12)
        ax.set_title('Professional Partner Success Rate', fontsize=14)
        ax.invert_xaxis()  # 反转x轴，让排名1在右边
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_estimated_votes(self, season: int, estimator: FanVoteEstimator,
                             save_path: str = None):
        """
        绘制某赛季估算的粉丝投票分布
        """
        if season not in estimator.estimated_votes:
            estimator.estimate_season(season)
        
        votes = estimator.estimated_votes[season]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (week, week_votes) in enumerate(list(votes.items())[:6]):
            ax = axes[idx]
            
            names = list(week_votes.keys())
            values = list(week_votes.values())
            
            # 截断长名字
            short_names = [n[:15] + '...' if len(n) > 15 else n for n in names]
            
            bars = ax.barh(short_names, values, color='coral', alpha=0.7)
            ax.set_xlabel('Estimated Fan Votes')
            ax.set_title(f'Week {week}')
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        plt.suptitle(f'Season {season}: Estimated Fan Votes by Week', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_mechanism_comparison(self, analyzer: VotingMechanismAnalyzer,
                                  seasons: List[int], save_path: str = None):
        """
        绘制两种投票机制的对比结果
        """
        all_results = []
        
        for season in seasons:
            comparison = analyzer.compare_mechanisms(season)
            comparison['season'] = season
            all_results.append(comparison)
        
        combined = pd.concat(all_results, ignore_index=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 图1：两种方法的准确率对比
        ax1 = axes[0]
        accuracy_by_season = combined.groupby('season').agg({
            'pct_correct': 'mean',
            'rank_correct': 'mean'
        })
        
        x = np.arange(len(accuracy_by_season))
        width = 0.35
        
        ax1.bar(x - width/2, accuracy_by_season['pct_correct'], width, 
                label='Percentage Method', color='steelblue', alpha=0.7)
        ax1.bar(x + width/2, accuracy_by_season['rank_correct'], width,
                label='Ranking Method', color='coral', alpha=0.7)
        
        ax1.set_xlabel('Season')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Prediction Accuracy by Voting Method')
        ax1.set_xticks(x)
        ax1.set_xticklabels(accuracy_by_season.index)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # 图2：两种方法是否一致
        ax2 = axes[1]
        agreement = combined.groupby('season')['methods_agree'].mean()
        ax2.bar(agreement.index, agreement.values, color='green', alpha=0.7)
        ax2.set_xlabel('Season')
        ax2.set_ylabel('Agreement Rate')
        ax2.set_title('Agreement Between Two Methods')
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    
    # 1. 数据路径（请根据实际情况修改）
    DATA_PATH = r"d:\Data\xwechat_files\wxid_m7pucc5xg5m522_f378\msg\file\2026-01\中文版赛题 (1)\中文版赛题\2026_MCM_Problem_C_Data.csv"
    
    print("=" * 60)
    print("2026 MCM Problem C: Dancing with the Stars 数据分析")
    print("=" * 60)
    
    # 2. 数据预处理
    print("\n[Step 1] 数据加载与预处理...")
    processor = DWTSDataProcessor(DATA_PATH)
    df = processor.preprocess()
    
    summary = processor.summary()
    print(f"  - 总选手数: {summary['total_contestants']}")
    print(f"  - 总赛季数: {summary['total_seasons']}")
    print(f"  - 平均年龄: {summary['avg_age']:.1f}")
    print(f"  - 行业分布: {list(summary['industries'].keys())[:5]}...")
    
    # 3. 粉丝投票估算
    print("\n[Step 2] 粉丝投票估算...")
    estimator = FanVoteEstimator(processor)
    
    # 估算几个关键赛季
    key_seasons = [2, 4, 11, 27]  # 争议赛季
    for season in key_seasons:
        if season in processor.get_all_seasons():
            votes = estimator.estimate_season(season, 'percentage')
            validation = estimator.validate_estimation(season, 'percentage')
            print(f"  - Season {season}: 准确率 = {validation['accuracy']:.2%}")
    
    # 4. 投票机制对比
    print("\n[Step 3] 投票机制对比分析...")
    analyzer = VotingMechanismAnalyzer(processor, estimator)
    
    # 分析争议案例
    controversial = analyzer.analyze_controversial_cases()
    print("  争议案例分析:")
    for _, row in controversial.iterrows():
        print(f"    - Season {row['season']} {row['celebrity']}: "
              f"最终排名={row['final_placement']}, 平均评委排名={row['avg_judge_rank']:.1f}")
    
    # 5. 可视化
    print("\n[Step 4] 生成可视化图表...")
    visualizer = DWTSVisualizer(processor)
    
    # 创建输出目录
    import os
    output_dir = r"d:\桌面\代码仓库\数据结构\data-structure-learning\MCM_2026_C\figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成图表
    visualizer.plot_industry_distribution(f"{output_dir}/industry_distribution.png")
    visualizer.plot_partner_success(f"{output_dir}/partner_success.png")
    
    if 2 in processor.get_all_seasons():
        visualizer.plot_season_scores(2, f"{output_dir}/season2_scores.png")
        visualizer.plot_estimated_votes(2, estimator, f"{output_dir}/season2_votes.png")
    
    print(f"  图表已保存到: {output_dir}")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    return processor, estimator, analyzer, visualizer


if __name__ == "__main__":
    processor, estimator, analyzer, visualizer = main()
