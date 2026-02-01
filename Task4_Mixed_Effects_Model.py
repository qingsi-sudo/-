"""
================================================================================
2026 MCM Problem C - Task 4: Impact Analysis (最终修复版)
================================================================================
修复:
1. 舞者随机效应提取兼容性
2. TWFE参数错误
3. 添加舞者固定效应作为替代分析
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    print("Warning: linearmodels not installed. TWFE analysis will be skipped.")
    HAS_LINEARMODELS = False


# ==============================================================================
# 1. 数据准备
# ==============================================================================

class DataPreparator:
    """数据预处理类"""
    
    def __init__(self, fan_vote_path: str, original_data_path: str):
        self.fan_vote_df = pd.read_csv(fan_vote_path)
        self.original_df = pd.read_csv(original_data_path).replace('N/A', np.nan)
        
    def merge_data(self) -> pd.DataFrame:
        """合并粉丝投票估计和原始数据"""
        feature_cols = ['celebrity_name', 'season', 'ballroom_partner', 
                        'celebrity_industry', 'celebrity_age_during_season',
                        'celebrity_homestate', 'celebrity_homecountry/region']
        
        features = self.original_df[feature_cols].drop_duplicates()
        merged = self.fan_vote_df.merge(features, on=['celebrity_name', 'season'], how='left')
        return merged
    
    def encode_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """编码类别变量"""
        df = df.copy()
        
        industry_order = {
            'Singer/Rapper': 5, 'Actor/Actress': 4, 'Athlete': 4,
            'Olympian': 4, 'TV Personality': 3, 'Model': 3,
            'YouTube/TikTok': 5, 'Comedian': 3, 'News Anchor': 2, 'Other': 2
        }
        
        def encode_industry(ind):
            if pd.isna(ind):
                return 2
            for key, val in industry_order.items():
                if key.lower() in str(ind).lower():
                    return val
            return 2
        
        df['industry_score'] = df['celebrity_industry'].apply(encode_industry)
        df['age'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].median())
        df['age_centered'] = df['age'] - df['age'].mean()
        df['dancer_encoded'] = pd.Categorical(df['ballroom_partner']).codes
        df['week'] = df['week'].astype(int)
        df['season'] = df['season'].astype(int)
        
        return df
    
    def create_analysis_dataset(self) -> pd.DataFrame:
        """创建完整的分析数据集"""
        merged = self.merge_data()
        encoded = self.encode_variables(merged)
        
        valid = encoded[
            (encoded['judge_score'] > 0) &
            (encoded['hybrid_votes'] > 0) &
            (~encoded['age'].isna())
        ].copy()
        
        valid['log_judge_score'] = np.log(valid['judge_score'] + 1)
        valid['log_fan_votes'] = np.log(valid['hybrid_votes'] + 1)
        
        return valid


# ==============================================================================
# 2. OLS分析
# ==============================================================================

class OLSAnalysis:
    """普通最小二乘回归分析"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        
    def run_ols(self, outcome: str, name: str):
        """运行OLS回归（包含年龄二次项）"""
        formula = f"{outcome} ~ age_centered + I(age_centered**2) + industry_score + dancer_encoded + C(season) + week"
        model = smf.ols(formula, data=self.df).fit(
            cov_type='cluster',
            cov_kwds={'groups': self.df['celebrity_name']}
        )
        self.results[name] = model
        return model
    
    def run_all(self):
        """运行所有OLS模型"""
        print("\n" + "="*80)
        print("探索性分析: OLS回归 (聚类稳健标准误)")
        print("="*80)
        
        print("\n[模型1] 评委分数 ~ 年龄 + 年龄² + 名人特征 + 舞者 + 控制变量")
        result_judge = self.run_ols('judge_score', 'judge_ols')
        print(result_judge.summary())
        
        print("\n[模型2] 粉丝投票 ~ 年龄 + 年龄² + 名人特征 + 舞者 + 控制变量")
        result_fan = self.run_ols('hybrid_percentage', 'fan_ols')
        print(result_fan.summary())
        
        return self.results
    
    def check_multicollinearity(self):
        """检查多重共线性"""
        print("\n【多重共线性检验 - VIF】")
        self.df['age_centered_sq'] = self.df['age_centered'] ** 2
        X = self.df[['age_centered', 'age_centered_sq', 'industry_score', 'dancer_encoded', 'week']]
        X = sm.add_constant(X)
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        print(vif_data)
        print("注: VIF < 10 表示无严重共线性")


# ==============================================================================
# 3. 混合效应模型
# ==============================================================================

class MixedEffectsAnalysis:
    """混合效应模型分析"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        self.dancer_effects = {}
        
    def run_mixed_model(self, outcome: str, name: str):
        """运行混合效应模型（包含年龄二次项）"""
        formula = f"{outcome} ~ age_centered + I(age_centered**2) + industry_score + week + C(season)"
        model = smf.mixedlm(formula, data=self.df, groups=self.df["ballroom_partner"], re_formula="1")
        result = model.fit(method='lbfgs', maxiter=200)
        self.results[name] = result
        return result
    
    def run_all(self):
        """运行所有混合效应模型"""
        print("\n" + "="*80)
        print("主要分析: 混合效应模型")
        print("="*80)
        
        print("\n[模型1] 评委分数 ~ 年龄 + 年龄² + 名人特征 + (1|舞者) + 控制变量")
        result_judge = self.run_mixed_model('judge_score', 'judge_mem')
        print(result_judge.summary())
        
        print("\n[模型2] 粉丝投票 ~ 年龄 + 年龄² + 名人特征 + (1|舞者) + 控制变量")
        result_fan = self.run_mixed_model('hybrid_percentage', 'fan_mem')
        print(result_fan.summary())
        
        return self.results
    
    def extract_dancer_effects(self):
        """提取舞者随机效应 - 兼容版本"""
        print("\n【舞者随机效应排名】")
        
        for outcome_name, result in self.results.items():
            effects = []
            for dancer, effect_val in result.random_effects.items():
                if isinstance(effect_val, pd.Series):
                    effect = effect_val.iloc[0] if len(effect_val) > 0 else 0.0
                elif isinstance(effect_val, (list, np.ndarray)):
                    effect = effect_val[0] if len(effect_val) > 0 else 0.0
                else:
                    effect = float(effect_val)
                
                effects.append({'dancer': dancer, 'effect': effect})
            
            df_effects = pd.DataFrame(effects).sort_values('effect', ascending=False)
            self.dancer_effects[outcome_name] = df_effects
            
            print(f"\n{outcome_name.upper()} - Top 10 舞者:")
            print(df_effects.head(10).to_string(index=False))
        
        return self.dancer_effects
    
    def calculate_icc(self):
        """计算组内相关系数 (ICC)"""
        print("\n【方差分解 - ICC (Intraclass Correlation)】")
        
        for name, result in self.results.items():
            tau_squared = result.cov_re.iloc[0, 0]
            sigma_squared = result.scale
            icc = tau_squared / (tau_squared + sigma_squared)
            
            print(f"\n{name}:")
            print(f"  随机效应方差 (τ²): {tau_squared:.4f}")
            print(f"  残差方差 (σ²):     {sigma_squared:.4f}")
            print(f"  ICC:              {icc:.4f}")
            print(f"  解释: {icc*100:.1f}% 的总方差由舞者间差异解释")


# ==============================================================================
# 4. 舞者固定效应分析（替代TWFE）
# ==============================================================================

class DancerFixedEffectsAnalysis:
    """舞者固定效应分析 - 作为TWFE的替代"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        
    def run_dancer_fe(self, outcome: str, name: str):
        """运行包含舞者固定效应的OLS"""
        # 只使用出现频率高的舞者（至少出现5次），减少虚拟变量数量
        dancer_counts = self.df['ballroom_partner'].value_counts()
        top_dancers = dancer_counts[dancer_counts >= 5].index.tolist()
        
        # 创建舞者分类变量：高频舞者保留，其他归为"Other"
        df_temp = self.df.copy()
        df_temp['dancer_cat'] = df_temp['ballroom_partner'].apply(
            lambda x: x if x in top_dancers else 'Other'
        )
        
        formula = f"{outcome} ~ age_centered + I(age_centered**2) + industry_score + C(dancer_cat) + C(season) + week"
        
        # 使用HC3稳健标准误而不是聚类标准误，避免约束矩阵不满秩警告
        model = smf.ols(formula, data=df_temp).fit(cov_type='hc3')
        
        self.results[name] = model
        return model
    
    def run_all(self):
        """运行所有舞者固定效应模型"""
        print("\n" + "="*80)
        print("稳健性检验: 舞者固定效应模型")
        print("="*80)
        
        print("\n[模型1] 评委分数 ~ 年龄 + 年龄² + 名人特征 + 舞者FE + 控制变量")
        result_judge = self.run_dancer_fe('judge_score', 'judge_dancer_fe')
        print(result_judge.summary())
        
        print("\n[模型2] 粉丝投票 ~ 年龄 + 年龄² + 名人特征 + 舞者FE + 控制变量")
        result_fan = self.run_dancer_fe('hybrid_percentage', 'fan_dancer_fe')
        print(result_fan.summary())
        
        return self.results
    
    def extract_dancer_coefficients(self):
        """提取舞者固定效应系数"""
        print("\n【舞者固定效应系数排名】")
        
        dancer_coefs = {}
        
        for outcome_name, result in self.results.items():
            coefs = []
            for param_name, coef in result.params.items():
                # 更新为dancer_cat
                if 'C(dancer_cat)' in param_name:
                    dancer_name = param_name.replace('C(dancer_cat)[T.', '').replace(']', '')
                    coefs.append({'dancer': dancer_name, 'coefficient': coef})
            
            # 添加基准组（系数为0）
            df_temp = self.df.copy()
            dancer_counts = df_temp['ballroom_partner'].value_counts()
            top_dancers = dancer_counts[dancer_counts >= 5].index.tolist()
            df_temp['dancer_cat'] = df_temp['ballroom_partner'].apply(
                lambda x: x if x in top_dancers else 'Other'
            )
            all_dancer_cats = df_temp['dancer_cat'].unique()
            included_dancers = [c['dancer'] for c in coefs]
            for dancer in all_dancer_cats:
                if dancer not in included_dancers:
                    coefs.append({'dancer': dancer, 'coefficient': 0.0})
            
            df_coefs = pd.DataFrame(coefs).sort_values('coefficient', ascending=False)
            dancer_coefs[outcome_name] = df_coefs
            
            print(f"\n{outcome_name.upper()} - Top 10 舞者:")
            print(df_coefs.head(10).to_string(index=False))
        
        return dancer_coefs


# ==============================================================================
# 5. 差异分析
# ==============================================================================

class DifferentialImpactAnalysis:
    """评委分数 vs 粉丝投票的差异影响分析"""
    
    def __init__(self, judge_results: dict, fan_results: dict):
        self.judge_results = judge_results
        self.fan_results = fan_results
        
    def compare_coefficients(self):
        """对比固定效应系数"""
        print("\n" + "="*80)
        print("系数对比: 评委分数 vs 粉丝投票")
        print("="*80)
        
        judge_mem = self.judge_results.get('judge_mem')
        fan_mem = self.fan_results.get('fan_mem')
        
        if judge_mem is None or fan_mem is None:
            print("Error: Mixed effects results not found")
            return None
        
        # 匹配变量名（statsmodels可能转换I()表达式）
        common_vars = ['age_centered', 'industry_score', 'week']
        age_sq_patterns = ['I(age_centered ** 2)', 'I(age_centered**2)', 'age_centered_sq']
        
        comparison = []
        
        # 先处理普通变量
        for var in common_vars:
            if var in judge_mem.params.index and var in fan_mem.params.index:
                judge_coef = judge_mem.params[var]
                judge_se = judge_mem.bse[var]
                fan_coef = fan_mem.params[var]
                fan_se = fan_mem.bse[var]
                
                diff = fan_coef - judge_coef
                diff_se = np.sqrt(judge_se**2 + fan_se**2)
                z_stat = diff / diff_se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                comparison.append({
                    'Variable': var,
                    'Judge_Coef': judge_coef,
                    'Judge_SE': judge_se,
                    'Fan_Coef': fan_coef,
                    'Fan_SE': fan_se,
                    'Difference': diff,
                    'Diff_SE': diff_se,
                    'Z_stat': z_stat,
                    'P_value': p_value,
                    'Significant': '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.1 else ''))
                })
        
        # 处理年龄二次项（尝试匹配不同的变量名格式）
        age_sq_var = None
        for pattern in age_sq_patterns:
            if pattern in judge_mem.params.index and pattern in fan_mem.params.index:
                age_sq_var = pattern
                break
        
        if age_sq_var:
            judge_coef = judge_mem.params[age_sq_var]
            judge_se = judge_mem.bse[age_sq_var]
            fan_coef = fan_mem.params[age_sq_var]
            fan_se = fan_mem.bse[age_sq_var]
            
            diff = fan_coef - judge_coef
            diff_se = np.sqrt(judge_se**2 + fan_se**2)
            z_stat = diff / diff_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            comparison.append({
                'Variable': 'age_centered^2',
                'Judge_Coef': judge_coef,
                'Judge_SE': judge_se,
                'Fan_Coef': fan_coef,
                'Fan_SE': fan_se,
                'Difference': diff,
                'Diff_SE': diff_se,
                'Z_stat': z_stat,
                'P_value': p_value,
                'Significant': '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.1 else ''))
            })
        
        df_compare = pd.DataFrame(comparison)
        print("\n系数对比表:")
        print(df_compare.to_string(index=False))
        
        return df_compare
    
    def interaction_model(self, df: pd.DataFrame):
        """堆叠数据交互效应模型"""
        print("\n【交互效应模型】")
        
        df_judge = df[['celebrity_name', 'season', 'week', 'age_centered', 
                       'industry_score', 'judge_score', 'ballroom_partner']].copy()
        df_judge['outcome_type'] = 'judge'
        df_judge['outcome'] = df_judge['judge_score']
        
        df_fan = df[['celebrity_name', 'season', 'week', 'age_centered', 
                     'industry_score', 'hybrid_percentage', 'ballroom_partner']].copy()
        df_fan['outcome_type'] = 'fan'
        df_fan['outcome'] = df_fan['hybrid_percentage']
        
        df_stacked = pd.concat([
            df_judge[['celebrity_name', 'season', 'week', 'age_centered', 
                     'industry_score', 'outcome_type', 'outcome', 'ballroom_partner']],
            df_fan[['celebrity_name', 'season', 'week', 'age_centered', 
                   'industry_score', 'outcome_type', 'outcome', 'ballroom_partner']]
        ])
        
        formula = """outcome ~ age_centered*C(outcome_type) + I(age_centered**2)*C(outcome_type) + 
                     industry_score*C(outcome_type) + 
                     week*C(outcome_type) + C(season)"""
        
        model = smf.ols(formula, data=df_stacked).fit(
            cov_type='cluster',
            cov_kwds={'groups': df_stacked['celebrity_name']}
        )
        
        print(model.summary())
        
        print("\n【关键交互项】")
        interaction_terms = [p for p in model.params.index if ':C(outcome_type)' in p]
        for term in interaction_terms:
            coef = model.params[term]
            se = model.bse[term]
            pval = model.pvalues[term]
            sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
            print(f"{term:40s}: {coef:8.4f} ({se:.4f}) {sig}")
        
        return model


# ==============================================================================
# 6. 可视化
# ==============================================================================

class Visualizer:
    """结果可视化"""
    
    @staticmethod
    def plot_dancer_effects(dancer_effects_judge, dancer_effects_fan, output_path='dancer_effects.png'):
        """绘制舞者效应对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        top_judge = dancer_effects_judge.head(15)
        axes[0].barh(range(len(top_judge)), top_judge['effect'])
        axes[0].set_yticks(range(len(top_judge)))
        axes[0].set_yticklabels(top_judge['dancer'])
        axes[0].set_xlabel('Random Effect')
        axes[0].set_title('Top 15 Dancers - Judge Score Impact')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
        
        top_fan = dancer_effects_fan.head(15)
        axes[1].barh(range(len(top_fan)), top_fan['effect'])
        axes[1].set_yticks(range(len(top_fan)))
        axes[1].set_yticklabels(top_fan['dancer'])
        axes[1].set_xlabel('Random Effect')
        axes[1].set_title('Top 15 Dancers - Fan Vote Impact')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n舞者效应图已保存: {output_path}")
    
    @staticmethod
    def plot_age_effect(df, output_path='age_effect.png'):
        """绘制年龄效应对比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(df['age'], df['judge_score'], alpha=0.3, s=20)
        z = np.polyfit(df['age'], df['judge_score'], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(df['age'].min(), df['age'].max(), 100)
        axes[0].plot(x_smooth, p(x_smooth), 'r-', linewidth=2, label='Quadratic Fit')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Judge Score')
        axes[0].set_title('Age vs Judge Score')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(df['age'], df['hybrid_percentage'], alpha=0.3, s=20)
        z = np.polyfit(df['age'], df['hybrid_percentage'], 2)
        p = np.poly1d(z)
        axes[1].plot(x_smooth, p(x_smooth), 'r-', linewidth=2, label='Quadratic Fit')
        axes[1].set_xlabel('Age')
        axes[1].set_ylabel('Fan Vote %')
        axes[1].set_title('Age vs Fan Vote')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"年龄效应图已保存: {output_path}")
    
    @staticmethod
    def plot_industry_effect(df, output_path='industry_effect.png'):
        """绘制行业效应对比"""
        industry_stats = df.groupby('celebrity_industry').agg({
            'judge_score': 'mean',
            'hybrid_percentage': 'mean',
            'celebrity_name': 'count'
        }).reset_index()
        
        industry_stats = industry_stats[industry_stats['celebrity_name'] >= 5]
        industry_stats = industry_stats.sort_values('judge_score', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].barh(range(len(industry_stats)), industry_stats['judge_score'])
        axes[0].set_yticks(range(len(industry_stats)))
        axes[0].set_yticklabels(industry_stats['celebrity_industry'], fontsize=9)
        axes[0].set_xlabel('Average Judge Score')
        axes[0].set_title('Industry Effect on Judge Score')
        
        axes[1].barh(range(len(industry_stats)), industry_stats['hybrid_percentage'])
        axes[1].set_yticks(range(len(industry_stats)))
        axes[1].set_yticklabels(industry_stats['celebrity_industry'], fontsize=9)
        axes[1].set_xlabel('Average Fan Vote %')
        axes[1].set_title('Industry Effect on Fan Vote')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"行业效应图已保存: {output_path}")


# ==============================================================================
# 7. 主程序
# ==============================================================================

def main():
    import os
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    FAN_VOTE_PATH = "/Users/fangyu/Documents/com/ms/code/-/Task1_Results/fan_vote_hybrid_mcmc.csv"
    if not os.path.exists(FAN_VOTE_PATH):
        FAN_VOTE_PATH = os.path.join(_SCRIPT_DIR, "Task1_Results", "fan_vote_hybrid_mcmc.csv")
    
    ORIGINAL_DATA_PATH = "/Users/fangyu/Documents/com/ms/code/-/Data.csv"
    if not os.path.exists(ORIGINAL_DATA_PATH):
        ORIGINAL_DATA_PATH = os.path.join(_SCRIPT_DIR, "Data.csv")
        if not os.path.exists(ORIGINAL_DATA_PATH):
            ORIGINAL_DATA_PATH = os.path.join(_SCRIPT_DIR, "2026_MCM_Problem_C_Data.csv")
    
    OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "Task4_Results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("Task 4: 舞者与名人特征的影响分析 (最终版)")
    print("="*80)
    
    # 数据准备
    print("\n[1] 数据准备...")
    try:
        prep = DataPreparator(FAN_VOTE_PATH, ORIGINAL_DATA_PATH)
        df = prep.create_analysis_dataset()
        print(f"    分析数据集: {len(df)} 条观测")
        print(f"    唯一选手: {df['celebrity_name'].nunique()}")
        print(f"    唯一舞者: {df['ballroom_partner'].nunique()}")
        print(f"    赛季范围: {df['season'].min()} - {df['season'].max()}")
    except Exception as e:
        print(f"    错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # OLS分析
    ols_analyzer = OLSAnalysis(df)
    ols_results = ols_analyzer.run_all()
    ols_analyzer.check_multicollinearity()
    
    # 混合效应模型
    mem_analyzer = MixedEffectsAnalysis(df)
    mem_results = mem_analyzer.run_all()
    dancer_effects = mem_analyzer.extract_dancer_effects()
    mem_analyzer.calculate_icc()
    
    # 舞者固定效应（替代TWFE）
    dancer_fe_analyzer = DancerFixedEffectsAnalysis(df)
    dancer_fe_results = dancer_fe_analyzer.run_all()
    dancer_coefs = dancer_fe_analyzer.extract_dancer_coefficients()
    
    # 差异分析
    diff_analyzer = DifferentialImpactAnalysis(mem_results, mem_results)
    coef_comparison = diff_analyzer.compare_coefficients()
    interaction_model = diff_analyzer.interaction_model(df)
    
    # 可视化
    print("\n[6] 生成可视化...")
    viz = Visualizer()
    
    if 'judge_mem' in dancer_effects and 'fan_mem' in dancer_effects:
        viz.plot_dancer_effects(
            dancer_effects['judge_mem'],
            dancer_effects['fan_mem'],
            output_path=os.path.join(OUTPUT_DIR, 'dancer_effects.png')
        )
    
    viz.plot_age_effect(df, output_path=os.path.join(OUTPUT_DIR, 'age_effect.png'))
    viz.plot_industry_effect(df, output_path=os.path.join(OUTPUT_DIR, 'industry_effect.png'))
    
    # 保存结果
    print("\n[7] 保存结果...")
    
    if 'judge_mem' in dancer_effects:
        dancer_effects['judge_mem'].to_csv(
            os.path.join(OUTPUT_DIR, 'dancer_ranking_judge.csv'), index=False
        )
    if 'fan_mem' in dancer_effects:
        dancer_effects['fan_mem'].to_csv(
            os.path.join(OUTPUT_DIR, 'dancer_ranking_fan.csv'), index=False
        )
    
    if 'judge_dancer_fe' in dancer_coefs:
        dancer_coefs['judge_dancer_fe'].to_csv(
            os.path.join(OUTPUT_DIR, 'dancer_fe_coefs_judge.csv'), index=False
        )
    if 'fan_dancer_fe' in dancer_coefs:
        dancer_coefs['fan_dancer_fe'].to_csv(
            os.path.join(OUTPUT_DIR, 'dancer_fe_coefs_fan.csv'), index=False
        )
    
    if coef_comparison is not None:
        coef_comparison.to_csv(
            os.path.join(OUTPUT_DIR, 'coefficient_comparison.csv'), index=False
        )
    
    with open(os.path.join(OUTPUT_DIR, 'model_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TASK 4 MODEL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        if 'judge_mem' in mem_results:
            f.write("MIXED EFFECTS MODEL - JUDGE SCORE\n")
            f.write("-"*80 + "\n")
            f.write(str(mem_results['judge_mem'].summary()))
            f.write("\n\n")
        
        if 'fan_mem' in mem_results:
            f.write("MIXED EFFECTS MODEL - FAN VOTE\n")
            f.write("-"*80 + "\n")
            f.write(str(mem_results['fan_mem'].summary()))
    
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("\n分析完成!")
    
    # 关键发现
    print("\n" + "="*80)
    print("关键发现摘要")
    print("="*80)
    
    if 'judge_mem' in dancer_effects and 'fan_mem' in dancer_effects:
        print("\n【1. 舞者随机效应排名 (混合效应模型)】")
        print("\n评委分数 - Top 5:")
        print(dancer_effects['judge_mem'].head(5)[['dancer', 'effect']].to_string(index=False))
        print("\n粉丝投票 - Top 5:")
        print(dancer_effects['fan_mem'].head(5)[['dancer', 'effect']].to_string(index=False))
    
    if coef_comparison is not None:
        print("\n【2. 名人特征影响对比】")
        print(coef_comparison[['Variable', 'Judge_Coef', 'Fan_Coef', 'Difference', 'Significant']])


if __name__ == "__main__":
    main()