# 2026 MCM Problem C: 可视化设计方案

## 一、推荐图表清单

以下是为 C 题论文建议的关键可视化图表，按优先级排序。

---

### 1. 基础数据探索（EDA）图表

#### 1.1 行业分布饼图 / 条形图
- **用途**：展示参赛明星的职业分布
- **放置位置**：论文 Background 或 Data Description 部分
- **代码函数**：`visualizer.plot_industry_distribution()`

```
预期效果：
┌─────────────────────────────────┐
│      Industry Distribution      │
│                                 │
│    Actor/Actress ████████ 35%   │
│    Athlete       █████   22%    │
│    Singer        ████    18%    │
│    TV Personality███     12%    │
│    Other         ██       8%    │
│    Model         █        5%    │
└─────────────────────────────────┘
```

#### 1.2 专业舞伴成功率排名
- **用途**：展示哪些舞伴的搭档更容易获胜
- **放置位置**：Feature Analysis 部分
- **代码函数**：`visualizer.plot_partner_success()`

#### 1.3 年龄与成绩关系散点图
- **用途**：探索年龄是否影响最终排名
- **放置位置**：Feature Analysis 部分
- **建议添加**：趋势线 + 置信区间

---

### 2. 粉丝投票估算图表

#### 2.1 某赛季每周投票估算柱状图
- **用途**：展示模型估算的粉丝投票分布
- **放置位置**：Model Results 部分
- **代码函数**：`visualizer.plot_estimated_votes(season, estimator)`

```
预期效果（每周一个子图）：
Week 1                    Week 2
┌──────────────────┐     ┌──────────────────┐
│ Drew     ████████│     │ Drew     ████████│
│ Stacy    ███████ │     │ Stacy    ███████ │
│ Jerry    ██████  │     │ Jerry    ██████  │
│ Lisa     █████   │     │ Lisa     █████   │
│ George   ████    │     │ George   ████    │
│ Tatum    ███ ←淘汰│    │ Tia      ██ ←淘汰 │
└──────────────────┘     └──────────────────┘
```

#### 2.2 估算置信区间图
- **用途**：展示投票估算的不确定性
- **类型**：带误差棒的柱状图或箱线图
- **关键信息**：哪些选手的投票更难估算

---

### 3. 投票机制对比图表

#### 3.1 排名制 vs 百分比制对比热力图
- **用途**：展示两种机制在不同赛季的差异
- **横轴**：赛季
- **纵轴**：周数
- **颜色**：结果一致（绿色）vs 结果不同（红色）

```
预期效果：
          Week 1  Week 2  Week 3  Week 4  ...
Season 1   ✓       ✓       ✓       ✗
Season 2   ✓       ✗       ✓       ✗
Season 3   ✓       ✓       ✓       ✓
...
```

#### 3.2 机制偏向性分析图
- **用途**：量化哪种机制更偏向粉丝/评委
- **类型**：双轴柱状图或雷达图

---

### 4. 争议案例分析图表

#### 4.1 争议选手每周排名变化折线图
- **用途**：展示 Jerry Rice、Bobby Bones 等人的评委排名变化
- **代码**：见 Notebook 中的 `analyze_controversial_case()` 函数
- **关键设计**：
  - 用红色标注评委排名垫底的周
  - 添加淘汰线（理论上应在此处被淘汰）

```
预期效果：
       Jerry Rice (Season 2) - Judge Ranking
       ┌────────────────────────────────────┐
   1   │                                    │
   2   │                                    │
   3   │    ●─────●                         │
   4   │   ╱       ╲       ●               │
   5   │  ●         ●─────●─●─●─●  ← 垫底   │
   6   │                                    │
       └────────────────────────────────────┘
       Week 1  2  3  4  5  6  7  8
```

#### 4.2 假设情景分析图
- **用途**：展示如果使用另一种机制，争议选手会在第几周被淘汰
- **类型**：时间线对比图

---

### 5. 新机制提案图表

#### 5.1 新旧机制效果对比
- **用途**：证明你提出的新机制更"公平"
- **指标**：
  - 评委垫底却存活的次数
  - 高分选手被淘汰的次数
  - 争议案例的处理结果

#### 5.2 敏感性分析热力图
- **用途**：展示权重参数变化对结果的影响
- **横轴**：评委权重 (0-100%)
- **纵轴**：粉丝权重 (100%-0%)
- **颜色**：争议选手被淘汰的周数

---

## 二、论文图表布局建议

### 建议的图表数量
- 论文正文：6-8 张精选图表
- 附录：补充图表（如所有赛季的分析）

### 每部分建议图表

| 论文章节 | 建议图表 |
|---------|---------|
| Background | 行业分布饼图、赛季选手数量变化图 |
| Data Analysis | 年龄-成绩散点图、舞伴成功率排名图 |
| Model | 粉丝投票估算示意图（流程图）、某赛季投票估算结果 |
| Results | 机制对比热力图、争议案例时间线图 |
| Recommendation | 新机制效果对比图、敏感性分析图 |

---

## 三、配色方案建议

### 推荐配色
- **主色调**：Steelblue (#4682B4) - 用于正常数据
- **强调色**：Coral (#FF7F50) - 用于被淘汰/异常数据
- **成功色**：Gold (#FFD700) - 用于冠军/胜出
- **中性色**：Gray (#808080) - 用于辅助线/次要信息

### Matplotlib 设置
```python
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# 自定义颜色
COLORS = {
    'primary': '#4682B4',
    'highlight': '#FF7F50',
    'success': '#FFD700',
    'neutral': '#808080'
}
```

---

## 四、图表制作检查清单

每张图表发布前，请确认：

- [ ] 有清晰的标题
- [ ] 坐标轴有标签和单位
- [ ] 图例位置合适且不遮挡数据
- [ ] 字体大小足够大（打印后仍可读）
- [ ] 分辨率至少 150 DPI
- [ ] 保存为 PNG 或 PDF 格式
- [ ] 文件名有意义（如 `fig2_season2_votes.png`）

---

## 五、快速生成所有图表

在 Python 中运行以下代码即可生成所有基础图表：

```python
from dwts_analysis import DWTSDataProcessor, FanVoteEstimator, VotingMechanismAnalyzer, DWTSVisualizer

# 1. 加载数据
DATA_PATH = "path/to/2026_MCM_Problem_C_Data.csv"
processor = DWTSDataProcessor(DATA_PATH)
processor.preprocess()

# 2. 创建分析器
estimator = FanVoteEstimator(processor)
analyzer = VotingMechanismAnalyzer(processor, estimator)
visualizer = DWTSVisualizer(processor)

# 3. 生成图表
import os
os.makedirs('figures', exist_ok=True)

visualizer.plot_industry_distribution('figures/fig1_industry.png')
visualizer.plot_partner_success('figures/fig2_partner.png')
visualizer.plot_season_scores(2, 'figures/fig3_season2_scores.png')
visualizer.plot_estimated_votes(2, estimator, 'figures/fig4_season2_votes.png')

print("图表生成完成！")
```
