import streamlit as st
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize_scalar

st.set_page_config(page_title="Shadow Price Simulator", layout="wide")

SLACK_TOLERANCE = 1e-4  # 余裕がゼロ（Active制約）とみなす許容誤差

# --- Language Selection ---
lang = st.sidebar.radio("Language / 言語", ["English", "日本語"], index=1)

# --- Translations Dictionary ---
T = {
    "title": {"日本語": "Shadow Price (Dual Variable) Simulator", "English": "Shadow Price (Dual Variable) Simulator"},
    "subtitle": {
        "日本語": "最適化における**シャドウプライス（双対変数）**を用いて、設備投資の限界価値と投資コストを比較し、データドリブンな意思決定をシミュレーションします。",
        "English": "Interactive analysis of capital investment bottlenecks and ROI using **Shadow Prices (Dual Variables)** in mathematical optimization."
    },
    "scenarios": {
        "日本語": ["シナリオ1: 機械1がボトルネック (基本)", "シナリオ2: 機械2がボトルネック (機械2が安い)", "シナリオ3: 両方に余裕あり (需要低)", "カスタム (手動設定)"],
        "English": ["Scenario 1: Machine 1 is Bottleneck (Default)", "Scenario 2: Machine 2 is Bottleneck (M2 is cheaper)", "Scenario 3: No Bottlenecks (Low Demand)", "Custom Settings"]
    },
    "scenario_title": {"日本語": "📖 シナリオ選択", "English": "📖 Scenario Selection"},
    "scenario_desc": {"日本語": "シナリオを選ぶとパラメータが自動設定されます", "English": "Selecting a scenario will auto-fill parameters"},
    "param_title": {"日本語": "⚙️ ビジネスパラメータ設定", "English": "⚙️ Business Parameters"},
    "cost_header": {"日本語": "生産コスト (Production Cost)", "English": "Production Cost"},
    "m1_c": {"日本語": "機械1の生産コスト (C1)", "English": "Machine 1 Cost (C1)"},
    "m2_c": {"日本語": "機械2の生産コスト (C2)", "English": "Machine 2 Cost (C2)"},
    "cap_header": {"日本語": "生産能力 (Capacity)", "English": "Capacity"},
    "m1_n": {"日本語": "機械1の上限 (Nmax1)", "English": "Machine 1 Limit (Nmax1)"},
    "m2_n": {"日本語": "機械2の上限 (Nmax2)", "English": "Machine 2 Limit (Nmax2)"},
    "inv_header": {"日本語": "増設コスト (Investment Cost per unit)", "English": "Investment Cost (per unit)"},
    "m1_inv": {"日本語": "機械1の増設コスト", "English": "Machine 1 Expansion Cost"},
    "m2_inv": {"日本語": "機械2の増設コスト", "English": "Machine 2 Expansion Cost"},
    "dem_header": {"日本語": "需要 (Demand)", "English": "Demand"},
    "dem_lbl": {"日本語": "総需要 (D)", "English": "Total Demand (D)"},
    "run_btn": {"日本語": "最適化を実行する", "English": "Run Optimization"},
    "error_solver_missing": {
        "日本語": "❌ CBCソルバーが見つかりません。\n\n**インストール手順:**\n- Streamlit Cloud: `packages.txt` に `coinor-cbc` と `coinor-libcbc-dev` を追加してください\n- Mac: `brew tap coin-or-tools/coinor && brew install cbc`\n- Windows: [Coin-OR Cbc](https://github.com/coin-or/Cbc) からバイナリをダウンロード",
        "English": "❌ CBC solver not found.\n\n**Install instructions:**\n- Streamlit Cloud: Add `coinor-cbc` and `coinor-libcbc-dev` to `packages.txt`\n- Mac: `brew tap coin-or-tools/coinor && brew install cbc`\n- Windows: Download from [Coin-OR Cbc](https://github.com/coin-or/Cbc)"
    },
    "error_infeasible": {
        "日本語": "現在のパラメータでは実行可能解が見つかりません。（需要に対して生産能力が不足しています）",
        "English": "No feasible solution found with current parameters. (Capacity is insufficient for the demand)"
    },
    "sol_exp_title": {"日本語": "💡 解決策 (Solutions)", "English": "💡 Solutions"},
    "sol_desc1": {"日本語": "- 需要(D)を {max_d} 以下に減らしてください。", "English": "- Reduce Demand (D) to {max_d} or less."},
    "sol_desc2": {"日本語": "- または、機械の生産能力(Nmax1, Nmax2)を合計で {d} 以上に増やしてください。", "English": "- Alternatively, increase total Machine Capacity (Nmax1 + Nmax2) to at least {d}."},
    "tab1": {"日本語": "📊 シミュレーター", "English": "📊 Simulator"},
    "tab2": {"日本語": "📖 問題設定と定式化", "English": "📖 Formulation & Scenarios"},
    "tab3": {"日本語": "📈 利益最大化 (価格戦略)", "English": "📈 Profit Maximization (Pricing Strategy)"},
    "h1_kpi": {"日本語": "1. 最適な生産計画とKPI", "English": "1. Optimal Production Plan & KPI"},
    "metric_cost": {"日本語": "総製造コスト (最小化)", "English": "Total Cost (Minimized)"},
    "metric_m1": {"日本語": "機械1の生産量", "English": "Machine 1 Production"},
    "metric_m2": {"日本語": "機械2の生産量", "English": "Machine 2 Production"},
    "usage_lbl": {"日本語": "使用率: {pct:.1f}%", "English": "Usage: {pct:.1f}%"},
    "lbl_bot": {"日本語": "🔥 ボトルネック", "English": "🔥 Bottleneck"},
    "warn_degen": {
        "日本語": "⚠️ **退化解の警告**: 機械1と機械2のコストが同一で、かつ余裕（Slack）が存在するため、最適解が複数存在する可能性があります。双対変数（シャドウプライス）が不安定になる場合があります。",
        "English": "⚠️ **Degeneracy Warning**: Costs for M1 and M2 are identical and slack exists, leading to multiple optimal solutions. Shadow prices may be unstable."
    },
    "h2_insight": {"日本語": "2. ビジネスインサイト (意思決定)", "English": "2. Business Insights (Decision Making)"},
    "insight_demand": {
        "日本語": "📈 **需要の限界コスト (価格戦略):**\n\n需要を1単位増やすと、総コストは **${val:.1f}** 増加します。製品の販売価格をこれ以上に設定すれば利益が出ます。",
        "English": "📈 **Marginal Cost of Demand (Pricing Strategy):**\n\nIncreasing demand by 1 unit increases total cost by **${val:.1f}**. Selling price should be set higher than this to be profitable."
    },
    "c1_sign_note": {
        "日本語": "（※ 正の場合はコスト増、負の場合はコスト減）",
        "English": "(* Positive means cost increases, negative means cost decreases)"
    },
    "insight_m_bot_invest": {
        "日本語": "💡 **投資推奨 (機械{i}):** 限界価値(${sp:.1f}) が 増設コスト(${inv:.1f}) を上回っています！増設により **${net:.1f}** の純利益が得られます。(投資回収: {payback:.1f} 期)",
        "English": "💡 **Invest (Machine {i}):** Marginal Value (${sp:.1f}) > Expansion Cost (${inv:.1f})! Expanding yields a net profit of **${net:.1f}**. (Payback: {payback:.1f} periods)"
    },
    "insight_m_bot_pass": {
        "日本語": "⚠️ **投資見送り (機械{i}):** 限界価値(${sp:.1f}) はありますが、増設コスト(${inv:.1f}) の方が高いため、増設すべきではありません。",
        "English": "⚠️ **Do Not Invest (Machine {i}):** Expansion Cost (${inv:.1f}) exceeds Marginal Value (${sp:.1f}). Do not expand."
    },
    "insight_m_ok": {
        "日本語": "ℹ️ **能力余裕 (機械{i}):** 生産能力に余裕があります（限界価値: $0）。投資は不要です。",
        "English": "ℹ️ **Excess Capacity (Machine {i}):** Capacity is not fully utilized (Marginal Value: $0). No investment needed."
    },
    "sp_local_warn": {
        "日本語": "※ 限界価値は「微小な変化」に対する局所的な近似です。",
        "English": "* Marginal value is a local approximation valid only for small changes."
    },
    "sp_large_warn": {
        "日本語": "大規模な増設を行う場合は、前提条件が変わるため再最適化が必要です。",
        "English": "Large capacity expansions require re-optimization as the bottleneck may shift."
    },
    "h3_sens": {"日本語": "3. 感度分析 (機械1の能力と総コスト)", "English": "3. Sensitivity Analysis (Machine 1 Capacity vs Total Cost)"},
    "sens_desc": {"日本語": "機械1の生産能力($Nmax_1$)を変化させたときに、総コストと限界価値（シャドウプライス）がどう推移するかを可視化します。", "English": "Visualizes how total cost and Marginal Value (Shadow Price) change as Machine 1's capacity ($Nmax_1$) varies."},
    "spinner": {"日本語": "最適化計算を実行中...", "English": "Running optimization calculations..."},
    "plot_cur": {"日本語": "現在の状態", "English": "Current State"},
    "plot_x": {"日本語": "機械1の最大生産能力 (Nmax1)", "English": "Machine 1 Max Capacity (Nmax1)"},
    "plot_y": {"日本語": "総コスト (Objective)", "English": "Total Cost (Objective)"},
    "plot_y2": {"日本語": "限界価値 (Marginal Value)", "English": "Marginal Value (Shadow Price)"},
    "h4_table": {"日本語": "4. 制約と双対変数の詳細", "English": "4. Constraints & Dual Variables Details"},
    "table_desc": {
        "日本語": "💡 **重要な法則**: シャドウプライスは「制約がActive（余裕 Slack = 0）なときのみ」価値を持ちます。Slack > 0 の制約の限界価値は必ず 0 になります。\n\n制約の双対変数（元の符号）と、投資判断に使う限界価値（Marginal Value）の違いを確認できます。（最小化問題の `<=` 制約では、限界価値は元の双対変数の符号を反転させたものになります）",
        "English": "💡 **Key Rule**: Shadow Price has value ONLY when the constraint is Active (Slack = 0). If Slack > 0, Marginal Value is always 0.\n\nDetailed mathematical data showing the original dual variables and the interpreted Marginal Values (Shadow Prices)."
    },
    "math_note_title": {"日本語": "🎓 双対変数の符号に関する数学的補足 (Math Note)", "English": "🎓 Mathematical Note on Dual Variable Signs"},
    "math_note_text": {
        "日本語": "最適化ソルバー（Pyomo/CBC）は、制約式を標準形にして解くため、最小化問題における `<=` 制約の双対変数は**負の値**として出力されます。\nしかし、ビジネス上の意思決定（限界価値）としては、「制約を緩めたときにどれだけコストが減るか」という**正の値**として扱うのが自然です。\nそのため、本シミュレーターではビジネスインサイトを導き出す際に符号を反転（`Marginal Value = -Raw Dual`）させています。",
        "English": "Optimization solvers like Pyomo/CBC evaluate constraints in standard form, so the dual variables for `<=` constraints in a minimization problem are output as **negative values**.\nHowever, for business decision-making (Marginal Value), it is more natural to treat this as a **positive value** representing \"how much cost is reduced when the constraint is relaxed.\"\nTherefore, this simulator inverses the sign (`Marginal Value = -Raw Dual`) to derive actionable business insights."
    },
    "tbl_col1": {"日本語": "制約", "English": "Constraint"},
    "tbl_col2": {"日本語": "式", "English": "Equation"},
    "tbl_col3": {"日本語": "現在の値", "English": "Current Value"},
    "tbl_col4": {"日本語": "制限値", "English": "Limit"},
    "tbl_col5": {"日本語": "余裕 (Slack)", "English": "Slack"},
    "tbl_col6": {"日本語": "元の双対変数 (Raw Dual)", "English": "Raw Dual Variable"},
    "tbl_col7": {"日本語": "限界価値 (Marginal Value)", "English": "Marginal Value"},
    "c1_name": {"日本語": "C1 (需要)", "English": "C1 (Demand)"},
    "c2_name": {"日本語": "C2 (機械1能力)", "English": "C2 (Mach. 1 Cap)"},
    "c3_name": {"日本語": "C3 (機械2能力)", "English": "C3 (Mach. 2 Cap)"},
    
    # Profit Maximization tab texts
    "h1_profit": {"日本語": "ビジネスモデル拡張: 価格と需要の関係", "English": "Business Model Extension: Price vs Demand"},
    "profit_desc": {
        "日本語": "製品の販売価格(Price)を変えると需要(Demand)が変化する「需要曲線」を導入し、**利益(Profit)を最大化する最適価格**を見つけます。\n\n需要モデル: $Demand = a - b \\times Price$", 
        "English": "Introducing a demand curve where Demand changes with Price. Find the **Optimal Price that maximizes Profit**.\n\nDemand Model: $Demand = a - b \\times Price$"
    },
    "lbl_a": {"日本語": "最大潜在需要 (a)", "English": "Max Potential Demand (a)"},
    "lbl_b": {"日本語": "価格弾力性 (b: 価格1単位あたりの需要減)", "English": "Price Elasticity (b: Demand drop per unit price)"},
    "run_profit": {"日本語": "利益最大化シミュレーションを実行", "English": "Run Profit Simulation"},
    "opt_price": {"日本語": "最適販売価格", "English": "Optimal Price"},
    "opt_profit": {"日本語": "最大予測利益", "English": "Max Predicted Profit"},
    "opt_demand": {"日本語": "予測需要量", "English": "Predicted Demand"},
    "plot_profit_x": {"日本語": "販売価格 (Price)", "English": "Selling Price (Price)"},
    "plot_profit_y": {"日本語": "利益 (Profit)", "English": "Profit"},
    "opt_result_txt": {
        "日本語": "💡 連続最適化（scipy.optimize）による正確な最適価格設定は **${price:,.1f}** です。このとき需要は **{demand:,.1f}** となり、機械1で {n1:.1f}個、機械2で {n2:.1f}個を生産することで、最大の利益 **${profit:,.1f}** が得られます。",
        "English": "💡 The exact optimal price (via scipy.optimize) is **${price:,.1f}**. Demand becomes **{demand:,.1f}**, producing {n1:.1f} on M1 and {n2:.1f} on M2, yielding a max profit of **${profit:,.1f}**."
    },
    
    "formulation": {
        "日本語": """
### ビジネスシナリオ
あなたは工場の生産管理者です。現在、ある製品を**合計D個**（需要）生産しなければなりません。
工場には**機械1**と**機械2**の2つの設備があります。

* **機械1**: 生産コストが安いですが、1度に生産できる上限が制限されています。
* **機械2**: 生産能力には余裕がありますが、生産コストが高いです。

**目標**: 総需要を確実に満たしつつ、2つの機械をうまく使い分けて**総製造コストを最小化**してください。

---

### 数理最適化モデルの定式化

#### 1. 変数 (Variables)
* $N_1$ : 機械1での生産量
* $N_2$ : 機械2での生産量

#### 2. 目的関数 (Objective Function)
製造コストの合計を最小化します。（※係数はサイドバーの設定値）
$$ \\min \\quad C_1 N_1 + C_2 N_2 $$

#### 3. 制約条件 (Constraints)
* **需要制約 (C1)**: 総生産量は必ず需要 $D$ を満たさなければならない。
  $$ N_1 + N_2 = D $$
* **機械1の能力制約 (C2)**: 機械1は最大 $Nmax_1$ までしか生産できない。
  $$ N_1 \\le Nmax_1 $$
* **機械2の能力制約 (C3)**: 機械2は最大 $Nmax_2$ までしか生産できない。
  $$ N_2 \\le Nmax_2 $$
* **非負制約**: 生産量は0以上でなければならない。
  $$ N_1, N_2 \\ge 0 $$

---
### 双対変数と限界価値（Shadow Price）の正確な理解

この問題を解くと、単に「N1を何個、N2を何個作る」という答えが出るだけでなく、**双対変数（Dual Variable）**が得られます。これは**「制約式の右辺定数を1単位増やしたときの、目的関数の変化量」**を意味します。

#### 1. 需要制約 (C1: N1 + N2 = D) の双対変数
* $D$ を1増やす（需要が増える）と、より多くの製品を作る必要があり生産コストは**増加**します。
* したがって、C1の双対変数は**正の値**になります。
* **ビジネス解釈（限界費用）**: 「製品を1個多く作るにはコストが○○ドルかかる。だから販売価格はそれ以上に設定すべき」という**価格戦略**の基準になります。

#### 2. 能力制約 (C2: N1 <= Nmax1) の双対変数
* $Nmax_1$ を1増やす（設備を拡張する）と、より安い機械で作れる量が増えるため、総コストは**減少**します。
* したがって、Pyomo/CBCが出力するC2の双対変数は**負の値**（例: -200）になります。
* **ビジネス解釈（限界価値）**: コストが200減るということは、能力拡張がもたらす**価値は +200** であると言えます（`限界価値 = -元の双対変数`）。
* **意思決定**: この限界価値（+200）と、設備の「増設コスト」を比較することで、合理的な投資判断が可能になります。
""",
        "English": """
### Business Scenario
You are a production manager. You need to produce $D$ units of a product to meet total demand. You have two machines:
* **Machine 1**: Low production cost, but limited capacity.
* **Machine 2**: High capacity, but high production cost.

**Goal**: Meet the total demand while minimizing the **total production cost**.

---

### Mathematical Formulation

#### 1. Variables
* $N_1$ : Production amount on Machine 1
* $N_2$ : Production amount on Machine 2

#### 2. Objective Function
Minimize total production cost.
$$ \\min \\quad C_1 N_1 + C_2 N_2 $$

#### 3. Constraints
* **Demand (C1)**: Total production must equal demand $D$.
  $$ N_1 + N_2 = D $$
* **Machine 1 Capacity (C2)**: Cannot exceed max capacity $Nmax_1$.
  $$ N_1 \\le Nmax_1 $$
* **Machine 2 Capacity (C3)**: Cannot exceed max capacity $Nmax_2$.
  $$ N_2 \\le Nmax_2 $$
* **Non-negativity**:
  $$ N_1, N_2 \\ge 0 $$

---
### Accurate Understanding of Dual Variables and Marginal Values

Solving this problem yields **Dual Variables**, representing **"the rate of change in the objective function per unit increase in the RHS of a constraint."**

#### 1. Demand Constraint (C1: N1 + N2 = D)
* Increasing $D$ by 1 forces more production, **increasing** total cost.
* Thus, C1's dual variable is **positive**.
* **Business Interpretation (Marginal Cost)**: This sets a baseline for **Pricing Strategy**—you must sell the product for more than this marginal cost to make a profit.

#### 2. Capacity Constraint (C2: N1 <= Nmax1)
* Increasing capacity $Nmax_1$ allows more use of the cheaper machine, **decreasing** total cost.
* Thus, Pyomo/CBC outputs a **negative** dual variable for C2 (e.g., -200).
* **Business Interpretation (Marginal Value)**: A cost reduction of 200 implies the value of expansion is **+200** (`Marginal Value = -Dual Variable`).
* **Decision Making**: Comparing this Marginal Value (+200) with the actual "Expansion Cost" allows you to make data-driven investment decisions.
"""
    }
}

TL = T
def t(key):
    return TL[key][lang]

# --- Pyomo Model Setup with Parameter Updating (Warm-start/Speedup) ---
def build_model():
    m = pyo.ConcreteModel()
    m.N1 = pyo.Var(within=pyo.NonNegativeReals)
    m.N2 = pyo.Var(within=pyo.NonNegativeReals)
    
    m.c1 = pyo.Param(mutable=True, default=100)
    m.c2 = pyo.Param(mutable=True, default=300)
    m.nmax1 = pyo.Param(mutable=True, default=31)
    m.nmax2 = pyo.Param(mutable=True, default=100)
    m.d = pyo.Param(mutable=True, default=50)
    
    m.obj = pyo.Objective(expr=m.c1 * m.N1 + m.c2 * m.N2, sense=pyo.minimize)
    m.C1 = pyo.Constraint(expr=m.N1 + m.N2 == m.d)
    m.C2 = pyo.Constraint(expr=m.N1 <= m.nmax1)
    m.C3 = pyo.Constraint(expr=m.N2 <= m.nmax2)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    opt = SolverFactory('cbc')
    return m, opt

if 'pyomo_model' not in st.session_state:
    m, opt = build_model()
    if not opt.available():
        st.error(t("error_solver_missing"))
        st.stop()
    st.session_state.pyomo_model = m
    st.session_state.pyomo_opt = opt

# Caching solving for sensitivity and profit grids
@st.cache_data(show_spinner=False)
def solve_model(c1, c2, nmax1, nmax2, d):
    m = st.session_state.pyomo_model
    opt = st.session_state.pyomo_opt
    
    m.c1.set_value(c1)
    m.c2.set_value(c2)
    m.nmax1.set_value(nmax1)
    m.nmax2.set_value(nmax2)
    m.d.set_value(d)
    
    # Suppress solver output for inner loops
    results = opt.solve(m, tee=False)
    status = results.solver.termination_condition
    
    if status == pyo.TerminationCondition.optimal:
        raw_dual_c1 = m.dual[m.C1]
        raw_dual_c2 = m.dual[m.C2]
        raw_dual_c3 = m.dual[m.C3]
        
        return {
            'status': 'Optimal',
            'obj': pyo.value(m.obj),
            'N1': pyo.value(m.N1),
            'N2': pyo.value(m.N2),
            'raw_dual_C1': raw_dual_c1,
            'raw_dual_C2': raw_dual_c2,
            'raw_dual_C3': raw_dual_c3,
            'sp_C1': raw_dual_c1,       
            'sp_C2': -raw_dual_c2,      
            'sp_C3': -raw_dual_c3,      
            'slack_C2': nmax1 - pyo.value(m.N1),
            'slack_C3': nmax2 - pyo.value(m.N2)
        }
    else:
        return {'status': 'Infeasible'}

@st.cache_data(show_spinner=False)
def generate_sensitivity_data(c1, c2, nmax1, nmax2, d):
    plot_data = []
    x_range = list(range(0, d + 20, 2))
    for x in x_range:
        sim_res = solve_model(c1, c2, x, nmax2, d)
        if sim_res['status'] == 'Optimal':
            plot_data.append({
                'Nmax1': x, 
                'Total Cost': sim_res['obj'],
                'Marginal Value': sim_res['sp_C2']
            })
    return plot_data

@st.cache_data(show_spinner=False)
def generate_profit_data(c1, c2, nmax1, nmax2, a_val, b_val):
    max_price = int(a_val / b_val)
    
    def negative_profit(p):
        d_val = a_val - b_val * p
        if d_val <= 0 or d_val > (nmax1 + nmax2):
            return 0.0
        sim_res = solve_model(c1, c2, nmax1, nmax2, d_val)
        if sim_res['status'] == 'Optimal':
            return -(p * d_val - sim_res['obj'])
        return 0.0
    
    res_opt = minimize_scalar(negative_profit, bounds=(1, max_price), method='bounded')
    best_p = res_opt.x
    
    profit_data = []
    p_grid = np.linspace(1, max_price, 200)
    p_grid = np.append(p_grid, best_p)
    p_grid = np.sort(p_grid)

    for p in p_grid:
        d_val = a_val - b_val * p
        if d_val <= 0 or d_val > (nmax1 + nmax2):
            continue
        sim_res = solve_model(c1, c2, nmax1, nmax2, d_val)
        if sim_res['status'] == 'Optimal':
            profit_data.append({
                'Price': p, 'Demand': d_val, 'Cost': sim_res['obj'],
                'Revenue': p * d_val, 'Profit': p * d_val - sim_res['obj'],
                'N1': sim_res['N1'], 'N2': sim_res['N2']
            })
    return best_p, profit_data

st.title(t("title"))
st.markdown(t("subtitle"))

# --- Session State Initialization ---
if 'c1' not in st.session_state:
    st.session_state.c1 = 100
    st.session_state.c2 = 300
    st.session_state.nmax1 = 31
    st.session_state.nmax2 = 100
    st.session_state.d = 50
    st.session_state.inv_cost1 = 150
    st.session_state.inv_cost2 = 150

def apply_scenario():
    scenario_str = st.session_state.scenario_selector
    scenarios_list = t("scenarios")
    if scenario_str not in scenarios_list:
        return
    idx = scenarios_list.index(scenario_str)
    
    # 完全に初期化することで前シナリオの残留を防止
    st.session_state.inv_cost1 = 150
    st.session_state.inv_cost2 = 150
    
    if idx == 0:
        st.session_state.c1 = 100
        st.session_state.c2 = 300
        st.session_state.nmax1 = 31
        st.session_state.nmax2 = 100
        st.session_state.d = 50
    elif idx == 1:
        st.session_state.c1 = 300
        st.session_state.c2 = 100
        st.session_state.nmax1 = 100
        st.session_state.nmax2 = 31
        st.session_state.d = 50
    elif idx == 2:
        st.session_state.c1 = 100
        st.session_state.c2 = 300
        st.session_state.nmax1 = 100
        st.session_state.nmax2 = 100
        st.session_state.d = 50

# --- Sidebar Inputs ---
st.sidebar.header(t("scenario_title"))
st.sidebar.selectbox(
    t("scenario_desc"),
    t("scenarios"),
    key="scenario_selector",
    on_change=apply_scenario
)

st.sidebar.header(t("param_title"))
with st.sidebar.form("param_form"):
    st.subheader(t("cost_header"))
    c1 = st.slider(t("m1_c"), 50, 500, key="c1", step=10)
    c2 = st.slider(t("m2_c"), 50, 500, key="c2", step=10)
    
    st.subheader(t("cap_header"))
    nmax1 = st.slider(t("m1_n"), 0, 150, key="nmax1", step=1)
    nmax2 = st.slider(t("m2_n"), 0, 150, key="nmax2", step=1)
    
    st.subheader(t("dem_header"))
    d = st.slider(t("dem_lbl"), 10, 200, key="d", step=1)
    
    st.subheader(t("inv_header"))
    inv_cost1 = st.number_input(t("m1_inv"), 0, 1000, key="inv_cost1", step=10)
    inv_cost2 = st.number_input(t("m2_inv"), 0, 1000, key="inv_cost2", step=10)
    
    submitted = st.form_submit_button(t("run_btn"), type="primary", width="stretch")

# --- Solve the current model ---
res = solve_model(c1, c2, nmax1, nmax2, d)

if res['status'] != 'Optimal':
    st.error(t("error_infeasible"))
    with st.expander(t("sol_exp_title")):
        max_d = nmax1 + nmax2
        st.write(t("sol_desc1").format(max_d=max_d))
        st.write(t("sol_desc2").format(d=d))
    st.stop()

# 退化解（複数最適解）の警告
if abs(c1 - c2) < 1e-6 and (res['slack_C2'] > SLACK_TOLERANCE or res['slack_C3'] > SLACK_TOLERANCE):
    st.warning(t("warn_degen"))

# --- Main Dashboard ---
tab1, tab_profit, tab_form = st.tabs([t("tab1"), t("tab3"), t("tab2")])

with tab1:
    st.header(t("h1_kpi"))
    col1, col2, col3 = st.columns(3)
    col1.metric(t("metric_cost"), f"${res['obj']:,.1f}")
    
    with col2:
        st.metric(t("metric_m1"), f"{res['N1']:.1f} / {nmax1}")
        usage1 = min(res['N1'] / nmax1 if nmax1 > 0 else 1.0, 1.0)
        st.progress(usage1)
        st.caption(t("usage_lbl").format(pct=usage1 * 100))
        if res['slack_C2'] <= SLACK_TOLERANCE and nmax1 > 0:
            st.error(t("lbl_bot"))
            
    with col3:
        st.metric(t("metric_m2"), f"{res['N2']:.1f} / {nmax2}")
        usage2 = min(res['N2'] / nmax2 if nmax2 > 0 else 1.0, 1.0)
        st.progress(usage2)
        st.caption(t("usage_lbl").format(pct=usage2 * 100))
        if res['slack_C3'] <= SLACK_TOLERANCE and nmax2 > 0:
            st.error(t("lbl_bot"))

    # --- Business Insights ---
    st.header(t("h2_insight"))
    
    st.info(t("insight_demand").format(val=res['sp_C1']) + f"\n\n{t('c1_sign_note')}")

    if res['sp_C2'] > 0:
        net1 = res['sp_C2'] - inv_cost1
        payback1 = inv_cost1 / res['sp_C2'] if res['sp_C2'] > 0 else float('inf')
        if net1 > 0:
            st.success(t("insight_m_bot_invest").format(i=1, sp=res['sp_C2'], inv=inv_cost1, net=net1, payback=payback1))
        else:
            st.warning(t("insight_m_bot_pass").format(i=1, sp=res['sp_C2'], inv=inv_cost1))
    else:
        st.info(t("insight_m_ok").format(i=1))

    if res['sp_C3'] > 0:
        net2 = res['sp_C3'] - inv_cost2
        payback2 = inv_cost2 / res['sp_C3'] if res['sp_C3'] > 0 else float('inf')
        if net2 > 0:
            st.success(t("insight_m_bot_invest").format(i=2, sp=res['sp_C3'], inv=inv_cost2, net=net2, payback=payback2))
        else:
            st.warning(t("insight_m_bot_pass").format(i=2, sp=res['sp_C3'], inv=inv_cost2))
    else:
        st.info(t("insight_m_ok").format(i=2))

    st.caption(t("sp_local_warn"))
    st.caption(f"⚠️ {t('sp_large_warn')}")

    # --- Sensitivity Analysis (Plotly) ---
    st.header(t("h3_sens"))
    st.markdown(t("sens_desc"))

    with st.spinner(t("spinner")):
        plot_data = generate_sensitivity_data(c1, c2, nmax1, nmax2, d)
        df_plot = pd.DataFrame(plot_data)

        if not df_plot.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(go.Scatter(
                x=df_plot['Nmax1'], y=df_plot['Total Cost'],
                mode='lines',
                name='Total Cost',
                line=dict(color='royalblue', width=3)
            ), secondary_y=False)
            
            fig.add_trace(go.Scatter(
                x=df_plot['Nmax1'], y=df_plot['Marginal Value'],
                mode='lines',
                name='Marginal Value',
                line=dict(color='orange', width=2, dash='dot')
            ), secondary_y=True)
            
            fig.add_trace(go.Scatter(
                x=[nmax1], y=[res['obj']],
                mode='markers+text',
                name='Current Nmax1',
                text=[t("plot_cur")],
                textposition="top right",
                marker=dict(color='red', size=12, symbol='circle')
            ), secondary_y=False)
            
            fig.update_layout(
                xaxis_title=t("plot_x"),
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                template="plotly_white"
            )
            fig.update_yaxes(title_text=t("plot_y"), secondary_y=False)
            fig.update_yaxes(title_text=t("plot_y2"), secondary_y=True)
            
            st.plotly_chart(fig, width="stretch")

    # --- Dual Variables Table ---
    st.header(t("h4_table"))
    st.markdown(t("table_desc"))
    
    with st.expander(t("math_note_title")):
        st.markdown(t("math_note_text"))

    df_constraints = pd.DataFrame({
        t("tbl_col1"): [t("c1_name"), t("c2_name"), t("c3_name")],
        t("tbl_col2"): ["N1 + N2 = D", "N1 <= Nmax1", "N2 <= Nmax2"],
        t("tbl_col3"): [res['N1'] + res['N2'], res['N1'], res['N2']],
        t("tbl_col4"): [d, nmax1, nmax2],
        t("tbl_col5"): [0.0, float(res['slack_C2']), float(res['slack_C3'])],
        t("tbl_col6"): [res['raw_dual_C1'], res['raw_dual_C2'], res['raw_dual_C3']],
        t("tbl_col7"): [res['sp_C1'], res['sp_C2'], res['sp_C3']]
    })

    st.dataframe(df_constraints.style.format({
        t("tbl_col3"): "{:.1f}",
        t("tbl_col4"): "{:.1f}",
        t("tbl_col5"): "{:.1f}",
        t("tbl_col6"): "{:.2f}",
        t("tbl_col7"): "{:.2f}"
    }), width="stretch")

with tab_profit:
    st.header(t("h1_profit"))
    st.markdown(t("profit_desc"))
    
    col_a, col_b = st.columns(2)
    a_val = col_a.number_input(t("lbl_a"), 100, 1000, 200, 10)
    b_val = col_b.number_input(t("lbl_b"), 0.1, 10.0, 1.0, 0.1)
    
    if st.button(t("run_profit"), type="primary", width="stretch"):
        with st.spinner(t("spinner")):
            best_p, profit_data = generate_profit_data(c1, c2, nmax1, nmax2, a_val, b_val)
            
            if profit_data:
                df_profit = pd.DataFrame(profit_data)
                best_row = df_profit.iloc[(df_profit['Price'] - best_p).abs().argsort()[:1]].iloc[0]
                
                st.info(t("opt_result_txt").format(
                    price=best_row['Price'], 
                    demand=best_row['Demand'], 
                    n1=best_row['N1'], 
                    n2=best_row['N2'], 
                    profit=best_row['Profit']
                ))
                
                c1_opt, c2_opt, c3_opt = st.columns(3)
                c1_opt.metric(t("opt_price"), f"${best_row['Price']:,.1f}")
                c2_opt.metric(t("opt_demand"), f"{best_row['Demand']:,.1f}")
                c3_opt.metric(t("opt_profit"), f"${best_row['Profit']:,.1f}")
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df_profit['Price'], y=df_profit['Profit'], 
                    mode='lines', name='Profit', line=dict(color='green', width=3)
                ))
                fig2.add_trace(go.Scatter(
                    x=[best_row['Price']], y=[best_row['Profit']],
                    mode='markers+text', text=["Max Profit"], textposition="top center",
                    marker=dict(color='red', size=12, symbol='star')
                ))
                fig2.update_layout(
                    xaxis_title=t("plot_profit_x"), 
                    yaxis_title=t("plot_profit_y"), 
                    showlegend=False, 
                    template="plotly_white"
                )
                st.plotly_chart(fig2, width="stretch")
            else:
                st.warning("この条件下で利益が出る（生産可能な）価格帯が見つかりませんでした。")

with tab_form:
    st.header(t("tab2"))
    st.markdown(t("formulation"))
