# Strategic Production Optimizer 🏭 📈

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-app-url.streamlit.app/](https://strategic-appuction-optimizer-eega7fc6bcgrgaqe9usuc6.streamlit.app/))

**Strategic Production Optimizer** is a professional, interactive web application built with Streamlit and Pyomo. It demonstrates how mathematical optimization (Linear Programming) can be directly translated into **Strategic Business Decisions** such as pricing, demand generation, and capital investment.

[🇯🇵 日本語のREADMEはこちら](#japanese-readme)

## 🌟 Key Features

1. **ROI-Driven Investment Insights (Shadow Prices)**
   Instead of just showing "how much to produce", this app uses **Dual Variables (Shadow Prices)** to calculate the true Marginal Value of expanding factory capacity. It automatically compares this Marginal Value against Investment Costs to deliver clear **"Invest"** or **"Pass"** recommendations.
2. **Profit Maximization & Pricing Strategy (Scipy + Pyomo)**
   Integrates a dynamic price-demand curve ($Demand = a - b \times Price$). The app performs a continuous optimization (using `scipy.optimize.minimize_scalar`) alongside Pyomo LP models to find the exact global optimal selling price that maximizes net profit.
3. **High-Performance Architecture (Warm-start)**
   Uses Pyomo's `Mutable Parameters` and Streamlit's `@st.cache_data`. Instead of rebuilding the optimization model on every iteration, the app reuses the pre-compiled structure and only injects new parameters. This enables extremely fast real-time sensitivity analysis sweeps.
4. **Bilingual UI**
   Fully supports English and Japanese through an intuitive sidebar toggle.

## 🛠 Tech Stack
- **Frontend & UI**: Streamlit, Plotly
- **Mathematical Optimization**: Pyomo, CBC Solver
- **Continuous Optimization**: SciPy, NumPy
- **Data Handling**: Pandas

## 🚀 How to Run Locally

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install the CBC Solver (Mac/Linux):
   ```bash
   brew tap coin-or-tools/coinor
   brew install cbc
   ```
   *(For Ubuntu/Debian: `sudo apt-get install coinor-cbc coinor-libcbc-dev`)*
3. Run the app:
   ```bash
   streamlit run strategic_production_optimizer.py
   ```

---
<a name="japanese-readme"></a>
# Strategic Production Optimizer 🏭 📈 (日本語)

**Strategic Production Optimizer** は、数理最適化（線形計画法）を単なる計算ツールで終わらせず、**「経営戦略・投資判断」**へと昇華させるためのプロフェッショナルなStreamlitダッシュボードです。

## 🌟 主な機能とビジネス価値

1. **投資対効果 (ROI) の自動判定**
   Pyomo/CBCソルバーから得られる**双対変数（シャドウプライス）**を「限界価値」として解釈。ボトルネックとなっている機械の増設コストと限界価値を比較し、**「投資推奨」**または**「投資見送り」**のインサイトを自動で出力します。
2. **価格戦略と利益最大化の融合**
   「価格を下げれば需要が増える」という需要曲線をモデルに組み込みました。LP（線形計画）の枠を超え、`scipy.optimize` の連続最適化と連携させることで、**「利益が最大となる最適な販売価格」**を正確に算出・可視化します。
3. **実務レベルの高速アーキテクチャ**
   Pyomoの `Mutable Parameters`（変更可能なパラメータ）と Streamlitのキャッシュを活用。感度分析などでモデルを何十回も解く際、毎回モデルを再構築するのではなく**構造を使い回して（Warm-start）計算を爆速化**しています。

## 🚀 ローカル環境での動かし方

1. Pythonパッケージのインストール:
   ```bash
   pip install -r requirements.txt
   ```
2. CBCソルバーのインストール (Mac):
   ```bash
   brew tap coin-or-tools/coinor
   brew install cbc
   ```
3. アプリの起動:
   ```bash
   streamlit run strategic_production_optimizer.py
   ```
