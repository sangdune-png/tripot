import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="트라이팟 오리지널 백테스터", layout="wide")
st.title("🛡️ 트라이팟 퀀트 백테스터 (오리지널 룰 완벽 적용)")
st.caption("250일선 완충지대(-5%~+1%), VIX 10일선, 52주 고점 기반 1.5배수 정밀 시뮬레이션")

st.sidebar.header("⚙️ 전략 및 기간 설정")

st.sidebar.subheader("📅 백테스트 기간 설정")
year_options = list(range(1986, 2027))
month_options = list(range(1, 13))

col_s1, col_s2 = st.sidebar.columns(2)
start_year = col_s1.selectbox("시작 연도", year_options, index=14) 
start_month = col_s2.selectbox("시작 월", month_options, index=0)   

col_e1, col_e2 = st.sidebar.columns(2)
end_year = col_e1.selectbox("종료 연도", year_options, index=39)   
end_month = col_e2.selectbox("종료 월", month_options, index=11)  

start_dt = pd.Timestamp(year=start_year, month=start_month, day=1)
end_dt = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(1)

st.sidebar.subheader("💱 통화 및 자본 설정")
currency = st.sidebar.radio("표시 통화", ["USD ($)", "KRW (원)"])
exchange_rate = st.sidebar.number_input("적용 환율 (원/달러)", value=1350, step=10)
initial_capital = st.sidebar.number_input("초기 자본 (USD)", value=10000, step=1000)

st.sidebar.subheader("💰 세금 및 거래비용")
apply_tax = st.sidebar.checkbox("미국 양도세 22% 차감 적용", value=True)
tax_deduction_usd = st.sidebar.number_input("연간 기본공제 (USD)", value=2500, help="약 250만 원")
fee_rate = st.sidebar.number_input("매매 수수료 (편도 %)", value=0.1, step=0.05) / 100

st.sidebar.subheader("트라이팟 오리지널 파라미터")
st.sidebar.markdown("- 상승 전환: 250일선 + 1%\n- 하락 전환: 250일선 - 5%\n- 주의 VIX: 28 / 패닉 VIX: 18\n- 고점 낙폭 기준: 9%")

@st.cache_data
def load_40yr_data():
    raw = yf.download(["^NDX", "^VIX", "TQQQ", "QLD"], start="1985-01-01")["Close"]
    
    df = pd.DataFrame(index=raw.index)
    df["NDX"] = raw["^NDX"]
    df["VIX"] = raw["^VIX"]
    df["TQQQ_ACTUAL"] = raw["TQQQ"]
    df["QLD_ACTUAL"] = raw["QLD"]
    
    df = df.dropna(subset=["NDX"]).copy()
    df["VIX"] = df["VIX"].bfill().fillna(18.0)
    
    ndx_ret = df["NDX"].pct_change().fillna(0)
    
    daily_drag_3x = 0.015 / 252
    synth_tqqq_ret = (ndx_ret * 3.0) - daily_drag_3x
    first_tqqq_date = df["TQQQ_ACTUAL"].first_valid_index()
    first_tqqq_price = df["TQQQ_ACTUAL"].loc[first_tqqq_date]
    cum_synth_3x = (1.0 + synth_tqqq_ret).cumprod()
    scale_3x = first_tqqq_price / cum_synth_3x.loc[first_tqqq_date]
    df["TQQQ"] = df["TQQQ_ACTUAL"].combine_first(cum_synth_3x * scale_3x)

    daily_drag_2x = 0.0095 / 252
    synth_qld_ret = (ndx_ret * 2.0) - daily_drag_2x
    first_qld_date = df["QLD_ACTUAL"].first_valid_index()
    first_qld_price = df["QLD_ACTUAL"].loc[first_qld_date]
    cum_synth_2x = (1.0 + synth_qld_ret).cumprod()
    scale_2x = first_qld_price / cum_synth_2x.loc[first_qld_date]
    df["QLD"] = df["QLD_ACTUAL"].combine_first(cum_synth_2x * scale_2x)

    df["QQQ"] = df["NDX"]
    return df[["QQQ", "VIX", "TQQQ", "QLD"]]

with st.spinner("오리지널 룰 기반 데이터 산출 중..."):
    full_data = load_40yr_data()

# 오리지널 3대 지표 완벽 구현
full_data["SMA250"] = full_data["QQQ"].rolling(window=250).mean()
full_data["VIX10"] = full_data["VIX"].rolling(window=10).mean()
full_data["QQQ_52W_High"] = full_data["QQQ"].rolling(window=252).max()
full_data["QQQ_DD"] = (full_data["QQQ"] - full_data["QQQ_52W_High"]) / full_data["QQQ_52W_High"]

# 워밍업 구간 제거 후 유저 설정 기간 필터링
full_data = full_data.dropna(subset=["SMA250", "QQQ_52W_High", "VIX10"])
sim_data = full_data[(full_data.index >= start_dt) & (full_data.index <= end_dt)].copy()

# 1단계: 시장 판독 (휩쏘 방지 밴드 적용)
trend_state = []
current_trend = "Bull"
for i in range(len(sim_data)):
    qqq = sim_data["QQQ"].iloc[i]
    sma = sim_data["SMA250"].iloc[i]
    
    if qqq > sma * 1.01:
        current_trend = "Bull"
    elif qqq < sma * 0.95:
        current_trend = "Bear"
    # 그 사이 구간은 '어제 상태 유지'
    trend_state.append(current_trend)
sim_data["Trend"] = trend_state

# 2단계: 무엇을 살 것인가 (Target State 결정)
target_state = []
for i in range(len(sim_data)):
    t = sim_data["Trend"].iloc[i]
    v10 = sim_data["VIX10"].iloc[i]
    dd = sim_data["QQQ_DD"].iloc[i]
    
    if t == "Bull":
        if v10 < 28 and dd >= -0.09:
            target_state.append("TQQQ")
        else:
            target_state.append("1.5X")
    else: # Bear
        if v10 < 18:
            target_state.append("1.5X")
        else:
            target_state.append("CASH")

sim_data["Target"] = target_state
sim_data["Exec_Target"] = sim_data["Target"].shift(1).fillna("CASH")

# 3단계: 정밀 매매 엔진 ("어제와 같으면 아무것도 하지 않는다")
assets = ["TQQQ", "QQQ", "QLD"]
shares = {a: 0.0 for a in assets}
avg_cost = {a: 0.0 for a in assets}
cash = initial_capital
annual_realized_gain = 0.0
total_tax_paid = 0.0
current_year = sim_data.index[0].year

portfolio_values = []
current_state = "CASH"

for i in range(len(sim_data)):
    date = sim_data.index[i]
    year = date.year
    target = sim_data["Exec_Target"].iloc[i]
    prices = {a: sim_data[a].iloc[i] for a in assets}
    
    # 연말 양도세 정산
    if year != current_year:
        if apply_tax and annual_realized_gain > tax_deduction_usd:
            taxable = annual_realized_gain - tax_deduction_usd
            tax = taxable * 0.22
            cash -= tax
            total_tax_paid += tax
        annual_realized_gain = 0.0
        current_year = year

    # 리밸런싱 (상태가 변했을 때만 전량 교체 매매 진행)
    if target != current_state:
        # 1. 기존 보유 전량 매도
        for a in assets:
            if shares[a] > 0:
                actual_price = prices[a] * (1 - fee_rate)
                realized = (actual_price - avg_cost[a]) * shares[a]
                annual_realized_gain += realized
                cash += shares[a] * actual_price
                shares[a] = 0.0
                avg_cost[a] = 0.0
                
        # 2. 새로운 타겟 비율대로 매수
        if target == "TQQQ":
            alloc = {"TQQQ": 1.0, "QQQ": 0.0, "QLD": 0.0}
        elif target == "1.5X":
            alloc = {"TQQQ": 0.0, "QQQ": 0.5, "QLD": 0.5}
        else: # CASH
            alloc = {"TQQQ": 0.0, "QQQ": 0.0, "QLD": 0.0}
            
        total_cash_to_deploy = cash
        for a in assets:
            weight = alloc[a]
            if weight > 0:
                alloc_cash = total_cash_to_deploy * weight
                actual_price = prices[a] * (1 + fee_rate)
                shares_bought = alloc_cash / actual_price
                shares[a] += shares_bought
                avg_cost[a] = actual_price
                cash -= alloc_cash
                
        current_state = target

    # 매일 포트폴리오 가치 기록
    current_val = cash + sum(shares[a] * prices[a] for a in assets)
    portfolio_values.append(current_val)

sim_data["Portfolio"] = portfolio_values
sim_data["QQQ_Hold"] = (initial_capital / sim_data["QQQ"].iloc[0]) * sim_data["QQQ"]
sim_data["QLD_Hold"] = (initial_capital / sim_data["QLD"].iloc[0]) * sim_data["QLD"]
sim_data["TQQQ_Hold"] = (initial_capital / sim_data["TQQQ"].iloc[0]) * sim_data["TQQQ"]

def get_metrics(series):
    cagr = ((series.iloc[-1] / series.iloc[0]) ** (252 / len(series)) - 1) * 100
    peak = np.maximum.accumulate(series)
    mdd = np.min((series - peak) / peak) * 100
    return cagr, mdd

strat_cagr, strat_mdd = get_metrics(sim_data["Portfolio"])
qqq_cagr, qqq_mdd = get_metrics(sim_data["QQQ_Hold"])
qld_cagr, qld_mdd = get_metrics(sim_data["QLD_Hold"])
tqqq_cagr, tqqq_mdd = get_metrics(sim_data["TQQQ_Hold"])

if currency == "KRW (원)":
    sim_data["Portfolio"] *= exchange_rate
    sim_data["QQQ_Hold"] *= exchange_rate
    sim_data["QLD_Hold"] *= exchange_rate
    sim_data["TQQQ_Hold"] *= exchange_rate
    display_tax = total_tax_paid * exchange_rate
    curr_symbol = "₩"
else:
    display_tax = total_tax_paid
    curr_symbol = "$"

# --- 결과 출력 ---
st.subheader(f"📊 백테스트 기간: {start_dt.strftime('%Y년 %m월')} ~ {end_dt.strftime('%Y년 %m월')}")

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"트라이팟 전략", f"{curr_symbol}{sim_data['Portfolio'].iloc[-1]:,.0f}", f"CAGR {strat_cagr:.1f}% / MDD {strat_mdd:.1f}%")
col2.metric("QQQ (1배) 단순보유", f"{curr_symbol}{sim_data['QQQ_Hold'].iloc[-1]:,.0f}", f"CAGR {qqq_cagr:.1f}% / MDD {qqq_mdd:.1f}%")
col3.metric("QLD (2배) 단순보유", f"{curr_symbol}{sim_data['QLD_Hold'].iloc[-1]:,.0f}", f"CAGR {qld_cagr:.1f}% / MDD {qld_mdd:.1f}%")
col4.metric("TQQQ (3배) 단순보유", f"{curr_symbol}{sim_data['TQQQ_Hold'].iloc[-1]:,.0f}", f"CAGR {tqqq_cagr:.1f}% / MDD {tqqq_mdd:.1f}%")

if apply_tax:
    st.info(f"💡 해당 구간 누적 납부된 미국 양도소득세 총액: **{curr_symbol}{display_tax:,.0f}**")

# --- 차트 시각화 (색상 분리 완벽 적용) ---
fig = go.Figure()

fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["QQQ_Hold"], mode='lines', name='QQQ (1X)', line=dict(color='#619cff', width=1.2), hovertemplate=f'{curr_symbol}%{{y:,.0f}}'))
fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["QLD_Hold"], mode='lines', name='QLD (2X)', line=dict(color='#9c27b0', width=1.2, dash='dash'), hovertemplate=f'{curr_symbol}%{{y:,.0f}}'))
fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["TQQQ_Hold"], mode='lines', name='TQQQ (3X)', line=dict(color='#b3b3b3', width=1.2, dash='dot'), hovertemplate=f'{curr_symbol}%{{y:,.0f}}'))

fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["Portfolio"], mode='lines', line=dict(color='lightgray', width=1), showlegend=False, hoverinfo='skip'))

idx_tqqq = sim_data["Exec_Target"] == "TQQQ"
idx_15x = sim_data["Exec_Target"] == "1.5X"
idx_cash = sim_data["Exec_Target"] == "CASH"

fig.add_trace(go.Scatter(x=sim_data.index[idx_tqqq], y=sim_data["Portfolio"][idx_tqqq], mode='markers', name='TQQQ 100% (상승)', marker=dict(color='#00ba38', size=3), hovertemplate=f'TQQQ 100%: {curr_symbol}%{{y:,.0f}}'))
fig.add_trace(go.Scatter(x=sim_data.index[idx_15x], y=sim_data["Portfolio"][idx_15x], mode='markers', name='QQQ 50%+QLD 50% (1.5배수)', marker=dict(color='#e79f00', size=3), hovertemplate=f'1.5배수: {curr_symbol}%{{y:,.0f}}'))
fig.add_trace(go.Scatter(x=sim_data.index[idx_cash], y=sim_data["Portfolio"][idx_cash], mode='markers', name='현금 100% (하락)', marker=dict(color='#f8766d', size=3), hovertemplate=f'현금 방어: {curr_symbol}%{{y:,.0f}}'))

fig.update_layout(
    title="자산 성장 곡선 (로그 스케일 / 트라이팟 오리지널)",
    yaxis_type="log",
    xaxis_title="날짜",
    yaxis_title=f"계좌 평가액 ({curr_symbol})",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)
