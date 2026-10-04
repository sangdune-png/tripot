import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="트라이팟 40년 백테스터", layout="wide")
st.title("🛡️ 트라이팟 퀀트 백테스터")
st.caption("나스닥 100 기반 동적 배분 & 미국 양도세 22% 반영 (TQQQ vs QLD 비교)")

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 전략 및 기간 설정")

target_asset = st.sidebar.selectbox("운용 대상 레버리지 ETF", ["TQQQ (3배)", "QLD (2배)"])
asset_key = "TQQQ" if "TQQQ" in target_asset else "QLD"

st.sidebar.subheader("📅 백테스트 기간 설정")
year_options = list(range(1986, 2027))
month_options = list(range(1, 13))

col_s1, col_s2 = st.sidebar.columns(2)
start_year = col_s1.selectbox("시작 연도", year_options, index=14) # 2000년
start_month = col_s2.selectbox("시작 월", month_options, index=0)   

col_e1, col_e2 = st.sidebar.columns(2)
end_year = col_e1.selectbox("종료 연도", year_options, index=39)   # 2025년
end_month = col_e2.selectbox("종료 월", month_options, index=11)  

start_dt = pd.Timestamp(year=start_year, month=start_month, day=1)
end_dt = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(1)

if start_dt >= end_dt:
    st.sidebar.error("⚠️ 시작 시점이 종료 시점보다 앞서야 합니다.")
    st.stop()

st.sidebar.subheader("💱 통화 환산 설정")
currency = st.sidebar.radio("표시 통화", ["USD ($)", "KRW (원)"])
exchange_rate = st.sidebar.number_input("적용 환율 (원/달러)", value=1350, step=10)

st.sidebar.subheader("💰 자본 및 세금")
initial_capital = st.sidebar.number_input("초기 자본 (USD)", value=10000, step=1000)
apply_tax = st.sidebar.checkbox("미국 양도세 22% 차감 적용", value=True)
tax_deduction_usd = st.sidebar.number_input("연간 기본공제 (USD)", value=2000, help="약 250만 원 상당 달러 기준")
fee_rate = st.sidebar.number_input("매매 수수료 + 슬리피지 (편도 %)", value=0.1, step=0.05) / 100

st.sidebar.subheader("트라이팟 3대 조건 기준치")
vix_panic_threshold = st.sidebar.slider("VIX 공포 임계치", 20, 40, 28)
mdd_warning_threshold = st.sidebar.slider("나스닥 전고점 대비 하락 경보 (-%)", -30, -5, -10) / 100

st.sidebar.subheader(f"비중 배분 규칙 ({asset_key} : 현금)")
alloc_normal = st.sidebar.slider("1. 정상 구간 (250일선 상회 & 안정)", 0, 100, 100) / 100
alloc_caution = st.sidebar.slider("2. 주의 구간 (낙폭 발생 or VIX 상승)", 0, 100, 50) / 100
alloc_bear = st.sidebar.slider("3. 위험/하락장 (250일선 하회 & 고변동성)", 0, 100, 0) / 100

# --- 40년 치 원천 데이터 수집 ---
@st.cache_data
def load_40yr_data():
    raw = yf.download(["^NDX", "^VIX", "TQQQ", "QLD"], start="1985-10-01")["Close"]
    
    df = pd.DataFrame(index=raw.index)
    df["NDX"] = raw["^NDX"]
    df["VIX"] = raw["^VIX"]
    df["TQQQ_ACTUAL"] = raw["TQQQ"]
    df["QLD_ACTUAL"] = raw["QLD"]
    
    df = df.dropna(subset=["NDX"]).copy()
    df["VIX"] = df["VIX"].bfill().fillna(18.0)
    
    ndx_ret = df["NDX"].pct_change().fillna(0)
    
    # 합성 TQQQ (3X)
    daily_drag_3x = 0.015 / 252
    synth_tqqq_ret = (ndx_ret * 3.0) - daily_drag_3x
    first_tqqq_date = df["TQQQ_ACTUAL"].first_valid_index()
    first_tqqq_price = df["TQQQ_ACTUAL"].loc[first_tqqq_date]
    cum_synth_3x = (1.0 + synth_tqqq_ret).cumprod()
    scale_3x = first_tqqq_price / cum_synth_3x.loc[first_tqqq_date]
    synth_tqqq = cum_synth_3x * scale_3x
    df["TQQQ"] = df["TQQQ_ACTUAL"].combine_first(synth_tqqq)

    # 합성 QLD (2X)
    daily_drag_2x = 0.0095 / 252
    synth_qld_ret = (ndx_ret * 2.0) - daily_drag_2x
    first_qld_date = df["QLD_ACTUAL"].first_valid_index()
    first_qld_price = df["QLD_ACTUAL"].loc[first_qld_date]
    cum_synth_2x = (1.0 + synth_qld_ret).cumprod()
    scale_2x = first_qld_price / cum_synth_2x.loc[first_qld_date]
    synth_qld = cum_synth_2x * scale_2x
    df["QLD"] = df["QLD_ACTUAL"].combine_first(synth_qld)

    df["QQQ"] = df["NDX"]
    return df[["QQQ", "VIX", "TQQQ", "QLD"]]

with st.spinner("금융 데이터 로딩 및 인덱스 정합성 검증 중..."):
    full_data = load_40yr_data()

# 지표 워밍업 왜곡 방지
full_data["QQQ_SMA250"] = full_data["QQQ"].rolling(window=250).mean()
full_data["QQQ_Peak"] = full_data["QQQ"].cummax()
full_data["QQQ_DD"] = (full_data["QQQ"] - full_data["QQQ_Peak"]) / full_data["QQQ_Peak"]

sim_data = full_data.dropna(subset=["QQQ_SMA250"]).copy()
sim_data = sim_data[(sim_data.index >= start_dt) & (sim_data.index <= end_dt)].copy()

if len(sim_data) < 10:
    st.warning("선택하신 기간의 유효 거래일 데이터가 부족합니다.")
    st.stop()

# 시그널 판독
signals = []
for i in range(len(sim_data)):
    qqq = sim_data["QQQ"].iloc[i]
    sma = sim_data["QQQ_SMA250"].iloc[i]
    vix = sim_data["VIX"].iloc[i]
    dd = sim_data["QQQ_DD"].iloc[i]

    is_above_sma = qqq >= sma
    is_vix_safe = vix < vix_panic_threshold
    is_dd_safe = dd >= mdd_warning_threshold

    if is_above_sma and is_vix_safe and is_dd_safe:
        target = alloc_normal
    elif not is_above_sma and (not is_vix_safe or not is_dd_safe):
        target = alloc_bear
    else:
        target = alloc_caution
    signals.append(target)

sim_data["Target_Alloc"] = signals
sim_data["Exec_Alloc"] = sim_data["Target_Alloc"].shift(1).fillna(alloc_normal)

# --- 정밀 회계 엔진 (세금 포함) ---
dates = sim_data.index
prices = sim_data[asset_key].values
allocs = sim_data["Exec_Alloc"].values

cash = initial_capital
shares = 0.0
avg_cost = 0.0
annual_realized_gain = 0.0
current_year = dates[0].year

portfolio_values = []
qqq_benchmark = (initial_capital / sim_data["QQQ"].values[0]) * sim_data["QQQ"].values
tqqq_benchmark = (initial_capital / sim_data["TQQQ"].values[0]) * sim_data["TQQQ"].values
qld_benchmark = (initial_capital / sim_data["QLD"].values[0]) * sim_data["QLD"].values

total_tax_paid = 0.0

for i, date in enumerate(dates):
    year = date.year
    price = prices[i]
    alloc = allocs[i]

    if year != current_year:
        if apply_tax and annual_realized_gain > tax_deduction_usd:
            taxable = annual_realized_gain - tax_deduction_usd
            tax = taxable * 0.22
            cash -= tax
            total_tax_paid += tax
        annual_realized_gain = 0.0
        current_year = year

    total_val = cash + (shares * price)
    target_equity_val = total_val * alloc
    current_equity_val = shares * price
    diff_val = target_equity_val - current_equity_val

    if diff_val > 0:
        buy_val = diff_val
        actual_price = price * (1 + fee_rate)
        shares_to_buy = buy_val / actual_price
        if cash >= buy_val:
            new_shares = shares + shares_to_buy
            avg_cost = ((shares * avg_cost) + (shares_to_buy * actual_price)) / new_shares
            shares = new_shares
            cash -= buy_val
    elif diff_val < 0:
        sell_val = abs(diff_val)
        actual_price = price * (1 - fee_rate)
        shares_to_sell = min(shares, sell_val / price)
        if shares_to_sell > 0:
            realized = (actual_price - avg_cost) * shares_to_sell
            annual_realized_gain += realized
            shares -= shares_to_sell
            cash += shares_to_sell * actual_price
            if shares == 0:
                avg_cost = 0.0

    current_val = cash + (shares * price)
    portfolio_values.append(current_val)

sim_data["Portfolio"] = portfolio_values
sim_data["QQQ_Hold"] = qqq_benchmark
sim_data["QLD_Hold"] = qld_benchmark
sim_data["TQQQ_Hold"] = tqqq_benchmark

def get_metrics(series):
    cagr = ((series[-1] / series[0]) ** (252 / len(series)) - 1) * 100
    peak = np.maximum.accumulate(series)
    mdd = np.min((series - peak) / peak) * 100
    return cagr, mdd

strat_cagr, strat_mdd = get_metrics(sim_data["Portfolio"].values)
qqq_cagr, qqq_mdd = get_metrics(sim_data["QQQ_Hold"].values)
qld_cagr, qld_mdd = get_metrics(sim_data["QLD_Hold"].values)
tqqq_cagr, tqqq_mdd = get_metrics(sim_data["TQQQ_Hold"].values)

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
col1.metric(f"트라이팟 전략 ({asset_key})", f"{curr_symbol}{sim_data['Portfolio'].iloc[-1]:,.0f}", f"CAGR {strat_cagr:.1f}% / MDD {strat_mdd:.1f}%")
col2.metric("QQQ (1배) 단순보유", f"{curr_symbol}{sim_data['QQQ_Hold'].iloc[-1]:,.0f}", f"CAGR {qqq_cagr:.1f}% / MDD {qqq_mdd:.1f}%")
col3.metric("QLD (2배) 단순보유", f"{curr_symbol}{sim_data['QLD_Hold'].iloc[-1]:,.0f}", f"CAGR {qld_cagr:.1f}% / MDD {qld_mdd:.1f}%")
col4.metric("TQQQ (3배) 단순보유", f"{curr_symbol}{sim_data['TQQQ_Hold'].iloc[-1]:,.0f}", f"CAGR {tqqq_cagr:.1f}% / MDD {tqqq_mdd:.1f}%")

if apply_tax:
    st.info(f"💡 해당 구간 누적 납부된 미국 양도소득세 총액: **{curr_symbol}{display_tax:,.0f}**")

# --- 차트 시각화 (색상 분리 핵심 코드) ---
fig = go.Figure()

# 1. 벤치마크 선들
fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["QQQ_Hold"], mode='lines', name='QQQ (1X)', line=dict(color='#619cff', width=1.2), hovertemplate=f'{curr_symbol}%{{y:,.0f}}'))
fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["QLD_Hold"], mode='lines', name='QLD (2X)', line=dict(color='#9c27b0', width=1.2, dash='dash'), hovertemplate=f'{curr_symbol}%{{y:,.0f}}'))
fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["TQQQ_Hold"], mode='lines', name='TQQQ (3X)', line=dict(color='#b3b3b3', width=1.2, dash='dot'), hovertemplate=f'{curr_symbol}%{{y:,.0f}}'))

# 2. 트라이팟 베이스 라인 (끊김 방지용 연한 선)
fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["Portfolio"], mode='lines', line=dict(color='lightgray', width=1), showlegend=False, hoverinfo='skip'))

# 3. 비중별 색상 마커 덧칠 (초록, 노랑, 빨강)
idx_normal = np.isclose(sim_data["Target_Alloc"], alloc_normal)
idx_caution = np.isclose(sim_data["Target_Alloc"], alloc_caution)
idx_bear = np.isclose(sim_data["Target_Alloc"], alloc_bear)

fig.add_trace(go.Scatter(x=sim_data.index[idx_normal], y=sim_data["Portfolio"][idx_normal], mode='markers', name=f'{asset_key} (정상 100%)', marker=dict(color='#00ba38', size=3), hovertemplate=f'정상진입: {curr_symbol}%{{y:,.0f}}'))
fig.add_trace(go.Scatter(x=sim_data.index[idx_caution], y=sim_data["Portfolio"][idx_caution], mode='markers', name='1.5배수 (비중 50%)', marker=dict(color='#e79f00', size=3), hovertemplate=f'절반매도: {curr_symbol}%{{y:,.0f}}'))
fig.add_trace(go.Scatter(x=sim_data.index[idx_bear], y=sim_data["Portfolio"][idx_bear], mode='markers', name='현금 (위험 0%)', marker=dict(color='#f8766d', size=3), hovertemplate=f'현금방어: {curr_symbol}%{{y:,.0f}}'))

fig.update_layout(
    title=f"자산 성장 곡선 (로그 스케일 / 색상별 전략 표기)",
    yaxis_type="log",
    xaxis_title="날짜",
    yaxis_title=f"계좌 평가액 ({curr_symbol})",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)
