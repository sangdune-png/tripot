import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 페이지 기본 설정 (모바일 최적화) ---
st.set_page_config(page_title="트라이팟(Tripod) 백테스터", layout="wide")
st.title("🛡️ 트라이팟 퀀트 백테스터")
st.caption("나스닥 250일선 + VIX + 고점 낙폭 기반 동적 배분 & 미국 양도세 22% 반영")

# --- 사이드바 파라미터 설정 ---
st.sidebar.header("⚙️ 전략 및 비용 설정")

start_year = st.sidebar.slider("시작 연도", 2011, 2024, 2015)
initial_capital = st.sidebar.number_input("초기 자본 ($)", value=10000, step=1000)

st.sidebar.subheader("세금 및 거래비용")
apply_tax = st.sidebar.checkbox("미국 양도세 22% 차감 적용", value=True)
tax_deduction_usd = st.sidebar.number_input("연간 기본공제 ($)", value=2000, help="약 250만 원 상당 달러 기준")
fee_rate = st.sidebar.number_input("매매 수수료 + 슬리피지 (편도 %)", value=0.1, step=0.05) / 100

st.sidebar.subheader("트라이팟 3대 조건 기준치")
vix_panic_threshold = st.sidebar.slider("VIX 공포 임계치", 20, 40, 28)
mdd_warning_threshold = st.sidebar.slider("나스닥 전고점 대비 하락 경보 (-%)", -30, -5, -10) / 100

st.sidebar.subheader("비중 배분 규칙 (TQQQ : 현금)")
alloc_normal = st.sidebar.slider("1. 정상 구간 (250일선 상회 & 안정)", 0, 100, 100) / 100
alloc_caution = st.sidebar.slider("2. 주의 구간 (낙폭 발생 or VIX 상승)", 0, 100, 50) / 100
alloc_bear = st.sidebar.slider("3. 위험/하락장 (250일선 하회 & 고변동성)", 0, 100, 0) / 100

# --- 데이터 로드 (캐싱 적용) ---
@st.cache_data
def load_data(start_date):
    tickers = ["QQQ", "^VIX", "TQQQ"]
    df = yf.download(tickers, start=start_date)["Close"]
    df = df.dropna()
    df.columns = ["QQQ", "TQQQ", "VIX"]
    return df

with st.spinner("과거 금융 데이터를 불러오는 중..."):
    data = load_data(f"{start_year}-01-01")

# --- 트라이팟 3대 지표 계산 ---
data["QQQ_SMA250"] = data["QQQ"].rolling(window=250).mean()
data["QQQ_Peak"] = data["QQQ"].cummax()
data["QQQ_DD"] = (data["QQQ"] - data["QQQ_Peak"]) / data["QQQ_Peak"]

# 250일선 워밍업 이후 데이터만 추출
sim_data = data.dropna().copy()

# 시그널 생성 (당일 종가 기준 판독 -> 익일 장 시작 시 체결 원칙)
signals = []
for i in range(len(sim_data)):
    qqq = sim_data["QQQ"].iloc[i]
    sma = sim_data["QQQ_SMA250"].iloc[i]
    vix = sim_data["VIX"].iloc[i]
    dd = sim_data["QQQ_DD"].iloc[i]

    # 조건 판단
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

# --- 정밀 회계 엔진 (이동평균법 평단가 + 연말 양도세 정산) ---
dates = sim_data.index
prices = sim_data["TQQQ"].values
allocs = sim_data["Exec_Alloc"].values

cash = initial_capital
shares = 0.0
avg_cost = 0.0
annual_realized_gain = 0.0
current_year = dates[0].year

portfolio_values = []
benchmark_values = (initial_capital / prices[0]) * prices  # TQQQ 단순보유
qqq_benchmark = (initial_capital / sim_data["QQQ"].values[0]) * sim_data["QQQ"].values

total_tax_paid = 0.0

for i, date in enumerate(dates):
    year = date.year
    price = prices[i]
    alloc = allocs[i]

    # 연도 변경 시: 전년도 양도세 22% 일시 차감 정산
    if year != current_year:
        if apply_tax and annual_realized_gain > tax_deduction_usd:
            taxable = annual_realized_gain - tax_deduction_usd
            tax = taxable * 0.22
            cash -= tax
            total_tax_paid += tax
        annual_realized_gain = 0.0
        current_year = year

    # 당일 총 포트폴리오 가치
    total_val = cash + (shares * price)
    target_equity_val = total_val * alloc
    current_equity_val = shares * price
    diff_val = target_equity_val - current_equity_val

    # 리밸런싱 주문 체결
    if diff_val > 0:  # 매수
        buy_val = diff_val
        actual_price = price * (1 + fee_rate)
        shares_to_buy = buy_val / actual_price
        if cash >= buy_val:
            # 평단가 이동평균법 갱신
            new_shares = shares + shares_to_buy
            avg_cost = ((shares * avg_cost) + (shares_to_buy * actual_price)) / new_shares
            shares = new_shares
            cash -= buy_val
    elif diff_val < 0:  # 매도 (비중 축소)
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
sim_data["TQQQ_Hold"] = benchmark_values
sim_data["QQQ_Hold"] = qqq_benchmark

# --- 성과 지표 계산 함수 ---
def get_metrics(series):
    cagr = ((series[-1] / series[0]) ** (252 / len(series)) - 1) * 100
    peak = np.maximum.accumulate(series)
    mdd = np.min((series - peak) / peak) * 100
    return cagr, mdd

strat_cagr, strat_mdd = get_metrics(sim_data["Portfolio"].values)
tqqq_cagr, tqqq_mdd = get_metrics(sim_data["TQQQ_Hold"].values)
qqq_cagr, qqq_mdd = get_metrics(sim_data["QQQ_Hold"].values)

# --- 결과 지표 출력 (모바일 카드 형태) ---
col1, col2, col3 = st.columns(3)
col1.metric("트라이팟 전략", f"${sim_data['Portfolio'].iloc[-1]:,.0f}", f"CAGR {strat_cagr:.1f}% / MDD {strat_mdd:.1f}%")
col2.metric("QQQ 단순보유", f"${sim_data['QQQ_Hold'].iloc[-1]:,.0f}", f"CAGR {qqq_cagr:.1f}% / MDD {qqq_mdd:.1f}%")
col3.metric("TQQQ 단순보유", f"${sim_data['TQQQ_Hold'].iloc[-1]:,.0f}", f"CAGR {tqqq_cagr:.1f}% / MDD {tqqq_mdd:.1f}%")

if apply_tax:
    st.info(f"💡 백테스트 기간 동안 누적 납부된 미국 양도소득세 총액: **${total_tax_paid:,.0f}**")

# --- 차트 시각화 ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["Portfolio"], mode='lines', name='트라이팟 (세후)', line=dict(color='#00ba38', width=2)))
fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["QQQ_Hold"], mode='lines', name='QQQ 보유', line=dict(color='#619cff', width=1.5)))
fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data["TQQQ_Hold"], mode='lines', name='TQQQ 보유', line=dict(color='#f8766d', width=1, dash='dot')))

fig.update_layout(
    title="자산 성장 곡선 (로그 스케일)",
    yaxis_type="log",
    xaxis_title="날짜",
    yaxis_title="계좌 평가액 ($)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# 지표 상태 현황 테이블
st.subheader("최근 10거래일 트라이팟 판독 현황")
recent_df = sim_data[["QQQ", "QQQ_SMA250", "VIX", "QQQ_DD", "Exec_Alloc"]].tail(10).copy()
recent_df["QQQ_DD"] = (recent_df["QQQ_DD"] * 100).round(2).astype(str) + "%"
recent_df["Exec_Alloc"] = (recent_df["Exec_Alloc"] * 100).round(0).astype(str) + "% (TQQQ)"
st.dataframe(recent_df, use_container_width=True)