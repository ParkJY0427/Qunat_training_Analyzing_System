import datetime
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

# Streamlit 페이지 설정
st.set_page_config(
    page_title="퀀트 분석가 | Dual-Track Multi-Factor 퀀트 데이터 파이프라인",
    layout="wide"
)

# 티커 한글 사명 매핑 딕셔너리
TICKER_NAME_MAP = {
    "NVDA": "엔비디아 (NVDA)",
    "GOOGL": "알파벳 (GOOGL)",
    "META": "메타 플랫폼스 (META)",
    "XLK": "기술 셀렉터 SPDR 펀드 (XLK)",
    "TSM": "타이완 세미컨덕터 (TSM)",
    "CONY": "YieldMax CONY 옵션 인컴 ETF (CONY)",
    "MSTY": "YieldMax MSTY 옵션 인컴 ETF (MSTY)",
    "005930.KS": "삼성전자 (005930.KS)",
    "000660.KS": "SK하이닉스 (000660.KS)",
    "012330.KS": "현대모비스 (012330.KS)"
}

# 1. 펀더멘털 데이터 수집 (NaN 및 예외 방어 강화)
def fetch_fundamentals(ticker):
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        per = info.get("trailingPE", None)
        pbr = info.get("priceToBook", None)
        div = info.get("dividendYield", None)

        per_str = f"{per:.2f}" if isinstance(per, (int, float)) else "데이터 수집 제한(N/A)"
        pbr_str = f"{pbr:.2f}" if isinstance(pbr, (int, float)) else "데이터 수집 제한(N/A)"
        div_str = f"{div * 100:.1f}%" if isinstance(div, (int, float)) else "정보 없음"

        return {"ticker": ticker, "PER": per_str, "PBR": pbr_str, "DivYield": div_str}
    except:
        return {"ticker": ticker, "PER": "데이터 수집 제한(N/A)", "PBR": "데이터 수집 제한(N/A)", "DivYield": "정보 없음"}

# 2. 거시 지표 수집기
def fetch_macro_indicators():
    try:
        macro_tickers = ["^VIX", "USDKRW=X", "^TNX"]
        macro_df = yf.download(macro_tickers, period="5d", progress=False)["Close"]
        vix_val = float(macro_df["^VIX"].dropna().iloc[-1]) if not macro_df["^VIX"].dropna().empty else 14.53
        krw_val = float(macro_df["USDKRW=X"].dropna().iloc[-1]) if not macro_df["USDKRW=X"].dropna().empty else 1351.1
        tnx_val = float(macro_df["^TNX"].dropna().iloc[-1]) if not macro_df["^TNX"].dropna().empty else 4.784
    except:
        vix_val, krw_val, tnx_val = 14.53, 1351.1, 4.784

    fear_greed_val = 50.0
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
        if res.status_code == 200:
            fear_greed_val = float(res.json()["fear_and_greed"]["score"])
    except:
        pass

    return vix_val, krw_val, tnx_val, fear_greed_val

# 3. 기술적 지표 벡터 연산기
def calculate_indicators(df_close, df_vol):
    df_calc = pd.DataFrame(index=df_close.index)
    df_calc["Close"] = df_close

    ma25 = df_close.rolling(window=25).mean()
    df_calc["Env_Upper"] = ma25 * 1.15
    df_calc["Env_Lower"] = ma25 * 0.85

    exp1 = df_close.ewm(span=12, adjust=False).mean()
    exp2 = df_close.ewm(span=26, adjust=False).mean()
    df_calc["MACD"] = exp1 - exp2
    df_calc["Signal"] = df_calc["MACD"].ewm(span=9, adjust=False).mean()
    df_calc["MACD_Hist"] = df_calc["MACD"] - df_calc["Signal"]

    delta = df_close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/18, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/18, adjust=False).mean()
    df_calc["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    if df_vol is not None and not df_vol.empty:
        direction = np.sign(delta).fillna(0)
        df_calc["OBV"] = (direction * df_vol).cumsum()
    else:
        df_calc["OBV"] = 0

    return df_calc

# 4. 화면 대시보드 전용 7단계 스코어링 함수
def evaluate_quant_opinion(close_p, env_up, env_low, macd_v, sig_v, rsi_v, obv_series):
    score = 0
    macd_diff = macd_v - sig_v
    if macd_v > sig_v:
        score += 3 if macd_diff > 0 else 2
    else:
        score += -3 if macd_diff < 0 else -2
        
    if rsi_v < 25:
        score += 3
    elif 25 <= rsi_v < 40:
        score += 2
    elif 40 <= rsi_v <= 60:
        score += 0
    elif 60 < rsi_v <= 75:
        score += 1
    elif rsi_v > 75:
        score += -3

    if close_p <= env_low:
        score += 2
    elif close_p >= env_up:
        score += -2

    if len(obv_series) >= 5:
        obv_trend = obv_series.iloc[-1] - obv_series.iloc[-5]
        if obv_trend > 0:
            score += 2
        else:
            score += -2

    if score >= 7:
        return "🚀 강력 매수 (Strong Buy)", score
    elif score >= 4:
        return "🟢 분할 매수 우위 (Accumulation)", score
    elif score >= 2:
        return "🔵 소액 저점 진입 (Light Entry)", score
    elif -1 <= score <= 1:
        return "🟡 중립 관망 (Neutral Hold)", score
    elif -3 <= score <= -2:
        return "🟠 리스크 관리 / 비중 축소 (Reduce)", score
    elif -6 <= score <= -4:
        return "🔴 분할 매도 (Scale Out)", score
    else:
        return "⚠️ 전량 현금화 / 적극 매도 (Strong Sell)", score

# 5. 화면 출력용 대시보드 요약 데이터프레임 생성
def get_quant_summary_df(tickers_str, period_yf):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    summary_list = []

    try:
        raw_data = yf.download(tickers, period=period_yf, progress=False, group_by='ticker')
    except:
        raw_data = None

    for ticker in tickers:
        display_name = TICKER_NAME_MAP.get(ticker, ticker)
        try:
            if len(tickers) > 1:
                df_close = raw_data[ticker]['Close'].dropna()
                df_vol = raw_data[ticker]['Volume'].dropna() if 'Volume' in raw_data[ticker] else None
            else:
                df_close = raw_data['Close'].dropna()
                df_vol = raw_data['Volume'].dropna() if 'Volume' in raw_data else None

            if len(df_close) < 30: continue

            df_calc = calculate_indicators(df_close, df_vol)
            last = df_calc.iloc[-1]

            close_p = last["Close"]
            env_up = last["Env_Upper"]
            env_low = last["Env_Lower"]
            macd_v = last["MACD"]
            sig_v = last["Signal"]
            rsi_v = last["RSI"]
            obv_series = df_calc["OBV"]

            if close_p >= env_up:
                env_status = "과열(고점 위험)"
            elif close_p <= env_low:
                env_status = "과매도(저점 기회)"
            else:
                env_status = "정상"

            macd_status = "골든크로스(매수우세)" if macd_v > sig_v else "데드크로스(하방압력)"
            opinion, score = evaluate_quant_opinion(close_p, env_up, env_low, macd_v, sig_v, rsi_v, obv_series)

            summary_list.append({
                "종목명": display_name,
                "현재 주가": f"{close_p:,.2f}",
                "Envelope 위치": env_status,
                "MACD 상태": macd_status,
                "RSI(18)": f"{rsi_v:.2f}",
                "멀티팩터 점수": f"{score:+d}점",
                "최종 진단": opinion
            })
        except:
            summary_list.append({
                "종목명": display_name,
                "현재 주가": "N/A",
                "Envelope 위치": "데이터 오류",
                "MACD 상태": "데이터 오류",
                "RSI(18)": "N/A",
                "멀티팩터 점수": "0점",
                "최종 진단": "🟡 중립 관망 (Neutral Hold)"
            })

    return pd.DataFrame(summary_list)

# 6. 마크다운(.md) 파일 생성 엔진
def build_markdown_report(filename, market_title, tickers_str, analysis_mode, period_yf, vix, krw, tnx, fg):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    raw_data = yf.download(tickers, period=period_yf, progress=False, group_by='ticker')
    with ThreadPoolExecutor(max_workers=10) as executor:
        fund_data = {res['ticker']: res for res in executor.map(fetch_fundamentals, tickers)}

    md_content = []
    md_content.append(f"# 📊 [Raw Data Package] {market_title} 퀀트 분석 데이터")
    md_content.append(f"- **생성 일시:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append(f"- **분석 모드:** {analysis_mode}")
    md_content.append(f"- **데이터 수집 기간:** {period_yf}")
    md_content.append(f"- **거시 지표 패키지:** VIX({vix:.2f}) | 공포탐욕지수({fg:.1f}) | 원/달러 환율({krw:,.1f}원) | 미10년물금리({tnx:.3f}%)")
    md_content.append("\n---\n")

    for ticker in tickers:
        display_name = TICKER_NAME_MAP.get(ticker, ticker)
        fund = fund_data.get(ticker, {})

        try:
            if len(tickers) > 1:
                df_close = raw_data[ticker]['Close'].dropna()
                df_vol = raw_data[ticker]['Volume'].dropna() if 'Volume' in raw_data[ticker] else None
            else:
                df_close = raw_data['Close'].dropna()
                df_vol = raw_data['Volume'].dropna() if 'Volume' in raw_data else None

            if len(df_close) < 30: continue

            df_calc = calculate_indicators(df_close, df_vol)
            last = df_calc.iloc[-1]

            close_p = last["Close"]
            env_up = last["Env_Upper"]
            env_low = last["Env_Lower"]
            macd_v = last["MACD"]
            sig_v = last["Signal"]
            hist_v = last["MACD_Hist"]
            rsi_v = last["RSI"]
            obv_v = last["OBV"]
            
            hist_mean = df_close.mean()
            price_vs_mean_pct = ((close_p - hist_mean) / hist_mean) * 100
            
            ret_1m = ((close_p - df_close.iloc[-20]) / df_close.iloc[-20]) * 100 if len(df_close) >= 20 else 0.0
            ret_3m = ((close_p - df_close.iloc[-60]) / df_close.iloc[-60]) * 100 if len(df_close) >= 60 else 0.0
            ret_6m = ((close_p - df_close.iloc[-120]) / df_close.iloc[-120]) * 100 if len(df_close) >= 120 else 0.0

            daily_ret = df_close.pct_change().dropna()
            annualized_vol = daily_ret.std() * np.sqrt(252) * 100
            rolling_max = df_close.cummax()
            drawdown = (df_close - rolling_max) / rolling_max
            mdd = drawdown.min() * 100

            dist_from_upper = ((close_p - env_up) / env_up) * 100
            dist_from_lower = ((close_p - env_low) / env_low) * 100

            md_content.append(f"## 📌 Ticker: {display_name} (`{ticker}`)")
            md_content.append(f"### 1. Price, Return & Risk Metrics")
            md_content.append(f"| Metric | Value | Description |")
            md_content.append(f"| :--- | :--- | :--- |")
            md_content.append(f"| Current Close | `{close_p:,.2f}` | 최신 종가 |")
            md_content.append(f"| Historical Mean | `{hist_mean:,.2f}` | 기간 평균 주가 |")
            md_content.append(f"| Mean Deviation (%) | `{price_vs_mean_pct:+.2f}%` | 평균 대비 괴리율 |")
            md_content.append(f"| 1M / 3M / 6M Return | `{ret_1m:+.2f}%` / `{ret_3m:+.2f}%` / `{ret_6m:+.2f}%` | 단중기 모멘텀 수익률 |")
            md_content.append(f"| Annualized Volatility | `{annualized_vol:.2f}%` | 연환산 변동성 |")
            md_content.append(f"| Maximum Drawdown (MDD) | `{mdd:.2f}%` | 기간 내 최대 낙폭 |")
            md_content.append(f"| Cumulative OBV | `{obv_v:,.0f}` | 누적 거래량 수급 지표 |")

            md_content.append(f"\n### 2. Technical Indicators (Granular Raw Values)")
            md_content.append(f"| Indicator | Parameter / Raw Value | Detailed Status / Deviation |")
            md_content.append(f"| :--- | :--- | :--- |")
            md_content.append(f"| Envelope (25, 15%) | Upper: `{env_up:,.2f}` <br> Lower: `{env_low:,.2f}` | 상단 대비: `{dist_from_upper:+.2f}%` <br> 하단 대비: `{dist_from_lower:+.2f}%` |")
            md_content.append(f"| MACD (12, 26, 9) | MACD: `{macd_v:.4f}` <br> Signal: `{sig_v:.4f}` | 히스토그램: `{hist_v:.4f}` <br> 교차상태: `{'Golden Cross' if macd_v > sig_v else 'Dead Cross'}` |")
            md_content.append(f"| RSI | Period: 18 <br> Value: `{rsi_v:.2f}` | 모멘텀 구간 수치 완충 완료 |")

            md_content.append(f"\n### 3. Fundamental Data")
            md_content.append(f"| Factor | Value |")
            md_content.append(f"| :--- | :--- |")
            md_content.append(f"| PER (Trailing) | `{fund.get('PER', 'N/A')}` |")
            md_content.append(f"| PBR | `{fund.get('PBR', 'N/A')}` |")
            md_content.append(f"| Dividend Yield | `{fund.get('DivYield', 'N/A')}` |")

            md_content.append("\n---\n")

        except Exception as e:
            continue

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    return filename

# --- Streamlit UI 구현 ---
st.title("🤖 20년 경력 퀀트 분석가 | Dual-Track Multi-Factor 데이터 파이프라인")
st.markdown("대시보드에서 요약 표를 확인하고, **AI 분석용 순수 팩트 마크다운 파일(.md)**을 다운로드할 수 있습니다.")

col1, col2 = st.columns(2)

with col1:
    ov_box = st.text_input("🌍 해외 시장 티커 (쉼표로 구분)", value="NVDA, GOOGL, META, CONY, MSTY")
    dom_box = st.text_input("🇰🇷 국내 시장 티커 (쉼표로 구분)", value="005930.KS, 000660.KS, 012330.KS")

with col2:
    mode_radio = st.radio("분석 모드 선택", ["단기 모멘텀 트레이딩 진단", "중장기 펀더멘털 진단"])
    period_radio = st.radio("기술적 지표 수집 기간 선택", ["1년 (1y)", "3년 (3y)", "5년 (5y)"], index=1)

# 세션 상태 초기화 (데이터 유지용)
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.ov_summary_df = None
    st.session_state.dom_summary_df = None
    st.session_state.fn_overseas = None
    st.session_state.fn_domestic = None

if st.button("🚀 퀀트 분석 실행 및 데이터 파일 생성", type="primary"):
    with st.spinner("데이터 수집 및 팩터 연산 중..."):
        period_map = {
            "1년 (1y)": "1y",
            "3년 (3y)": "3y",
            "5년 (5y)": "5y"
        }
        period_yf = period_map.get(period_radio, "3y")

        vix, krw, tnx, fg = fetch_macro_indicators()

        st.session_state.ov_summary_df = get_quant_summary_df(ov_box, period_yf)
        st.session_state.dom_summary_df = get_quant_summary_df(dom_box, period_yf)

        date_prefix = datetime.datetime.now().strftime("%y.%m.%d")
        clean_mode = mode_radio.replace(" ", "_")
        clean_period = period_radio.split(" ")[0]

        st.session_state.fn_overseas = build_markdown_report(
            f"{date_prefix}_{clean_mode}({clean_period})_해외주식_데이터패키지.md", 
            "해외 시장", ov_box, mode_radio, period_yf, vix, krw, tnx, fg
        )
        st.session_state.fn_domestic = build_markdown_report(
            f"{date_prefix}_{clean_mode}({clean_period})_국내주식_데이터패키지.md", 
            "국내 시장", dom_box, mode_radio, period_yf, vix, krw, tnx, fg
        )
        
        st.session_state.analyzed = True

# 분석이 완료된 상태라면 세션에 저장된 데이터를 화면에 계속 유지
if st.session_state.analyzed:
    st.success("✅ 마크다운 데이터 패키지 파일 빌드 완료!")

    st.subheader("📊 대시보드 요약 표: 해외 시장")
    st.dataframe(st.session_state.ov_summary_df, use_container_width=True)

    st.subheader("📊 대시보드 요약 표: 국내 시장")
    st.dataframe(st.session_state.dom_summary_df, use_container_width=True)

    st.subheader("📥 데이터 패키지 다운로드")
    dcol1, dcol2 = st.columns(2)
    
    with dcol1:
        if st.session_state.fn_overseas and os.path.exists(st.session_state.fn_overseas):
            with open(st.session_state.fn_overseas, "rb") as f:
                st.download_button(
                    label="📥 해외 시장 데이터 패키지 (.md)",
                    data=f,
                    file_name=st.session_state.fn_overseas,
                    mime="text/markdown"
                )
            
    with dcol2:
        if st.session_state.fn_domestic and os.path.exists(st.session_state.fn_domestic):
            with open(st.session_state.fn_domestic, "rb") as f:
                st.download_button(
                    label="📥 국내 시장 데이터 패키지 (.md)",
                    data=f,
                    file_name=st.session_state.fn_domestic,
                    mime="text/markdown"
                )
