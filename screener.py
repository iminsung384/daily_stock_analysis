"""
장기투자용 한국/미국 우량 가치주 + 눌림목 자동 스크리너
- 기존 고정 관심 종목 + 스크리너가 뽑은 한국 2종목 + 미국 2종목을 쉼표(,)로 구분해 stdout으로 출력합니다.
- 로그 메시지는 stderr로 출력하므로, 셸에서 `STOCKS=$(python screener.py)` 형태로 바로 받을 수 있습니다.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import yfinance as yf

# 기본 고정 관심 종목 (환경변수 STOCK_LIST가 설정되어 있으면 해당 목록을 우선 사용)
DEFAULT_FIXED_STOCKS = [
    "069500.KS",
    "360750.KS",
    "MSFT",
    "AMZN",
    "AAPL",
    "PLTR",
    "AMD",
    "QTUM",
    "SOXX",
]

# 한국 코스피 대표 우량 대형주 풀 (6자리 숫자 .KS 코드만 사용하여 daily_stock_analysis 파서와 100% 호환)
KR_BLUECHIPS = [
    "005930.KS",  # 삼성전자
    "000660.KS",  # SK하이닉스
    "373220.KS",  # LG에너지솔루션
    "207940.KS",  # 삼성바이오로직스
    "005380.KS",  # 현대차
    "000270.KS",  # 기아
    "068270.KS",  # 셀트리온
    "035420.KS",  # NAVER
    "035720.KS",  # 카카오
    "105560.KS",  # KB금융
    "055550.KS",  # 신한지주
    "086790.KS",  # 하나금융지주
    "005490.KS",  # POSCO홀딩스
    "051910.KS",  # LG화학
    "006400.KS",  # 삼성SDI
    "028260.KS",  # 삼성물산
    "012330.KS",  # 현대모비스
    "066570.KS",  # LG전자
    "003550.KS",  # LG
    "032830.KS",  # 삼성생명
    "015760.KS",  # 한국전력
    "034730.KS",  # SK
    "017670.KS",  # SK텔레콤
    "030200.KS",  # KT
    "096770.KS",  # SK이노베이션
    "010950.KS",  # S-Oil
    "009150.KS",  # 삼성전기
    "000810.KS",  # 삼성화재
    "018260.KS",  # 삼성에스디에스
    "033780.KS",  # KT&G
    "003670.KS",  # 포스코퓨처엠
    "086280.KS",  # 현대글로비스
    "010130.KS",  # 고려아연
    "042660.KS",  # 한화오션
    "034020.KS",  # 두산에너빌리티
    "259960.KS",  # 크래프톤
    "329180.KS",  # HD현대중공업
    "009540.KS",  # HD한국조선해양
    "012450.KS",  # 한화에어로스페이스
    "000100.KS",  # 유한양행
]

# 미국 대형 우량주 기본 풀 (Wikipedia 조회 실패 시 폴백용)
US_BLUECHIPS_FALLBACK = [
    "NVDA", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM", "V", "MA", "UNH",
    "XOM", "COST", "HD", "PG", "JNJ", "WMT", "NFLX", "CRM", "ORCL", "CVX",
    "BAC", "KO", "PEP", "ABBV", "MRK", "TMO", "ACN", "MCD", "CSCO", "LIN",
    "ADBE", "TXN", "QCOM", "INTU", "ISRG", "AMAT", "NOW", "UBER", "CAT", "GE",
]


def log(msg: str) -> None:
    print(f"[screener] {msg}", file=sys.stderr)


def get_fixed_stocks() -> list[str]:
    env_stocks = os.getenv("STOCK_LIST", "").strip()
    if env_stocks:
        return [s.strip() for s in env_stocks.split(",") if s.strip()]
    return list(DEFAULT_FIXED_STOCKS)


def get_us_universe(limit: int = 80) -> list[str]:
    """S&P 500 구성 종목을 가져오되, 실패 시 기본 우량주 풀을 반환합니다."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            storage_options={"User-Agent": "Mozilla/5.0"},
        )
        symbols = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        if symbols:
            return symbols[:limit]
    except Exception as exc:
        log(f"S&P 500 목록 조회 실패, 기본 풀 사용: {exc}")
    return US_BLUECHIPS_FALLBACK


def compute_rsi(series: pd.Series, period: int = 14) -> float | None:
    if len(series) < period + 2:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else None


def filter_technical(tickers: list[str], exclude: set[str], max_candidates: int = 15) -> list[tuple[str, float]]:
    """배치 다운로드로 RSI와 200일 이동평균선을 확인해 과열되지 않은 눌림목 후보를 추립니다."""
    pool = [t for t in tickers if t not in exclude]
    if not pool:
        return []

    try:
        data = yf.download(
            pool,
            period="1y",
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as exc:
        log(f"시세 배치 다운로드 실패: {exc}")
        return []

    candidates: list[tuple[str, float]] = []
    single_ticker = len(pool) == 1

    for ticker in pool:
        try:
            df = data if single_ticker else data[ticker]
            close = df["Close"].dropna()
            if len(close) < 120:
                continue

            rsi = compute_rsi(close)
            if rsi is None or rsi > 60:
                # RSI 60 초과는 단기 과열로 간주하여 제외
                continue

            sma200 = close.rolling(window=min(200, len(close))).mean().iloc[-1]
            latest = float(close.iloc[-1])
            if pd.isna(sma200) or latest < float(sma200) * 0.85:
                # 장기 추세가 심하게 훼손된 하락 종목 제외
                continue

            # RSI가 40에 가까울수록 건강한 눌림목으로 우선순위 부여
            candidates.append((ticker, rsi))
        except Exception:
            continue

    candidates.sort(key=lambda x: abs(x[1] - 40.0))
    return candidates[:max_candidates]


def pick_top_value_quality(candidates: list[tuple[str, float]], top_n: int = 2) -> list[str]:
    """기술적 후보 중 PER(저평가)과 ROE(수익성)가 우수한 종목을 최종 선정합니다."""
    scored: list[tuple[str, float]] = []

    for ticker, rsi in candidates:
        try:
            info = yf.Ticker(ticker).info or {}
            pe = info.get("trailingPE") or info.get("forwardPE")
            roe = info.get("returnOnEquity")

            # 적자 기업이거나 극단적 고평가(PER 35 초과) 제외
            if pe is None or pe <= 0 or pe > 35:
                continue

            # ROE 정보가 있는 경우 최소 8% 이상만 통과 (일부 한국 종목은 누락될 수 있어 None 허용)
            if roe is not None and roe < 0.08:
                continue

            pe_score = max(0.0, (35.0 - float(pe)) / 35.0) * 40.0
            roe_score = min(float(roe), 0.35) / 0.35 * 40.0 if roe is not None else 15.0
            rsi_score = max(0.0, (25.0 - abs(rsi - 40.0)) / 25.0) * 20.0

            total = pe_score + roe_score + rsi_score
            scored.append((ticker, total))
            log(f"후보 {ticker}: PER={pe:.1f}, ROE={roe}, RSI={rsi:.1f} -> 점수={total:.1f}")
        except Exception:
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]


def main() -> None:
    fixed = get_fixed_stocks()
    exclude = set(fixed)

    kr_candidates = filter_technical(KR_BLUECHIPS, exclude, max_candidates=12)
    kr_picks = pick_top_value_quality(kr_candidates, top_n=2)

    us_pool = get_us_universe(limit=80)
    us_candidates = filter_technical(us_pool, exclude, max_candidates=15)
    us_picks = pick_top_value_quality(us_candidates, top_n=2)

    final_list: list[str] = []
    for symbol in [*fixed, *kr_picks, *us_picks]:
        if symbol not in final_list:
            final_list.append(symbol)

    log(f"고정 종목({len(fixed)}): {','.join(fixed)}")
    log(f"자동 선정 한국({len(kr_picks)}): {','.join(kr_picks)}")
    log(f"자동 선정 미국({len(us_picks)}): {','.join(us_picks)}")

    print(",".join(final_list))


if __name__ == "__main__":
    main()
