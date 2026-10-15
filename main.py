
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온 분석 · 선형 회귀 · 장기 및 최근 추세 비교")

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
END_YEAR = 2025
MIN_DAYS = 300


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = ["날짜", "지점", "평균기온", "최저기온", "최고기온"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필요한 열이 없습니다: {', '.join(missing)}")

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(), errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= END_YEAR]

    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    return (
        yearly[yearly["관측일수"] >= MIN_DAYS]
        .sort_values("연도")
        .reset_index(drop=True)
    )


def fit_trend(data):
    """연도와 연평균기온의 선형 회귀 결과를 반환한다."""
    x = data["연도"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    if len(data) < 2 or np.unique(x).size < 2:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    correlation = np.corrcoef(x, y)[0, 1]

    return {
        "slope": slope,
        "intercept": intercept,
        "per_100_years": slope * 100,
        "correlation": correlation,
        "count": len(data),
        "start": int(data["연도"].min()),
        "end": int(data["연도"].max())
    }


try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러올 수 없습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석에 사용할 연도가 충분하지 않습니다.")
    st.stop()

# 전체 기간 회귀
overall = fit_trend(yearly)

# 최근 20년: 분석에 사용되는 가장 최근 연도를 기준으로 20년 구간 설정
latest_year = int(yearly["연도"].max())
recent_start = latest_year - 19
recent_data = yearly[yearly["연도"] >= recent_start].copy()
recent = fit_trend(recent_data)

if recent is None:
    st.error("최근 20년의 회귀 분석에 필요한 데이터가 부족합니다.")
    st.stop()

# -----------------------------
# 기울기 비교: 100년당 변화량
# -----------------------------
st.subheader("📈 100년당 기온 변화량 비교")

st.caption(
    f"전체 기간: {overall['start']}~{overall['end']}년 · "
    f"최근 20년 구간: {recent_start}~{latest_year}년"
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"전체 기간 ({overall['count']}개 연도)",
        value=f"{overall['per_100_years']:+.2f} °C",
        help="전체 분석 기간의 선형 회귀 기울기를 100배 한 값"
    )
    st.caption(
        f"분석 구간: {overall['start']}~{overall['end']}년"
    )

with col2:
    st.metric(
        label=f"최근 20년 구간 ({recent['count']}개 연도)",
        value=f"{recent['per_100_years']:+.2f} °C",
        help="최근 20년 구간에 포함되는 유효 연도만으로 계산한 값"
    )
    st.caption(
        f"분석 구간: {recent['start']}~{recent['end']}년"
    )

difference = recent["per_100_years"] - overall["per_100_years"]
st.info(
    f"최근 구간과 전체 기간의 차이: {difference:+.2f} °C / 100년"
)

st.caption(
    "양수는 기온 상승 추세, 음수는 하강 추세를 뜻합니다. "
    "이는 실제 100년 동안 관측된 변화가 아니라 회귀 직선을 "
    "100년 기준으로 환산한 값입니다."
)

# -----------------------------
# 연도별 예상 기온
# -----------------------------
selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1
)

predicted_temp = (
    overall["slope"] * selected_year + overall["intercept"]
)

st.subheader(f"{selected_year}년 예상 연평균기온")
st.metric("전체 기간 회귀 모델의 예상 기온", f"{predicted_temp:.2f} °C")

st.caption(
    "과거 전체 기간의 추세를 연장한 통계적 추정치이며, "
    "실제 기상 예보가 아닙니다."
)

# -----------------------------
# 분석 정보
# -----------------------------
st.subheader("분석 데이터 정보")

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 분석 연도 수", f"{overall['count']}개")
c2.metric("시작 연도", f"{overall['start']}년")
c3.metric("끝 연도", f"{overall['end']}년")
c4.metric("전체 기간 상관계수", f"{overall['correlation']:.4f}")

# -----------------------------
# 산점도 및 회귀 직선
# -----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=yearly["연도"],
    y=yearly["연평균기온"],
    mode="markers",
    name="실제 연평균기온",
    customdata=yearly[["관측일수"]].to_numpy(),
    hovertemplate=(
        "연도: %{x}년<br>"
        "연평균기온: %{y:.2f} °C<br>"
        "관측일수: %{customdata[0]}일<extra></extra>"
    )
))

# 전체 기간 회귀 직선
line_x = np.linspace(overall["start"], 2100, 300)
line_y = overall["slope"] * line_x + overall["intercept"]

fig.add_trace(go.Scatter(
    x=line_x,
    y=line_y,
    mode="lines",
    name="전체 기간 회귀 직선",
    line=dict(width=3),
    hovertemplate="연도: %{x:.0f}년<br>기온: %{y:.2f} °C<extra></extra>"
))

# 최근 20년 회귀 직선은 해당 구간에만 표시
recent_x = np.linspace(recent["start"], recent["end"], 100)
recent_y = recent["slope"] * recent_x + recent["intercept"]

fig.add_trace(go.Scatter(
    x=recent_x,
    y=recent_y,
    mode="lines",
    name="최근 20년 회귀 직선",
    line=dict(width=3, dash="dash"),
    hovertemplate="연도: %{x:.0f}년<br>기온: %{y:.2f} °C<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name="선택 연도 예상 기온",
    marker=dict(size=13, symbol="diamond"),
    hovertemplate="연도: %{x:.0f}년<br>예상 기온: %{y:.2f} °C<extra></extra>"
))

fig.update_layout(
    title="서울 연평균기온과 회귀 직선 비교",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    xaxis=dict(range=[1900, 2100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=85, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 회귀식
# -----------------------------
with st.expander("회귀식과 연도별 데이터 보기"):
    st.write("전체 기간 회귀식")
    st.latex(
        rf"\hat{{y}} = {overall['slope']:.5f}x "
        rf"{'+' if overall['intercept'] >= 0 else '-'} "
        rf"{abs(overall['intercept']):.2f}"
    )

    st.write("최근 20년 회귀식")
    st.latex(
        rf"\hat{{y}} = {recent['slope']:.5f}x "
        rf"{'+' if recent['intercept'] >= 0 else '-'} "
        rf"{abs(recent['intercept']):.2f}"
    )

    st.dataframe(
        yearly.rename(columns={
            "연평균기온": "연평균기온 (°C)",
            "관측일수": "관측일수 (일)"
        }),
        use_container_width=True,
        hide_index=True
    )
