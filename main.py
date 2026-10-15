
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고 "
    "선형회귀를 이용해 원하는 연도의 예상 기온을 확인합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜와 기온 열 정리
    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )

    for col in ["평균기온", "최저기온", "최고기온"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
    df["연도"] = df["날짜"].dt.year

    # 기준 기간: 2025년까지
    df = df[df["연도"] <= 2025].copy()

    # 연도별 평균기온과 유효한 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    yearly = yearly[yearly["관측일수"] >= 300].copy()

    return yearly.sort_values("연도").reset_index(drop=True)


try:
    yearly = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석에 필요한 데이터가 부족합니다.")
    st.stop()

# -----------------------------
# 선형회귀 계산
# -----------------------------
years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# y = a*x + b
a, b = np.polyfit(years, temps, 1)

# 상관계수
correlation = np.corrcoef(years, temps)[0, 1]

start_year = int(years.min())
end_year = int(years.max())
year_count = len(yearly)

# -----------------------------
# 회귀 분석 결과
# -----------------------------
st.subheader("📊 회귀 분석 결과")

col1, col2, col3, col4 = st.columns(4)

col1.metric("회귀에 사용한 연도 수", f"{year_count}년")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수", f"{correlation:.4f}")

st.caption(
    f"회귀식: 연평균기온 = {a:.5f} × 연도 + {b:.3f}"
)

# -----------------------------
# 연도별 산점도와 회귀 직선
# -----------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

line_years = np.linspace(start_year, end_year, 300)
line_temps = a * line_years + b

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="선형회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "회귀 예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.10),
    margin=dict(l=20, r=20, t=40, b=20)
)

fig.update_xaxes(dtick=10)
fig.update_yaxes(showgrid=True, zeroline=False)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 연도별 기온 예측기
# -----------------------------
st.subheader("🔮 원하는 연도의 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

predicted_temp = a * selected_year + b

st.markdown(f"### {selected_year}년 예상 연평균기온")

st.markdown(
    f"""
    <div style="
        background-color: #eaf3ff;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 15px;
    ">
        <div style="font-size: 20px; color: #234;">
            선형회귀 예측값
        </div>
        <div style="
            font-size: 52px;
            font-weight: bold;
            color: #1769aa;
        ">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# 선택한 연도를 그래프에서 확인
fig_prediction = go.Figure()

fig_prediction.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 기온",
        marker=dict(size=6)
    )
)

fig_prediction.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3)
    )
)

fig_prediction.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name="선택한 연도의 예측값",
        marker=dict(size=15, symbol="star"),
        hovertemplate=(
            "연도: %{x}년<br>"
            "예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig_prediction.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig_prediction, use_container_width=True)

# -----------------------------
# 데이터 및 분석 설명
# -----------------------------
with st.expander("분석에 사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly.rename(
            columns={
                "연도": "연도",
                "연평균기온": "연평균기온(℃)",
                "관측일수": "관측일수(일)"
            }
        ).round(2),
        use_container_width=True,
        hide_index=True
    )

with st.expander("분석 방법과 주의사항"):
    st.markdown("""
    - 날짜별 평균기온을 연도별로 묶어 산술평균을 계산합니다.
    - 유효한 평균기온 관측값이 300개 미만인 연도는 제외합니다.
    - 2025년까지의 자료만 사용합니다.
    - 상관계수는 연도와 연평균기온 사이의 선형 관계를 나타냅니다.
    - 회귀 직선은 최소제곱법으로 구합니다.
    - 1900~2100년의 예상값은 동일한 직선으로 계산합니다.
    - 미래 예측값은 통계적 추정치이며 실제 기온과 다를 수 있습니다.
    """)
