
# -----------------------------
# 전체 기간 및 최근 20년 회귀 분석
# -----------------------------
years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# 전체 기간 회귀 직선
a, b = np.polyfit(years, temps, 1)

# 상관계수
correlation = np.corrcoef(years, temps)[0, 1]

start_year = int(years.min())
end_year = int(years.max())
year_count = len(yearly)

# 최근 20년: 가장 최근 관측 연도를 기준으로 20년 범위 설정
recent_start = end_year - 19
recent = yearly[yearly["연도"] >= recent_start].copy()

if len(recent) >= 2:
    recent_years = recent["연도"].to_numpy(dtype=float)
    recent_temps = recent["연평균기온"].to_numpy(dtype=float)

    recent_a, recent_b = np.polyfit(
        recent_years, recent_temps, 1
    )

    recent_correlation = np.corrcoef(
        recent_years, recent_temps
    )[0, 1]

    recent_count = len(recent)
    recent_actual_start = int(recent["연도"].min())
    recent_actual_end = int(recent["연도"].max())
else:
    recent_a = recent_b = np.nan
    recent_correlation = np.nan
    recent_count = len(recent)
    recent_actual_start = recent_actual_end = None

# 연간 기울기를 100년당 기온 변화로 환산
rise_100_years = a * 100
recent_rise_100_years = recent_a * 100

# -----------------------------
# 분석 결과 표시
# -----------------------------
st.subheader("🌡️ 100년당 기온 변화 비교")

st.caption(
    f"전체 기간: {start_year}~{end_year}년 · "
    f"회귀 사용 연도 {year_count}개"
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="전체 기간 · 100년당 기온 변화",
        value=f"{rise_100_years:+.2f}℃",
        help="전체 분석 기간의 선형회귀 기울기를 100배 한 값"
    )
    st.caption(
        f"상관계수: {correlation:.4f}"
    )

with col2:
    if len(recent) >= 2:
        st.metric(
            label="최근 20년 · 100년당 기온 변화",
            value=f"{recent_rise_100_years:+.2f}℃",
            help=(
                f"{recent_actual_start}~{recent_actual_end}년의 "
                "자료로 계산한 기울기를 100배 한 값"
            )
        )
        st.caption(
            f"실제 분석 기간: {recent_actual_start}~"
            f"{recent_actual_end}년 · "
            f"사용 연도 {recent_count}개 · "
            f"상관계수 {recent_correlation:.4f}"
        )
    else:
        st.warning("최근 기간의 데이터가 부족해 비교할 수 없습니다.")

st.info(
    "양수(+)는 기온 상승 추세, 음수(-)는 기온 하락 추세를 뜻합니다. "
    "100년당 변화량은 해당 기간의 직선 기울기를 환산한 값이며, "
    "실제 미래 기온이 그만큼 변한다는 보장은 없습니다."
)
