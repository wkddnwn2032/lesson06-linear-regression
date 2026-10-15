
# 전체 기간 회귀 직선
line_years = np.linspace(start_year, end_year, 300)
line_temps = a * line_years + b

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3)
    )
)

# 최근 20년 회귀 직선
if len(recent) >= 2:
    recent_line_years = np.linspace(
        recent_actual_start,
        recent_actual_end,
        100
    )
    recent_line_temps = (
        recent_a * recent_line_years + recent_b
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_years,
            y=recent_line_temps,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, dash="dash")
        )
    )
