"""Streamlit dashboard for warehouse layout and picking simulation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from warehouse_sim.layout.grid import WarehouseGrid
from warehouse_sim.simulation import (
    CargoSize,
    OptimizationResult,
    PickTask,
    PickingOrder,
    SimulationResult,
    optimize_conditions,
    simulate_picking,
)
from warehouse_sim.visualization import (
    plot_warehouse_3d,
    plot_warehouse_animation_3d,
    plot_warehouse_grid,
)


@st.cache_data(show_spinner=False)
def prepare_scenario(shelf_count: int) -> tuple[pd.DataFrame, list[PickingOrder]]:
    """Generate reproducible product and order data for the selected SKU count."""
    rng = np.random.default_rng(42)
    product_ids = [f"SKU-{index + 1:03d}" for index in range(shelf_count)]
    frequencies = rng.zipf(1.7, size=shelf_count).astype(float)
    products = pd.DataFrame(
        {"product_id": product_ids, "pick_frequency": frequencies}
    )
    order_count = max(20, shelf_count * 2)
    pick_probabilities = frequencies / frequencies.sum()
    orders: list[PickingOrder] = []
    for order_number in range(order_count):
        pick_count = min(shelf_count, int(rng.integers(1, 5)))
        selected = rng.choice(
            shelf_count,
            size=pick_count,
            replace=False,
            p=pick_probabilities,
        )
        picks = [
            PickTask(product_ids[int(index)], quantity=int(rng.integers(1, 4)))
            for index in selected
        ]
        orders.append(PickingOrder(f"ORD-{order_number + 1:03d}", picks))
    return products, orders


@st.cache_data(show_spinner=False)
def run_scenario(
    shelf_count: int,
    worker_count: int,
    forklift_count: int,
    cargo_weight_kg: float = 10.0,
    cargo_size_class: str = CargoSize.MEDIUM.value,
    max_transport_units: int = 5,
    placement_rule: str = "abc",
    trace_interval: float = 0.5,
) -> tuple[np.ndarray, pd.DataFrame, SimulationResult, int]:
    """Generate a reproducible layout and workload for the selected capacities."""
    warehouse = WarehouseGrid(width=50, height=50, aisle_spacing=8)
    products, orders = prepare_scenario(shelf_count)
    slotting = warehouse.optimize_slotting(
        products,
        placement_rule=placement_rule,
    )
    shelf_positions = {
        row.product_id: (int(row.row), int(row.column))
        for row in slotting.itertuples(index=False)
    }
    result = simulate_picking(
        orders,
        shelf_positions=shelf_positions,
        io_point=warehouse.io_points[0],
        worker_count=worker_count,
        forklift_count=forklift_count,
        container_count=worker_count,
        cargo_weight_kg=cargo_weight_kg,
        cargo_size_class=cargo_size_class,
        max_transport_units=max_transport_units,
        trace_interval=trace_interval,
    )
    return warehouse.grid.copy(), slotting, result, len(orders)


@st.cache_data(show_spinner=False)
def run_optimization(
    products: pd.DataFrame,
    orders: list[PickingOrder],
    io_point: tuple[int, int],
    cargo_weight_kg: float,
    cargo_size_class: str,
    max_transport_units: int,
    target_makespan: float,
    max_workers: int,
    max_forklifts: int,
    worker_hourly_cost: float,
    forklift_hourly_cost: float,
    mode: str,
    budget: float | None,
    min_worker_utilization: float,
    min_forklift_utilization: float,
    co2_per_forklift_cell: float,
    co2_per_forklift_hour: float,
) -> OptimizationResult:
    """Cache a sensitivity search for identical workload and constraints."""
    return optimize_conditions(
        products,
        orders,
        io_point,
        width=50,
        height=50,
        aisle_spacing=8,
        cargo_weight_kg=cargo_weight_kg,
        cargo_size_class=cargo_size_class,
        max_transport_units=max_transport_units,
        target_makespan=target_makespan,
        max_workers=max_workers,
        max_forklifts=max_forklifts,
        worker_hourly_cost=worker_hourly_cost,
        forklift_hourly_cost=forklift_hourly_cost,
        mode=mode,
        budget=budget,
        min_worker_utilization=min_worker_utilization,
        min_forklift_utilization=min_forklift_utilization,
        co2_per_forklift_cell=co2_per_forklift_cell,
        co2_per_forklift_hour=co2_per_forklift_hour,
    )


def apply_recommendation() -> None:
    """Apply the selected configuration before Streamlit recreates widgets."""
    recommendation = st.session_state.get("recommended_configuration")
    if recommendation is None:
        return
    st.session_state["worker_count_input"] = int(recommendation["workers"])
    st.session_state["forklift_count_input"] = int(recommendation["forklifts"])
    st.session_state["placement_rule_input"] = str(
        recommendation["placement_rule"]
    )


def render_layout(
    grid: np.ndarray,
    slotting: pd.DataFrame,
    positions: pd.DataFrame,
    key_suffix: str = "timeline",
) -> None:
    layout_2d_tab, layout_3d_tab = st.tabs(["2D", "3D（インタラクティブ）"])
    with layout_2d_tab:
        figure = plot_warehouse_grid(grid, slotting, positions)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
    with layout_3d_tab:
        figure_3d = plot_warehouse_3d(
            grid,
            slotting,
            agent_positions=positions,
        )
        st.plotly_chart(
            figure_3d,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["resetCameraDefault3d"],
            },
            key=f"warehouse-layout-3d-{key_suffix}",
        )


def optimization_charts(
    candidates: pd.DataFrame,
) -> tuple[go.Figure, go.Figure]:
    """Build worker/makespan and estimated-cost/makespan sensitivity plots."""
    makespan_figure = go.Figure()
    for (rule, forklifts), group in candidates.groupby(
        ["placement_rule", "forklifts"], sort=True
    ):
        ordered = group.sort_values("workers")
        makespan_figure.add_trace(
            go.Scatter(
                x=ordered["workers"],
                y=ordered["makespan"],
                mode="lines+markers",
                name=f"{rule} / {forklifts}台",
            )
        )
    makespan_figure.update_layout(
        title="作業員数とメイクスパン",
        xaxis_title="作業員数",
        yaxis_title="メイクスパン",
        legend_title="棚配置ルール / フォークリフト数",
    )
    cost_figure = go.Figure(
        go.Scatter(
            x=candidates["estimated_cost"],
            y=candidates["makespan"],
            mode="markers",
            marker={
                "color": candidates["workers"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "作業員数"},
                "symbol": [
                    "circle" if rule == "abc" else "diamond"
                    for rule in candidates["placement_rule"]
                ],
                "size": 9,
            },
            customdata=candidates[
                ["workers", "forklifts", "placement_rule", "feasible"]
            ].to_numpy(),
            hovertemplate=(
                "費用=%{x:.1f}<br>メイクスパン=%{y:.1f}<br>"
                "作業員=%{customdata[0]}<br>フォークリフト=%{customdata[1]}<br>"
                "棚配置=%{customdata[2]}<br>制約内=%{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )
    cost_figure.update_layout(
        title="リソース費用と処理効率のトレードオフ",
        xaxis_title="推定リソース費用（作業員費用＋フォークリフト稼働費用）",
        yaxis_title="メイクスパン",
    )
    return makespan_figure, cost_figure


def main() -> None:
    st.set_page_config(page_title="倉庫シミュレーション", layout="wide")
    st.title("倉庫レイアウト・ピッキングシミュレーション")
    st.caption(
        "レイアウト・シミュレーション・最適化の結果をタブで確認できます。"
    )

    with st.sidebar:
        st.header("基本条件")
        shelf_count = st.number_input(
            "棚の数（商品SKU数）",
            min_value=1,
            max_value=1000,
            value=134,
            step=1,
            key="shelf_count_input",
        )
        worker_count = st.number_input(
            "作業員数",
            min_value=1,
            max_value=100,
            value=13,
            step=1,
            key="worker_count_input",
        )
        forklift_count = st.number_input(
            "フォークリフト数",
            min_value=0,
            max_value=50,
            value=4,
            step=1,
            key="forklift_count_input",
        )
        st.caption(f"コンテナ数は作業員数と同数（{worker_count}台）で設定します。")
        with st.expander("詳細な詳細設定・運搬物パラメータ", expanded=False):
            placement_rule = st.selectbox(
                "棚配置ルール",
                options=["abc", "frequency"],
                format_func=lambda rule: (
                    "ABC優先" if rule == "abc" else "ピッキング頻度優先"
                ),
                key="placement_rule_input",
            )
            st.subheader("運搬物条件")
            cargo_weight_kg = st.number_input(
                "平均荷物重量 (kg)",
                min_value=0.5,
                max_value=100.0,
                value=10.0,
                step=0.5,
                key="cargo_weight_input",
            )
            cargo_size_class = st.selectbox(
                "荷物サイズ区分",
                options=[size.value for size in CargoSize],
                index=1,
                key="cargo_size_input",
            )
            max_transport_units = st.number_input(
                "1回あたりの最大同時搬送数 (個)",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="transport_capacity_input",
            )
            trace_interval = st.selectbox(
                "シミュレーション記録間隔（時間単位）",
                options=[0.25, 0.5, 1.0],
                index=1,
                format_func=lambda interval: f"{interval:g}",
                key="trace_interval_input",
                help="間隔を短くすると位置ログが細かくなり、表示が滑らかになります。",
            )
            st.divider()
            st.subheader("最適化の制約・コスト")
            target_makespan = st.number_input(
                "目標メイクスパン（時間単位）",
                min_value=1.0,
                value=500.0,
                step=10.0,
                key="optimization_target_input",
            )
            max_search_workers = st.number_input(
                "作業員の探索上限",
                min_value=1,
                max_value=20,
                value=20,
                step=1,
                key="optimization_workers_input",
            )
            max_search_forklifts = st.number_input(
                "フォークリフトの探索上限",
                min_value=0,
                max_value=8,
                value=4,
                step=1,
                key="optimization_forklifts_input",
            )
            worker_hourly_cost = st.number_input(
                "作業員の時間単価",
                min_value=0.0,
                value=30.0,
                step=1.0,
                key="worker_cost_input",
            )
            forklift_hourly_cost = st.number_input(
                "フォークリフトの時間単価",
                min_value=0.0,
                value=50.0,
                step=1.0,
                key="forklift_cost_input",
            )
            optimization_mode = st.selectbox(
                "最適化の目標",
                options=["time", "cost", "balance", "eco"],
                format_func=lambda mode: {
                    "time": "時間優先（メイクスパン最小化）",
                    "cost": "コスト最小化（予算内）",
                    "balance": "バランス優先（費用対効果）",
                    "eco": "環境・エコ優先（CO2換算推定値）",
                }[mode],
                key="optimization_mode_input",
            )
            optimization_budget = st.number_input(
                "予算上限（コスト最小化時）",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                key="optimization_budget_input",
                help="コスト最小化モードでのみ適用されます。",
            )
            min_worker_utilization = st.number_input(
                "作業員の最低稼働率（バランス優先時）",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                key="minimum_worker_utilization_input",
            )
            min_forklift_utilization = st.number_input(
                "フォークリフトの最低稼働率（バランス優先時）",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="minimum_forklift_utilization_input",
            )
            co2_per_forklift_cell = st.number_input(
                "フォークリフト1グリッド移動あたりのCO2換算係数 (kg-CO2)",
                min_value=0.0,
                value=0.2,
                step=0.05,
                key="co2_per_cell_input",
                help="モデル比較用の仮定値です。実測・認証済み排出係数ではありません。",
            )
            co2_per_forklift_hour = st.number_input(
                "フォークリフト1稼働時間あたりのCO2換算係数 (kg-CO2)",
                min_value=0.0,
                value=0.5,
                step=0.1,
                key="co2_per_hour_input",
                help="モデル比較用の仮定値です。実測・認証済み排出係数ではありません。",
            )

    cargo_weight_kg = float(st.session_state["cargo_weight_input"])
    cargo_size_class = str(st.session_state["cargo_size_input"])
    max_transport_units = int(st.session_state["transport_capacity_input"])
    trace_interval = float(st.session_state["trace_interval_input"])
    placement_rule = str(st.session_state["placement_rule_input"])
    target_makespan = float(st.session_state["optimization_target_input"])
    max_search_workers = int(st.session_state["optimization_workers_input"])
    max_search_forklifts = int(st.session_state["optimization_forklifts_input"])
    worker_hourly_cost = float(st.session_state["worker_cost_input"])
    forklift_hourly_cost = float(st.session_state["forklift_cost_input"])
    optimization_mode = str(st.session_state["optimization_mode_input"])
    optimization_budget = float(st.session_state["optimization_budget_input"])
    min_worker_utilization = float(
        st.session_state["minimum_worker_utilization_input"]
    )
    min_forklift_utilization = float(
        st.session_state["minimum_forklift_utilization_input"]
    )
    co2_per_forklift_cell = float(st.session_state["co2_per_cell_input"])
    co2_per_forklift_hour = float(st.session_state["co2_per_hour_input"])
    minimum_forklifts = 0 if cargo_size_class == CargoSize.SMALL.value else 1
    if forklift_count == 0 and cargo_size_class != CargoSize.SMALL.value:
        st.sidebar.warning(
            f"{cargo_size_class}はフォークリフトが必要です。1台以上を指定してください。"
        )

    try:
        with st.spinner("レイアウト生成・シミュレーション中..."):
            grid, slotting, result, order_count = run_scenario(
                int(shelf_count),
                int(worker_count),
                int(forklift_count),
                float(cargo_weight_kg),
                cargo_size_class,
                int(max_transport_units),
                placement_rule,
                trace_interval,
            )
    except (ValueError, RuntimeError) as exc:
        st.error(f"シミュレーションを実行できませんでした: {exc}")
        return

    average_utilization = (
        (result.worker_utilization + result.forklift_utilization) / 2
        if forklift_count > 0
        else result.worker_utilization
    )
    total_wait = (
        result.worker_wait_time_total
        + result.forklift_wait_time_total
        + result.container_wait_time_total
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric("メイクスパン", f"{result.makespan:.1f}")
    summary_columns[1].metric("平均稼働率", f"{average_utilization:.1%}")
    summary_columns[2].metric("総待ち時間", f"{total_wait:.1f}")
    summary_columns[3].metric("完了オーダー数", f"{order_count}")

    tabs = st.tabs(
        [
            "倉庫レイアウト・3Dモデル",
            "リアルタイム・アニメーション",
            "最適化・KPI分析",
        ]
    )
    with tabs[0]:
        st.subheader("倉庫レイアウト")
        st.caption("2Dグリッドとインタラクティブな3Dモデルを切り替えて確認できます。")
        render_layout(grid, slotting, pd.DataFrame(), key_suffix="overview")
        with st.expander("棚割りとオーダー完了時間", expanded=False):
            table_columns = st.columns(2)
            with table_columns[0]:
                st.markdown("**棚割り（先頭20件）**")
                st.dataframe(
                    slotting[
                        [
                            "product_id",
                            "pick_frequency",
                            "abc_class",
                            "row",
                            "column",
                            "distance_from_io",
                        ]
                    ].head(20),
                    use_container_width=True,
                    hide_index=True,
                )
            with table_columns[1]:
                st.markdown("**オーダー完了時間（先頭20件）**")
                completion_frame = pd.DataFrame(
                    result.order_completion_times.items(),
                    columns=["order_id", "completion_time"],
                )
                st.dataframe(
                    completion_frame.head(20),
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[1]:
        st.subheader("シミュレーション再生")
        if result.trace.empty:
            st.info("再生できるエージェント記録はありません。")
        else:
            control_column, map_column = st.columns([1, 3], gap="large")
            with control_column:
                fps = st.select_slider(
                    "再生フレームレート",
                    options=[12, 20, 30],
                    value=20,
                    format_func=lambda value: f"{value} FPS",
                    key="animation_fps",
                )
                playback_speed = st.select_slider(
                    "再生速度",
                    options=[
                        "0.5x",
                        "1.0x (標準)",
                        "2.0x (高速)",
                        "5.0x (超高速)",
                    ],
                    value="1.0x (標準)",
                    key="animation_speed",
                )
                max_frames = st.slider(
                    "最大表示フレーム数（負荷調整）",
                    min_value=30,
                    max_value=240,
                    value=120,
                    step=30,
                    key="animation_max_frames",
                )
                st.caption(
                    f"モデル時間: 0～{result.makespan:.1f} / "
                    f"記録間隔: {trace_interval:g}"
                )
                st.caption(
                    "3Dシーン上部のKPI表示は、再生・一時停止・時間スライダーと同期します。"
                )
            with map_column:
                animation = plot_warehouse_animation_3d(
                    grid,
                    result.trace,
                    slotting,
                    fps=fps,
                    max_frames=max_frames,
                    speed_multiplier={
                        "0.5x": 0.5,
                        "1.0x (標準)": 1.0,
                        "2.0x (高速)": 2.0,
                        "5.0x (超高速)": 5.0,
                    }[playback_speed],
                )
                st.plotly_chart(
                    animation,
                    use_container_width=True,
                    config={"scrollZoom": True, "displaylogo": False},
                    key="warehouse-agent-animation",
                )

    optimization_signature = (
        int(shelf_count),
        float(cargo_weight_kg),
        cargo_size_class,
        int(max_transport_units),
        float(target_makespan),
        int(max_search_workers),
        int(max_search_forklifts),
        float(worker_hourly_cost),
        float(forklift_hourly_cost),
        optimization_mode,
        float(optimization_budget),
        float(min_worker_utilization),
        float(min_forklift_utilization),
        float(co2_per_forklift_cell),
        float(co2_per_forklift_hour),
    )
    with tabs[2]:
        st.subheader("最適化・KPI分析")
        st.caption(
            "目的に応じて作業員数・フォークリフト数・棚配置を探索します。"
        )
        settings_column, current_kpi_column = st.columns([1, 2], gap="large")
        with settings_column:
            st.markdown("**探索条件**")
            st.write(f"目標メイクスパン: {target_makespan:.1f}")
            st.write(
                f"作業員: 1～{max_search_workers} 人 / "
                f"フォークリフト: {minimum_forklifts}～{max_search_forklifts} 台"
            )
            st.write(
                f"時間単価: 作業員 {worker_hourly_cost:.1f} / "
                f"フォークリフト {forklift_hourly_cost:.1f}"
            )
            st.write(
                "最適化目標: "
                + {
                    "time": "時間優先（メイクスパン最小化）",
                    "cost": "コスト最小化（予算内）",
                    "balance": "バランス優先（費用対効果）",
                    "eco": "環境・エコ優先（CO2換算推定値）",
                }[optimization_mode]
            )
            if optimization_mode == "cost":
                st.write(f"予算上限: {optimization_budget:,.1f}")
            if optimization_mode == "balance":
                st.write(
                    f"最低稼働率: 作業員 {min_worker_utilization:.0%} / "
                    f"フォークリフト {min_forklift_utilization:.0%}"
                )
            optimize_clicked = st.button(
                "最適な条件を自動計算する",
                type="primary",
                disabled=(max_search_forklifts < minimum_forklifts),
                use_container_width=True,
                key="run_optimization_button",
            )
            if max_search_forklifts < minimum_forklifts:
                st.warning("現在の荷物サイズではフォークリフト1台以上が必要です。")
        with current_kpi_column:
            st.markdown("**現在のシミュレーションKPI**")
            current_kpis = st.columns(3)
            current_kpis[0].metric("メイクスパン", f"{result.makespan:.1f}")
            current_kpis[1].metric("平均稼働率", f"{average_utilization:.1%}")
            current_kpis[2].metric("総待ち時間", f"{total_wait:.1f}")

        if optimize_clicked:
            try:
                products, orders = prepare_scenario(int(shelf_count))
                with st.spinner("リソース数と棚配置ルールを比較しています..."):
                    search_result = run_optimization(
                        products,
                        orders,
                        (49, 25),
                        float(cargo_weight_kg),
                        cargo_size_class,
                        int(max_transport_units),
                        float(target_makespan),
                        int(max_search_workers),
                        int(max_search_forklifts),
                        float(worker_hourly_cost),
                        float(forklift_hourly_cost),
                        optimization_mode,
                        float(optimization_budget)
                        if optimization_mode == "cost"
                        else None,
                        float(min_worker_utilization),
                        float(min_forklift_utilization),
                        float(co2_per_forklift_cell),
                        float(co2_per_forklift_hour),
                    )
                st.session_state["optimization_candidates"] = (
                    search_result.candidates
                )
                st.session_state["optimization_feasible"] = search_result.feasible
                st.session_state["optimization_reason"] = (
                    search_result.recommendation_reason
                )
                st.session_state["optimization_signature"] = optimization_signature
                st.session_state["recommended_configuration"] = (
                    search_result.candidates.iloc[0][
                        ["workers", "forklifts", "placement_rule"]
                    ].to_dict()
                )
            except (ValueError, RuntimeError) as exc:
                st.error(f"最適条件を算出できませんでした: {exc}")

        candidates = st.session_state.get("optimization_candidates")
        if (
            st.session_state.get("optimization_signature")
            != optimization_signature
        ):
            candidates = None
        if isinstance(candidates, pd.DataFrame) and not candidates.empty:
            st.divider()
            st.markdown("**推奨条件**")
            recommendation = candidates.iloc[0]
            st.info(
                "推奨理由: "
                + str(st.session_state.get("optimization_reason", ""))
            )
            if st.session_state.get("optimization_feasible", False):
                st.success(
                    f"作業員 {int(recommendation['workers'])} 人、"
                    f"フォークリフト {int(recommendation['forklifts'])} 台、"
                    f"棚配置 {recommendation['placement_rule']}。"
                    f"メイクスパン {recommendation['makespan']:.1f}、"
                    f"推定費用 {recommendation['estimated_cost']:.1f}、"
                    f"CO2換算推定 {recommendation['estimated_co2_kg']:.1f} kg。"
                )
            else:
                st.warning(
                    "探索範囲内に設定目標・最低稼働率を満たす候補がありません。"
                    "目的に最も近い構成を提示します。"
                )
            current_estimated_cost = (
                result.makespan * int(worker_count) * worker_hourly_cost
                + result.forklift_active_time_total * forklift_hourly_cost
            )
            current_estimated_co2 = (
                result.forklift_distance_total * co2_per_forklift_cell
                + result.forklift_active_time_total * co2_per_forklift_hour
            )
            st.markdown("**現在条件と推奨条件のKPI比較**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "KPI": "作業員数",
                            "現在": str(int(worker_count)),
                            "推奨": str(int(recommendation["workers"])),
                        },
                        {
                            "KPI": "フォークリフト数",
                            "現在": str(int(forklift_count)),
                            "推奨": str(int(recommendation["forklifts"])),
                        },
                        {
                            "KPI": "棚配置ルール",
                            "現在": str(placement_rule),
                            "推奨": str(recommendation["placement_rule"]),
                        },
                        {
                            "KPI": "メイクスパン",
                            "現在": f"{result.makespan:.1f}",
                            "推奨": f"{recommendation['makespan']:.1f}",
                        },
                        {
                            "KPI": "推定費用",
                            "現在": f"{current_estimated_cost:.1f}",
                            "推奨": f"{recommendation['estimated_cost']:.1f}",
                        },
                        {
                            "KPI": "1アイテムあたり推定費用",
                            "現在": (
                                f"{current_estimated_cost / max(result.item_count_total, 1):.2f}"
                            ),
                            "推奨": f"{recommendation['cost_per_item']:.2f}",
                        },
                        {
                            "KPI": "作業員稼働率",
                            "現在": f"{result.worker_utilization:.1%}",
                            "推奨": f"{recommendation['worker_utilization']:.1%}",
                        },
                        {
                            "KPI": "フォークリフト稼働率",
                            "現在": f"{result.forklift_utilization:.1%}",
                            "推奨": f"{recommendation['forklift_utilization']:.1%}",
                        },
                        {
                            "KPI": "フォークリフト走行距離",
                            "現在": f"{result.forklift_distance_total:.1f}",
                            "推奨": f"{recommendation['forklift_distance']:.1f}",
                        },
                        {
                            "KPI": "CO2換算推定値 (kg)",
                            "現在": f"{current_estimated_co2:.1f}",
                            "推奨": f"{recommendation['estimated_co2_kg']:.1f}",
                        },
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.button(
                "推奨条件を入力欄に適用",
                on_click=apply_recommendation,
                key="apply_recommendation_button",
            )
            makespan_figure, cost_figure = optimization_charts(candidates)
            chart_columns = st.columns(2, gap="large")
            with chart_columns[0]:
                st.plotly_chart(
                    makespan_figure,
                    use_container_width=True,
                    key="optimization_makespan_chart",
                )
            with chart_columns[1]:
                st.plotly_chart(
                    cost_figure,
                    use_container_width=True,
                    key="optimization_cost_chart",
                )
            with st.expander("候補構成の一覧", expanded=False):
                st.dataframe(candidates, use_container_width=True, hide_index=True)
        else:
            st.info("探索条件を設定し、「最適な条件を自動計算する」を実行してください。")

    st.caption(
        "移動距離はグリッド上のマンハッタン距離で計算します。"
        "時間単位はモデル上の任意単位です。"
    )


if __name__ == "__main__":
    main()
