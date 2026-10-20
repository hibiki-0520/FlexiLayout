"""Streamlit dashboard for warehouse layout and picking simulation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from warehouse_sim.layout.grid import WarehouseGrid
from warehouse_sim.simulation import (
    PickTask,
    PickingOrder,
    SimulationResult,
    simulate_picking,
)
from warehouse_sim.visualization import plot_warehouse_grid


@st.cache_data(show_spinner=False)
def run_scenario(
    shelf_count: int,
    worker_count: int,
    forklift_count: int,
) -> tuple[np.ndarray, pd.DataFrame, SimulationResult, int]:
    """Generate a reproducible layout and workload for the selected capacities."""
    warehouse = WarehouseGrid(width=50, height=50, aisle_spacing=8)
    rng = np.random.default_rng(42)
    product_ids = [f"SKU-{index + 1:03d}" for index in range(shelf_count)]
    frequencies = rng.zipf(1.7, size=shelf_count).astype(float)
    products = pd.DataFrame(
        {"product_id": product_ids, "pick_frequency": frequencies}
    )
    slotting = warehouse.optimize_slotting(products)

    shelf_positions = {
        row.product_id: (int(row.row), int(row.column))
        for row in slotting.itertuples(index=False)
    }
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

    io_point = warehouse.io_points[0]
    result = simulate_picking(
        orders,
        shelf_positions=shelf_positions,
        io_point=io_point,
        worker_count=worker_count,
        forklift_count=forklift_count,
        container_count=worker_count,
    )
    return warehouse.grid.copy(), slotting, result, order_count


def main() -> None:
    st.set_page_config(page_title="倉庫シミュレーション", layout="wide")
    st.title("倉庫レイアウト・ピッキングシミュレーション")
    st.caption(
        "棚数やリソース数を変更すると、ABC棚割りとSimPyシミュレーションを再実行します。"
    )

    with st.sidebar:
        st.header("シミュレーション条件")
        shelf_count = st.slider("棚の数（商品SKU数）", 1, 200, 50)
        worker_count = st.slider("作業員数", 1, 20, 4)
        forklift_count = st.slider("フォークリフト数", 1, 10, 2)
        st.caption(f"コンテナ数は作業員数と同数（{worker_count}台）で設定します。")

    try:
        with st.spinner("レイアウト生成・シミュレーション中..."):
            grid, slotting, result, order_count = run_scenario(
                shelf_count, worker_count, forklift_count
            )
    except (ValueError, RuntimeError) as exc:
        st.error(f"シミュレーションを実行できませんでした: {exc}")
        return

    st.subheader("シミュレーション結果")
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("メイクスパン", f"{result.makespan:.1f}")
    kpi_columns[1].metric("作業員稼働率", f"{result.worker_utilization:.1%}")
    kpi_columns[2].metric("フォークリフト稼働率", f"{result.forklift_utilization:.1%}")
    kpi_columns[3].metric("処理オーダー数", f"{order_count}")

    wait_columns = st.columns(3)
    wait_columns[0].metric(
        "作業員待ち時間（合計）", f"{result.worker_wait_time_total:.1f}"
    )
    wait_columns[1].metric(
        "フォークリフト待ち時間（合計）", f"{result.forklift_wait_time_total:.1f}"
    )
    wait_columns[2].metric(
        "コンテナ待ち時間（合計）", f"{result.container_wait_time_total:.1f}"
    )

    layout_column, table_column = st.columns([1.2, 1])
    with layout_column:
        st.subheader("グリッドレイアウト")
        figure = plot_warehouse_grid(grid, slotting)
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
    with table_column:
        st.subheader("棚割り（先頭20件）")
        st.dataframe(
            slotting[
                ["product_id", "pick_frequency", "abc_class", "row", "column", "distance_from_io"]
            ].head(20),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("オーダー完了時間（先頭20件）")
        completion_frame = pd.DataFrame(
            result.order_completion_times.items(),
            columns=["order_id", "completion_time"],
        )
        st.dataframe(completion_frame.head(20), use_container_width=True, hide_index=True)

    st.caption(
        "移動距離はグリッド上のマンハッタン距離で計算します。"
        "作業員速度・フォークリフト速度は各1・2セル/時間単位、"
        "ピッキング時間は1個あたり1時間単位です。"
    )


if __name__ == "__main__":
    main()
