"""Plotly frame animation for warehouse agents and synchronized simulation KPIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from warehouse_sim.visualization.scene_3d import plot_warehouse_3d


_TRACE_COLUMNS = {
    "time",
    "agent_id",
    "agent_type",
    "x",
    "y",
    "state",
    "wait_time_total",
    "queue_count",
    "completed_order_count",
}


def _snapshot_at(
    histories: dict[str, pd.DataFrame],
    kpi_times: np.ndarray,
    kpi_waits: np.ndarray,
    kpi_queues: np.ndarray,
    kpi_completions: np.ndarray,
    time: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for agent_id, history in histories.items():
        timestamps = history["time"].to_numpy(dtype=float)
        next_index = int(np.searchsorted(timestamps, time, side="right"))
        current_index = max(0, next_index - 1)
        current = history.iloc[current_index]
        x, y = float(current["x"]), float(current["y"])
        state = str(current["state"])
        if next_index < len(history) and state == "moving":
            following = history.iloc[next_index]
            next_time = float(following["time"])
            current_time = float(current["time"])
            if next_time > current_time:
                fraction = min(
                    max((time - current_time) / (next_time - current_time), 0),
                    1,
                )
                x += (float(following["x"]) - x) * fraction
                y += (float(following["y"]) - y) * fraction
        rows.append(
            {
                "agent_id": agent_id,
                "agent_type": current["agent_type"],
                "x": x,
                "y": y,
                "state": state,
            }
        )

    snapshot = pd.DataFrame(rows)
    kpi_index = max(0, int(np.searchsorted(kpi_times, time, side="right")) - 1)
    snapshot["wait_time_total"] = float(kpi_waits[kpi_index])
    snapshot["queue_count"] = int(kpi_queues[kpi_index])
    snapshot["completed_order_count"] = int(kpi_completions[kpi_index])
    snapshot["time"] = float(time)
    return snapshot


def _kpi_annotations(snapshot: pd.DataFrame, time: float) -> list[dict[str, object]]:
    active_workers = snapshot.loc[
        (snapshot["agent_type"] == "worker")
        & snapshot["state"].isin(["moving", "picking"]),
        "agent_id",
    ].nunique()
    queue_count = int(snapshot["queue_count"].max())
    total_wait = float(snapshot["wait_time_total"].max())
    completed_orders = int(snapshot["completed_order_count"].max())
    items = (
        ("時刻", f"{time:.1f}"),
        ("稼働中の作業員", str(active_workers)),
        ("累積待ち時間", f"{total_wait:.1f}"),
        (
            "待ち要求数（渋滞指標） / 完了オーダー",
            f"{queue_count} / {completed_orders}",
        ),
    )
    return [
        {
            "xref": "paper",
            "yref": "paper",
            "x": x,
            "y": 1.16,
            "text": f"<b>{label}</b><br>{value}",
            "showarrow": False,
            "align": "center",
            "font": {"size": 13, "color": "#243247"},
            "bgcolor": "#f4f7fb",
            "bordercolor": "#dce3eb",
            "borderwidth": 1,
            "borderpad": 7,
        }
        for (label, value), x in zip(items, (0.1, 0.36, 0.62, 0.88))
    ]


def plot_warehouse_animation(
    grid: np.ndarray,
    trace: pd.DataFrame,
    slotting: pd.DataFrame | None = None,
    *,
    fps: int = 20,
    max_frames: int = 120,
) -> go.Figure:
    """Create an interactive 2D Plotly animation with synchronized KPI labels.

    The complete simulation log is resampled to no more than ``max_frames``
    evenly spaced animation frames. Agent positions are linearly interpolated
    between logged movement samples. Play, pause, and timeline controls execute
    in the browser; KPI annotations are part of each Plotly frame and therefore
    advance in sync with the animated agents.
    """
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty two-dimensional array")
    if not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= 60:
        raise ValueError("fps must be an integer between 1 and 60")
    if (
        not isinstance(max_frames, int)
        or isinstance(max_frames, bool)
        or not 2 <= max_frames <= 500
    ):
        raise ValueError("max_frames must be an integer between 2 and 500")
    missing_columns = _TRACE_COLUMNS.difference(trace.columns)
    if missing_columns:
        raise ValueError(
            f"trace is missing required columns: {sorted(missing_columns)}"
        )
    if trace.empty:
        raise ValueError("trace must contain at least one agent sample")
    if trace[["time", "x", "y"]].isna().any().any():
        raise ValueError("trace times and coordinates must not contain null values")
    if not set(trace["agent_type"]).issubset({"worker", "forklift"}):
        raise ValueError("trace agent_type values must be worker or forklift")
    unknown_states = set(trace["state"]).difference(
        {"moving", "picking", "waiting"}
    )
    if unknown_states:
        raise ValueError(f"trace contains unsupported states: {sorted(unknown_states)}")

    minimum_time = float(trace["time"].min())
    maximum_time = float(trace["time"].max())
    observed_times = trace["time"].nunique()
    frame_count = min(max_frames, max(2, observed_times))
    frame_times = (
        np.linspace(minimum_time, maximum_time, frame_count)
        if maximum_time > minimum_time
        else np.array([minimum_time, minimum_time])
    )
    histories = {
        str(agent_id): history.sort_values("time").reset_index(drop=True)
        for agent_id, history in trace.groupby("agent_id", sort=False)
    }
    kpi_history = (
        trace.groupby("time", as_index=False)[
            ["wait_time_total", "queue_count", "completed_order_count"]
        ]
        .max()
        .sort_values("time")
    )
    kpi_times = kpi_history["time"].to_numpy(dtype=float)
    kpi_waits = kpi_history["wait_time_total"].to_numpy(dtype=float)
    kpi_queues = kpi_history["queue_count"].to_numpy(dtype=int)
    kpi_completions = kpi_history["completed_order_count"].to_numpy(dtype=int)

    cell_codes = {".": 0, "A": 1, "I": 2, "S": 3}
    unknown_cells = set(np.unique(grid)).difference(cell_codes)
    if unknown_cells:
        raise ValueError(f"grid contains unknown cell types: {sorted(unknown_cells)}")
    map_trace = go.Heatmap(
        z=np.vectorize(cell_codes.__getitem__)(grid),
        x=np.arange(grid.shape[1]),
        y=np.arange(grid.shape[0]),
        zmin=0,
        zmax=3,
        colorscale=[
            [0.0, "#f2f4f7"],
            [0.249, "#f2f4f7"],
            [0.25, "#8793a1"],
            [0.499, "#8793a1"],
            [0.5, "#e5484d"],
            [0.749, "#e5484d"],
            [0.75, "#3777c8"],
            [1.0, "#3777c8"],
        ],
        showscale=False,
        hovertemplate="行=%{y}, 列=%{x}<extra></extra>",
        name="倉庫グリッド",
    )
    figure = go.Figure(data=[map_trace])

    if slotting is not None and not slotting.empty:
        required_slotting = {"row", "column", "abc_class"}
        missing_slotting = required_slotting.difference(slotting.columns)
        if missing_slotting:
            raise ValueError(
                f"slotting is missing required columns: {sorted(missing_slotting)}"
            )
        for rank, color, symbol in (
            ("A", "#174ea6", "square"),
            ("B", "#4385d1", "square"),
            ("C", "#9ac4f4", "square"),
        ):
            rank_shelves = slotting[slotting["abc_class"].astype(str).str.upper() == rank]
            if rank_shelves.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=rank_shelves["column"],
                    y=rank_shelves["row"],
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": color,
                        "symbol": symbol,
                        "line": {"color": "white", "width": 0.5},
                    },
                    name=f"棚 - {rank}",
                    hovertemplate=f"棚ランク {rank}<br>行=%{{y}}, 列=%{{x}}<extra></extra>",
                )
            )

    agent_trace_indices: list[int] = []
    initial_snapshot = _snapshot_at(
        histories,
        kpi_times,
        kpi_waits,
        kpi_queues,
        kpi_completions,
        float(frame_times[0]),
    )
    for agent_type, name, color, symbol in (
        ("worker", "作業員", "#1677ff", "circle"),
        ("forklift", "フォークリフト", "#f28c28", "square"),
    ):
        agents = initial_snapshot[initial_snapshot["agent_type"] == agent_type]
        figure.add_trace(
            go.Scatter(
                x=agents["x"],
                y=agents["y"],
                mode="markers+text",
                marker={
                    "size": 13,
                    "color": color,
                    "symbol": symbol,
                    "line": {"color": "white", "width": 1.5},
                },
                text=agents["agent_id"],
                textposition="top center",
                customdata=agents[["state"]].to_numpy(),
                name=name,
                hovertemplate=(
                    "%{text}<br>状態=%{customdata[0]}<br>"
                    "列=%{x:.2f}, 行=%{y:.2f}<extra></extra>"
                ),
            )
        )
        agent_trace_indices.append(len(figure.data) - 1)

    frames: list[go.Frame] = []
    for frame_index, frame_time in enumerate(frame_times):
        snapshot = (
            initial_snapshot
            if frame_index == 0
            else _snapshot_at(
                histories,
                kpi_times,
                kpi_waits,
                kpi_queues,
                kpi_completions,
                float(frame_time),
            )
        )
        updates = []
        for agent_type in ("worker", "forklift"):
            agents = snapshot[snapshot["agent_type"] == agent_type]
            updates.append(
                go.Scatter(
                    x=agents["x"],
                    y=agents["y"],
                    text=agents["agent_id"],
                    customdata=agents[["state"]].to_numpy(),
                )
            )
        frames.append(
            go.Frame(
                name=f"frame-{frame_index}",
                data=updates,
                traces=agent_trace_indices,
                layout=go.Layout(
                    annotations=_kpi_annotations(snapshot, float(frame_time))
                ),
            )
        )

    figure.frames = frames
    figure.update_layout(
        title="倉庫内エージェント動線",
        template="plotly_white",
        height=680,
        margin={"l": 35, "r": 30, "t": 125, "b": 85},
        annotations=_kpi_annotations(initial_snapshot, float(frame_times[0])),
        xaxis={
            "title": "X（列）",
            "range": [-0.5, grid.shape[1] - 0.5],
            "dtick": max(1, grid.shape[1] // 10),
            "scaleanchor": "y",
            "scaleratio": 1,
            "showgrid": True,
            "gridcolor": "#d7dce2",
        },
        yaxis={
            "title": "Y（行）",
            "range": [grid.shape[0] - 0.5, -0.5],
            "dtick": max(1, grid.shape[0] // 10),
            "showgrid": True,
            "gridcolor": "#d7dce2",
        },
        legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.13,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ 再生",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "mode": "immediate",
                                "frame": {
                                    "duration": round(1000 / fps),
                                    "redraw": False,
                                },
                                "transition": {
                                    "duration": max(20, round(500 / fps)),
                                    "easing": "linear",
                                },
                            },
                        ],
                    },
                    {
                        "label": "⏸ 一時停止",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.25,
                "y": -0.13,
                "len": 0.73,
                "currentvalue": {"prefix": "時刻: "},
                "steps": [
                    {
                        "label": f"{frame_time:.1f}",
                        "method": "animate",
                        "args": [
                            [f"frame-{index}"],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for index, frame_time in enumerate(frame_times)
                ],
            }
        ],
    )
    return figure


def plot_warehouse_animation_3d(
    grid: np.ndarray,
    trace: pd.DataFrame,
    slotting: pd.DataFrame | None = None,
    *,
    fps: int = 20,
    max_frames: int = 120,
    speed_multiplier: float = 1.0,
) -> go.Figure:
    """Create a 3D warehouse animation with browser-side playback controls."""
    if not isinstance(speed_multiplier, (int, float)) or speed_multiplier not in {
        0.5,
        1.0,
        2.0,
        5.0,
    }:
        raise ValueError("speed_multiplier must be one of 0.5, 1.0, 2.0, or 5.0")
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty two-dimensional array")
    if not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= 60:
        raise ValueError("fps must be an integer between 1 and 60")
    if (
        not isinstance(max_frames, int)
        or isinstance(max_frames, bool)
        or not 2 <= max_frames <= 500
    ):
        raise ValueError("max_frames must be an integer between 2 and 500")
    missing_columns = _TRACE_COLUMNS.difference(trace.columns)
    if missing_columns:
        raise ValueError(
            f"trace is missing required columns: {sorted(missing_columns)}"
        )
    if trace.empty:
        raise ValueError("trace must contain at least one agent sample")
    if trace[["time", "x", "y"]].isna().any().any():
        raise ValueError("trace times and coordinates must not contain null values")
    if not set(trace["agent_type"]).issubset({"worker", "forklift"}):
        raise ValueError("trace agent_type values must be worker or forklift")
    unknown_states = set(trace["state"]).difference(
        {"moving", "picking", "waiting"}
    )
    if unknown_states:
        raise ValueError(f"trace contains unsupported states: {sorted(unknown_states)}")

    minimum_time = float(trace["time"].min())
    maximum_time = float(trace["time"].max())
    frame_count = min(max_frames, max(2, trace["time"].nunique()))
    frame_times = (
        np.linspace(minimum_time, maximum_time, frame_count)
        if maximum_time > minimum_time
        else np.array([minimum_time, minimum_time])
    )
    histories = {
        str(agent_id): history.sort_values("time").reset_index(drop=True)
        for agent_id, history in trace.groupby("agent_id", sort=False)
    }
    kpi_history = (
        trace.groupby("time", as_index=False)[
            ["wait_time_total", "queue_count", "completed_order_count"]
        ]
        .max()
        .sort_values("time")
    )
    kpi_times = kpi_history["time"].to_numpy(dtype=float)
    kpi_waits = kpi_history["wait_time_total"].to_numpy(dtype=float)
    kpi_queues = kpi_history["queue_count"].to_numpy(dtype=int)
    kpi_completions = kpi_history["completed_order_count"].to_numpy(dtype=int)

    figure = plot_warehouse_3d(grid, slotting)
    initial_snapshot = _snapshot_at(
        histories,
        kpi_times,
        kpi_waits,
        kpi_queues,
        kpi_completions,
        float(frame_times[0]),
    )
    agent_trace_indices: list[int] = []
    for agent_type, name, color, symbol in (
        ("worker", "作業員", "#168bff", "circle"),
        ("forklift", "フォークリフト", "#ff9c2a", "square"),
    ):
        agents = initial_snapshot[initial_snapshot["agent_type"] == agent_type]
        figure.add_trace(
            go.Scatter3d(
                x=agents["x"],
                y=grid.shape[0] - 1 - agents["y"],
                z=[0.5] * len(agents),
                mode="markers+text",
                marker={
                    "size": 13,
                    "color": color,
                    "symbol": symbol,
                    "line": {"color": "#111820", "width": 1},
                },
                text=agents["agent_id"],
                textposition="top center",
                customdata=agents[["state"]].to_numpy(),
                name=name,
                hovertemplate=(
                    "%{text}<br>状態=%{customdata[0]}<br>"
                    "X=%{x:.2f}, Y=%{y:.2f}<extra></extra>"
                ),
            )
        )
        agent_trace_indices.append(len(figure.data) - 1)

    frames: list[go.Frame] = []
    for frame_index, frame_time in enumerate(frame_times):
        snapshot = (
            initial_snapshot
            if frame_index == 0
            else _snapshot_at(
                histories,
                kpi_times,
                kpi_waits,
                kpi_queues,
                kpi_completions,
                float(frame_time),
            )
        )
        updates = []
        for agent_type in ("worker", "forklift"):
            agents = snapshot[snapshot["agent_type"] == agent_type]
            updates.append(
                go.Scatter3d(
                    x=agents["x"],
                    y=grid.shape[0] - 1 - agents["y"],
                    z=[0.5] * len(agents),
                    text=agents["agent_id"],
                    customdata=agents[["state"]].to_numpy(),
                )
            )
        frames.append(
            go.Frame(
                name=f"frame-{frame_index}",
                data=updates,
                traces=agent_trace_indices,
                layout=go.Layout(
                    annotations=_kpi_annotations(snapshot, float(frame_time))
                ),
            )
        )

    duration = round(1000 / (fps * float(speed_multiplier)))
    transition_duration = max(
        0,
        round(500 / (fps * float(speed_multiplier))),
    )
    figure.frames = frames
    figure.update_layout(
        title="倉庫内エージェント 3D アニメーション",
        height=760,
        margin={"l": 0, "r": 0, "t": 130, "b": 85},
        annotations=_kpi_annotations(initial_snapshot, float(frame_times[0])),
        scene_camera={
            "eye": {"x": 0.0, "y": 0.0, "z": 2.5},
            "up": {"x": 0.0, "y": 1.0, "z": 0.0},
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.11,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ 再生",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "mode": "immediate",
                                "frame": {"duration": duration, "redraw": True},
                                "transition": {
                                    "duration": transition_duration,
                                    "easing": "linear",
                                },
                            },
                        ],
                    },
                    {
                        "label": "⏸ 一時停止",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.25,
                "y": -0.11,
                "len": 0.73,
                "currentvalue": {"prefix": "時刻: "},
                "steps": [
                    {
                        "label": f"{frame_time:.1f}",
                        "method": "animate",
                        "args": [
                            [f"frame-{index}"],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for index, frame_time in enumerate(frame_times)
                ],
            }
        ],
    )
    return figure
