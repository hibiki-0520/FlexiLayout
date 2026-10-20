"""Interactive Plotly 3D visualization for warehouse layouts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

GridCoordinate = tuple[int, int]
_BOX_TRIANGLES = (
    (0, 1, 2), (0, 2, 3),
    (4, 6, 5), (4, 7, 6),
    (0, 5, 1), (0, 4, 5),
    (1, 6, 2), (1, 5, 6),
    (2, 7, 3), (2, 6, 7),
    (3, 4, 0), (3, 7, 4),
)


def _append_box(
    geometry: dict[str, list[float] | list[int]],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> None:
    """Append a cuboid to a combined Mesh3d geometry."""
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds
    vertices = (
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
    )
    base = len(geometry["x"])
    for x, y, z in vertices:
        geometry["x"].append(x)
        geometry["y"].append(y)
        geometry["z"].append(z)
    for first, second, third in _BOX_TRIANGLES:
        geometry["i"].append(base + first)
        geometry["j"].append(base + second)
        geometry["k"].append(base + third)


def _add_mesh(
    figure: go.Figure,
    geometry: dict[str, list[float] | list[int]],
    *,
    color: str,
    name: str,
    hovertemplate: str,
) -> None:
    if not geometry["i"]:
        return
    figure.add_trace(
        go.Mesh3d(
            **geometry,
            color=color,
            opacity=1,
            flatshading=True,
            lighting={
                "ambient": 0.45,
                "diffuse": 0.75,
                "specular": 0.5,
                "roughness": 0.55,
                "fresnel": 0.15,
            },
            lightposition={"x": -100, "y": 150, "z": 200},
            name=name,
            hovertemplate=hovertemplate,
        )
    )


def plot_warehouse_3d(
    grid: np.ndarray,
    slotting: pd.DataFrame | None = None,
    worker_paths: Sequence[Sequence[GridCoordinate]] | None = None,
    forklift_paths: Sequence[Sequence[GridCoordinate]] | None = None,
    shelf_height: float = 4.0,
    agent_positions: pd.DataFrame | None = None,
) -> go.Figure:
    """Build an interactive 3D warehouse scene from a 2D grid.

    Grid columns map to X, rows map to Y, and shelf height maps to Z. Each
    occupied shelf cell is drawn as a steel rack bay with multiple shelf decks.
    Optional worker and forklift paths are sequences of ``(row, column)``
    coordinates and are drawn just above the floor.
    """
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty two-dimensional array")
    if not np.isfinite(shelf_height) or shelf_height <= 0:
        raise ValueError("shelf_height must be a finite positive number")

    cell_values = {".", "A", "I", "S"}
    unknown_cells = set(np.unique(grid)).difference(cell_values)
    if unknown_cells:
        raise ValueError(f"grid contains unknown cell types: {sorted(unknown_cells)}")

    abc_by_position: dict[GridCoordinate, str] = {}
    if slotting is not None:
        required_columns = {"row", "column", "abc_class"}
        missing_columns = required_columns.difference(slotting.columns)
        if missing_columns:
            raise ValueError(
                f"slotting is missing required columns: {sorted(missing_columns)}"
            )
        for item in slotting.itertuples(index=False):
            row, column = int(item.row), int(item.column)
            if not (0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]):
                raise ValueError(
                    f"slotting coordinate {(row, column)} is outside the grid"
                )
            if grid[row, column] != "S":
                raise ValueError(f"slotting coordinate {(row, column)} is not a shelf")
            abc_class = str(item.abc_class).upper()
            if abc_class not in {"A", "B", "C"}:
                raise ValueError(
                    f"ABC class at {(row, column)} must be A, B, or C"
                )
            abc_by_position[(row, column)] = abc_class

    def validate_paths(
        paths: Sequence[Sequence[GridCoordinate]] | None, name: str
    ) -> list[list[GridCoordinate]]:
        validated: list[list[GridCoordinate]] = []
        for path in paths or []:
            points = list(path)
            for point in points:
                if (
                    len(point) != 2
                    or any(not isinstance(value, (int, np.integer)) for value in point)
                    or not (
                        0 <= point[0] < grid.shape[0]
                        and 0 <= point[1] < grid.shape[1]
                    )
                ):
                    raise ValueError(f"{name} contains invalid grid coordinate {point}")
            if points:
                validated.append(points)
        return validated

    workers = validate_paths(worker_paths, "worker_paths")
    forklifts = validate_paths(forklift_paths, "forklift_paths")
    width, height = grid.shape[1], grid.shape[0]
    figure = go.Figure()

    figure.add_trace(
        go.Surface(
            x=[-0.5, width - 0.5],
            y=[-0.5, height - 0.5],
            z=[[0, 0], [0, 0]],
            colorscale=[[0, "#202833"], [1, "#202833"]],
            showscale=False,
            opacity=1,
            name="Concrete floor",
            hoverinfo="skip",
            lighting={"ambient": 1, "diffuse": 0.2, "roughness": 0.95},
        )
    )

    aisle_geometry: dict[str, list[float] | list[int]] = {
        "x": [], "y": [], "z": [], "i": [], "j": [], "k": []
    }
    for row, column in zip(*np.where(grid == "A")):
        _append_box(
            aisle_geometry,
            (column - 0.5, column + 0.5),
            (height - 1 - row - 0.5, height - 1 - row + 0.5),
            (0.015, 0.045),
        )
    _add_mesh(
        figure,
        aisle_geometry,
        color="#485461",
        name="Aisle surface",
        hovertemplate="Aisle<br>X=%{x}, Y=%{y}<extra></extra>",
    )

    minor_x: list[float | None] = []
    minor_y: list[float | None] = []
    minor_z: list[float | None] = []
    major_x: list[float | None] = []
    major_y: list[float | None] = []
    major_z: list[float | None] = []
    for column in range(width + 1):
        target_x, target_y, target_z = (
            (major_x, major_y, major_z)
            if column % 5 == 0
            else (minor_x, minor_y, minor_z)
        )
        target_x.extend((column - 0.5, column - 0.5, None))
        target_y.extend((-0.5, height - 0.5, None))
        target_z.extend((0.055, 0.055, None))
    for row in range(height + 1):
        target_x, target_y, target_z = (
            (major_x, major_y, major_z)
            if row % 5 == 0
            else (minor_x, minor_y, minor_z)
        )
        target_x.extend((-0.5, width - 0.5, None))
        target_y.extend((row - 0.5, row - 0.5, None))
        target_z.extend((0.055, 0.055, None))
    figure.add_trace(
        go.Scatter3d(
            x=minor_x,
            y=minor_y,
            z=minor_z,
            mode="lines",
            line={"color": "#66717e", "width": 1},
            opacity=0.38,
            name="Floor grid",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=major_x,
            y=major_y,
            z=major_z,
            mode="lines",
            line={"color": "#a3adb8", "width": 2},
            opacity=0.5,
            name="5-cell markings",
            hoverinfo="skip",
        )
    )

    abc_colors = {
        "A": "#e04b42",
        "B": "#e5a93d",
        "C": "#42a66b",
        "Unclassified": "#3982c4",
    }
    post_geometry: dict[str, list[float] | list[int]] = {
        "x": [], "y": [], "z": [], "i": [], "j": [], "k": []
    }
    deck_geometries: dict[str, dict[str, list[float] | list[int]]] = {
        category: {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []}
        for category in abc_colors
    }
    rack_height = shelf_height
    post_width = 0.065
    bay_width = 0.78
    bay_depth = 0.72
    shelf_levels = [rack_height * fraction for fraction in (0.12, 0.48, 0.84)]

    for row, column in zip(*np.where(grid == "S")):
        row, column = int(row), int(column)
        center_y = height - 1 - row
        category = abc_by_position.get((row, column), "Unclassified")
        x_left, x_right = column - bay_width / 2, column + bay_width / 2
        y_front, y_back = center_y - bay_depth / 2, center_y + bay_depth / 2
        for x_post in (x_left, x_right):
            for y_post in (y_front, y_back):
                _append_box(
                    post_geometry,
                    (x_post - post_width / 2, x_post + post_width / 2),
                    (y_post - post_width / 2, y_post + post_width / 2),
                    (0.05, rack_height),
                )

        deck = deck_geometries[category]
        deck_thickness = max(rack_height * 0.018, 0.035)
        for level in shelf_levels:
            _append_box(
                deck,
                (x_left, x_right),
                (y_front, y_back),
                (level, level + deck_thickness),
            )
            beam_height = max(rack_height * 0.025, 0.045)
            for y_beam in (y_front, y_back):
                _append_box(
                    post_geometry,
                    (x_left, x_right),
                    (y_beam - 0.04, y_beam + 0.04),
                    (level - beam_height, level),
                )

    _add_mesh(
        figure,
        post_geometry,
        color="#9ba8b5",
        name="Steel rack frame",
        hovertemplate="Steel rack frame<extra></extra>",
    )
    for category, geometry in deck_geometries.items():
        _add_mesh(
            figure,
            geometry,
            color=abc_colors[category],
            name=f"Shelf deck - {category}",
            hovertemplate=f"Storage deck - {category}<br>level height=%{{z}}<extra></extra>",
        )

    dock_geometry: dict[str, list[float] | list[int]] = {
        "x": [], "y": [], "z": [], "i": [], "j": [], "k": []
    }
    io_rows, io_columns = np.where(grid == "I")
    for row, column in zip(io_rows, io_columns):
        center_y = height - 1 - int(row)
        _append_box(
            dock_geometry,
            (int(column) - 0.48, int(column) + 0.48),
            (center_y - 0.48, center_y + 0.48),
            (0.06, 0.18),
        )
    _add_mesh(
        figure,
        dock_geometry,
        color="#e04b42",
        name="Loading dock",
        hovertemplate="Loading dock<extra></extra>",
    )
    if len(io_rows):
        figure.add_trace(
            go.Scatter3d(
                x=io_columns,
                y=height - 1 - io_rows,
                z=np.full(len(io_rows), rack_height * 0.38),
                mode="markers+text",
                marker={
                    "size": 10,
                    "color": "#ff3030",
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 2},
                },
                text=["I/O"] * len(io_rows),
                textposition="top center",
                name="Entrance / I/O",
                hovertemplate="Entrance / I/O<br>X=%{x}, Y=%{y}<extra></extra>",
            )
        )

    if agent_positions is not None and not agent_positions.empty:
        agent_styles = {
            "worker": {"color": "#20e0a0", "symbol": "circle", "name": "Workers"},
            "forklift": {"color": "#ffd166", "symbol": "square", "name": "Forklifts"},
        }
        for agent_type, style in agent_styles.items():
            positions = agent_positions[
                agent_positions["agent_type"] == agent_type
            ]
            if positions.empty:
                continue
            figure.add_trace(
                go.Scatter3d(
                    x=positions["x"],
                    y=height - 1 - positions["y"],
                    z=[0.5] * len(positions),
                    mode="markers+text",
                    marker={
                        "size": 8,
                        "color": style["color"],
                        "symbol": style["symbol"],
                        "line": {"color": "#111820", "width": 1},
                    },
                    text=positions["agent_id"],
                    textposition="top center",
                    name=style["name"],
                    customdata=positions[["state"]].to_numpy(),
                    hovertemplate=(
                        "%{text}<br>state=%{customdata[0]}<br>"
                        "X=%{x}, Y=%{y}<extra></extra>"
                    ),
                )
            )

    for name, paths, color in (
        ("Worker path", workers, "#20e0a0"),
        ("Forklift path", forklifts, "#ffd166"),
    ):
        for path_index, path in enumerate(paths, start=1):
            figure.add_trace(
                go.Scatter3d(
                    x=[column for _, column in path],
                    y=[height - 1 - row for row, _ in path],
                    z=[0.14] * len(path),
                    mode="lines+markers",
                    line={"color": color, "width": 7},
                    marker={"size": 4},
                    name=f"{name} {path_index}",
                    legendgroup=name,
                    hovertemplate=f"{name}<br>X=%{{x}}, Y=%{{y}}<extra></extra>",
                )
            )

    figure.update_layout(
        title=f"Warehouse 3D model ({width} x {height} grid)",
        scene={
            "xaxis_title": "X (column)",
            "yaxis_title": "Y (row)",
            "zaxis_title": "Z (height)",
            "xaxis": {
                "range": [-0.5, width - 0.5],
                "showbackground": False,
                "gridcolor": "#8d98a5",
                "zerolinecolor": "#d5dce3",
            },
            "yaxis": {
                "range": [-0.5, height - 0.5],
                "showbackground": False,
                "gridcolor": "#8d98a5",
                "zerolinecolor": "#d5dce3",
            },
            "zaxis": {
                "range": [0, rack_height * 1.2],
                "showbackground": True,
                "backgroundcolor": "#343c47",
                "gridcolor": "#8d98a5",
                "zerolinecolor": "#d5dce3",
            },
            "aspectmode": "manual",
            "aspectratio": {
                "x": width,
                "y": height,
                "z": max(rack_height * 2.3, 1),
            },
            "camera": {
                "eye": {"x": 1.55, "y": 1.6, "z": 1.15},
                "up": {"x": 0, "y": 0, "z": 1},
            },
            "dragmode": "orbit",
        },
        paper_bgcolor="#111820",
        plot_bgcolor="#111820",
        font={"color": "#edf2f7"},
        legend={
            "title": "Warehouse elements",
            "bgcolor": "rgba(17,24,32,0.82)",
            "bordercolor": "#66717e",
            "borderwidth": 1,
        },
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )
    return figure
