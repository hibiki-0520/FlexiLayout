"""Matplotlib visualization for warehouse grids."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def plot_warehouse_grid(
    grid: np.ndarray,
    slotting: pd.DataFrame | None = None,
    agent_positions: pd.DataFrame | None = None,
) -> Figure:
    """Draw a warehouse grid with a legend for floor, aisles, I/O, and shelves.

    When a slotting DataFrame with ``row``, ``column``, and ``abc_class``
    columns is provided, shelves are colored by ABC class. Shelf cells without
    a matching slotting row use the generic shelf color.
    """
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty two-dimensional array")

    cell_values = {".": 0, "A": 1, "I": 2, "S": 6}
    unknown_cells = set(np.unique(grid)).difference(cell_values)
    if unknown_cells:
        raise ValueError(f"grid contains unknown cell types: {sorted(unknown_cells)}")
    image = np.vectorize(cell_values.__getitem__)(grid)

    if slotting is not None:
        required_columns = {"row", "column", "abc_class"}
        missing_columns = required_columns.difference(slotting.columns)
        if missing_columns:
            raise ValueError(
                f"slotting is missing required columns: {sorted(missing_columns)}"
            )
        for row in slotting.itertuples(index=False):
            cell_row, cell_column = int(row.row), int(row.column)
            if not (
                0 <= cell_row < grid.shape[0]
                and 0 <= cell_column < grid.shape[1]
            ):
                raise ValueError(
                    f"slotting coordinate {(cell_row, cell_column)} is outside the grid"
                )
            if grid[cell_row, cell_column] != "S":
                raise ValueError(
                    f"slotting coordinate {(cell_row, cell_column)} is not a shelf cell"
                )
            abc_class = str(row.abc_class).upper()
            if abc_class not in {"A", "B", "C"}:
                raise ValueError(
                    f"ABC class at {(cell_row, cell_column)} must be A, B, or C"
                )
            image[cell_row, cell_column] = {"A": 3, "B": 4, "C": 5}[abc_class]

    colors = [
        "#f4f5f7",
        "#9ca3af",
        "#e11d48",
        "#1d4ed8",
        "#3b82f6",
        "#93c5fd",
        "#2563eb",
    ]
    labels = [
        "Floor / empty",
        "Aisle",
        "Entrance / I/O",
        "Shelf - A",
        "Shelf - B",
        "Shelf - C",
        "Shelf - unclassified",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(image, cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_title(f"Warehouse layout ({grid.shape[1]} x {grid.shape[0]} grid)")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.set_xticks(np.arange(-0.5, grid.shape[1], 5), minor=True)
    axis.set_yticks(np.arange(-0.5, grid.shape[0], 5), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.6)
    axis.tick_params(which="minor", bottom=False, left=False)

    legend_indexes = [0, 1, 2, 6]
    if slotting is not None and not slotting.empty:
        legend_indexes.extend([3, 4, 5])
    handles = [
        Patch(facecolor=colors[index], label=labels[index])
        for index in legend_indexes
    ]
    axis.legend(handles=handles, loc="upper right", ncols=2, fontsize="small")
    if agent_positions is not None and not agent_positions.empty:
        marker_styles = {
            "worker": {"color": "#16a34a", "marker": "o", "label": "Worker"},
            "forklift": {"color": "#f59e0b", "marker": "s", "label": "Forklift"},
        }
        for agent_type, style in marker_styles.items():
            positions = agent_positions[agent_positions["agent_type"] == agent_type]
            if positions.empty:
                continue
            axis.scatter(
                positions["x"],
                positions["y"],
                c=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.5,
                s=55,
                label=style["label"],
                zorder=5,
            )
        axis.legend(loc="upper right", ncols=2, fontsize="small")
    figure.tight_layout()
    return figure
