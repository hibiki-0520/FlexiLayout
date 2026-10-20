"""2D warehouse grid generation and ABC-based shelf slotting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class WarehouseGrid:
    """Create a grid with aisles and I/O points, then assign products to shelves.

    Coordinates are ``(row, column)``. Grid cells contain ``.`` for floor,
    ``A`` for aisle, ``I`` for an I/O point, and ``S`` for an occupied shelf.
    """

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        aisle_spacing: int = 8,
        io_points: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        if width < 3 or height < 3:
            raise ValueError("width and height must both be at least 3")
        if aisle_spacing < 1:
            raise ValueError("aisle_spacing must be a positive integer")

        self.width = width
        self.height = height
        self.aisle_spacing = aisle_spacing
        if io_points is None:
            points = [(height - 1, width // 2)]
        else:
            points = list(io_points)
        if not points:
            raise ValueError("At least one I/O point is required")
        if len(set(points)) != len(points):
            raise ValueError("I/O points must not contain duplicates")
        for row, column in points:
            if not (0 <= row < height and 0 <= column < width):
                raise ValueError(f"I/O point {(row, column)} is outside the grid")
            if row not in (0, height - 1) and column not in (0, width - 1):
                raise ValueError(f"I/O point {(row, column)} must be on the grid boundary")

        self.io_points = points
        self.grid = np.full((height, width), ".", dtype="<U1")
        self._shelf_positions: list[tuple[int, int]] = []
        self._create_aisles()
        self._connect_io_points()
        self._shelf_positions = [
            (row, column)
            for row in range(1, height - 1)
            for column in range(1, width - 1)
            if self.grid[row, column] == "."
        ]
        self._shelf_positions.sort(key=self._shelf_sort_key)

    def _aisle_indices(self, size: int) -> list[int]:
        indices = list(range(self.aisle_spacing, size - 1, self.aisle_spacing))
        if not indices:
            indices = [size // 2]
        return indices

    def _create_aisles(self) -> None:
        for row in self._aisle_indices(self.height):
            self.grid[row, :] = "A"
        for column in self._aisle_indices(self.width):
            self.grid[:, column] = "A"

    def _connect_io_points(self) -> None:
        aisle_cells = np.argwhere(self.grid == "A")
        for point in self.io_points:
            row, column = point
            nearest_row, nearest_column = min(
                ((int(r), int(c)) for r, c in aisle_cells),
                key=lambda cell: abs(row - cell[0]) + abs(column - cell[1]),
            )
            step = 1 if nearest_row >= row else -1
            for route_row in range(row, nearest_row + step, step):
                if (route_row, column) != point:
                    self.grid[route_row, column] = "A"
            step = 1 if nearest_column >= column else -1
            for route_column in range(column, nearest_column + step, step):
                if (nearest_row, route_column) != point:
                    self.grid[nearest_row, route_column] = "A"
            self.grid[point] = "I"

    def _shelf_sort_key(self, position: tuple[int, int]) -> tuple[int, int, int]:
        row, column = position
        distance = min(
            abs(row - io_row) + abs(column - io_column)
            for io_row, io_column in self.io_points
        )
        return distance, row, column

    @staticmethod
    def _derive_abc_classes(frequencies: np.ndarray) -> np.ndarray:
        """Classify products by cumulative pick-frequency share (80/15/5)."""
        classes = np.full(len(frequencies), "C", dtype="<U1")
        total = frequencies.sum()
        if total == 0:
            return classes

        sorted_indices = np.argsort(-frequencies, kind="stable")
        cumulative_before = np.cumsum(frequencies[sorted_indices]) - frequencies[sorted_indices]
        cumulative_share_before = cumulative_before / total
        for index, share in zip(sorted_indices, cumulative_share_before):
            if share < 0.80:
                classes[index] = "A"
            elif share < 0.95:
                classes[index] = "B"
        return classes

    def optimize_slotting(
        self,
        products: pd.DataFrame,
        item_col: str = "product_id",
        frequency_col: str = "pick_frequency",
        abc_col: str | None = None,
    ) -> pd.DataFrame:
        """Assign products to shelf cells, prioritizing ABC class and I/O distance.

        If ``abc_col`` is omitted, an existing ``abc_class`` column is used when
        available; otherwise classes are derived from cumulative pick frequency.
        The returned DataFrame includes ``abc_class``, ``row``, ``column``, and
        ``distance_from_io`` columns.
        """
        required_columns = {item_col, frequency_col}
        missing_columns = required_columns.difference(products.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        if products[item_col].isna().any() or products[item_col].duplicated().any():
            raise ValueError(f"{item_col} values must be non-null and unique")

        frequencies = pd.to_numeric(products[frequency_col], errors="raise").to_numpy(
            dtype=float
        )
        if not np.isfinite(frequencies).all() or (frequencies < 0).any():
            raise ValueError(f"{frequency_col} values must be finite and non-negative")

        category_column = abc_col
        if category_column is None and "abc_class" in products.columns:
            category_column = "abc_class"
        if category_column is not None:
            if category_column not in products.columns:
                raise ValueError(f"ABC class column {category_column!r} was not found")
            classes = products[category_column].astype("string").str.upper()
            if classes.isna().any() or not classes.isin(["A", "B", "C"]).all():
                raise ValueError(f"{category_column} values must be A, B, or C")
            classes = classes.to_numpy(dtype=str)
        else:
            classes = self._derive_abc_classes(frequencies)

        if len(products) > len(self._shelf_positions):
            raise ValueError(
                f"Not enough shelf cells: {len(products)} products for "
                f"{len(self._shelf_positions)} available cells"
            )

        self.grid[self.grid == "S"] = "."
        abc_rank = pd.Series(classes).map({"A": 0, "B": 1, "C": 2}).to_numpy()
        ordered = products.copy()
        ordered["abc_class"] = classes
        ordered["_abc_rank"] = abc_rank
        ordered["_frequency"] = frequencies
        ordered = ordered.sort_values(
            ["_abc_rank", "_frequency"],
            ascending=[True, False],
            kind="stable",
        ).drop(columns=["_abc_rank", "_frequency"])

        positions = self._shelf_positions[: len(ordered)]
        ordered["row"] = [row for row, _ in positions]
        ordered["column"] = [column for _, column in positions]
        ordered["distance_from_io"] = [
            min(
                abs(row - io_row) + abs(column - io_column)
                for io_row, io_column in self.io_points
            )
            for row, column in positions
        ]
        for row, column in positions:
            self.grid[row, column] = "S"
        return ordered.reset_index(drop=True)
