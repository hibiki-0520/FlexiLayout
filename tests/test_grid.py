import unittest

import pandas as pd

from warehouse_sim.layout.grid import WarehouseGrid


class WarehouseGridTests(unittest.TestCase):
    def test_creates_connected_io_and_aisles(self) -> None:
        warehouse = WarehouseGrid(width=20, height=18, aisle_spacing=5)

        self.assertEqual(warehouse.grid.shape, (18, 20))
        self.assertEqual(warehouse.grid[17, 10], "I")
        self.assertTrue((warehouse.grid == "A").any())

    def test_slotting_prioritizes_abc_then_nearby_shelves(self) -> None:
        warehouse = WarehouseGrid(width=20, height=20, aisle_spacing=5)
        products = pd.DataFrame(
            {
                "product_id": ["c", "b", "a"],
                "pick_frequency": [10, 100, 1000],
            }
        )

        assigned = warehouse.optimize_slotting(products)

        self.assertEqual(assigned["product_id"].tolist(), ["a", "b", "c"])
        self.assertEqual(assigned["abc_class"].tolist(), ["A", "B", "C"])
        self.assertEqual(assigned.iloc[0]["distance_from_io"], min(assigned["distance_from_io"]))
        for row, column in zip(assigned["row"], assigned["column"]):
            self.assertEqual(warehouse.grid[row, column], "S")

    def test_uses_provided_abc_classes(self) -> None:
        warehouse = WarehouseGrid(width=12, height=12, aisle_spacing=4)
        products = pd.DataFrame(
            {
                "product_id": ["class-c", "class-a"],
                "pick_frequency": [1000, 1],
                "abc_class": ["C", "A"],
            }
        )

        assigned = warehouse.optimize_slotting(products)

        self.assertEqual(assigned["product_id"].tolist(), ["class-a", "class-c"])

    def test_shelves_are_aisle_accessible_at_maximum_app_sku_count(self) -> None:
        warehouse = WarehouseGrid(width=50, height=50, aisle_spacing=5)
        products = pd.DataFrame(
            {
                "product_id": [f"sku-{index}" for index in range(1000)],
                "pick_frequency": list(range(1000, 0, -1)),
            }
        )

        assigned = warehouse.optimize_slotting(products)

        self.assertEqual(len(assigned), 1000)
        for row, column in zip(assigned["row"], assigned["column"]):
            self.assertTrue(
                any(
                    0 <= row + row_delta < warehouse.height
                    and 0 <= column + column_delta < warehouse.width
                    and warehouse.grid[row + row_delta, column + column_delta]
                    in {"A", "I"}
                    for row_delta, column_delta in (
                        (-1, 0),
                        (0, -1),
                        (0, 1),
                        (1, 0),
                    )
                )
            )

    def test_rejects_invalid_io_and_frequency(self) -> None:
        with self.assertRaises(ValueError):
            WarehouseGrid(width=10, height=10, io_points=[(5, 5)])

        warehouse = WarehouseGrid(width=10, height=10)
        products = pd.DataFrame({"product_id": ["x"], "pick_frequency": [-1]})
        with self.assertRaises(ValueError):
            warehouse.optimize_slotting(products)


if __name__ == "__main__":
    unittest.main()
