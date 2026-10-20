import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from warehouse_sim.visualization import plot_warehouse_grid


class WarehouseGridVisualizationTests(unittest.TestCase):
    def test_draws_grid_and_abc_legend(self) -> None:
        grid = np.array(
            [
                [".", "A", "."],
                ["I", "S", "S"],
            ]
        )
        slotting = pd.DataFrame(
            {
                "row": [1],
                "column": [1],
                "abc_class": ["A"],
            }
        )

        figure = plot_warehouse_grid(grid, slotting)
        try:
            axis = figure.axes[0]
            labels = [text.get_text() for text in axis.get_legend().get_texts()]
            self.assertEqual(axis.images[0].get_array().shape, grid.shape)
            self.assertIn("Floor / empty", labels)
            self.assertIn("Aisle", labels)
            self.assertIn("Entrance / I/O", labels)
            self.assertIn("Shelf - A", labels)
            self.assertIn("Shelf - unclassified", labels)
        finally:
            plt.close(figure)

    def test_rejects_unknown_grid_cells(self) -> None:
        with self.assertRaises(ValueError):
            plot_warehouse_grid(np.array([["?"]]))

    def test_rejects_slotting_that_does_not_match_shelf(self) -> None:
        grid = np.array([[".", "S"]])
        slotting = pd.DataFrame(
            {"row": [0], "column": [0], "abc_class": ["A"]}
        )

        with self.assertRaises(ValueError):
            plot_warehouse_grid(grid, slotting)

    def test_draws_current_agent_positions(self) -> None:
        positions = pd.DataFrame(
            {
                "agent_id": ["worker-1", "forklift-1"],
                "agent_type": ["worker", "forklift"],
                "x": [1.0, 2.0],
                "y": [0.0, 1.0],
            }
        )
        figure = plot_warehouse_grid(
            np.array([["I", ".", "S"], [".", ".", "."]]),
            agent_positions=positions,
        )
        try:
            axis = figure.axes[0]
            self.assertEqual(len(axis.collections), 2)
            labels = [text.get_text() for text in axis.get_legend().get_texts()]
            self.assertIn("Worker", labels)
            self.assertIn("Forklift", labels)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
