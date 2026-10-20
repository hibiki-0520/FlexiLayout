import unittest

import numpy as np
import pandas as pd

from warehouse_sim.visualization import plot_warehouse_3d


class WarehouseGrid3DVisualizationTests(unittest.TestCase):
    def test_draws_abc_shelves_and_io_in_3d(self) -> None:
        grid = np.array(
            [
                [".", "A", "S"],
                ["I", "S", "."],
            ]
        )
        slotting = pd.DataFrame(
            {
                "row": [0, 1],
                "column": [2, 1],
                "abc_class": ["A", "C"],
            }
        )

        figure = plot_warehouse_3d(grid, slotting, shelf_height=3)

        self.assertTrue(all(trace.type in {"surface", "scatter3d", "mesh3d"} for trace in figure.data))
        trace_names = [trace.name for trace in figure.data]
        self.assertIn("Shelf deck - A", trace_names)
        self.assertIn("Shelf deck - C", trace_names)
        self.assertIn("Steel rack frame", trace_names)
        self.assertIn("Aisle surface", trace_names)
        self.assertIn("Floor grid", trace_names)
        self.assertIn("5-cell markings", trace_names)
        self.assertIn("Entrance / I/O", trace_names)
        floor = next(trace for trace in figure.data if trace.name == "Concrete floor")
        self.assertEqual(floor.colorscale[0][1], "#202833")
        rack_frame = next(
            trace for trace in figure.data if trace.name == "Steel rack frame"
        )
        self.assertGreater(max(rack_frame.z), 0)
        self.assertEqual(figure.layout.scene.zaxis.range[0], 0)
        self.assertAlmostEqual(figure.layout.scene.zaxis.range[1], 3.6)

    def test_draws_optional_worker_and_forklift_paths(self) -> None:
        grid = np.array([["I", ".", "S"]])

        figure = plot_warehouse_3d(
            grid,
            worker_paths=[[(0, 0), (0, 1), (0, 2)]],
            forklift_paths=[[(0, 2), (0, 0)]],
        )

        trace_names = [trace.name for trace in figure.data]
        self.assertIn("Worker path 1", trace_names)
        self.assertIn("Forklift path 1", trace_names)

    def test_draws_current_agent_positions(self) -> None:
        positions = pd.DataFrame(
            {
                "agent_id": ["worker-1", "forklift-1"],
                "agent_type": ["worker", "forklift"],
                "x": [0.0, 2.0],
                "y": [0.0, 0.0],
                "state": ["moving", "waiting"],
            }
        )
        figure = plot_warehouse_3d(
            np.array([["I", ".", "S"]]),
            agent_positions=positions,
        )
        trace_names = [trace.name for trace in figure.data]
        self.assertIn("Workers", trace_names)
        self.assertIn("Forklifts", trace_names)

    def test_rejects_invalid_grid_and_path(self) -> None:
        with self.assertRaises(ValueError):
            plot_warehouse_3d(np.array([["?"]]))
        with self.assertRaises(ValueError):
            plot_warehouse_3d(np.array([["."]]), worker_paths=[[(1, 0)]])


if __name__ == "__main__":
    unittest.main()
