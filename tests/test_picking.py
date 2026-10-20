import unittest

from warehouse_sim.simulation import PickTask, PickingOrder, simulate_picking


class PickingSimulationTests(unittest.TestCase):
    def test_calculates_makespan_and_resource_utilization(self) -> None:
        result = simulate_picking(
            [PickingOrder("order-1", [PickTask("sku", quantity=2)])],
            shelf_positions={"sku": (0, 2)},
            io_point=(0, 0),
            worker_count=1,
            forklift_count=1,
            container_count=1,
            worker_speed=1,
            forklift_speed=1,
            pick_time_per_unit=1,
        )

        self.assertEqual(result.makespan, 6)
        self.assertAlmostEqual(result.worker_utilization, 4 / 6)
        self.assertAlmostEqual(result.forklift_utilization, 2 / 6)
        self.assertEqual(result.order_completion_times, {"order-1": 6})
        self.assertEqual(result.worker_wait_time_total, 0)
        self.assertEqual(result.forklift_wait_time_total, 0)

    def test_reports_wait_for_single_forklift(self) -> None:
        orders = [
            PickingOrder("first", [PickTask("sku")]),
            PickingOrder("second", [PickTask("sku")]),
        ]

        result = simulate_picking(
            orders,
            shelf_positions={"sku": (0, 1)},
            io_point=(0, 0),
            worker_count=2,
            forklift_count=1,
            container_count=2,
            worker_speed=1,
            forklift_speed=1,
            pick_time_per_unit=1,
        )

        self.assertGreater(result.makespan, 0)
        self.assertGreater(result.forklift_wait_time_total, 0)
        self.assertEqual(len(result.order_completion_times), 2)
        self.assertLessEqual(result.worker_utilization, 1)
        self.assertLessEqual(result.forklift_utilization, 1)

    def test_validates_missing_shelf_and_resource_counts(self) -> None:
        with self.assertRaises(ValueError):
            simulate_picking(
                [PickingOrder("order", [PickTask("missing")])],
                shelf_positions={},
                io_point=(0, 0),
                worker_count=1,
                forklift_count=1,
                container_count=1,
            )

        with self.assertRaises(ValueError):
            simulate_picking(
                [],
                shelf_positions={},
                io_point=(0, 0),
                worker_count=0,
                forklift_count=1,
                container_count=1,
            )


if __name__ == "__main__":
    unittest.main()
