import unittest

import numpy as np

from warehouse_sim.simulation import CargoSize, PickTask, PickingOrder, simulate_picking


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

        self.assertEqual(result.makespan, 10)
        self.assertAlmostEqual(result.worker_utilization, 6 / 10)
        self.assertAlmostEqual(result.forklift_utilization, 4 / 10)
        self.assertEqual(result.order_completion_times, {"order-1": 10})
        self.assertEqual(result.forklift_distance_total, 4)
        self.assertEqual(result.item_count_total, 2)
        self.assertEqual(result.worker_wait_time_total, 0)
        self.assertEqual(result.forklift_wait_time_total, 0)
        self.assertEqual(result.trace["agent_id"].nunique(), 2)
        self.assertTrue(
            result.trace.groupby("time")["agent_id"].nunique().eq(2).all()
        )
        final_worker = result.trace[
            (result.trace["agent_id"] == "worker-1")
            & (result.trace["time"] == result.makespan)
        ].iloc[-1]
        self.assertEqual((final_worker["x"], final_worker["y"]), (0, 0))
        self.assertEqual(final_worker["completed_order_count"], 1)

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

    def test_heavier_cargo_increases_makespan(self) -> None:
        order = [PickingOrder("order", [PickTask("sku", quantity=2)])]
        common = {
            "orders": order,
            "shelf_positions": {"sku": (0, 2)},
            "io_point": (0, 0),
            "worker_count": 1,
            "forklift_count": 1,
            "container_count": 1,
            "cargo_size_class": CargoSize.MEDIUM,
            "max_transport_units": 5,
            "worker_speed": 1,
            "forklift_speed": 1,
            "pick_time_per_unit": 1,
        }

        light = simulate_picking(**common, cargo_weight_kg=1)
        heavy = simulate_picking(**common, cargo_weight_kg=50)

        self.assertGreater(heavy.makespan, light.makespan)
        self.assertGreater(heavy.forklift_utilization, 0)

    def test_small_cargo_can_be_hand_carried_without_forklift(self) -> None:
        result = simulate_picking(
            [PickingOrder("order", [PickTask("sku", quantity=2)])],
            shelf_positions={"sku": (0, 2)},
            io_point=(0, 0),
            worker_count=1,
            forklift_count=0,
            container_count=1,
            cargo_size_class=CargoSize.SMALL,
            max_transport_units=1,
        )

        self.assertGreater(result.makespan, 0)
        self.assertEqual(result.forklift_utilization, 0)

    def test_large_cargo_requires_forklift_and_batches_by_capacity(self) -> None:
        order = [PickingOrder("order", [PickTask("sku", quantity=3)])]
        common = {
            "orders": order,
            "shelf_positions": {"sku": (0, 1)},
            "io_point": (0, 0),
            "worker_count": 1,
            "container_count": 1,
            "cargo_size_class": CargoSize.LARGE,
            "max_transport_units": 1,
        }
        with self.assertRaisesRegex(ValueError, "requires at least one forklift"):
            simulate_picking(**common, forklift_count=0)

        batched_result = simulate_picking(**common, forklift_count=1)
        full_load_result = simulate_picking(
            **{**common, "max_transport_units": 3},
            forklift_count=1,
        )
        self.assertEqual(batched_result.forklift_wait_time_total, 0)
        self.assertGreater(batched_result.forklift_utilization, 0)
        self.assertGreater(batched_result.makespan, full_load_result.makespan)

    def test_trace_records_intermediate_positions_and_waiting_time(self) -> None:
        result = simulate_picking(
            [
                PickingOrder("first", [PickTask("sku")]),
                PickingOrder("second", [PickTask("sku")]),
            ],
            shelf_positions={"sku": (0, 2)},
            io_point=(0, 0),
            worker_count=1,
            forklift_count=0,
            container_count=1,
            cargo_size_class=CargoSize.SMALL,
            trace_interval=0.5,
        )

        worker_trace = result.trace[result.trace["agent_type"] == "worker"]
        self.assertTrue({"moving", "picking", "waiting"}.intersection(worker_trace["state"]))
        self.assertTrue(worker_trace["x"].between(0, 2).all())
        self.assertGreater(worker_trace["wait_time_total"].max(), 0)
        self.assertGreater(worker_trace["queue_count"].max(), 0)
        self.assertEqual(
            int(worker_trace["completed_order_count"].max()),
            len(result.order_completion_times),
        )

    def test_routes_around_shelves_using_aisle_grid(self) -> None:
        grid = np.array(
            [
                [".", ".", "."],
                ["I", "S", "S"],
                [".", ".", "."],
            ]
        )
        result = simulate_picking(
            [PickingOrder("order", [PickTask("sku")])],
            shelf_positions={"sku": (1, 2)},
            io_point=(1, 0),
            worker_count=1,
            forklift_count=0,
            container_count=1,
            cargo_size_class=CargoSize.SMALL,
            trace_interval=0.25,
            warehouse_grid=grid,
        )

        worker_trace = result.trace[result.trace["agent_type"] == "worker"]
        self.assertFalse(
            ((worker_trace["x"] == 1) & (worker_trace["y"] == 1)).any()
        )
        self.assertEqual(result.makespan, 9)
        self.assertTrue(
            (
                (worker_trace["x"] == 0)
                | (worker_trace["x"] == 2)
                | (worker_trace["y"] != 1)
            ).all()
        )
        movement = worker_trace[worker_trace["state"] == "moving"]
        self.assertTrue(
            (
                movement["x"].diff().fillna(0).abs()
                + movement["y"].diff().fillna(0).abs()
            ).le(0.25).all()
        )

    def test_reports_shelf_without_aisle_route(self) -> None:
        grid = np.array(
            [
                ["I", "S", "S"],
                ["S", "S", "S"],
                ["S", "S", "S"],
            ]
        )
        with self.assertRaisesRegex(ValueError, "No aisle route"):
            simulate_picking(
                [PickingOrder("order", [PickTask("sku")])],
                shelf_positions={"sku": (2, 2)},
                io_point=(0, 0),
                worker_count=1,
                forklift_count=0,
                container_count=1,
                cargo_size_class=CargoSize.SMALL,
                warehouse_grid=grid,
            )


if __name__ == "__main__":
    unittest.main()
