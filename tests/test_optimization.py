import unittest

import pandas as pd

from warehouse_sim.simulation import (
    CargoSize,
    PickTask,
    PickingOrder,
    optimize_conditions,
)


class OptimizationTests(unittest.TestCase):
    def test_finds_least_cost_configuration_within_target(self) -> None:
        products = pd.DataFrame(
            {"product_id": ["sku"], "pick_frequency": [10]}
        )
        orders = [PickingOrder("order", [PickTask("sku")])]

        result = optimize_conditions(
            products,
            orders,
            (9, 5),
            width=10,
            height=10,
            aisle_spacing=3,
            cargo_weight_kg=1.0,
            cargo_size_class=CargoSize.SMALL,
            max_transport_units=5,
            target_makespan=100,
            max_workers=2,
            max_forklifts=0,
        )

        self.assertTrue(result.feasible)
        self.assertEqual(len(result.candidates), 4)
        self.assertTrue(result.candidates.iloc[0]["feasible"])
        self.assertEqual(result.candidates.iloc[0]["workers"], 1)
        self.assertEqual(result.candidates.iloc[0]["forklifts"], 0)
        self.assertEqual(
            set(result.candidates["placement_rule"]),
            {"abc", "frequency"},
        )

    def test_reports_fastest_candidate_when_target_is_unreachable(self) -> None:
        products = pd.DataFrame(
            {"product_id": ["sku"], "pick_frequency": [10]}
        )
        result = optimize_conditions(
            products,
            [PickingOrder("order", [PickTask("sku")])],
            (9, 5),
            width=10,
            height=10,
            aisle_spacing=3,
            cargo_weight_kg=1.0,
            cargo_size_class=CargoSize.SMALL,
            max_transport_units=5,
            target_makespan=0.01,
            max_workers=2,
            max_forklifts=0,
        )

        self.assertFalse(result.feasible)
        self.assertEqual(
            result.candidates.iloc[0]["makespan"],
            result.candidates["makespan"].min(),
        )

    def test_supports_all_optimization_objectives_and_budget_fallback(self) -> None:
        products = pd.DataFrame(
            {"product_id": ["sku"], "pick_frequency": [10]}
        )
        orders = [PickingOrder("order", [PickTask("sku", quantity=2)])]
        common = {
            "products": products,
            "orders": orders,
            "io_point": (9, 5),
            "width": 10,
            "height": 10,
            "aisle_spacing": 3,
            "cargo_weight_kg": 1.0,
            "cargo_size_class": CargoSize.MEDIUM,
            "max_transport_units": 5,
            "target_makespan": 100,
            "max_workers": 2,
            "max_forklifts": 1,
        }

        time_result = optimize_conditions(**common, mode="time")
        cost_result = optimize_conditions(
            **common,
            mode="cost",
            budget=0,
        )
        balance_result = optimize_conditions(
            **common,
            mode="balance",
            min_worker_utilization=0.0,
            min_forklift_utilization=0.0,
        )
        eco_result = optimize_conditions(**common, mode="eco")

        self.assertEqual(time_result.mode, "time")
        self.assertEqual(
            time_result.candidates.iloc[0]["makespan"],
            time_result.candidates["makespan"].min(),
        )
        self.assertFalse(cost_result.feasible)
        self.assertIn("予算", cost_result.recommendation_reason)
        self.assertTrue(balance_result.feasible)
        self.assertEqual(
            balance_result.candidates.iloc[0]["cost_per_item"],
            balance_result.candidates[balance_result.candidates["feasible"]][
                "cost_per_item"
            ].min(),
        )
        self.assertGreater(eco_result.candidates.iloc[0]["forklift_distance"], 0)
        self.assertEqual(
            eco_result.candidates.iloc[0]["estimated_co2_kg"],
            eco_result.candidates["estimated_co2_kg"].min(),
        )
        self.assertTrue(
            {
                "estimated_co2_kg",
                "forklift_distance",
                "cost_per_item",
                "worker_utilization",
            }.issubset(eco_result.candidates.columns)
        )


if __name__ == "__main__":
    unittest.main()
