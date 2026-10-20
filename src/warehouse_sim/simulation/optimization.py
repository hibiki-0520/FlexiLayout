"""Multi-objective search for warehouse resource configurations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import pandas as pd

from warehouse_sim.layout.grid import WarehouseGrid
from warehouse_sim.simulation.picking import (
    CargoSize,
    PickingOrder,
    simulate_picking,
)

OPTIMIZATION_MODES = ("time", "cost", "balance", "eco")


@dataclass(frozen=True)
class OptimizationResult:
    """Candidate configurations and the selected objective mode."""

    candidates: pd.DataFrame
    feasible: bool
    mode: str
    recommendation_reason: str


def optimize_conditions(
    products: pd.DataFrame,
    orders: Sequence[PickingOrder],
    io_point: tuple[int, int],
    *,
    width: int,
    height: int,
    aisle_spacing: int,
    cargo_weight_kg: float,
    cargo_size_class: CargoSize | str,
    max_transport_units: int,
    target_makespan: float,
    max_workers: int,
    max_forklifts: int,
    worker_hourly_cost: float = 30.0,
    forklift_hourly_cost: float = 50.0,
    mode: str = "cost",
    budget: float | None = None,
    min_worker_utilization: float = 0.25,
    min_forklift_utilization: float = 0.2,
    co2_per_forklift_cell: float = 0.2,
    co2_per_forklift_hour: float = 0.5,
) -> OptimizationResult:
    """Evaluate configurations and rank them for a selected objective.

    Modes are ``time`` (minimum makespan), ``cost`` (minimum estimated
    labor/equipment cost within time and optional budget constraints),
    ``balance`` (minimum cost per picked item after utilization thresholds),
    and ``eco`` (minimum estimated forklift CO2-equivalent proxy).
    The eco estimate uses caller-supplied emission factors per grid cell and
    active hour; defaults are illustrative rather than certified factors.
    """
    if mode not in OPTIMIZATION_MODES:
        raise ValueError(f"mode must be one of {OPTIMIZATION_MODES}")
    if (
        not isinstance(max_workers, int)
        or isinstance(max_workers, bool)
        or not 1 <= max_workers <= 100
    ):
        raise ValueError("max_workers must be between 1 and 100")
    if (
        not isinstance(max_forklifts, int)
        or isinstance(max_forklifts, bool)
        or not 0 <= max_forklifts <= 50
    ):
        raise ValueError("max_forklifts must be between 0 and 50")
    if not isinstance(target_makespan, (int, float)) or not isfinite(
        target_makespan
    ) or target_makespan <= 0:
        raise ValueError("target_makespan must be positive and finite")
    for name, value in (
        ("worker_hourly_cost", worker_hourly_cost),
        ("forklift_hourly_cost", forklift_hourly_cost),
        ("co2_per_forklift_cell", co2_per_forklift_cell),
        ("co2_per_forklift_hour", co2_per_forklift_hour),
    ):
        if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if budget is not None and (
        not isinstance(budget, (int, float)) or not isfinite(budget) or budget < 0
    ):
        raise ValueError("budget must be finite and non-negative")
    for name, utilization in (
        ("min_worker_utilization", min_worker_utilization),
        ("min_forklift_utilization", min_forklift_utilization),
    ):
        if (
            not isinstance(utilization, (int, float))
            or not isfinite(utilization)
            or not 0 <= utilization <= 1
        ):
            raise ValueError(f"{name} must be between 0 and 1")

    cargo_size = CargoSize(cargo_size_class)
    minimum_forklifts = 0 if cargo_size is CargoSize.SMALL else 1
    if max_forklifts < minimum_forklifts:
        raise ValueError(
            f"{cargo_size.value} requires max_forklifts >= {minimum_forklifts}"
        )
    configuration_count = (
        2 * max_workers * (max_forklifts - minimum_forklifts + 1)
    )
    if configuration_count > 500:
        raise ValueError(
            "Search exceeds 500 configurations; reduce the resource search limits"
        )

    item_count = sum(pick.quantity for order in orders for pick in order.picks)
    candidates: list[dict[str, int | float | str | bool]] = []
    for placement_rule in ("abc", "frequency"):
        warehouse = WarehouseGrid(
            width=width,
            height=height,
            aisle_spacing=aisle_spacing,
            io_points=[io_point],
        )
        slotting = warehouse.optimize_slotting(
            products,
            placement_rule=placement_rule,
        )
        shelf_positions = {
            row.product_id: (int(row.row), int(row.column))
            for row in slotting.itertuples(index=False)
        }

        for worker_count in range(1, max_workers + 1):
            for forklift_count in range(minimum_forklifts, max_forklifts + 1):
                result = simulate_picking(
                    orders,
                    shelf_positions=shelf_positions,
                    io_point=io_point,
                    worker_count=worker_count,
                    forklift_count=forklift_count,
                    container_count=worker_count,
                    cargo_weight_kg=cargo_weight_kg,
                    cargo_size_class=cargo_size,
                    max_transport_units=max_transport_units,
                    record_trace=False,
                    warehouse_grid=warehouse.grid,
                )
                labor_cost = result.makespan * worker_count * worker_hourly_cost
                equipment_cost = (
                    result.forklift_active_time_total * forklift_hourly_cost
                )
                estimated_cost = labor_cost + equipment_cost
                cost_per_item = (
                    estimated_cost / item_count if item_count else 0.0
                )
                estimated_co2 = (
                    result.forklift_distance_total * co2_per_forklift_cell
                    + result.forklift_active_time_total * co2_per_forklift_hour
                )
                meets_time = result.makespan <= target_makespan
                meets_budget = budget is None or estimated_cost <= budget
                meets_utilization = (
                    result.worker_utilization >= min_worker_utilization
                    and (
                        forklift_count == 0
                        or result.forklift_utilization >= min_forklift_utilization
                    )
                )
                if mode == "cost":
                    eligible = meets_time and meets_budget
                elif mode == "balance":
                    eligible = meets_utilization
                else:
                    eligible = True
                candidates.append(
                    {
                        "workers": worker_count,
                        "forklifts": forklift_count,
                        "placement_rule": placement_rule,
                        "makespan": result.makespan,
                        "estimated_cost": estimated_cost,
                        "cost_per_item": cost_per_item,
                        "estimated_co2_kg": estimated_co2,
                        "forklift_distance": result.forklift_distance_total,
                        "forklift_active_time": result.forklift_active_time_total,
                        "worker_utilization": result.worker_utilization,
                        "forklift_utilization": result.forklift_utilization,
                        "meets_time_target": meets_time,
                        "meets_budget": meets_budget,
                        "meets_utilization_target": meets_utilization,
                        "feasible": eligible,
                    }
                )

    frame = pd.DataFrame(candidates)
    any_feasible = bool(frame["feasible"].any())
    if mode == "time":
        frame = frame.sort_values(
            ["makespan", "estimated_cost"],
            ascending=[True, True],
            kind="stable",
        )
        reason = "総費用を評価対象から外し、メイクスパンを最小化しました。"
    elif mode == "cost":
        if any_feasible:
            frame = frame.sort_values(
                ["feasible", "estimated_cost", "makespan"],
                ascending=[False, True, True],
                kind="stable",
            )
            reason = "時間目標と予算を満たす候補から推定費用が最小の構成を選びました。"
        else:
            frame = frame.sort_values(
                ["makespan", "estimated_cost"],
                ascending=[True, True],
                kind="stable",
            )
            reason = "時間目標または予算を満たす候補がなく、最短時間の構成を提示します。"
    elif mode == "balance":
        if any_feasible:
            frame = frame.sort_values(
                ["feasible", "cost_per_item", "estimated_cost"],
                ascending=[False, True, True],
                kind="stable",
            )
            reason = (
                "指定された最低稼働率を満たす候補から、1アイテムあたりの"
                "推定費用が最小の構成を選びました。"
            )
        else:
            frame["_utilization_shortfall"] = (
                (min_worker_utilization - frame["worker_utilization"]).clip(lower=0)
                + (
                    (min_forklift_utilization - frame["forklift_utilization"]).clip(lower=0)
                    * (frame["forklifts"] > 0)
                )
            )
            frame = frame.sort_values(
                ["_utilization_shortfall", "cost_per_item", "estimated_cost"],
                ascending=[True, True, True],
                kind="stable",
            )
            frame = frame.drop(columns="_utilization_shortfall")
            reason = (
                "最低稼働率をすべて満たす候補がなく、稼働率不足が最小の構成を"
                "費用対効果順に提示します。"
            )
    else:
        frame = frame.sort_values(
            ["estimated_co2_kg", "forklift_distance", "forklift_active_time", "makespan"],
            ascending=[True, True, True, True],
            kind="stable",
        )
        reason = (
            "フォークリフトの移動距離と稼働時間から算出したCO2換算推定値を"
            "最小化しました。"
        )

    return OptimizationResult(
        candidates=frame.reset_index(drop=True),
        feasible=any_feasible,
        mode=mode,
        recommendation_reason=reason,
    )
