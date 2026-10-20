"""SimPy-based warehouse order-picking simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import simpy

Coordinate = tuple[int, int]


@dataclass(frozen=True)
class PickTask:
    """A quantity of one product to pick for an order."""

    product_id: str
    quantity: int = 1


@dataclass(frozen=True)
class PickingOrder:
    """An order containing the products and quantities to pick."""

    order_id: str
    picks: Sequence[PickTask]


@dataclass(frozen=True)
class SimulationResult:
    """Aggregate completion and resource KPIs; time values use simulation units."""

    makespan: float
    worker_utilization: float
    forklift_utilization: float
    worker_wait_time_total: float
    worker_wait_time_average: float
    forklift_wait_time_total: float
    forklift_wait_time_average: float
    container_wait_time_total: float
    container_wait_time_average: float
    order_completion_times: Mapping[str, float]


def _validate_coordinate(name: str, coordinate: Coordinate) -> None:
    if (
        not isinstance(coordinate, tuple)
        or len(coordinate) != 2
        or any(not isinstance(value, int) or value < 0 for value in coordinate)
    ):
        raise ValueError(f"{name} must be a pair of non-negative integers")


def simulate_picking(
    orders: Sequence[PickingOrder],
    shelf_positions: Mapping[str, Coordinate],
    io_point: Coordinate,
    worker_count: int,
    forklift_count: int,
    container_count: int,
    *,
    worker_speed: float = 1.0,
    forklift_speed: float = 2.0,
    pick_time_per_unit: float = 1.0,
) -> SimulationResult:
    """Run order picking and return makespan, utilization, and wait-time KPIs.

    Distances use Manhattan distance between grid cells. Workers begin each
    order at the I/O point, visit its shelves in the supplied order, then wait
    for a forklift to return the container to the I/O point. Speeds are cells
    per simulation-time unit; picking time is simulation-time units per item.
    Resource utilization is productive time divided by fleet capacity over the
    makespan. Queue wait times are reported as both totals and per-request
    averages.
    """
    if worker_count <= 0 or forklift_count <= 0 or container_count <= 0:
        raise ValueError("worker, forklift, and container counts must be positive")
    for name, value in (
        ("worker_speed", worker_speed),
        ("forklift_speed", forklift_speed),
        ("pick_time_per_unit", pick_time_per_unit),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    _validate_coordinate("io_point", io_point)
    for product_id, position in shelf_positions.items():
        if not product_id:
            raise ValueError("Product IDs in shelf_positions must not be empty")
        _validate_coordinate(f"shelf_positions[{product_id!r}]", position)

    order_ids = [order.order_id for order in orders]
    if any(not order_id for order_id in order_ids) or len(set(order_ids)) != len(
        order_ids
    ):
        raise ValueError("Order IDs must be non-empty and unique")
    for order in orders:
        for pick in order.picks:
            if pick.product_id not in shelf_positions:
                raise ValueError(
                    f"No shelf position is defined for product {pick.product_id!r}"
                )
            if not isinstance(pick.quantity, int) or pick.quantity <= 0:
                raise ValueError("Pick quantities must be positive integers")

    env = simpy.Environment()
    workers = simpy.Resource(env, capacity=worker_count)
    forklifts = simpy.Resource(env, capacity=forklift_count)
    containers = simpy.Resource(env, capacity=container_count)
    worker_active_time = 0.0
    forklift_active_time = 0.0
    worker_waits: list[float] = []
    forklift_waits: list[float] = []
    container_waits: list[float] = []
    completion_times: dict[str, float] = {}

    def process_order(order: PickingOrder):
        nonlocal worker_active_time, forklift_active_time

        requested_at = env.now
        with workers.request() as worker_request:
            yield worker_request
            worker_waits.append(env.now - requested_at)

            requested_at = env.now
            with containers.request() as container_request:
                yield container_request
                container_waits.append(env.now - requested_at)

                current_position = io_point
                for pick in order.picks:
                    shelf_position = shelf_positions[pick.product_id]
                    distance = abs(current_position[0] - shelf_position[0]) + abs(
                        current_position[1] - shelf_position[1]
                    )
                    travel_time = distance / worker_speed
                    yield env.timeout(travel_time)
                    worker_active_time += travel_time

                    pick_duration = pick.quantity * pick_time_per_unit
                    yield env.timeout(pick_duration)
                    worker_active_time += pick_duration
                    current_position = shelf_position

                requested_at = env.now
                with forklifts.request() as forklift_request:
                    yield forklift_request
                    forklift_waits.append(env.now - requested_at)

                    distance = abs(current_position[0] - io_point[0]) + abs(
                        current_position[1] - io_point[1]
                    )
                    transport_time = distance / forklift_speed
                    yield env.timeout(transport_time)
                    forklift_active_time += transport_time

            completion_times[order.order_id] = env.now

    processes = [env.process(process_order(order)) for order in orders]
    if processes:
        env.run(until=env.all_of(processes))
    makespan = env.now

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return SimulationResult(
        makespan=makespan,
        worker_utilization=(
            worker_active_time / (makespan * worker_count) if makespan else 0.0
        ),
        forklift_utilization=(
            forklift_active_time / (makespan * forklift_count) if makespan else 0.0
        ),
        worker_wait_time_total=sum(worker_waits),
        worker_wait_time_average=average(worker_waits),
        forklift_wait_time_total=sum(forklift_waits),
        forklift_wait_time_average=average(forklift_waits),
        container_wait_time_total=sum(container_waits),
        container_wait_time_average=average(container_waits),
        order_completion_times=completion_times,
    )
