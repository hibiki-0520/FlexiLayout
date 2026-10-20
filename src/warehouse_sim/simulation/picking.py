"""SimPy-based warehouse order-picking simulation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
from functools import lru_cache
from heapq import heappop, heappush

import numpy as np
import pandas as pd
import simpy

Coordinate = tuple[int, int]
RouteFunction = Callable[[Coordinate, Coordinate], tuple[Coordinate, ...]]
_GRID_ROUTE_CACHES: OrderedDict[
    tuple[tuple[int, int], bytes],
    RouteFunction,
] = OrderedDict()


class CargoSize(str, Enum):
    """Supported handling classes shown in the Streamlit application."""

    SMALL = "小型 (手持ち可)"
    MEDIUM = "中型 (コンテナ必須)"
    LARGE = "大型 (フォークリフト必須)"


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
    """Aggregate completion KPIs and a time-series agent trace."""

    makespan: float
    worker_utilization: float
    forklift_utilization: float
    worker_wait_time_total: float
    worker_wait_time_average: float
    forklift_wait_time_total: float
    forklift_wait_time_average: float
    container_wait_time_total: float
    container_wait_time_average: float
    forklift_active_time_total: float
    forklift_distance_total: float
    item_count_total: int
    order_completion_times: Mapping[str, float]
    trace: pd.DataFrame


def _validate_coordinate(name: str, coordinate: Coordinate) -> None:
    if (
        not isinstance(coordinate, tuple)
        or len(coordinate) != 2
        or any(not isinstance(value, int) or value < 0 for value in coordinate)
    ):
        raise ValueError(f"{name} must be a pair of non-negative integers")


def _route_path(
    origin: Coordinate,
    destination: Coordinate,
    warehouse_grid: np.ndarray | None,
) -> list[Coordinate]:
    if warehouse_grid is None:
        row, column = origin
        path = [origin]
        row_step = 1 if destination[0] >= row else -1
        while row != destination[0]:
            row += row_step
            path.append((row, column))
        column_step = 1 if destination[1] >= column else -1
        while column != destination[1]:
            column += column_step
            path.append((row, column))
        return path

    return list(_grid_route_function(warehouse_grid)(origin, destination))


def _astar_route(
    origin: Coordinate,
    destination: Coordinate,
    warehouse_grid: np.ndarray,
) -> tuple[Coordinate, ...]:
    rows, columns = warehouse_grid.shape
    for point in (origin, destination):
        if not (0 <= point[0] < rows and 0 <= point[1] < columns):
            raise ValueError(f"Route coordinate {point} is outside warehouse_grid")

    frontier: list[tuple[int, int, Coordinate]] = []
    heappush(
        frontier,
        (
            abs(origin[0] - destination[0]) + abs(origin[1] - destination[1]),
            0,
            origin,
        ),
    )
    came_from: dict[Coordinate, Coordinate | None] = {origin: None}
    cost_so_far = {origin: 0}
    neighbors = ((-1, 0), (0, -1), (0, 1), (1, 0))

    while frontier:
        _, current_cost, current = heappop(frontier)
        if current == destination:
            path = [current]
            while came_from[path[-1]] is not None:
                previous = came_from[path[-1]]
                if previous is None:
                    break
                path.append(previous)
            path.reverse()
            return tuple(path)
        if current_cost != cost_so_far[current]:
            continue

        for row_delta, column_delta in neighbors:
            neighbor = (current[0] + row_delta, current[1] + column_delta)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < columns):
                continue
            if neighbor != destination and warehouse_grid[neighbor] == "S":
                continue
            next_cost = current_cost + 1
            if next_cost >= cost_so_far.get(neighbor, float("inf")):
                continue
            cost_so_far[neighbor] = next_cost
            came_from[neighbor] = current
            heuristic = abs(neighbor[0] - destination[0]) + abs(
                neighbor[1] - destination[1]
            )
            heappush(frontier, (next_cost + heuristic, next_cost, neighbor))

    raise ValueError(f"No aisle route found from {origin} to {destination}")


def _grid_route_function(warehouse_grid: np.ndarray) -> RouteFunction:
    normalized_grid = warehouse_grid.astype("<U1", copy=False)
    grid_data = normalized_grid.tobytes()
    cache_key = (
        normalized_grid.shape,
        hashlib.blake2b(grid_data, digest_size=16).digest(),
    )
    route_function = _GRID_ROUTE_CACHES.get(cache_key)
    if route_function is not None:
        _GRID_ROUTE_CACHES.move_to_end(cache_key)
        return route_function

    route_function = lru_cache(maxsize=8192)(
        lambda origin, destination: _astar_route(
            origin,
            destination,
            normalized_grid,
        )
    )
    _GRID_ROUTE_CACHES[cache_key] = route_function
    if len(_GRID_ROUTE_CACHES) > 4:
        _GRID_ROUTE_CACHES.popitem(last=False)
    return route_function


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
    cargo_weight_kg: float = 0.0,
    cargo_size_class: CargoSize | str = CargoSize.MEDIUM,
    max_transport_units: int = 5,
    trace_interval: float = 1.0,
    record_trace: bool = True,
    warehouse_grid: np.ndarray | None = None,
) -> SimulationResult:
    """Simulate orders and optionally record agent positions at each time step.

    Trace coordinates use ``x=column`` and ``y=row``. If ``warehouse_grid``
    is provided, agents follow an A* route through floor, aisle, and I/O cells;
    shelf cells are only entered when they are the route destination. Agent
    states are ``moving``, ``picking``, or ``waiting``. Movement is sampled
    and interpolated along each grid edge at ``trace_interval`` intervals.
    """
    if worker_count <= 0 or container_count <= 0:
        raise ValueError("worker and container counts must be positive")
    if (
        not isinstance(forklift_count, int)
        or isinstance(forklift_count, bool)
        or forklift_count < 0
    ):
        raise ValueError("forklift_count must be a non-negative integer")
    for name, value in (
        ("worker_speed", worker_speed),
        ("forklift_speed", forklift_speed),
        ("pick_time_per_unit", pick_time_per_unit),
    ):
        if not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError(f"{name} must be finite and positive")
    if (
        not isinstance(cargo_weight_kg, (int, float))
        or not 0 <= cargo_weight_kg < float("inf")
    ):
        raise ValueError("cargo_weight_kg must be finite and non-negative")
    if (
        not isinstance(max_transport_units, int)
        or isinstance(max_transport_units, bool)
        or max_transport_units <= 0
    ):
        raise ValueError("max_transport_units must be a positive integer")
    if not isinstance(trace_interval, (int, float)) or not (
        0 < trace_interval < float("inf")
    ):
        raise ValueError("trace_interval must be finite and positive")
    try:
        cargo_size = CargoSize(cargo_size_class)
    except ValueError as exc:
        raise ValueError(f"Unsupported cargo_size_class: {cargo_size_class!r}") from exc
    if cargo_size is not CargoSize.SMALL and forklift_count <= 0:
        raise ValueError(f"{cargo_size.value} requires at least one forklift")

    _validate_coordinate("io_point", io_point)
    if warehouse_grid is not None:
        if (
            not isinstance(warehouse_grid, np.ndarray)
            or warehouse_grid.ndim != 2
            or warehouse_grid.size == 0
        ):
            raise ValueError("warehouse_grid must be a non-empty 2D NumPy array")
        unknown_cells = set(np.unique(warehouse_grid)).difference(
            {".", "A", "I", "S"}
        )
        if unknown_cells:
            raise ValueError(
                f"warehouse_grid contains unknown cells: {sorted(unknown_cells)}"
            )
        for name, point in (("io_point", io_point), *shelf_positions.items()):
            if not (
                0 <= point[0] < warehouse_grid.shape[0]
                and 0 <= point[1] < warehouse_grid.shape[1]
            ):
                raise ValueError(f"{name} coordinate {point} is outside warehouse_grid")
        if warehouse_grid[io_point] != "I":
            raise ValueError("io_point must refer to an I cell in warehouse_grid")
        for product_id, point in shelf_positions.items():
            if warehouse_grid[point] != "S":
                raise ValueError(
                    f"shelf_positions[{product_id!r}] must refer to an S cell"
                )
    for product_id, position in shelf_positions.items():
        if not product_id:
            raise ValueError("Product IDs in shelf_positions must not be empty")
        _validate_coordinate(f"shelf_positions[{product_id!r}]", position)
    route_function: RouteFunction = (
        _grid_route_function(warehouse_grid)
        if warehouse_grid is not None
        else lambda origin, destination: tuple(
            _route_path(origin, destination, None)
        )
    )

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

    worker_size_factor = {
        CargoSize.SMALL: 1.0,
        CargoSize.MEDIUM: 1.0,
        CargoSize.LARGE: 0.9,
    }[cargo_size]
    forklift_size_factor = {
        CargoSize.SMALL: 1.0,
        CargoSize.MEDIUM: 1.0,
        CargoSize.LARGE: 0.8,
    }[cargo_size]
    pick_size_factor = {
        CargoSize.SMALL: 1.0,
        CargoSize.MEDIUM: 1.0,
        CargoSize.LARGE: 1.25,
    }[cargo_size]
    effective_worker_speed = worker_speed * worker_size_factor / (
        1 + cargo_weight_kg / 50
    )
    effective_forklift_speed = forklift_speed * forklift_size_factor / (
        1 + cargo_weight_kg / 200
    )
    effective_pick_time = pick_time_per_unit * pick_size_factor * (
        1 + cargo_weight_kg / 100
    )

    env = simpy.Environment()
    workers = simpy.Resource(env, capacity=worker_count)
    forklifts = simpy.Resource(env, capacity=max(forklift_count, 1))
    containers = simpy.Resource(env, capacity=container_count)
    worker_active_time = 0.0
    forklift_active_time = 0.0
    forklift_distance_total = 0.0
    worker_waits: list[float] = []
    forklift_waits: list[float] = []
    container_waits: list[float] = []
    completion_times: dict[str, float] = {}

    agents: dict[str, dict[str, float | str]] = {}
    for index in range(worker_count):
        agents[f"worker-{index + 1}"] = {
            "agent_type": "worker",
            "x": float(io_point[1]),
            "y": float(io_point[0]),
            "state": "waiting",
        }
    for index in range(forklift_count):
        agents[f"forklift-{index + 1}"] = {
            "agent_type": "forklift",
            "x": float(io_point[1]),
            "y": float(io_point[0]),
            "state": "waiting",
        }

    worker_assigned = [False] * worker_count
    forklift_assigned = [False] * forklift_count
    cumulative_wait = {"worker": 0.0, "forklift": 0.0, "container": 0.0}
    pending_waits: dict[int, tuple[str, float]] = {}
    next_wait_id = 0
    trace_snapshots: dict[float, list[dict[str, float | str | int]]] = {}

    def record_snapshot() -> None:
        if not record_trace:
            return
        time = round(float(env.now), 9)
        wait_time_total = sum(cumulative_wait.values()) + sum(
            max(0.0, float(env.now) - started_at)
            for _, started_at in pending_waits.values()
        )
        trace_snapshots[time] = [
            {
                "time": time,
                "agent_id": agent_id,
                "agent_type": str(agent["agent_type"]),
                "x": float(agent["x"]),
                "y": float(agent["y"]),
                "state": str(agent["state"]),
                "wait_time_total": wait_time_total,
                "queue_count": len(pending_waits),
                "completed_order_count": len(completion_times),
            }
            for agent_id, agent in agents.items()
        ]

    def set_state(agent_id: str, state: str) -> None:
        agents[agent_id]["state"] = state
        record_snapshot()

    def sample_trace():
        while True:
            yield env.timeout(trace_interval)
            record_snapshot()

    def move_agent(agent_id: str, destination: Coordinate, speed: float):
        agent = agents[agent_id]
        origin = (int(round(float(agent["y"]))), int(round(float(agent["x"]))))
        route = route_function(origin, destination)
        route_destination = route[-1]
        if len(route) == 1:
            agent["x"] = float(route_destination[1])
            agent["y"] = float(route_destination[0])
            record_snapshot()
            return route_destination

        set_state(agent_id, "moving")
        for start, end in zip(route, route[1:]):
            duration = 1.0 / speed
            elapsed = 0.0
            while elapsed < duration:
                step_duration = min(float(trace_interval), duration - elapsed)
                yield env.timeout(step_duration)
                elapsed += step_duration
                progress = min(elapsed / duration, 1.0)
                agent["y"] = start[0] + (end[0] - start[0]) * progress
                agent["x"] = start[1] + (end[1] - start[1]) * progress
                record_snapshot()
        agent["x"] = float(route_destination[1])
        agent["y"] = float(route_destination[0])
        record_snapshot()
        return route_destination

    def perform_picking(agent_id: str, duration: float):
        nonlocal worker_active_time
        worker_active_time += duration
        remaining = duration
        set_state(agent_id, "picking")
        while remaining > 0:
            step_duration = min(float(trace_interval), remaining)
            yield env.timeout(step_duration)
            remaining -= step_duration
            record_snapshot()

    def process_batch(
        batch: list[tuple[str, int]],
        worker_id: str,
        forklift_id: str | None,
    ):
        nonlocal worker_active_time, forklift_active_time, forklift_distance_total

        current_position = (
            int(round(float(agents[worker_id]["y"]))),
            int(round(float(agents[worker_id]["x"]))),
        )
        batch_units = sum(quantity for _, quantity in batch)
        for product_id, quantity in batch:
            shelf_position = shelf_positions[product_id]
            distance = len(route_function(current_position, shelf_position)) - 1
            worker_active_time += distance / effective_worker_speed
            current_position = yield from move_agent(
                worker_id,
                shelf_position,
                effective_worker_speed,
            )
            yield from perform_picking(
                worker_id, quantity * effective_pick_time
            )

        if cargo_size is CargoSize.SMALL:
            distance = len(route_function(current_position, io_point)) - 1
            worker_active_time += distance / effective_worker_speed
            yield from move_agent(worker_id, io_point, effective_worker_speed)
            return

        if forklift_id is None:
            raise RuntimeError("A forklift must be assigned for this cargo size")
        set_state(worker_id, "waiting")
        forklift_position = (
            int(round(float(agents[forklift_id]["y"]))),
            int(round(float(agents[forklift_id]["x"]))),
        )
        unloaded_distance = len(
            route_function(forklift_position, current_position)
        ) - 1
        loaded_distance = len(route_function(current_position, io_point)) - 1
        forklift_distance_total += unloaded_distance + loaded_distance
        transport_time = (
            unloaded_distance + loaded_distance
        ) / effective_forklift_speed
        loading_time = batch_units * pick_time_per_unit * cargo_weight_kg / 100
        forklift_active_time += transport_time + loading_time
        yield from move_agent(
            forklift_id, current_position, effective_forklift_speed
        )
        set_state(forklift_id, "picking")
        remaining_loading = loading_time
        while remaining_loading > 0:
            step_duration = min(float(trace_interval), remaining_loading)
            yield env.timeout(step_duration)
            remaining_loading -= step_duration
            record_snapshot()
        yield from move_agent(forklift_id, io_point, effective_forklift_speed)
        set_state(forklift_id, "waiting")

    def process_order(order: PickingOrder):
        nonlocal next_wait_id, worker_active_time
        requested_at = env.now
        next_wait_id += 1
        wait_token = next_wait_id
        pending_waits[wait_token] = ("worker", requested_at)
        with workers.request() as worker_request:
            yield worker_request
            pending_waits.pop(wait_token)
            cumulative_wait["worker"] += env.now - requested_at
            worker_waits.append(env.now - requested_at)
            worker_index = next(
                index for index, assigned in enumerate(worker_assigned) if not assigned
            )
            worker_assigned[worker_index] = True
            worker_id = f"worker-{worker_index + 1}"

            try:
                chunks = [
                    (pick.product_id, quantity)
                    for pick in order.picks
                    for quantity in (
                        [max_transport_units]
                        * (pick.quantity // max_transport_units)
                        + (
                            [pick.quantity % max_transport_units]
                            if pick.quantity % max_transport_units
                            else []
                        )
                    )
                ]
                batches: list[list[tuple[str, int]]] = []
                for product_id, quantity in chunks:
                    if (
                        not batches
                        or sum(units for _, units in batches[-1]) + quantity
                        > max_transport_units
                    ):
                        batches.append([])
                    batches[-1].append((product_id, quantity))

                for batch in batches:
                    forklift_id: str | None = None
                    if cargo_size is CargoSize.MEDIUM:
                        requested_at = env.now
                        set_state(worker_id, "waiting")
                        next_wait_id += 1
                        wait_token = next_wait_id
                        pending_waits[wait_token] = ("container", requested_at)
                        with containers.request() as container_request:
                            yield container_request
                            pending_waits.pop(wait_token)
                            wait = env.now - requested_at
                            cumulative_wait["container"] += wait
                            container_waits.append(wait)
                            yield from run_batch_with_forklift(batch, worker_id)
                    elif cargo_size is CargoSize.LARGE:
                        yield from run_batch_with_forklift(batch, worker_id)
                    else:
                        yield from process_batch(batch, worker_id, forklift_id)

                if cargo_size is not CargoSize.SMALL:
                    worker_position = (
                        int(round(float(agents[worker_id]["y"]))),
                        int(round(float(agents[worker_id]["x"]))),
                    )
                    return_distance = len(
                        route_function(worker_position, io_point)
                    ) - 1
                    worker_active_time += return_distance / effective_worker_speed
                    yield from move_agent(
                        worker_id,
                        io_point,
                        effective_worker_speed,
                    )
                completion_times[order.order_id] = float(env.now)
                record_snapshot()
            finally:
                agents[worker_id]["x"] = float(io_point[1])
                agents[worker_id]["y"] = float(io_point[0])
                set_state(worker_id, "waiting")
                worker_assigned[worker_index] = False

    def run_batch_with_forklift(
        batch: list[tuple[str, int]], worker_id: str
    ):
        nonlocal next_wait_id, worker_active_time
        requested_at = env.now
        set_state(worker_id, "waiting")
        next_wait_id += 1
        wait_token = next_wait_id
        pending_waits[wait_token] = ("forklift", requested_at)
        with forklifts.request() as forklift_request:
            yield forklift_request
            pending_waits.pop(wait_token)
            wait = env.now - requested_at
            cumulative_wait["forklift"] += wait
            forklift_waits.append(wait)
            forklift_index = next(
                index
                for index, assigned in enumerate(forklift_assigned)
                if not assigned
            )
            forklift_assigned[forklift_index] = True
            forklift_id = f"forklift-{forklift_index + 1}"
            try:
                yield from process_batch(batch, worker_id, forklift_id)
            finally:
                forklift_assigned[forklift_index] = False
                set_state(forklift_id, "waiting")
        worker_position = (
            int(float(agents[worker_id]["y"])),
            int(float(agents[worker_id]["x"])),
        )
        return_distance = abs(worker_position[0] - io_point[0]) + abs(
            worker_position[1] - io_point[1]
        )
        worker_active_time += return_distance / effective_worker_speed
        yield from move_agent(worker_id, io_point, effective_worker_speed)

    record_snapshot()
    processes = [env.process(process_order(order)) for order in orders]
    if processes:
        if record_trace:
            env.process(sample_trace())
        env.run(until=env.all_of(processes))
    record_snapshot()
    makespan = float(env.now)
    item_count_total = sum(
        pick.quantity for order in orders for pick in order.picks
    )

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    trace_rows = [
        row
        for time in sorted(trace_snapshots)
        for row in trace_snapshots[time]
    ]
    trace = pd.DataFrame(
        trace_rows,
        columns=[
            "time",
            "agent_id",
            "agent_type",
            "x",
            "y",
            "state",
            "wait_time_total",
            "queue_count",
            "completed_order_count",
        ],
    )
    return SimulationResult(
        makespan=makespan,
        worker_utilization=(
            worker_active_time / (makespan * worker_count) if makespan else 0.0
        ),
        forklift_utilization=(
            forklift_active_time / (makespan * forklift_count)
            if makespan and forklift_count
            else 0.0
        ),
        worker_wait_time_total=sum(worker_waits),
        worker_wait_time_average=average(worker_waits),
        forklift_wait_time_total=sum(forklift_waits),
        forklift_wait_time_average=average(forklift_waits),
        container_wait_time_total=sum(container_waits),
        container_wait_time_average=average(container_waits),
        forklift_active_time_total=forklift_active_time,
        forklift_distance_total=forklift_distance_total,
        item_count_total=item_count_total,
        order_completion_times=completion_times,
        trace=trace,
    )
