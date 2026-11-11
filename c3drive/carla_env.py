"""Synchronous CARLA recording and policy rollouts."""

import json
import math
import os
import queue
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from networkx import NetworkXNoPath, NodeNotFound

from .data import INPUTS, SPLITS, write_episode
from .model import HISTORY, HORIZON, IMAGE_HEIGHT, IMAGE_WIDTH, LIDAR_SIZE, POSITION_SCALE, SPEED_SCALE

DT = 0.1
LABEL_STRIDE = 4


def local_xy(xy, origin, yaw):
    angle = math.radians(yaw)
    c, s = math.cos(angle), math.sin(angle)
    return (np.asarray(xy) - origin) @ np.array([[c, -s], [s, c]])


def location_xy(location):
    return np.array([location.x, location.y], dtype=np.float32)


def sensor_frame(inbox, tick):
    while True:
        packet = inbox.get(timeout=12)
        if packet.frame == tick:
            return packet
        if packet.frame > tick:
            raise RuntimeError(f"Missing sensor frame {tick}, received {packet.frame}")


class Perturbation:
    def __init__(self, domain, seed):
        self.domain = domain
        self.rng = np.random.default_rng(seed)
        self.last_image = None

    def image(self, image, step):
        if self.domain == "camera-drop" and step % 40 >= 20 and self.last_image is not None:
            return self.last_image
        self.last_image = image
        if self.domain == "occlusion":
            image = image.copy()
            image[:, IMAGE_HEIGHT // 4:3 * IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4:3 * IMAGE_WIDTH // 4] = 0
        return image

    def state(self, position, speed):
        if self.domain == "gps":
            position = position + self.rng.normal(0, 3, 2)
        if self.domain == "speed":
            speed = max(0, speed * self.rng.normal(0.65, 0.15))
        return position, speed


class Route:
    def __init__(self, plan):
        self.points = np.stack([location_xy(waypoint.transform.location) for waypoint, _ in plan])
        self.commands = [option.name for _, option in plan]
        self.distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]
        self.index = 0
        self.navigation_indices = {"clean": 0, "observed": 0}

    def advance(self, position):
        left, right = max(0, self.index - 4), min(len(self.points), self.index + 20)
        nearest = left + int(np.argmin(np.linalg.norm(self.points[left:right] - position, axis=1)))
        self.index = max(nearest, self.index)
        return float(np.linalg.norm(self.points[nearest] - position))

    @property
    def completion(self):
        return float(self.distances[self.index] / self.distances[-1])

    def state(self, position, yaw, speed, source):
        # Keep clean and observed navigation tracks independent of scoring progress.
        index = self.navigation_indices[source]
        first = max(0, index - 10)
        near = self.points[first:index + 11]
        index = first + int(np.argmin(np.linalg.norm(near - position, axis=1)))
        self.navigation_indices[source] = index
        indices = np.searchsorted(self.distances, self.distances[index] + np.array([8, 20]))
        targets = local_xy(self.points[np.minimum(indices, len(self.points) - 1)], position, yaw)
        command = self.commands[min(indices[0], len(self.points) - 1)]
        return np.asarray([speed / SPEED_SCALE, *(targets.flatten() / 20),
                           command == "LEFT", command == "RIGHT", command == "STRAIGHT"], np.float32)


class CarlaEnv:
    def __init__(self, host, port, town, carla_root):
        if carla_root:
            sys.path.insert(0, str(Path(carla_root) / "PythonAPI" / "carla"))
        import carla
        from agents.navigation.basic_agent import BasicAgent
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        self.carla = carla
        self.expert_class = BasicAgent
        self.client = carla.Client(host, port)
        self.client.set_timeout(40)
        if self.client.get_client_version() != self.client.get_server_version():
            raise RuntimeError("Install the CARLA Python wheel that matches the server.")
        self.world = self.client.get_world()
        if self.world.get_map().name.rsplit("/", 1)[-1] != town:
            self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.town = town
        self.planner = GlobalRoutePlanner(self.map, 2.0)
        self.spawns = self.map.get_spawn_points()
        self.traffic = self.client.get_trafficmanager(port + 1)

    def make_routes(self, count, seed):
        rng = np.random.default_rng(seed)
        partitions = np.array_split(rng.permutation(len(self.spawns)), len(SPLITS))
        routes = {}
        for split, pool in zip(SPLITS, partitions):
            candidates = [(int(start), int(end)) for start in pool for end in pool if start != end]
            rng.shuffle(candidates)
            routes[split] = []
            for start, end in candidates:
                try:
                    plan = self.planner.trace_route(self.spawns[start].location, self.spawns[end].location)
                except (LookupError, NetworkXNoPath, NodeNotFound, ValueError):
                    continue
                if len(plan) < 2 or not 70 <= Route(plan).distances[-1] <= 300:
                    continue
                routes[split].append({"start": start, "end": end, "seed": int(rng.integers(1, 2**31 - 1))})
                if len(routes[split]) == count:
                    break
            if len(routes[split]) != count:
                raise RuntimeError(f"Could only find {len(routes[split])} routes for {split}. Choose fewer routes or another town.")
        return {"town": self.town, "routes": routes}

    def add_sensor(self, actor_list, parent, sensor_id, pose, attributes, callback):
        blueprint = self.world.get_blueprint_library().find(sensor_id)
        for key, value in attributes.items():
            blueprint.set_attribute(key, str(value))
        sensor = self.world.spawn_actor(blueprint, pose, attach_to=parent)
        actor_list.append(sensor)
        sensor.listen(callback)

    def run(self, spec, domain, ticks, traffic_count=0, policy=None, record_path=None):
        if max(spec["start"], spec["end"]) >= len(self.spawns):
            raise ValueError(f"Route endpoints are outside the spawn list for {self.town}: {spec}")
        carla = self.carla
        world = self.world
        old_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = False
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        actors = []
        sensor_actors = []
        collisions = []
        invasions = []
        frames, poses, speeds = [], [], []
        try:
            world.apply_settings(settings)
            self.traffic.set_synchronous_mode(True)
            self.traffic.set_random_device_seed(spec["seed"])
            world.reset_all_traffic_lights()
            start, end = self.spawns[spec["start"]], self.spawns[spec["end"]]
            library = world.get_blueprint_library()
            ego = world.spawn_actor(library.find("vehicle.tesla.model3"), start)
            actors.append(ego)
            rng = np.random.default_rng(spec["seed"])
            for index in rng.permutation(len(self.spawns)):
                if len(actors) > traffic_count:
                    break
                if index == spec["start"]:
                    continue
                car = world.try_spawn_actor(library.find("vehicle.tesla.model3"), self.spawns[index])
                if car:
                    actors.append(car)
                    car.set_autopilot(True, self.traffic.get_port())
            plan = self.planner.trace_route(start.location, end.location)
            route = Route(plan)
            expert = self.expert_class(ego, target_speed=25, map_inst=self.map, grp_inst=self.planner)
            expert.set_global_plan(plan)
            self.add_sensor(sensor_actors, ego, "sensor.other.collision", carla.Transform(), {},
                            lambda event: collisions.append(event.frame))
            self.add_sensor(sensor_actors, ego, "sensor.other.lane_invasion", carla.Transform(), {},
                            lambda event: invasions.append(event.frame)
                            if any(m.type not in (carla.LaneMarkingType.Broken, carla.LaneMarkingType.NONE)
                                   for m in event.crossed_lane_markings) else None)
            camera_queues = [queue.Queue() for _ in range(3)]
            for yaw, inbox in zip((-55, 0, 55), camera_queues):
                self.add_sensor(sensor_actors, ego, "sensor.camera.rgb",
                                carla.Transform(carla.Location(x=1.4, z=1.6), carla.Rotation(yaw=yaw)),
                                {"image_size_x": IMAGE_WIDTH, "image_size_y": IMAGE_HEIGHT, "fov": 90}, inbox.put)
            lidar_queue = queue.Queue()
            self.add_sensor(sensor_actors, ego, "sensor.lidar.ray_cast", carla.Transform(carla.Location(z=2.1)),
                            {"range": 32, "rotation_frequency": 10, "channels": 16,
                             "points_per_second": 32000}, lidar_queue.put)
            perturb = Perturbation(domain, spec["seed"])
            history = deque(maxlen=HISTORY)
            delayed = deque()
            route_error = 0.0
            tick_ids = []
            infractions = []
            for step in range(ticks):
                frame_id = world.tick(15)
                tick_ids.append(frame_id)
                images = []
                for inbox in camera_queues:
                    packet = sensor_frame(inbox, frame_id)
                    bgra = np.frombuffer(packet.raw_data, np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 4)
                    images.append(bgra[:, :, 2::-1].transpose(2, 0, 1))
                image = np.concatenate(images)
                lidar = lidar_bev(sensor_frame(lidar_queue, frame_id))
                pose = ego.get_transform()
                position, yaw = location_xy(pose.location), pose.rotation.yaw
                speed = ego.get_velocity().length()
                route_error = route.advance(position)
                clean_state = route.state(position, yaw, speed, "clean")
                noisy_position, noisy_speed = perturb.state(position, speed)
                observation = dict(image=perturb.image(image, step), lidar=lidar,
                                   state=route.state(noisy_position, yaw, noisy_speed, "observed"),
                                   clean_image=image, clean_state=clean_state)
                history.append(observation)
                if record_path is not None:
                    frames.append(observation)
                    poses.append((position, yaw))
                    speeds.append(speed)
                if collisions or route_error > 12 or (route.completion >= 0.99 and np.linalg.norm(position - route.points[-1]) < 4):
                    break
                if policy is None:
                    control = expert.run_step()
                else:
                    prediction = policy(history)
                    control = self.control(prediction, noisy_speed)
                if domain == "latency":
                    delayed.append(control)
                    control = delayed.popleft() if len(delayed) > 3 else carla.VehicleControl(brake=1.0)
                ego.apply_control(control)
            # Let event callbacks for the final observation finish before scoring.
            world.tick(15)
            if collisions:
                infractions.append("collision")
            if invasions:
                infractions.append("solid_lane_crossing")
            if route_error > 12:
                infractions.append("route_deviation")
            completion = route.completion
            result = dict(route=spec, domain=domain, town=self.town, requested_ticks=ticks,
                          requested_traffic=traffic_count, completion=completion, collisions=len(collisions),
                          lane_crossings=len(invasions), ticks=len(tick_ids), infractions=infractions,
                          success=completion >= 0.99 and not collisions and route_error <= 12,
                          score=completion * (0.5 ** len(collisions)) * (0.95 ** len(invasions)))
            if record_path is not None:
                self.record(record_path, frames, poses, speeds, tick_ids, collisions, invasions, result)
            return result
        finally:
            for sensor in sensor_actors:
                sensor.stop()
            if sensor_actors or actors:
                self.client.apply_batch([carla.command.DestroyActor(actor) for actor in sensor_actors + actors])
            self.traffic.set_synchronous_mode(False)
            world.apply_settings(old_settings)

    def control(self, prediction, speed):
        path = prediction[:-1].reshape(HORIZON, 2) * POSITION_SCALE
        x, y = path[1]
        steering = np.clip(1.4 * math.atan2(y, max(0.7, x)), -1, 1)
        target_speed = np.clip(prediction[-1] * SPEED_SCALE, 0, 12)
        error = target_speed - speed
        return self.carla.VehicleControl(steer=float(steering), throttle=float(np.clip(0.25 * error, 0, 0.8)),
                                         brake=float(np.clip(-0.4 * error, 0, 1)))

    def record(self, path, frames, poses, speeds, tick_ids, collisions, invasions, result):
        horizon = HORIZON * LABEL_STRIDE
        count = len(poses) - horizon
        if count <= 0:
            return
        positions = np.stack([pose[0] for pose in poses])
        targets, severity, critical = [], [], []
        for step in range(count):
            future = np.arange(step + LABEL_STRIDE, step + horizon + 1, LABEL_STRIDE)
            local = local_xy(positions[future], *poses[step]) / POSITION_SCALE
            targets.append([*local.flatten(), np.mean(np.asarray(speeds)[future]) / SPEED_SCALE])
            first, last = tick_ids[step], tick_ids[step + horizon]
            impacts = sum(first <= event <= last for event in collisions)
            crossings = sum(first <= event <= last for event in invasions)
            failure = impacts > 0 or (last == tick_ids[-1] and "route_deviation" in result["infractions"])
            critical.append(failure)
            severity.append(1 - (0.5 ** impacts) * (0.95 ** crossings))
        write_episode(path, frames, targets, severity, critical, result)


def lidar_bev(packet):
    points = np.frombuffer(packet.raw_data, np.float32).reshape(-1, 4)
    ix = np.floor((points[:, :2] + 24) * LIDAR_SIZE / 48).astype(np.int32)
    valid = ((ix >= 0) & (ix < LIDAR_SIZE)).all(axis=1) & (points[:, 2] > -1.9)
    cells = ix[valid]
    height = (points[valid, 2] > -0.9).astype(np.int32)
    grid = np.zeros((2, LIDAR_SIZE, LIDAR_SIZE), dtype=np.int32)
    np.add.at(grid, (height, cells[:, 0], cells[:, 1]), 1)
    return (np.minimum(grid, 8) * 31).astype(np.uint8)


class Policy:
    def __init__(self, model, detector=None, on_block=None):
        self.model = model
        self.detector = detector
        self.on_block = on_block

    @torch.inference_mode()
    def __call__(self, history):
        frames = [history[0]] * (HISTORY - len(history)) + list(history)
        device = next(self.model.parameters()).device
        inputs = [torch.as_tensor(np.stack([frame[key] for frame in frames]), device=device)[None] for key in INPUTS]
        features = self.model.encode(*inputs)
        slot = -1
        if self.detector:
            vector = features[0].cpu().numpy()
            triggered = self.detector.observe(vector)
            slot = self.detector.slot
            if triggered and self.on_block:
                self.on_block()
        return self.model.plan(features, slot)[0].cpu().numpy()


def connect(args, town):
    return CarlaEnv(args.host, args.port, town, args.carla_root or os.environ.get("CARLA_ROOT"))


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not manifest.get("town") or set(manifest.get("routes", {})) != set(SPLITS):
        raise ValueError("Route manifest must specify a town and all required splits.")
    endpoints = {}
    for split, routes in manifest["routes"].items():
        if not isinstance(routes, list) or not routes:
            raise ValueError(f"Route split {split} is empty or invalid.")
        seen = set()
        for route in routes:
            if any(type(route.get(key)) is not int or route[key] < 0 for key in ("start", "end", "seed")):
                raise ValueError(f"Invalid route in {split}: {route}")
            pair = (route["start"], route["end"])
            if pair in seen or pair[0] == pair[1]:
                raise ValueError(f"Repeated or zero-length route in {split}: {pair}")
            seen.add(pair)
            for endpoint in pair:
                other = endpoints.setdefault(endpoint, split)
                if other != split:
                    raise ValueError(f"Spawn point {endpoint} is shared by {other} and {split}.")
    return manifest
