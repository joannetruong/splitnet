#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import os
import random
from typing import Dict, List, Optional

import cv2
import habitat
import habitat.datasets.pointnav.pointnav_dataset as mp3d_dataset
import numpy as np
from habitat.config.default import get_config
from habitat.datasets import make_dataset

USE_SPOT = False
USE_GRAY = False
if USE_SPOT:
    if USE_GRAY:
        CFG = "/coc/testnvme/jtruong33/google_nav/habitat-lab/configs/tasks/outdoor_spotnav_hm3d.yaml"
    else:
        CFG = "/coc/testnvme/jtruong33/google_nav/habitat-lab/configs/tasks/pvp_spotnav_hm3d.yaml"
else:
    CFG = "configs/habitat_nav_task_config.yaml"


def make_config(gpu_id, split, data_path, sensors, resolution):
    config = get_config(CFG)
    config.defrost()
    config.TASK.NAME = "Nav-v0"
    config.TASK.MEASUREMENTS = []
    config.DATASET.SPLIT = split
    config.DATASET.DATA_PATH = data_path
    config.HEIGHT = resolution
    config.WIDTH = resolution

    # for sensor in sensors:
    #     config.SIMULATOR[sensor]["HEIGHT"] = resolution
    #     config.SIMULATOR[sensor]["WIDTH"] = resolution
    #     config.SIMULATOR[sensor]["POSITION"] = [0, 1.09, 0]
    #     config.SIMULATOR[sensor]["HFOV"] = 45

    config.TASK.HEIGHT = resolution
    config.TASK.WIDTH = resolution
    config.SIMULATOR.AGENT_0.SENSORS = sensors

    config.ENVIRONMENT.MAX_EPISODE_STEPS = 2**32
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
    return config


class RandomImageGenerator(object):
    def __init__(
        self,
        gpu_id: int,
        unique_dataset_name: str,
        split: str,
        data_path: str,
        images_before_reset: int,
        sensors: Optional[List[str]] = None,
        resolution: Optional[int] = 256,
    ) -> None:
        if sensors is None:
            if USE_SPOT:
                if USE_GRAY:
                    sensors = [
                        "SPOT_LEFT_GRAY_SENSOR",
                        "SPOT_RIGHT_GRAY_SENSOR",
                        "SPOT_LEFT_DEPTH_SENSOR",
                        "SPOT_RIGHT_DEPTH_SENSOR",
                    ]
                else:
                    sensors = [
                        "SPOT_LEFT_RGB_SENSOR",
                        "SPOT_RIGHT_RGB_SENSOR",
                        "SPOT_LEFT_DEPTH_SENSOR",
                        "SPOT_RIGHT_DEPTH_SENSOR",
                    ]
            else:
                sensors = ["RGB_SENSOR", "DEPTH_SENSOR"]
        self.images_before_reset = images_before_reset
        config = make_config(gpu_id, split, data_path, sensors, resolution)
        data_dir = os.path.join(
            "data/scene_episodes/", unique_dataset_name + "_" + split
        )
        self.dataset_name = config.DATASET.TYPE
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, "dataset_one_ep_per_scene_v2.json.gz")
        # Creates a dataset where each episode is a random spawn point in each scene.
        if not os.path.exists(data_path):
            dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
            # Get one episode per scene in dataset
            scene_episodes = {}
            for episode in dataset.episodes:
                if episode.scene_id not in scene_episodes:
                    scene_episodes[episode.scene_id] = episode
            scene_episodes = list(scene_episodes.values())
            print("scene episodes", len(scene_episodes))
            dataset.episodes = scene_episodes
            if not os.path.exists(data_path):
                # Multiproc do check again before write.
                json = dataset.to_json().encode("utf-8")
                with gzip.GzipFile(data_path, "w") as fout:
                    fout.write(json)
        dataset = mp3d_dataset.PointNavDatasetV1()
        print("dataset: ", dataset)
        with gzip.open(data_path, "rt") as f:
            dataset.from_json(f.read())

        config.TASK.SENSORS = ["POINTGOAL_SENSOR"]
        if "SEMANTIC_SENSOR" in sensors:
            config.TASK.SENSOR.append("CLASS_SEGMENTATION_SENSOR")
            config.TASK.CLASS_SEGMENTATION_SENSOR.HEIGHT = config.TASK.HEIGHT
            config.TASK.CLASS_SEGMENTATION_SENSOR.WIDTH = config.TASK.WIDTH

        config.freeze()
        print("DATASET: ", dataset)
        self.env = habitat.Env(config=config, dataset=dataset)
        random.shuffle(self.env.episodes)
        self.num_samples = 0

    def get_sample(self) -> Dict[str, np.ndarray]:
        if self.num_samples % self.images_before_reset == 0:
            self.env.reset()

        rand_location = self.env.sim.sample_navigable_point()
        num_tries = 0
        while rand_location[1] > 1:
            rand_location = self.env.sim.sample_navigable_point()
            num_tries += 1
            if num_tries > 1000:
                self.env.reset()

        rand_angle = np.random.uniform(0, 2 * np.pi)
        rand_rotation = [0, np.sin(rand_angle / 2), 0, np.cos(rand_angle / 2)]
        self.env.sim.set_agent_state(rand_location, rand_rotation)
        obs = self.env.sim._sensor_suite.get_observations(
            self.env.sim.get_sensor_observations()
        )

        class_semantic = None
        if "semantic" in obs:
            # Currently unused
            semantic = obs["semantic"]
            class_semantic = self.env.sim.scene_obj_id_to_semantic_class[
                semantic
            ].astype(np.int32)
        if USE_SPOT:
            if USE_GRAY:
                img = np.concatenate(
                    [
                        # Spot is cross-eyed; right is on the left on the FOV
                        obs["spot_right_gray"][:, :, :],
                        obs["spot_left_gray"][:, :, :],
                    ],
                    axis=1,
                )
                img = cv2.resize(img, (256, 256))
                img = img.reshape([*img.shape[:2], 1])
            else:
                img = np.concatenate(
                    [
                        # Spot is cross-eyed; right is on the left on the FOV
                        obs["spot_right_rgb"][:, :, :3],
                        obs["spot_left_rgb"][:, :, :3],
                    ],
                    axis=1,
                )
                img = cv2.resize(img, (256, 256))
            depth = np.concatenate(
                [
                    # Spot is cross-eyed; right is on the left on the FOV
                    obs["spot_right_depth"],
                    obs["spot_left_depth"],
                ],
                axis=1,
            ).squeeze()
            depth = cv2.resize(depth, (256, 256))
        else:
            img = obs["rgb"][:, :, :3]
            depth = obs["depth"].squeeze()
            print("IMG SHAPE: ", img.shape)
            print("depth SHAPE: ", depth.shape)

        self.num_samples += 1
        return {"rgb": img, "depth": depth, "class_semantic": class_semantic}
