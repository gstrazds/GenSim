import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from typing import Dict, Any
from gensim.robocodegen import RobotScript, EnvironmentExt, PickAndPlaceAction

class PlaceRedBlocksInGreenBowls(RobotScript):
    """pick up red blocks and place them into green bowls."""

    def __init__(self, env: EnvironmentExt, task_spec):
        super().__init__(env, task_name=task_spec['task-name'], instructions=task_spec['task-description'])
        scene_objects = env.detect_objects()  # obtain a list of all the visible objects

        # a list of blocks that are red
        self.red_blocks = list(filter(lambda oid: env.is_object_type(oid, 'block') and env.is_object_color(oid, 'red'), scene_objects))

        # a list of bowls that are green
        self.green_bowls = list(filter(lambda oid: env.is_object_type(oid, 'bowl') and env.is_object_color(oid, 'green'), scene_objects))
        self.current_target_index = -1

    def get_target_id(self):
        if not self.green_bowls:  # if there are no target bowls available
            self.current_target_index = -1
        else:
            # cycle through the available target bowls
            self.current_target_index += 1
            if self.current_target_index >= len(self.green_bowls):
                self.current_target_index = 0
        if self.current_target_index >= 0:
            return self.green_bowls[self.current_target_index]
        return None

    def act(self, obs: str, info: Dict[str, Any]):
        ''' Each time this method is invoked, move one red block into a green bowl.
        In order to distribute the blocks as evenly as possible, we cycle through the available bowls.
        '''

        if not self.red_blocks:  # if there are no more red blocks available, there's nothing to do.
            return None
        block_id = self.red_blocks.pop()  # select one red block, removing it from the list of items that need to be moved
        bowl_id = self.get_target_id()  # choose a destination
        if bowl_id is None:
            return None
        pick_pose = self.env.get_pick_pose(block_id)
        place_pose = self.env.get_place_pose(block_id, bowl_id)  # a pose that's appropriate for placing the first object onto or in the target object

        return PickAndPlaceAction(pick_pose, place_pose, block_id)  # arguments for performing a pick and place action in the environment