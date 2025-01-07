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

class PutBluesAroundRed(RobotScript):
    """Place the blue blocks around the red block."""

    def __init__(self, env: EnvironmentExt, task_spec):
        super().__init__(env, task_name=task_spec['task-name'], instructions=task_spec['task-description'])
        self.env_reset(env)

    def env_reset(self, env):
        super().env_reset(env)
        # a list of blocks that are blue
        self.blue_blocks = list(filter(lambda oid: env.is_object_type(oid, 'block') and env.is_object_color(oid, 'blue'), self.scene_objects))

        # the red block
        self.red_block = next(filter(lambda oid: env.is_object_type(oid, 'block') and env.is_object_color(oid, 'red'), self.scene_objects), None)
        self.placement_index = 0

    def act(self, obs, info):
        ''' Each time this method is invoked, move one blue block around the red block.
        The blue blocks are placed in a circular pattern around the red block.
        '''

        if not self.blue_blocks:  # if there are no more blue blocks available, there's nothing to do.
            print("NO BLUE BLOCKS")
            return None
        if not self.red_block:  # if there is no red block, we cannot complete the task.
            print("NO RED BLOCKS")
            return None

        block_id = self.blue_blocks.pop()  # select one blue block, removing it from the list of items that need to be moved

        # Calculate the placement position around the red block
        red_block_pose = self.env.get_object_pose(self.red_block)
        angle = 2 * np.pi * (self.placement_index / len(self.blue_blocks))  # distribute blocks evenly in a circle
        offset = np.array([np.cos(angle), np.sin(angle), 0]) * self.env.get_object_size(self.red_block)
        #place_pose = red_block_pose + offset
        place_pose = (red_block_pose[0] + offset, red_block_pose[1])

        self.placement_index += 1  # increment the placement index for the next block

        pick_pose = self.env.get_pick_pose(block_id)
        print(f"pick_pose={pick_pose} place_pose={place_pose}")

        return PickAndPlaceAction(pick_pose, place_pose, block_id)  # arguments for performing a pick and place action in the environment