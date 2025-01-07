import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from typing import Dict, List, Any
from gensim.robocodegen import RobotScript, EnvironmentExt, PickAndPlaceAction

class StackColorCoordinatedBlocks(RobotScript):
    """Stack blocks of different colors in a specific order on a pallet."""

    def __init__(self, env: EnvironmentExt, task_spec):
        super().__init__(env, task_name=task_spec['task-name'], instructions=task_spec['task-description'])
        self.env_reset(env)

    def env_reset(self, env):
        super().env_reset(env)
        # Define the order of the blocks for each stack
        self.stack_one_order = ['red', 'blue', 'green']
        self.stack_two_order = ['yellow', 'orange', 'purple']
        # Initialize lists to hold the blocks for each stack
        self.stack_one_blocks = []
        self.stack_two_blocks = []
        # Find all blocks and sort them into the two stacks based on color
        for block_id in self.scene_objects:
            if env.is_object_type(block_id, 'block'):
                color = env.get_object_color(block_id)
                if color in self.stack_one_order:
                    self.stack_one_blocks.append((self.stack_one_order.index(color), block_id))
                elif color in self.stack_two_order:
                    self.stack_two_blocks.append((self.stack_two_order.index(color), block_id))
        # Sort the blocks by their order in the stack (red/blue/green or yellow/orange/purple)
        self.stack_one_blocks.sort()
        self.stack_two_blocks.sort()
        # Get the pallet ID to place the stacks on
        self.pallet_id = next(filter(lambda oid: env.is_object_type(oid, 'pallet'), self.scene_objects), None)

    def act(self, obs, info):
        ''' Each time this method is invoked, move one block to the correct position on the pallet to form the stacks.
        '''
        # Check if there are blocks left to stack for stack one
        if self.stack_one_blocks:
            color_index, block_id = self.stack_one_blocks.pop(0)  # Get the next block to place
            place_pose = self.env.get_place_pose(block_id, self.pallet_id, stack_index=0, level=color_index)
        # If stack one is complete, move on to stack two
        elif self.stack_two_blocks:
            color_index, block_id = self.stack_two_blocks.pop(0)  # Get the next block to place
            place_pose = self.env.get_place_pose(block_id, self.pallet_id, stack_index=1, level=color_index)
        else:
            return None  # All blocks have been placed, the task is complete

        pick_pose = self.env.get_pick_pose(block_id)
        return PickAndPlaceAction(pick_pose, place_pose, block_id)  # Perform the pick and place action