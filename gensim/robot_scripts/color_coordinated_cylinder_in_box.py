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

class ColorCoordinatedCylinderInBox(RobotScript):
    """Place colored cylinders into the matching colored boxes in a specific sequence."""

    def __init__(self, env: EnvironmentExt, task_spec):
        super().__init__(env, task_name=task_spec['task-name'], instructions=task_spec['task-description'])
        self.env_reset(env)

    def env_reset(self, env):
        super().env_reset(env)
        # Initialize the sequence in which cylinders will be placed
        self.sequence = ['red', 'blue', 'green', 'yellow']
        # Find all cylinders and boxes in the environment
        self.cylinders = {color: None for color in self.sequence}
        self.boxes = {color: None for color in self.sequence}
        for oid in self.scene_objects:
            for color in self.sequence:
                if env.is_object_type(oid, 'cylinder') and env.is_object_color(oid, color):
                    self.cylinders[color] = oid
                elif env.is_object_type(oid, 'box') and env.is_object_color(oid, color):
                    self.boxes[color] = oid

    def act(self, obs, info):
        ''' Each time this method is invoked, move the next cylinder in the sequence into the matching colored box.
        '''
        # Check if the sequence is empty, if so, the task is complete
        if not self.sequence:
            return None

        # Get the next color in the sequence
        color = self.sequence.pop(0)
        cylinder_id = self.cylinders[color]
        box_id = self.boxes[color]

        # If either the cylinder or the box is missing, the task cannot be completed
        if cylinder_id is None or box_id is None:
            return None

        # Get the pick and place poses for the cylinder and box
        pick_pose = self.env.get_pick_pose(cylinder_id)
        place_pose = self.env.get_place_pose(cylinder_id, box_id)

        # Perform the pick and place action
        return PickAndPlaceAction(pick_pose, place_pose, cylinder_id)