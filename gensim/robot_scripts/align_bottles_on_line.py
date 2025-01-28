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

class AlignBottlesOnLine(RobotScript):
    """Align bottles on lines of matching colors in a specific sequence."""

    def __init__(self, env: EnvironmentExt, task_spec):
        super().__init__(env, task_name=task_spec['task-name'], instructions=task_spec['task-description'])
        self.env_reset(env)

    def env_reset(self, env):
        super().env_reset(env)
        # Create a dictionary to hold the bottles and lines of each color
        self.bottles = {}
        self.lines = {}
        # Define the sequence in which the bottles should be aligned
        self.sequence = ['red', 'blue', 'green', 'yellow']
        # Populate the dictionary with the bottles and lines
        for color in self.sequence:
            self.bottles[color] = list(filter(lambda oid: env.is_object_type(oid, 'bottle') and env.is_object_color(oid, color), self.scene_objects))
            self.lines[color] = list(filter(lambda oid: env.is_object_type(oid, 'line') and env.is_object_color(oid, color), self.scene_objects))
        # Initialize the current color index to start with the first color in the sequence
        self.current_color_index = 0

    def get_next_bottle_and_line(self):
        # Get the current color based on the sequence
        current_color = self.sequence[self.current_color_index]
        # If there are no bottles or lines of the current color, move to the next color
        if not self.bottles[current_color] or not self.lines[current_color]:
            self.current_color_index += 1
            return None, None
        # Get the bottle and line of the current color
        bottle_id = self.bottles[current_color].pop()
        line_id = self.lines[current_color][0]  # Assuming there's only one line per color
        return bottle_id, line_id

    def act(self, obs: str, info: Dict[str, Any]):
        ''' Each time this method is invoked, align one bottle on its matching color line.
        The bottles are aligned in the sequence defined by the 'sequence' attribute.
        '''

        # If we have gone through all colors, there's nothing to do
        if self.current_color_index >= len(self.sequence):
            return None

        # Get the next bottle and line to align
        bottle_id, line_id = self.get_next_bottle_and_line()
        # If there's no bottle or line, it means we need to move to the next color
        if bottle_id is None or line_id is None:
            return self.act(obs, info)  # Recursively call act to handle the next color

        # Get the pick and place poses for the bottle and line
        pick_pose = self.env.get_pick_pose(bottle_id)
        place_pose = self.env.get_place_pose(bottle_id, line_id)

        # Return the action to pick up the bottle and place it on the line
        return PickAndPlaceAction(pick_pose, place_pose, bottle_id)