import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class MoveBottlesIntoBlueZone(Task):
    """
    The task requires the robot to pick up three bottles of different colors (red, green, and yellow),
    navigate around a small blue block acting as an obstacle, and place each bottle neatly within a
    designated blue zone on the table without knocking over the block.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "move the {color} bottle into the blue zone"
        self.task_completed_desc = "done moving bottles into the blue zone."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the colors for the bottles and the obstacle block.
        bottle_colors = ['red', 'green', 'yellow', 'blue']
        obstacle_color = utils.COLORS['cyan']
        zone_color = utils.COLORS['blue']

        # Define the sizes for the bottles, obstacle block, and the zone.
        bottle_size = (0.05, 0.05, 0.15)  # (x, y, z) dimensions
        obstacle_size = (0.05, 0.05, 0.05)  # Small blue block size
        zone_size = (0.5, 0.5, 0)  # Zone size (x, y, z), z is 0 because it's a flat zone
        zone2_size = (0.12, 0.12, 0)  # Zone size (x, y, z), z is 0 because it's a flat zone

        # Add the blue zone where bottles need to be placed.
        zone_pose = self.get_random_pose(env, zone_size)
        env.add_object('zone/zone_large.urdf', zone_pose, 'fixed', color=zone_color)
        zone2_pose = self.get_random_pose(env, zone2_size)
        # zone2_pose[0][0] = 0.7  #zone_pose[0][0]+0.7
        env.add_object('zone/zone.urdf', zone2_pose, 'fixed', color='red')

        # Add the small blue block obstacle.
        obstacle_pose = self.get_random_pose(env, obstacle_size)
        env.add_object('block/block.urdf', obstacle_pose, 'fixed', color=obstacle_color)

        # Add the three bottles.
        bottles = []
        for i, color in enumerate(bottle_colors):
            # Get a random pose for the bottle that doesn't intersect with the zone or the obstacle.
            bottle_pose = self.get_random_pose(env, bottle_size)
            bottle_pose[0][2] = -0.065  # adjust for mesh coord offest
            bottle_urdf = 'bottle/bottle.urdf'
            replace = {'DIM': bottle_size, 'COLOR': color}
            # IMPORTANT: REPLACE THE TEMPLATE URDF with `fill_template`
            # urdf = self.fill_template(bottle_urdf, replace)
            bottle_id = env.add_object(bottle_urdf, bottle_pose, color=color)
            bottles.append(bottle_id)

            # Define the language goal for each bottle.
            color_name = bottle_colors[i]
            language_goal = self.lang_template.format(color=color_name)

            # Add a goal for each bottle to be in the blue zone.
            self.add_goal(objs=[bottle_id], matches=np.int32([[1]]), targ_poses=[zone_pose], replace=False,
                          rotations=True, metric='zone', params=[(zone_pose, zone_size)], step_max_reward=1 / len(bottle_colors),
                          language_goal=language_goal)

        # The number of language goals matches the number of motion goals.
        # Each bottle has a corresponding language goal that instructs the robot to move it into the blue zone.