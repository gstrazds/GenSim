import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
import random
from cliport.tasks.task import Task
from cliport.utils import utils

class AlignBottlesOnLine(Task):
    """Align colored blocks on the line of the same color in a specific sequence."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "align the {color} bottle on the {color} line"
        self.task_completed_desc = "done aligning bottles."
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add lines.
        line_size = (0.3, 0.01, 0.01)
        line_urdf = 'line/line-template.urdf'
        line_poses = []
        delta_y_pos = random.randint(0,4) / 100.0
        for i, color in enumerate(self.colors):
            line_pose_rand = self.get_random_pose(env, line_size)
            line_pose_y = line_poses[0][0][1] if line_poses else line_pose_rand[0][1]
 
            line_pose = ((0.30+i*(0.10+delta_y_pos), line_pose_y, line_pose_rand[0][2]), utils.eulerXYZ_to_quatXYZW((0,0,0)))
            env.add_object(line_urdf, line_pose, category='fixed', color=utils.COLORS[color])
            line_poses.append(line_pose)

        # Add blocks.
        block_size = (0.05, 0.05, 0.05)
        block_urdf = 'bottle/bottle.urdf'
        blocks = []
        for color in self.colors:
            block_pose = self.get_random_pose(env, block_size)
            block_pose[0][2] = -0.065

            block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[color])
            blocks.append(block_id)

        # Goal: each block is on the line of the same color.
        for i in range(len(self.colors)):
            language_goal = self.lang_template.format(color=self.colors[i])
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[line_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/len(self.colors),
                          language_goal=language_goal)