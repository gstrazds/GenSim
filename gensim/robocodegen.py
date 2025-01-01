"""Generate robot control script for given task."""

import os
import hydra
import numpy as np
import openai
import random
import traceback

import time
from datetime import datetime
from pprint import pprint

from typing import Dict  #, List, Union, Optional

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

from cliport import tasks
# from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
from gensim.memory import Memory
from cliport.utils.utils import COLORS_NAMES, COLORS
from gensim.utils import (
    mkdir_if_missing,
    save_text,
    add_to_txt,
    extract_code,
    extract_dict,
    extract_list,
    set_gpt_model,
    clear_messages,
    format_dict_prompt,
    sample_list_reference,
    generate_feedback,
)

def _get_obj_class_from_urdf(urdf:str) -> str:
    return urdf


def name_for_color(color) -> str:
    if type(color) is str:
        return color
    assert len(color) >= 3, f"Expected {color} to be a tuple or list of length 3 or 4" 
    for color_name in COLORS_NAMES:
        ref_color = COLORS[color_name]
        if ref_color[0] == color[0] and ref_color[1] == color[1] and ref_color[2] == color[2]:
            return color_name
    return "UNKNOWN_COLOR"

def print_pose(pose):
    print("({:0.3f} {:0.3f} {:0.3f}) ({:0.3f} {:0.3f} {:0.3f})".format(pose[0][0], pose[0][1], pose[0][2], pose[1][0], pose[1][1], pose[1][2]), end="")

class EnvironmentExt(Environment):
    def __init__(self, assets_root, task=None, disp=False, shared_memory=False, hz=240, record_cfg=None):
        super().__init__(assets_root, task, disp, shared_memory, hz, record_cfg)
        self.obj_colors = {}
        self.obj_classes = {}

    def add_object(self, urdf, pose, category='rigid', color=None, **kwargs):
        """List of (fixed, rigid, or deformable) objects in env."""
        
        input_color = color  # base class modifies the color arg
        obj_id = super().add_object(urdf, pose, category, color, **kwargs)
    
        obj_class = _get_obj_class_from_urdf(urdf)
        self.obj_classes[obj_id] = obj_class
        print(f"obj_classes[{obj_id}] = {obj_class}")
        # if input_color is not None:
        #     color = input_color
        if color is not None:
            self.obj_colors[obj_id] = color
            print(f"obj_colors[{obj_id}] = {color}")
        return obj_id
    
    def set_color(self, obj_id, color):
        self.obj_colors[obj_id] = color
        print(f"set_color({obj_id}) = {color}")
        return super().set_color(obj_id, color)

    def get_object_class(self, obj_id) -> str:
        return self.obj_classes.get(obj_id, None)

    def get_object_color(self, obj_id) -> str:
        color = self.obj_colors.get(obj_id, None)
        if color is None:
            return color
        return name_for_color(color)            

class RobotScript:
    def __init__(self, env:EnvironmentExt, task_spec:Dict[str,str]):
        self.env = env
        self.task_name = task_spec['task-name']
        self.instruction = task_spec['task-description']


class RoboScriptGenAgent:
    """
    class that gemerates robot control script for simulation environments
    """
    def __init__(self, cfg, memory):
        self.cfg = cfg
        self.model_output_dir = cfg["model_output_dir"]
        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.memory = memory
        self.chat_log = memory.chat_log
        self.use_template = cfg['use_template']

    def api_review(self, task_name):
        """review the task api"""
        if os.path.exists(f"{self.prompt_folder}/cliport_prompt_api_template.txt"):
            add_to_txt(
                self.chat_log, "================= API Preview!", with_print=True)
            api_prompt_text = open(
                f"{self.prompt_folder}/cliport_prompt_api_template.txt").read()
            if 'task' in self.cfg:
                api_prompt_text = api_prompt_text.replace("TASK_NAME_TEMPLATE", task_name)

            res = generate_feedback(
                api_prompt_text, temperature=0, interaction_txt=self.chat_log)
            print(res)

    def _build_template_reference_prompt(self, task_spec):
        """ select which code reference to reference """
        if os.path.exists(f"{self.prompt_folder}/cliport_prompt_code_reference_selection_template.txt"):
            self.chat_log = add_to_txt(self.chat_log, "================= Code Reference!", with_print=True)
            code_reference_question = open(f'{self.prompt_folder}/cliport_prompt_code_reference_selection_template.txt').read()
            code_reference_question = code_reference_question.replace("TASK_NAME_TEMPLATE", task_spec["task-name"])
            code_reference_question = code_reference_question.replace("TASK_CODE_LIST_TEMPLATE", str(list(self.memory.online_code_buffer.keys())))

            code_reference_question = code_reference_question.replace("TASK_STRING_TEMPLATE", str(task_spec))
            res = generate_feedback(code_reference_question, temperature=0., interaction_txt=self.chat_log)
            code_reference_cmd = extract_list(res, prefix='code_reference')
            exec(code_reference_cmd, globals())  # creates variable 'code_reference' which is a list of keys 
            task_code_reference_replace_prompt = ''
            for key in code_reference:
                if key in self.memory.online_code_buffer:
                    task_code_reference_replace_prompt += f'```\n{self.memory.online_code_buffer[key]}\n```\n\n'
                else:
                    print("missing task reference code:", key)

        return task_code_reference_replace_prompt

    def implement_task(self, task_spec):
        """Generate Code for the task"""
        code_prompt_text = open(f"{self.prompt_folder}/cliport_prompt_code_split_template.txt").read()
        code_prompt_text = code_prompt_text.replace("TASK_NAME_TEMPLATE", task_spec["task-name"])
        code_prompt_text = code_prompt_text.replace("TASK_SPEC_TEMPLATE", 
                        f"{{'task-name': '{task_spec['task-name']}', 'task-description': \"{task_spec['task-description']}\"}}")

        if self.use_template or os.path.exists(f"{self.prompt_folder}/cliport_prompt_code_reference_selection_template.txt"):
            task_code_reference_replace_prompt = self._build_template_reference_prompt(task_spec)
            code_prompt_text = code_prompt_text.replace("TASK_CODE_REFERENCE_TEMPLATE", task_code_reference_replace_prompt)

        elif os.path.exists(f"{self.prompt_folder}/cliport_prompt_code_split_template.txt"):
            self.chat_log = add_to_txt(self.chat_log, "================= Code Generation!", with_print=True)
            code_prompt_text = code_prompt_text.replace("TASK_STRING_TEMPLATE", str(task_spec))

        print(code_prompt_text)
        return None
        res = generate_feedback(
                code_prompt_text, temperature=0, interaction_txt=self.chat_log)
        code, task_name = extract_code(res)
        if len(task_name) == 0:
            print("empty task name:", task_name)
            return None, task_name
        if code is not None and len(code):
            print("Save code to:", self.model_output_dir, task_name + "_code_output")
            save_text(self.model_output_dir, task_name + "_code_output", code)

        return code


class GenCodeRunner:
    """ the main class that runs simulation loop """
    def __init__(self, cfg, agent, critic, memory):
        self.cfg = cfg
        self.agent = agent
        self.critic = critic
        self.memory = memory

        self.task_spec = None  #dict('task-name': task_name, ...)
        self.task = None  # instance of cliport.tasks.Task

        self.code_generation_pass = False
        # statistics
        self.syntax_pass_rate = 0
        self.runtime_pass_rate = 0
        self.env_pass_rate = 0
        self.curr_trials = 0

        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.chat_log = memory.chat_log
        self.task_asset_logs = []
        self.n_episodes = 0

    def print_current_stats(self):
        """ print the current statistics of the generation attempts """
        print("=========================================================")
        print(f"{self.cfg['prompt_folder']} Trial {self.curr_trials} SYNTAX_PASS_RATE: {(self.syntax_pass_rate / (self.curr_trials)) * 100:.1f}% RUNTIME_PASS_RATE: {(self.runtime_pass_rate / (self.curr_trials)) * 100:.1f}% ENV_PASS_RATE: {(self.env_pass_rate / (self.curr_trials)) * 100:.1f}%")
        print("=========================================================")

    def setup_env(self, task_name):
        # self.cfg['task'] = task_name
        add_to_txt(self.chat_log, f"================= Retrieve Task Spec ({task_name})...", with_print=True)

        _default_task_spec = {"task-name": "dummy", "assets-used": [], "task_descriptions": ""}
        task_spec = self.memory.online_task_buffer.get(task_name, None) #_default_task_spec )
        if task_spec is None:
            print("TASK SPEC not found!!!", self.memory.online_task_buffer.keys())
        else:
            print(task_spec)
        current_task_name = task_spec['task-name']
        self.task_spec = task_spec
        """ build the new task"""
        env = EnvironmentExt(
                self.cfg['assets_root'],
                disp=self.cfg['disp'],
                shared_memory=self.cfg['shared_memory'],
                hz=480,
                record_cfg=[] #self.cfg['record']
            )

        task = tasks.names[current_task_name]()  # an instance of the Task
        task.mode = self.cfg['mode']
        self.task = task
        # save_data = self.cfg['save_data']

        # Initialize scripted oracle agent and dataset.
        # expert = task.oracle(env)
        data_path = os.path.join(self.cfg['data_dir'], "{}-{}".format(current_task_name, task.mode))
        # dataset = RavensDataset(data_path, self.cfg, n_demos=0, augment=False)
        dataset = None
        return env

    def run_one_episode(self, env, episode, seed, use_oracle=False):
        """ run the new task for one episode """
        add_to_txt(
                self.chat_log, f"================= TRIAL: {self.curr_trials}", with_print=True)
        np.random.seed(seed)
        random.seed(seed)

        print(f"{'Oracle ' if use_oracle else ''}demo: {self.n_episodes + 1}/{self.cfg['max_eps']} | Seed: {seed}")
        expert = self.task.oracle(env)
        env.set_task(self.task)
        obs = env.reset()
        print(f"***** Task: {type(self.task).__name__} mode={self.task.mode} lang_goal='{self.task.get_lang_goal()}'")

        info = env.info
        reward = 0
        episode_reward = 0

        # Rollout expert policy
        for _ in range(self.task.max_steps):
            act = expert.act(obs, info)
            episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            episode_reward += reward
            print(f'Episode Reward (Cummulative): {episode_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
        episode.append((obs, None, reward, info))  # episode is a list of (obs, act, reward, info) tuples
                                                   # act has 2 fields: act['pose0'] and act['pose1']
        return episode_reward

    def run_n_episodes(self, env, n_eps:int = 1, initial_seed=-2, use_oracle=False):
        num_run_eps = 0
        total_rews = 0

        if initial_seed < 0:
            if self.task.mode == 'train':
                initial_seed = -2
            elif self.task.mode == 'val': # NOTE: beware of increasing val set to >100
                initial_seed = -1
            elif self.task.mode == 'test':
                initial_seed = -1 + 10000
            else:
                raise Exception("Invalid mode. Valid options: train, val, test")

        current_seed = initial_seed
        while num_run_eps < n_eps:
        # for epi_idx in range(cfg['n']):
            episode = []
            current_seed += 2
            num_run_eps += 1

            try:
                episode_reward = self.run_one_episode(env, episode, current_seed, use_oracle=use_oracle)
            except Exception as e:
                to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
                print(to_print)
            # Only save completed demonstrations.
            if episode_reward > 0.99: # and save_data:
                # dataset.add(seed, episode)
                total_rews += 1

            # if hasattr(env, 'blender_recorder'):
            #     print("blender pickle saved to ", '{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))
            #     env.blender_recorder.save('{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))

            print(f"*** {'(Oracle) ' if use_oracle else ''}Total Reward: {total_rews} / Episodes: {num_run_eps}")
        return total_rews

    def code_generation(self):   #, task_spec, task: tasks.Task):
        """ generate robot control script through interactions of agent and critic """
        self.code_generation_pass = True
        # _task_name = task_spec['task-name']
        mkdir_if_missing(self.cfg['model_output_dir'])

        try:
            start_time = time.time()
            # self.agent.api_review(fask_spec['task-name'])

            self.generated_code = self.agent.implement_task(self.task_spec)  # _task_name

            if self.critic:
                self.critic.error_review(self.generated_code)
            # self.generated_task_name = self.generated_task["task-name"]

            # self.generated_tasks.append(self.generated_task)
            # self.generated_task_assets.append(self.generated_asset)
            # self.generated_task_programs.append(self.generated_code)
            # self.generated_task_names.append(self.generated_task_name)
        except:
            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
            print("Task Creation Exception:", to_print)
            self.code_generation_pass = False

        print("task creation time {:.3f}".format(time.time() - start_time))



@hydra.main(config_path='../cliport/cfg', config_name='codegen')
def main(cfg):
    cfg['task'] = cfg['task'].replace("_", "-")
    print(f"\n---- Robot Control Script Generation ---- \n\t\tTask: {cfg['task']} ( prompt_folder= {cfg['prompt_folder']} )\n")
    openai.api_key = cfg['openai_key']

    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    cfg['model_output_dir'] = os.path.join(cfg['output_folder'], cfg['prompt_folder'] + "_" + model_time)
    if 'seed' in cfg:
        cfg['model_output_dir'] = cfg['model_output_dir'] + f"_{cfg['seed']}"

    set_gpt_model(cfg['gpt_model'])
    memory = Memory(cfg)
    agent = RoboScriptGenAgent(cfg, memory)
    critic = None #Critic(cfg, memory)
    runner = GenCodeRunner(cfg, agent, critic, memory)

    env = runner.setup_env(cfg['task'])
    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = -1 #dataset.max_seed
    max_eps = cfg['max_eps'] # 3 * cfg['n']

    # if 'regenerate_data' in cfg:
    #     dataset.n_episodes = 0

    runner.run_n_episodes(env, n_eps=max_eps, initial_seed=seed, use_oracle=True)
    print(f"obj_colors = {env.obj_colors}")
    print(f"obj_classes = {env.obj_classes}")
    if "fixed" in env.obj_ids and len(env.obj_ids["fixed"]) > 0:
        print("FIXED objects:")
        for obj_id in env.obj_ids['fixed']:
            print(f"\t({obj_id}]: type={env.get_object_class(obj_id)} color={env.get_object_color(obj_id)}")
        print()
    if "rigid" in env.obj_ids and len(env.obj_ids["rigid"]) > 0:
        print("Moveable rigid objects:")
        for obj_id in env.obj_ids['rigid']:
            print(f"\t({obj_id}]: type={env.get_object_class(obj_id)} color={env.get_object_color(obj_id)} ", end="")
            print_pose(env.get_object_pose(obj_id))
            print()
        print()
    # print()
    runner.code_generation()  # runner.task_spec['task-name']
    #runner.run_n_episodes(env, n_eps=max_eps, initial_seed=seed, use_oracle=False)
    print()
if __name__ == '__main__':
    main()
