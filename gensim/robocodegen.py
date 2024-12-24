"""Generate robot control script for given task."""

import os
import hydra
import numpy as np
import openai
import random

from datetime import datetime
from pprint import pprint

from cliport import tasks
# from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
from gensim.memory import Memory
from gensim.utils import (
    mkdir_if_missing,
    save_text,
    save_stat,
    set_gpt_model,
    add_to_txt,
    clear_messages
)

class GenCodeRunner:
    """ the main class that runs simulation loop """
    def __init__(self, cfg, agent, critic, memory):
        self.cfg = cfg
        self.agent = agent
        self.critic = critic
        self.memory = memory

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
        self.cfg['task'] = task_name
        self.current_task_name = task_name
        """ build the new task"""
        env = Environment(
                self.cfg['assets_root'],
                disp=self.cfg['disp'],
                shared_memory=self.cfg['shared_memory'],
                hz=480,
                record_cfg=[] #self.cfg['record']
            )

        task = tasks.names[self.current_task_name]()
        task.mode = self.cfg['mode']
        # save_data = self.cfg['save_data']

        # Initialize scripted oracle agent and dataset.
        # expert = task.oracle(env)
        data_path = os.path.join(self.cfg['data_dir'], "{}-{}".format(self.current_task_name, task.mode))
        # dataset = RavensDataset(data_path, self.cfg, n_demos=0, augment=False)
        dataset = None
        print(f"***** Task: {self.cfg['task']} mode={task.mode} lang_goal='{task.get_lang_goal()}'")
        return task, env

    def run_one_episode(self, env, task, episode, seed):
        """ run the new task for one episode """
        add_to_txt(
                self.chat_log, f"================= TRIAL: {self.curr_trials}", with_print=True)
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{} | Seed: {}'.format(self.n_episodes + 1, self.cfg['max_eps'], seed))
        expert = task.oracle(env)
        env.set_task(task)
        obs = env.reset()

        info = env.info
        reward = 0
        episode_reward = 0

        # Rollout expert policy
        for _ in range(task.max_steps):
            act = expert.act(obs, info)
            episode.append((obs, act, reward, info))
            lang_goal = info['lang_goal']
            obs, reward, done, info = env.step(act)
            episode_reward += reward
            print(f'Total Reward: {episode_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
        episode.append((obs, None, reward, info))  # episode is a list of (obs, act, reward, info) tuples
                                                   # act has 2 fields: act['pose0'] and act['pose1']
        return episode_reward


@hydra.main(config_path='../cliport/cfg', config_name='codegen')
def main(cfg):
    cfg['task'] = cfg['task'].replace("_", "-")
    if False:
        # Initialize environment and task.
        env = Environment(
            cfg['assets_root'],
            disp=cfg['disp'],
            shared_memory=cfg['shared_memory'],
            hz=480,
            record_cfg=[] #cfg['record']
        )
        task = tasks.names[cfg['task']]()
        task.mode = cfg['mode']
        save_data = cfg['save_data']

        # Initialize scripted oracle agent and dataset.
        agent = task.oracle(env)
        # data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
        # dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
        # print(f"Saving to: {data_path}")
        print(f"***** Task: {cfg['task']} mode={task.mode} lang_goal='{task.get_lang_goal()}'")
    else:
        openai.api_key = cfg['openai_key']

        model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        cfg['model_output_dir'] = os.path.join(cfg['output_folder'], cfg['prompt_folder'] + "_" + model_time)
        if 'seed' in cfg:
            cfg['model_output_dir'] = cfg['model_output_dir'] + f"_{cfg['seed']}"

        set_gpt_model(cfg['gpt_model'])
        memory = Memory(cfg)
        agent = None #Agent(cfg, memory)
        critic = None #Critic(cfg, memory)
        runner = GenCodeRunner(cfg, agent, critic, memory)

    task, env = runner.setup_env(cfg['task'])
    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = -1 #dataset.max_seed
    max_eps = cfg['max_eps'] # 3 * cfg['n']

    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # if 'regenerate_data' in cfg:
    #     dataset.n_episodes = 0

    num_run_eps = 0
    total_rews = 0

    # Collect training data from oracle demonstrations.
    while num_run_eps < max_eps:
    # for epi_idx in range(cfg['n']):
        episode = []
        seed += 2
        num_run_eps += 1

        try:
            episode_reward = runner.run_one_episode(env, task, episode, seed)
        except Exception as e:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import TerminalFormatter
            import traceback

            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
            print(to_print)
        # Only save completed demonstrations.
        if episode_reward > 0.99: # and save_data:
            # dataset.add(seed, episode)
            total_rews += 1

        # if hasattr(env, 'blender_recorder'):
        #     print("blender pickle saved to ", '{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))
        #     env.blender_recorder.save('{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))

        print(f"Cumulative Reward: {total_rews} / Episodes: {num_run_eps}")

if __name__ == '__main__':
    main()
