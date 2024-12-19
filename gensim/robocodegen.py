"""Generate robot control script for given task."""

import os
import hydra
import numpy as np
import random

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
import IPython
import random

@hydra.main(config_path='../cliport/cfg', config_name='codegen')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=[] #cfg['record']
    )
    cfg['task'] = cfg['task'].replace("_", "-")
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    # dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

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

    curr_run_eps = 0
    total_rews = 0

    # Collect training data from oracle demonstrations.
    while curr_run_eps < max_eps:
    # for epi_idx in range(cfg['n']):
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)
        print('Seed: {}'.format(seed))
        try:
            curr_run_eps += 1 # make sure exits the loop
            env.set_task(task)
            obs = env.reset()
            info = env.info
            reward = 0

            # Unlikely, but a safety check to prevent leaks.
            if task.mode == 'val' and seed > (-1 + 10000):
                raise Exception("!!! Seeds for val set will overlap with the test set !!!")


            # Rollout expert policy
            for _ in range(task.max_steps):
                act = agent.act(obs, info)
                episode.append((obs, act, reward, info))
                lang_goal = info['lang_goal']
                obs, reward, done, info = env.step(act)
                total_reward += reward
                print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
                if done:
                    break

        except Exception as e:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import TerminalFormatter
            import traceback

            to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
            print(to_print)
            continue

        episode.append((obs, None, reward, info))  # episode is a list of (obs, act, reward, info) tuples
                                                   # act has 2 fields: act['pose0'] and act['pose1']
        # Only save completed demonstrations.
        if total_reward > 0.99: # and save_data:
            # dataset.add(seed, episode)
            total_rews += 1

        # if hasattr(env, 'blender_recorder'):
        #     print("blender pickle saved to ", '{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))
        #     env.blender_recorder.save('{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))

        print(f"Cumulative Reward: {total_rews} / Episodes: {curr_run_eps}")

if __name__ == '__main__':
    main()
