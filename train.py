from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger

import gym_snake.envs.snake_env
from gym_snake.envs.snake.view import LocalAction
from gym_snake.envs.snake.snake import Snake
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy

import os
import datetime
import time
import tensorflow as tf
import numpy as np

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=str, default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ts = time.time()
    # ts_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    # save_path = os.path.join('/home/pasa/deeplearning/tf_models/snake/', ts_str)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # print("Model storage/load path: " + save_path)
    #
    # logger.configure()
    env = gym_snake.envs.SnakeEnv(grid_size=[11, 11], unit_size=1, snake_size=2, unit_gap=0, n_snakes=1, n_foods=13)
    # model = deepq.models.cnn_to_mlp(
    #     convs=[(32, 5, 1), (64, 3, 1), (64, 3, 1)],
    #     hiddens=[512, 256],
    #     dueling=True,
    # )

    # def callback(lcl, _glb):
    #     # stop training if reward exceeds 199
    #     print("eprewmean: " + str(sum(lcl['episode_rewards'][-101:-1]) / 100))
    #     is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    #     return is_solved
    #
    # act = deepq.learn(
    #     env,
    #     q_func=model,
    #     lr=1e-5,
    #     max_timesteps=300000,
    #     buffer_size=10000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.0001,
    #     #train_freq=4,
    #     #learning_starts=5,
    #     target_network_update_freq=1000,
    #     gamma=1,
    #     prioritized_replay=True,
    #     prioritized_replay_alpha=0.6,
    #     checkpoint_freq=100,
    #     checkpoint_path=save_path,
    #     print_freq=10,
    #     only_load=False
    # )
    tf.Session().__enter__()

    num_timesteps = 1000000
    policy =  CnnPolicy
    model = ppo2.learn(policy=policy, env=env, nsteps=64, nminibatches=4,
        lam=1, gamma=1, noptepochs=4, log_interval=10,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.3,
        total_timesteps=int(num_timesteps * 1.1),
        save_interval=50)
        #load_path="/tmp/openai-2018-08-08-01-55-50-649928/checkpoints/00450")



    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    while True:
        actions = model.step(obs)[0]
        obs[:]  = env.step(actions)[0]
        env.render()


if __name__ == '__main__':
    main()