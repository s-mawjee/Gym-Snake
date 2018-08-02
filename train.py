from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger

import gym_snake.envs.snake_env
from gym_snake.envs.snake.view import LocalAction
from gym_snake.envs.snake.snake import Snake

import os
import datetime
import time

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=str, default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ts = time.time()
    ts_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join('/home/pasa/deeplearning/tf_models/snake/', ts_str, args.gpu)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Model storage/load path: " + save_path)

    logger.configure()
    env = gym_snake.envs.SnakeEnv(grid_size=[13, 13], unit_size=1, snake_size=4, unit_gap=0, action_transformer=LocalAction())
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 1), (64, 4, 1), (64, 3, 1)],
        hiddens=[256],
        dueling=False,
    )

    def callback(lcl, _glb):
        # stop training if reward exceeds 199
        print("eprewmean: " + str(sum(lcl['episode_rewards'][-101:-1]) / 100))
        is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
        return is_solved

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        #train_freq=4,
        #learning_starts=5,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=100,
        checkpoint_path=save_path,
        print_freq=10,
        #callback=callback
    )

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while True:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
            if done:
                break
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()