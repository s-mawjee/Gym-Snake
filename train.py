from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger

import gym_snake.envs.snake_env
from gym_snake.envs.snake.view import LocalAction
from gym_snake.envs.snake.snake import Snake

import os

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=str, default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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

    deepq.learn(
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
        #checkpoint_freq=args.checkpoint_freq,
        #checkpoint_path=args.checkpoint_path,
        print_freq=10,
        #callback=callback
    )

    env.close()


if __name__ == '__main__':
    main()