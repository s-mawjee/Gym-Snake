from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger

import gym_snake.envs.snake_env
from gym_snake.envs.snake.view import LocalAction
from gym_snake.envs.snake.snake import Snake

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-path', type=str, default=None)

    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = gym_snake.envs.SnakeEnv(grid_size=[13, 13], unit_size=1, snake_size=4, unit_gap=0, action_transformer=LocalAction())
    env = bench.Monitor(env, logger.get_dir())
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
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        #train_freq=4,
        #learning_starts=5,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        #checkpoint_freq=args.checkpoint_freq,
        #checkpoint_path=args.checkpoint_path,
        print_freq=10,
        #callback=callback
    )

    env.close()


if __name__ == '__main__':
    main()