from ppo import PPO
import gym
from gym.wrappers.record_video import RecordVideo
from simulators.discrete_grid_world import ImageGridWorld

if __name__ == '__main__':
    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id, render_mode="rgb_array")
    env = ImageGridWorld(size=(5, 5), goal_pos=24, max_steps=30)
    # env = RecordVideo(env, './video', episode_trigger=lambda episode_number: True)
    # print(env.action_space.n)
    agent = PPO(env, h_layers=[32, 32, 32], seed=0, steps_per_epoch=600, epochs=150, gamma=0.99, lam=0.95,
                clip_ratio=0.2, lr_a=1e-4, lr_c=1e-3, train_a_iters=80, train_c_iters=80,
                max_ep_len=2000, kl_target=0.01, ent_weight=0.001, save_freq=100, save_path='./checkpoints/',
                mode="discrete")

    # # training
    agent.train()

    # test
    path = './checkpoints/actor_checkpoint899'
    agent.test(path)
