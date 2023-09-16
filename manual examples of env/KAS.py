import pygame
from pettingzoo.butterfly import knights_archers_zombies_v10
import supersuit as ss

config = {
    "render_mode": "human",
    "spawn_rate": 5,
    "num_archers": 1, # <--- Single agent env
    "num_knights": 0,
    "max_zombies": 5,
    "max_arrows": 20,
    "killable_knights": True,
    "killable_archers": True,
    "pad_observation": True,
    "line_death": False,
    "max_cycles": 900,
    "vector_state": False, # <--- Image observation
    "use_typemasks": False,
}

env = knights_archers_zombies_v10.parallel_env(**config)
# env = ss.color_reduction_v0(env, mode="B")
# env = ss.dtype_v0(env, "float32")
# env = ss.resize_v1(env, x_size=84, y_size=84)
observations = env.reset(seed=42)

# print(observations)
# print(type(observations))
# print(observations["archer_0"])
print(observations["archer_0"].size)
print(observations["archer_0"].shape)
print(type(observations["archer_0"]))
print(type(observations["archer_0"][0][0]))

# clock = pygame.time.Clock()
# for agent in env.agents:
#     clock.tick(env.metadata["render_fps"])
#     observation, reward, termination, truncation, info = env.last()
#     print(observation.size)
#     print(observation.shape)
#     print(type(observation))
#     input(">>>")

#     if agent == env.agent:
#         # get user input (controls are WASD and space)
#         action = env(observation, agent)
#     else:
#         # this is where you would insert your policy (for non-player agents)
#         action = env.action_space(agent).sample()

#     env.step(action)
env.close()
