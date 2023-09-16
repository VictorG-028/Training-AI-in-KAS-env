from pettingzoo.butterfly import pistonball_v6
import supersuit as ss

env = pistonball_v6.parallel_env(render_mode="human")
env = ss.color_reduction_v0(env, mode="B")
env = ss.dtype_v0(env, "float32")
env = ss.resize_v1(env, x_size=84, y_size=84)
observations = env.reset()

print(observations["piston_0"].size)
print(observations["piston_0"].shape)
# print(observations["piston_0"])
print(type(observations["piston_0"]))
print(type(observations["piston_0"][0][0]))
input(">>>")

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
