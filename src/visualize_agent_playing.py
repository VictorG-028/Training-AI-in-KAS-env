import os
import ray
import supersuit as ss
from PIL import Image
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Imports arquivos src locais communs
from env_setup import not_parallel_env_creator, ENV_NAME
from cnn import CNNModelV2

os.environ["SDL_VIDEODRIVER"] = "dummy"

PLAY_WITH_DQN = True

current_dir = os.path.dirname(os.path.abspath(__file__))
if PLAY_WITH_DQN:
    checkpoint_filename = "checkpoints\\DQN\\checkpoint_000880"
else:
    checkpoint_filename = "checkpoints\\PPO\\checkpoint_000100"

print(current_dir)
absolute_checkpoint_path = os.path.join(current_dir, checkpoint_filename)
checkpoint_path = os.path.expanduser(absolute_checkpoint_path)
ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

env = not_parallel_env_creator(None)
register_env(
    ENV_NAME, 
    lambda config: PettingZooEnv(not_parallel_env_creator(config))
)


ray.init(num_cpus=12, num_gpus=1)

# PPOagent = PPO.from_checkpoint(checkpoint_path)
if PLAY_WITH_DQN:
    algorithm_agent = DQN.from_checkpoint(checkpoint_path)
else:
    algorithm_agent = PPO.from_checkpoint(checkpoint_path)

reward_sum = 0
frame_list = []
i = 0
env.reset()

for mock_agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    reward_sum += reward
    if termination or truncation:
        action = None
    else:
        action = algorithm_agent.compute_single_action(observation)

    env.step(action)
    i += 1
    if i % (len(env.possible_agents) + 1) == 0:
        img = Image.fromarray(env.render())
        frame_list.append(img)
env.close()


print("Recompensa", reward_sum)
frame_list[0].save(
    "DQN_playing.gif" if PLAY_WITH_DQN else "PPO_playing.gif", 
    save_all=True, 
    append_images=frame_list[1:], 
    duration=3, 
    loop=0
)
