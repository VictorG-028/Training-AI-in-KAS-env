# Imports
import os
import ray
from ray import tune
from ray.air import session, CheckpointConfig
# from ray.air.checkpoint import Checkpoint
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv, PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Imports do algoritmo
from ray.rllib.algorithms.ppo import PPOConfig, PPO

# Imports arquivos src locais communs
from env_setup import env_creator, ENV_NAME
from cnn import CNNModelV2


# main
if __name__ == "__main__":
    ray.init(num_cpus=12, num_gpus=1)

    register_env(
        ENV_NAME, 
        lambda config: ParallelPettingZooEnv(env_creator(config))
    )
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    config = (
        PPOConfig()
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .environment(
            env=ENV_NAME, 
            # clip_actions=True, 
            disable_env_checking=True # 'True' devido ao erro "not passing checking"
        )
        .rollouts(
            num_rollout_workers=11, 
            # rollout_fragment_length=128
        )
        .training(
            train_batch_size=512,
            lr=3e-5,    # <----- Otimizado com optuna
            gamma=0.95, # <----- Otimizado com optuna
            lambda_=0.9,
            use_gae=True,
            # clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1, # <----- Otimizado com optuna
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,

            model={"custom_model": "CNNModelV2"}
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    # config.exploration_config.update({ # Decaying epsylon-greddy
    #     "initial_epsilon": 1.5,
    #     "final_epsilon": 0.01,
    #     "epsilon_timesteps": 1_000_000,
    # })

    print("Começou run")
    trial_response = tune.run(
        "PPO",
        name="PPO_V2", # Qualquer nome para o experimento
        stop={
            # Critério de parada do experimento
            # "episodes_total": 1000,
            # "timesteps_total": 1_000_000, # 1 milhão de passos
            "time_total_s": 300, # tempo em segundos
            # "training_iteration": 2,
        },
        checkpoint_config=CheckpointConfig(checkpoint_frequency=100),
        # checkpoint_freq=5, # Deprecated, ^^^ usar checkpoint_config instead ^^^
        storage_path="~\\ray_results\\" + ENV_NAME,
        # local_dir="~\\ray_results\\" + ENV_NAME, # Deprecated, ^^^ usar storage_path instead ^^^
        config=config.to_dict(), # O espaço de busca e outras configs são passadas nesse parâmetro
        resume=False, # resume: [True, False, "LOCAL", "REMOTE", "PROMPT", "AUTO"] para continuar o treino de onde parou
    )

    print(trial_response.results_df["hist_stats/episode_reward"]) # [8.0, 2.0, 2.0, 6.0, 3.0, 4.0, 3.0, 5.0, 12.0, ... [0.0, 3.0, 2.0, 3.0, 1.0, 1.0, 3.0, 3.0, 3.0, ... [8.0, 6.0, 4.0, 11.0, 3.0, 8.0, 8.0, 18.0, 4.0... [6.0, 10.0, 7.0, 6.0, 1.0, 10.0, 7.0, 4.0, 2.0... [1.0, 0.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 3.0, ... [8.0, 2.0, 5.0, 4.0, 4.0, 4.0, 6.0, 2.0, 3.0, ...
    print(trial_response.results_df["hist_stats/episode_lengths"])# [223, 163, 163, 223, 163, 203, 163, 163, 363, ... [163, 223, 183, 163, 183, 163, 163, 163, 183, ... [303, 283, 163, 323, 163, 323, 283, 503, 203, ... [223, 323, 203, 183, 163, 323, 203, 203, 163, ... [163, 163, 223, 183, 203, 203, 183, 163, 183, ... [303, 183, 183, 183, 203, 183, 203, 183, 223, ...

    # V1
    # primeiro trial ID 50058_00000 com 57_856 timesteps (12hr 43min 27s, 6cpus)
    # segundo trial ID 5a3bd_00000 com 100_352 timesteps (22hr 2min 57s, 6cpus)
    # terceiro trial ID 74e74_00000 com 56_832 timesteps (12hr 28min 52s, 12cpus)
    # quarto trial ID bf209_00000 com 64_512 timesteps (14hr 23min 58s, 12cpus)
    # quinto trial ID cbe8a_00000 com 94_720 timesteps (21hr 36min 41s, 12cpus)
    # sexto trial ID ace0c_00000 com 60_416 timesteps (15hr 17min 7s, 12cpus)
