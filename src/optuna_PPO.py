# Imports
import os

import ray
from ray import tune
from ray.air import session, CheckpointConfig
# from ray.air.checkpoint import Checkpoint
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv, PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import optuna
import numpy as np

# Arquivos src locais communs
from env_setup import env_creator, ENV_NAME
from cnn import CNNModelV2

trial_details_to_store_in_text = []

def train(trial: optuna.Trial):
    lr = trial.suggest_categorical('lr', [1e-5, 2e-5, 3e-5])
    gamma = trial.suggest_categorical('gamma', [0.90, 0.95, 0.99])
    entropy_coeff = trial.suggest_categorical('entropy_coeff', [0.05, 0.1, 0.15])

    config = (
        PPOConfig()
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
            # lr=tune.grid_search([1e-5, 2e-5, 3e-5]),    # <----- Grid Search
            lr=lr,    # <----- optuna
            # gamma=tune.grid_search([0.90, 0.95, 0.99]), # <----- Grid Search
            gamma=gamma,    # <----- optuna
            lambda_=0.9,
            use_gae=True,
            # clip_param=0.4,
            grad_clip=None,
            # entropy_coeff=tune.grid_search([0.05, 0.1, 0.15]), # <----- Grid Search
            entropy_coeff=entropy_coeff,    # <----- optuna
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    print("Começou trial")
    trial_response = tune.run(
        "PPO",
        name="PPO_optuna",
        stop={
            "time_total_s": 2700, # 2700 segundos = 45 minutos
        },
        checkpoint_freq=100,
        local_dir="~/ray_results/" + ENV_NAME, # "~\\ray_results\\" é a pasta default em que a biblioteca salva os resultados
        config=config.to_dict(), # O espaço de busca e outras configs podem ser passadas nesse parâmetro
        resume=False, # resume: [True, False, "LOCAL", "REMOTE", "PROMPT", "AUTO"] para continuar o treino de onde parou
        # num_samples=2 # Quantos experimentos rodam em paralelo, permitindo explorar o espaço de busca mais rapidamente
    )

    # print(trial_response.results_df["hist_stats/episode_reward"])
    # print(trial_response.results_df["hist_stats/episode_lengths"])

    # Ao fim de cada tune.run(name, local_dir, storage_path, ...), o resultado é salvo em 2 pastas:
    # 1. pasta padrão -> "~/ray_results/{name}/{algoritmo}_{ambiente}_{random_hash}_00000_0_{DateTime}"
    # 2. pasta local_dir ou storage_path (depende de qual input foi utilizado no tune.run) "{local_dir ou storage_path}/{name}/{algoritmo}_{ambiente}_{random_hash}_00000_0_{DateTime}"
    # Ao que parece, o tune salava primeiro na pasta padrão e, então, copia o conteudo de "~/ray_results/{name}/" para "{local_dir ou storage_path}/{name}/" 
    # (foi observado que arquivos colocados manualemnte são copiados para a outra pasta)
    # Dentro da pasta do experiemnto, existe um arquivo progress.csv que armazeno as recompensas
    # A coluna do .csv "hist_stats/episode_reward" contem listas com os valores das recompensas por episódio
    # Cada linha dessa coluna representa uma iteração do experimento tune.run(). 
    # Cada iteração tem 0 ou mais episódios.
    # Exemplo: Se a primeira iteração tem 4 episódios, então a primeira linha da coluna "hist_stats/episode_reward" terá uma lista com 4 valores.
    # Exemplo: Se a N-ésima iteração tem M episódios, então a N-ésima linha da coluna "hist_stats/episode_reward" terá uma lista com M valores.
    # print(trial_response.results_df["hist_stats/episode_reward"].values[-20:][0])
    # print(trial_response.results_df["hist_stats/episode_reward"].values[-20:])
    last_20_rewards = trial_response.results_df["hist_stats/episode_reward"].values[-20:][0]
    avg_last_20_rewards = np.mean(last_20_rewards)

    # Guarda informações do trial num .txt
    trial_details_to_store_in_text.append(
        str(trial_response.results_df["time_total_s"].values[0]) + ", " + \
        str(trial_response.results_df["timesteps_total"].values[0]) + ", " + \
        str(trial_response.results_df["episodes_total"].values[0]) + "\n"
    )

    return avg_last_20_rewards



if __name__ == "__main__":
    ray.init(
        num_cpus=12, 
        num_gpus=1
    )

    register_env(
        ENV_NAME, 
        lambda config: ParallelPettingZooEnv(env_creator(config))
    )
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    study = optuna.create_study(
        direction='maximize',
        storage='sqlite:///PPO_optuna_results.db',
        study_name='PPO_KAZ',
        load_if_exists=True
    )

    study.optimize(train, n_trials=27)

    print("=------------------=")
    print("MELHORES PARÂMETROS:")
    print(study.best_params) # {'lr': 3e-05, 'gamma': 0.95, 'entropy_coeff': 0.1}
    print("=------------------=")
    print(trial_details_to_store_in_text)
    with open("all_optuna_trial_details.txt", "w") as f:
        f.writelines(trial_details_to_store_in_text)
