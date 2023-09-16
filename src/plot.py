import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Optional, Union

def smooth(data, window):
  data = np.array(data)
  n = len(data)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i-window+1)
    y[i] = data[start:(i+1)].mean()
  return y

def plot_rewards(
        rewards_values: Union[List[Union[int, float]], np.ndarray],
        rewards_values_2: Optional[Union[List[Union[int, float]], np.ndarray]] = None,
        y_max_suggested: Optional[Union[int, float, np.number]] = None, 
        show_x_in_log_scale: bool = False, 
        smooth_window_len: int = 10, 
        x_label: str = 'episode', 
        algorithm_name: str = 'Algorithm_name name here',
        should_plot_std_deviation: bool = False,
        should_save: bool = False,
        filename: Optional[str] = "plot", 
        cumulative: bool = False
    ):
    '''Exibe um gráfico "episódio/passo x retorno", fazendo a média a cada `window` retornos, para suavizar.
    
    Parâmetros:
    - x_values: se return_type=='episode', este parâmetro é uma lista de retornos a cada episódio; se return_type=='step', é uma lista de pares (passo,retorno) 
    - ymax_suggested (opcional): valor máximo de retorno (eixo y), se tiver um valor máximo conhecido previamente
    - show_x_in_log_scale: se for True, mostra o eixo x na escala log (para detalhar mais os resultados iniciais)
    - smooth_window_len: permite fazer a média dos últimos resultados, para suavizar o gráfico
    - filename: se for fornecida uma string, salva um arquivo de imagem ao invés de exibir.
    '''

    if rewards_values_2 is not None:
        length_diff = rewards_values.size - rewards_values_2.size
        if length_diff > 0:
            print(rewards_values_2[-1])
            padding = np.full(length_diff, rewards_values_2[-1])
            rewards_values_2 = np.concatenate((rewards_values_2, padding))


    plt.figure(figsize=(12,7))

    if x_label == 'episode':
        plt.xlabel('Episódios')
        x_label = "episodio"
    elif x_label == 'step':
        plt.xlabel('Passos')
        x_label = "step"
    else:
        assert x_label in ['episode', 'step'], \
            "[plot_result ERROR] x_label must be one of " \
            "the valid options ['episode', 'step']"
    
    if cumulative:
            rewards_values = np.cumsum(rewards_values)
            rewards_values_2 = np.cumsum(rewards_values_2)
            acumulative_title = "acumulada "
    else:
        acumulative_title = ""

    plt.ylabel('Recompensa do episódio')

    x_values = np.arange(1, len(rewards_values)+1)
    y_values = smooth(rewards_values, smooth_window_len)
    if rewards_values_2 is not None: 
        y_values_2 = smooth(rewards_values_2, smooth_window_len)
    plt.plot(x_values, y_values, label=algorithm_name)
    if rewards_values_2 is not None:
        plt.plot(x_values, y_values_2, label='PPO_5')
    plt.title(f"Recompensa {acumulative_title}por {x_label} (suavizada com janela de tamanho {smooth_window_len})")

    if should_plot_std_deviation:
        std_dev = np.std(y_values)
        upper_bound = y_values + std_dev
        lower_bound = y_values - std_dev
        plt.fill_between(x_values, lower_bound, upper_bound, alpha=0.2, label=f'Std Dev ({algorithm_name})')
        
        if rewards_values_2 is not None:
            std_dev = np.std(y_values_2)
            upper_bound = y_values_2 + std_dev
            lower_bound = y_values_2 - std_dev
            plt.fill_between(x_values, lower_bound, upper_bound, alpha=0.2, label='Std Dev (PPO_5)')

    if show_x_in_log_scale:
        plt.xscale('log')

    if y_max_suggested:
        y_max = np.max([y_max_suggested, np.max(y_values)])
        plt.ylim(top=y_max)

    plt.legend(fontsize=18)

    if should_save:
        plt.savefig("output/plot/" + filename)
        print(f"Arquivo salvo: {filename}")

    plt.show()
    plt.close()

def load_all_optuna_progress_csv() -> pd.DataFrame:
    results_optuna_path = "C:\\Users\\victo\\ray_results\\knights_archers_zombies_v10\\PPO_optuna"
    optuna_results_df = [] # List with 27 dataframes

    # Loop through each folder in the specified path
    for folder_name in os.listdir(results_optuna_path):
        folder_path = os.path.join(results_optuna_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Construct the path to the progress.csv file in each folder
            progress_csv_path = os.path.join(folder_path, "progress.csv")
            
            # Check if the progress.csv file exists in the folder
            if os.path.isfile(progress_csv_path):
                # Read the CSV file into a DataFrame and append it to the list
                df = pd.read_csv(progress_csv_path)
                optuna_results_df.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(optuna_results_df, ignore_index=True)
    return combined_df


def process_dataframe_episode_rewards(
        dataframe: pd.DataFrame, 
        column_name='hist_stats/episode_reward'
    ) -> np.ndarray:

    episode_rewards = []
    convert_string_to_array_of_floats = lambda some_string: eval(some_string)

    for index, row in dataframe.iterrows(): # for each linha no dataframe:
        # Cada elemento da coluna 'hist_stats/episode_reward' é uma string 
        # Cada string representa um array de recompensa de 1 iteração 
        # 1 iteração tem 0 ou mais episódios
        # Converte e junta a string em um array de float 
        # Cada float é uma reward de um episódio
        one_iteration_episode_rewards = convert_string_to_array_of_floats(row[column_name])
        episode_rewards.extend(one_iteration_episode_rewards)

    # print("size -> ", np.array(episode_rewards, dtype=float).size)
    return np.array(episode_rewards, dtype=float)

# Read the files data
results_DQN = pd.read_csv("C:\\Users\\victo\\ray_results\\knights_archers_zombies_v10\\DQN_V2\\DQN_knights_archers_zombies_v10_282b3_00000_0_2023-08-28_03-28-52\\progress.csv")
results_PPO_1 = pd.read_csv("C:\\Users\\victo\\ray_results\\knights_archers_zombies_v10\\PPO_V1\\PPO_knights_archers_zombies_v10_50058_00000_0_2023-08-31_05-35-17\\progress.csv")
results_PPO_2 = pd.read_csv("C:\\Users\\victo\\ray_results\\knights_archers_zombies_v10\\PPO_V1\\PPO_knights_archers_zombies_v10_5a3bd_00000_0_2023-09-01_01-31-00\\progress.csv")
results_PPO_3 = pd.read_csv("C:\\Users\\victo\\ray_results\\knights_archers_zombies_v10\\PPO_V1\\PPO_knights_archers_zombies_v10_74e74_00000_0_2023-09-02_06-31-12\\progress.csv")
results_PPO_4 = pd.read_csv("C:\\Users\\victo\\ray_results\\knights_archers_zombies_v10\\PPO_V1\\PPO_knights_archers_zombies_v10_bf209_00000_0_2023-09-03_03-25-59\\progress.csv")
results_PPO_5 = pd.read_csv("C:\\Users\\victo\\ray_results\\knights_archers_zombies_v10\\PPO_V1\\PPO_knights_archers_zombies_v10_cbe8a_00000_0_2023-09-04_00-47-40\\progress.csv")
results_PPO_6 = pd.read_csv("C:\\Users\\victo\\ray_results\\knights_archers_zombies_v10\\PPO_V1\\PPO_knights_archers_zombies_v10_ace0c_00000_0_2023-09-05_02-18-40\\progress.csv")
results_optuna = load_all_optuna_progress_csv()

# Process each dataframe to get episode rewards
episode_rewards_DQN = process_dataframe_episode_rewards(results_DQN)
# episode_rewards_PPO_1 = process_dataframe_episode_rewards(results_PPO_1)
episode_rewards_PPO_2 = process_dataframe_episode_rewards(results_PPO_2) # Experimento com mais episódios
# episode_rewards_PPO_3 = process_dataframe_episode_rewards(results_PPO_3)
# episode_rewards_PPO_4 = process_dataframe_episode_rewards(results_PPO_4)
episode_rewards_PPO_5 = process_dataframe_episode_rewards(results_PPO_5) # Segundo experimento com mais episódio
# episode_rewards_PPO_6 = process_dataframe_episode_rewards(results_PPO_6)

# Plot the graph and save image
SHOULD_SAVE = True
SHOULD_PLOT_STD_DEVIATION = False
plot_rewards(episode_rewards_DQN, algorithm_name="DQN", cumulative=False, smooth_window_len=1000, should_save=SHOULD_SAVE, filename="episode_rewards_DQN", should_plot_std_deviation=SHOULD_PLOT_STD_DEVIATION)
# plot_rewards(episode_rewards_DQN, algorithm_name="DQN", cumulative=True, smooth_window_len=10, should_save=SHOULD_SAVE, filename="cum_episode_rewards_DQN", should_plot_std_deviation=SHOULD_PLOT_STD_DEVIATION)
# plot_rewards(episode_rewards_PPO_2, algorithm_name="PPO_2", cumulative=False, smooth_window_len=1000, should_save=SHOULD_SAVE, filename="episode_rewards_PPO_2", should_plot_std_deviation=SHOULD_PLOT_STD_DEVIATION)
# plot_rewards(episode_rewards_PPO_2, algorithm_name="PPO_2", cumulative=True, smooth_window_len=10, should_save=SHOULD_SAVE, filename="cum_episode_rewards_PPO_2", should_plot_std_deviation=SHOULD_PLOT_STD_DEVIATION)
# plot_rewards(episode_rewards_PPO_5, algorithm_name="PPO_2", cumulative=False, smooth_window_len=1000, should_save=SHOULD_SAVE, filename="episode_rewards_PPO_5", should_plot_std_deviation=SHOULD_PLOT_STD_DEVIATION)
# plot_rewards(episode_rewards_DQN, episode_rewards_PPO_2, cumulative=False, smooth_window_len=1000, should_save=SHOULD_SAVE, filename="episode_rewards_DQN_PPO")
# plot_rewards(episode_rewards_PPO_2, episode_rewards_PPO_5, algorithm_name="PPO_2", cumulative=False, smooth_window_len=1000, should_save=SHOULD_SAVE, filename="episode_rewards_PPO2_PPO5", should_plot_std_deviation=SHOULD_PLOT_STD_DEVIATION)

##### Observa as colunas do dataframe #####
# for c in results_DQN.columns:
#     print(c)

##### Verifica quanto tempo demorou para rodar o algoritmo DQN #####

# tempo_total_execucao = results_DQN['time_total_s'].iloc[-1]
# results_DQN['timestamp'] = pd.to_datetime(results_DQN['timestamp'], unit='s')
# inicio_execucao = results_DQN['timestamp'].min()
# fim_execucao = results_DQN['timestamp'].max()
# diferenca_tempo = (fim_execucao - inicio_execucao).total_seconds()

# print(results_DQN['time_total_s'])
# print(results_DQN['date'])
# print(results_DQN['timestamp'])

# def segundos_para_hms(segundos):
#     horas = segundos // 3600
#     minutos = (segundos % 3600) // 60
#     segundos = segundos % 60
#     return horas, minutos, segundos
# horas, minutos, segundos = segundos_para_hms(diferenca_tempo)

# print(f"Tempo total da execução (time_total_s): {tempo_total_execucao} segundos")
# print(f"Tempo total da execução (diferença de timestamp): {diferenca_tempo} segundos")
# print(f"Tempo total da execução (conversão da diferença de timestamp): {horas} horas, {minutos} minutos, {segundos} segundos")

##### Faz um teste para encontrar quantos episódios foram rodados #####
# print(results_DQN['episodes_total'])
# print(results_DQN['episodes_total'].sum())
# print(results_DQN['hist_stats/episode_reward'])

# for index, row in results_DQN.iterrows():
#     value_column1 = eval(row['hist_stats/episode_reward'])
#     value_column2 = row['episodes_total']

#     test = (len(value_column1) == value_column2)

#     if not test:
#         print(f"({len(value_column1)}, {value_column2})")
