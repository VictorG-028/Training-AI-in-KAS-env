# Setup Video

[Tutorial Setup RAY RL project](https://youtu.be/nsfjjUV5dvs)

# RL DQN and PPO in KAS environment with image observation

<div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div style="text-align: center; width: 25%;">
        <img 
            src="https://github.com/VictorG-028/Training-AI-in-KAS-env/blob/main/output/gif/DQN_880_playing.gif" 
            alt="DQN playing GIF" 
            width="25%"
        >
        <p>DQN playing GIF</p>
    </div>
    <div style="text-align: center; width: 25%;">
        <img 
            src="https://github.com/VictorG-028/Training-AI-in-KAS-env/blob/main/output/gif/PPO_100_playing.gif" 
            alt="PPO playing GIF" 
            width="25%"
        >
        <p>PPO playing GIF</p>
    </div>
</div>

# Informação relevante para o uso futuro da RAY[rllib]

- `"~\\ray_results\\"` é a pasta default em que a biblioteca salva os resultados e fica em `“C:\\Users\\{nome_usuário}\\Desktop\\RL\\ray_results\\”`.
- Se `tune.run` receber `storage_path`, a finalização do experimento salva os mesmos arquivos em 2 pastas: pasta padrão e `storage_path`.
- É possível parar um experimento e salvar um checkpoint ao usar `ctrl + C` **apenas uma vez**, a segunda vez não salva o checkpoint.
- O experimento finaliza somente quando termina uma iteração e verifica que chegou no critério de parada passado no parâmetro `stop` do `tuner.run`.
- O arquivo `experiment_state-{DateTime}.json` tem a informação `"_resumed": true`, caso o experimento tenha sido resumido com sucesso, False caso contrário.
- `progress.csv` guarda as mesmas informações dos arquivos `result.json` e `params.json` que estão na mesma pasta, ou seja, basta procurar resultados do experimento em `progress.csv`.
- O arquivo `progress.csv` tem uma coluna `“hist_stats/episode_reward”` com a recompensa de cada episódio.
- Qualquer coluna do `progress.csv` pode ser usada como critério de parada no parâmetro `stop`
- `ray.init(*num_cpus*=12, *num_gpus*=1)` determina o limite máximo de CPU e GPU, enquanto `.rollouts(num_rollout_workers=MAX_CPU-1)` determina quantas CPUs o tuner pode usar. Aviso: `num_rollout_workers` deve ser, no máximo, `MAX_CPU-1` segundo [essa thread](https://discuss.ray.io/t/most-efficient-way-to-use-only-a-cpu-for-training/1762) que diz “1 CPU é alocada sempre para o driver/local-worker process”.
- Aumentar o número de agentes no ambiente, aumenta muito o tempo de treino

# Erros e soluções encontrados

[**ERRO 1**] Ao executar `pip install -r requirements.txt`, ocorre um conflito de versões de dependência entre RAY[rllib] (≥ 2.6.2) e PettingZoo[Classic, Butterfly] (≥ 1.23.0) devido à divergência nas exigências de versões do Gynasium, onde RAY[rllib] requer uma versão mais antiga e PettingZoo[Classic, Butterfly] exige uma versão mais recente. 

[**SOLUÇÃO 1**] Resolvido ao trocar a versão do PettingZoo[Classic, Butterfly] para **== 1.22.3** (como sugerido nesse [pull request (closed)](https://github.com/ray-project/ray/pull/34696) )


[**ERRO 2**] Na versão do PettingZoo[Classic, Butterfly] == 1.22.3, **todos** os ambientes e wrappers são definidos com um método `reset` que espera 3 argumentos (o terceiro argumento é `return_info: bool` que deve ser `True`). Na versão latest do SuperSuit (biblioteca de wrappers para PettingZoo) e no wrapper RAY[rllib]/env/wrappers/pettingzoo_env.py esse mesmo método espera apenas 2 argumentos. Nas versões 1.23 em diante do PettingZoo[Classic, Butterfly], os ambiente e wrappers passaram a definir esse método com apenas 2 argumentos. O erro ocorre ao baixar versões diferentes dessas bibliotecas e tentar chamar o método `reset` sem passar o terceito argumento, ou ao tentar passar argumento demais.

[**SOLUÇÃO 2**] Seguindo os passos a seguir:

1. Criar virtual env `python 3.9`  (não pode ser 3.10, pois ainda não é suportada) com anaconda no VS Code com o comando `>Python:Create Environment` na command pallete que abre usando `ctrl + shift + p`.
2. `pip install -r requirements.txt` com as versões obrigatórias `PettingZoo[classic, butterfly]==1.22.3`, `ray[rllib]>=2.6.2`, `SuperSuit==3.8.0`
3. Abrir os arquivos e modificar para ficar parecido com exemplo abaixo
    
    Observação: É possível observar versões diferens dessa biblioteca no github e usar de referência para consertar o método reset
    
    3.1 `.conda/Lib/site-packages/pettingzoo/utils/wrappers/base_parallel.py` > Classe `BaseParallelWrapper` > Método `reset` na Linha 24 > Colocar `return_info=True`
    
    3.2 `.conda/Lib/site-packages/pettingzoo/utils/conversions.py` > Classe `aec_to_parallel_wrapper` > Método `reset` na Linha 126
    
    ```python
    def reset(self, seed=None, return_info=False, options=None):
            # print("@@@@@@@@@@@@@@@@@@@")
            # print("ESSE PRINT APARECE NO TERIMNAL, MAS É DIFICIL DE ACHAR")
            # print(f"{self.aec_env}")
            # print("@@@@@@@@@@@@@@@@@@@")
    				
    				# Antes
            # self.aec_env.reset(seed=seed, return_info=return_info, options=options)
    
            # Depois
    				self.aec_env.reset(seed=seed, options=options)
    
            self.agents = self.aec_env.agents[:]
            observations = {
                agent: self.aec_env.observe(agent)
                for agent in self.aec_env.agents
                if not (self.aec_env.terminations[agent] or self.aec_env.truncations[agent])
            }
    
            if not return_info:
                return observations
            else:
                infos = dict(**self.aec_env.infos)
                return observations, infos
    ```
    
    3.3 `.conda\Lib\site-packages\pettingzoo\utils\__init__.py` > Renomear import `BaseParallelWraper` para `BaseParallelWrapper` (fix typo)
    
    3.4 `.conda\Lib\site-packages\pettingzoo\utils\wrappers\__init__.py` > Renomear import `BaseParallelWraper` para `BaseParallelWrapper` (fix typo)
    
    3.5 `.conda\Lib\site-packages\pettingzoo\utils\wrappers\base_parallel.py` > Renomear classe `BaseParallelWraper` para `BaseParallelWrapper` (fix typo)

    3.6 `.conda\Lib\site-packages\ray\rllib\env\wrappers\pettingzoo_env.py` > Método `reset` na Linha 224 > remover `return_info` deixando apenas os outros 2 argumentos
    
4. Ativar ambiente executando `conda activate “path_do_ambiente”` ao abrir um novo terminal no VS Code


[**ERRO 3**] Falha ao carregar checkpoint com tune.run(resume=True) ou tune.run(resume=”LOCAL”)

[**SOLUÇÃO 3 (incompleta)**] Parcialmente resolvido ao modificar o arquivo `.conda/Lib/site-packages/ray/tune/execution/experiment_state.py` > função `_experiment_checkpoint_exists` para retornar `True` em vez de chamar pela segunda vez `_find_newest_experiment_checkpoint`. A primeira chamada retorna uma lista com o arquivo de checkpoint, enquanto a segunda chamada retorna None (bug provavelmente)  e faz a continuação do código falhar em carregar o checkpoint. Essa solução permite a continuação do experimento, mas o resultado salvo em progress.csv é errado.
