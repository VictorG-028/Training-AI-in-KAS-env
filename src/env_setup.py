from pettingzoo.butterfly import knights_archers_zombies_v10
import supersuit as ss

# Environment setup
ENV_NAME = "knights_archers_zombies_v10"

def env_creator(args):
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=20,
        max_zombies=15,
        num_archers=2, # <----- Quantos agentes no ambiente
        num_knights=2, # <-|
        max_arrows=25,
        killable_knights=True,
        killable_archers=True,
        pad_observation=True,
        line_death=False,
        max_cycles=900, # Quantidade máxima de steps em cada episódio
        vector_state=False, # <--- Observação em formato de imagem
        use_typemasks=False,
    )
    # Wrappers que processam a observação
    env = ss.color_reduction_v0(env, mode="B") # mode="B" reduz observação RGB para Gray Scale
    env = ss.dtype_v0(env, "float32") # Ambiente PettingZoo[Butterfly] tem observação(imagem) com tipo uint8 -> float32
    env = ss.resize_v1(env, x_size=84, y_size=84) # Reduz a imagem do tamanho 512x512 original para 84x84 
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1) # Transforma valores da imagem de [0, 255] para [0, 1]
    env = ss.frame_stack_v1(env, stack_size=3) # Aplica frame stack com 3 frames
    return env


def not_parallel_env_creator(args):
    env = knights_archers_zombies_v10.env(
        spawn_rate=20,
        max_zombies=15,
        num_archers=2, # <----- Quantos agentes no ambiente
        num_knights=2, # <-|
        max_arrows=25,
        killable_knights=True,
        killable_archers=True,
        pad_observation=True,
        line_death=False,
        max_cycles=900, # Quantidade máxima de steps em cada episódio
        vector_state=False, # <--- Observação em formato de imagem
        use_typemasks=False,
        render_mode="rgb_array",
    )
    # Wrappers que processam a observação
    env = ss.color_reduction_v0(env, mode="B") # mode="B" reduz observação RGB para Gray Scale
    env = ss.dtype_v0(env, "float32") # Ambiente PettingZoo[Butterfly] tem observação(imagem) com tipo uint8 -> float32
    env = ss.resize_v1(env, x_size=84, y_size=84) # Reduz a imagem do tamanho 512x512 original para 84x84 
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1) # Transforma valores da imagem de [0, 255] para [0, 1]
    env = ss.frame_stack_v1(env, stack_size=3) # Aplica frame stack com 3 frames
    return env
