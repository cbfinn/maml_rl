import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, save_video=True, video_filename='sim_out.mp4', reset_arg=None):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    images = []
    o = env.reset(reset_args=reset_arg)
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d: # and not animated:  # TODO testing
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
            if save_video:
                from PIL import Image
                image = env.wrapped_env.wrapped_env.get_viewer().get_image()
                pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
                images.append(np.flipud(np.array(pil_image)))

    if animated:
        if save_video and len(images) >= max_path_length:
            import moviepy.editor as mpy
            clip = mpy.ImageSequenceClip(images, fps=20*speedup)
            if video_filename[-3:] == 'gif':
                clip.write_gif(video_filename, fps=20*speedup)
            else:
                clip.write_videofile(video_filename, fps=20*speedup)
        #return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
