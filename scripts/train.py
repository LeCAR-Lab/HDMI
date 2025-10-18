import torch

# import warp
import hydra
import numpy as np
import einops
import wandb
import logging
import os
import time
import datetime

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

import active_adaptation as aa
import mjlab

# from active_adaptation.utils.torchrl import SyncDataCollector
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase
from tensordict import TensorDict

# local import
from scripts.helpers import make_env_policy, EpisodeStats, evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")


@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    print(
        f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    run = wandb.init(
        job_type=cfg.wandb.job_type,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )
    run.config.update(OmegaConf.to_container(cfg))

    default_run_name = (
        f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    run_idx = run.name.split("-")[-1]
    run.name = f"{run_idx}-{default_run_name}"
    setproctitle(run.name)

    cfg_save_path = os.path.join(run.dir, "cfg.yaml")
    OmegaConf.save(cfg, cfg_save_path)
    run.save(cfg_save_path, policy="now")
    run.save(os.path.join(run.dir, "config.yaml"), policy="now")

    env, policy, vecnorm = make_env_policy(cfg)

    import inspect
    import shutil

    source_path = inspect.getfile(policy.__class__)
    target_path = os.path.join(run.dir, source_path.split("/")[-1])
    shutil.copy(source_path, target_path)
    wandb.save(target_path, policy="now")

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str, artifact: bool = False):
        ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        if "vecnorm" in locals():
            state_dict["vecnorm"] = vecnorm.state_dict()
        torch.save(state_dict, ckpt_path)
        if artifact:
            artifact = wandb.Artifact(
                f"{type(env).__name__}-{type(policy).__name__}", type="model"
            )
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact)
        run.save(ckpt_path, policy="now", base_path=run.dir)
        logging.info(f"Saved checkpoint to {str(ckpt_path)}")

    assert env.training

    def should_save(i):
        if not aa.is_main_process():
            return False
        return i > 0 and save_interval > 0 and i % save_interval == 0

    # 4. --- Training Loop ---
    carry = env.reset()
    rollout_policy: TensorDictModuleBase = policy.get_rollout_policy("train")

    with torch.inference_mode():
        tmp_carry = rollout_policy(carry.clone(False))
        tmp_td, _ = env.step_and_maybe_reset(tmp_carry.clone(False))
        tmp_td["next"] = tmp_td["next"].select(
            "done",
            "terminated",
            "discount",
            "reward",
            "stats",
            "is_init",
            "adapt_hx",
            strict=False,
        )

    N = env.num_envs
    T = cfg.algo.train_every
    device = env.device

    data_buf = TensorDict({}, batch_size=[N, T], device=device)
    for key, value in tmp_td.items(include_nested=True, leaves_only=True):
        shape_tail = value.shape[1:]
        buf = torch.empty((N, T, *shape_tail), dtype=value.dtype, device=device)
        data_buf.set(key, buf)
    logging.info(f"Data buffer size: {data_buf.bytes() / 1e6:.2f} MB")

    if aa.is_main_process():
        progress = tqdm(range(total_iters))
    else:
        progress = range(total_iters)

    env_frames = 0
    start_iter = env.current_iter
    for i in progress:
        rollout_start = time.perf_counter()
        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            torch.compiler.cudagraph_mark_step_begin()  # for compiled policy
            env.set_progress(start_iter + i)
            for step in range(cfg.algo.train_every):
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                td["next"] = td["next"].select(
                    "done",
                    "terminated",
                    "discount",
                    "reward",
                    "stats",
                    "is_init",
                    "adapt_hx",
                    strict=False,
                )
                data_buf[:, step] = td
            policy.critic(data_buf)
            values = data_buf["state_value"]
            data_buf["next", "state_value"] = torch.where(
                data_buf["next", "done"],
                values,  # a walkaround to avoid storing the next states
                torch.cat(
                    [
                        values[:, 1:],
                        policy.critic(carry.copy())["state_value"].unsqueeze(1),
                    ],
                    dim=1,
                ),
            )
        rollout_time = time.perf_counter() - rollout_start

        episode_stats.add(data_buf)
        env_frames += data_buf.numel()

        info = {}
        if i % log_interval == 0 and len(episode_stats):
            for k, v in sorted(episode_stats.pop().items(True, True)):
                key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                info[key] = torch.mean(v.float()).item()
        training_start = time.perf_counter()
        info.update(policy.train_op(data_buf))
        training_time = time.perf_counter() - training_start
        info.update(env.extra)
        info.update(env.stats_ema)

        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters)

        info["env_frames"] = env_frames
        info["rollout_fps"] = data_buf.numel() / rollout_time
        info["training_time"] = training_time

        if should_save(i):
            save(policy, f"checkpoint_{i}")

        if aa.is_main_process():
            # print(OmegaConf.to_yaml({k: v for k, v in info.items() if (isinstance(v, (float, int)) and not k.startswith("performance_reward"))}))
            run.log(info)

    # 5. --- Finalization and Cleanup ---
    if aa.is_main_process():
        save(policy, "checkpoint_final", artifact=True)

    policy_eval = policy.get_rollout_policy("eval")
    info, trajs, stats, policy_trajs = evaluate(
        env, policy_eval, render=cfg.eval_render, seed=cfg.seed
    )
    run.log(info)

    wandb.finish()
    os._exit(0)
    env.close()

    run_id = run.id
    project = run.project
    entity = run.entity
    run_path = f"{entity}/{project}/{run_id}"

    return run_path


if __name__ == "__main__":
    main()
