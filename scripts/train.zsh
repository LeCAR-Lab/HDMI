# exp combinations
# 1. data=data/motion/g1/omomo/sub1_suitcase_011, eef_contact_exp=true, exp_name=move_suitcase_originalmotion_contact
# 2. data=data/motion/g1/omomo/sub1_suitcase_011, eef_contact_exp=false, exp_name=move_suitcase_originalmotion_nocontact
# 3. data=data/retarget/output, eef_contact_exp=false, exp_name=move_suitcase_retargetmotion_nocontact

NUM_ENVS=8192
HEADLESS=true

# Experiment 1: Original motion with contact
uv run scripts/train.py \
    eval_render=true \
    algo=ppo_roa_train \
    task=G1/hdmi/move_suitcase \
    wandb.mode=online \
    headless=$HEADLESS \
    task.command.data_path=data/motion/g1/omomo/sub1_suitcase_011 \
    task.num_envs=$NUM_ENVS \
    task.reward.object_tracking.eef_contact_exp.enabled=true \
    wandb.name=move_suitcase_originalmotion_contact

# Experiment 2: Original motion without contact
# uv run scripts/train.py \
#     algo=ppo_roa_train \
#     task=G1/hdmi/move_suitcase \
#     wandb.mode=online \
#     headless=$HEADLESS \
#     task.command.data_path=data/motion/g1/omomo/sub1_suitcase_011 \
#     task.num_envs=$NUM_ENVS \
#     task.reward.object_tracking.eef_contact_exp.enabled=false \
#     wandb.name=move_suitcase_originalmotion_nocontact

# Experiment 3: Retarget motion with contact
# uv run scripts/train.py \
#     algo=ppo_roa_train \
#     task=G1/hdmi/move_suitcase \
#     wandb.mode=online \
#     headless=$HEADLESS \
#     task.command.data_path=data/retarget/output \
#     task.num_envs=$NUM_ENVS \
#     task.reward.object_tracking.eef_contact_exp.enabled=true \
#     wandb.name=move_suitcase_retargetmotion_contact

# Experiment 4: Retarget motion without contact
# uv run scripts/train.py \
#     algo=ppo_roa_train \
#     task=G1/hdmi/move_suitcase \
#     wandb.mode=online \
#     headless=$HEADLESS \
#     task.command.data_path=data/retarget/output \
#     task.num_envs=$NUM_ENVS \
#     task.reward.object_tracking.eef_contact_exp.enabled=false \
#     wandb.name=move_suitcase_retargetmotion_nocontact

# Experiment 5: Retarget motion without contact, with joint ctrl
# uv run scripts/train.py \
#     algo=ppo_roa_train \
#     task=G1/hdmi/move_suitcase \
#     wandb.mode=online \
#     headless=$HEADLESS \
#     task.command.data_path=data/retarget/output \
#     task.num_envs=$NUM_ENVS \
#     task.reward.object_tracking.eef_contact_exp.enabled=false \
#     task.observation.ref_joint_pos_.ref_joint_pos_action_policy.use_joint_ctrl=true \
#     wandb.name=move_suitcase_retargetmotion_nocontact_forwardJointCtrl

# Experiment 6: Retarget motion without contact, with joint ctrl and smaller action scale
# uv run scripts/train.py \
#     algo=ppo_roa_train \
#     task=G1/hdmi/move_suitcase \
#     wandb.mode=online \
#     headless=$HEADLESS \
#     task.command.data_path=data/retarget/output \
#     task.num_envs=$NUM_ENVS \
#     task.reward.object_tracking.eef_contact_exp.enabled=false \
#     task.observation.ref_joint_pos_.ref_joint_pos_action_policy.use_joint_ctrl=true \
#     '~task.action.action_scaling' \
#     '+task.action.action_scaling={.*: 0.2}' \
#     wandb.name=move_suitcase_retargetMotion_noContact_forwardJointCtrl_02actionscale

# cheatsheet:
    # replay motion:
    # '+task.command.replay_motion=true'
    # disable termination:
    # '~task.termination' \
    # '+task.termination.cum_body_pos_error={body_names: "torso_link", min_steps: 25, threshold: 100.0}'