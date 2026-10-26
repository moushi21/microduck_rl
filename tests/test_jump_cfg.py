from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    STAND_Z,
    MicroduckJumpRlCfg,
    make_microduck_jump_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_jump_obs_parity_with_velocity():
    """61D actor/critic obs term order must match velocity env for ONNX hot-swap."""
    jump = make_microduck_jump_env_cfg()
    vel = make_microduck_velocity_env_cfg()
    for grp in ("actor", "critic"):
        assert list(jump.observations[grp].terms.keys()) == list(
            vel.observations[grp].terms.keys()
        ), f"Observation layout mismatch in group {grp}"


def test_jump_upstream_push_robot_is_removed():
    """Regression: ENABLE_VELOCITY_PUSHES=False must pop upstream quadruped push_robot."""
    cfg = make_microduck_jump_env_cfg()
    assert "push_robot" not in cfg.events
    play_cfg = make_microduck_jump_env_cfg(play=True)
    assert "push_robot" not in play_cfg.events


def test_jump_curricula_do_not_crush_takeoff_or_swing_head():
    """Regression: action_rate_weight must match initial action_rate_l2 (-0.01)
    and walking head_pose_range widening must be absent."""
    cfg = make_microduck_jump_env_cfg()
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["weight"] == cfg.rewards["action_rate_l2"].weight == -0.01
    assert stages[-1]["weight"] >= -0.08
    assert "head_pose_range" not in cfg.curriculum
    assert "head_pose_bias_weight" not in cfg.curriculum


def test_jump_reward_signs_and_stand_z():
    """Positive task terms > 0, cost terms (returning >= 0) strictly < 0."""
    cfg = make_microduck_jump_env_cfg()
    r = cfg.rewards
    assert STAND_Z == 0.115
    assert r["jump_height"].params["stand_height"] == STAND_Z
    assert r["air_time"].func is microduck_mdp.both_feet_air_time
    assert r["jump_height"].func is microduck_mdp.both_feet_jump_height
    assert r["base_lin_vel_z"].func is microduck_mdp.jump_takeoff_velocity
    assert r["single_foot_air"].func is microduck_mdp.single_foot_air_cost

    for pos_name in ("air_time", "jump_height", "base_lin_vel_z", "upright", "pose", "head_pose_tracking"):
        assert r[pos_name].weight > 0.0, pos_name
    assert r["head_pose_tracking"].weight <= 0.3

    for cost_name in (
        "single_foot_air",
        "termination_penalty",
        "base_xy_vel_penalty",
        "action_rate_l2",
        "body_ang_vel",
        "angular_momentum",
        "foot_slip",
        "self_collisions",
        "dof_pos_limits",
    ):
        assert r[cost_name].weight < 0.0, cost_name
    assert r["foot_slip"].params["command_threshold"] < 0.0


def test_jump_commands_keep_normalizer_alive_and_hide_arrows():
    cfg = make_microduck_jump_env_cfg()
    twist = cfg.commands["twist"]
    assert twist.debug_vis is False
    assert twist.heading_command is False
    assert twist.ranges.heading is None
    assert twist.rel_forward_envs < 0.0
    assert 0.0 < twist.rel_standing_envs < 1.0
    assert twist.ranges.lin_vel_x[0] < 0.0 < twist.ranges.lin_vel_x[1]

    play_cfg = make_microduck_jump_env_cfg(play=True)
    assert play_cfg.commands["twist"].debug_vis is False
    assert play_cfg.commands["twist"].rel_standing_envs == 1.0
    assert play_cfg.commands["twist"].ranges.lin_vel_x == (0.0, 0.0)


def test_jump_rl_cfg_has_distinct_experiment_and_symmetry():
    assert MicroduckJumpRlCfg.experiment_name == "jump"
    assert MicroduckJumpRlCfg.run_name == "jump"
    assert MicroduckJumpRlCfg.algorithm.symmetry_cfg is not None
