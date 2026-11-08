"""Configuration and MDP unit tests for Mjlab-Run-Flat-MicroDuck and its Backlash twin."""

import math
import re
from types import SimpleNamespace

import mujoco
import pytest
import torch

from mjlab_microduck.robot.microduck_constants import (
    BACKLASH_HOME_FRAME,
    HOME_FRAME,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
    get_walk_backlash_spec,
    get_walk_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backlash import make_backlash_variant
from mjlab_microduck.tasks.microduck_run_env_cfg import (
    HEAD_BODY_NAMES,
    MAX_RUN_SPEED,
    NUM_STEPS_PER_ENV,
    SPRINT_HEAD_OFFSET,
    SPRINT_HEAD_STD,
    TARGET_FORWARD_LEAN_RAD,
    MicroduckRunRlCfg,
    make_microduck_run_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_run_obs_parity_with_velocity():
    """61D actor and 76D critic obs term order must match velocity env for ONNX hot-swap."""
    run_cfg = make_microduck_run_env_cfg()
    vel_cfg = make_microduck_velocity_env_cfg()
    for grp in ("actor", "critic"):
        assert list(run_cfg.observations[grp].terms.keys()) == list(
            vel_cfg.observations[grp].terms.keys()
        ), f"Observation layout mismatch in group {grp}"


def test_run_upstream_push_robot_is_removed():
    """Regression: ENABLE_VELOCITY_PUSHES=False must pop upstream quadruped push_robot."""
    cfg = make_microduck_run_env_cfg()
    assert "push_robot" not in cfg.events
    play_cfg = make_microduck_run_env_cfg(play=True)
    assert "push_robot" not in play_cfg.events


def test_run_bam_and_nan_guards_are_wired():
    """Standalone env must wire expand_bam_friction_fields, critic _safe terms, nan_state, and collapse terminations."""
    cfg = make_microduck_run_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
    for name in ("foot_contact_forces", "foot_height", "foot_air_time"):
        assert cfg.observations["critic"].terms[name].func.__name__.endswith("_safe"), (
            f"critic/{name} is missing its NaN guard"
        )
    assert cfg.terminations["nan_state"].params.get("sensor_names") == (
        "feet_ground_contact",
    )
    assert math.isclose(
        cfg.terminations["fell_over"].params["limit_angle"], math.radians(60.0)
    )
    assert cfg.terminations["base_too_low"].params["minimum_height"] == 0.065


def test_run_biomechanics_and_reward_signs():
    """Verify stride std progression, unified forward lean, anti-head-whip & chin-down gaze, and signs."""
    cfg = make_microduck_run_env_cfg()
    r = cfg.rewards

    # 1. Stride & 3D hip coordination: hip_yaw/hip_roll stay tight when standing (0.06 / 0.05),
    # then all 5 leg joint groups (hip_yaw, hip_roll, hip_pitch, knee, ankle) expand from
    # standing -> walking -> running so the swing foot can angle inward toward the midline
    # and the rear push-off foot can angle outward (Arena 2M Sprint style).
    std_s = r["pose"].params["std_standing"]
    std_w = r["pose"].params["std_walking"]
    std_r = r["pose"].params["std_running"]
    assert std_s[r".*hip_yaw.*"] == 0.06
    assert std_s[r".*hip_roll.*"] == 0.05
    for leg_key in (
        r".*hip_yaw.*",
        r".*hip_roll.*",
        r".*hip_pitch.*",
        r".*knee.*",
        r".*ankle.*",
    ):
        assert std_s[leg_key] < std_w[leg_key] < std_r[leg_key], leg_key
    assert r["pose"].params["walking_threshold"] == 0.05
    assert r["pose"].params["running_threshold"] == 0.45
    assert r["pose"].weight == 0.25

    # 2. Unified forward lean: both upright and forward_lean target TARGET_FORWARD_LEAN_RAD (0.30 rad ≈ 17.2 deg)
    assert r["upright"].func is microduck_mdp.sprint_lean_upright_reward
    assert r["upright"].params["target_pitch"] == TARGET_FORWARD_LEAN_RAD == 0.30
    assert r["forward_lean"].func is microduck_mdp.forward_lean_reward
    assert r["forward_lean"].params["target_pitch"] == TARGET_FORWARD_LEAN_RAD

    # 3. Anti-head-whip & chin-down gaze: strong head tracking with SPRINT_HEAD_OFFSET + neck_vel penalty
    assert r["head_pose_tracking"].weight >= 1.5
    assert r["head_pose_tracking"].params["default_offset"] == SPRINT_HEAD_OFFSET
    assert r["head_pose_tracking"].params["std"] == SPRINT_HEAD_STD
    assert r["neck_vel"].func is microduck_mdp.neck_joint_vel_l2
    assert "head_pose_bias" not in r

    # 4. Pitch/bounce-free horizontal velocity tracking, flight phase, & high knee-lift clearance
    assert r["track_linear_velocity"].func is microduck_mdp.track_horizontal_linear_velocity
    assert r["flight_phase"].func is microduck_mdp.flight_phase_reward
    assert r["foot_clearance"].params["target_height"] == 0.035
    assert r["foot_swing_height"].params["target_height"] == 0.035
    assert r["heading_hold"].func is microduck_mdp.heading_hold_reward
    assert r["forward_speed"].func is microduck_mdp.forward_speed_reward
    assert r["gait_symmetry"].func is microduck_mdp.gait_symmetry_penalty

    # 5. Sign convention: positive task rewards > 0, cost penalties strictly < 0
    for pos_name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "forward_speed",
        "flight_phase",
        "heading_hold",
        "forward_lean",
        "upright",
        "pose",
        "air_time",
        "head_pose_tracking",
    ):
        assert r[pos_name].weight > 0.0, pos_name
    assert r["body_pose_tracking"].weight == 0.0

    for cost_name in (
        "action_rate_l2",
        "neck_vel",
        "body_ang_vel",
        "angular_momentum",
        "foot_slip",
        "foot_clearance",
        "foot_swing_height",
        "self_collisions",
        "gait_symmetry",
        "termination_penalty",
        "dof_pos_limits",
    ):
        assert r[cost_name].weight < 0.0, cost_name


def test_run_curricula_five_stages_co_scheduled():
    """Verify 5-stage from-scratch sprint curriculum reaches 0.9–1.0 m/s with flight_phase ramp."""
    cfg = make_microduck_run_env_cfg()
    expected_steps = [
        0,
        1000 * NUM_STEPS_PER_ENV,
        2200 * NUM_STEPS_PER_ENV,
        3500 * NUM_STEPS_PER_ENV,
        4800 * NUM_STEPS_PER_ENV,
    ]

    vel_stages = cfg.curriculum["velocity_command_ranges"].params["velocity_stages"]
    flt_stages = cfg.curriculum["flight_phase_weight"].params["weight_stages"]
    act_stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]

    assert [s["step"] for s in vel_stages] == expected_steps
    assert [s["step"] for s in flt_stages] == expected_steps
    assert [s["step"] for s in act_stages] == expected_steps

    # Initial stage values match cfg initial values, final stage reaches (0.90, 1.00)
    assert vel_stages[0]["lin_vel_range"] == cfg.commands["twist"].ranges.lin_vel_x == (0.30, 0.50)
    assert vel_stages[-1]["lin_vel_range"] == (0.90, 1.00)
    assert MAX_RUN_SPEED == 1.00
    assert cfg.rewards["air_time"].params["threshold_min"] == 0.025  # Fixed 25 ms so high cadence is never penalized
    assert flt_stages[0]["weight"] == cfg.rewards["flight_phase"].weight == 0.5
    assert flt_stages[-1]["weight"] == 2.5
    assert act_stages[0]["weight"] == cfg.rewards["action_rate_l2"].weight == -0.02
    assert act_stages[-1]["weight"] == -0.10

    assert "air_time_threshold" not in cfg.curriculum
    assert "standing_envs" not in cfg.curriculum
    assert "head_pose_range" not in cfg.curriculum
    assert "head_pose_bias_weight" not in cfg.curriculum


def test_run_commands_straight_line_hide_arrows_and_play_mode():
    """Twist/head/body commands hide blue viewer arrow, keep normalizer alive, and lock to 1.0 m/s in play."""
    cfg = make_microduck_run_env_cfg()
    twist = cfg.commands["twist"]
    assert twist.debug_vis is False
    assert twist.heading_command is False
    assert twist.ranges.heading is None
    assert twist.rel_standing_envs == 0.02
    assert twist.rel_forward_envs == 0.5
    assert twist.rel_turn_in_place_envs == 0.0
    assert twist.ranges.lin_vel_x == (0.30, 0.50)
    assert twist.ranges.lin_vel_y[0] < 0.0 < twist.ranges.lin_vel_y[1]
    assert twist.ranges.ang_vel_z[0] < 0.0 < twist.ranges.ang_vel_z[1]
    assert cfg.commands["head_pose"].zero_command_prob == 0.5
    assert cfg.commands["body_pose"].zero_command_prob == 0.5

    play_cfg = make_microduck_run_env_cfg(play=True)
    assert play_cfg.commands["twist"].debug_vis is False
    assert play_cfg.commands["twist"].rel_standing_envs == 0.0
    assert play_cfg.commands["twist"].ranges.lin_vel_x == (1.00, 1.00)
    assert play_cfg.commands["twist"].ranges.lin_vel_y == (0.0, 0.0)
    assert play_cfg.commands["twist"].ranges.ang_vel_z == (0.0, 0.0)
    assert "velocity_command_ranges" not in play_cfg.curriculum
    assert "flight_phase_weight" not in play_cfg.curriculum


def test_run_sprint_head_offset_points_gaze_downward_in_mujoco():
    """Physics check: SPRINT_HEAD_OFFSET=(-0.75, -0.50, 0, 0) pitches neck deeply forward (>3 cm CoM shift) and tilts face downward."""
    model = get_walk_spec().compile()
    data_home = mujoco.MjData(model)
    data_sprint = mujoco.MjData(model)

    for d in (data_home, data_sprint):
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            qadr = model.jnt_qposadr[j]
            for pat, val in HOME_FRAME.joint_pos.items():
                if re.match(pat, j_name):
                    d.qpos[qadr] = val
                    break

    neck_names = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
    for name, delta in zip(neck_names, SPRINT_HEAD_OFFSET):
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data_sprint.qpos[model.jnt_qposadr[j_id]] += delta

    mujoco.mj_forward(model, data_home)
    mujoco.mj_forward(model, data_sprint)

    head_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    cam_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_camera")

    # 1. Deep forward neck extension shifts 280 g head CoM >3.0 cm forward (+x in trunk frame)
    assert data_sprint.xipos[head_body_id, 0] > data_home.xipos[head_body_id, 0] + 0.030

    # 2. Forward gaze axis (+x of head_camera site, which is horizontal 0.0 at HOME)
    # pitches downward (-z in world: -sin(0.25) ≈ -0.247) and up-axis (+z) tilts to cos(0.25) ≈ 0.969.
    cam_mat_home = data_home.site_xmat[cam_site_id].reshape(3, 3)
    cam_mat_sprint = data_sprint.site_xmat[cam_site_id].reshape(3, 3)
    forward_gaze_z_home = cam_mat_home[2, 0]
    forward_gaze_z_sprint = cam_mat_sprint[2, 0]
    assert math.isclose(forward_gaze_z_home, 0.0, abs_tol=1e-4)
    assert forward_gaze_z_sprint < -0.20  # Tilted ~14.3 deg downward vs horizontal HOME gaze


def test_run_rl_cfg_has_distinct_experiment_and_symmetry():
    assert MicroduckRunRlCfg.experiment_name == "run"
    assert MicroduckRunRlCfg.run_name == "run"
    assert MicroduckRunRlCfg.algorithm.symmetry_cfg is not None


def test_run_flat_and_backlash_tasks_are_registered():
    """Both Mjlab-Run-Flat-MicroDuck and Mjlab-Run-Flat-Backlash-MicroDuck are registered."""
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    from mjlab_microduck.tasks import _BACKLASH_TASKS

    tasks = list_tasks()
    run_tasks = [t for t in tasks if "Run" in t]
    assert sorted(run_tasks) == [
        "Mjlab-Run-Flat-Backlash-MicroDuck",
        "Mjlab-Run-Flat-MicroDuck",
    ]

    rows = [r for r in _BACKLASH_TASKS if r[0].startswith("Mjlab-Run-")]
    assert len(rows) == 1
    task_id, make_fn, kw, rl_cfg, robot_cfg = rows[0]
    assert task_id == "Mjlab-Run-Flat-Backlash-MicroDuck"
    assert make_fn is make_microduck_run_env_cfg
    assert kw == {}
    assert rl_cfg is MicroduckRunRlCfg
    assert robot_cfg is MICRODUCK_WALK_BACKLASH_ROBOT_CFG

    for play in (False, True):
        reg_bl = load_env_cfg("Mjlab-Run-Flat-Backlash-MicroDuck", play=play)
        assert reg_bl.scene.entities["robot"].spec_fn is get_walk_backlash_spec
    assert load_rl_cfg("Mjlab-Run-Flat-Backlash-MicroDuck").experiment_name == "run"


def test_run_backlash_variant_mirrors_walk_model_and_preserves_invariants():
    """Backlash twin must swap to MICRODUCK_WALK_BACKLASH_ROBOT_CFG, remap both
    actor/critic obs to encoder-through-backlash, preserve encoder-bias split,
    scope dof_pos_limits away from ±1° backlash stops, and leave base cfg unmutated."""
    base_cfg = make_microduck_run_env_cfg()
    orig_pose_joints = base_cfg.rewards["pose"].params["asset_cfg"].joint_names

    bl_cfg = make_backlash_variant(
        make_microduck_run_env_cfg(), MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    )
    vel_cfg = make_microduck_velocity_env_cfg()

    assert bl_cfg.scene.entities["robot"] is MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    assert bl_cfg.scene.entities["robot"].init_state is BACKLASH_HOME_FRAME

    for grp in ("actor", "critic"):
        assert list(bl_cfg.observations[grp].terms.keys()) == list(
            vel_cfg.observations[grp].terms.keys()
        ), f"Backlash obs layout mismatch in group {grp}"
        jp = bl_cfg.observations[grp].terms["joint_pos"]
        jv = bl_cfg.observations[grp].terms["joint_vel"]
        assert jp.func is microduck_mdp.joint_pos_rel_backlash
        assert jv.func is microduck_mdp.joint_vel_rel_backlash
        assert jp.params["asset_cfg"].joint_names == (r"^(?!passive_).*",)
        assert jv.params["asset_cfg"].joint_names == (r"^(?!passive_).*",)

    assert bl_cfg.observations["actor"].terms["joint_pos"].params["biased"] is True
    assert bl_cfg.observations["critic"].terms["joint_pos"].params["biased"] is False

    assert bl_cfg.rewards["dof_pos_limits"].params["asset_cfg"].joint_names == (
        r"^(?!passive_).*",
    )

    bl_pose_joints = bl_cfg.rewards["pose"].params["asset_cfg"].joint_names
    assert all("_backlash" in pat for pat in bl_pose_joints)
    assert base_cfg.rewards["pose"].params["asset_cfg"].joint_names == orig_pose_joints


@pytest.mark.parametrize("spec_fn", [get_walk_spec, get_walk_backlash_spec])
def test_run_selectors_resolve_on_walk_and_backlash_mjcf(spec_fn):
    """Compile actual walk and walk_backlash MJCFs on CPU to lock joint/geom/site/body selectors."""
    model = spec_fn().compile()
    is_backlash = spec_fn is get_walk_backlash_spec

    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    servo_pat = re.compile(r"^(?!passive_).*")
    servos = [n for n in articulated if servo_pat.match(n)]
    assert len(servos) == 14

    backlash_joints = [
        (j, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("passive_")
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).endswith("_backlash")
    ]
    if is_backlash:
        assert len(backlash_joints) == 14
    else:
        assert len(backlash_joints) == 0

    bl_cfg = make_backlash_variant(
        make_microduck_run_env_cfg(), MICRODUCK_WALK_BACKLASH_ROBOT_CFG
    )
    pose_pats = [re.compile(p) for p in bl_cfg.rewards["pose"].params["asset_cfg"].joint_names]
    pose_matched = [n for n in articulated if any(p.match(n) for p in pose_pats)]
    assert len(pose_matched) == 10  # 5 left leg + 5 right leg servos only
    assert all(not n.startswith("passive_") for n in pose_matched)

    geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)
    }
    assert {"left_foot_collision", "right_foot_collision"} <= geom_names

    site_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in range(model.nsite)
    }
    assert {"left_foot", "right_foot", "imu"} <= site_names

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)
    ]
    assert "trunk_base" in body_names
    for pat in HEAD_BODY_NAMES:
        rx = re.compile(f"^{pat}$")
        assert any(b and rx.match(b) for b in body_names), f"Unmatched head DR body: {pat}"


def test_mdp_velocity_command_ranges_curriculum_supports_tuple_windows():
    """Verify velocity_command_ranges_curriculum updates (lo, hi) windows without overwriting vy/wz."""
    ranges = SimpleNamespace(
        lin_vel_x=(0.30, 0.50),
        lin_vel_y=(-0.01, 0.01),
        ang_vel_z=(-0.02, 0.02),
    )
    cmd_term = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
    command_manager = SimpleNamespace(get_term=lambda name: cmd_term)

    stages = [
        {"step": 0, "lin_vel_range": (0.30, 0.50)},
        {"step": 1000 * 24, "lin_vel_range": (0.50, 0.70)},
        {"step": 2200 * 24, "lin_vel_range": (0.65, 0.85)},
        {"step": 3500 * 24, "lin_vel_range": (0.80, 0.95)},
        {"step": 4800 * 24, "lin_vel_range": (0.90, 1.00)},
    ]

    env = SimpleNamespace(
        device="cpu",
        common_step_counter=4800 * 24,
        command_manager=command_manager,
    )
    out = microduck_mdp.velocity_command_ranges_curriculum(
        env,
        env_ids=torch.tensor([0]),
        command_name="twist",
        velocity_stages=stages,
        update_lin_vel_y=False,
        update_ang_vel_z=False,
        forward_only=True,
    )
    assert ranges.lin_vel_x == (0.90, 1.00)
    assert ranges.lin_vel_y == (-0.01, 0.01)
    assert ranges.ang_vel_z == (-0.02, 0.02)
    assert torch.isclose(out, torch.tensor([1.00]))


def test_mdp_sprint_lean_upright_and_forward_lean():
    """Unit test sprint_lean_upright_reward and forward_lean_reward on CPU mock."""
    # env 0: leaning at target pitch (0.25), level roll (0.0) -> peaks at 1.0
    # env 1: standing vertical (0.0, 0.0) while commanded to run -> penalized (< 0.10)
    # env 2: standing vertical (0.0, 0.0) when idle (cmd_x = 0.0) -> 1.0
    g_b = torch.tensor(
        [
            [0.25, 0.0, -0.968],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    asset = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=g_b))
    cmds = torch.tensor([[0.9, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)

    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_command=lambda name: cmds),
    )

    slu = microduck_mdp.sprint_lean_upright_reward(
        env, command_name="twist", target_pitch=0.25, pitch_std=0.15, roll_std=0.12
    )
    assert torch.isclose(slu[0], torch.tensor(1.0), atol=1e-5)
    assert slu[1].item() < 0.10  # Upright fast-walking is no longer rewarded!
    assert torch.isclose(slu[2], torch.tensor(1.0), atol=1e-5)


def test_mdp_track_horizontal_velocity_and_flight_phase():
    """Verify track_horizontal_linear_velocity ignores pitch/vz bounce and flight_phase requires both feet airborne at speed."""
    pitch = 0.30
    quat_pitched = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
    quat_w = torch.tensor(
        [quat_pitched, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    v_w = torch.tensor(
        [
            [1.0, 0.0, 0.35],
            [0.0, 0.0, 0.40],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cmds = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    air_time = torch.tensor(
        [
            [0.04, 0.03],  # both feet airborne -> flight phase!
            [0.05, 0.05],  # both airborne, but vx=0 (hopping in place)
            [0.08, 0.00],  # single-support walk (right foot grounded)
        ],
        dtype=torch.float32,
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(root_link_lin_vel_w=v_w, root_link_quat_w=quat_w)
    )
    sensor = SimpleNamespace(data=SimpleNamespace(current_air_time=air_time))
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene={"robot": asset, "feet_ground_contact": sensor},
        command_manager=SimpleNamespace(get_command=lambda name: cmds),
    )

    trk = microduck_mdp.track_horizontal_linear_velocity(
        env, std=math.sqrt(0.08), command_name="twist"
    )
    assert torch.isclose(trk[0], torch.tensor(1.0), atol=1e-5)

    flt = microduck_mdp.flight_phase_reward(
        env, sensor_name="feet_ground_contact", command_name="twist", vel_ref=0.50
    )
    assert torch.isclose(flt[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(flt[1], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(flt[2], torch.tensor(0.0), atol=1e-5)
