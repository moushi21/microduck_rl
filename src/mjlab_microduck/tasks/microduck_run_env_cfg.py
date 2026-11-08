"""Microduck straight-line running environment (Mjlab-Run-Flat-MicroDuck).

High-speed straight-line forward running up to 1.0 m/s (~4.0 body lengths/s,
Froude number Fr ≈ 0.78) trained from scratch on flat terrain, registered both
as ``Mjlab-Run-Flat-MicroDuck`` (``robot_walk.xml``) and its ±1° serial-play
backlash twin ``Mjlab-Run-Flat-Backlash-MicroDuck`` (``robot_walk_backlash.xml``,
preserving the shared 61D actor observation and 14D action contract):

  - **Stride & 3D hip coordination (腰・膝・足首の蹴り方と歩幅 + 振り上げ足内旋・蹴り足外旋)**:
    speed-staged ``pose`` tolerances (light weight ``0.25``) keep ``std_standing``
    tight, then progressively open **both** sagittal joints (``hip_pitch`` 0.55 -> 0.75,
    ``knee`` 0.60 -> 0.80, ``ankle`` 0.45 -> 0.55) **and** 3D hip rotation
    (``hip_yaw`` 0.25 -> 0.35, ``hip_roll`` 0.12 -> 0.18) above
    ``running_threshold = 0.45 m/s``. This allows the forward-swinging foot to
    rotate slightly inward toward the CoM midline (reducing lateral waddle) and
    the rear push-off foot to rotate slightly outward (sprinter/skater-style 3D
    hip extension combining ``hip_pitch + hip_yaw + hip_roll``).
  - **Pronounced forward sprint lean (前傾姿勢)**: both ``upright``
    (``sprint_lean_upright_reward``) and ``forward_lean`` target a unified
    ``TARGET_FORWARD_LEAN_RAD = 0.30 rad`` (~17.2° nose-down trunk pitch) when
    moving forward, eliminating the pull toward vertical 0° pitch that caused
    upright fast-walking.
  - **Deep forward neck extension & anti-head-whip (首を前に倒す前傾重心・首振り抑制・顔面下向き)**:
    At ``HOME_FRAME``, ``neck_pitch = +0.35 rad (+20°)`` is tilted backward over the
    rear battery. ``head_pose_tracking`` (``weight = 2.0``) shifts the nominal target by
    ``SPRINT_HEAD_OFFSET = (-0.75, -0.50, 0.0, 0.0)`` so the 5 cm neck link pitches
    deeply forward (``neck_pitch ≈ -0.40 rad = -23°`` in trunk frame, ~-40° in world
    frame, shifting the 280 g head CoM >3.5 cm forward ahead of the hips) while
    ``head_pitch`` compensates so the face points ~14.3° downward relative to the
    trunk (~31° downward in world frame). Paired with tight yaw/roll stds
    ``(0.18, 0.18, 0.08, 0.08)`` and ``neck_joint_vel_l2`` (``weight = -0.08``) to
    suppress step-by-step lateral head whipping.
  - **Pitch- & bounce-free horizontal velocity tracking + aerial flight phase**:
    Replaces stock ``mdp.track_linear_velocity`` (which reads 3D tilted body-frame
    velocity and penalizes ``v_b,z²``, punishing both forward trunk pitch
    ``v_b,z = v_x·sin(θ)`` and vertical flight-phase bounce) with
    ``track_horizontal_linear_velocity`` in the horizontal yaw plane, paired with
    ``flight_phase_reward`` (rewarding both feet simultaneously airborne at speed)
    and a high 3.5 cm swing-foot target height (folding the swing knee to minimize
    leg rotational inertia ``I = m·r²``).
  - **Clean viewer**: blue velocity command arrow is hidden (``debug_vis = False``).
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24
MAX_RUN_SPEED = 1.00  # Target top sprint velocity (m/s)
TARGET_FORWARD_LEAN_RAD = 0.30  # ~17.2° forward trunk pitch during sprint
# Nominal head delta from HOME [neck_pitch, head_pitch, head_yaw, head_roll]:
# At HOME, neck_pitch=+0.349 rad (+20° tilted backward) and head_pitch=+0.349 rad.
# Delta (-0.75, -0.50, 0.0, 0.0) pitches the 5 cm neck link deeply forward to
# -0.40 rad (-23° in trunk frame, shifting the 280 g head CoM ~3.6 cm forward as
# seen in Microduck Arena 2M Sprint), while lifting head_pitch relative to the
# neck so net head pitch in trunk frame is (-0.50) - (-0.75) = +0.25 rad (+14.3°
# nose-down relative to trunk, ~28.6° downward in world frame during sprint).
SPRINT_HEAD_OFFSET = (-0.75, -0.50, 0.0, 0.0)
SPRINT_HEAD_STD = (0.18, 0.18, 0.08, 0.08)

# Straight-line running task — zero turn-in-place fraction.
TURN_IN_PLACE_FRACTION = 0.0

# Symmetry — straight-line running is bilaterally symmetric; PPO mirror loss
# enforces equal left/right stride mechanics and prevents veering.
ENABLE_SYMMETRY = True

# Domain randomization toggles (sim2real parity with velocity/jump)
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies
ENABLE_KP_RANDOMIZATION = False
ENABLE_KD_RANDOMIZATION = False
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed).
ENABLE_VELOCITY_PUSHES = False  # Disabled: lateral kicks disrupt straight-line sprint cadence discovery
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates IMU mounting errors
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias)
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False

# Head/body pose command resampling intervals (kept alive for 61D runtime obs parity).
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# Observation configuration
USE_PROJECTED_GRAVITY = True  # If True, use projected gravity instead of raw accelerometer

# Domain randomization ranges
COM_RANDOMIZATION_RANGE = 0.003  # ±3 mm initial, ramped to ±8 mm via curriculum
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck -> neck_pitch -> yaw_roll_motion -> head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. NOTE: bearing_roll is NOT a head body — in both models
# it is the right-hip-yaw link (child of trunk_base); it has always been listed
# here by mistake and is kept only to preserve existing DR behavior.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3 mm initial, ramped to ±8 mm via curriculum
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.98, 1.02)  # ±2% applied to BOTH mass and inertia together
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10%
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5% BAM friction scale
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±5% reflected rotor inertia
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 3.0  # up-to-3° zero-centered random-axis IMU mounting error
ENCODER_BIAS_RANGE = (-0.005, 0.005)  # ±0.005 rad (~±0.29°) per-joint encoder offset (constant per env)
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0

import mujoco as _mujoco
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


# Microduck-specific rough terrain: much gentler than the default ROUGH_TERRAINS_CFG.
# The robot can only lift its feet ~1-2 cm, so steps are capped at 1.5 cm.
MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),  # max 1.5 cm (vs 10 cm default)
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),  # max 1 cm
            platform_width=1.5,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)


def _soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
    """Soften terrain box geom contacts to reduce edge-contact NaN instability."""
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:
        geom.solref = [0.04, 1.0]
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]
        count += 1
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")


def make_microduck_run_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck straight-line running environment configuration."""

    std_standing = {
        # Tight posture when standing idle (command < walking_threshold)
        r".*hip_yaw.*": 0.06,
        r".*hip_roll.*": 0.05,  # Hold the 5°-inward stance (sole sits flat), stop leg splay
        r".*hip_pitch.*": 0.15,
        r".*knee.*": 0.15,
        r".*ankle.*": 0.10,
    }

    std_walking = {
        # Stage 0 moderate speed regime (0.05 <= speed < 0.45 m/s):
        # open sagittal joints for longer strides and allow moderate hip_yaw/hip_roll
        # rotation so the swing foot can angle toward the CoM midline.
        r".*hip_yaw.*": 0.25,
        r".*hip_roll.*": 0.12,
        r".*hip_pitch.*": 0.55,
        r".*knee.*": 0.60,
        r".*ankle.*": 0.45,
    }

    std_running = {
        # Stage 1–4 sprint regime (speed >= 0.45 m/s):
        # wide sagittal freedom for high-cadence stride extension, swing knee
        # folding, and ankle push-off, PLUS 3D hip rotation (hip_yaw=0.35, hip_roll=0.18)
        # so the forward swing foot turns slightly inward toward the midline and
        # the rear push-off foot turns outward (Microduck Arena 2M Sprint style).
        r".*hip_yaw.*": 0.35,
        r".*hip_roll.*": 0.18,
        r".*hip_pitch.*": 0.75,
        r".*knee.*": 0.80,
        r".*ankle.*": 0.55,
    }

    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Per-foot terrain-height ray sensor driving critic foot_height obs and swing clearance rewards.
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    foot_frictions_geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # Base configuration
    cfg = make_velocity_env_cfg()

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 1.0

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    # Pose reward: light weight (0.25) so HOME does not pull the robot back to an
    # upright standing stance; keeps hip_yaw/hip_roll tight while freeing sagittal joints.
    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_running
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.05
    cfg.rewards["pose"].params["running_threshold"] = 0.45
    cfg.rewards["pose"].weight = 0.25

    # Sprint-lean upright orientation: centers the pitch target at TARGET_FORWARD_LEAN_RAD
    # (0.25 rad ≈ 14.3° nose-down) when moving forward while keeping roll strictly level (0.0).
    # Eliminates the pull toward 0° vertical pitch that previously suppressed forward lean.
    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.sprint_lean_upright_reward,
        weight=1.5,
        params={
            "command_name": "twist",
            "target_pitch": TARGET_FORWARD_LEAN_RAD,
            "pitch_std": 0.15,
            "roll_std": 0.12,
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Forward lean reward: reinforces the same ~14.3° (0.25 rad) forward trunk pitch
    # proportional to commanded sprint speed so the CoM stays well ahead of stance push-off.
    cfg.rewards["forward_lean"] = RewardTermCfg(
        func=microduck_mdp.forward_lean_reward,
        weight=1.5,
        params={
            "command_name": "twist",
            "target_pitch": TARGET_FORWARD_LEAN_RAD,
            "std": 0.12,
        },
    )

    # Straight-line heading hold: angle-based reward against the spawn yaw so the
    # policy actively steers back if foot impact asymmetry causes slight yaw drift.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.5,
        params={"std": 0.30, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Neck angular-velocity damping: directly penalizes step-by-step lateral head
    # whipping (neck_pitch, head_pitch, head_yaw, head_roll velocities) so the 280 g
    # head stays steady and does not yank the trunk off course.
    cfg.rewards["neck_vel"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_vel_l2,
        weight=-0.08,
    )

    # Foot-specific configurations
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # Foot slip penalty: keeps stance push-off crisp without over-constraining transitions.
    cfg.rewards["foot_slip"].weight = -0.15
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.05

    cfg.rewards.pop("soft_landing", None)

    # Self-collision penalty
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Bilateral stride balance: penalizes cumulative left/right foot swing-time
    # imbalance |L - R| / (L + R) so the robot never resorts to one-legged limping.
    cfg.rewards["gait_symmetry"] = RewardTermCfg(
        func=microduck_mdp.gait_symmetry_penalty,
        weight=-1.0,
        params={"sensor_name": feet_ground_cfg.name},
    )

    # Terminal fall penalty: moderate (-10.0, not -50.0) so the policy is not
    # terrified of leaning aggressively forward during sprint acceleration.
    cfg.rewards["termination_penalty"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-10.0,
    )

    # Stride air-time reward: keep threshold_min fixed at 0.025 s (~1 step) so
    # high stride frequency f is NEVER penalized by the -f*t_min dead-zone loss.
    cfg.rewards["air_time"].weight = 1.5
    cfg.rewards["air_time"].params["command_threshold"] = 0.05
    cfg.rewards["air_time"].params["threshold_min"] = 0.025
    cfg.rewards["air_time"].params["threshold_max"] = 0.220

    # Aerial flight phase reward: directly rewards BOTH feet simultaneously
    # airborne while travelling forward at speed (the physical definition of
    # running vs race-walking, as seen in Microduck Arena 2M Sprint).
    cfg.rewards["flight_phase"] = RewardTermCfg(
        func=microduck_mdp.flight_phase_reward,
        weight=0.5,
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "vel_ref": 0.50,
            "command_threshold": 0.05,
        },
    )

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    # Primary velocity tracking rewards:
    # - Uses track_horizontal_linear_velocity in the yaw-aligned horizontal plane
    #   so neither forward trunk lean (v_b,z = v_x*sin(pitch)) nor vertical
    #   flight-phase bounce (v_w,z != 0) is penalized as z_error!
    # - Keeps track_angular_velocity std at sqrt(0.5) so its internal roll/pitch
    #   rate term does not choke dynamic sagittal push-off pitch oscillation.
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
        func=microduck_mdp.track_horizontal_linear_velocity,
        weight=5.0,
        params={"command_name": "twist", "std": math.sqrt(0.08)},
    )
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.50)

    cfg.rewards["forward_speed"] = RewardTermCfg(
        func=microduck_mdp.forward_speed_reward,
        weight=2.0,
        params={"vel_ref": 0.80},
    )

    # Action smoothness: starts light (-0.02) so explosive push-off discovery is
    # untaxed in Stage 0, then ramps to -0.10 by iter 4800 via curriculum.
    cfg.rewards["action_rate_l2"].weight = -0.02

    # Foot clearance & swing height: 3.5 cm target height (with softened -0.5
    # clearance weight) encourages high knee-folding during the forward recovery
    # swing (reducing leg rotational moment of inertia I = m*r^2 as seen in Arena
    # 2M Sprint) instead of penalizing foot lift above 1.5 cm.
    cfg.rewards["foot_clearance"].weight = -0.5
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.05
    cfg.rewards["foot_clearance"].params["target_height"] = 0.035

    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.05
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.035

    # === EVENTS ===
    # BAM writes per-env dof_frictionloss/dof_damping every step; register for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Terminate environments that hit numerical instability before NaNs reach rsl_rl.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    # 60° tilt limit allows deep forward sprint acceleration drive without premature reset.
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)
    # robot_walk.xml only has foot collision geoms; terminate if the trunk collapses below 6.5 cm.
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        time_out=False,
        params={"minimum_height": 0.065},
    )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # Velocity pushes: when disabled, explicitly remove the base template's push_robot event
    # (make_velocity_env_cfg ships a ±0.5 m/s quadruped kick enabled by default).
    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S

        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
    else:
        cfg.events.pop("push_robot", None)

    # Domain randomization — non-accumulating per-episode resets via mjlab 1.3.0 dr.* ops.
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_BASE_ORIENTATION_RANDOMIZATION:
        cfg.events["randomize_base_orientation"] = EventTermCfg(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )

    # === OBSERVATIONS (shared 61D actor layout for runtime ONNX hot-swap) ===
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Keep base_lin_vel in critic only (privileged information)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    if not USE_PROJECTED_GRAVITY:
        del cfg.observations["actor"].terms["projected_gravity"]
        cfg.observations["actor"].terms["raw_accelerometer"] = ObservationTermCfg(
            func=microduck_mdp.raw_accelerometer,
            scale=1.0,
        )

    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Sanitize sensor-derived critic terms against transient MuJoCo contact/raycast NaNs.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # Observation noise configuration
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (applied to actor only; critic sees true frame).
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel matching Dynamixel present_velocity firmware windowing.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude passive_* joints so joint_pos/vel obs match the 14 actuated servos.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Per-env constant encoder bias on actor joint_pos only.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # === COMMANDS ===
    # Straight-line forward velocity command:
    # - Stage 0 starts at lin_vel_x = (0.30, 0.50) and ramps to (0.90, 1.00) via curriculum.
    # - rel_forward_envs = 0.5 zeroes vy/wz on half the envs (pure straight-line command)
    #   while the other half samples tiny non-zero vy/wz noise so obs[49:51] normalizer
    #   std never collapses to eps.
    # - debug_vis = False hides the blue command arrow in the viewer.
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.02
    command.rel_heading_envs = 0.0
    command.rel_forward_envs = 0.5
    command.ranges.lin_vel_x = (0.30, 0.50)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.02, 0.02)
    command.debug_vis = False
    command.viz.z_offset = 0.0
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION
    cfg.commands["twist"].debug_vis = False

    # Head pose command: keep nominal command near zero (50% exact zero + tiny range for normalizer std).
    # SPRINT_HEAD_OFFSET in head_pose_tracking biases the physical target to a slight chin-down
    # forward neck lean while keeping head_yaw and head_roll locked straight ahead.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        zero_command_prob=0.5,
        ranges=(
            (-0.03, 0.03),    # neck_pitch
            (-0.03, 0.03),    # head_pitch
            (-0.03, 0.03),    # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )
    # Body pose command (6D slot kept alive with tiny ranges + 50% exact zero for 61D ONNX parity).
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        zero_command_prob=0.5,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
    )

    # Append head + body command obs terms in the canonical runtime order:
    # [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # === Pose tracking rewards ===
    # Strong head stabilization (weight=2.0) around SPRINT_HEAD_OFFSET=(-0.15, 0.10, 0.0, 0.0)
    # with tight yaw/roll std (0.08 rad ≈ 4.5°): prevents skyward gaze and lateral head whipping.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={
            "command_name": "head_pose",
            "std": SPRINT_HEAD_STD,
            "default_offset": SPRINT_HEAD_OFFSET,
        },
    )
    # body_pose: infra kept intact at weight 0 for 61D runtime compatibility.
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # Terrain
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        cfg.scene.spec_fn = _soften_terrain_contacts
        cfg.sim.nconmax = 200
        cfg.sim.mujoco.iterations = 30
        cfg.sim.mujoco.ls_iterations = 50

        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # === 5-STAGE FROM-SCRATCH SPRINT CURRICULUM (UP TO 1.0 m/s) ===
    # 1. Forward velocity window curriculum: progressively shifts the commanded
    #    speed band from [0.30, 0.50] up to full sprint [0.90, 1.00] m/s.
    cfg.curriculum["velocity_command_ranges"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "update_lin_vel_y": False,
            "update_ang_vel_z": False,
            "forward_only": True,
            "velocity_stages": [
                {"step": 0,                         "lin_vel_range": (0.30, 0.50)},  # Stage 0: fast stride discovery
                {"step": 1000 * NUM_STEPS_PER_ENV, "lin_vel_range": (0.50, 0.70)},  # Stage 1: transition to running
                {"step": 2200 * NUM_STEPS_PER_ENV, "lin_vel_range": (0.65, 0.85)},  # Stage 2: high-speed running
                {"step": 3500 * NUM_STEPS_PER_ENV, "lin_vel_range": (0.80, 0.95)},  # Stage 3: sprint acceleration
                {"step": 4800 * NUM_STEPS_PER_ENV, "lin_vel_range": (0.90, 1.00)},  # Stage 4: 0.9–1.0 m/s top-speed sprint
            ],
        },
    )

    # 2. Aerial flight-phase weight curriculum: ramps the reward for having BOTH
    #    feet simultaneously airborne at speed from 0.5 in Stage 0 up to 2.5 in
    #    Stage 4, while air_time.threshold_min stays fixed at 0.025 s so high
    #    stride frequency f is never penalized.
    cfg.curriculum["flight_phase_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "flight_phase",
            "weight_stages": [
                {"step": 0,                         "weight": 0.5},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 1.0},
                {"step": 2200 * NUM_STEPS_PER_ENV, "weight": 1.5},
                {"step": 3500 * NUM_STEPS_PER_ENV, "weight": 2.0},
                {"step": 4800 * NUM_STEPS_PER_ENV, "weight": 2.5},
            ],
        },
    )

    # 3. Action-rate smoothness curriculum: kept very light (-0.02) during Stage 0
    #    so dynamic push-off is not taxed before the gait exists, then tightened to -0.10.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                         "weight": -0.02},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.04},
                {"step": 2200 * NUM_STEPS_PER_ENV, "weight": -0.06},
                {"step": 3500 * NUM_STEPS_PER_ENV, "weight": -0.08},
                {"step": 4800 * NUM_STEPS_PER_ENV, "weight": -0.10},
            ],
        },
    )

    # 4. CoM randomization range curricula — start at ±3 mm and ramp to ±8 mm for robustness.
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,                         "range": 0.003},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "range": 0.005},
                    {"step": 3000 * NUM_STEPS_PER_ENV, "range": 0.008},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,                         "range": 0.003},
                    {"step": 1500 * NUM_STEPS_PER_ENV, "range": 0.005},
                    {"step": 3000 * NUM_STEPS_PER_ENV, "range": 0.008},
                ],
            },
        )

    # Disable default velocity curricula
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    if play:
        cfg.commands["twist"].debug_vis = False
        cfg.commands["twist"].rel_standing_envs = 0.0
        cfg.commands["twist"].ranges.lin_vel_x = (MAX_RUN_SPEED, MAX_RUN_SPEED)
        cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
        cfg.commands["body_pose"].ranges = ((0.0, 0.0),) * 6
        cfg.curriculum.pop("velocity_command_ranges", None)
        cfg.curriculum.pop("flight_phase_weight", None)

    return cfg


MicroduckRunRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="run",  # Directory name
    run_name="run",  # Appended to datetime in wandb: <datetime>_run
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=7_000,
)
