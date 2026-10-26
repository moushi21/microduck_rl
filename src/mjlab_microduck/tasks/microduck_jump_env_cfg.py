"""Microduck stationary vertical jump environment (Mjlab-Jump-Flat-MicroDuck).

Symmetric two-footed vertical jumping in place on flat terrain:
  - Unit-scaled dense simultaneous air-time + trunk-lift above STAND_Z (0.115 m),
    gated by uprightness and post-spawn settle (eliminates both the sparse-reward
    do-nothing basin and spawn-drop/tuck-collapse exploits).
  - Height-gated takeoff velocity bootstrap (only above 0.108 m, so ground
    knee-bobbing between 0.075 and 0.108 m earns strictly zero).
  - Single-foot air cost + bilateral PPO mirror loss (ENABLE_SYMMETRY=True) to
    enforce simultaneous two-foot takeoff and simultaneous two-foot landing.
  - Gentle post-discovery action-rate curriculum (-0.01 -> -0.06) and steady
    near-HOME head pose so the 280 g head stays centered over the support polygon.
"""

import math
from copy import deepcopy

NUM_STEPS_PER_ENV = 24
STAND_Z = 0.115  # Measured Microduck standing trunk height on flat ground (m)

# Fraction of envs commanded to spin on the spot (lin=0, |ang| ∈ [0.4·max, max]).
TURN_IN_PLACE_FRACTION = 0.0

# Symmetry — ジャンプは完全左右対称動作のため有効化
ENABLE_SYMMETRY = True

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies
ENABLE_KP_RANDOMIZATION = False # Was True
ENABLE_KD_RANDOMIZATION = False # Was True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed).
ENABLE_VELOCITY_PUSHES = False  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias)
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # Randomize initial tilt to force reactive behavior

# Head/body pose command tracking (replaces the old neck-offset disturbance scheme).
# Head pose: 4D deltas-from-HOME on neck/head joints; vel env tracks these as a
# primary objective. Body pose: 6D delta in [x, y, z, roll, pitch, yaw]; vel env
# samples small ranges + tiny reward weight so input neurons stay alive but
# tracking isn't the priority (standup env raises the weight).
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# Observation configuration
USE_PROJECTED_GRAVITY = True  # If True, use projected gravity instead of raw accelerometer

# Domain randomization ranges (adjust as needed)
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped to ±8mm via curriculum
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck → neck_pitch → yaw_roll_motion → head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. NOTE: bearing_roll is NOT a head body — in both models
# it is the right-hip-yaw link (child of trunk_base); it has always been listed
# here by mistake and is kept only to preserve existing DR behavior.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3mm initial, ramped via curriculum
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.98, 1.02)  # ±5% applied to BOTH mass and inertia together.
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # ±15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # ±10% (can increase to 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.95, 1.05)  # ±10% reflected rotor inertia (microban: dr.joint_armature, same range)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # Velocity change range in m/s. Was ±0.5 — an
# ADDITIVE kick larger than max walk speed (0.4) every 3-6 s trains a permanently
# nervous fall-recovery gait (2026-07 audit). ±0.3 keeps push robustness while
# letting a calmer gait be optimal.
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 3.0  # up-to-6° random-axis IMU mounting error. NOTE: zero-centered (random axis) — trains tolerance to misalignment *magnitude*, NOT a pitch bias. The real board's systematic ~5° pitch offset is corrected at the source in the runtime (imu-pitch-offset), not here.
ENCODER_BIAS_RANGE = (-0.005, 0.005)  # ±0.86° per-joint encoder offset (constant per env)
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # ±10° forward/backward tilt at episode start
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # ±5° side-to-side tilt at episode start

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
        # NOTE: BoxInvertedPyramidStairsTerrainCfg removed — it sets env_origin_z to the pit
        # bottom (negative), causing resets at root_z = 0.12 + env_origin_z ≈ −0.10 m which
        # places the robot below the pit floor and makes it fall through the ground.
        # Uneven cobblestone-like ground: random per-cell height offsets.
        # grid_width=0.12 on an 8m patch = 66×66 = 4 356 boxes/patch → ~261 K total → OOM.
        # 0.45 m gives 17×17 = 289 boxes/patch → ~17 K total (border = 0.35 m ✓).
        # Must not divide evenly into terrain size (8.0 m): 0.45 × 17 = 7.65 ✓
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),  # max 1 cm
            platform_width=1.5,
        ),
        # Gentle slopes (heightfield pyramid, platform on TOP — robot spawns on
        # the flat platform and walks down/up/across the slope as commands
        # resample). slope_range is rise/run: 0.03→0.10 ≈ 1.7°→5.7° by
        # difficulty — small robot, small slopes. NOT inverted (see the
        # inverted-pyramid env_origin note above — same pit-spawn risk class).
        # vertical_scale=0.001 keeps quantization steps at 1 mm so a gentle
        # slope is smooth instead of a staircase of 5 mm ledges.
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
    """Soften terrain box geom contacts to reduce edge-contact NaN instability.

    Box terrains place adjacent geoms at different heights. The hard edges where
    heights change cause contact normal instability when feet land on them, which
    can produce impulsive NaN forces in the MuJoCo solver.

    Doubling the solref time constant (0.02 → 0.04 s) makes contact springs
    2× softer — enough to damp the instability without noticeably changing the
    macro-level walking physics. Applied to all geoms in the "terrain" body,
    which contains every box generated by TerrainGenerator.
    """
    body = spec.body("terrain")
    count = 0
    for geom in body.geoms:
        geom.solref = [0.04, 1.0]   # 2× softer time constant (default: 0.02)
        geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # slightly softer impedance
        count += 1
    print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")


def make_microduck_jump_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck stationary vertical jump environment configuration."""

    std_standing = {
        # hip_yaw / hip_roll は厳格にHOME付近へ拘束して開脚・ねじれを防ぎつつ、
        # hip_pitch / knee / ankle は大きく緩めて深い屈伸と跳躍・着地吸収を許容する
        r".*hip_yaw.*": 0.08,
        r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.80,
        r".*knee.*": 0.90,
        r".*ankle.*": 0.60,
    }

    std_walking = std_standing

    site_names = ["left_foot", "right_foot"]

    # Contact sensor for feet - LEFT, RIGHT order
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT foot first, RIGHT foot second
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

    # mjlab 1.3.0: foot_height obs + foot_clearance/foot_swing_height rewards are
    # now driven by a per-foot terrain-height ray sensor (was site_pos based).
    # Mirrors microban's foot_height_scan.
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
    cfg.episode_length_s = 8.0  # 複数回のジャンプ＋着地後の姿勢立て直しまで一貫して学習させるため8秒に設定

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
    # Pose reward configuration
    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 10.0  # 常に std_standing を適用
    # hip_yaw / hip_roll のねじれ・開脚を抑えつつ、膝・股関節・足首の深い屈伸は許容する
    cfg.rewards["pose"].weight = 0.2

    # Body-specific reward configurations
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    # 直立維持報酬: 静止しているだけで大黒字になるのを防ぐため 0.5 に抑制（空中・着地時の姿勢制御用）
    cfg.rewards["upright"].weight = 0.5
    cfg.rewards["upright"].params["std"] = math.sqrt(0.15)

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # command=0 でも接地中の足滑りを常時抑止するため command_threshold = -1.0 に設定
    cfg.rewards["foot_slip"].weight = -0.2
    cfg.rewards["foot_slip"].params["command_threshold"] = -1.0

    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # 滞空時間報酬: 両足が「同時に」浮いている時間に単位正規化（Unit-scaled）で連続加点
    cfg.rewards["air_time"] = RewardTermCfg(
        func=microduck_mdp.both_feet_air_time,
        weight=12.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "threshold_min": 0.0,
            "threshold_max": 0.50,
            "min_com_z": 0.100,
        },
    )

    # 跳躍高度報酬: 両足が同時離地している時【限定】で STAND_Z (0.115 m) からの上昇高度を加点
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=microduck_mdp.both_feet_jump_height,
        weight=15.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "stand_height": STAND_Z,
            "ref_lift": 0.03,
        },
    )

    # 片足浮きペナルティ: 足踏み・片足ホップ・左右バラバラの離着地を直接抑制（両足接地 or 両足同時滞空なら 0）
    cfg.rewards["single_foot_air"] = RewardTermCfg(
        func=microduck_mdp.single_foot_air_cost,
        weight=-1.5,
        params={"sensor_name": feet_ground_cfg.name},
    )

    # 転倒・潰れ終了への即時ペナルティ（1回大きく飛んで着地崩壊するカミカゼ特攻ハックを防止）
    cfg.rewards["termination_penalty"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-80.0,
    )

    # 飛び立ちや空中の姿勢ブレを許容（過度なペナルティで静止が勝つのを防止）
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.01

    # 水平移動（歩行・助走・前後滑り）のペナルティ
    cfg.rewards["base_xy_vel_penalty"] = RewardTermCfg(
        func=microduck_mdp.base_xy_velocity_penalty,
        weight=-1.0,
    )

    # Velocity tracking rewards
    cfg.rewards["track_linear_velocity"].weight = 0.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.2)
    # 静止で毎ステップ稼ぐのを防ぐため重みを極小化 (0.2 → 0.05)
    cfg.rewards["track_angular_velocity"].weight = 0.05
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # Action smoothness — 初期探索時は極めて軽く開始し、ジャンプ習得後にカリキュラムで徐々に引き締める
    cfg.rewards["action_rate_l2"].weight = -0.01

    # 歩行専用の足スイング報酬は無効化
    cfg.rewards["foot_clearance"].weight = 0.0
    cfg.rewards["foot_swing_height"].weight = 0.0

    # Z軸（垂直上方向）への蹴り出しブートストラップ（0.108m以上の伸び上がり区間のみ加点し、床での高速屈伸ハックを封じる）
    cfg.rewards["base_lin_vel_z"] = RewardTermCfg(
        func=microduck_mdp.jump_takeoff_velocity,
        weight=3.0,
        params={
            "min_height": 0.108,
            "max_height": 0.175,
            "max_vz": 1.2,
        },
    )

    # NOTE: no neck-only action-rate term — the shared action_rate_l2 sums over
    # ALL action dims (neck included), and head_pose_tracking below gives the
    # 4 neck/head DOFs a position objective, so the neck is fully shaped.

    # Events
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
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
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # Grippier footpad — narrowed from (0.3, 1.2)
    # Terminate environments that have gone numerically unstable (NaN physics).
    # MuJoCo can produce NaN joint positions on extreme contact impulses.
    # Terminating immediately resets to a valid state before NaN propagates
    # into the observation buffer and corrupts network weights.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )
    # 45°以上の傾き・転倒で即座にエピソード終了
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(45.0)
    # robot_walk.xml は足裏のみ衝突判定を持つため、胴体が 6.5cm 未満まで潰れたら即リセット（床貫通・座り込み防止）
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        time_out=False,
        params={"minimum_height": 0.065},
    )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # Velocity-based pushes: ENABLE_VELOCITY_PUSHES=False の時はベース環境の巨大外乱 push_robot を必ず削除する！
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

    # Domain randomization — re-sampled per episode at reset. In mjlab 1.3.0 the
    # stock dr.* ops with operation="add"/"scale" read from the compile-time
    # default field each reset (Operation.uses_defaults=True), so they are
    # NON-accumulating natively — this upstream behavior replaces microduck's old
    # custom restore-then-add functions that worked around the accumulation footgun.
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
        # Randomize the CoM of the head assembly bodies (per-body fresh offset each reset).
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
        # Randomize motor PD gains
        # Uses custom function that handles DelayedActuator
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
        # Physics-consistent mass + inertia randomization via mjlab's pseudo_inertia:
        # alpha scales BOTH mass and inertia by e^(2*alpha) with the CoM unchanged
        # (so it does NOT conflict with randomize_com). alpha_range is derived from
        # the ±5% mass scale range: e^(2*alpha) ∈ [0.95, 1.05].
        # Replaces the old custom randomize_mass_and_inertia, which was a silent
        # no-op under mjlab 1.3.0 (direct per-env body_mass/body_inertia writes are
        # not expanded and collapse to a single shared value). Startup mode = fixed
        # per env for the whole run (standard for mass DR; no accumulation).
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
        # Joint-friction DR under BAM: scales BAM's velocity-independent friction
        # budget (Coulomb + Stribeck + load) per-env via the FrictionDRBamActuator
        # friction_scale hook. MuJoCo's dof_frictionloss is zeroed under BAM, so the
        # stock dr.dof_frictionloss is a no-op — this is the BAM-native path.
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_JOINT_DAMPING_RANDOMIZATION:
        # Randomize joint damping (lubrication, temperature effects).
        # Custom non-accumulating scaler. NOTE: no-op under BAM (dof_damping
        # zeroed in edit_spec); only affects the XML position actuator.
        cfg.events["randomize_joint_damping"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",  # required by domain_randomization=True
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Randomize reflected rotor inertia (armature), microban-exact
        # (dr.joint_armature, scale, ±10%). Non-accumulating (uses_defaults). DOES
        # affect the BAM actuator — BAM sets dof_armature (~0.0018), it isn't zeroed.
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # IMU orientation randomization (mounting error) is applied at the OBSERVATION
    # level below (per-env constant rotation of projected_gravity + base_ang_vel).
    # The old event-based randomize_imu_orientation wrote site_quat, which under
    # mjlab 1.3.0 is neither per-env expanded nor read by these obs — a no-op.

    # Base orientation randomization (forces reactive behavior)
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

    # Observations
    del cfg.observations["actor"].terms["base_lin_vel"]
    # mjlab 1.3.0 adds a height_scan term (terrain ray scan) to both groups by
    # default. The microduck has no such body-mounted terrain sensor for the
    # policy, so drop it from both (mirrors microban).
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # Add base_lin_vel to critic only (privileged information)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # Determine gravity/accelerometer term name based on flag
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # Replace projected_gravity with raw_accelerometer if flag is False
    if not USE_PROJECTED_GRAVITY:
        # Remove projected_gravity and add raw_accelerometer
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
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64

    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1  # was 3 (=60 ms worst case); real dxl IMU path is fast — ±20 ms envelope (2026-07 audit)
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # The critic's sensor-derived terms are the one obs path `nan_state` cannot
    # protect (it checks joint + root state; these read raycast/contact sensor
    # data, which MuJoCo can return non-finite for while the state is still
    # clean). A single NaN here kills the whole run via rsl_rl's check_nan —
    # that is the 2026-08-21 Velocity2-Rough-Backlash crash. Critic-only, so
    # sanitizing costs the policy nothing.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # Observation noise configuration (edit these values as needed)
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03) # was 0.2
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)  # was 0.15
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)  # was 0.05
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)  # was 2.0

    # IMU mounting-misalignment DR (per-env constant rotation of the IMU-derived
    # observations). Applied to the ACTOR only (the policy sees a slightly rotated
    # IMU frame, like a real mounting error); the critic keeps the true values.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        if USE_PROJECTED_GRAVITY:
            g = cfg.observations["actor"].terms[gravity_term_name]
            g.func = microduck_mdp.projected_gravity_imu_misaligned
            g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel: the Dynamixel firmware computes
    # present_velocity via a moving-average over the previous position-sample
    # window, so the value the policy actually reads is ~1 control period old.
    # Matches reality and stops the policy relying on instantaneous qdot feedback.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude passive_* joints (jaw linkage) from joint_pos/vel obs so the
    # observation dim matches the action dim (14) instead of the raw articulation (16).
    # Deepcopy each joint_pos/joint_vel term first — actor and critic share the
    # same term objects/params dicts from the base template, so mutating one would
    # leak into the other (e.g. the encoder-bias `biased` flag below).
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR: the base template samples a per-env constant joint-encoder
    # offset (startup event "encoder_bias"), but joint_pos_rel ignores it unless
    # biased=True. Feed the biased joint pos to the ACTOR only (what the real
    # encoders report); the critic keeps the true joint pos (privileged).
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Commands — deepcopy to avoid shared-state corruption from other env cfgs.
    # AGENTS.md invariant: keep 50% exact-zero idle command + 50% tiny non-zero noise
    # so obs[48:51] normalizer std never collapses to eps while idle [0,0,0] is trained.
    command: UniformVelocityCommandCfg = deepcopy(cfg.commands["twist"])
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.5
    command.rel_heading_envs = 0.0
    command.rel_forward_envs = -1.0  # r.uniform_(0,1) <= -1.0 が常にFalseになり前進上書き・青矢印を完全根絶
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.02, 0.02)
    command.debug_vis = False
    command.viz.z_offset = 0.0
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    cfg.commands["twist"].rel_turn_in_place_envs = 0.0
    cfg.commands["twist"].debug_vis = False

    # Head pose command: 280gの頭部をHOME付近に安定保持（微小ノイズ + 50%完全ゼロ）
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
    # Body pose command (6D slot kept alive with tiny ranges + 50% exact zero for 61D ONNX parity)
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

    # Append head + body command obs terms to both policy and critic groups.
    # Order matters for the runtime obs layout: [twist(3), head_pose(4), body_pose(6)].
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
    # 頭部をHOME付近に穏やかに保つ（旧 weight=2.0 は「棒立ちで+2.0/sの大黒字」になっていたため 0.2 に縮小）
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.2,
        params={"command_name": "head_pose", "std": 0.4},
    )
    # body_pose: infra kept intact but DISABLED (weight 0) — the obs slot and
    # command stay alive for runtime 61D compatibility.
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

    # action_rate_l2 カリキュラム:
    # 旧設定（0iter=-0.10 → 1500iter=-0.30）がジャンプの爆発的蹴り出しを完全に潰していたため、
    # 初期は -0.01 で自由に跳躍を発見させ、習得後に -0.06 まで穏やかに引き締める。
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                         "weight": -0.01},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.02},
                {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -0.04},
                {"step": 3000 * NUM_STEPS_PER_ENV, "weight": -0.06},
            ],
        },
    )

    # NOTE: 歩行用の head_pose_range カリキュラム（頭を±80°振り回す外乱）および
    # head_pose_bias_weight はジャンプ中の重心崩壊を招くため意図的に除外する。

    # CoM randomization range curriculum
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 1500 * 24, "range": 0.005},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 1500 * 24, "range": 0.005},
                ],
            },
        )

    # Disable default velocity curricula
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # 学習の進展に合わせて必要滞空時間のボーナス基準（threshold_min）を段階的に引き上げ
    cfg.curriculum["jump_height_curriculum"] = CurriculumTermCfg(
        func=microduck_mdp.reward_param_curriculum,
        params={
            "reward_name": "air_time",
            "param_name": "threshold_min",
            "stages": [
                {"step": 0,                         "value": 0.0},   # 0~1000 iter: 少しでも両足同時離地すれば即加点
                {"step": 1000 * NUM_STEPS_PER_ENV, "value": 0.04},  # 1000~2000 iter: 0.04s ボーナス
                {"step": 2000 * NUM_STEPS_PER_ENV, "value": 0.08},  # 2000~3000 iter: 0.08s ボーナス
                {"step": 3000 * NUM_STEPS_PER_ENV, "value": 0.11},  # 3000~4000 iter: 0.11s ボーナス
                {"step": 4000 * NUM_STEPS_PER_ENV, "value": 0.14},  # 4000+ iter:     0.14s フルジャンプボーナス
            ],
        },
    )

    if play:
        cfg.commands["twist"].debug_vis = False
        cfg.commands["twist"].rel_standing_envs = 1.0
        cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
        cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
        cfg.commands["body_pose"].ranges = ((0.0, 0.0),) * 6
        cfg.curriculum.pop("jump_height_curriculum", None)

    return cfg


MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="jump",
    run_name="jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
