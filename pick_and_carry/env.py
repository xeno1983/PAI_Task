"""Vectorized pick-and-carry environment on Genesis.

One scene holds ``num_envs`` copies. ``reset`` returns observations of shape
``(num_envs, num_obs)`` and ``step`` returns ``(obs, reward, done, info)``.

The task math lives outside this file so it can be read without the simulator:
``observation.pack_observation``, ``reward.task_reward``, and
``control`` for the leg, arm, and gripper targets.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from pick_and_carry.arms import get_arm
from pick_and_carry.config import (
    DEFAULT_LEG_ANGLES,
    FOOT_LINKS,
    LEG_JOINTS,
    EnvConfig,
    layout_dim,
    layout_slices,
    leg_limit,
)
from pick_and_carry.control import arm_gripper_command, gripper_open_amount, learned_leg_target
from pick_and_carry.cpg import LegCPG
from pick_and_carry.curriculum import (
    BALL_RADIUS,
    BALL_STAGES,
    BALL_VELOCITY_SCALE,
    CATCH_LEVELS,
    FLYING_LEVELS,
    LOCOMOTION_STAGES,
    PREDICTION_SCALE,
    STAGES,
    STAND_STILL_SHARE,
    WALK_COMMAND_SECONDS,
    holds_arm_pose,
    ball_in_base,
    feet_air_time_reward,
    locomotion_reward,
    manipulation_reward,
    sample_ball,
    sample_command,
    sample_reach_target,
    stage_layouts,
)
from pick_and_carry.helpers import (
    detect_wrist_target,
    fleet_camera_height,
    fleet_view_indices,
    grasp_mask,
    gripper_is_closed,
    pairwise_distance,
    task_phase,
    quat_apply,
    quat_conjugate,
    world_to_base,
    yaw_quat,
)
from pick_and_carry.ik import (
    CARRY_POINT_IN_BASE,
    GRASP_HEIGHT_OFFSET,
    PLACE_HEIGHT_OFFSET,
    base_to_mount,
    solve_ik,
)
from pick_and_carry.observation import object_in_base, pack_observation
from pick_and_carry.reward import task_reward
from pick_and_carry.ros_bridge import clamp_goal_xy, downsample_rgb
from pick_and_carry.terrain import make_course, make_supply_yard, make_terrain_bank, paste_tile, sample_height


# Thighs and calves hitting the ground or a wall count as collisions. The base counts as a fall.
BASE_LINK = "base"
COLLISION_LINKS = tuple(f"{leg}_{part}" for leg in ("FL", "FR", "RL", "RR") for part in ("thigh", "calf"))
COLLISION_FORCE = 0.1


def _joint_dof_index(joint) -> int:
    local = getattr(joint, "dofs_idx_local", None)
    if local is not None and len(local) > 0:
        return int(local[0])
    if hasattr(joint, "dof_idx_local"):
        return int(joint.dof_idx_local)
    return int(joint.dof_start)


def _link_index(link) -> int:
    for attr in ("idx_local", "local_idx"):
        if hasattr(link, attr):
            return int(getattr(link, attr))
    raise AttributeError("Link has no local index attribute.")


def _as_batch(tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.ndim == 0:
        raise RuntimeError(f"Expected a batched tensor, got shape {tuple(tensor.shape)}")
    if tensor.ndim >= 1 and tensor.shape[0] == num_envs:
        return tensor
    if num_envs == 1:
        return tensor.unsqueeze(0)
    raise RuntimeError(f"Cannot broadcast shape {tuple(tensor.shape)} to num_envs={num_envs}")


def _on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.dtype != torch.float32 or tensor.device != device:
        return tensor.to(device=device, dtype=torch.float32)
    return tensor


def _rgb_uint8(rgb) -> np.ndarray:
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    image = np.asarray(rgb)
    if image.ndim == 4:
        image = image[0]
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class PickAndCarryEnv:
    """Go2 base with a selectable arm fixed on its back, plus a cube and a goal."""

    def __init__(self, num_envs: int | None = None, show_viewer: bool | None = None, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        if num_envs is not None:
            self.cfg.num_envs = num_envs
        if show_viewer is not None:
            self.cfg.show_viewer = show_viewer
        if self.cfg.num_envs < 1:
            raise ValueError("num_envs must be >= 1 so every tensor keeps a batch dimension.")
        if self.cfg.curriculum not in STAGES:
            raise ValueError(f"curriculum must be one of {', '.join(STAGES)}, got {self.cfg.curriculum}.")
        if self.cfg.catch_level not in CATCH_LEVELS:
            raise ValueError(f"catch_level must be one of {', '.join(CATCH_LEVELS)}, got {self.cfg.catch_level}.")

        self.num_envs = self.cfg.num_envs
        self.arm = get_arm(self.cfg.arm)
        self.cfg.arm_action_scale = self.arm.action_scale
        self.cfg.arm_kp = self.arm.kp
        self.cfg.arm_kd = self.arm.kd
        self.cfg.gripper_kp = self.arm.gripper_kp
        self.cfg.gripper_kd = self.arm.gripper_kd
        self.cfg.gripper_open = abs(self.arm.joint_open - self.arm.joint_closed)
        self.cfg.gripper_closed = self.arm.closed_threshold
        self.camera = None
        self.fleet_camera = None
        self.wrist_camera = None
        self._fleet_indices: list[int] = []
        self._recording = False
        self._wrist_warning_sent = False
        self._overview_warning_sent = False
        obs_layout, action_layout = stage_layouts(self.cfg.curriculum, self.arm.dof)
        self.num_obs = layout_dim(obs_layout)
        self.num_actions = layout_dim(action_layout)
        self.obs_slices = layout_slices(obs_layout)
        self.action_slices = layout_slices(action_layout)
        self.max_episode_length = self.cfg.max_episode_length
        self._contact_warning_sent = False

        import genesis as gs

        backend = gs.gpu if self.cfg.backend == "gpu" else gs.cpu

        def _init_genesis(with_seed: bool) -> None:
            if with_seed:
                gs.init(backend=backend, precision="32", seed=self.cfg.seed)
            else:
                gs.init(backend=backend, precision="32")

        def _already_initialized(exc: Exception) -> bool:
            message = str(exc).lower()
            return "already" in message

        try:
            _init_genesis(True)
        except TypeError:
            try:
                _init_genesis(False)
            except Exception as exc:
                if not _already_initialized(exc):
                    raise
        except Exception as exc:
            if not _already_initialized(exc):
                raise
        self.device = torch.device(gs.device if not isinstance(gs.device, torch.device) else gs.device)
        torch.manual_seed(self.cfg.seed)

        if self.cfg.urdf_path:
            self.urdf_path = self.cfg.urdf_path
        else:
            self.urdf_path = str(self.arm.urdf_path)
        if self.cfg.fleet_snapshot and not self.cfg.wrist_camera and self.num_envs > 1:
            self._fleet_indices = fleet_view_indices(self.num_envs)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.cfg.dt, substeps=self.cfg.substeps),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=self.cfg.max_collision_pairs,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=self._rendered_env_indices(),
                shadow=False,
                segmentation_level="link",
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.2, -2.2, 1.4),
                camera_lookat=(0.4, 0.0, 0.4),
                camera_fov=40,
            ),
            show_viewer=self.cfg.show_viewer,
        )
        self._terrain_grid = None
        self._terrain_origin = (0.0, 0.0)
        self._terrain_scale = 1.0
        self._tile_cols = 0
        self._tile_stride = 0
        self._tile_shifts = None
        self._terrain_rng = np.random.default_rng(self.cfg.seed)
        self._terrain_mix = 1.0
        self._visual_map = None
        self.terrain = None
        self._yard_pickups = None
        self._yard_goals = None
        if self.cfg.terrain and self.cfg.yard_stations > 0:
            height, origin, scale, pickups, goals = make_supply_yard(
                self.cfg.yard_stations,
                self.cfg.terrain_amplitude,
                self.cfg.terrain_horizontal_scale,
            )
            shifts = np.zeros((self.num_envs, 2), dtype=np.float32)
            cols = height.shape[1]
            stride = cols
            self._yard_pickups = pickups
            self._yard_goals = goals
        elif self.cfg.terrain:
            height, origin, scale, shifts, cols, stride = make_terrain_bank(
                self.num_envs,
                self.cfg.terrain_amplitude,
                self.cfg.terrain_horizontal_scale,
                self._terrain_rng,
                steps=self.cfg.terrain_steps,
            )
        if self.cfg.terrain:
            self._terrain_grid = height
            self._terrain_origin = origin
            self._terrain_scale = scale
            self._tile_cols = cols
            self._tile_stride = stride
            self._tile_shifts = torch.as_tensor(shifts, dtype=torch.float32, device=self.device)
            self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                    height_field=height,
                    horizontal_scale=scale,
                    vertical_scale=1.0,
                    pos=(origin[0], origin[1], 0.0),
                ),
                surface=gs.surfaces.Default(color=(0.45, 0.42, 0.36, 1.0)),
            )
        else:
            self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.urdf_path,
                pos=self.cfg.base_init_pos,
                quat=self.cfg.base_init_quat,
                fixed=False,
                # Fixed joints are merged by default. Feet and the gripper frame must survive that merge.
                links_to_keep=[*FOOT_LINKS, self.arm.tcp_link],
            )
        )
        cube = self.cfg.cube_size
        supply_colors = (
            (0.95, 0.35, 0.15, 1.0),
            (0.20, 0.45, 0.95, 1.0),
            (0.95, 0.80, 0.15, 1.0),
            (0.20, 0.75, 0.40, 1.0),
            (0.70, 0.30, 0.85, 1.0),
            (0.95, 0.55, 0.15, 1.0),
            (0.15, 0.70, 0.75, 1.0),
            (0.85, 0.25, 0.40, 1.0),
        )
        supply_count = self.cfg.yard_stations if self.cfg.yard_stations > 0 else 1
        self.supplies = []
        for index in range(supply_count):
            self.supplies.append(
                self.scene.add_entity(
                    morph=gs.morphs.Box(size=(cube, cube, cube), pos=(0.4, 0.0, cube * 0.5), fixed=False),
                    material=gs.materials.Rigid(rho=self.cfg.cube_density),
                    surface=gs.surfaces.Default(color=supply_colors[index % len(supply_colors)]),
                )
            )
        self.cube = self.supplies[0]
        self.ball = None
        if self.cfg.curriculum in BALL_STAGES:
            # In reach the ball only marks the target point: it is pinned in the air and the hand passes through.
            self.ball = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    radius=BALL_RADIUS,
                    pos=(0.8, 0.0, BALL_RADIUS),
                    fixed=False,
                    collision=self.cfg.curriculum == "catch",
                ),
                material=gs.materials.Rigid(rho=180.0),
                surface=gs.surfaces.Default(color=(0.85, 0.15, 0.15, 1.0)),
            )
        marker = self.cfg.goal_marker_size
        self.goal_marker = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(marker[0], marker[1], 0.40),
                pos=(1.0, 0.0, 0.20),
                fixed=False,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(0.15, 0.85, 0.30, 0.85)),
        )
        if self.cfg.wrist_camera:
            self.wrist_camera = self.scene.add_camera(
                res=self.cfg.wrist_camera_res,
                pos=(0.0, 0.0, 0.5),
                lookat=(0.2, 0.0, 0.4),
                fov=self.cfg.wrist_camera_fov,
                GUI=False,
            )
        if self._fleet_indices:
            # Drawn once at startup. The pose is moved over the block after the grid offsets exist.
            self.fleet_camera = self.scene.add_camera(
                res=(1280, 720),
                pos=(0.0, 0.0, 8.0),
                lookat=(0.0, 0.0, 0.2),
                fov=50,
                # Replaced with the fitted distance before the one snapshot. The default 20 m
                # clips a wide block and the picture is only the background color.
                far=2000.0,
                GUI=False,
            )
        if self.cfg.record_path or self.cfg.enable_camera:
            self.camera = self.scene.add_camera(
                res=self.cfg.camera_res,
                pos=self.cfg.camera_pos,
                lookat=self.cfg.camera_lookat,
                fov=self.cfg.camera_fov,
                GUI=False,
            )
        self.scene.build(n_envs=self.num_envs, env_spacing=self.cfg.env_spacing)
        if self.terrain is not None:
            self._prepare_terrain_runtime()
        if self.wrist_camera is not None:
            # The tcp link is kept through the fixed-joint merge. The offset looks out through the fingers.
            self.wrist_camera.attach(self.robot.get_link(self.arm.tcp_link), self.arm.wrist_offset())
            self._supply_link_idx = [int(cube.links[0].idx) for cube in self.supplies]
        if self.camera is not None and self.cfg.record_path:
            Path(self.cfg.record_path).parent.mkdir(parents=True, exist_ok=True)
            self.camera.start_recording(save_to_filename=self.cfg.record_path, fps=25)
            self._recording = True

        self.leg_names = list(LEG_JOINTS)
        self.arm_names = list(self.arm.joints)
        self.finger_names = [name for name in self.arm.fingers if self._has_joint(name)]
        if len(self.finger_names) == 0:
            raise RuntimeError(f"URDF {self.urdf_path} has none of the gripper joints {self.arm.fingers}.")
        self.actuated_names = self.leg_names + self.arm_names + self.finger_names
        self.dof_idx = [_joint_dof_index(self.robot.get_joint(name)) for name in self.actuated_names]
        self.ee_link_name = self._resolve_ee_link()
        self.ee_link_idx = _link_index(self.robot.get_link(self.ee_link_name))
        self.foot_idx = [_link_index(self.robot.get_link(name)) for name in FOOT_LINKS]
        self.base_link_idx = _link_index(self.robot.get_link(BASE_LINK))
        self.collision_idx = [_link_index(self.robot.get_link(name)) for name in COLLISION_LINKS]

        self._configure_pd()
        self._allocate_buffers()
        self.extras: dict = {}
        self._randomize_friction()
        self.reset()

    def _has_joint(self, name: str) -> bool:
        try:
            self.robot.get_joint(name)
        except Exception:
            return False
        return True

    def _resolve_ee_link(self) -> str:
        for name in self.arm.ee_links:
            try:
                self.robot.get_link(name)
                return name
            except Exception:
                continue
        raise RuntimeError(
            f"None of the end-effector links {self.arm.ee_links} exist in {self.urdf_path}."
        )

    def _configure_pd(self) -> None:
        kp, kd, lower, upper = [], [], [], []
        for name in self.leg_names:
            lo, hi, effort = leg_limit(name)
            kp.append(self.cfg.leg_kp)
            kd.append(self.cfg.leg_kd)
            lower.append(-effort)
            upper.append(effort)
        for name in self.arm_names:
            _lo, _hi, effort = self.arm.limits[name]
            kp.append(self.cfg.arm_kp)
            kd.append(self.cfg.arm_kd)
            lower.append(-effort)
            upper.append(effort)
        for name in self.finger_names:
            kp.append(self.cfg.gripper_kp)
            kd.append(self.cfg.gripper_kd)
            effort = self.arm.limits[name][2]
            lower.append(-effort)
            upper.append(effort)
        self.robot.set_dofs_kp(kp, self.dof_idx)
        self.robot.set_dofs_kv(kd, self.dof_idx)
        self.robot.set_dofs_force_range(lower, upper, self.dof_idx)
        self.leg_lower = self._tensor([leg_limit(name)[0] for name in self.leg_names])
        self.leg_upper = self._tensor([leg_limit(name)[1] for name in self.leg_names])
        self.arm_lower = self._tensor([self.arm.limits[name][0] for name in self.arm_names])
        self.arm_upper = self._tensor([self.arm.limits[name][1] for name in self.arm_names])
        self.default_leg = self._tensor([DEFAULT_LEG_ANGLES[name] for name in self.leg_names])
        self.default_arm = self._tensor([self.arm.default_angles[name] for name in self.arm_names])
        self.leg_effort = self._tensor([leg_limit(name)[2] for name in self.leg_names])
        self.arm_effort = self._tensor([self.arm.limits[name][2] for name in self.arm_names])

    def _tensor(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _allocate_buffers(self) -> None:
        n = self.num_envs
        self.obs_buf = torch.zeros((n, self.num_obs), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.done_buf = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self.actions = torch.zeros((n, self.num_actions), dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros((n, self.num_actions), dtype=torch.float32, device=self.device)
        # Expanded 12-D leg joint actions (from CPG); used in rewards / last_leg_action obs.
        self.leg_joint_actions = torch.zeros((n, len(LEG_JOINTS)), dtype=torch.float32, device=self.device)
        self.prev_leg_joint_actions = torch.zeros((n, len(LEG_JOINTS)), dtype=torch.float32, device=self.device)
        self.cpg = LegCPG(
            n,
            dt=self.cfg.dt,
            leg_action_scale=self.cfg.leg_action_scale,
            device=self.device,
        )
        # When True, skip auto command resampling (teacher / locked demos own ``command``).
        self.hold_commands = False
        self.q_ik = self.default_arm.expand(n, -1).clone()
        self.ik_error = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros((n,), dtype=torch.int32, device=self.device)
        self.episode_return = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.base_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.base_quat = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        self.base_lin_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.base_ang_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.projected_gravity = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.dof_pos = torch.zeros((n, len(self.actuated_names)), dtype=torch.float32, device=self.device)
        self.dof_vel = torch.zeros((n, len(self.actuated_names)), dtype=torch.float32, device=self.device)
        self.ee_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.obj_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.goal_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.foot_contact = torch.zeros((n, len(FOOT_LINKS)), dtype=torch.float32, device=self.device)
        # Locomotion bookkeeping: how long each foot has been up, last step's leg speed and torque.
        self.feet_air_time = torch.zeros((n, len(FOOT_LINKS)), dtype=torch.float32, device=self.device)
        self.last_contact = torch.zeros((n, len(FOOT_LINKS)), dtype=torch.bool, device=self.device)
        self.air_time_reward = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.prev_leg_vel = torch.zeros((n, len(self.leg_names)), dtype=torch.float32, device=self.device)
        self.leg_torque = torch.zeros((n, len(self.leg_names)), dtype=torch.float32, device=self.device)
        self.body_contact = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self.leg_collisions = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.push_timer = torch.zeros((n,), dtype=torch.int32, device=self.device)
        self.grasped = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self.prev_grasped = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self.was_grasped = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self.place_latched = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self.prev_ee_dist = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.prev_goal_dist = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.prev_base_dist = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.phase = torch.zeros((n,), dtype=torch.long, device=self.device)
        self._supply_index = torch.zeros((n,), dtype=torch.long, device=self.device)
        self.supply_finished = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self._release_hold = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self._goal_pose = torch.zeros((len(self.supplies), 3), dtype=torch.float32, device=self.device)
        self.base_obj_dist = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.obj_goal_dist = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.wrist_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.wrist_visible = torch.zeros((n,), dtype=torch.bool, device=self.device)
        self.command = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.ball_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.ball_vel = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.base_lin_vel_world = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.hold_steps = torch.zeros((n,), dtype=torch.int32, device=self.device)
        # Walk moves the arm at random: where it is heading, where it is now, and when to pick a new target.
        self.arm_goal = self.default_arm.expand(n, -1).clone()
        self.arm_cmd = self.default_arm.expand(n, -1).clone()
        self.arm_timer = torch.zeros((n,), dtype=torch.int32, device=self.device)
        # Reach pins the ball here, in world coordinates.
        self.reach_target = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.command_timer = torch.zeros((n,), dtype=torch.int32, device=self.device)
        self._ball_velocity_warning = False
        self.gravity = torch.tensor((0.0, 0.0, -1.0), dtype=torch.float32, device=self.device)
        self.init_base_pos = self._tensor(self.cfg.base_init_pos)
        self.init_base_quat = self._tensor(self.cfg.base_init_quat)
        self.reward_terms: dict[str, torch.Tensor] = {}

    def reset(self) -> torch.Tensor:
        self._reset_idx(None)
        self._read_state()
        self._update_grasp_flags()
        self._update_phase()
        self._capture_distances(None)
        self._write_observation()
        return self.obs_buf

    def step(self, action: torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"action must have shape {(self.num_envs, self.num_actions)}, got {tuple(action.shape)}")
        self.actions = action.clamp(-self.cfg.action_clip, self.cfg.action_clip)
        self._push_robots()
        self._apply_action(self.actions)
        self.scene.step()
        self.episode_length += 1
        self._read_state()
        self._pin_goal(None)
        self._pin_ball()
        self._update_grasp_flags()
        self._assist_grasp()
        self._update_phase()
        self._update_air_time()

        reward, terms, newly_placed, fallen = self.compute_reward()
        if self.cfg.cycle_supplies and self.cfg.curriculum == "carry" and bool(newly_placed.any()):
            self._advance_supplies(newly_placed)
            newly_placed = torch.zeros_like(newly_placed)
        self.rew_buf = reward
        self.reward_terms = terms
        self.episode_return += reward
        grasped_now = self.grasped.clone()
        was_grasped_now = self.was_grasped.clone()

        timeout = self.episode_length >= self.max_episode_length
        missed = self._ball_missed()
        if self.cfg.curriculum in LOCOMOTION_STAGES:
            success = timeout & ~fallen
            done = fallen | timeout
        elif self.cfg.curriculum == "haul":
            dropped = ~self.grasped
            success = timeout & ~fallen & self.grasped
            done = fallen | timeout | dropped
        else:
            success = newly_placed
            done = success | fallen | timeout | missed
        terminal = success | fallen | missed
        if self.cfg.curriculum == "haul":
            terminal = terminal | ~self.grasped
        self.done_buf = done
        self.extras = {
            "success": success,
            "fallen": fallen,
            "timeout": timeout & ~terminal,
            "terminal": terminal,
            "time_outs": (timeout & ~terminal).to(dtype=torch.float32),
            "reward_terms": {name: value.detach() for name, value in terms.items()},
            "ik_error": self.ik_error.detach(),
        }
        if bool(done.any()):
            self.extras["episode_return"] = self.episode_return[done].mean().detach()
            self.extras["success_rate"] = success[done].to(dtype=torch.float32).mean().detach()
            self._reset_idx(done)
            self._read_state()
            self.grasped = torch.where(done, torch.zeros_like(grasped_now), grasped_now)
            self.was_grasped = torch.where(done, torch.zeros_like(was_grasped_now), was_grasped_now)
            self._update_phase()
            self._capture_distances(done)
            self.dof_vel = torch.where(done[:, None], 0.0, self.dof_vel)
            self.base_lin_vel = torch.where(done[:, None], 0.0, self.base_lin_vel)
            self.base_ang_vel = torch.where(done[:, None], 0.0, self.base_ang_vel)
            self.foot_contact = torch.where(done[:, None], 0.0, self.foot_contact)
        if self.cfg.curriculum in LOCOMOTION_STAGES:
            self._refresh_commands()
        self.prev_actions = torch.where(done[:, None], torch.zeros_like(self.actions), self.actions)
        self.prev_leg_joint_actions = torch.where(
            done[:, None],
            torch.zeros_like(self.leg_joint_actions),
            self.leg_joint_actions,
        )
        self.prev_leg_vel = self.dof_vel[:, : len(self.leg_names)].clone()
        self.prev_grasped = torch.where(done, torch.zeros_like(grasped_now), grasped_now)
        self._write_observation()
        return self.obs_buf, self.rew_buf, self.done_buf, self.extras

    def compute_reward(self):
        """Return the staged reward, per-term values, place events, and falls."""
        if self.cfg.curriculum in LOCOMOTION_STAGES:
            reward, terms, fallen = locomotion_reward(
                base_lin_vel=self.base_lin_vel,
                base_ang_vel=self.base_ang_vel,
                command=self.command,
                projected_gravity=self.projected_gravity,
                base_height=self.base_pos[:, 2],
                leg_actions=self.leg_joint_actions,
                prev_leg_actions=self.prev_leg_joint_actions,
                leg_vel=self.dof_vel[:, : len(self.leg_names)],
                prev_leg_vel=self.prev_leg_vel,
                leg_torque=self.leg_torque,
                foot_contact=self.foot_contact,
                air_time_reward=self.air_time_reward,
                collisions=self.leg_collisions,
                body_contact=self.body_contact,
                max_tilt_deg=self.cfg.max_tilt_deg,
                min_base_height=self.cfg.min_base_height,
                dt=self.cfg.dt,
                scales=self.cfg.locomotion_scales,
            )
            return reward, terms, torch.zeros_like(fallen), fallen
        if self.cfg.curriculum == "haul" or self.cfg.curriculum in BALL_STAGES:
            ball_stage = self.cfg.curriculum in BALL_STAGES
            target_pos = self.ball_pos if ball_stage else self.obj_pos
            target_vel = self.ball_vel if ball_stage else torch.zeros_like(self.obj_pos)
            closed = gripper_is_closed(self._gripper_opening(), self.cfg.gripper_closed)
            reward, terms, caught, fallen, distance = manipulation_reward(
                stage=self.cfg.curriculum,
                ee_pos=self.ee_pos,
                object_pos=target_pos,
                object_vel=target_vel,
                base_lin_vel_world=self.base_lin_vel_world,
                base_lin_vel=self.base_lin_vel,
                base_ang_vel=self.base_ang_vel,
                command=self.command,
                projected_gravity=self.projected_gravity,
                base_height=self.base_pos[:, 2],
                grasped=self.grasped,
                closed=closed,
                actions=self.actions,
                prev_distance=self.prev_ee_dist,
                max_tilt_deg=self.cfg.max_tilt_deg,
                min_base_height=self.cfg.min_base_height,
                dt=self.cfg.dt,
            )
            self.prev_ee_dist = distance
            if ball_stage:
                self.hold_steps = torch.where(caught, self.hold_steps + 1, torch.zeros_like(self.hold_steps))
                newly = self.hold_steps >= self.cfg.catch_hold_steps
            else:
                newly = torch.zeros_like(fallen)
            return reward, terms, newly, fallen
        (
            reward,
            terms,
            newly_placed,
            fallen,
            self.place_latched,
            self.prev_ee_dist,
            self.prev_goal_dist,
            self.prev_base_dist,
        ) = task_reward(
            ee_pos=self.ee_pos,
            obj_pos=self.obj_pos,
            goal_pos=self.goal_pos,
            base_pos=self.base_pos,
            grasped=self.grasped,
            prev_grasped=self.prev_grasped,
            was_grasped=self.was_grasped,
            place_latched=self.place_latched,
            projected_gravity=self.projected_gravity,
            phase=self.phase,
            prev_ee_dist=self.prev_ee_dist,
            prev_goal_dist=self.prev_goal_dist,
            prev_base_dist=self.prev_base_dist,
            base_obj_dist=self.base_obj_dist,
            base_lin_vel=self.base_lin_vel,
            base_ang_vel=self.base_ang_vel,
            gripper_opening=self._gripper_opening(),
            actions=self.actions,
            cfg=self.cfg,
        )
        return reward, terms, newly_placed, fallen

    def _update_phase(self) -> None:
        self.base_obj_dist = torch.linalg.norm(self.base_pos[:, :2] - self.obj_pos[:, :2], dim=-1)
        self.obj_goal_dist = torch.linalg.norm(self.obj_pos[:, :2] - self.goal_pos[:, :2], dim=-1)
        self.phase = task_phase(
            self.grasped,
            self.base_obj_dist,
            self.obj_goal_dist,
            self.cfg.pickup_distance,
            self.cfg.release_distance,
        )
        if self.cfg.curriculum == "haul":
            self.phase.fill_(2)
        elif self.cfg.curriculum in (*LOCOMOTION_STAGES, *BALL_STAGES):
            self.phase.zero_()
        if self.cfg.cycle_supplies:
            # The hand reaches forward of the body, so pick up and set down only
            # when that target is in front. Otherwise keep walking and turning.
            obj_ahead, obj_aside = self._body_offset(self.obj_pos)
            goal_ahead, goal_aside = self._body_offset(self.goal_pos)
            lined_up = (
                ~self.grasped
                & ~self.supply_finished
                & (obj_ahead > 0.26)
                & (obj_ahead < 0.44)
                & (obj_aside.abs() < 0.12)
            )
            self.phase = torch.where((~self.grasped) & ~lined_up, torch.zeros_like(self.phase), self.phase)
            self.phase = torch.where(lined_up, torch.ones_like(self.phase), self.phase)
            carrying = self.grasped & ~self.supply_finished & ~self._release_hold
            self.phase = torch.where(carrying, torch.full_like(self.phase, 2), self.phase)
            near_pad = (
                self.grasped
                & ~self.supply_finished
                & (goal_ahead > 0.22)
                & (goal_ahead < 0.55)
                & (goal_aside.abs() < 0.24)
            )
            self._release_hold |= near_pad
            self._release_hold &= self.grasped & ~self.supply_finished
            self.phase = torch.where(self._release_hold, torch.full_like(self.phase, 3), self.phase)

    def _capture_distances(self, envs_idx) -> None:
        target = self.ball_pos if self.cfg.curriculum in BALL_STAGES else self.obj_pos
        dist_ee = pairwise_distance(self.ee_pos, target)
        dist_goal = pairwise_distance(self.obj_pos, self.goal_pos)
        dist_base = torch.linalg.norm(self.base_pos[:, :2] - self.obj_pos[:, :2], dim=-1)
        if envs_idx is None:
            self.prev_ee_dist.copy_(dist_ee)
            self.prev_goal_dist.copy_(dist_goal)
            self.prev_base_dist.copy_(dist_base)
            return
        idx = self._index(envs_idx)
        if idx.numel() == 0:
            return
        self.prev_ee_dist[idx] = dist_ee[idx]
        self.prev_goal_dist[idx] = dist_goal[idx]
        self.prev_base_dist[idx] = dist_base[idx]

    def _apply_action(self, action: torch.Tensor) -> None:
        cpg_act = action[:, self.action_slices["cpg"]]
        arm_act = action[:, self.action_slices["arm"]]
        grip_act = action[:, self.action_slices["gripper"]]
        leg_act = self.cpg.step(cpg_act)
        self.leg_joint_actions = leg_act
        leg_target = self._leg_command(leg_act)
        self._estimate_leg_torque(leg_target)
        if holds_arm_pose(self.cfg.curriculum):
            arm_target = self._held_arm_target()
            finger_target = self._rest_fingers(action.shape[0])
        else:
            q_ik, ik_err, ee_near = self._solve_arm_ik()
            self.q_ik = q_ik
            self.ik_error = ik_err
            arm_target, opening = arm_gripper_command(
                arm_act,
                grip_act,
                self.default_arm,
                self.phase,
                control_split=self.cfg.control_split,
                arm_scale=self.cfg.arm_action_scale,
                gripper_open=self.arm.joint_open,
                gripper_closed=self.arm.joint_closed,
                lower=self.arm_lower,
                upper=self.arm_upper,
                q_ik=q_ik,
                grip_residual_scale=self.cfg.gripper_residual_scale,
                curriculum=self.cfg.curriculum,
                ee_near=ee_near,
            )
            finger_target = opening.unsqueeze(-1).expand(-1, len(self.finger_names))
        target = torch.cat((leg_target, arm_target, finger_target), dim=-1)
        self.robot.control_dofs_position(target, self.dof_idx)

    def _arm_ik_target_world(self) -> torch.Tensor:
        """World-frame EE position the IK should chase (one row per env)."""
        stage = self.cfg.curriculum
        n = self.num_envs
        if stage == "reach":
            return self.reach_target if self.reach_target is not None else self.ball_pos
        if stage == "catch":
            return self.ball_pos
        # haul / carry: phase-dependent
        carry = self.base_pos + quat_apply(
            self.base_quat,
            torch.tensor(CARRY_POINT_IN_BASE, device=self.device, dtype=torch.float32).expand(n, -1),
        )
        pick = self.obj_pos.clone()
        pick[:, 2] = pick[:, 2] + GRASP_HEIGHT_OFFSET
        place = self.goal_pos.clone()
        place[:, 2] = place[:, 2] + PLACE_HEIGHT_OFFSET
        target = carry.clone()
        target = torch.where((self.phase == 1).unsqueeze(-1), pick, target)
        target = torch.where((self.phase == 3).unsqueeze(-1), place, target)
        return target

    def _solve_arm_ik(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q_ik, error_norm, ee_near)`` for residual arm control."""
        world = self._arm_ik_target_world()
        base = world_to_base(world, self.base_pos, self.base_quat)
        mount = base_to_mount(base)
        # Warm-start from current arm joints when available.
        arm_slice = slice(len(self.leg_names), len(self.leg_names) + len(self.arm_names))
        q0 = self.dof_pos[:, arm_slice]
        if q0.shape[-1] != len(self.arm_names):
            q0 = self.q_ik
        q_ik, err = solve_ik(q0, mount, self.arm_lower, self.arm_upper)
        # Fall back to the previous solution when the solve stays far from the target.
        bad = err > 0.08
        if bool(bad.any()):
            q_ik = torch.where(bad.unsqueeze(-1), self.q_ik, q_ik)
        ee_dist = pairwise_distance(self.ee_pos, world)
        ee_near = ee_dist < self.cfg.grasp_distance
        return q_ik, err, ee_near

    def _pin_ball(self) -> None:
        """Reach: keep the marker ball on its target point."""
        if self.cfg.curriculum != "reach" or self.ball is None:
            return
        self.ball.set_pos(self.reach_target, zero_velocity=True)
        self.ball_pos = self.reach_target.clone()

    def _ball_missed(self) -> torch.Tensor:
        """A thrown ball that has landed can no longer be caught, so the throw ends."""
        none = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if self.cfg.curriculum != "catch" or self.cfg.catch_level not in FLYING_LEVELS:
            return none
        return (self.ball_pos[:, 2] < BALL_RADIUS + 0.015) & (self.episode_length > 5)

    def _held_arm_target(self) -> torch.Tensor:
        """Arm targets when the policy does not drive the arm: the reset pose, or a random wander."""
        if not self.cfg.arm_disturb:
            return self.default_arm.expand(self.num_envs, -1)
        self.arm_timer += 1
        due = self.arm_timer >= max(1, int(self.cfg.arm_disturb_interval_s / self.cfg.dt))
        if bool(due.any()):
            span_up = (self.arm_upper - self.default_arm).clamp(min=0.0)
            span_down = (self.default_arm - self.arm_lower).clamp(min=0.0)
            unit = torch.rand((self.num_envs, len(self.arm_names)), device=self.device) * 2.0 - 1.0
            offset = torch.where(unit > 0, unit * span_up, unit * span_down) * self.cfg.arm_disturb_scale
            self.arm_goal = torch.where(due[:, None], self.default_arm + offset, self.arm_goal)
            self.arm_timer = torch.where(due, torch.zeros_like(self.arm_timer), self.arm_timer)
        self.arm_cmd = self.arm_cmd + (self.arm_goal - self.arm_cmd) * self.cfg.arm_disturb_rate
        return self.arm_cmd

    def _initial_push_timer(self, count: int) -> torch.Tensor:
        """Start each robot at a random point of the push cycle so the pushes do not all land together."""
        steps = max(1, int(self.cfg.push_interval_s / self.cfg.dt))
        return torch.randint(0, steps, (count,), dtype=torch.int32, device=self.device)

    def _update_air_time(self) -> None:
        if self.cfg.curriculum not in LOCOMOTION_STAGES:
            return
        self.air_time_reward, self.feet_air_time, self.last_contact = feet_air_time_reward(
            self.feet_air_time,
            self.last_contact,
            self.foot_contact,
            self.command,
            self.cfg.dt,
            self.cfg.locomotion_scales.feet_air_time_target,
        )

    def _estimate_leg_torque(self, leg_target: torch.Tensor) -> None:
        """PD torque the simulator will apply this step, limited by each joint's effort."""
        leg = slice(0, len(self.leg_names))
        torque = self.cfg.leg_kp * (leg_target - self.dof_pos[:, leg]) - self.cfg.leg_kd * self.dof_vel[:, leg]
        self.leg_torque = torch.maximum(torch.minimum(torque, self.leg_effort), -self.leg_effort)

    def _randomize_friction(self) -> None:
        """Give every environment its own ground friction, as legged_gym does."""
        if not self.cfg.randomize_friction:
            return
        low, high = self.cfg.friction_range
        try:
            ratio = torch.empty((self.num_envs, 1), dtype=torch.float32, device=self.device).uniform_(low, high)
            links = list(range(self.robot.n_links))
            self.robot.set_friction_ratio(ratio.expand(-1, len(links)).contiguous(), links)
        except Exception as error:
            print(f"Friction randomization is unavailable on this Genesis build ({error}); friction stays fixed.")

    def _push_robots(self) -> None:
        """Every push interval, give some robots a random planar velocity. legged_gym pushes all at once."""
        if not self.cfg.push_robots or self.cfg.curriculum not in LOCOMOTION_STAGES:
            return
        self.push_timer += 1
        due = self.push_timer >= int(self.cfg.push_interval_s / self.cfg.dt)
        if not bool(due.any()):
            return
        idx = torch.nonzero(due, as_tuple=False).squeeze(-1)
        self.push_timer[idx] = 0
        velocity = self.base_lin_vel_world[idx, :2]
        velocity = velocity + (torch.rand_like(velocity) * 2.0 - 1.0) * self.cfg.push_max_vel
        self.robot.set_dofs_velocity(velocity, [0, 1], envs_idx=idx)

    def _leg_command(self, leg_act: torch.Tensor) -> torch.Tensor:
        """Legs while walking: reference pose plus the policy."""
        walk = (self.phase == 0) | (self.phase == 2)
        return learned_leg_target(
            leg_act,
            self.default_leg,
            walk,
            control_split=self.cfg.control_split,
            action_scale=self.cfg.leg_action_scale,
            lower=self.leg_lower,
            upper=self.leg_upper,
        )

    def _base_yaw(self) -> torch.Tensor:
        w, qx, qy, qz = self.base_quat.unbind(dim=-1)
        return torch.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def _body_offset(self, point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward and left offsets of a world point, in the base heading."""
        yaw = self._base_yaw()
        delta = point[:, :2] - self.base_pos[:, :2]
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        ahead = c * delta[:, 0] + s * delta[:, 1]
        aside = -s * delta[:, 0] + c * delta[:, 1]
        return ahead, aside

    def _read_state(self) -> None:
        self.base_pos = self._batch(self.robot.get_pos())
        self.base_quat = self._batch(self.robot.get_quat())
        world_lin = self._batch(self.robot.get_vel())
        world_ang = self._batch(self.robot.get_ang())
        self.base_lin_vel_world = world_lin
        inv_quat = quat_conjugate(self.base_quat)
        self.base_lin_vel = quat_apply(inv_quat, world_lin)
        self.base_ang_vel = quat_apply(inv_quat, world_ang)
        self.projected_gravity = quat_apply(inv_quat, self.gravity.expand(self.num_envs, 3))
        self.dof_pos = self._batch(self.robot.get_dofs_position(self.dof_idx))
        self.dof_vel = self._batch(self.robot.get_dofs_velocity(self.dof_idx))
        ee = self._batch(self.robot.get_links_pos(links_idx_local=[self.ee_link_idx]))
        self.ee_pos = ee[:, 0, :] if ee.ndim == 3 else ee
        self._read_object()
        if self.ball is not None:
            self.ball_pos = self._batch(self.ball.get_pos())
            self.ball_vel = self._batch(self.ball.get_vel())
        self._read_contacts()
        self._sense_wrist()

    def _read_object(self) -> None:
        if len(self.supplies) == 1:
            self.obj_pos = self._batch(self.cube.get_pos())
            return
        stacked = torch.stack([self._batch(cube.get_pos()) for cube in self.supplies], dim=1)
        index = self._supply_index.view(-1, 1, 1).expand(-1, 1, 3)
        self.obj_pos = stacked.gather(1, index).squeeze(1)

    def _read_contacts(self) -> None:
        try:
            net = self._batch(self.robot.get_links_net_contact_force())
            foot_force = net[:, self.foot_idx, :]
            self.foot_contact = (torch.linalg.norm(foot_force, dim=-1) > self.cfg.contact_force_threshold).to(torch.float32)
            self.body_contact = torch.linalg.norm(net[:, self.base_link_idx, :], dim=-1) > self.cfg.contact_force_threshold
            hits = torch.linalg.norm(net[:, self.collision_idx, :], dim=-1) > COLLISION_FORCE
            self.leg_collisions = hits.sum(dim=-1).to(torch.float32)
        except Exception:
            if not self._contact_warning_sent:
                print("Foot contact is unavailable on this Genesis build; contact flags stay zero.")
                self._contact_warning_sent = True
            self.foot_contact.zero_()
            self.body_contact.zero_()
            self.leg_collisions.zero_()

    def _batch(self, tensor: torch.Tensor) -> torch.Tensor:
        return _on_device(_as_batch(tensor, self.num_envs), self.device)

    def _rest_fingers(self, count: int) -> torch.Tensor:
        """Gripper joint position written at reset. Holding it keeps the initial pose."""
        return torch.full(
            (count, len(self.finger_names)),
            0.5 * self.cfg.gripper_open,
            dtype=torch.float32,
            device=self.device,
        )

    def _finger_slice(self) -> slice:
        return slice(len(self.leg_names) + len(self.arm_names), len(self.actuated_names))

    def _gripper_opening(self) -> torch.Tensor:
        joint = self.dof_pos[:, self._finger_slice()].mean(dim=-1)
        return gripper_open_amount(joint, self.arm.joint_open, self.arm.joint_closed)

    def _update_grasp_flags(self) -> None:
        opening = self._gripper_opening()
        if self.cfg.wrist_camera:
            # A closed gripper counts only when the wrist camera sees the cube at the hand.
            # If the view is lost while the fingers stay closed, keep the hold.
            closed = gripper_is_closed(opening, self.cfg.gripper_closed)
            seen = grasp_mask(
                self.ee_pos,
                self.wrist_pos,
                opening,
                self.cfg.grasp_distance,
                self.cfg.gripper_closed,
            ) & self.wrist_visible
            self.grasped = seen | (self.prev_grasped & closed & ~self.wrist_visible)
        else:
            self.grasped = grasp_mask(
                self.ee_pos,
                self.obj_pos,
                opening,
                self.cfg.grasp_distance,
                self.cfg.gripper_closed,
            )
        self.was_grasped |= self.grasped
        if self.cfg.curriculum == "haul":
            self.grasped.fill_(True)
            self.was_grasped.fill_(True)

    @staticmethod
    def _camera_matrix(transform) -> np.ndarray:
        if hasattr(transform, "detach"):
            transform = transform.detach().cpu().numpy()
        return np.asarray(transform, dtype=np.float32)

    def _sense_wrist(self) -> None:
        if self.wrist_camera is None:
            return
        try:
            self.wrist_camera.move_to_attach()
            _, depth, segmentation, _ = self.wrist_camera.render(
                rgb=False,
                depth=True,
                segmentation=True,
                force_render=True,
            )
            target = [self._supply_link_idx[int(index)] for index in self._supply_index.tolist()]
            positions, visible = detect_wrist_target(
                depth,
                segmentation,
                target,
                focal=float(self.wrist_camera.f),
                cx=float(self.wrist_camera.cx),
                cy=float(self.wrist_camera.cy),
                transform=self._camera_matrix(self.wrist_camera.transform),
            )
        except Exception as exc:
            if not self._wrist_warning_sent:
                print(f"Wrist camera detection failed; the cube stays unseen. {exc}", flush=True)
                self._wrist_warning_sent = True
            self.wrist_visible.zero_()
            return
        self.wrist_pos.copy_(torch.as_tensor(positions, dtype=torch.float32, device=self.device))
        self.wrist_visible.copy_(torch.as_tensor(visible, dtype=torch.bool, device=self.device))

    def _assist_grasp(self) -> None:
        if not bool(self.grasped.any()):
            return
        if len(self.supplies) == 1:
            idx = torch.where(self.grasped)[0]
            self.cube.set_pos(self.ee_pos[idx], envs_idx=idx, zero_velocity=True)
            held_quat = self.init_base_quat.expand(idx.shape[0], 4)
            self.cube.set_quat(held_quat, envs_idx=idx, zero_velocity=True)
        else:
            held_quat = self.init_base_quat
            for supply_i, cube in enumerate(self.supplies):
                idx = torch.where(self.grasped & (self._supply_index == supply_i))[0]
                if idx.numel() == 0:
                    continue
                cube.set_pos(self.ee_pos[idx], envs_idx=idx, zero_velocity=True)
                cube.set_quat(held_quat.expand(idx.shape[0], 4), envs_idx=idx, zero_velocity=True)
        self.obj_pos = torch.where(self.grasped[:, None], self.ee_pos, self.obj_pos)

    def _place_yard(self, idx, n: int, shifts: torch.Tensor):
        half = self.cfg.cube_size * 0.5 + 0.002
        pickups = torch.as_tensor(self._yard_pickups, dtype=torch.float32, device=self.device)
        goals = torch.as_tensor(self._yard_goals, dtype=torch.float32, device=self.device)
        if idx is None:
            self._supply_index.zero_()
            self.supply_finished.fill_(False)
            self._release_hold.fill_(False)
        else:
            self._supply_index[idx] = 0
            self.supply_finished[idx] = False
            self._release_hold[idx] = False
        obj_pos = None
        quat = self.init_base_quat.expand(n, 4)
        for supply_i, cube in enumerate(self.supplies):
            xy = pickups[supply_i : supply_i + 1].expand(n, 2) + shifts
            pos = torch.cat((xy, self._ground_z(xy) + half), dim=-1)
            cube.set_pos(pos, envs_idx=idx, zero_velocity=True)
            cube.set_quat(quat, envs_idx=idx, zero_velocity=True)
            if supply_i == 0:
                obj_pos = pos
        for supply_i in range(len(self.supplies)):
            xy = goals[supply_i : supply_i + 1]
            self._goal_pose[supply_i, 0] = xy[0, 0]
            self._goal_pose[supply_i, 1] = xy[0, 1]
            self._goal_pose[supply_i, 2] = self._ground_z(xy)[0, 0] + self.cfg.goal_marker_size[2] * 0.5
        goal_pos = self._goal_pose[0].expand(n, 3).clone()
        goal_pos[:, 0] += shifts[:, 0]
        goal_pos[:, 1] += shifts[:, 1]
        return obj_pos, quat, goal_pos

    def _advance_supplies(self, placed: torch.Tensor) -> None:
        """Leave the delivered box where it is and aim at the next one."""
        for env_i in torch.where(placed)[0].tolist():
            nxt = int(self._supply_index[env_i]) + 1
            if nxt >= len(self.supplies):
                self.supply_finished[env_i] = True
                continue
            self._supply_index[env_i] = nxt
            self.goal_pos[env_i] = self._goal_pose[nxt]
        self.was_grasped[placed] = False
        self.grasped[placed] = False
        self.place_latched[placed] = False
        self.prev_grasped[placed] = False
        self._release_hold[placed] = False

    def _pin_goal(self, envs_idx) -> None:
        """Place the visible marker. Its center sits above goal_pos so the column is easy to see."""
        if envs_idx is None:
            shown = self.goal_pos
            quat = self.init_base_quat.expand(self.num_envs, 4)
            idx = None
        else:
            idx = self._index(envs_idx)
            if idx.numel() == 0:
                return
            shown = self.goal_pos[idx]
            quat = self.init_base_quat.expand(idx.shape[0], 4)
        shown = shown.clone()
        shown[:, 2] = shown[:, 2] + 0.20
        if idx is None:
            self.goal_marker.set_pos(shown, zero_velocity=True)
            self.goal_marker.set_quat(quat, zero_velocity=True)
            return
        self.goal_marker.set_pos(shown, envs_idx=idx, zero_velocity=True)
        self.goal_marker.set_quat(quat, envs_idx=idx, zero_velocity=True)

    def set_goal_xy(self, x: float, y: float) -> tuple[float, float]:
        """Move the drop point of the single inference environment and refresh the observation.

        The new point is kept for later episode resets. The supply position is unchanged.
        """
        if self.num_envs != 1:
            raise RuntimeError("set_goal_xy is for one environment.")
        cx, cy = clamp_goal_xy(x, y)
        xy = torch.tensor([[cx, cy]], dtype=torch.float32, device=self.device)
        z = self._ground_z(xy) + float(self.cfg.goal_marker_size[2]) * 0.5
        self.goal_pos[0, 0] = cx
        self.goal_pos[0, 1] = cy
        self.goal_pos[0, 2] = float(z[0, 0])
        self.cfg.fixed_goal_xy = (cx, cy)
        self._release_hold.zero_()
        self._pin_goal(None)
        self._update_phase()
        self.prev_goal_dist.copy_(pairwise_distance(self.obj_pos, self.goal_pos))
        self._write_observation()
        return cx, cy

    def inference_state(self) -> dict[str, object]:
        """One environment, as plain numbers for the ROS bridge."""

        def row(tensor: torch.Tensor) -> list[float]:
            return [float(value) for value in tensor.detach().reshape(-1).cpu().tolist()]

        return {
            "type": "state",
            "arm": self.arm.name,
            "urdf": Path(self.urdf_path).name,
            "joint_names": list(self.actuated_names),
            "joint_pos": row(self.dof_pos[0]),
            "base_pos": row(self.base_pos[0]),
            "base_quat_wxyz": row(self.base_quat[0]),
            "ee_pos": row(self.ee_pos[0]),
            "object_pos": row(self.obj_pos[0]),
            "goal_pos": row(self.goal_pos[0]),
            "phase": int(self.phase[0].item()),
            "cube_size": float(self.cfg.cube_size),
        }

    def _set_camera_far(self, camera, far: float) -> None:
        """Push the clip plane out. It is fixed when the camera is built, and update_camera leaves it."""
        camera._far = float(far)
        rasterizer = self.scene.visualizer.rasterizer
        node = getattr(rasterizer, "_camera_nodes", {}).get(camera.uid)
        if node is not None and node.camera is not None:
            node.camera.zfar = float(far)

    def _rendered_env_indices(self) -> list[int]:
        """Environments the rasterizer draws. Physics still steps every copy."""
        if self._fleet_indices:
            return self._fleet_indices
        return list(range(self.num_envs if self.cfg.wrist_camera else 1))

    def save_fleet_snapshot(self) -> Path | None:
        """Write one overhead still of the drawn robots. The viewer stays closed."""
        if self.fleet_camera is None or not self.cfg.fleet_snapshot:
            return None
        try:
            offsets = np.asarray(self.scene.envs_offset, dtype=np.float64)[self._fleet_indices]
            origin = offsets[0]
            center = offsets.mean(axis=0)
            center[2] = 0.2
            height = fleet_camera_height(offsets[:, :2], max(self.cfg.env_spacing))
            look = center - origin
            pos = look + np.array([0.0, 0.0, height])
            self._set_camera_far(self.fleet_camera, max(height * 4.0, 50.0))
            self.fleet_camera.set_pose(pos=pos.tolist(), lookat=look.tolist())
            rgb, *_ = self.fleet_camera.render(
                rgb=True, depth=False, segmentation=False, normal=False, force_render=True
            )
            image = _rgb_uint8(rgb)
            path = Path(self.cfg.fleet_snapshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            import cv2

            if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"cv2.imwrite failed for {path}")
        except Exception as exc:
            print(f"並びの静止画を残せません。{exc}", flush=True)
            return None
        print(
            f"並びの静止画を残しました: {path} （描画 {len(self._fleet_indices)} 台 / 計算 {self.num_envs} 台）",
            flush=True,
        )
        try:
            import os

            os.startfile(path)
        except Exception:
            pass
        return path

    def record_initial_pose(self, hold_s: float = 1.0) -> None:
        """Write the pose from reset at the front of the mp4, before any action is applied."""
        camera = self.camera
        if camera is None or not self._recording or not camera._recorded_encoders:
            return
        from genesis.utils.misc import tensor_to_array

        rgb, *_ = camera.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)
        frame = tensor_to_array(rgb)
        if not np.issubdtype(np.asarray(frame).dtype, np.integer):
            frame = _rgb_uint8(frame)
        frame = np.ascontiguousarray(frame)
        fps = float(camera._recorded_encoders[0]._fps)
        repeats = max(1, int(round(hold_s * fps)))
        batched = bool(getattr(camera, "_is_batched", False) and camera._env_idx is None)
        for _ in range(repeats):
            if batched:
                for env_frame, encoder in zip(frame, camera._recorded_encoders):
                    encoder.write(np.ascontiguousarray(env_frame))
            else:
                camera._recorded_encoders[0].write(frame.copy())
        print(f"動画の最初の {hold_s:.0f} 秒は、動かす前の初期姿勢です。", flush=True)

    def overview_rgb(self) -> np.ndarray | None:
        """One downsampled overview frame. The interactive viewer stays closed."""
        if self.camera is None:
            return None
        try:
            rgb, *_ = self.camera.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)
        except Exception as exc:
            if not self._overview_warning_sent:
                print(f"俯瞰映像を出せません。{exc}", flush=True)
                self._overview_warning_sent = True
            return None
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        return downsample_rgb(rgb, stride=3)

    def _write_observation(self) -> None:
        leg = slice(0, len(self.leg_names))
        arm = slice(len(self.leg_names), len(self.leg_names) + len(self.arm_names))
        fingers = self._finger_slice()
        gripper_pos = self.dof_pos[:, fingers].mean(dim=-1, keepdim=True)
        gripper_vel = self.dof_vel[:, fingers].mean(dim=-1, keepdim=True)
        zeros3 = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        if self.cfg.curriculum == "carry":
            obj_b = object_in_base(
                self.obj_pos,
                self.base_pos,
                self.base_quat,
                use_wrist=self.cfg.wrist_camera,
                wrist_pos=self.wrist_pos,
                wrist_visible=self.wrist_visible,
            )
            goal_b = world_to_base(self.goal_pos, self.base_pos, self.base_quat)
            command = zeros3
            phase = self.phase
            grasped = self.grasped
        elif self.cfg.curriculum == "reach":
            obj_b = world_to_base(self.ball_pos, self.base_pos, self.base_quat)
            goal_b = zeros3
            command = zeros3
            phase = self.phase
            grasped = self.grasped
        elif self.cfg.curriculum == "catch":
            # The goal slot carries the ball velocity and the command slot a short prediction of where it
            # will be. Both are unused in this stage. The prediction is scaled down so that the legs, which
            # learned to read that slot as a speed command in walk, are not told to run at the ball.
            obj_b, ball_v, ball_ahead = ball_in_base(
                self.ball_pos, self.ball_vel, self.base_pos, self.base_quat, self.cfg.catch_level
            )
            goal_b = ball_v * BALL_VELOCITY_SCALE
            command = ball_ahead * PREDICTION_SCALE
            phase = self.phase
            grasped = self.grasped
        elif self.cfg.curriculum == "haul":
            obj_b = world_to_base(self.obj_pos, self.base_pos, self.base_quat)
            goal_b = zeros3
            command = self.command
            phase = self.phase
            grasped = self.grasped
        else:
            obj_b = zeros3
            goal_b = zeros3
            command = self.command if self.cfg.curriculum in LOCOMOTION_STAGES else zeros3
            phase = self.phase
            grasped = torch.zeros_like(self.grasped)
        self.obs_buf = pack_observation(
            leg_pos=self.dof_pos[:, leg] - self.default_leg,
            leg_vel=self.dof_vel[:, leg],
            projected_gravity=self.projected_gravity,
            base_lin_vel=self.base_lin_vel,
            base_ang_vel=self.base_ang_vel,
            base_height=self.base_pos[:, 2:3],
            foot_contact=self.foot_contact,
            arm_pos=self.dof_pos[:, arm],
            arm_vel=self.dof_vel[:, arm],
            gripper_pos=gripper_pos,
            gripper_vel=gripper_vel,
            ee_pos=world_to_base(self.ee_pos, self.base_pos, self.base_quat),
            object_pos=obj_b,
            goal_pos=goal_b,
            grasped=grasped,
            phase=phase,
            command=command,
            last_leg_action=self.prev_leg_joint_actions,
        )
        if self.obs_buf.shape[-1] != self.num_obs:
            raise RuntimeError(
                f"Observation dim {self.obs_buf.shape[-1]} does not match {self.num_obs} for {self.cfg.curriculum}."
            )

    def _reset_idx(self, envs_idx) -> None:
        idx = None if envs_idx is None else self._index(envs_idx)
        if idx is not None and idx.numel() == 0:
            return
        n = self.num_envs if idx is None else int(idx.shape[0])
        if self._terrain_grid is not None and self.cfg.randomize_terrain and self.cfg.yard_stations == 0:
            ids = list(range(self.num_envs)) if idx is None else [int(value) for value in idx.detach().cpu().tolist()]
            self._retile(ids)
            shifts = self._tile_shifts if idx is None else self._tile_shifts[idx]
        else:
            shifts = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        base_pos = self.init_base_pos.expand(n, 3).clone()
        base_pos[:, 0] += shifts[:, 0]
        base_pos[:, 1] += shifts[:, 1]
        base_quat = self.init_base_quat.expand(n, 4).clone()
        leg = self.default_leg.expand(n, -1)
        arm = self.default_arm.expand(n, -1)
        fingers = self._rest_fingers(n)
        dof_pos = torch.cat((leg, arm, fingers), dim=-1)
        if idx is None:
            self.q_ik = arm.clone()
            self.ik_error.zero_()
        else:
            self.q_ik[idx] = arm
            self.ik_error[idx] = 0

        self.robot.set_pos(base_pos, envs_idx=idx, zero_velocity=True)
        self.robot.set_quat(base_quat, envs_idx=idx, zero_velocity=True)
        self.robot.set_dofs_position(dof_pos, self.dof_idx, envs_idx=idx, zero_velocity=True)

        if self._yard_pickups is not None:
            obj_pos, obj_quat, goal_pos = self._place_yard(idx, n, shifts)
        else:
            obj_pos, obj_quat = self._sample_object(n, shifts)
            goal_pos = self._sample_goal(n, shifts)
        self.cube.set_pos(obj_pos, envs_idx=idx, zero_velocity=True)
        self.cube.set_quat(obj_quat, envs_idx=idx, zero_velocity=True)

        if idx is None:
            self.goal_pos.copy_(goal_pos)
            self.obj_pos.copy_(obj_pos)
            self.base_pos.copy_(base_pos)
            self.base_quat.copy_(base_quat)
            self.actions.zero_()
            self.prev_actions.zero_()
            self.leg_joint_actions.zero_()
            self.prev_leg_joint_actions.zero_()
            self.cpg.reset()
            self.feet_air_time.zero_()
            self.last_contact.fill_(False)
            self.air_time_reward.zero_()
            self.prev_leg_vel.zero_()
            self.push_timer.copy_(self._initial_push_timer(n))
            self.episode_length.zero_()
            self.episode_return.zero_()
            self.grasped.fill_(False)
            self.prev_grasped.fill_(False)
            self.was_grasped.fill_(False)
            self.place_latched.fill_(False)
            self.dof_pos.copy_(dof_pos)
            self.dof_vel.zero_()
        else:
            self.goal_pos[idx] = goal_pos
            self.obj_pos[idx] = obj_pos
            self.base_pos[idx] = base_pos
            self.base_quat[idx] = base_quat
            self.actions[idx] = 0
            self.prev_actions[idx] = 0
            self.leg_joint_actions[idx] = 0
            self.prev_leg_joint_actions[idx] = 0
            self.cpg.reset(idx)
            self.feet_air_time[idx] = 0
            self.last_contact[idx] = False
            self.air_time_reward[idx] = 0
            self.prev_leg_vel[idx] = 0
            self.push_timer[idx] = self._initial_push_timer(n)
            self.episode_length[idx] = 0
            self.episode_return[idx] = 0
            self.grasped[idx] = False
            self.prev_grasped[idx] = False
            self.was_grasped[idx] = False
            self.place_latched[idx] = False
            self.dof_pos[idx] = dof_pos
            self.dof_vel[idx] = 0
        self._pin_goal(idx)
        self._reset_curriculum(idx, base_pos)

    def _reset_curriculum(self, idx, base_pos: torch.Tensor) -> None:
        count = base_pos.shape[0]
        if self.cfg.curriculum in ("walk", "haul"):
            share = STAND_STILL_SHARE if self.cfg.curriculum == "walk" else 0.0
            command = sample_command(count, self.device, share)
            self._write_rows(self.command, command, idx)
        else:
            self._write_rows(self.command, torch.zeros((count, 3), dtype=torch.float32, device=self.device), idx)
        if self.cfg.curriculum in LOCOMOTION_STAGES:
            self._park_cube(idx, count)
            # The arm starts at the reset pose and wanders from its first random target on.
            self._write_rows(self.arm_cmd, self.default_arm.expand(count, -1), idx)
            self._write_rows(self.arm_goal, self.default_arm.expand(count, -1), idx)
            interval = max(1, int(self.cfg.arm_disturb_interval_s / self.cfg.dt))
            self._write_rows(self.arm_timer, torch.randint(0, interval, (count,), dtype=torch.int32, device=self.device), idx)
        if self.cfg.curriculum == "haul":
            held = base_pos.clone()
            held[:, 0] = held[:, 0] + 0.32
            held[:, 2] = 0.28
            self.cube.set_pos(held, envs_idx=idx, zero_velocity=True)
            self._write_rows(self.obj_pos, held, idx)
        if self.cfg.curriculum == "reach" and self.ball is not None:
            target = sample_reach_target(base_pos)
            self.ball.set_pos(target, envs_idx=idx, zero_velocity=True)
            self._write_rows(self.reach_target, target, idx)
            self._write_rows(self.ball_pos, target, idx)
            self._write_rows(self.ball_vel, torch.zeros_like(target), idx)
            self._park_cube(idx, count)
        if self.cfg.curriculum == "catch" and self.ball is not None:
            position, linear, angular = sample_ball(self.cfg.catch_level, base_pos)
            self.ball.set_pos(position, envs_idx=idx, zero_velocity=True)
            velocity = torch.cat((linear, angular), dim=-1)
            try:
                self.ball.set_dofs_velocity(velocity, envs_idx=idx)
            except Exception as exc:
                if not self._ball_velocity_warning:
                    print(f"ボールの初速を入れられません。{exc}", flush=True)
                    self._ball_velocity_warning = True
            self._write_rows(self.ball_pos, position, idx)
            self._write_rows(self.ball_vel, linear, idx)
            self._park_cube(idx, count)
        self._write_rows(self.hold_steps, torch.zeros(count, dtype=torch.int32, device=self.device), idx)
        self._write_rows(self.command_timer, torch.zeros(count, dtype=torch.int32, device=self.device), idx)

    def _park_cube(self, idx, count: int) -> None:
        parked = torch.zeros((count, 3), dtype=torch.float32, device=self.device)
        parked[:, 2] = -2.0
        self.cube.set_pos(parked, envs_idx=idx, zero_velocity=True)
        self._write_rows(self.obj_pos, parked, idx)

    def _write_rows(self, buffer: torch.Tensor, values: torch.Tensor, idx) -> None:
        if idx is None:
            buffer.copy_(values)
        else:
            buffer[idx] = values

    def _refresh_commands(self) -> None:
        if self.hold_commands:
            return
        period = max(1, int(round(WALK_COMMAND_SECONDS / self.cfg.dt)))
        self.command_timer += 1
        due = self.command_timer >= period
        if not bool(due.any()):
            return
        share = STAND_STILL_SHARE if self.cfg.curriculum == "walk" else 0.0
        fresh = sample_command(self.num_envs, self.device, share)
        self.command = torch.where(due[:, None], fresh, self.command)
        self.command_timer = torch.where(due, torch.zeros_like(self.command_timer), self.command_timer)

    def set_commands(self, command: torch.Tensor, *, hold: bool = True) -> torch.Tensor:
        """Overwrite velocity commands and patch them into ``obs_buf`` (command slice).

        Use ``hold=True`` for teacher / BC demos so ``_refresh_commands`` does not
        overwrite the locked command mid-rollout.
        """
        if command.shape != self.command.shape:
            raise ValueError(f"command shape {tuple(command.shape)} != {tuple(self.command.shape)}")
        self.command.copy_(command)
        self.hold_commands = bool(hold)
        cmd_slice = self.obs_slices["command"]
        self.obs_buf[:, cmd_slice] = self.command
        return self.obs_buf

    def _ground_z(self, xy: torch.Tensor) -> torch.Tensor:
        """Height of the course under xy points. Flat zero when terrain is off."""
        if self._terrain_grid is None:
            return torch.zeros((xy.shape[0], 1), dtype=torch.float32, device=self.device)
        sampled = sample_height(
            self._terrain_grid,
            self._terrain_origin,
            self._terrain_scale,
            xy.detach().cpu().numpy(),
        )
        return torch.as_tensor(sampled, dtype=torch.float32, device=self.device).unsqueeze(-1)

    def _retile(self, env_ids: list[int]) -> None:
        """Give each resetting env a new bump pattern. Other envs keep their ground."""
        for env_id in env_ids:
            tile, _, _ = make_course(
                self.cfg.terrain_amplitude * self._terrain_mix,
                self._terrain_scale,
                self._terrain_rng,
                steps=self.cfg.terrain_steps,
            )
            paste_tile(self._terrain_grid, env_id, tile, self._tile_cols, self._tile_stride)
        self._publish_terrain()

    def _prepare_terrain_runtime(self) -> None:
        info = self.scene.rigid_solver.collider.collider_info
        bounds = info.terrain_xyz_maxmin.to_numpy().copy()
        bounds[2] = max(float(bounds[2]), float(self.cfg.terrain_amplitude))
        info.terrain_xyz_maxmin.from_numpy(bounds)
        self._visual_map = self._match_visual_vertices()

    def _match_visual_vertices(self) -> np.ndarray | None:
        """Indices of rendered vertices that sit on the height-field grid, in grid order."""
        positions = self.scene.rigid_solver.dyn_info.vverts.init_pos.to_numpy()
        if positions.ndim != 2 or positions.shape[1] != 3:
            return None
        scale = self._terrain_scale
        rows, cols = self._terrain_grid.shape
        ix = np.rint(positions[:, 0] / scale).astype(np.int32)
        iy = np.rint(positions[:, 1] / scale).astype(np.int32)
        on_grid = (
            (ix >= 0)
            & (ix < rows)
            & (iy >= 0)
            & (iy < cols)
            & (np.abs(positions[:, 0] - ix * scale) < scale * 0.2)
            & (np.abs(positions[:, 1] - iy * scale) < scale * 0.2)
        )
        flat = ix * cols + iy
        chosen = np.full(rows * cols, -1, dtype=np.int32)
        chosen[flat[on_grid]] = np.flatnonzero(on_grid)
        if int(np.count_nonzero(chosen >= 0)) < rows * cols // 2:
            return None
        return chosen

    def set_catch_level(self, level: str) -> None:
        """Next reset uses this ball. The scene stays the same."""
        if level not in CATCH_LEVELS:
            raise ValueError(f"catch_level must be one of {', '.join(CATCH_LEVELS)}, got {level}.")
        self.cfg.catch_level = level

    def set_terrain_mix(self, mix: float) -> None:
        """1 is the selected rough ground. 0 is flat. New bumps use the mix from the next reset."""
        self._terrain_mix = min(1.0, max(0.0, float(mix)))
        if self._terrain_grid is None:
            return
        self._retile(list(range(self.num_envs)))

    def _publish_terrain(self) -> None:
        height = np.ascontiguousarray(self._terrain_grid, dtype=np.float32)
        info = self.scene.rigid_solver.collider.collider_info
        info.terrain_hf.from_numpy(height)
        self.terrain._terrain_height_field = torch.as_tensor(height, dtype=torch.float32, device=self.device)
        if self._visual_map is None:
            return
        positions = self.scene.rigid_solver.dyn_info.vverts.init_pos.to_numpy()
        known = self._visual_map >= 0
        positions[self._visual_map[known], 2] = height.reshape(-1)[known]
        self.scene.rigid_solver.dyn_info.vverts.init_pos.from_numpy(np.ascontiguousarray(positions, dtype=np.float32))

    def _sample_object(self, n: int, shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.fixed_object_xy is None:
            xy = self._uniform_xy(n, self.cfg.object_x_range, self.cfg.object_y_range) + shift
            yaw = torch.empty((n,), device=self.device).uniform_(-math.pi, math.pi)
        else:
            xy = self._fixed_xy(n, self.cfg.fixed_object_xy) + shift
            yaw = torch.zeros((n,), device=self.device)
        z = self._ground_z(xy) + self.cfg.cube_size * 0.5 + 0.002
        return torch.cat((xy, z), dim=-1), yaw_quat(yaw)

    def _sample_goal(self, n: int, shift: torch.Tensor) -> torch.Tensor:
        if self.cfg.fixed_goal_xy is None:
            xy = self._uniform_xy(n, self.cfg.goal_x_range, self.cfg.goal_y_range) + shift
        else:
            xy = self._fixed_xy(n, self.cfg.fixed_goal_xy) + shift
        z = self._ground_z(xy) + self.cfg.goal_marker_size[2] * 0.5
        return torch.cat((xy, z), dim=-1)

    def _fixed_xy(self, n: int, xy: tuple[float, float]) -> torch.Tensor:
        point = torch.tensor(xy, dtype=torch.float32, device=self.device).view(1, 2)
        return point.expand(n, 2).clone()

    def _uniform_xy(self, n: int, x_range: tuple[float, float], y_range: tuple[float, float]) -> torch.Tensor:
        x = torch.empty((n, 1), device=self.device).uniform_(*x_range)
        y = torch.empty((n, 1), device=self.device).uniform_(*y_range)
        return torch.cat((x, y), dim=-1)

    def _index(self, envs_idx) -> torch.Tensor:
        if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
            return torch.where(envs_idx)[0]
        return torch.as_tensor(envs_idx, dtype=torch.long, device=self.device)

    def close(self) -> None:
        if self.camera is not None and self._recording:
            self.camera.stop_recording()
            self._recording = False
            self.camera = None
        destroy = getattr(self.scene, "destroy", None)
        if callable(destroy):
            destroy()
