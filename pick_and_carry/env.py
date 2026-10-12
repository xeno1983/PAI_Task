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

from pick_and_carry.arms import get_arm, layouts
from pick_and_carry.config import (
    DEFAULT_LEG_ANGLES,
    FOOT_LINKS,
    LEG_JOINTS,
    EnvConfig,
    default_urdf_path,
    layout_dim,
    layout_slices,
    leg_limit,
)
from pick_and_carry.control import arm_gripper_command, gripper_open_amount, learned_leg_target, scripted_trot_pose
from pick_and_carry.helpers import (
    detect_wrist_target,
    grasp_mask,
    gripper_is_closed,
    pairwise_distance,
    task_phase,
    quat_apply,
    quat_conjugate,
    world_to_base,
    yaw_quat,
)
from pick_and_carry.observation import object_in_base, pack_observation
from pick_and_carry.reward import task_reward
from pick_and_carry.ros_bridge import clamp_goal_xy, downsample_rgb
from pick_and_carry.terrain import make_course, make_supply_yard, make_terrain_bank, paste_tile, sample_height


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
        if self.cfg.control_mode not in ("position", "torque"):
            raise ValueError("control_mode must be 'position' or 'torque'.")

        self.num_envs = self.cfg.num_envs
        self.arm = get_arm(self.cfg.arm)
        if self.arm.name != "panda":
            self.cfg.arm_action_scale = self.arm.action_scale
            self.cfg.arm_kp = self.arm.kp
            self.cfg.arm_kd = self.arm.kd
            self.cfg.gripper_kp = self.arm.gripper_kp
            self.cfg.gripper_kd = self.arm.gripper_kd
            self.cfg.gripper_open = abs(self.arm.joint_open - self.arm.joint_closed)
            self.cfg.gripper_closed = self.arm.closed_threshold
        self.camera = None
        self.wrist_camera = None
        self._recording = False
        self._wrist_warning_sent = False
        self._overview_warning_sent = False
        obs_layout, action_layout = layouts(self.arm.dof)
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
        self.gs = gs
        self.device = torch.device(gs.device if not isinstance(gs.device, torch.device) else gs.device)
        torch.manual_seed(self.cfg.seed)

        if self.cfg.urdf_path:
            self.urdf_path = self.cfg.urdf_path
        elif self.arm.name == "panda":
            self.urdf_path = str(default_urdf_path())
        else:
            self.urdf_path = str(self.arm.urdf_path)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.cfg.dt, substeps=self.cfg.substeps),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=self.cfg.max_collision_pairs,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.num_envs if self.cfg.wrist_camera else 1)),
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

        self._configure_pd()
        self._allocate_buffers()
        self.extras: dict = {}
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
        self.gait_time = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.gait_blend = torch.ones((n,), dtype=torch.float32, device=self.device)
        self.base_obj_dist = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.obj_goal_dist = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.wrist_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.wrist_visible = torch.zeros((n,), dtype=torch.bool, device=self.device)
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
        if self.cfg.freeze_legs:
            self.actions = self.actions.clone()
            self.actions[:, self.action_slices["leg"]] = 0.0
        self._apply_action(self.actions)
        self.scene.step()
        self.episode_length += 1
        self._read_state()
        self._pin_goal(None)
        self._update_grasp_flags()
        self._assist_grasp()
        self._update_phase()

        reward, terms, newly_placed, fallen = self.compute_reward()
        if self.cfg.cycle_supplies and bool(newly_placed.any()):
            self._advance_supplies(newly_placed)
            newly_placed = torch.zeros_like(newly_placed)
        self.rew_buf = reward
        self.reward_terms = terms
        self.episode_return += reward
        grasped_now = self.grasped.clone()
        was_grasped_now = self.was_grasped.clone()

        timeout = self.episode_length >= self.max_episode_length
        success = newly_placed
        done = success | fallen | timeout
        terminal = success | fallen
        self.done_buf = done
        self.extras = {
            "success": success,
            "fallen": fallen,
            "timeout": timeout & ~terminal,
            "terminal": terminal,
            "time_outs": (timeout & ~terminal).to(dtype=torch.float32),
            "reward_terms": {name: value.detach() for name, value in terms.items()},
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
        self.prev_grasped = torch.where(done, torch.zeros_like(grasped_now), grasped_now)
        self._write_observation()
        return self.obs_buf, self.rew_buf, self.done_buf, self.extras

    def compute_reward(self):
        """Return the staged reward, per-term values, place events, and falls."""
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
        dist_ee = pairwise_distance(self.ee_pos, self.obj_pos)
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
        leg_act = action[:, self.action_slices["leg"]]
        arm_act = action[:, self.action_slices["arm"]]
        grip_act = action[:, self.action_slices["gripper"]]
        if self.cfg.control_mode == "position":
            leg_target = self._leg_command(leg_act)
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
            )
            finger_target = opening.unsqueeze(-1).expand(-1, len(self.finger_names))
            target = torch.cat((leg_target, arm_target, finger_target), dim=-1)
            self.robot.control_dofs_position(target, self.dof_idx)
            return
        leg_torque = leg_act * self.leg_effort
        arm_torque = arm_act * self.arm_effort
        finger_torque = grip_act.expand(-1, len(self.finger_names)) * 20.0
        force = torch.cat((leg_torque, arm_torque, finger_torque), dim=-1)
        self.robot.control_dofs_force(force, self.dof_idx)

    def _leg_command(self, leg_act: torch.Tensor) -> torch.Tensor:
        """Legs while walking. Learned by default; a fixed trot only if scripted_gait is set."""
        if self.cfg.freeze_legs:
            return self.default_leg.expand(leg_act.shape[0], -1).clone()
        walk = (self.phase == 0) | (self.phase == 2)
        if not self.cfg.scripted_gait:
            return learned_leg_target(
                leg_act,
                self.default_leg,
                walk,
                control_split=self.cfg.control_split,
                action_scale=self.cfg.leg_action_scale,
                lower=self.leg_lower,
                upper=self.leg_upper,
            )
        yaw_err = None
        if self.cfg.cycle_supplies:
            target = torch.where(self.grasped[:, None], self.goal_pos, self.obj_pos)
            _, _, yaw_err = self._aim_error(target)
        pose, self.gait_blend, self.gait_time = scripted_trot_pose(
            leg_act,
            self.default_leg,
            walk,
            self.gait_blend,
            self.gait_time,
            dt=self.cfg.dt,
            gait_period=self.cfg.gait_period,
            action_scale=self.cfg.leg_action_scale,
            lower=self.leg_lower,
            upper=self.leg_upper,
            yaw_err=yaw_err,
        )
        return pose

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

    def _aim_error(self, point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Heading error toward ``point``. Positive means the body is aimed left of it."""
        yaw = self._base_yaw()
        delta = point[:, :2] - self.base_pos[:, :2]
        desired = torch.atan2(delta[:, 1], delta[:, 0])
        yaw_err = torch.atan2(torch.sin(yaw - desired), torch.cos(yaw - desired))
        return yaw, desired, yaw_err

    def _read_state(self) -> None:
        self.base_pos = self._batch(self.robot.get_pos())
        self.base_quat = self._batch(self.robot.get_quat())
        world_lin = self._batch(self.robot.get_vel())
        world_ang = self._batch(self.robot.get_ang())
        inv_quat = quat_conjugate(self.base_quat)
        self.base_lin_vel = quat_apply(inv_quat, world_lin)
        self.base_ang_vel = quat_apply(inv_quat, world_ang)
        self.projected_gravity = quat_apply(inv_quat, self.gravity.expand(self.num_envs, 3))
        self.dof_pos = self._batch(self.robot.get_dofs_position(self.dof_idx))
        self.dof_vel = self._batch(self.robot.get_dofs_velocity(self.dof_idx))
        ee = self._batch(self.robot.get_links_pos(links_idx_local=[self.ee_link_idx]))
        self.ee_pos = ee[:, 0, :] if ee.ndim == 3 else ee
        self._read_object()
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
        except Exception:
            if not self._contact_warning_sent:
                print("Foot contact is unavailable on this Genesis build; contact flags stay zero.")
                self._contact_warning_sent = True
            self.foot_contact.zero_()

    def _batch(self, tensor: torch.Tensor) -> torch.Tensor:
        return _on_device(_as_batch(tensor, self.num_envs), self.device)

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
        if not self.cfg.assist_grasp or not bool(self.grasped.any()):
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
        obj_b = object_in_base(
            self.obj_pos,
            self.base_pos,
            self.base_quat,
            use_wrist=self.cfg.wrist_camera,
            wrist_pos=self.wrist_pos,
            wrist_visible=self.wrist_visible,
        )
        self.obs_buf = pack_observation(
            leg_pos=self.dof_pos[:, leg],
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
            goal_pos=world_to_base(self.goal_pos, self.base_pos, self.base_quat),
            grasped=self.grasped,
            phase=self.phase,
        )
        if self.obs_buf.shape[-1] != self.num_obs:
            raise RuntimeError(f"Observation dim {self.obs_buf.shape[-1]} does not match {self.num_obs} for arm {self.arm.name}.")

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
        fingers = torch.full((n, len(self.finger_names)), 0.5 * self.cfg.gripper_open, dtype=torch.float32, device=self.device)
        dof_pos = torch.cat((leg, arm, fingers), dim=-1)

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
            self.episode_length.zero_()
            self.episode_return.zero_()
            self.grasped.fill_(False)
            self.prev_grasped.fill_(False)
            self.was_grasped.fill_(False)
            self.place_latched.fill_(False)
            self.gait_time.zero_()
            self.gait_blend.fill_(1.0)
            self.dof_pos.copy_(dof_pos)
            self.dof_vel.zero_()
        else:
            self.goal_pos[idx] = goal_pos
            self.obj_pos[idx] = obj_pos
            self.base_pos[idx] = base_pos
            self.base_quat[idx] = base_quat
            self.actions[idx] = 0
            self.episode_length[idx] = 0
            self.episode_return[idx] = 0
            self.grasped[idx] = False
            self.prev_grasped[idx] = False
            self.was_grasped[idx] = False
            self.place_latched[idx] = False
            self.gait_time[idx] = 0
            self.gait_blend[idx] = 1.0
            self.dof_pos[idx] = dof_pos
            self.dof_vel[idx] = 0
        self._pin_goal(idx)

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
                self.cfg.terrain_amplitude,
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
