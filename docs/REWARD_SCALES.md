# 全タスク報酬一覧

Source: `pick_and_carry/config.py` (`RewardScales`, `LocomotionScales`),
`curriculum.py` (`locomotion_reward`, `manipulation_reward`),
`reward.py` / `helpers.py` (`task_reward`).

Default `dt = 0.02`. Updated 2026-10-19 (clamp = positives only; standstill_vel; BC imit on PPO).

## タスク → 報酬関数

| curriculum | 関数 | 係数の置き場 | 備考 |
|---|---|---|---|
| **walk** | `locomotion_reward` | `LocomotionScales` | easy / hard 同じ係数 |
| **reach** | `manipulation_reward`（非 haul） | ハードコード | 静止ターゲット |
| **catch** | `manipulation_reward`（非 haul） | ハードコード | reach と同式。成功は hold 持続 |
| **haul** | `manipulation_reward`（haul） | ハードコード | 箱を持って速度追従 |
| **carry** | `task_reward` → `compute_stage_reward` | `RewardScales` + phase gate | phase 0→3 で項を切替 |

**walk の clamp:** `only_positive_rewards=True` のとき、**正スケールの基底項だけ**を合計して `min 0` に clamp。罰スケール（`lin_vel_z`, `action_rate` 等）は常に加算。その後 `walk_progress` / `gait` / `standstill` / `standstill_vel` / `termination` を加算。

**PPO + BC warm-start:** `--init-from`（または easy-walk 自動）後、既定で `r += −0.1 · MSE(a, a_BC(s))`（`--bc-imit-coef` で変更、`0` で無効）。

---

## walk（`LOCOMOTION_STAGES`）

### 正スケール基底項（clamp 対象）

| 項 | 係数 | 信号 | 適用 | 備考 |
|---|---|---|---|---|
| tracking_lin_vel | 2.5 × dt | `exp(−‖v_xy − cmd_xy‖² / σ)`, σ=0.25 | 毎ステップ | |
| tracking_ang_vel | 0.5 × dt | `exp(−(ω_z − cmd_yaw)² / σ)` | 毎ステップ | |
| orientation | 0.0 × dt | `g_xy²` | — | 係数 0（無効） |
| feet_air_time | 1.5 × dt | 着地時 `Σ(t_air − 0.5)`; ‖cmd_xy‖>0.1 のみ | 着地瞬間 | target=0.5 s |

### 罰スケール基底項（常に加算）

| 項 | 係数 | 信号 | 適用 | 備考 |
|---|---|---|---|---|
| lin_vel_z | −2.0 × dt | `v_z²` | 毎ステップ | |
| ang_vel_xy | −0.05 × dt | `ω_x² + ω_y²` | 毎ステップ | |
| base_height | −20.0 × dt | `(h − 0.30)²` | 毎ステップ | target=0.30 m |
| torques | −1e−5 × dt | `Σ τ_leg²` | 毎ステップ | 脚のみ |
| dof_acc | −2.5e−7 × dt | `Σ ((q̇ − q̇_prev)/dt)²` | 毎ステップ | |
| action_rate | −0.002 × dt | `Σ (a − a_prev)²` | 毎ステップ | |
| collision | −1.0 × dt | 腿・脛の接触数 | 毎ステップ | |

### 基底のあと（常に加算）

| 項 | 係数 | 信号 | 適用 | 備考 |
|---|---|---|---|---|
| walk_progress | 1.5 × dt | `walk_score ∈ [0,1]`（指令速度の達成率） | 後段 | always on |
| gait | 1.5 × dt | `max(hard_trot, soft_trot) × moving` | 後段 | 対角 / 反相同期 |
| standstill | −2.0 × dt | `(1 − walk_score) × moving` | 後段 | `moving`=指令 |
| standstill_vel | −1.0 × dt | `1(‖v_xy‖ < 0.08) × moving` | 後段 | 指令中のほぼ停止 |
| termination | −5.0 | 倒れた / 胴体接地 | 後段・1 回 | ×dt なし |

**ログのみ（報酬に入らない）:** `tracking`, `walk_score`, `steady`, `gait_score`, `jerk`

---

## haul（箱搬送・速度指令）

| 項 | 係数 | 信号 | 適用 | 備考 |
|---|---|---|---|---|
| hold | 2.0 × dt | `exp(−d_ee/0.12) × grasped × upright` | 把持中 | |
| track | 3.0 × dt | `exp(−‖v − cmd‖² / 0.25) × grasped × upright` | 把持中 | |
| alive | 0.3 × dt | `grasped × upright` | 把持中 | |
| effort | −0.005 × dt | `Σ a²` | 毎ステップ | |
| fall | −5.0 | fallen | 1 回 | |
| drop | −2.0 | 未把持かつ未転倒 | 毎ステップ | 落とした罰 |

**ログのみ:** `tracking`, `walk_score`

---

## reach / catch（手先到達・捕捉・同式）

| 項 | 係数 | 信号 | 適用 | 備考 |
|---|---|---|---|---|
| approach | 8.0 | `clamp(Δd, ±0.05) × upright` | 毎ステップ | ×dt なし |
| near | 1.5 × dt | `exp(−d/0.12) × upright` | 毎ステップ | |
| align | 0.4 × dt | 手→球と球の来方向の一致 `[0,1]` | 毎ステップ | 静止球は 1 |
| catch | 0.5 | `d<0.09 ∧ ‖v_rel‖<0.25 ∧ closed ∧ ¬fallen` | 条件成立時 | |
| alive | 0.2 × dt | upright | 毎ステップ | |
| effort | −0.005 × dt | `Σ a²` | 毎ステップ | |
| fall | −5.0 | fallen | 1 回 | |
| still | −1.0 × dt | `v_xy² + ω_z²` | 毎ステップ | 脚は静止 |

catch のエピソード成功は `hold_steps >= catch_hold_steps`（環境側）。

---

## carry（Pick-and-Carry · phase gated）

### フェーズ

| phase | 意味 | 有効な主な項 |
|---|---|---|
| 0 | 物体へ歩行 | `approach_base`, `progress_base`, `forward` |
| 1 | ピックアップ | `approach_ee`, `grasp_shape`, `progress_ee`, `grasp_bonus` |
| 2 | ゴールへ歩行 | `transport`, `progress_goal`, `forward` |
| 3 | リリース | `release`（開握）, `place` |

### 項

| 項 | 係数 | 信号 | 適用 | 備考 |
|---|---|---|---|---|
| approach_ee | 1.5 × dt | `exp(−d_ee / 0.25) × free` | phase1 pickup | σ=`approach_sigma` |
| approach_base | 0.4 × dt | `exp(−d_base / 0.60) × free` | phase0 →物体 | σ=`base_sigma` |
| grasp_shape | 6.0 × dt | `close_frac × near_ee × free` | phase1 | |
| grasp_bonus | 5.0 | newly_grasped | イベント | スパース |
| transport | 2.0 × dt | `exp(−d_goal / 0.40) × held` | phase2 →ゴール | σ=`transport_sigma` |
| place | 10.0 | newly_placed | イベント | スパース |
| progress_ee | 8.0 | `clamp(Δd_ee, ±0.05) × pickup` | phase1 | ×dt なし |
| progress_goal | 8.0 | `clamp(Δd_goal, ±0.05) × walk_goal` | phase2 | |
| progress_base | 8.0 | `clamp(Δd_base, ±0.05) × walk_object` | phase0 | |
| forward | 4.0 × dt（split） | `exp(−((vx−0.4)/0.3)²) − 0.4·slip`; slip=`vy²+(0.5 ω_z)²` | 歩行 phase | `control_split=walk` は **6.0 × dt** |
| release | 4.0 × dt | `open_frac × release` | phase3 | |
| fall | −5.0 | fallen | 1 回 | |
| energy | −0.02 × dt | `Σ a²` | 毎ステップ | |
| alive | 0.3 × dt | 定数 | 毎ステップ | |

`control_split=walk` のときは操作系項をゼロにし、`forward` 係数だけ 6.0×dt。通常の carry は `split`。`ee_dist` はログのみ。

---

## 補足

- 正 = 接近・追従・歩容・イベントボーナス。負 = 罰（転倒・トルク・静止・エネルギー）。
- ValueDICE 学習時は環境報酬に加え `imit_coef × centered ν + env_coef × env` を混合（デフォルト `env_coef=10`, imit≈2）。上表は環境報酬そのもの。
