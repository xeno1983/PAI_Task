# Go2 Pick-and-Carry（Genesis）

Unitree Go2 の背中に腕を固定し、不整地を渡って立方体を拾い、目標へ置く環境です。学習は PPO が既定で、起動時に GRPO と Symmetric PPO（左右対称の増強）も選べます。手法ごとの保存先は `logs/{algo}_*` で、メニュー 8 で曲線を重ねて比較できます。画面は出さず、結果は mp4 で確認します。この GPU では Genesis の 3D ウィンドウが落ちるため、`--viewer` は使いません。

確認済みの組み合わせは Windows、Python 3.12、PyTorch 2.11.0+cu128、Genesis 1.4.3、RTX 5070 Laptop（8 GB）です。

## 1. 必要なもの

- NVIDIA GPU。RTX 50 番台は CUDA 12.8 以降の PyTorch が必要です。CPU 版の torch では動きません。
- Python 3.12。
- プロジェクトを置いたフォルダ。以下ではその場所をプロジェクトルートと呼びます。

## 2. 環境構築（Windows）

PowerShell で、プロジェクトルートに仮想環境を作ります。システム Python の PyTorch は触りません。Genesis より先に CUDA 付き PyTorch を入れます。

```powershell
cd <プロジェクトルート>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` は Genesis、TensorBoard、NumPy です。

文字化けを避けるため、以後のコマンドの前に次を置きます。起動スクリプトはこれを自分で設定します。

```powershell
$env:PYTHONUTF8 = "1"
```

動作確認は、ビューアなしで 1 体を数ステップ進めます。初回はシミュレーションカーネルのコンパイルで 1〜2 分かかり、2 回目以降は十数秒です。最後に `smoke ok` と出れば成功です。

```powershell
.\.venv\Scripts\python.exe scripts\smoke_local.py
```

Genesis がなくても、報酬と方策のテストは通ります。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## 3. 起動スクリプトで実行する

デスクトップの `起動スクリプト\PickAndCarry\PickAndCarry_run.bat` をダブルクリックします。中身は `launcher\PickAndCarry_run.ps1` と同じメニューです。ログの窓は、`exit code` が出るまで閉じません。

| 番号 | 内容 | 主な出力 |
| --- | --- | --- |
| 1 | 模倣学習（トロット教師 → BC / GAIL）。デモ保存の有無を選べる | `bc_trot.pt` / `gail_trot.pt`、`videos/...` |
| 2 | 段階を選んで学習（歩行 / 到達 / 持ち運び / 箱運び / キャッチ） | 各 `logs/{algo}_*` |
| 3 | 推論。段階を選んで保存した重みを再生 | mp4（`videos/<タスク>/...`） |
| 4 | 箱運びの途中経過を並べた比較動画 | `PickAndCarry_z1_compare.mp4` |
| 5 | 学習曲線をブラウザで開く | `PickAndCarry_curves.html` |
| 6 | スライドと録画を、説明音声つきの 1 本にする | `PickAndCarry_presentation.mp4` |
| 7 | Colab 用 zip | `起動スクリプト\PickAndCarry\PAI_Task_colab.zip` |
| 8 | 学習手法（PPO / GRPO / Symmetric PPO）の曲線比較 | `PickAndCarry_compare_curves.html` |

出力ファイルは、bat と同じ `起動スクリプト\PickAndCarry` フォルダにできます。デモ・再生 mp4 は `logs/.../videos/<タスク>/<学習回>/` に分かれ、起動フォルダへもコピーします。

2 番を選ぶと、次を順に聞きます。Enter は括弧内の既定です。

1. カリキュラム段階（歩行 / 到達 / 持ち運び / 箱運び / キャッチ）。歩行→簡単では BC / GAIL 暖機も選べます。
2. 学習回数。今の iteration にこの回数を足します。
3. 同時に動かすロボット数（既定 64）。
4. 歩行では地形の高さ。到達以降は平地です。

途中で止めても、最後に `saved ppo_XXXX.pt` と出た時点から次回続けられます。保存は 20 iteration ごとです。画面の iteration がファイルより先に進んでいることがあります。

3 番の箱運び推論では、クリックで物資と運び先を決められます。ROS 2 を選ぶと rviz2 のクリックで運び先だけが動きます。物資の位置は、最初のクリック画面のままです。

## 4. コマンドで実行する

起動スクリプトと同じ学習を、プロジェクトルートから直接行う例です。歩行は `logs/ppo_curriculum_walk`（簡単モードは `*_walk_easy`）、箱運びは `logs/ppo_z1` です。`--resume latest` は iteration の数値順で最新を選びます。

```powershell
$env:PYTHONUTF8 = "1"
# 1a) トロット教師 → BC。出力: logs/ppo_curriculum_walk_easy/bc_trot.pt
.\.venv\Scripts\python.exe -m pick_and_carry.train_bc --easy-walk --num-envs 32 --seconds 90 --epochs 80
# 1b) トロット教師 → GAIL（BC 暖機あり）。出力: logs/ppo_curriculum_walk_easy/gail_trot.pt
.\.venv\Scripts\python.exe -m pick_and_carry.train_gail --easy-walk --num-envs 32 --seconds 90 --bc-epochs 40 --iterations 1000
# 追加学習（既存 gail_trot.pt から）
.\.venv\Scripts\python.exe -m pick_and_carry.train_gail --easy-walk --resume latest --iterations 1000
# 2) 模倣重みから easy-walk PPO（--init-from を省略すると gail_trot.pt → bc_trot.pt の順で自動採用）
.\.venv\Scripts\python.exe -m pick_and_carry.train_ppo --easy-walk --resume latest --num-envs 64 --iterations 1000 --num-steps 128 --save-interval 20 --curriculum walk --arm z1
# 明示する場合:
# --init-from logs/ppo_curriculum_walk_easy/gail_trot.pt
# --init-from logs/ppo_curriculum_walk_easy/bc_trot.pt
```

起動メニュー:

- **1** … 模倣学習（BC / GAIL）。教師デモ mp4 の保存有無を選べます。動画は `logs/.../videos/<タスク>/<種別>_ep####_iter######_<時刻>/` に分けて保存（リール `*_demo.mp4` とスキル別クリップ）
- **2 → 歩行 → 簡単 → BC / GAIL してから PPO** … 学習の一連（ここでも教師デモ保存の有無を選べます）
- **3 → 歩行 → 簡単** … 模倣重みや教師デモの再生

教師は立つ・座る・ジャンプ・前後進・左右旋回・停止と、対角トロット歩行を CPG パラメータ（10 次元）として出します。環境側の CPG が 12 関節指令へ展開します。GAIL は識別器報酬 `-log(1-D(s,a))` で PPO を回します（既定は環境報酬なし・BC 暖機あり）。`--easy-walk` で PPO チェックポイントが無いとき、`--init-from` 未指定なら `gail_trot.pt` があればそれを、無ければ `bc_trot.pt` を自動で載せます（起動スクリプトと同じ順）。

再生は学習せず、保存した重みで mp4 を書きます。保存先は同様に `videos/<タスク>/<algo>_iter######_<時刻>/` です（起動フォルダ側にも同じ木をコピーします）。

```powershell
.\.venv\Scripts\python.exe -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --seconds 20 --record "$env:USERPROFILE\Desktop\起動スクリプト\PickAndCarry\PickAndCarry_ppo.mp4" --log-dir logs/ppo_curriculum_walk --curriculum walk --arm z1
```

よく使う引数は次です。

| 引数 | 意味 |
| --- | --- |
| `--arm z1` | 腕。Unitree Z1（既定） |
| `--algo ppo` / `grpo` / `sym_ppo` | 学習手法。保存先は `logs/{algo}_*` |
| `--control-split split` / `walk` / `both` | 歩行と把持の分け方。既定は split |
| `--wrist-camera` | 手先カメラで箱を見てからつかむ |
| `--terrain-steps` | 移動区間に段差を足す |
| `--terrain-amplitude 0.045` | 波または段の高さ（メートル） |
| `--resume latest` | そのログフォルダの最新の適合する重みから続ける（iteration 数値順） |
| `--init-from PATH` | 模倣重みなどから重みだけ載せて iteration 0 から PPO |
| `--easy-walk` | 平坦・腕固定・押しなし。保存先 `*_walk_easy`。PPO 無し時は `gail_trot.pt` → `bc_trot.pt` を自動初期化 |
| `--play` | 学習せず再生する |
| `--ros` | 推論だけ。状態を送り、実行中の運び先を受け取る。学習中に付けると止まる |

### 推論中に運び先を動かす

Genesis の画面は開きません。表示は WSL2 の rviz2 です。シミュレーションは今の Windows の仮想環境のままです。ROS の Python はこちらには入れません。

WSL で、プロジェクト（例: `/mnt/c/Users/xeno1/PAI_Task`）に移動してから次を実行します。`ros-<distro>-desktop` が入っている必要があります。

```bash
bash ros/display.sh z1
```

Z1 のときは `bash ros/display.sh z1` です。箱の URDF を rviz2 に出します。公式メッシュは使いません。

Windows では起動スクリプトで「ROS 2 で表示する」を選びます。rviz2 は WSL の Ubuntu（ROS 2 Jazzy）で開き、姿勢は共有フォルダ経由で渡します。手で開くときは、先に WSL で `bash ros/display.sh z1` を実行します。

```powershell
.\.venv\Scripts\python.exe scripts\infer_gui.py --log-dir logs/ppo_curriculum_walk --record "$env:USERPROFILE\Desktop\起動スクリプト\PickAndCarry\PickAndCarry_ppo.mp4" --seconds 60 --ros --curriculum walk --arm z1
```

クリック画面の 1 点目が物資、2 点目が最初の運び先です。起動スクリプトの 5 番で ROS 2 を選ぶと rviz2 が開きます。`world` 座標は最初から出るので、ロボットモデルのエラーは推論を待たなくても消えます。推論が始まると、Windows にシミュレーション画面が開いて順に更新されます。rviz2 上部の Publish Point で地面をクリックすると、黄色い球がその場に出ます。推論中なら運び先がそこへ変わり、シミュレーション画面の緑の位置も動きます。コースの外はコース内に収めます。終わったあとの mp4 も残ります。ROS 2 を選ばないときは、これまでどおり推論が終わってから動画を開きます。

荷物を持っているときに遠くをクリックすると、また運び先へ歩きます。近くをクリックすると、置く動作に入ります。まだ掴む前のクリックは、置き場所の記憶だけを更新します。時間切れでは出発点に戻りません。転倒したときと、置き終えたときは次の試行が始まり、運び先は最後のクリックのままです。

学習曲線と比較動画です。

```powershell
.\.venv\Scripts\python.exe scripts\plot_curves.py --log-dir logs/ppo_walk --output "$env:USERPROFILE\Desktop\起動スクリプト\PickAndCarry\PickAndCarry_curves.html"
.\.venv\Scripts\python.exe scripts\compare_snapshots.py --log-dir logs/ppo_walk --seconds 16 --panels 4 --terrain-steps --terrain-amplitude 0.045
```

曲線は報酬、把持率、手先と箱の距離、設置率です。TensorBoard は `tensorboard --logdir logs/ppo_walk` です。ここには方策 Loss、価値 Loss、エントロピーも残ります。

## 5. タスクの要点

コースは、平地から出発し、不整地、拾う平地、不整地、置く平地の順です。1 試行は 20 秒（`dt` 0.02 秒で 1000 ステップ）です。転倒は傾き 60 度超、またはベース高さ 0.18 m 未満です。時間切れは失敗ではなく、価値関数で次状態を補います。

歩き方は方策が学びます。`--control-split` の 3 通りは観測の次元を変えません。

- `split`（既定）: 歩いている間は脚。拾う間と置く間は腕と指。
- `walk`: 脚だけ。腕は固定、指は開いたまま。報酬は前進と、転ばないこと。
- `both`: 脚・腕・指を試行のあいだずっと出す。

腕の初期姿勢で、手先は胴体の前方およそ 0.4 m に来ます。箱は一辺 4 cm です。つかんだとみなす距離は 0.12 m、指が閉じた基準は Z1 のグリッパ関節で閉じ側です。

### 段階的な学習

段階は `--curriculum` で選びます。全段階が観測 81・行動 22（CPG パラメータ + 腕）で、重みは次の順に引き継ぎます。

| 段階 | 内容 | 重みの元 |
| --- | --- | --- |
| `walk` | 立つ・歩く・旋回。平地から始め、成績が良いと傾斜と段差へ広げる。指令の 20% は静止。腕は勝手に動かす。ときどき押し、床の摩擦もばらす | なし |
| `reach` | 胴の前の空中の点へ手先を届かせる。脚は歩行の姿勢を保つ | `walk` |
| `haul` | 箱を持ったまま歩く | `reach` |
| `carry` | ピック＆プレース（箱運び）。`logs/ppo_z1` | `haul` |
| `catch` | ボールのキャッチ。`--catch-level` は `static`、`slow`、`random`、`fly`、`noisy` | `reach`（上の水準から順に） |

旧い立つ・座る・ジャンプ・不整地の段階は、歩行に統合しました。以前の `logs/ppo_curriculum_stand` などは使いません。

`catch` の観測は、物体位置の欄にボール位置、目標位置の欄にボール速度 × 0.5、速度指令の欄に 0.3 秒先の予測位置 × 0.1 を入れます。`fly` と `noisy` は転がしたあと宙を飛ばし、床に落ちると試行を終えます。

### 観測と行動（Z1）

`reset()` は `(環境数, 81)`、`step(action)` は `obs, reward, done, info` を返します。行動は `(環境数, 22)` で、各成分はおおよそ -1 から 1 です。

| 区間 | 内容 |
| --- | --- |
| 0:12 | 脚の関節角（立ち姿勢との差。FL, FR, RL, RR の順に hip, thigh, calf） |
| 12:24 | 脚の関節速度 × 0.05 |
| 24:27 | ベース座標系の重力 |
| 27:30 | ベース線速度 × 2 |
| 30:33 | ベース角速度 × 0.25 |
| 33:34 | ベース高さ |
| 34:38 | 足先の接地（接触力 1 N 超） |
| 38:44 | 腕の関節角 |
| 44:50 | 腕の関節速度 × 0.05 |
| 50:51 | グリッパ |
| 51:52 | グリッパ速度 × 0.05 |
| 52:55 | 手先位置（ベース座標） |
| 55:58 | 物体位置（ベース座標）。到達・キャッチではボール。手先カメラで見えないときは (0, 0, -1) |
| 58:61 | 目標位置（ベース座標）。キャッチではボール速度 × 0.5 |
| 61:62 | 把持フラグ |
| 62:66 | 段階 one-hot（箱へ歩く、拾う、運ぶ、置く） |
| 66:69 | 速度指令。キャッチでは 0.3 秒先のボール位置 × 0.1 |
| 69:81 | 直前の脚の指令（CPG が展開した 12 関節） |

行動は `[CPG 15 | 腕 Δq 6 | グリッパ Δg 1]` です。脚は環境内の対角トロット CPG（`pick_and_carry/cpg.py`）が脚ごとの位相を進め、12 関節指令へ展開してから PD（kp 60、kd 1.5）で追従します。CPG は対角オフセット・軟結合・接地引き込み（`dphi_*` / `couple` / `entrain`）を方策から調整できます。reach / haul / carry / catch では腕は数値 IK（`pick_and_carry/ik.py`）が EE 目標へベース姿勢を出し、方策は小さな残差（×0.12 rad）だけを足します。グリッパも段階ベースの開閉に残差（×0.35 rad）を足します。walk では腕は方策外です。旧・行動 17（CPG 10）の重みは使えません。

歩行の報酬は legged_gym の標準の組み合わせです（速度の追従、縦揺れ・横揺れ、高さ 0.30 m、トルク、加速度、行動の変化、足を上げた時間、衝突、終了）。方策は 512×256×128 の ELU で、行動は正規分布です。学習率は KL 0.01 を目標に自動で調整します。

箱運びの報酬の式は `pick_and_carry/reward.py` です。到達とキャッチは `pick_and_carry/curriculum.py` の `manipulation_reward`（手先の接近、速度の一致、保持）です。前進（秒速 0.40 m 付近）、接近、把持、運搬、設置（+10）、転倒（-5）、操作量、生存を足します。歩行だけの学習では把持と設置の点は出しません。`ee_dist` は記録用で、報酬には足しません。

## 6. 腕

腕は Unitree Z1 のみです。公式の関節位置と慣性を使い、グリッパ込みで約 5.2 kg、6 軸です。URDF は `assets/urdf/quadruped_z1.urdf` です。観測 81、行動 22（CPG+腕）です。

Go2 の `base` は +X が前、+Z が上です。腕は `arm_mount_joint` で `(-0.05, 0, 0.08)` に固定します。胴体の少し後ろ、上面のすぐ上です。

## 7. Colab

手順は [docs/COLAB.md](docs/COLAB.md) です。ノートブックは `notebooks/pick_and_carry_colab.ipynb` です。起動スクリプトと同じく段階（walk/reach/haul/carry/catch）と手法（PPO/GRPO/Symmetric PPO）を選び、同じ `logs/{algo}_*` に学習・推論します。GPU ランタイムが必須です。3D ウィンドウは使わず、mp4 をノートの中で再生します。

## 8. 実機への適用

シミュレータで学習できた方策を Go2 + Z1 実機へ載せる手順・観測行動の契約・安全な段階適用は [docs/SIM_TO_REAL.md](docs/SIM_TO_REAL.md) です。現状リポジトリは実機 SDK 送信までは含みません（rviz 表示用の ROS ブリッジまで）。

## 9. コードの場所

シミュレータがなくても読める計算は次のファイルです。

| ファイル | 内容 |
| --- | --- |
| `pick_and_carry/config.py` | 環境設定、Z1 の観測 81 と行動 22（CPG 15） |
| `pick_and_carry/cpg.py` | 脚用の学習可能 CPG（パラメータ → 12 関節） |
| `pick_and_carry/ik.py` | Z1 の FK / 数値 IK（腕ベース指令） |
| `pick_and_carry/video_paths.py` | デモ・再生 mp4 の `videos/<タスク>/<学習回>/` 配置 |
| `pick_and_carry/arms.py` | Z1 の関節、質量の前提、次元 |
| `pick_and_carry/observation.py` | 観測ベクトルの組み立て |
| `pick_and_carry/reward.py` | 箱運びの報酬 |
| `pick_and_carry/curriculum.py` | 段階の定義、重みの引き継ぎ、歩行・到達・キャッチの報酬 |
| `pick_and_carry/control.py` | 脚・腕・指の目標角度 |
| `pick_and_carry/helpers.py` | 座標変換、把持・設置・転倒、手先カメラの重心 |
| `pick_and_carry/env.py` | Genesis のシーン |
| `pick_and_carry/agents.py` | PPO の方策と clip 付きの更新 |
| `pick_and_carry/train_ppo.py` | 学習と再生（easy-walk の模倣自動初期化含む） |
| `pick_and_carry/train_bc.py` | トロット教師デモの Behavioral Cloning |
| `pick_and_carry/train_gail.py` | トロット教師デモの GAIL |
| `pick_and_carry/checkpoints.py` | 重みの保存、互換する最新の探索、古い保存の退避 |
| `pick_and_carry/difficulty.py` | 学習中の難しさ（地形、ボール）を、成績で上げ下げする |
| `pick_and_carry/ros_bridge.py` | 推論中の状態送信と運び先の受信。ROS 自体は入れない |
| `ros/bridge_node.py` | WSL 側で関節、姿勢、マーカーに変換し、クリックを返す |
| `ros/display.sh` | robot_state_publisher と rviz2 を起動する |
