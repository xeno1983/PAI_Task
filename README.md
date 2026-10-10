# Go2 + Franka Pick-and-Carry（Genesis / Colab）

Genesis 上で、四足の Unitree Go2 の背中に Franka Panda を固定し、立方体を拾って目標エリアまで運ぶ環境です。強化学習のループは PPO と SAC の両方を同梱しています。

## マウント

Go2 の `base` は +X が前、+Z が上です。Franka の `panda_link0` も +Z が上なので、回転なしでアームは直立します。

固定ジョイント `arm_mount_joint` の原点は `base` から `(-0.05, 0, 0.08)` です。前後方向の中心より少し後ろ、胴体上面（約 0.057 m）のすぐ上にフランジが来ます。

`assets/urdf/quadruped_arm.urdf` はメッシュなしのプリミティブ版です。関節名は公式の Go2 / Panda と揃えてあり、アームリンクの質量は合計約 3 kg に落としてあります。公式 URDF を結合したファイルがあると、環境はそちらを優先します。

```bash
python scripts/merge_urdf.py --go2 /path/go2.urdf --panda /path/panda.urdf
```

出力は `assets/urdf/quadruped_arm_merged.urdf` です。メッシュパスは絶対パスに書き換えます。公式 Franka は約 18 kg で Go2（約 15 kg）より重いので、最初の学習では `--arm-mass-scale 0.25` を推奨します。

## 環境 API

タスクは「不整地を渡って物体へ近づく → 平らな足場で拾う → 不整地を渡って運ぶ → 平らな足場で置く」です。段差は二つの移動区間だけで、拾う足場と置く足場は平らです。物体も目標も、腕が届く前に不整地を渡る位置に出ます。エピソードが始まるたびに、その環境の段差の形と高さが変わります。

歩き方は学習です。移動中、方策が立ち姿勢を中心に脚 12 関節を動かします。拾う間と置く間は、脚を立ち姿勢に固定するプログラムです。腕とグリッパは、拾う間と置く間だけ方策が動かし、移動中は既定の姿勢と開閉に固定します。`--scripted-gait` を付けると、以前の対角トロットに戻せます。

`PickAndCarryEnv.reset()` は `(num_envs, 68)` の観測を返します。`step(action)` は `obs, reward, done, info` を返します。`action` は `(num_envs, 20)` で、各成分は `[-1, 1]` です。

| 区間 | 内容 |
| --- | --- |
| 0:12 | 脚の関節角（FL, FR, RL, RR の順に hip, thigh, calf） |
| 12:24 | 脚の関節速度 × 0.05 |
| 24:27 | ベース座標系の重力 |
| 27:30 | ベース線速度 × 2 |
| 30:33 | ベース角速度 × 0.25 |
| 33:34 | ベース高さ |
| 34:38 | 足先の接地フラグ |
| 38:45 | アーム 7 関節角 |
| 45:52 | アーム関節速度 × 0.05 |
| 52:53 | グリッパ開き（0 が閉じ、0.04 が全開） |
| 53:54 | グリッパ速度 × 0.05 |
| 54:57 | 手先位置（ベース座標） |
| 57:60 | 物体位置（ベース座標） |
| 60:63 | 目標位置（ベース座標） |
| 63:64 | 把持フラグ |
| 64:68 | 段階 one-hot（不整地→物体、拾う、不整地→目標、放す） |

行動は `[脚 12 | アーム 7 | グリッパ 1]` です。位置制御では脚とアームは立ち姿勢からの差分です。移動中の脚は最大約 0.5 rad まで方策が動かします。グリッパは `-1` で閉じ、`+1` で全開（0.04 m）です。移動中の報酬は、前方 0.4 m/s に近いほど高く、横滑りすると下がります。`EnvConfig.control_mode="torque"` にすると同じ次元をトルク指令として使います。

報酬は `compute_reward()` が次の項を足します。

- 手先とベースの、物体への接近（未把持のとき）
- 物体の近くでグリッパを閉じる整形項と、把持が成立した瞬間のボーナス
- 把持中の、目標への運搬
- 一度把持した物体を目標半径内の低い位置で離したときの設置ボーナス
- 転倒（傾き 60 度超、またはベース高さ 0.18 m 未満）のペナルティとエピソード終了
- 行動の二乗によるエネルギー正則化
- 生存ボーナス

デフォルトでは、把持判定（手先が物体から 0.10 m 以内かつグリッパ開きが 0.012 m 未満）が真のあいだ、物体を手先に固定します。指先の摩擦だけで持つ段階に進むときは `assist_grasp=False` です。自己衝突は切ってあります。立方体とロボット、地面の衝突は有効です。

## ローカル（Windows + NVIDIA GPU）

このマシンではプロジェクト直下の `.venv` を使います。システム Python の PyTorch は CPU 版のままです。RTX 50 番台は CUDA 12.8 以降の PyTorch が必要なので、Genesis より先に cu128 を入れます。

PowerShell:

```powershell
cd C:\Users\xeno1\PAI_Task
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\smoke_local.py
```

`PYTHONUTF8=1` は、Genesis のログに含まれる罫線文字を cp932 のコンソールが落とさないようにするためです。`smoke_local.py` は環境を 1 つ作り、ゼロ行動で 5 ステップ進めます。初回はシミュレーションカーネルのコンパイルで 1〜2 分かかり、2 回目以降は数秒です。

確認済みの組み合わせは Python 3.12、PyTorch 2.11.0+cu128、Genesis 1.4.3、RTX 5070 Laptop（8 GB）です。

## Colab

GPU ランタイムで、このリポジトリを `/content/PAI_Task` に置きます。

```python
!pip install -q -r requirements.txt
%cd /content/PAI_Task
!python scripts/validate_urdf.py
!python -m pick_and_carry.train_ppo --num-envs 64 --iterations 20
```

初回の `scene.build` はカーネルコンパイルで数十秒から数分かかります。T4 では環境数 64 から始め、メモリが余れば 256 まで上げます。ビューアは Colab ではオフのままにします。TensorBoard のログは `logs/ppo` と `logs/sac` です。

```bash
python -m pick_and_carry.train_sac --num-envs 16 --steps 5000
```

同じ手順は `notebooks/pick_and_carry_colab.ipynb` にも分けてあります。

## 学習の比較動画

PPO は学習前の重みを `logs/ppo/ppo_0000.pt` に残し、その後は `--save-interval`（起動メニューでは 20 iteration）ごとに `ppo_0020.pt` のように保存します。途中経過は、同じ物体と目標で再生して 1 本の動画に並べます。パネルは左上から学習の早い順です。

```powershell
.\.venv\Scripts\python.exe scripts\compare_snapshots.py --seconds 16 --panels 4
```

出力は `デスクトップ\起動スクリプト\PickAndCarry_compare.mp4` です。観測のサイズが違う古い重みはスキップします。

## 学習曲線

PPO は iteration ごとに `logs/ppo/metrics.jsonl` へ報酬、把持率、手先と箱の距離、設置率を追記します。同時に TensorBoard のイベントも `logs/ppo` に残します。実行条件は `logs/ppo/run.json` です。曲線のページは次で作れます。

```powershell
.\.venv\Scripts\python.exe scripts\plot_curves.py
```

出力は `デスクトップ\起動スクリプト\PickAndCarry_curves.html` です。細い線が各 iteration、太い線が移動平均です。TensorBoard を直接見るときは `tensorboard --logdir logs/ppo` です。

## ローカルで確認できること

Genesis がなくても、URDF の整合と報酬の段階はテストできます。

```bash
python scripts/build_primitive_urdf.py
python -m unittest discover -s tests
```
