# Go2 Pick-and-Carry（Genesis）

Unitree Go2 の背中に腕を固定し、不整地を渡って立方体を拾い、目標へ置く環境です。学習は PPO が本体で、短い SAC も同梱しています。画面は出さず、結果は mp4 で確認します。この GPU では Genesis の 3D ウィンドウが落ちるため、`--viewer` は使いません。

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

Genesis がなくても、報酬と URDF のテストは通ります。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## 3. 起動スクリプトで実行する

デスクトップの `起動スクリプト\PickAndCarry_run.bat` をダブルクリックします。中身は `launcher\PickAndCarry_run.ps1` と同じメニューです。ログの窓は、`exit code` が出るまで閉じません。

| 番号 | 内容 | 主な出力 |
| --- | --- | --- |
| 1 | 8 秒間立たせて録画 | `PickAndCarry_smoke.mp4`（Z1 なら `PickAndCarry_z1_smoke.mp4`） |
| 2 | PPO で学習し、終わると曲線と 20 秒の再生 | `PickAndCarry_ppo.mp4` または `PickAndCarry_z1.mp4` |
| 3 | 短い SAC（Panda のみ、8 環境、500 ステップ） | `PickAndCarry_sac.mp4` |
| 4 | Colab 用 zip | `起動スクリプト\PAI_Task_colab.zip` |
| 5 | クリックで物資と運び先を決めて再生 | 同じ mp4 |
| 6 | 学習の早い時点と遅い時点を 1 本に並べる | `PickAndCarry_compare.mp4` |
| 7 | 学習曲線をブラウザで開く | `PickAndCarry_curves.html` |
| 8 | スライドと録画を、説明音声つきの 1 本にする | `PickAndCarry_presentation.mp4` |

出力ファイルは、bat と同じ `起動スクリプト` フォルダにできます。

2 番を選ぶと、次を順に聞きます。Enter は括弧内の既定です。

1. 腕。Panda（既定）か Z1。
2. 学習回数（1000）。今の iteration にこの回数を足します。
3. 同時に動かすロボット数（16）。32 を超えると 8 GB の GPU ではメモリ不足になることがあります。手先カメラを使うと、同じ台数でも遅くなります。
4. 走行環境。低い波 3.5 cm、段差 4.5 cm（既定）、段差 6 cm。段は箱の前に 1 段、置く前に 2 段で、箱と目標の下は平らです。
5. 学習の分け方。歩行のみ、歩行と把持を分割（既定）、同時。
6. 対象物の見つけ方。シミュレータの位置（既定）か、手先カメラ。

途中で止めても、最後に `saved ppo_XXXX.pt` と出た時点から次回続けられます。保存は 20 iteration ごとです。画面の iteration がファイルより先に進んでいることがあります。

## 4. コマンドで実行する

起動スクリプトと同じ学習を、プロジェクトルートから直接行う例です。Panda の続きは `logs/ppo_walk`、Z1 は `logs/ppo_z1` です。引数を付けない `train_ppo` の既定ログは `logs/ppo` なので、続きから学ぶときは `--log-dir` を明示します。

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pick_and_carry.train_ppo --resume latest --num-envs 16 --iterations 1000 --num-steps 128 --save-interval 20 --log-dir logs/ppo_walk --terrain-steps --terrain-amplitude 0.045
```

Z1 で最初から学ぶときは `--arm z1 --log-dir logs/ppo_z1` を付けます。Panda の重みは寸法が違うので読みません。

再生は学習せず、保存した重みで mp4 を書きます。

```powershell
.\.venv\Scripts\python.exe -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --seconds 20 --record "$env:USERPROFILE\Desktop\起動スクリプト\PickAndCarry_ppo.mp4" --log-dir logs/ppo_walk --terrain-steps --terrain-amplitude 0.045
```

よく使う引数は次です。

| 引数 | 意味 |
| --- | --- |
| `--arm panda` / `--arm z1` | 腕。既定は panda |
| `--control-split split` / `walk` / `both` | 歩行と把持の分け方。既定は split |
| `--wrist-camera` | 手先カメラで箱を見てからつかむ |
| `--terrain-steps` | 移動区間に段差を足す |
| `--terrain-amplitude 0.045` | 波または段の高さ（メートル） |
| `--resume latest` | そのログフォルダの最新の適合する重みから続ける |
| `--play` | 学習せず再生する |
| `--no-assist-grasp` | 指の接触だけで持つ。既定は、つかんだ判定のあいだ箱を手先へ固定する |

学習曲線と比較動画です。

```powershell
.\.venv\Scripts\python.exe scripts\plot_curves.py --log-dir logs/ppo_walk --output "$env:USERPROFILE\Desktop\起動スクリプト\PickAndCarry_curves.html"
.\.venv\Scripts\python.exe scripts\compare_snapshots.py --log-dir logs/ppo_walk --seconds 16 --panels 4 --terrain-steps --terrain-amplitude 0.045
```

曲線は報酬、把持率、手先と箱の距離、設置率です。TensorBoard は `tensorboard --logdir logs/ppo_walk` です。ここには方策 Loss、価値 Loss、エントロピーも残ります。

## 5. タスクの要点

コースは、平地から出発し、不整地、拾う平地、不整地、置く平地の順です。1 試行は 20 秒（`dt` 0.02 秒で 1000 ステップ）です。転倒は傾き 60 度超、またはベース高さ 0.18 m 未満です。時間切れは失敗ではなく、価値関数で次状態を補います。

歩き方は方策が学びます。`--control-split` の 3 通りは観測の次元を変えません。

- `split`（既定）: 歩いている間は脚。拾う間と置く間は腕と指。
- `walk`: 脚だけ。腕は固定、指は開いたまま。報酬は前進と、転ばないこと。
- `both`: 脚・腕・指を試行のあいだずっと出す。

腕の初期姿勢で、手先は胴体の前方およそ 0.4 m に来ます。箱は一辺 4 cm です。つかんだとみなす距離は 0.12 m、指が閉じた基準は Panda で開き 0.012 m 未満です。

### 観測と行動（Panda）

`reset()` は `(環境数, 68)`、`step(action)` は `obs, reward, done, info` を返します。行動は `(環境数, 20)` で、各成分は -1 から 1 です。Z1 は腕が 6 関節なので、観測 66、行動 19 です。並びは同じで、腕の角度と角速度だけ 1 関節分短くなります。

| 区間 | 内容 |
| --- | --- |
| 0:12 | 脚の関節角（FL, FR, RL, RR の順に hip, thigh, calf） |
| 12:24 | 脚の関節速度 × 0.05 |
| 24:27 | ベース座標系の重力 |
| 27:30 | ベース線速度 × 2 |
| 30:33 | ベース角速度 × 0.25 |
| 33:34 | ベース高さ |
| 34:38 | 足先の接地（接触力 1 N 超） |
| 38:45 | 腕の関節角（Panda は 7、Z1 は 6） |
| 45:52 | 腕の関節速度 × 0.05 |
| 52:53 | グリッパ。Panda は 0 が閉じ、0.04 m が全開 |
| 53:54 | グリッパ速度 × 0.05 |
| 54:57 | 手先位置（ベース座標） |
| 57:60 | 物体位置（ベース座標）。手先カメラで見えないときは (0, 0, -1) |
| 60:63 | 目標位置（ベース座標） |
| 63:64 | 把持フラグ |
| 64:68 | 段階 one-hot（箱へ歩く、拾う、運ぶ、置く） |

行動は `[脚 12 | 腕 | グリッパ 1]` です。脚と腕は立ち姿勢からの差分で、脚は最大約 0.5 rad です。グリッパは -1 で閉じ、+1 で開きます。

報酬の式は `pick_and_carry/reward.py` です。前進（秒速 0.40 m 付近）、接近、把持、運搬、設置（+10）、転倒（-5）、操作量、生存を足します。歩行だけの学習では把持と設置の点は出しません。`ee_dist` は記録用で、報酬には足しません。

## 6. 腕

既定は Franka Panda です。実機は約 18 kg で Go2 の定格積載（約 8 kg）を超えます。学習用 URDF `assets/urdf/quadruped_arm.urdf` は腕の質量を約 3.4 kg に落としてあり、見た目のメッシュは公式の形状を参照します。保存済みの Panda の重みは、この 68 次元のまま続きます。

Unitree Z1 は `--arm z1` です。公式の関節位置と慣性を使い、グリッパ込みで約 5.2 kg、6 軸です。URDF は `assets/urdf/quadruped_z1.urdf` です。観測が変わるので、Panda の重みは読まず `logs/ppo_z1` で最初から学習します。

Go2 の `base` は +X が前、+Z が上です。腕は `arm_mount_joint` で `(-0.05, 0, 0.08)` に固定します。胴体の少し後ろ、上面のすぐ上です。

## 7. Colab

手順は [docs/COLAB.md](docs/COLAB.md) です。ノートブックは `notebooks/pick_and_carry_colab.ipynb` です。GPU ランタイムが必須で、CPU のままでは地形の準備で止まります。短い確認は 8 環境です。3D ウィンドウは使わず、mp4 をノートの中で再生します。

## 8. コードの場所

シミュレータがなくても読める計算は次のファイルです。

| ファイル | 内容 |
| --- | --- |
| `pick_and_carry/config.py` | 環境設定、Panda の観測 68 と行動 20 |
| `pick_and_carry/arms.py` | Panda と Z1 の関節、質量の前提、次元 |
| `pick_and_carry/observation.py` | 観測ベクトルの組み立て |
| `pick_and_carry/reward.py` | 段階ごとの報酬 |
| `pick_and_carry/control.py` | 脚・腕・指の目標角度 |
| `pick_and_carry/helpers.py` | 座標変換、把持・設置・転倒、手先カメラの重心 |
| `pick_and_carry/env.py` | Genesis のシーン |
| `pick_and_carry/agents.py` | PPO の方策と clip 付きの更新 |
| `pick_and_carry/train_ppo.py` | 学習と再生 |
