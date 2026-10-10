# Colab で Pick-and-Carry を実行する手順

対象は Genesis 上の Go2 + Franka（Pick-and-Carry）です。ノートブックは `notebooks/pick_and_carry_colab.ipynb` です。Colab の GPU は T4 または L4 を使います。ビューアは使いません。

ローカルの `.venv` は Windows 用なので、Colab には上げません。Colab には CUDA 付き PyTorch が最初から入っています。Windows 用の cu128 を入れ直す必要はありません。

## 1. アップロード用の zip を作る

デスクトップの `PickAndCarry_run.bat` を開き、`4` を選びます。デスクトップに `PAI_Task_colab.zip` ができます。中身から `.venv`、`logs`、`__pycache__` は除いてあります。

同じ zip を手で作る場合は、PowerShell で次を実行します。

```powershell
tar -a -c -f "$env:USERPROFILE\Desktop\PAI_Task_colab.zip" --exclude=.venv --exclude=logs --exclude=__pycache__ --exclude=.git -C C:\Users\xeno1 PAI_Task
```

## 2. Colab のランタイムを GPU にする

1. [https://colab.research.google.com/](https://colab.research.google.com/) で新しいノートブックを開く。
2. 「ランタイム」→「ランタイムのタイプを変更」→ ハードウェアアクセラレータを **GPU** にする。
3. 無料枠は T4、Colab Pro は L4 が付くことが多いです。学習を長く回すときは Pro の方が切れにくいです。

## 3. プロジェクトを置く

左のファイルパネルから `PAI_Task_colab.zip` をアップロードし、先頭のセルで展開します。

```python
!unzip -q -o /content/PAI_Task_colab.zip -d /content
%cd /content/PAI_Task
!ls
```

`pick_and_carry` と `assets` と `requirements.txt` が見えれば成功です。

Google ドライブに置いた場合は、代わりに次を使います。

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/PAI_Task
```

ドライブ上のパスは、実際に置いた場所に合わせてください。

## 4. GPU とパッケージを確認する

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

`True` と `Tesla T4` または `NVIDIA L4` が出れば、そのランタイムは使えます。`False` のときはランタイムを GPU に変えて、セッションを再起動してから同じセルを再実行します。

続けて依存パッケージを入れます。

```python
!pip install -q -r requirements.txt
```

このコマンドは Genesis と TensorBoard を入れます。PyTorch は Colab の CUDA 版をそのまま使います。

## 5. URDF を確認する

プリミティブの統合 URDF でそのまま動きます。

```python
!python scripts/validate_urdf.py
```

`mount=base->panda_link0` と `xyz=(-0.05, 0.0, 0.08)` が出れば、背中への固定ジョイントは読めます。

公式の `go2.urdf` と `panda.urdf` をファイルパネルへ置いたときだけ、次でメッシュ版を作ります。アームが重くて転ぶ場合は質量スケールを `0.25` のままにします。

```python
!python scripts/merge_urdf.py --go2 go2.urdf --panda panda.urdf --arm-mass-scale 0.25
```

`assets/urdf/quadruped_arm_merged.urdf` ができると、環境はこちらを優先して読みます。

## 6. 短い動作確認

初回の `scene.build` はカーネルコンパイルで 1〜3 分かかります。2 回目以降は短くなります。

```python
import torch
from pick_and_carry.config import EnvConfig
from pick_and_carry.env import PickAndCarryEnv

env = PickAndCarryEnv(cfg=EnvConfig(num_envs=4, backend="gpu", show_viewer=False, seed=0))
obs = env.reset()
action = torch.zeros(env.num_envs, env.num_actions, device=obs.device)
obs, reward, done, info = env.step(action)
print(tuple(obs.shape), reward.detach().cpu().tolist())
env.close()
```

観測は `(4, 64)`、行動は 20 次元です。`show_viewer=True` にはしません。Colab には Genesis のウィンドウを出す画面がありません。

ログに `Neutral robot position (qpos0) exceeds joint limits` と出ることがあります。URDF の中立角が子牛関節の範囲外だという警告で、`reset()` が立ち姿勢に置き直したあともシミュレーションは続きます。

## 7. 学習する

T4（約 15 GB）では環境数 64 から始めます。メモリ不足になったら 32 に下げ、余裕があれば 256 まで上げます。L4 も同じ順で増やします。

PPO:

```python
!python -m pick_and_carry.train_ppo --num-envs 64 --iterations 20 --num-steps 32
```

SAC:

```python
!python -m pick_and_carry.train_sac --num-envs 16 --steps 5000
```

チェックポイントと TensorBoard のイベントは `logs/ppo` と `logs/sac` に残ります。ノートブックで見る場合は次です。

```python
%load_ext tensorboard
%tensorboard --logdir logs/ppo
```

## 8. 付属ノートブックを使う場合

`notebooks/pick_and_carry_colab.ipynb` を Colab で開いても同じ流れです。開いたあと、作業ディレクトリが `/content/PAI_Task` であることを最初のコードセルで確認します。セルは上から順に実行します。

1. パッケージのインストールと作業ディレクトリの移動
2. 観測レイアウトと URDF の検証
3. 公式 URDF があるときだけ結合
4. 4 環境の `reset` / `step`
5. 短い PPO（32 環境、5 iteration）

## 9. セッションが切れたとき

無料枠は時間でセッションが切れ、ランタイムのディスクは消えます。切れたあとは「3. プロジェクトを置く」からやり直します。学習の続きを残すときは、`logs/` をドライブへコピーしてからセッションを閉じます。

```python
!cp -r /content/PAI_Task/logs /content/drive/MyDrive/PAI_Task_logs
```
