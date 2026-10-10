# Colab で実行する

ノートブックは `notebooks/pick_and_carry_colab.ipynb` です。セルは上から順に実行します。3D ウィンドウは使わず、確認用の mp4 をノートの中で再生します。

ローカルの `.venv` は Windows 用なので zip に入れません。Colab には CUDA 付き PyTorch が入っているので、Windows 用の cu128 は入れません。

## 1. ランタイムを GPU にする

1. [Colab](https://colab.research.google.com/) で `pick_and_carry_colab.ipynb` を開く。
2. 「ランタイム」→「ランタイムのタイプを変更」→ ハードウェア アクセラレータを **T4 GPU** または **L4 GPU** にする。
3. 変更したあとは「ランタイムを再起動」し、先頭のセルからやり直す。

CPU のまま実行すると、Genesis は GPU を見つけられず CPU に落ち、地形の準備で止まります。ノートの最初のセルは `nvidia-smi` が無いとそこで止まります。

## 2. プロジェクトを置く

ローカルでは、デスクトップの `起動スクリプト\PickAndCarry_run.bat` を開き、`4` を選びます。同じフォルダに `PAI_Task_colab.zip` ができます。`.venv`、`logs`、`__pycache__`、`.git` は除いてあります。

Colab のファイルパネルから、この zip をアップロードして展開します。

```python
!unzip -q -o /content/PAI_Task_colab.zip -d /content
```

ノートブックは次の順でプロジェクトを探します。

1. `/content/PAI_Task`
2. `/content/drive/MyDrive/PAI_Task`
3. `/content/drive/MyDrive/content/PAI_Task`

ドライブにだけ置いてある場合、先頭セルがドライブをマウントします。`pick_and_carry` が見えないときは、置いた場所をこのどれかに合わせてください。

## 3. パッケージと GPU を確認する

2 番目のコードセルが `requirements.txt` から Genesis と TensorBoard を入れ、PyTorch から GPU が見えることを確認します。`cuda True` と GPU 名が出れば続けます。`False` のときはランタイムを GPU にしてセッションを再起動します。

描画は EGL を使います。先頭セルが `PYOPENGL_PLATFORM=egl` を設定します。

## 4. URDF を確認する

次のセルが観測の並びと、背中への固定ジョイントを表示します。Panda の既定は観測 68、行動 20 です。`mount=base->panda_link0` と `xyz=(-0.05, 0.0, 0.08)` が出れば、プリミティブの URDF を読めます。

`go2.urdf` と `panda.urdf` を同じフォルダに置いたときだけ、次のセルが公式メッシュ版を作ります。無ければプリミティブのまま進みます。公式 Franka は重いので、結合するときは質量スケール `0.25` のままにします。

## 5. 描画を確認する

1 体を 4 秒、台本のトロットで録画し、`logs/colab_preview.mp4` をノートに表示します。初回の `scene.build` はカーネルのコンパイルで数分かかります。Genesis のデバイスが CPU になっていたら、そのセルは学習へ進まず止まります。

ログに `Neutral robot position (qpos0) exceeds joint limits` と出ることがあります。URDF の中立角が一部の関節範囲の端にあるという警告で、`reset()` が立ち姿勢にしたあともシミュレーションは続きます。

## 6. 短く学習して再生する

T4 ではまず 8 環境、5 iteration、1 iteration あたり 16 ステップです。通ったら同じセルの `--num-envs` を 32 まで上げられます。64 以上は、このノートの短い確認には使いません。地形のメッシュが大きく、CPU に落ちた状態で環境数を増やすと準備に非常に時間がかかります。

チェックポイントは `logs/ppo` です。最後のセルが最新の重みを 1 体で 8 秒再生し、`logs/colab_ppo.mp4` を表示します。

ローカルの `logs/ppo_walk` とは別フォルダです。Colab の結果をローカルの続きに使うときは、観測 68・行動 20・Panda であることを確認してから `logs/ppo_walk` へコピーします。

TensorBoard をノートで見る場合は次です。

```python
%load_ext tensorboard
%tensorboard --logdir logs/ppo
```

## 7. セッションが切れたとき

無料枠は時間でセッションが切れ、ランタイムのディスクは消えます。切れたあとは「2. プロジェクトを置く」からやり直します。学習の続きを残すときは、切る前に `logs/` をドライブへコピーします。

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r /content/PAI_Task/logs /content/drive/MyDrive/PAI_Task_logs
```
