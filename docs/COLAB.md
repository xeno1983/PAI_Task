# Colab で実行する

ノートブックは `notebooks/pick_and_carry_colab.ipynb` です。セルは上から順に実行します。起動スクリプトと同じく、段階・学習手法を選んで `train_ppo` で学習・推論します。3D ウィンドウは使わず、確認は mp4 です。

ローカルの `.venv` は Windows 用なので zip に入れません。Colab には CUDA 付き PyTorch が入っているので、Windows 用の cu128 は入れません。

## 1. ランタイムを GPU にする

1. [Colab](https://colab.research.google.com/) で `pick_and_carry_colab.ipynb` を開く。
2. 「ランタイム」→「ランタイムのタイプを変更」→ ハードウェア アクセラレータを **T4 GPU** または **L4 GPU** にする。
3. 変更したあとは「ランタイムを再起動」し、先頭のセルからやり直す。

CPU のまま実行すると、Genesis は `gs.cpu` に落ち、地形の準備で止まります。GPU 確認セルは `nvidia-smi` が無いとそこで止まります。

## 2. Google Drive をマウントする（任意・推奨）

ノート先頭付近の「Google Drive をマウント」セルを実行します。認証ダイアログが出たら許可してください。

```python
from google.colab import drive
drive.mount("/content/drive")
```

Drive にリポジトリを置く場合の例: `MyDrive/PAI_Task`。以前退避した `MyDrive/PAI_Task_logs/logs` があれば、次のセットアップセルがローカルに `logs/` が無いときだけ復元します。

## 3. プロジェクトを置く

ローカルでは、デスクトップの `起動スクリプト\PickAndCarry\PickAndCarry_run.bat` を開き、`6` を選びます。同じフォルダに `PAI_Task_colab.zip` ができます。`.venv`、`logs`、`__pycache__`、`.git` は除いてあります。

Colab のファイルパネルから、この zip をアップロードして展開します。

```python
!unzip -q -o /content/PAI_Task_colab.zip -d /content
```

ノートブックは次の順でプロジェクトを探します。

1. `/content/PAI_Task`
2. `/content/drive/MyDrive/PAI_Task`
3. `/content/drive/MyDrive/content/PAI_Task`

マウントセルを飛ばしても、セットアップセルが未マウントなら自動で `drive.mount` を試します。

## 4. パッケージと GPU を確認する

セットアップの次のコードセルが `requirements.txt` から Genesis と TensorBoard を入れ、PyTorch から GPU が見えることを確認します。`cuda True` と GPU 名が出れば続けます。

描画は EGL を使います。セットアップセルが `PYOPENGL_PLATFORM=egl` を設定します。

## 5. 設定セル（起動スクリプトと同じ）

次の変数を選びます。保存先はローカルと同じです。

| 変数 | 内容 | 例 |
| --- | --- | --- |
| `STAGE` | 段階 | `walk` / `reach` / `haul` / `carry` / `catch` |
| `CATCH_LEVEL` | キャッチの水準 | `static` / `slow` / `random` / `fly` / `noisy` |
| `ALGO` | 学習手法 | `ppo` / `grpo` / `sym_ppo` |
| `NUM_ENVS` | 並列数 | T4 は `8` から。通ったら `32` |
| `ITERATIONS` | 追加学習回数 | `50` など |
| `NUM_STEPS` | 1 iteration のステップ | 起動と同じ `128` |

保存先の例:

- 歩行 PPO → `logs/ppo_curriculum_walk`
- 箱運び GRPO → `logs/grpo_z1`
- キャッチ Symmetric PPO（静止）→ `logs/sym_ppo_curriculum_catch_static`

## 6. 学習

学習セルは次と同等です。

```text
python -m pick_and_carry.train_ppo --resume latest --algo ... --curriculum ... --log-dir ...
```

同じフォルダに重みがあれば続きから、無ければ前段階の重みから、それも無ければ初期値から始めます。

## 7. 推論

推論セルは `--play --resume latest --num-envs 1 --record ...` で mp4 を作り、ノート内で再生します。Colab ではクリック GUI と ROS 2 は使いません。

## 8. 曲線と比較

任意セルで `scripts/plot_curves.py` を呼びます。複数手法を同じ段階で学習していれば、比較 HTML も作れます。

## 9. セッションが切れたとき（Drive へ退避）

無料枠は時間でセッションが切れ、ランタイムのディスクは消えます。切れる前にノート末尾のセルを実行してください。未マウントならその場でマウントし、`logs/` を `MyDrive/PAI_Task_logs` へコピーします。
