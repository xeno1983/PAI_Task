# Go2 Pick-and-Carry（提出用）

Unitree Go2 + Z1 で、不整地を含むコース上の箱運びを学習する環境です。
提出手法は **トロット教師（CPG）→ 行動クローニング（BC）→ カリキュラム PPO** です。

## 構成

| パス | 内容 |
| --- | --- |
| `pick_and_carry/` | 環境・報酬・観測・CPG/IK・BC・PPO |
| `scripts/` | 起動補助、スライド／提出動画の生成 |
| `launcher/` | メニュー起動（`PickAndCarry_run.bat`） |
| `tests/` | 単体テスト（Genesis なしでも一部実行可） |
| `assets/urdf/` | Z1 付き簡易 URDF |
| `output/` | 実行時に生成（動画・スライド・曲線） |

## 環境構築（Windows）

```powershell
cd <このフォルダ>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\smoke_local.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## 起動

`PickAndCarry_run.bat` または `launcher\PickAndCarry_run.bat` を実行します。

1. 模倣学習（教師 → BC）
2. 段階学習（歩行 → 到達 → 持ち運び → 箱運び → キャッチ）
3. 推論・録画
4. 比較動画
5. 学習曲線
6. プレゼン動画

コマンド例:

```powershell
# BC
.\.venv\Scripts\python.exe -m pick_and_carry.train_bc --easy-walk --num-envs 32 --seconds 90 --epochs 3000 --log-dir logs\imitation --arm z1
# 歩行 PPO（BC から）
.\.venv\Scripts\python.exe -m pick_and_carry.train_ppo --easy-walk --init-from logs\imitation\bc_trot.pt --num-envs 64 --iterations 1000 --curriculum walk --arm z1 --log-dir logs\ppo_curriculum_walk_easy
```

## 補足

- 画面表示は使わず、結果は `output/` の mp4 で確認する想定です。
- ROS 2 連携は任意です（`ros/`）。必須ではありません。
