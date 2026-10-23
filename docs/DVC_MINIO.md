# Git + MLflow + DVC + MinIO

学習を複数マシン／再開で回すときの役割分担です。

| 層 | 役割 | 置き場 |
| --- | --- | --- |
| Git | コード・設定・DVC メタ（`.dvc` / `dvc.lock`） | GitHub |
| MLflow | 実験パラメータ・メトリクス・（任意）成果物 URI | `logs/mlflow` + MinIO `pai-mlflow` |
| DVC | 重み・抜粋メトリクスなど大きい成果物の版管理 | ポインタは Git、実体は MinIO `pai-dvc` |
| MinIO | S3 互換オブジェクトストレージ | `infra/minio`（ローカル） |
| Mender | 実機への OTA（方策 `best.pt`） | `docs/MENDER_OTA.md` / `artifacts/mender/` |

```
train → logs/<stage>/ppo_*.pt + metrics.jsonl
              │
              ├─ MLflow metrics (毎回)
              └─ dvc_sync publish（区切りで）→ artifacts/checkpoints/<stage>/
                                                    │
                                                    └─ dvc push → MinIO
```

学習ループ中に毎回 `dvc push` はしません（I/O で遅くなるため）。段階の区切りや別 PC へ移すときにだけ publish / pull します。

## 1. MinIO 起動

起動スクリプト（`PickAndCarry_run`）の **メニュー 9** から操作できます。

| 番号 | 内容 |
| --- | --- |
| 9 → 1 | MinIO 起動（必要なら Docker Desktop も起動） |
| 9 → 2 | 初回セットアップ（pip + DVC remote） |
| 9 → 3 | **最良**チェックポイントを publish → push（`metrics.jsonl` で選ぶ） |
| 9 → 4 | **最新** iteration を publish → push |
| 9 → 5 | pull → restore |
| 9 → 6 | DVC の最良重みで推論（pull → `best.pt` → play） |
| 9 → 7 | `dvc status` |
| 9 → 8 | MinIO Console を開く |
| 9 → 9 | Mender Artifact 作成（`best.pt` → `.mender`、OTA 用） |

手動で起動する場合:

```powershell
copy .env.example .env
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_minio.ps1
# または
docker compose -f infra/minio/docker-compose.yml --env-file .env up -d
```

- API: http://127.0.0.1:9000
- Console: http://127.0.0.1:9001 （既定ユーザ / パスワードは `.env`）

> **Image note (2026):** Docker Hub から `minio/minio` / `minio/mc` が消えています。compose は `pull_policy: if_not_present` でローカルキャッシュを使います。未取得のマシンでは quay.io ログイン後に同タグを pull するか、既に持っている `minio/minio:latest` / `minio/mc:latest` を使ってください。

## 2. 依存関係

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dvc.txt
```

## 3. DVC 初期化（初回のみ・リポジトリでは済んでいる想定）

```powershell
.\scripts\load_pai_env.ps1   # .env を読み AWS_* をセット
dvc remote list
```

リモート名は `minio`、URL は `s3://pai-dvc` です。認証は環境変数の `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` を使います。

## 4. チェックポイントを共有する

学習後（または区切りで）:

```powershell
.\scripts\load_pai_env.ps1
# 既定は --select best（metrics ピークに最も近い重み + best.pt）
.\.venv\Scripts\python.exe scripts\dvc_sync.py publish --log-dir logs\ppo_z1 --select best --push
# 最新 iteration だけ欲しいとき
.\.venv\Scripts\python.exe scripts\dvc_sync.py publish --log-dir logs\ppo_curriculum_walk --select latest --push
```

別マシン / クリーンな作業ツリーで戻す・推論する:

```powershell
.\scripts\load_pai_env.ps1
.\.venv\Scripts\python.exe scripts\dvc_sync.py pull
.\.venv\Scripts\python.exe scripts\dvc_sync.py restore --log-dir logs\ppo_z1
# 推論用パス（best.pt）を表示
.\.venv\Scripts\python.exe scripts\dvc_sync.py resolve --name ppo_z1
.\.venv\Scripts\python.exe -m pick_and_carry.train_ppo --play --resume artifacts\checkpoints\ppo_z1\best.pt --num-envs 1 --log-dir logs\ppo_z1
```

`publish` は次を `artifacts/checkpoints/<name>/` に抜粋します。

- `--select best`（既定）: `metrics.jsonl` の段階別スコア最大に近い `ppo_*.pt`
- `--select latest`: 最大 iteration の `ppo_*.pt`
- 常に安定名 `best.pt` と `manifest.json`（選んだ iteration / score）
- `metrics.jsonl` / `run.json` / 模倣重み（あれば）

段階別の「最良」スコア（概略）:

- walk: tracking / walk_score / gait
- reach: approach / reward
- carry・z1・haul: grasp / ee_dist / reward
- catch: success / reward
- その他: reward

## 5. MLflow 成果物を MinIO へ

`.env` を読み込んだ状態で学習すると、`LocalTracker` は可能なら artifact を `s3://pai-mlflow/artifacts` に書きます。メトリクスの SQLite はこれまでどおり `logs/mlflow/mlflow.db` です。

```powershell
.\scripts\load_pai_env.ps1
.\.venv\Scripts\mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db --workers 1
```

## 6. やらないこと

- `logs/` 全体を DVC 管理にしない（毎 iteration の差分が巨大）
- 学習ホットパスで `dvc push` しない
- MinIO のパスワードを Git に載せない（`.env` は gitignore）
