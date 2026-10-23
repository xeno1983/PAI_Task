# 新しい Windows 環境でのセットアップ

Pick-and-Carry（Genesis）を別 PC に入れるときの手順です。確認済みの目安は **Windows 10/11、Python 3.12、PyTorch（CUDA 12.8）、Genesis 1.4.x、NVIDIA GPU** です。

---

## 0. 事前に入れるもの

| ソフト | 用途 | 備考 |
| --- | --- | --- |
| NVIDIA ドライバ | GPU | RTX 50 番台は新しいドライバが必要 |
| [Python 3.12](https://www.python.org/downloads/) | 実行環境 | 「Add python.exe to PATH」をオン |
| [Git for Windows](https://git-scm.com/download/win) | リポジトリ取得 | |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | MinIO（任意） | 起動したまま使う |
| PowerShell 5+ / 7 | 起動スクリプト | Windows 標準で可 |

任意:

| ソフト | 用途 |
| --- | --- |
| WSL2 + Ubuntu + ROS 2 | rviz2 で運び先表示（推論メニュー） |
| `mender-artifact` CLI | 実機 OTA 用 `.mender` 作成（[Downloads](https://docs.mender.io/downloads)） |

GPU 無し・CPU 版 PyTorch では学習・再生は動きません。

---

## 1. リポジトリを取る

```powershell
cd $env:USERPROFILE
git clone https://github.com/xeno1983/PAI_Task.git
cd PAI_Task
```

ブランチ例（DVC / MinIO / Mender 込み）:

```powershell
git checkout feat/dvc-minio-mlflow
git pull
```

以下、カレントはプロジェクトルート（`PAI_Task`）とします。

---

## 2. Python 仮想環境と依存関係

システム Python の torch は触らず、`.venv` に入れます。**Genesis より先に CUDA 付き PyTorch** を入れます。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

文字化け防止（毎回のシェル、または起動スクリプトが設定）:

```powershell
$env:PYTHONUTF8 = "1"
```

CUDA が見えているか:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

`True` と GPU 名が出れば OK です。

---

## 3. 動作確認

初回はシミュレーションカーネルのコンパイルで 1〜2 分かかります。

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\smoke_local.py
```

最後に `smoke ok` と出れば成功です。Genesis 無しでも通る単体テスト:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

---

## 4. デスクトップ起動スクリプト

リポジトリの `launcher\` をデスクトップへコピーします。

```powershell
$dest = Join-Path $env:USERPROFILE "Desktop\起動スクリプト\PickAndCarry"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -Force .\launcher\PickAndCarry_run.bat $dest
Copy-Item -Force .\launcher\PickAndCarry_run.ps1 $dest
```

`PickAndCarry_run.ps1` 内のパスが自分のクローン先を指しているか確認します。

```powershell
# 例: Set-Location C:\Users\<あなた>\PAI_Task
Select-String -Path $dest\PickAndCarry_run.ps1 -Pattern "Set-Location"
```

違う場合は、その 1 行を自分の `PAI_Task` の絶対パスに直します。`PickAndCarry_run.bat` も同様に `cd /d ...` を合わせてください。

あとは `PickAndCarry_run.bat` をダブルクリック。メニュー 1〜9 が使えます。

| 番号 | 内容 |
| --- | --- |
| 1 | 模倣学習 |
| 2 | カリキュラム学習 |
| 3 | 推論 |
| 4〜8 | 比較動画・曲線・プレゼン・zip・手法比較 |
| **9** | **DVC / MinIO / Mender** |

---

## 5. DVC + MinIO + MLflow（任意・推奨）

研究用の重み倉庫と実験ノートです。詳細は [DVC_MINIO.md](DVC_MINIO.md)。

### 5.1 初回

```powershell
copy .env.example .env
# 必要なら .env の minioadmin パスワードを変更

.\.venv\Scripts\python.exe -m pip install -r requirements-dvc.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_dvc_minio.ps1
```

`setup_dvc_minio.ps1` は Docker Desktop 起動 → MinIO 起動 → DVC remote 設定まで行います。

または起動スクリプト **メニュー 9 → 2**。

### 5.2 日常

| 操作 | メニュー / コマンド |
| --- | --- |
| MinIO 起動 | 9 → 1、または `scripts\start_minio.ps1` |
| 最良重みを倉庫へ | 9 → 3 |
| 倉庫から戻す | 9 → 5 |
| 最良で推論 | 9 → 6 |
| Console | http://127.0.0.1:9001 （user/pass は `.env`） |
| MLflow UI | 下記 |

```powershell
. .\scripts\load_pai_env.ps1
.\.venv\Scripts\mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

ブラウザ: http://127.0.0.1:5000

**注意（2026）:** Docker Hub から公式 `minio/minio` が消えていることがあります。compose は `pull_policy: if_not_present` です。イメージが無い PC では、別マシンのキャッシュを移すか quay.io 等から取得してください。

別 PC で同じ重みを使う:

```powershell
git pull
. .\scripts\load_pai_env.ps1
.\.venv\Scripts\python.exe scripts\dvc_sync.py pull
.\.venv\Scripts\python.exe scripts\dvc_sync.py restore --log-dir logs\ppo_curriculum_walk
```

（MinIO が同じネットワーク／同じデータを見えている必要があります。ローカルだけの MinIO なら、バケット内容の共有方法を別途決めてください。）

---

## 6. Mender OTA（任意・実機向け）

方策ファイルの端末配信です。詳細は [MENDER_OTA.md](MENDER_OTA.md)。

1. [mender-artifact](https://docs.mender.io/downloads) を PATH か `tools\` に置く  
2. DVC で `best.pt` を publish（メニュー 9 → 3）  
3. メニュー **9 → 9**、または:

```powershell
.\.venv\Scripts\python.exe scripts\mender_pack_model.py --name ppo_curriculum_walk --device-type go2-z1
```

4. できた `artifacts\mender\*.mender` を Hosted Mender 等へアップロード  
5. 端末には Client と `device/mender/modules/pai-model` を配置

---

## 7. よくあるつまずき

| 症状 | 確認 |
| --- | --- |
| `torch.cuda.is_available()` が False | ドライバ、`cu128` の torch、GPU 機種 |
| `smoke_local` が落ちる | 初回コンパイル中か、VRAM 不足（並列数を下げる） |
| 起動スクリプトが別フォルダを見に行く | `PickAndCarry_run.ps1` の `Set-Location` |
| MinIO が起動しない | Docker Desktop 起動、`scripts\start_minio.ps1` |
| `pull access denied for minio/minio` | ローカルにイメージが無い。上記「注意（2026）」 |
| DVC push で認証エラー | `. .\scripts\load_pai_env.ps1` 忘れ、`.env` のキー |
| 日本語が化ける | `$env:PYTHONUTF8="1"` |

---

## 8. 最短チェックリスト

```text
□ NVIDIA ドライバ / Python 3.12 / Git /（任意）Docker Desktop
□ git clone → cd PAI_Task
□ venv → torch(cu128) → requirements.txt
□ smoke_local.py → smoke ok
□ launcher を Desktop\起動スクリプト\PickAndCarry へコピーしパス修正
□（任意）setup_dvc_minio.ps1 → メニュー 9 で publish/push
□（任意）mlflow ui
□（任意）mender-artifact + メニュー 9 → 9
```

関連:

- 学習・メニューの詳細: [../README.md](../README.md)
- DVC / MinIO: [DVC_MINIO.md](DVC_MINIO.md)
- 実機: [SIM_TO_REAL.md](SIM_TO_REAL.md)
- OTA: [MENDER_OTA.md](MENDER_OTA.md)
- 使い方スライド再生成: `.\.venv\Scripts\python.exe scripts\make_dvc_mlflow_ppt.py`
