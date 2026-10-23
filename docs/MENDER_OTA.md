# Mender による方策（モデル）OTA

学習済み重みを実機へ配るときの役割分担です。OS 全体の A/B 更新と、**方策ファイルだけ**のアプリ更新を分けます。

| 層 | 役割 |
| --- | --- |
| Git | コード・DVC ポインタ・Update Module スクリプト |
| MLflow | 実験ノート（曲線・パラメータ） |
| DVC + MinIO | 研究室／PC 間の重み倉庫（`best.pt`） |
| **Mender** | **実機への OTA 配信・ロールバック** |

```
publish --select best → artifacts/checkpoints/<stage>/best.pt
                              │
                              ▼ mender_pack_model.py
                     artifacts/mender/*.mender
                              │
                              ▼ Hosted Mender / 自前サーバ
                     端末の Mender Client
                              │
                              ▼ Update Module「pai-model」
                     /data/pai/models/<stage>/best.pt
                     /data/pai/models/active → 上記
```

MinIO は「研究用の倉庫」、Mender は「現場端末への配達と版の確定」です。同じ `best.pt` を両方に載せる想定です。

## 1. 端末側（一度だけ）

1. 端末に [Mender Client](https://docs.mender.io/get-started) を入れ、Hosted Mender または自前サーバへ接続する。
2. `device_type` を揃える（既定例: `go2-z1`）。
3. Update Module を配置する:

```bash
sudo cp device/mender/modules/pai-model /usr/share/mender/modules/v3/pai-model
sudo chmod +x /usr/share/mender/modules/v3/pai-model
sudo mkdir -p /data/pai/models
```

インストール先は環境変数 `PAI_MODEL_DIR` で変更できます（既定 `/data/pai/models`）。

実機ランタイムはシンボリックリンク `active/best.pt` を読めばよいです:

```text
/data/pai/models/active/best.pt
/data/pai/models/active/manifest.json
```

## 2. PC 側：最良重み → `.mender`

```powershell
.\scripts\load_pai_env.ps1
# DVC で最良を確定（未なら）
.\.venv\Scripts\python.exe scripts\dvc_sync.py publish --log-dir logs\ppo_curriculum_walk --select best --push

# Mender Artifact 化（要 mender-artifact CLI）
.\.venv\Scripts\python.exe scripts\mender_pack_model.py --name ppo_curriculum_walk --device-type go2-z1
```

出力:

- `artifacts/mender/pai-....mender`
- 同名の `.mender.json`（stage / iteration / score のメモ）

`mender-artifact` の入手: [Mender Downloads](https://docs.mender.io/downloads)。Windows では WSL か `tools/mender-artifact.exe` に置く。

起動スクリプトの **メニュー 9 → 9** からも同じパックを呼べます。

## 3. サーバへ上げて配信

1. [Hosted Mender](https://mender.io/)（または自前 Mender Server）にログイン。
2. Releases に `.mender` をアップロード。
3. 対象デバイス（`device_type=go2-z1`）へデプロイ。
4. 端末で `ArtifactInstall` → `Commit` が通れば `/data/pai/models/active/best.pt` が更新される。失敗時は Module がバックアップへ Rollback。

## 4. やること / やらないこと

やること:

- 区切りの **最良**（または意図した版）だけを Artifact にする
- `manifest.json` を同梱し、どの iteration / score か端末でも追えるようにする
- OS イメージ更新は別 Artifact（rootfs）にし、方策は `pai-model` に限定する

やらないこと:

- 学習ホットパスで毎回 Mender 配信しない
- MinIO を端末に直接晒して「最新ファイルを wget」だけにしない（版・ロールバック・承認がない）
- `logs/` 全履歴を端末へ送らない

## 5. 実機推論とのつなぎ

`docs/SIM_TO_REAL.md` のランタイムは、ロードパスを例えば次にする:

```python
ckpt = torch.load("/data/pai/models/active/best.pt", map_location="cpu")
```

シミュ側の検証はこれまでどおり DVC の `9 → 6`（play）で行い、通った版だけ `mender_pack_model` → 配信します。
