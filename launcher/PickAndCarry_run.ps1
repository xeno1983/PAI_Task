# The .bat is ASCII so cmd can start it from a Japanese folder. Switch the
# window to UTF-8 only after PowerShell has already opened this file.
chcp.com 65001 > $null
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location C:\Users\xeno1\PAI_Task
$Host.UI.RawUI.WindowTitle = "Pick-and-Carry"
$out = $PSScriptRoot

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host ".venv がありません。README の手順で環境を作ってから再実行してください。"
    Read-Host "キーを押すと閉じます"
    exit 1
}

Write-Host ""
Write-Host " Pick-and-Carry"
Write-Host " プロジェクト: $(Get-Location)"
Write-Host ""
Write-Host " この画面はログです。閉じずに待ってください。"
Write-Host ""
Write-Host "   1  ロボットを 8 秒間立たせて録画"
Write-Host "      正解: 犬が平らな足場に立ち、前方に段差、その先にオレンジの立方体。"
Write-Host "            最後に smoke ok と exit code: 0"
Write-Host "            動画: 起動スクリプトフォルダの PickAndCarry_smoke.mp4"
Write-Host "   2  歩行から学習する。不整地→拾う→不整地→置く を 1000 iteration"
Write-Host "      歩き方は方策が学ぶ。拾うときと置くときは脚を立ち姿勢に固定する。"
Write-Host "      始める前に、走行環境と、歩行だけの学習か同時の把持かを選ぶ。"
Write-Host "      段の位置はエピソードごとに少し変わる。続きは logs\ppo_walk から読む。"
Write-Host "      動画: 起動スクリプトフォルダの PickAndCarry_ppo.mp4"
Write-Host "      目安: 40 分ほど。もう一度選ぶと、その続きから学習する。"
Write-Host "   3  短い SAC（8 環境、500 ステップ、録画あり）"
Write-Host "   4  Colab 用 zip を起動スクリプトフォルダに作る"
Write-Host "   5  今の学習結果だけを動画にする（追加学習なし）。走行環境を選ぶ"
Write-Host "   6  学習の途中経過を並べた比較動画を作る（追加学習なし）。走行環境を選ぶ"
Write-Host "      同じ初期配置で、早い時点と遅い時点を同時に再生する。"
Write-Host "      動画: 起動スクリプトフォルダの PickAndCarry_compare.mp4"
Write-Host "   7  学習曲線を開く（追加学習なし）"
Write-Host "      報酬、把持率、手先の距離、設置率。"
Write-Host "      ブラウザ: 起動スクリプトフォルダの PickAndCarry_curves.html"
Write-Host ""
$choice = Read-Host "番号を入力 [1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

function Select-Terrain {
    Write-Host ""
    Write-Host " 走行環境"
    Write-Host "   1  低い波（高さ 3.5cm）"
    Write-Host "   2  段差を足す（高さ 4.5cm）。箱の前に1段、置くまでにもう2段"
    Write-Host "   3  段差を高くする（高さ 6cm）"
    $picked = Read-Host "走行環境 [2]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "2" }
    switch ($picked) {
        "1" {
            Write-Host "低い波で進めます。"
            $script:terrainArgs = @()
        }
        "3" {
            Write-Host "高さ 6cm の段差で進めます。"
            $script:terrainArgs = @("--terrain-steps", "--terrain-amplitude", "0.06")
        }
        default {
            Write-Host "高さ 4.5cm の段差で進めます。"
            $script:terrainArgs = @("--terrain-steps", "--terrain-amplitude", "0.045")
        }
    }
}

function Select-Control {
    Write-Host ""
    Write-Host " 学習の分け方"
    Write-Host "   1  歩行を先に覚える。腕は固定。報酬は前へ進むことと転ばないこと"
    Write-Host "   2  分ける。歩くときは脚、拾う・置くときは腕"
    Write-Host "   3  歩行と把持を同時。エピソード中ずっと脚も腕も方策が出す"
    $picked = Read-Host "学習の分け方 [2]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "2" }
    switch ($picked) {
        "1" {
            Write-Host "歩行だけを学習します。いまの重みの脚から続きます。"
            $script:controlArgs = @("--control-split", "walk")
        }
        "3" {
            Write-Host "歩行と把持を同時に学習します。いまの重みから続きます。"
            $script:controlArgs = @("--control-split", "both")
        }
        default {
            Write-Host "歩くときと拾うときを分けて学習します。"
            $script:controlArgs = @()
        }
    }
}

$code = 0
switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "シーンを作っています。終わると動画を開きます。"
        $video = Join-Path $out "PickAndCarry_smoke.mp4"
        & $python -u scripts\smoke_local.py --record $video --seconds 8
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "2" {
        Write-Host ""
        Write-Host "不整地を渡って拾い、もう一度不整地を渡って置く学習を 1000 iteration 進めます。終わると学習曲線と動画を開きます。"
        Select-Terrain
        Select-Control
        & $python -u -m pick_and_carry.train_ppo --resume latest --num-envs 16 --iterations 1000 --num-steps 128 --save-interval 20 --log-dir logs/ppo_walk @terrainArgs @controlArgs
        $code = $LASTEXITCODE
        if ($code -eq 0) {
            $curves = Join-Path $out "PickAndCarry_curves.html"
            & $python -u scripts\plot_curves.py --log-dir logs/ppo_walk --output $curves
            if ($LASTEXITCODE -eq 0 -and (Test-Path $curves)) { Start-Process $curves }
            $video = Join-Path $out "PickAndCarry_ppo.mp4"
            & $python -u -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --record $video --seconds 20 --log-dir logs/ppo_walk @terrainArgs @controlArgs
            $code = $LASTEXITCODE
            if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
        }
    }
    "3" {
        Write-Host ""
        Write-Host "SAC を開始します。動画に映るのは 1 体だけです。"
        $video = Join-Path $out "PickAndCarry_sac.mp4"
        & $python -u -m pick_and_carry.train_sac --record $video --num-envs 8 --steps 500 --warmup 50 --log-interval 25 --save-interval 500
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "4" {
        $zip = Join-Path $out "PAI_Task_colab.zip"
        if (Test-Path $zip) { Remove-Item -Force $zip }
        tar -a -c -f $zip --exclude=.venv --exclude=logs --exclude=__pycache__ --exclude=.git -C C:\Users\xeno1 PAI_Task
        $code = $LASTEXITCODE
        if ($code -eq 0) { Write-Host "作成しました: $zip" }
    }
    "5" {
        Write-Host ""
        Write-Host "今の学習結果を再生して動画にします。"
        Select-Terrain
        Select-Control
        $video = Join-Path $out "PickAndCarry_ppo.mp4"
        & $python -u -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --record $video --seconds 20 --log-dir logs/ppo_walk @terrainArgs @controlArgs
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "6" {
        Write-Host ""
        Write-Host "保存済みの時点を同じ配置で再生し、並べた動画を作ります。"
        Select-Terrain
        Select-Control
        $video = Join-Path $out "PickAndCarry_compare.mp4"
        & $python -u scripts\compare_snapshots.py --log-dir logs/ppo_walk --output $video --seconds 16 --panels 4 @terrainArgs @controlArgs
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "7" {
        Write-Host ""
        Write-Host "保存済みの学習記録から曲線を作り、ブラウザで開きます。"
        $curves = Join-Path $out "PickAndCarry_curves.html"
        & $python -u scripts\plot_curves.py --log-dir logs/ppo_walk --output $curves
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $curves)) { Start-Process $curves }
    }
    default {
        Write-Host "不明な番号です: $choice"
        $code = 1
    }
}

Write-Host ""
Write-Host "exit code: $code"
Read-Host "この画面は確認が終わるまで開いたままです。キーを押すと閉じます"
exit $code
