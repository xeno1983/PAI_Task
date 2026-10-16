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
Write-Host "   1  学習。段階を選んでから回数を入れる"
Write-Host "      立つ、座る、歩行、ジャンプ、不整地、持ち歩き、箱運び、キャッチの順。"
Write-Host "      直前の段階の重みから続ける。腕は Z1。観測 81、行動 19。"
Write-Host "      不整地は段差の高さだけ。箱運びだけ、走行環境・学習の分け方・手先カメラを選ぶ。"
Write-Host "      ROS 2 は学習では使わない。"
Write-Host "      Enter だけなら 1000 回、並列 16 体。同じ段階をもう一度選ぶと続きから学習する。"
Write-Host "      終わると MLflow、その段階の学習曲線、録画を開いて、この画面は閉じます。5 の提出動画は、残った段階を順に流す。"
Write-Host "      各学習は logs\mlflow にも残る。親の重みとチェックポイントが実行ごとに分かれる。"
Write-Host "      キャッチは静止、ゆっくり、ばらす、ノイズで別のファイルになる。"
Write-Host "   2  推論。段階を選んで、その保存した重みを再生する"
Write-Host "      箱運びだけ、クリックで物資と運び先を決める。ROS 2 を選ぶと rviz2 で運び先を変えられる。"
Write-Host "      立つ、座る、歩行、ジャンプ、不整地、持ち歩き、キャッチは録画だけ。ROS 2 は使わない。"
Write-Host "   3  箱運びの途中経過を並べた比較動画（追加学習なし）"
Write-Host "      使うのは logs\ppo_z1 だけ。走行環境、学習の分け方、手先カメラを選ぶ。"
Write-Host "      動画: 起動スクリプトフォルダの PickAndCarry_z1_compare.mp4"
Write-Host "   4  学習曲線を開く（追加学習なし）。段階を選ぶと、その保存先を開く"
Write-Host "      箱運びは報酬、把持率、手先の距離、設置率。歩行系は速度の一致。"
Write-Host "      ブラウザ: 起動スクリプトフォルダの PickAndCarry_curves.html"
Write-Host "   5  プレゼン用の動画を作る（追加学習なし）"
Write-Host "      学習条件のスライドを作り直し、フォルダ内の録画を1本につなぐ。説明の音声も入る。"
Write-Host "      動画: 起動スクリプトフォルダの PickAndCarry_presentation.mp4"
Write-Host "   6  Colab 用 zip を起動スクリプトフォルダに作る"
Write-Host ""

function Get-LatestIteration {
    param([string]$Folder = "logs\ppo_z1")
    $dir = Join-Path (Get-Location) $Folder
    $best = 0
    if (Test-Path $dir) {
        foreach ($file in Get-ChildItem -Path $dir -Filter "ppo_*.pt" -ErrorAction SilentlyContinue) {
            if ($file.BaseName -match '^ppo_(\d+)$') {
                $value = [int]$Matches[1]
                if ($value -gt $best) { $best = $value }
            }
        }
    }
    return $best
}

$z1Iter = Get-LatestIteration "logs\ppo_z1"
Write-Host " 腕は Unitree Z1（約 5.2 kg、関節 6）。観測 81、行動 19。"
Write-Host " 箱運びの iteration: $z1Iter    保存: logs\ppo_z1"
Write-Host ""
$choice = Read-Host "番号を入力 [1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

function Select-Curriculum {
    param([switch]$ForPlay)
    Write-Host ""
    Write-Host " カリキュラム"
    Write-Host "   1  立つ。しゃがんだ状態から立ち上がる"
    Write-Host "   2  座る。体を下げて静止する"
    Write-Host "   3  歩行。平地で前後進と旋回"
    Write-Host "   4  ジャンプ。歩行の続き。足を離して着地する"
    Write-Host "   5  不整地の歩行。ジャンプの続き"
    Write-Host "   6  歩きながら持つ。不整地の重みから続ける"
    Write-Host "   7  ピック＆プレース。置いてある箱を掴んで運ぶ"
    Write-Host "   8  転がるボールを掴む"
    $picked = Read-Host "段階 [1]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "1" }
    $script:stage = "carry"
    $script:curriculumArgs = @()
    if ($picked -eq "1") {
        $script:stage = "stand"
        $script:curriculumArgs = @("--curriculum", "stand")
        $script:logDir = "logs/ppo_curriculum_stand"
        $script:videoName = "PickAndCarry_stand.mp4"
        Write-Host "立つ段階です。しゃがんだ姿勢から始まり、何もしなければしゃがんだままです。腕は初期姿勢のままです。"
        Write-Host "脚を伸ばして高さ約 0.40 m で 1 秒静止すると成功です。保存先は logs\ppo_curriculum_stand です。"
    } elseif ($picked -eq "2") {
        $script:stage = "sit"
        $script:curriculumArgs = @("--curriculum", "sit")
        $script:logDir = "logs/ppo_curriculum_sit"
        $script:videoName = "PickAndCarry_sit.mp4"
        Write-Host "座る段階です。立った姿勢から始まり、立つ段階で覚えた脚の伸ばし方から体を下げます。腕は初期姿勢のままです。"
        Write-Host "高さ約 0.28 m で 1 秒静止すると成功です。保存先は logs\ppo_curriculum_sit です。"
    } elseif ($picked -eq "3") {
        $script:stage = "walk"
        $script:curriculumArgs = @("--curriculum", "walk")
        $script:logDir = "logs/ppo_curriculum_walk"
        $script:videoName = "PickAndCarry_walk.mp4"
        Write-Host "平地の歩行です。座るの重みから続けます。腕は初期姿勢のままです。"
        Write-Host "保存先は logs\ppo_curriculum_walk です。"
    } elseif ($picked -eq "4") {
        $script:stage = "jump"
        $script:curriculumArgs = @("--curriculum", "jump")
        $script:logDir = "logs/ppo_curriculum_jump"
        $script:videoName = "PickAndCarry_jump.mp4"
        Write-Host "ジャンプです。平地の歩行の重みから続けます。足が離れてから着地すると成功です。腕は初期姿勢のままです。"
        Write-Host "保存先は logs\ppo_curriculum_jump です。追加で選ぶ項目はありません。"
    } elseif ($picked -eq "5") {
        $script:stage = "rough"
        $script:curriculumArgs = @("--curriculum", "rough")
        $script:logDir = "logs/ppo_curriculum_rough"
        $script:videoName = "PickAndCarry_rough.mp4"
        Write-Host "不整地の歩行です。ジャンプの重みから続けます。腕は初期姿勢のままです。"
        Write-Host "保存先は logs\ppo_curriculum_rough です。このあと選ぶのは段差の高さだけです。"
        Write-Host "成績が低いあいだは平地に戻し、回復したら段差を上げます。"
    } elseif ($picked -eq "6") {
        $script:stage = "haul"
        $script:curriculumArgs = @("--curriculum", "haul")
        $script:logDir = "logs/ppo_curriculum_haul"
        $script:videoName = "PickAndCarry_haul.mp4"
        Write-Host "掴んだまま歩く段階です。不整地の重みから、脚も腕も同じネットワークで続けます。"
        Write-Host "保存先は logs\ppo_curriculum_haul です。"
    } elseif ($picked -eq "7") {
        $script:stage = "carry"
        $script:curriculumArgs = @()
        $script:logDir = "logs/ppo_z1"
        $script:videoName = "PickAndCarry_z1.mp4"
        Write-Host "ピック＆プレースです。歩きながら持つ段階と同じ観測・行動なので、その重みから続けます。"
        Write-Host "このあと、走行環境・学習の分け方・手先カメラを選びます。"
        if ($ForPlay) {
            Write-Host "推論では、クリックで物資と運び先を決められます。ROS 2 を選ぶと rviz2 で運び先を変えられます。"
        }
    } elseif ($picked -eq "8") {
        $script:stage = "catch"
        Write-Host "   1  手の前で静止"
        Write-Host "   2  ゆっくりまっすぐ転がす"
        Write-Host "   3  速さと方向をばらす"
        Write-Host "   4  位置に少しノイズを足す"
        $levelPick = Read-Host "ボール [1]"
        if ([string]::IsNullOrWhiteSpace($levelPick)) { $levelPick = "1" }
        $level = @("static", "slow", "random", "noisy")[[Math]::Min(3, [Math]::Max(0, [int]$levelPick - 1))]
        $script:curriculumArgs = @("--curriculum", "catch", "--catch-level", $level)
        $script:logDir = "logs/ppo_curriculum_catch_$level"
        $script:videoName = "PickAndCarry_catch_$level.mp4"
        Write-Host "キャッチ（$level）です。箱運びと同じ観測・行動なので、その重みから続けます。"
        Write-Host "保存先は $script:logDir です。"
        if ($level -ne "static") {
            Write-Host "成績が低いあいだは易しいボールに戻し、回復したら上げます。"
        }
    } else {
        $script:stage = "stand"
        $script:curriculumArgs = @("--curriculum", "stand")
        $script:logDir = "logs/ppo_curriculum_stand"
        $script:videoName = "PickAndCarry_stand.mp4"
        Write-Host "不明な番号なので、立つから始めます。"
    }
    $script:activeIter = Get-LatestIteration ($script:logDir -replace "/", "\")
    if (-not $ForPlay -and ($script:stage -in @("stand", "walk", "jump", "haul", "catch"))) {
        Ask-Retreat
    }
}

function Ask-Retreat {
    Write-Host "   1  この段階の保存から続ける"
    Write-Host "   2  上の段階の重みで、この易しい課題を学び直す"
    $answer = Read-Host "易しい課題に戻す [1]"
    if ($answer -eq "2") {
        $script:curriculumArgs += @("--retreat")
        Write-Host "上の段階の重みから、この課題を学び直します。"
    }
}

function Select-Terrain {
    param([switch]$Walking)
    Write-Host ""
    Write-Host " 走行環境"
    Write-Host "   1  低い波（高さ 3.5cm）"
    if ($Walking) {
        Write-Host "   2  段差を足す（高さ 4.5cm）"
    } else {
        Write-Host "   2  段差を足す（高さ 4.5cm）。箱の前に1段、置くまでにもう2段"
    }
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
    Write-Host " 学習の分け方（箱運びだけ）"
    Write-Host "   1  脚だけ動かす。腕は初期姿勢のまま"
    Write-Host "   2  分ける。歩くときは脚、拾う・置くときは腕"
    Write-Host "   3  脚と腕を同時に出す"
    $picked = Read-Host "学習の分け方 [2]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "2" }
    switch ($picked) {
        "1" {
            Write-Host "箱運びのあいだは脚だけ動かします。腕は初期姿勢のままです。"
            $script:controlArgs = @("--control-split", "walk")
        }
        "3" {
            Write-Host "脚と腕を同時に出します。いまの重みから続きます。"
            $script:controlArgs = @("--control-split", "both")
        }
        default {
            Write-Host "歩くときと拾うときを分けて学習します。"
            $script:controlArgs = @()
        }
    }
}

function Select-Vision {
    Write-Host ""
    Write-Host " 対象物の見つけ方"
    Write-Host "   1  シミュレータの位置をそのまま使う"
    Write-Host "   2  手先のカメラで箱を見つける。見えているときだけつかむ"
    $picked = Read-Host "見つけ方 [1]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "1" }
    if ($picked -eq "2") {
        Write-Host "手先カメラで検知して把持します。"
        $script:visionArgs = @("--wrist-camera")
    } else {
        Write-Host "シミュレータの位置で把持します。"
        $script:visionArgs = @()
    }
}

function Select-Ros {
    Write-Host ""
    Write-Host " rviz2"
    Write-Host "   1  使わない。クリック画面の運び先のまま録画する"
    Write-Host "   2  ROS 2 で表示する。rviz2 を開き、推論中のクリックで運び先を動かす"
    $picked = Read-Host "rviz2 [1]"
    if ($picked -eq "2") {
        Start-Rviz
    } else {
        Write-Host "録画だけ行います。"
        $script:rosArgs = @()
        $script:inferSeconds = 20
    }
}

function Select-StageOptions {
    param([switch]$ForPlay)
    $script:terrainArgs = @()
    $script:controlArgs = @()
    $script:visionArgs = @()
    $script:rosArgs = @()
    $script:inferSeconds = 20
    if ($script:stage -eq "carry") {
        Select-Terrain
        Select-Control
        Select-Vision
        if ($ForPlay) { Select-Ros }
    } elseif ($script:stage -eq "rough") {
        Select-Terrain -Walking
    }
}

function Start-Rviz {
    $armName = "z1"
    $root = (Get-Location).Path
    $wslPath = "/mnt/" + $root.Substring(0, 1).ToLower() + ($root.Substring(2) -replace "\\", "/")
    $script:rosArgs = @("--ros")
    $script:inferSeconds = 60
    $command = "wsl.exe -d Ubuntu -- bash -lc `"cd $wslPath && bash ros/display.sh $armName`""
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k $command"
    Write-Host "rviz2 を開いています。表示まで十数秒かかることがあります。"
    Write-Host "推論が始まると、シミュレーション画面が Windows に開きます。姿勢は logs\ros_ipc 経由で rviz2 に渡ります。"
    Write-Host "rviz2 の Publish Point（十字）で、コースの薄い地面をクリックしてください。黄色い球がクリック、緑の柱がシミュレーションの運び先です。"
}

function Open-Mlflow {
    $db = "logs\mlflow\mlflow.db"
    if (-not (Test-Path $db)) {
        Write-Host "MLflow の記録が無いので、画面は開きません。"
        return
    }
    $mlflow = ".\.venv\Scripts\mlflow.exe"
    if (-not (Get-Process -Name mlflow -ErrorAction SilentlyContinue)) {
        # One worker. MLflow's default of 4 makes uvicorn hand the socket to a child, and Windows rejects that listen.
        $env:MLFLOW_DISABLE_AGENT_HINT = "1"
        Start-Process -FilePath $mlflow -ArgumentList @("ui", "--backend-store-uri", "sqlite:///logs/mlflow/mlflow.db", "--workers", "1", "--host", "127.0.0.1") -WorkingDirectory (Get-Location).Path
        Start-Sleep -Seconds 3
    }
    Write-Host "MLflow を開きます。サーバの窓は閉じないでください。"
    Start-Process "http://127.0.0.1:5000"
}

$script:armArgs = @("--arm", "z1")
$script:logDir = "logs/ppo_z1"
$script:videoName = "PickAndCarry_z1.mp4"
$script:stage = "carry"
$script:activeIter = $z1Iter

$code = 0
switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "段階を選んで学習します。終わると、MLflow、その保存先の学習曲線、動画を開きます。"
        Select-Curriculum
        $trainIters = 0
        while ($trainIters -lt 1) {
            $rawIters = Read-Host "学習回数 [1000]"
            if ([string]::IsNullOrWhiteSpace($rawIters)) { $rawIters = "1000" }
            $parsedIters = 0
            if (-not [int]::TryParse($rawIters, [ref]$parsedIters) -or $parsedIters -lt 1) {
                Write-Host "1 以上の整数を入力してください。"
                continue
            }
            $trainIters = $parsedIters
        }
        Write-Host "開始時の iteration: $script:activeIter    終了目標: $($script:activeIter + $trainIters)（$trainIters 回）"
        $numEnvs = 0
        while ($numEnvs -lt 1) {
            $rawEnvs = Read-Host "同時に動かすロボット数 [16]"
            if ([string]::IsNullOrWhiteSpace($rawEnvs)) { $rawEnvs = "16" }
            $parsedEnvs = 0
            if (-not [int]::TryParse($rawEnvs, [ref]$parsedEnvs) -or $parsedEnvs -lt 1) {
                Write-Host "1 以上の整数を入力してください。"
                continue
            }
            $numEnvs = $parsedEnvs
        }
        if ($numEnvs -gt 32) {
            Write-Host "この GPU は 8GB です。台数は制限せず、$numEnvs 台をそのまま使います。"
            Write-Host "ロボットの状態は GPU 上に複製されます。タスクマネージャーのメモリ欄はほとんど変わりません。"
            Write-Host "足りなくなると学習が止まります。開始ログの GPU 使用量で確認できます。"
        }
        Select-StageOptions
        & $python -u -m pick_and_carry.train_ppo --resume latest --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
        $code = $LASTEXITCODE
        Open-Mlflow
        if ($code -eq 0) {
            $curves = Join-Path $out "PickAndCarry_curves.html"
            & $python -u scripts\plot_curves.py --log-dir $logDir --output $curves
            if ($LASTEXITCODE -eq 0 -and (Test-Path $curves)) { Start-Process $curves }
            $video = Join-Path $out $videoName
            & $python -u -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --record $video --seconds 20 --log-dir $logDir @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
            $code = $LASTEXITCODE
            if ($code -eq 0 -and (Test-Path $video)) {
                Write-Host "提出動画用に残しました: $videoName"
                Start-Process $video
            }
        }
    }
    "2" {
        Write-Host ""
        Select-Curriculum -ForPlay
        Write-Host "推論します。使う重み: iteration $script:activeIter （$logDir）"
        $video = Join-Path $out $videoName
        Select-StageOptions -ForPlay
        if ($script:stage -eq "carry") {
            Write-Host "1回目のクリックが物資、2回目が運び先です。"
            & $python -u scripts\infer_gui.py --log-dir $logDir --record $video --seconds $script:inferSeconds @armArgs @terrainArgs @controlArgs @visionArgs @rosArgs
        } else {
            & $python -u -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --record $video --seconds 20 --log-dir $logDir @armArgs @terrainArgs @curriculumArgs
        }
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) {
            Write-Host "提出動画用に残しました: $videoName"
            Start-Process $video
        }
    }
    "3" {
        Write-Host ""
        Write-Host "比較動画は箱運びだけです。logs\ppo_z1 の時点を並べます。"
        Write-Host "保存済みの時点を同じ配置で再生し、並べた動画を作ります。最新は iteration $script:activeIter です。"
        Select-Terrain
        Select-Control
        Select-Vision
        $video = Join-Path $out "PickAndCarry_z1_compare.mp4"
        & $python -u scripts\compare_snapshots.py --log-dir $logDir --output $video --seconds 16 --panels 4 @armArgs @terrainArgs @controlArgs @visionArgs
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "4" {
        Write-Host ""
        Select-Curriculum -ForPlay
        Write-Host "保存済みの学習記録から曲線を作り、ブラウザで開きます。保存先: $logDir"
        $curves = Join-Path $out "PickAndCarry_curves.html"
        & $python -u scripts\plot_curves.py --log-dir $logDir --output $curves
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $curves)) { Start-Process $curves }
    }
    "5" {
        Write-Host ""
        Write-Host "学習条件のスライドを作り直し、このフォルダの録画をつないでプレゼン動画にします。"
        & $python -u scripts\make_summary_slide.py
        if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; break }
        $video = Join-Path $out "PickAndCarry_presentation.mp4"
        & $python -u scripts\make_presentation.py --pptx (Join-Path $out "PickAndCarry_学習条件.pptx") --videos $out --output $video
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "6" {
        $zip = Join-Path $out "PAI_Task_colab.zip"
        if (Test-Path $zip) { Remove-Item -Force $zip }
        tar -a -c -f $zip --exclude=.venv --exclude=logs --exclude=__pycache__ --exclude=.git -C C:\Users\xeno1 PAI_Task
        $code = $LASTEXITCODE
        if ($code -eq 0) { Write-Host "作成しました: $zip" }
    }
    default {
        Write-Host "不明な番号です: $choice"
        $code = 1
    }
}

Write-Host ""
Write-Host "exit code: $code"
if ($choice -eq "1" -and $code -eq 0) {
    exit $code
}
Read-Host "この画面は確認が終わるまで開いたままです。キーを押すと閉じます"
exit $code
