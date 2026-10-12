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
Write-Host "   2  歩行から学習する。不整地→拾う→不整地→置く。回数はあとで入力"
Write-Host "      歩き方は方策が学ぶ。拾うときと置くときは脚を立ち姿勢に固定する。"
Write-Host "      始める前に、腕、走行環境、学習の分け方、手先カメラを選ぶ。"
Write-Host "      Panda の続きは logs\ppo_walk、Z1 の続きは logs\ppo_z1 から読む。"
Write-Host "      動画: 起動スクリプトフォルダの PickAndCarry_ppo.mp4"
Write-Host "      Enter だけなら 1000 回、並列 16 体。もう一度選ぶと続きから学習する。"
Write-Host "   3  短い SAC（8 環境、500 ステップ、録画あり、Panda のみ）"
Write-Host "   4  Colab 用 zip を起動スクリプトフォルダに作る"
Write-Host "   5  推論。クリックで物資と運び先を決める。ROS 2 を選ぶと実行中に rviz2 で運び先を変えられる"
Write-Host "   6  学習の途中経過を並べた比較動画を作る（追加学習なし）。走行環境を選ぶ"
Write-Host "      同じ初期配置で、早い時点と遅い時点を同時に再生する。"
Write-Host "      動画: 起動スクリプトフォルダの PickAndCarry_compare.mp4"
Write-Host "   7  学習曲線を開く（追加学習なし）"
Write-Host "      報酬、把持率、手先の距離、設置率。"
Write-Host "      ブラウザ: 起動スクリプトフォルダの PickAndCarry_curves.html"
Write-Host "   8  プレゼン用の動画を作る（追加学習なし）"
Write-Host "      学習条件のスライドを作り直し、フォルダ内の録画を1本につなぐ。説明の音声も入る。"
Write-Host "      動画: 起動スクリプトフォルダの PickAndCarry_presentation.mp4"
Write-Host ""

function Get-LatestIteration {
    param([string]$Folder = "logs\ppo_walk")
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

$latestIter = Get-LatestIteration "logs\ppo_walk"
$z1Iter = Get-LatestIteration "logs\ppo_z1"
Write-Host " Panda の iteration: $latestIter    保存: logs\ppo_walk"
Write-Host " Z1 の iteration: $z1Iter    保存: logs\ppo_z1"
Write-Host " Z1 は約 5.2 kg、関節 6 つ。Panda の重みは使いません。"
Write-Host ""
$choice = Read-Host "番号を入力 [1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

function Select-Arm {
    Write-Host ""
    Write-Host " マニピュレータ"
    Write-Host "   1  Franka Panda（約 18 kg の実機。学習上の質量は約 3.4 kg）。今の重みから続く"
    Write-Host "   2  Unitree Z1（約 5.2 kg、6 軸）。最初から学習し、logs\ppo_z1 に保存する"
    $picked = Read-Host "腕 [1]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "1" }
    if ($picked -eq "2") {
        Write-Host "Z1 で進めます。Panda の重みは読みません。"
        $script:armArgs = @("--arm", "z1")
        $script:logDir = "logs/ppo_z1"
        $script:activeIter = Get-LatestIteration "logs\ppo_z1"
        $script:videoName = "PickAndCarry_z1.mp4"
    } else {
        Write-Host "Franka Panda で進めます。"
        $script:armArgs = @()
        $script:logDir = "logs/ppo_walk"
        $script:activeIter = Get-LatestIteration "logs\ppo_walk"
        $script:videoName = "PickAndCarry_ppo.mp4"
    }
}

function Select-Curriculum {
    Write-Host ""
    Write-Host " カリキュラム"
    Write-Host "   1  いまのピック＆プレース。置いてある箱を掴んで運ぶ"
    Write-Host "   2  歩行。平地で前後進と旋回"
    Write-Host "   3  不整地の歩行。2 の続きとして同じ観測"
    Write-Host "   4  歩きながら持つ。脚は決まった歩き、腕と進む速さを学ぶ"
    Write-Host "   5  転がるボールを掴む"
    $picked = Read-Host "段階 [1]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "1" }
    $script:stage = "carry"
    $script:curriculumArgs = @()
    if ($picked -eq "2") {
        $script:stage = "walk"
        $script:curriculumArgs = @("--curriculum", "walk")
        $script:logDir = "logs/ppo_curriculum_walk"
        $script:videoName = "PickAndCarry_walk.mp4"
        Write-Host "平地の歩行です。保存先は logs\ppo_curriculum_walk です。"
    } elseif ($picked -eq "3") {
        $script:stage = "rough"
        $script:curriculumArgs = @("--curriculum", "rough")
        $script:logDir = "logs/ppo_curriculum_rough"
        $script:videoName = "PickAndCarry_rough.mp4"
        Write-Host "不整地の歩行です。平地の重みがあればそこから続けます。保存先は logs\ppo_curriculum_rough です。"
    } elseif ($picked -eq "4") {
        $script:stage = "haul"
        $script:curriculumArgs = @("--curriculum", "haul")
        $script:logDir = "logs/ppo_curriculum_haul"
        $script:videoName = "PickAndCarry_haul.mp4"
        Write-Host "掴んだまま歩く段階です。脚は歩行の重みを使い、腕と進む速さを学習します。保存先は logs\ppo_curriculum_haul です。"
    } elseif ($picked -eq "5") {
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
        $script:videoName = "PickAndCarry_catch.mp4"
        Write-Host "キャッチ（$level）です。この段階の重みがなければ、一つ前から続けます。保存先は $script:logDir です。"
    } else {
        Write-Host "ピック＆プレースのまま進めます。"
    }
    $script:activeIter = Get-LatestIteration ($script:logDir -replace "/", "\")
}

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

function Start-Rviz {
    $armName = "panda"
    if ($script:logDir -eq "logs/ppo_z1") { $armName = "z1" }
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

$code = 0
switch ($choice) {
    "1" {
        Write-Host ""
        Select-Arm
        Write-Host "シーンを作っています。終わると動画を開きます。"
        $video = Join-Path $out $(if ($script:logDir -eq "logs/ppo_z1") { "PickAndCarry_z1_smoke.mp4" } else { "PickAndCarry_smoke.mp4" })
        & $python -u scripts\smoke_local.py --record $video --seconds 8 @armArgs
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "2" {
        Write-Host ""
        Write-Host "不整地を渡って拾い、もう一度不整地を渡って置く学習を進めます。終わると学習曲線と動画を開きます。"
        Select-Arm
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
            Write-Host "この GPU は 8GB です。32 を超えるとメモリ不足で止まることがあります。"
        }
        if ($script:stage -eq "carry") {
            Select-Terrain
            Select-Control
            Select-Vision
        } elseif ($script:stage -eq "rough") {
            Select-Terrain
            $script:controlArgs = @()
            $script:visionArgs = @()
        } else {
            $script:terrainArgs = @()
            $script:controlArgs = @()
            $script:visionArgs = @()
        }
        & $python -u -m pick_and_carry.train_ppo --resume latest --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
        $code = $LASTEXITCODE
        if ($code -eq 0) {
            $curves = Join-Path $out "PickAndCarry_curves.html"
            & $python -u scripts\plot_curves.py --log-dir $logDir --output $curves
            if ($LASTEXITCODE -eq 0 -and (Test-Path $curves)) { Start-Process $curves }
            $video = Join-Path $out $videoName
            & $python -u -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --record $video --seconds 20 --log-dir $logDir @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
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
        Select-Arm
        Write-Host "クリック画面を開きます。使う重み: iteration $script:activeIter （$logDir）"
        Write-Host "1回目のクリックが物資、2回目が運び先です。"
        Select-Terrain
        Select-Control
        Select-Vision
        Select-Ros
        $video = Join-Path $out $videoName
        & $python -u scripts\infer_gui.py --log-dir $logDir --record $video --seconds $script:inferSeconds @armArgs @terrainArgs @controlArgs @visionArgs @rosArgs
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "6" {
        Write-Host ""
        Select-Arm
        Write-Host "保存済みの時点を同じ配置で再生し、並べた動画を作ります。最新は iteration $script:activeIter です。"
        Select-Terrain
        Select-Control
        Select-Vision
        $video = Join-Path $out $(if ($script:logDir -eq "logs/ppo_z1") { "PickAndCarry_z1_compare.mp4" } else { "PickAndCarry_compare.mp4" })
        & $python -u scripts\compare_snapshots.py --log-dir $logDir --output $video --seconds 16 --panels 4 @armArgs @terrainArgs @controlArgs @visionArgs
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "7" {
        Write-Host ""
        Select-Arm
        Write-Host "保存済みの学習記録から曲線を作り、ブラウザで開きます。"
        $curves = Join-Path $out "PickAndCarry_curves.html"
        & $python -u scripts\plot_curves.py --log-dir $logDir --output $curves
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $curves)) { Start-Process $curves }
    }
    "8" {
        Write-Host ""
        Write-Host "学習条件のスライドを作り直し、このフォルダの録画をつないでプレゼン動画にします。"
        & $python -u scripts\make_summary_slide.py
        if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; break }
        $video = Join-Path $out "PickAndCarry_presentation.mp4"
        & $python -u scripts\make_presentation.py --pptx (Join-Path $out "PickAndCarry_学習条件.pptx") --videos $out --output $video
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
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
