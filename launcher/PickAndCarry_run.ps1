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
Write-Host "   1  模倣学習（トロット教師 → BC / ValueDICE）。デモ保存の有無を選べます"
Write-Host "      保存先: logs\<手法>_curriculum_walk_easy\ （bc_trot.pt / valuedice_trot.pt / デモ mp4）"
Write-Host "   2  学習。段階を選んでから回数を入れる"
Write-Host "      歩行、到達、歩きながら持つ、箱運び、キャッチ。到達と持ち歩きは歩行から、キャッチは到達から続ける。"
Write-Host "      歩行は平地から始めて地形へ広げ、腕を動かしながら歩く。腕は Z1。観測 81、行動 17（CPG+腕）。"
Write-Host "      歩行→簡単を選ぶと、トロット模倣学習してから PPO も選べます（単独は 1）。"
Write-Host "      Enter だけなら 1000 回、並列 64 体。同じ段階をもう一度選ぶと続きから学習する。"
Write-Host "   3  推論。段階を選んで、その保存した重みを再生する"
Write-Host "      箱運びだけ、クリックで物資と運び先を決める。簡単歩行では BC 重みの再生も選べます。"
Write-Host "   4  箱運びの途中経過を並べた比較動画（追加学習なし）"
Write-Host "   5  学習曲線を開く（追加学習なし）"
Write-Host "   6  プレゼン用の動画を作る（追加学習なし）"
Write-Host "   7  Colab 用 zip を起動スクリプトの PickAndCarry フォルダに作る"
Write-Host "   8  学習手法の結果を比較する（PPO / GRPO / Symmetric PPO）"
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

function Get-StageFolder {
    param([string]$Algo, [string]$Stage, [string]$Level = "static", [switch]$Easy)
    switch ($Stage) {
        "walk" {
            if ($Easy) { return "logs/${Algo}_curriculum_walk_easy" }
            return "logs/${Algo}_curriculum_walk"
        }
        "reach" { return "logs/${Algo}_curriculum_reach" }
        "haul" { return "logs/${Algo}_curriculum_haul" }
        "carry" { return "logs/${Algo}_z1" }
        "catch" { return "logs/${Algo}_curriculum_catch_$Level" }
        default { return "logs/${Algo}_z1" }
    }
}

function Format-Iter {
    param([int]$Value)
    if ($Value -le 0) { return "-" }
    return "$Value"
}

function Show-StageProgress {
    Write-Host " 腕は Unitree Z1（約 5.2 kg、関節 6）。観測 81、行動 17（CPG+腕）。"
    Write-Host " 学習済み iteration（最新の ppo_XXXX.pt）"
    Write-Host ("   {0,-14} {1,8} {2,8} {3,10}" -f "段階", "PPO", "GRPO", "Sym PPO")
    $rows = @(
        @{ Label = "歩行"; Stage = "walk"; Level = "static"; Easy = $false },
        @{ Label = "歩行(簡単)"; Stage = "walk"; Level = "static"; Easy = $true },
        @{ Label = "到達"; Stage = "reach"; Level = "static"; Easy = $false },
        @{ Label = "持ち歩き"; Stage = "haul"; Level = "static"; Easy = $false },
        @{ Label = "箱運び"; Stage = "carry"; Level = "static"; Easy = $false },
        @{ Label = "キャッチ静止"; Stage = "catch"; Level = "static"; Easy = $false },
        @{ Label = "キャッチ遅い"; Stage = "catch"; Level = "slow"; Easy = $false },
        @{ Label = "キャッチばらす"; Stage = "catch"; Level = "random"; Easy = $false },
        @{ Label = "キャッチ飛ぶ"; Stage = "catch"; Level = "fly"; Easy = $false },
        @{ Label = "キャッチノイズ"; Stage = "catch"; Level = "noisy"; Easy = $false }
    )
    foreach ($row in $rows) {
        $folderArgs = @{ Algo = "ppo"; Stage = $row.Stage; Level = $row.Level }
        if ($row.Easy) { $folderArgs.Easy = $true }
        $ppo = Get-LatestIteration ((Get-StageFolder @folderArgs) -replace "/", "\")
        $folderArgs.Algo = "grpo"
        $grpo = Get-LatestIteration ((Get-StageFolder @folderArgs) -replace "/", "\")
        $folderArgs.Algo = "sym_ppo"
        $sym = Get-LatestIteration ((Get-StageFolder @folderArgs) -replace "/", "\")
        Write-Host ("   {0,-14} {1,8} {2,8} {3,10}" -f $row.Label, (Format-Iter $ppo), (Format-Iter $grpo), (Format-Iter $sym))
    }
}

function Select-Algo {
    Write-Host ""
    Write-Host " 学習手法"
    Write-Host "   1  PPO（既定。価値関数つきの clipped PPO）"
    Write-Host "   2  GRPO（並列環境の相対報酬。批評家は更新しない）"
    Write-Host "   3  Symmetric PPO（左右反転のデータを足して学習）"
    $picked = Read-Host "手法 [1]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "1" }
    switch ($picked) {
        "2" {
            $script:algo = "grpo"
            Write-Host "GRPO で学習します。保存先は logs\grpo_* です。"
        }
        "3" {
            $script:algo = "sym_ppo"
            Write-Host "Symmetric PPO で学習します。保存先は logs\sym_ppo_* です。"
        }
        default {
            $script:algo = "ppo"
            Write-Host "PPO で学習します。保存先は logs\ppo_* です。"
        }
    }
    $script:algoArgs = @("--algo", $script:algo)
}

function Set-LogDirFromStage {
    $easy = [bool]$script:easyWalk
    $script:logDir = Get-StageFolder $script:algo $script:stage $(if ($script:catchLevel) { $script:catchLevel } else { "static" }) -Easy:$easy
    if ($script:stage -eq "carry") {
        $script:videoName = "PickAndCarry_$($script:algo)_z1.mp4"
    } elseif ($script:stage -eq "catch") {
        $script:videoName = "PickAndCarry_$($script:algo)_catch_$($script:catchLevel).mp4"
    } elseif ($script:stage -eq "walk" -and $easy) {
        $script:videoName = "PickAndCarry_$($script:algo)_walk_easy.mp4"
    } else {
        $script:videoName = "PickAndCarry_$($script:algo)_$($script:stage).mp4"
    }
    $script:activeIter = Get-LatestIteration ($script:logDir -replace "/", "\")
}
Show-StageProgress
Write-Host ""

$choice = Read-Host "番号を入力 [1]"
if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

function Select-Curriculum {
    param([switch]$ForPlay)
    Write-Host ""
    Write-Host " カリキュラム"
    Write-Host "   1  歩行。立つ・歩く・旋回。平地から始め、腕を動かしながら地形へ広げる"
    Write-Host "   2  到達。歩行の続き。空中の点へ手先を届かせる"
    Write-Host "   3  歩きながら持つ。到達の重みから続ける"
    Write-Host "   4  ピック＆プレース。置いてある箱を掴んで運ぶ"
    Write-Host "   5  ボールのキャッチ。到達の重みから続ける"
    $picked = Read-Host "段階 [1]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "1" }
    $script:stage = "carry"
    $script:catchLevel = "static"
    $script:easyWalk = $false
    $script:runBcFirst = $false
    $script:initFromArgs = @()
    $script:curriculumArgs = @()
    if ($picked -eq "2") {
        $script:stage = "reach"
        $script:curriculumArgs = @("--curriculum", "reach")
        Write-Host "到達です。歩行の重みから続けます。胴の前の空中に点が出て、手先がそこへ届いて少し保つと成功です。"
        Write-Host "脚は歩行の姿勢を保ちます。追加で選ぶ項目はありません。"
    } elseif ($picked -eq "3") {
        $script:stage = "haul"
        $script:curriculumArgs = @("--curriculum", "haul")
        Write-Host "掴んだまま歩く段階です。到達の重みから、脚も腕も同じネットワークで続けます。"
    } elseif ($picked -eq "4") {
        $script:stage = "carry"
        $script:curriculumArgs = @()
        Write-Host "ピック＆プレースです。歩きながら持つ段階と同じ観測・行動なので、その重みから続けます。"
        Write-Host "このあと、走行環境・学習の分け方・手先カメラを選びます。"
        if ($ForPlay) {
            Write-Host "推論では、クリックで物資と運び先を決められます。ROS 2 を選ぶと rviz2 で運び先を変えられます。"
        }
    } elseif ($picked -eq "5") {
        $script:stage = "catch"
        Write-Host "   1  手の前で静止"
        Write-Host "   2  ゆっくりまっすぐ転がす"
        Write-Host "   3  速さと方向をばらす"
        Write-Host "   4  転がして、宙を飛ばす。手の届く範囲を通る"
        Write-Host "   5  飛ぶボールに、位置のノイズを足す"
        $levelPick = Read-Host "ボール [1]"
        if ([string]::IsNullOrWhiteSpace($levelPick)) { $levelPick = "1" }
        $script:catchLevel = @("static", "slow", "random", "fly", "noisy")[[Math]::Min(4, [Math]::Max(0, [int]$levelPick - 1))]
        $script:curriculumArgs = @("--curriculum", "catch", "--catch-level", $script:catchLevel)
        Write-Host "キャッチ（$($script:catchLevel)）です。到達の重みから続けます。観測にはボールの速度と 0.3 秒先の位置が入ります。"
        if ($script:catchLevel -ne "static") {
            Write-Host "成績が低いあいだは易しいボールに戻し、回復したら上げます。"
        }
        if ($script:catchLevel -in @("fly", "noisy")) {
            Write-Host "投げたボールが床に落ちると、その試行は終わります。"
        }
    } else {
        $script:stage = "walk"
        $script:easyWalk = $false
        $script:curriculumArgs = @("--curriculum", "walk")
        if ($picked -ne "1") {
            Write-Host "不明な番号なので、歩行から始めます。"
        }
        Write-Host "歩行です。"
        Write-Host "   1  簡単。平坦・腕固定・押しなし（推奨のやり直し）。保存先 *_walk_easy"
        Write-Host "   2  本番。地形・腕外乱・押し・摩擦ばらつきあり"
        $walkMode = Read-Host "歩行モード [1]"
        if ([string]::IsNullOrWhiteSpace($walkMode)) { $walkMode = "1" }
        if ($walkMode -eq "2") {
            Write-Host "本番歩行です。平地から始め、成績が良いと地形を広げます。"
            Write-Host "止まる指令も混ぜ、腕は勝手に動かし、ときどき押し、床の摩擦もばらします。"
            Write-Host "このあと選ぶのは段差の高さだけです。"
        } else {
            $script:easyWalk = $true
            $script:curriculumArgs += @("--easy-walk")
            Write-Host "簡単歩行です。平坦で腕は固定、押しと摩擦ばらつきはオフです。"
            Write-Host "政策の腕アクションは学習から外します。段差の選択はスキップします。"
        }
    }
    Set-LogDirFromStage
    Write-Host "保存先は $($script:logDir -replace '/', '\') です。今の iteration: $($script:activeIter)"
    if (-not $ForPlay -and $script:easyWalk) {
        Ask-BcWarmStart
    }
    if (-not $ForPlay -and ($script:stage -in @("walk", "reach", "haul", "catch"))) {
        Ask-Retreat
    }
}

function Ask-BcWarmStart {
    $script:runBcFirst = $false
    $script:runGailFirst = $false
    $script:initFromArgs = @()
    Write-Host "   1  トロット教師を BC してから PPO（推奨・初回。PPO は iteration 0 から）"
    Write-Host "   2  トロット教師を ValueDICE してから PPO（PPO は iteration 0 から）"
    Write-Host "   3  既存の bc_trot.pt / valuedice_trot.pt / PPO 重みから PPO のみ"
    Write-Host "   4  ゼロから PPO のみ"
    Write-Host "   （模倣学習だけ・デモ再生だけはメニュー 1）"
    Write-Host "   （続き学習は 3。1/2 を選ぶと BC/ValueDICE のあと PPO を iteration 0 からやり直します）"
    $answer = Read-Host "初期化 [1]"
    if ([string]::IsNullOrWhiteSpace($answer)) { $answer = "1" }
    if ($answer -eq "2") {
        $script:runGailFirst = $true
        Write-Host "先にトロット教師のデモを集めて ValueDICE し、デモ動画のあと PPO を始めます（既存 ppo_*.pt は使いません）。"
    } elseif ($answer -eq "3") {
        $folder = Join-Path (Get-Location) ($script:logDir -replace "/", "\")
        $valuedice = Join-Path $folder "valuedice_trot.pt"
        $gail = Join-Path $folder "gail_trot.pt"
        $bc = Join-Path $folder "bc_trot.pt"
        $hasPpo = @(Get-ChildItem -Path $folder -Filter "ppo_*.pt" -File -ErrorAction SilentlyContinue).Count -gt 0
        $imit = $null
        $kind = ""
        if (Test-Path $valuedice) {
            $imit = $valuedice
            $kind = "ValueDICE"
        } elseif (Test-Path $gail) {
            $imit = $gail
            $kind = "GAIL"
        } elseif (Test-Path $bc) {
            $imit = $bc
            $kind = "BC"
        }
        if ($hasPpo -and $null -ne $imit) {
            Write-Host "PPO チェックポイントと $kind 重みの両方があります。"
            Write-Host "   1  PPO の続きから（進捗維持・推奨）"
            Write-Host "   2  $kind から暖気し直し（iteration 0。重みは $kind、乱数ゼロ開始ではない）"
            $pick = Read-Host "既存重み [1]"
            if ($pick -eq "2") {
                $script:initFromArgs = @("--init-from", $imit)
                Write-Host "$kind 重みで PPO を iteration 0 から始めます（--resume なし）: $imit"
            } else {
                Write-Host "既存の PPO を続行します（$kind 暖気は使いません）。"
            }
        } elseif ($null -ne $imit) {
            $script:initFromArgs = @("--init-from", $imit)
            Write-Host "既存の $kind 重みを初期値にします（--resume なし）: $imit"
        } elseif ($hasPpo) {
            Write-Host "既存の PPO を続行します。"
        } else {
            Write-Host "valuedice_trot.pt / gail_trot.pt / bc_trot.pt / ppo_*.pt が無いので、ゼロからになります。"
        }
    } elseif ($answer -eq "4") {
        Write-Host "模倣学習なしで PPO します（既存 ppo があれば --resume latest で続きます）。"
    } else {
        $script:runBcFirst = $true
        Write-Host "先にトロット教師のデモを集めて BC し、デモ動画のあと PPO を始めます（既存 ppo_*.pt は使いません）。"
    }
}

function Ask-TeacherDemoSave {
    Write-Host "   1  教師デモを保存する（リール＋スキル別 mp4）"
    Write-Host "   2  保存しない（学習だけ。動画は作らない）"
    $ans = Read-Host "教師デモ [1]"
    if ($ans -eq "2") {
        Write-Host "教師デモ mp4 は作りません。"
        return @("--save-demos", "none")
    }
    Write-Host "教師デモ mp4 を保存します。"
    return @("--save-demos", "all")
}

function Ask-EnvCoef {
    param([double]$Default = 10.0)
    while ($true) {
        $raw = Read-Host "env-coef (walk reward weight) [$Default]"
        if ([string]::IsNullOrWhiteSpace($raw)) { $raw = "$Default" }
        $parsed = 0.0
        if (-not [double]::TryParse($raw, [ref]$parsed) -or $parsed -lt 0) {
            Write-Host "Enter a number >= 0."
            continue
        }
        Write-Host "env-coef: $parsed"
        return $parsed
    }
}

function Ask-DiscEpochs {
    param([int]$Default = 2)
    while ($true) {
        $raw = Read-Host "nu-epochs (ValueDICE nu updates/iter) [$Default]"
        if ([string]::IsNullOrWhiteSpace($raw)) { $raw = "$Default" }
        $parsed = 0
        if (-not [int]::TryParse($raw, [ref]$parsed) -or $parsed -lt 1) {
            Write-Host "Enter an integer >= 1."
            continue
        }
        Write-Host "nu-epochs: $parsed"
        return $parsed
    }
}

function Ask-GailIterations {
    param([int]$Default = 1000)
    while ($true) {
        $raw = Read-Host "ValueDICE iterations [$Default]"
        if ([string]::IsNullOrWhiteSpace($raw)) { $raw = "$Default" }
        $parsed = 0
        if (-not [int]::TryParse($raw, [ref]$parsed) -or $parsed -lt 1) {
            Write-Host "Enter an integer >= 1."
            continue
        }
        Write-Host "ValueDICE iterations: $parsed"
        return $parsed
    }
}

function Ask-BcEpochs {
    param([int]$Default = 3000)
    $epochs = 0
    while ($epochs -lt 1) {
        $raw = Read-Host "BC エポック数 [$Default]"
        if ([string]::IsNullOrWhiteSpace($raw)) { $raw = "$Default" }
        $parsed = 0
        if (-not [int]::TryParse($raw, [ref]$parsed) -or $parsed -lt 1) {
            Write-Host "1 以上の整数を入力してください。"
            continue
        }
        $epochs = $parsed
    }
    Write-Host "BC エポック数: $epochs"
    return $epochs
}

function Ask-ValueDiceWarmStart {
    param([string]$LogDirRel)
    # Sets script:vdExtraArgs (flat) / vdBcEpochs for train_gail / train_valuedice.
    $script:vdExtraArgs = @()
    $script:vdBcEpochs = 3000
    $folder = Join-Path (Get-Location) ($LogDirRel -replace "/", "\")
    $valuedice = Join-Path $folder "valuedice_trot.pt"
    if (-not (Test-Path $valuedice)) { $valuedice = Join-Path $folder "gail_trot.pt" }
    $bc = Join-Path $folder "bc_trot.pt"
    $hasVd = Test-Path $valuedice
    $hasBc = Test-Path $bc
    if ($hasVd -and $hasBc) {
        Write-Host "既存の valuedice_trot.pt / bc_trot.pt があります。"
        Write-Host "   1  保存済み BC から ValueDICE（BC 再学習なし）"
        Write-Host "   2  ValueDICE 追加学習（続きから）"
        Write-Host "   3  BC からやり直して ValueDICE"
        $mode = Read-Host "ValueDICE [1]"
        if ([string]::IsNullOrWhiteSpace($mode)) { $mode = "1" }
        if ($mode -eq "2") {
            $script:vdExtraArgs = @("--resume", "latest", "--bc-epochs", "0")
            $script:vdBcEpochs = 0
            Write-Host "既存の ValueDICE 重みから追加学習します（BC なし）。"
        } elseif ($mode -eq "3") {
            $script:vdBcEpochs = 3000
            Write-Host "BC を温め直してから ValueDICE します。"
        } else {
            $script:vdExtraArgs = @("--init-from", $bc, "--bc-epochs", "0")
            $script:vdBcEpochs = 0
            Write-Host "保存 BC を使います（再学習しません）: $bc"
        }
    } elseif ($hasVd) {
        Write-Host "既存の valuedice_trot.pt / gail_trot.pt があります。"
        Write-Host "   1  最初から（BC 温め直し）"
        Write-Host "   2  追加学習（続きから・iter 数は次で指定）"
        $mode = Read-Host "ValueDICE [1]"
        if ($mode -eq "2") {
            $script:vdExtraArgs = @("--resume", "latest", "--bc-epochs", "0")
            $script:vdBcEpochs = 0
            Write-Host "既存の模倣重みから追加学習します（BC なし）。"
        }
    } elseif ($hasBc) {
        Write-Host "既存の bc_trot.pt があります。"
        Write-Host "   1  保存済み BC から ValueDICE（BC 再学習なし）"
        Write-Host "   2  BC からやり直して ValueDICE"
        $mode = Read-Host "ValueDICE [1]"
        if ($mode -eq "2") {
            $script:vdBcEpochs = 3000
            Write-Host "BC を温め直してから ValueDICE します。"
        } else {
            $script:vdExtraArgs = @("--init-from", $bc, "--bc-epochs", "0")
            $script:vdBcEpochs = 0
            Write-Host "保存 BC を使います（再学習しません）: $bc"
        }
    }
}

function Get-VideoTaskName {
    if ($script:stage -eq "carry") { return "carry" }
    if ($script:stage -eq "catch") { return "catch_$($script:catchLevel)" }
    if ($script:stage -eq "walk" -and $script:easyWalk) { return "walk_easy" }
    return $script:stage
}

function Copy-VideoTree {
    param([string]$LogDirRel)
    $srcRoot = Join-Path (Get-Location) (($LogDirRel -replace "/", "\") + "\videos")
    if (-not (Test-Path $srcRoot)) { return 0 }
    $dstRoot = Join-Path $out "videos"
    New-Item -ItemType Directory -Force -Path $dstRoot | Out-Null
    Copy-Item -Path (Join-Path $srcRoot "*") -Destination $dstRoot -Recurse -Force
    return @(Get-ChildItem -Path $srcRoot -Recurse -Filter "*.mp4" -ErrorAction SilentlyContinue).Count
}

function Open-BcDemos {
    param([string]$LogDirRel)
    $folder = Join-Path (Get-Location) ($LogDirRel -replace "/", "\")
    $videosRoot = Join-Path $folder "videos"
    $reels = @("trot_teacher_demo.mp4", "bc_trot_demo.mp4", "valuedice_trot_demo.mp4", "gail_trot_demo.mp4")
    $opened = 0
    # Prefer new layout: videos/<task>/<kind>_ep####_iter######_stamp/*.mp4
    if (Test-Path $videosRoot) {
        $runs = @(Get-ChildItem -Path $videosRoot -Directory -Recurse -ErrorAction SilentlyContinue |
            Where-Object { Get-ChildItem -Path $_.FullName -Filter "*.mp4" -File -ErrorAction SilentlyContinue } |
            Sort-Object LastWriteTime -Descending)
        foreach ($run in $runs) {
            $rel = $run.FullName.Substring($videosRoot.Length).TrimStart("\")
            Write-Host "デモフォルダ: videos\$rel"
            foreach ($name in $reels) {
                $path = Join-Path $run.FullName $name
                if (Test-Path $path) {
                    Write-Host "デモを開きます: $path"
                    Start-Process $path
                    $opened += 1
                }
            }
        }
        $copied = Copy-VideoTree $LogDirRel
        if ($copied -gt 0) {
            Write-Host "起動フォルダへコピー: $(Join-Path $out 'videos') （$copied 本）"
        }
    }
    # Legacy flat files in the log dir (pre-subfolder layout).
    if ($opened -eq 0) {
        foreach ($name in $reels) {
            $path = Join-Path $folder $name
            if (Test-Path $path) {
                Write-Host "デモを開きます: $path"
                Start-Process $path
                Copy-Item -Force $path (Join-Path $out $name)
                $opened += 1
            }
        }
    }
    if ($opened -eq 0) {
        Write-Host "デモ mp4 が見つかりません: $videosRoot"
    } else {
        Write-Host "デモ $opened 本を開きました（videos/<タスク>/<学習回>/ 以下）。"
    }
}

function Open-LatestPlayVideo {
    param([string]$LogDirRel, [string]$VideoName)
    $folder = Join-Path (Get-Location) ($LogDirRel -replace "/", "\")
    $videosRoot = Join-Path $folder "videos"
    $found = $null
    if (Test-Path $videosRoot) {
        $found = Get-ChildItem -Path $videosRoot -Recurse -Filter $VideoName -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
    }
    if (-not $found) {
        $flat = Join-Path $folder $VideoName
        if (Test-Path $flat) { $found = Get-Item $flat }
    }
    $copied = Copy-VideoTree $LogDirRel
    if ($found) {
        Write-Host "再生動画: $($found.FullName)"
        if ($copied -gt 0) { Write-Host "起動フォルダへコピー: $(Join-Path $out 'videos')" }
        Start-Process $found.FullName
    } else {
        Write-Host "再生動画が見つかりません: $VideoName"
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
    } elseif ($script:stage -eq "walk") {
        if (-not $script:easyWalk) {
            Select-Terrain -Walking
        }
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
$script:algo = "ppo"
$script:algoArgs = @("--algo", "ppo")
$script:catchLevel = "static"
$script:stage = "carry"
Set-LogDirFromStage
$code = 0

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "模倣学習（トロット教師 → BC / ValueDICE）。保存先は logs\<手法>_curriculum_walk_easy です。"
        Select-Algo
        $script:stage = "walk"
        $script:easyWalk = $true
        $script:catchLevel = "static"
        $script:curriculumArgs = @("--curriculum", "walk", "--easy-walk")
        Set-LogDirFromStage
        Write-Host "保存先: $($script:logDir -replace '/', '\')"
        Write-Host "   1  BC を実行（推奨・初回。デモの有無を選べます）"
        Write-Host "   2  ValueDICE を実行（ν＋PPO。必要なら BC で温める）"
        Write-Host "   3  既存の重みでデモを再生（bc_trot.pt / valuedice_trot.pt）"
        Write-Host "   4  教師デモだけ録画（立つ/座る/ジャンプ/前後/旋回/停止）"
        Write-Host "   5  既存のデモ動画だけ開く（学習なし）"
        $bcPick = Read-Host "模倣学習 [1]"
        if ([string]::IsNullOrWhiteSpace($bcPick)) { $bcPick = "1" }
        $logDir = $script:logDir
        if ($bcPick -eq "2") {
            $demoArgs = Ask-TeacherDemoSave
            Ask-ValueDiceWarmStart $logDir
            $bcEpochs = [int]$script:vdBcEpochs
            if ($bcEpochs -gt 0) {
                Write-Host "教師デモを集め（約90秒相当）たあと BC → ValueDICE します。"
                $bcEpochs = Ask-BcEpochs -Default $bcEpochs
            } else {
                Write-Host "教師デモを集め（約90秒相当）たあと ValueDICE のみ実行します（BC 学習なし）。"
            }
            $gailIters = Ask-GailIterations -Default 1000
            $envCoef = Ask-EnvCoef -Default 10
            $discEpochs = Ask-DiscEpochs -Default 2
            $gailCmd = @(
                "-u", "-m", "pick_and_carry.train_gail",
                "--easy-walk", "--num-envs", "32", "--seconds", "90",
                "--iterations", "$gailIters", "--env-coef", "$envCoef",
                "--nu-epochs", "$discEpochs", "--log-dir", "$logDir"
            ) + $algoArgs + $armArgs + $demoArgs
            if ($script:vdExtraArgs -and $script:vdExtraArgs.Count -gt 0) {
                $gailCmd += $script:vdExtraArgs
            } else {
                $gailCmd += @("--bc-epochs", "$bcEpochs")
            }
            Write-Host ("起動: python " + ($gailCmd -join " "))
            & $python @gailCmd
            $code = $LASTEXITCODE
            if ($code -eq 0) {
                Open-Mlflow
                if ($demoArgs -notcontains "none") { Open-BcDemos $logDir }
                Write-Host "続けてメニュー 2 → 歩行 → 簡単 →「既存の重みから PPO」を選ぶと valuedice_trot.pt を使えます。"
            }
        } elseif ($bcPick -eq "3") {
            $gailPath = Join-Path (Get-Location) (($logDir -replace "/", "\") + "\valuedice_trot.pt")
            if (-not (Test-Path $gailPath)) {
                $gailPath = Join-Path (Get-Location) (($logDir -replace "/", "\") + "\gail_trot.pt")
            }
            if (Test-Path $gailPath) {
                & $python -u -m pick_and_carry.train_gail --easy-walk --demo-only bc --log-dir $logDir @algoArgs @armArgs
            } else {
                & $python -u -m pick_and_carry.train_bc --easy-walk --demo-only bc --log-dir $logDir @algoArgs @armArgs
            }
            $code = $LASTEXITCODE
            if ($code -eq 0) { Open-BcDemos $logDir }
        } elseif ($bcPick -eq "4") {
            & $python -u -m pick_and_carry.train_bc --easy-walk --demo-only teacher --save-demos teacher --log-dir $logDir @algoArgs @armArgs
            $code = $LASTEXITCODE
            if ($code -eq 0) { Open-BcDemos $logDir }
        } elseif ($bcPick -eq "5") {
            $gailPath = Join-Path (Get-Location) (($logDir -replace "/", "\") + "\valuedice_trot.pt")
            if (-not (Test-Path $gailPath)) {
                $gailPath = Join-Path (Get-Location) (($logDir -replace "/", "\") + "\gail_trot.pt")
            }
            if (Test-Path $gailPath) {
                & $python -u -m pick_and_carry.train_gail --easy-walk --demo-only all --save-demos all --log-dir $logDir @algoArgs @armArgs
            } else {
                & $python -u -m pick_and_carry.train_bc --easy-walk --demo-only all --save-demos all --log-dir $logDir @algoArgs @armArgs
            }
            $code = $LASTEXITCODE
            if ($code -eq 0) { Open-BcDemos $logDir }
        } else {
            $demoArgs = Ask-TeacherDemoSave
            Write-Host "デモを集め（約90秒相当）て BC を実行します。"
            $bcEpochs = Ask-BcEpochs -Default 3000
            & $python -u -m pick_and_carry.train_bc --easy-walk --num-envs 32 --seconds 90 --epochs $bcEpochs --log-dir $logDir @algoArgs @armArgs @demoArgs
            $code = $LASTEXITCODE
            if ($code -eq 0) {
                Open-Mlflow
                if ($demoArgs -notcontains "none") { Open-BcDemos $logDir }
                Write-Host "続けてメニュー 2 → 歩行 → 簡単 →「既存の重みから PPO」を選ぶと、続けて PPO できます。"
            }
        }
    }
    "2" {
        Write-Host ""
        Write-Host "段階と学習手法を選んで学習を始めます。終わると MLflow、学習曲線、再生動画を開きます。"
        Select-Algo
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
            $rawEnvs = Read-Host "並列ロボット数 [64]"
            if ([string]::IsNullOrWhiteSpace($rawEnvs)) { $rawEnvs = "64" }
            $parsedEnvs = 0
            if (-not [int]::TryParse($rawEnvs, [ref]$parsedEnvs) -or $parsedEnvs -lt 1) {
                Write-Host "1 以上の整数を入力してください。"
                continue
            }
            $numEnvs = $parsedEnvs
        }
        if ($numEnvs -gt 32) {
            Write-Host "ノート GPU（例 8GB）では、並列を下げた方が安定します。$numEnvs 体のまま進めます。"
            Write-Host "他のロボットや重い GPU 作業と同時だと落ちやすいです。タスクマネージャで空きを確認してください。"
            Write-Host "足りなければ学習を止めます。開始前に GPU 使用率を確認してください。"
        }
        Select-StageOptions
        if ($script:runBcFirst) {
            Write-Host "トロット教師のデモを集めて模倣学習（BC）を実行します…"
            $demoArgs = Ask-TeacherDemoSave
            $bcEnvs = [Math]::Min(32, $numEnvs)
            $bcEpochs = Ask-BcEpochs -Default 3000
            & $python -u -m pick_and_carry.train_bc --easy-walk --num-envs $bcEnvs --seconds 90 --epochs $bcEpochs --log-dir $logDir @algoArgs @armArgs @demoArgs
            $code = $LASTEXITCODE
            if ($code -ne 0) { break }
            if ($demoArgs -notcontains "none") { Open-BcDemos $logDir }
            $bcPath = Join-Path (Get-Location) (($logDir -replace "/", "\") + "\bc_trot.pt")
            Write-Host "BC 完了。PPO は $bcPath から iteration 0 で開始します（既存の ppo_*.pt は使いません）。"
            & $python -u -m pick_and_carry.train_ppo --init-from $bcPath --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
        } elseif ($script:runGailFirst) {
            Write-Host "トロット教師のデモを集めて模倣学習（ValueDICE）を実行します…"
            $demoArgs = Ask-TeacherDemoSave
            $gailEnvs = [Math]::Min(32, $numEnvs)
            Ask-ValueDiceWarmStart $logDir
            $bcEpochs = [int]$script:vdBcEpochs
            if ($bcEpochs -gt 0) {
                $bcEpochs = Ask-BcEpochs -Default $bcEpochs
                Write-Host "教師デモ収集 → BC → ValueDICE の順で実行します。"
            } else {
                Write-Host "教師デモ収集 → ValueDICE のみ（BC 学習なし）。"
            }
            $gailIters = Ask-GailIterations -Default 1000
            $envCoef = Ask-EnvCoef -Default 10
            $discEpochs = Ask-DiscEpochs -Default 2
            $gailCmd = @(
                "-u", "-m", "pick_and_carry.train_gail",
                "--easy-walk", "--num-envs", "$gailEnvs", "--seconds", "90",
                "--iterations", "$gailIters", "--env-coef", "$envCoef",
                "--nu-epochs", "$discEpochs", "--log-dir", "$logDir"
            ) + $algoArgs + $armArgs + $demoArgs
            if ($script:vdExtraArgs -and $script:vdExtraArgs.Count -gt 0) {
                $gailCmd += $script:vdExtraArgs
            } else {
                $gailCmd += @("--bc-epochs", "$bcEpochs")
            }
            Write-Host ("起動: python " + ($gailCmd -join " "))
            & $python @gailCmd
            $code = $LASTEXITCODE
            if ($code -ne 0) { break }
            if ($demoArgs -notcontains "none") { Open-BcDemos $logDir }
            $gailPath = Join-Path (Get-Location) (($logDir -replace "/", "\") + "\valuedice_trot.pt")
            Write-Host "ValueDICE 完了。PPO は $gailPath から iteration 0 で開始します（既存の ppo_*.pt は使いません）。"
            & $python -u -m pick_and_carry.train_ppo --init-from $gailPath --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
        } else {
            if ($script:initFromArgs.Count -gt 0) {
                Write-Host "暖気起動: --init-from のみ（--resume は付けません）。"
                & $python -u -m pick_and_carry.train_ppo --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs @initFromArgs
            } else {
                & $python -u -m pick_and_carry.train_ppo --resume latest --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
            }
        }
        $code = $LASTEXITCODE
        Open-Mlflow
        if ($code -eq 0) {
            $curves = Join-Path $out "PickAndCarry_curves.html"
            & $python -u scripts\plot_curves.py --log-dir $logDir --output $curves
            if ($LASTEXITCODE -eq 0 -and (Test-Path $curves)) { Start-Process $curves }
            $video = Join-Path $out $videoName
            & $python -u -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --record $video --seconds 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
            $code = $LASTEXITCODE
            if ($code -eq 0) { Open-LatestPlayVideo $logDir $videoName }
        }
    }
    "3" {
        Write-Host ""
        Select-Algo
        Select-Curriculum -ForPlay
        $video = Join-Path $out $videoName
        Select-StageOptions -ForPlay
        if ($script:stage -eq "carry") {
            Write-Host "推論します。使う重み: iteration $script:activeIter （$logDir）"
            Write-Host "1回目のクリックが物資、2回目が運び先です。"
            & $python -u scripts\infer_gui.py --log-dir $logDir --record $video --seconds $script:inferSeconds @armArgs @terrainArgs @controlArgs @visionArgs @rosArgs
            $code = $LASTEXITCODE
            if ($code -eq 0) { Open-LatestPlayVideo $logDir $videoName }
        } elseif ($script:easyWalk) {
            $folder = Join-Path (Get-Location) ($logDir -replace "/", "\")
            $bcPath = Join-Path $folder "bc_trot.pt"
            $gailPath = Join-Path $folder "valuedice_trot.pt"
            if (-not (Test-Path $gailPath)) { $gailPath = Join-Path $folder "gail_trot.pt" }
            Write-Host "   1  PPO の最新重みを再生（iteration $script:activeIter）"
            Write-Host "   2  模倣の valuedice_trot.pt / bc_trot.pt を再生"
            Write-Host "   3  箱運びの途中経過を並べた比較動画（追加学習なし）"
            $playPick = Read-Host "手法 [1]"
            if ([string]::IsNullOrWhiteSpace($playPick)) { $playPick = "1" }
            if ($playPick -eq "2") {
                $imitPath = $null
                if (Test-Path $gailPath) { $imitPath = $gailPath }
                elseif (Test-Path $bcPath) { $imitPath = $bcPath }
                if ($null -eq $imitPath) {
                    Write-Host "valuedice_trot.pt / gail_trot.pt / bc_trot.pt がありません。先にメニュー 1 で模倣学習してください。"
                    $code = 1
                } else {
                    Write-Host "再生する重み: $imitPath"
                    & $python -u -m pick_and_carry.train_ppo --play --resume $imitPath --num-envs 1 --record $video --seconds 20 --log-dir $logDir --easy-walk @algoArgs @armArgs @curriculumArgs
                    $code = $LASTEXITCODE
                    if ($code -eq 0) { Open-LatestPlayVideo $logDir $videoName }
                }
            } elseif ($playPick -eq "3") {
                & $python -u -m pick_and_carry.train_bc --easy-walk --demo-only teacher --save-demos teacher --log-dir $logDir @algoArgs @armArgs
                $code = $LASTEXITCODE
                Open-BcDemos $logDir
            } else {
                Write-Host "推論します。使う重み: iteration $script:activeIter （$logDir）"
                & $python -u -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --record $video --seconds 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @curriculumArgs
                $code = $LASTEXITCODE
                if ($code -eq 0) { Open-LatestPlayVideo $logDir $videoName }
            }
        } else {
            Write-Host "推論します。使う重み: iteration $script:activeIter （$logDir）"
            & $python -u -m pick_and_carry.train_ppo --play --resume latest --num-envs 1 --record $video --seconds 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @curriculumArgs
            $code = $LASTEXITCODE
            if ($code -eq 0) { Open-LatestPlayVideo $logDir $videoName }
        }
    }
    "4" {
        Write-Host ""
        Select-Algo
        $script:stage = "carry"
        $script:catchLevel = "static"
        Set-LogDirFromStage
        Write-Host "比較用の箱運び動画を作ります。$($script:logDir -replace '/', '\') の途中チェックポイントを使います。"
        Write-Host "途中チェックポイントを等間隔に並べて、学習の進みを見比べます。最新は iteration $script:activeIter です。"
        Select-Terrain
        Select-Control
        Select-Vision
        $video = Join-Path $out "PickAndCarry_$($script:algo)_z1_compare.mp4"
        & $python -u scripts\compare_snapshots.py --log-dir $logDir --output $video --seconds 16 --panels 4 @armArgs @terrainArgs @controlArgs @visionArgs
        $code = $LASTEXITCODE
        if ($code -eq 0) { Open-LatestPlayVideo $logDir "PickAndCarry_$($script:algo)_z1_compare.mp4" }
    }
    "5" {
        Write-Host ""
        Select-Algo
        Select-Curriculum -ForPlay
        Write-Host "選択した学習記録から曲線を作り、ブラウザで開きます。対象: $logDir"
        $curves = Join-Path $out "PickAndCarry_curves.html"
        & $python -u scripts\plot_curves.py --log-dir $logDir --output $curves
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $curves)) { Start-Process $curves }
    }
    "6" {
        Write-Host ""
        Write-Host "学習条件のスライドを更新し、起動フォルダの動画からプレゼン動画を作ります。"
        & $python -u scripts\make_summary_slide.py
        if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; break }
        $video = Join-Path $out "PickAndCarry_presentation.mp4"
        & $python -u scripts\make_presentation.py --pptx (Join-Path $out "PickAndCarry_学習条件.pptx") --videos $out --output $video
        $code = $LASTEXITCODE
        if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
    }
    "7" {
        $zip = Join-Path $out "PAI_Task_colab.zip"
        if (Test-Path $zip) { Remove-Item -Force $zip }
        tar -a -c -f $zip --exclude=.venv --exclude=logs --exclude=__pycache__ --exclude=.git -C C:\Users\xeno1 PAI_Task
        $code = $LASTEXITCODE
        if ($code -eq 0) { Write-Host "作成しました: $zip" }
    }
    "8" {
        Write-Host ""
        Write-Host "同じ段階について、PPO / GRPO / Symmetric PPO の曲線を重ねます。"
        $script:algo = "ppo"
        $script:algoArgs = @("--algo", "ppo")
        Select-Curriculum -ForPlay
        $dirs = @()
        $labels = @()
        foreach ($name in @("ppo", "grpo", "sym_ppo")) {
            $folder = Get-StageFolder $name $script:stage $(if ($script:catchLevel) { $script:catchLevel } else { "static" }) -Easy:([bool]$script:easyWalk)
            $metrics = Join-Path (Get-Location) (($folder -replace "/", "\") + "\metrics.jsonl")
            $hasPt = (Get-LatestIteration ($folder -replace "/", "\")) -gt 0
            if ((Test-Path $metrics) -or $hasPt) {
                $dirs += $folder
                $labels += $name
            }
        }
        if ($dirs.Count -lt 2) {
            Write-Host "比較できる手法が 2 つ未満です。別手法でも同じ段階を学習してから再実行してください。"
            $code = 1
        } else {
            $curves = Join-Path $out "PickAndCarry_compare_curves.html"
            & $python -u scripts\plot_curves.py --compare @dirs --labels @labels --output $curves
            $code = $LASTEXITCODE
            if ($code -eq 0 -and (Test-Path $curves)) { Start-Process $curves }
        }
    }
    default {
        Write-Host "不明な番号です: $choice"
        $code = 1
    }
}
Write-Host ""
Write-Host "exit code: $code"
if ($choice -eq "2" -and $code -eq 0) {
    exit $code
}
Read-Host "内容を確認し終わったらキーを押してください"
exit $code
