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
Write-Host "   1  模倣学習（トロット教師 → BC / DAgger / ValueDICE）。デモ保存の有無を選べます"
Write-Host "      保存先: logs\imitation\ （BC/DAgger/ValueDICE。PPO の walk フォルダとは並列）"
Write-Host "   2  学習。段階を選んでから回数を入れる"
Write-Host "      歩行、到達、歩きながら持つ、箱運び、キャッチ。到達と持ち歩きは歩行から、キャッチは到達から続ける。"
Write-Host "      歩行は平地から始めて地形へ広げ、腕を動かしながら歩く。腕は Z1。観測 81、行動 17（CPG+腕）。"
Write-Host "      歩行→簡単の推奨は 保存DAgger/BC→PPO（無ければ BC→PPO）。ValueDICE は任意。模倣だけは 1。"
Write-Host "      Enter だけなら 1000 回、並列 64 体。同じ段階をもう一度選ぶと続きから学習する。"
Write-Host "   3  推論。段階を選んで、その保存した重みを再生する"
Write-Host "      箱運びだけ、クリックで物資と運び先を決める。簡単歩行では BC 重みの再生も選べます。"
Write-Host "   4  箱運びの途中経過を並べた比較動画（追加学習なし）"
Write-Host "   5  学習曲線を開く（追加学習なし）"
Write-Host "   6  プレゼン用の動画を作る（追加学習なし）"
Write-Host "   7  Colab 用 zip を起動スクリプトの PickAndCarry フォルダに作る"
Write-Host "   8  学習手法の結果を比較する（PPO / GRPO / Symmetric PPO）"
Write-Host "   9  DVC / MinIO（チェックポイント共有。Docker で MinIO 起動・publish / pull）"
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
    Write-Host ("   {0,-14} {1,8} {2,8} {3,8} {4,10}" -f "段階", "PPO", "GRPO", "DrGRPO", "Sym PPO")
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
        $folderArgs.Algo = "dr_grpo"
        $dr = Get-LatestIteration ((Get-StageFolder @folderArgs) -replace "/", "\")
        $folderArgs.Algo = "sym_ppo"
        $sym = Get-LatestIteration ((Get-StageFolder @folderArgs) -replace "/", "\")
        Write-Host ("   {0,-14} {1,8} {2,8} {3,8} {4,10}" -f $row.Label, (Format-Iter $ppo), (Format-Iter $grpo), (Format-Iter $dr), (Format-Iter $sym))
    }
}

function Select-Algo {
    Write-Host ""
    Write-Host " 学習手法"
    Write-Host "   1  PPO（既定。価値関数つきの clipped PPO）"
    Write-Host "   2  GRPO（群内 mean/std 相対。批評家なし）"
    Write-Host "   3  Dr. GRPO（群内 mean のみ。/std なし・COLM 2025）"
    Write-Host "   4  Symmetric PPO（左右反転のデータを足して学習）"
    $picked = Read-Host "手法 [1]"
    if ([string]::IsNullOrWhiteSpace($picked)) { $picked = "1" }
    switch ($picked) {
        "2" {
            $script:algo = "grpo"
            Write-Host "GRPO で学習します。保存先は logs\grpo_* です。"
        }
        "3" {
            $script:algo = "dr_grpo"
            Write-Host "Dr. GRPO で学習します。保存先は logs\dr_grpo_* です。"
        }
        "4" {
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
    $script:runDaggerFirst = $false
    $script:runGailFirst = $false
    $script:initFromArgs = @()
    $script:noAutoInit = $false
    $script:curriculumArgs = @()
    if ($picked -eq "2") {
        $script:stage = "reach"
        $script:curriculumArgs = @("--curriculum", "reach")
        Write-Host "到達です。歩行の重みから続けます。胴の前の空中に点が出て、手先がそこへ届いて少し保つと成功です。"
        Write-Host "脚は歩行の姿勢を保ちます。このあと初期重みを選べます。"
    } elseif ($picked -eq "3") {
        $script:stage = "haul"
        $script:curriculumArgs = @("--curriculum", "haul")
        Write-Host "掴んだまま歩く段階です。到達の重みから、脚も腕も同じネットワークで続けます。"
        Write-Host "このあと初期重みを選べます。"
    } elseif ($picked -eq "4") {
        $script:stage = "carry"
        $script:curriculumArgs = @()
        Write-Host "ピック＆プレースです。歩きながら持つ段階と同じ観測・行動なので、その重みから続けます。"
        Write-Host "このあと、走行環境・学習の分け方・手先カメラと、初期重みを選びます。"
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
        Write-Host "このあと初期重みを選べます。"
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
        Write-Host "   2  本番。地形・押し・摩擦ばらつきあり（腕は固定）"
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
            Write-Host "推奨の初期化は 保存DAgger/BC→PPO（歩行改善向き）。ValueDICE は任意です。"
        }
    }
    Set-LogDirFromStage
    Write-Host "保存先は $($script:logDir -replace '/', '\') です。今の iteration: $($script:activeIter)"
    if (-not $ForPlay -and $script:stage -eq "walk") {
        Ask-BcWarmStart
    }
    if (-not $ForPlay -and ($script:stage -in @("reach", "haul", "carry", "catch"))) {
        Ask-LaterStageInit
    }
    if (-not $ForPlay -and ($script:stage -in @("walk", "reach", "haul", "catch"))) {
        Ask-Retreat
    }
}

function Get-ImitationFolder {
    # Shared imitation dir, sibling of ppo_curriculum_walk_easy under logs/.
    return (Join-Path (Get-Location) "logs\imitation")
}

function Find-ImitationWeight {
    param([string]$Name)
    # logs/imitation first, then legacy nested/flat walk folders.
    $folders = @()
    $folders += (Join-Path (Get-Location) "logs\imitation")
    $rl = Join-Path (Get-Location) ($script:logDir -replace "/", "\")
    $folders += (Join-Path $rl "imitation")
    $folders += $rl
    if ($script:logDir -match 'curriculum_walk$' -and $script:logDir -notmatch 'curriculum_walk_easy$') {
        $easy = Join-Path (Get-Location) (($script:logDir + "_easy") -replace "/", "\")
        $folders += (Join-Path $easy "imitation")
        $folders += $easy
    } elseif ($script:logDir -match 'curriculum_walk_easy$') {
        $folders += (Join-Path $rl "imitation")
    }
    foreach ($dir in $folders) {
        $path = Join-Path $dir $Name
        if (Test-Path $path) { return $path }
    }
    return $null
}

function Get-ImitationWeightPath {
    param([string]$Name)
    $found = Find-ImitationWeight $Name
    if ($null -ne $found) { return $found }
    return (Join-Path (Get-ImitationFolder) $Name)
}

function Get-PriorStageLogDir {
    # Default folder to browse when picking a transfer weight for later stages.
    $algo = if ($script:algo) { $script:algo } else { "ppo" }
    switch ($script:stage) {
        "reach" { return (Get-StageFolder $algo "walk" "static") }
        "haul" { return (Get-StageFolder $algo "reach" "static") }
        "carry" { return (Get-StageFolder $algo "haul" "static") }
        "catch" { return (Get-StageFolder $algo "reach" "static") }
        default { return $script:logDir }
    }
}

function Select-InitWeightFile {
    param(
        [string]$PreferredDir = "",
        [string]$Title = "初期重みを選択（*.pt）"
    )
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($PreferredDir)) {
        $candidates += (Join-Path (Get-Location) ($PreferredDir -replace "/", "\"))
    }
    $candidates += (Get-ImitationFolder)
    $candidates += (Join-Path (Get-Location) ($script:logDir -replace "/", "\"))
    $candidates += (Join-Path (Get-Location) "logs")
    $candidates += (Get-Location).Path
    $startDir = $null
    foreach ($dir in $candidates) {
        if (Test-Path $dir) { $startDir = $dir; break }
    }
    if ($null -eq $startDir) { $startDir = (Get-Location).Path }
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.Title = $Title
        $dialog.Filter = "PyTorch weights (*.pt)|*.pt|All files (*.*)|*.*"
        $dialog.FilterIndex = 1
        $dialog.InitialDirectory = $startDir
        $dialog.Multiselect = $false
        $dialog.CheckFileExists = $true
        $result = $dialog.ShowDialog()
        if ($result -eq [System.Windows.Forms.DialogResult]::OK -and -not [string]::IsNullOrWhiteSpace($dialog.FileName)) {
            return $dialog.FileName
        }
    } catch {
        Write-Host "ファイル選択ウィンドウを開けませんでした: $($_.Exception.Message)"
    }
    return $null
}

function Ask-LaterStageInit {
    # Reach / haul / carry / catch: continue automatically, or pick an explicit .pt.
    $script:initFromArgs = @()
    $script:noAutoInit = $false
    $label = switch ($script:stage) {
        "reach" { "到達" }
        "haul" { "持ち歩き" }
        "carry" { "箱運び" }
        "catch" { "キャッチ" }
        default { $script:stage }
    }
    $prior = Get-PriorStageLogDir
    $priorWin = $prior -replace "/", "\"
    Write-Host ""
    Write-Host "$label の初期化:"
    Write-Host "   1  自動（この段階の続き。無ければ前段階の最新）← 推奨"
    Write-Host "   2  重みファイルを選ぶ（--init-from。別フォルダなら iteration 0）"
    Write-Host "      候補フォルダ例: $priorWin"
    $answer = Read-Host "初期化 [1]"
    if ([string]::IsNullOrWhiteSpace($answer)) { $answer = "1" }
    if ($answer -eq "2") {
        Write-Host "ファイル選択ウィンドウを開きます（キャンセル後はパスを手入力できます）。"
        $picked = Select-InitWeightFile -PreferredDir $prior -Title "$label の初期重みを選択（前段階の ppo_*.pt など）"
        if (-not [string]::IsNullOrWhiteSpace($picked)) {
            $script:initFromArgs = @("--init-from", $picked)
            Write-Host "指定重みを --init-from します: $picked"
        } else {
            Write-Host "キャンセルされました。手入力する場合は .pt のパスを入れてください。"
            $raw = Read-Host "初期重みパス（空=自動に戻す）"
            if ([string]::IsNullOrWhiteSpace($raw)) {
                Write-Host "未入力なので自動（--resume latest）にします。"
            } else {
                $script:initFromArgs = @("--init-from", $raw.Trim().Trim('"'))
                Write-Host "指定重みを --init-from します: $($script:initFromArgs[1])"
            }
        }
    } else {
        Write-Host "自動で続けます（この段階に重みが無ければ前段階から借ります）。"
    }
}

function Ask-BcWarmStart {
    $script:runBcFirst = $false
    $script:runDaggerFirst = $false
    $script:runGailFirst = $false
    $script:initFromArgs = @()
    $script:noAutoInit = $false
    $folder = Join-Path (Get-Location) ($script:logDir -replace "/", "\")
    $dagger = Find-ImitationWeight "dagger_trot.pt"
    $valuedice = Find-ImitationWeight "valuedice_trot.pt"
    $gail = Find-ImitationWeight "gail_trot.pt"
    $bc = Find-ImitationWeight "bc_trot.pt"
    $hasDagger = $null -ne $dagger
    $hasBc = $null -ne $bc
    $modeLabel = if ($script:easyWalk) { "簡単歩行" } else { "本番歩行" }
    Write-Host "$modeLabel の初期化（推奨は DAgger/BC → PPO。ファイルパス指定可）:"
    if ($hasDagger) {
        Write-Host "   1  保存 DAgger から PPO（推奨。再学習なし・iteration 0）"
        Write-Host "      $dagger"
    } elseif ($hasBc) {
        Write-Host "   1  保存 BC から PPO（推奨。BC 再学習なし・iteration 0）"
        Write-Host "      $bc"
    } else {
        Write-Host "   1  BC してから PPO（推奨・初回。保存模倣がまだ無い）"
    }
    Write-Host "   2  既存の模倣 / PPO 重みから PPO（優先: DAgger → BC → ValueDICE → PPO 続き）"
    Write-Host "   3  DAgger してから PPO（対話的模倣・推奨の次点）"
    Write-Host "   4  ValueDICE してから PPO（任意・上級。PPO は iteration 0）"
    Write-Host "   5  ゼロから PPO のみ"
    Write-Host "   6  重みファイルを選ぶ（--init-from。模倣は iter 0、同フォルダの ppo_*.pt は続き）"
    Write-Host "   （模倣学習だけ・デモ再生だけはメニュー 1）"
    $answer = Read-Host "初期化 [1]"
    if ([string]::IsNullOrWhiteSpace($answer)) { $answer = "1" }
    if ($answer -eq "2") {
        $hasPpo = @(Get-ChildItem -Path $folder -Filter "ppo_*.pt" -File -ErrorAction SilentlyContinue).Count -gt 0
        $imit = $null
        $kind = ""
        if ($hasDagger) {
            $imit = $dagger
            $kind = "DAgger"
        } elseif ($hasBc) {
            $imit = $bc
            $kind = "BC"
        } elseif ($null -ne $valuedice) {
            $imit = $valuedice
            $kind = "ValueDICE"
        } elseif ($null -ne $gail) {
            $imit = $gail
            $kind = "GAIL"
        }
        if ($hasPpo -and $null -ne $imit) {
            Write-Host "PPO チェックポイントと $kind 重みの両方があります。"
            Write-Host "   1  PPO の続きから（進捗維持・推奨）"
            Write-Host "   2  $kind から暖気し直し（iteration 0。重みは $kind）"
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
            Write-Host "dagger_trot.pt / bc_trot.pt / valuedice_trot.pt / ppo_*.pt が無いので、ゼロからになります。"
        }
    } elseif ($answer -eq "3") {
        $script:runDaggerFirst = $true
        Write-Host "先に DAgger（教師クエリ付き模倣）し、そのあと PPO を始めます（既存 ppo_*.pt は使いません）。"
    } elseif ($answer -eq "4") {
        $script:runGailFirst = $true
        Write-Host "先にトロット教師のデモを集めて ValueDICE し、そのあと PPO を始めます（既存 ppo_*.pt は使いません）。"
    } elseif ($answer -eq "5") {
        $script:noAutoInit = $true
        Write-Host "模倣学習なしで PPO します（既存 ppo があれば --resume latest。無い場合は本当にゼロから・模倣重みは使いません）。"
    } elseif ($answer -eq "6") {
        Write-Host "ファイル選択ウィンドウを開きます（キャンセル後はパスや別名を手入力できます）。"
        $picked = Select-InitWeightFile
        if (-not [string]::IsNullOrWhiteSpace($picked)) {
            $script:initFromArgs = @("--init-from", $picked)
            Write-Host "指定重みを --init-from します（模倣→iter 0、同フォルダ ppo_*.pt→続き）: $picked"
        } else {
            Write-Host "キャンセルされました。手入力する場合はパスまたは別名（bc / dagger / latest）を入れてください。"
            $raw = Read-Host "初期重みパスまたは別名（空=ゼロから）"
            if ([string]::IsNullOrWhiteSpace($raw)) {
                Write-Host "未入力なのでゼロから PPO します。"
                $script:noAutoInit = $true
            } else {
                $script:initFromArgs = @("--init-from", $raw.Trim().Trim('"'))
                Write-Host "指定重みを --init-from します（模倣→iter 0、同フォルダ ppo_*.pt→続き）: $($script:initFromArgs[1])"
            }
        }
    } else {
        if ($hasDagger) {
            $script:initFromArgs = @("--init-from", $dagger)
            Write-Host "保存 DAgger から PPO を iteration 0 で始めます: $dagger"
        } elseif ($hasBc) {
            $script:initFromArgs = @("--init-from", $bc)
            Write-Host "保存 BC から PPO を iteration 0 で始めます（BC 再学習なし）: $bc"
        } else {
            $script:runBcFirst = $true
            Write-Host "保存模倣が無いので、先に BC してから PPO を始めます（既存 ppo_*.pt は使いません）。"
        }
    }
}

function Ask-DaggerMode {
    Write-Host "   1  classic DAgger（全状態を教師ラベル）"
    Write-Host "   2  gated DAgger（方策と教師の差が大きいときだけラベル）"
    $ans = Read-Host "DAgger モード [1]"
    if ($ans -eq "2") { return "gated" }
    return "classic"
}

function Ask-DaggerRounds {
    param([int]$Default = 8)
    while ($true) {
        $raw = Read-Host "DAgger rounds [$Default]"
        if ([string]::IsNullOrWhiteSpace($raw)) { $raw = "$Default" }
        $parsed = 0
        if (-not [int]::TryParse($raw, [ref]$parsed) -or $parsed -lt 1) {
            Write-Host "1 以上の整数を入力してください。"
            continue
        }
        Write-Host "DAgger rounds: $parsed"
        return $parsed
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
    $imit = Join-Path (Get-Location) "logs\imitation"
    $legacyNested = Join-Path (Get-Location) (($LogDirRel -replace "/", "\") + "\imitation")
    $legacy = Join-Path (Get-Location) ($LogDirRel -replace "/", "\")
    $folder = if (Test-Path $imit) { $imit } elseif (Test-Path $legacyNested) { $legacyNested } else { $legacy }
    $valuedice = Join-Path $folder "valuedice_trot.pt"
    if (-not (Test-Path $valuedice)) { $valuedice = Join-Path $legacyNested "valuedice_trot.pt" }
    if (-not (Test-Path $valuedice)) { $valuedice = Join-Path $legacy "valuedice_trot.pt" }
    if (-not (Test-Path $valuedice)) { $valuedice = Join-Path $folder "gail_trot.pt" }
    if (-not (Test-Path $valuedice)) { $valuedice = Join-Path $legacy "gail_trot.pt" }
    $bc = Join-Path $folder "bc_trot.pt"
    if (-not (Test-Path $bc)) { $bc = Join-Path $legacyNested "bc_trot.pt" }
    if (-not (Test-Path $bc)) { $bc = Join-Path $legacy "bc_trot.pt" }
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
    $imit = Join-Path (Get-Location) "logs\imitation"
    $legacyNested = Join-Path (Get-Location) (($LogDirRel -replace "/", "\") + "\imitation")
    $legacy = Join-Path (Get-Location) ($LogDirRel -replace "/", "\")
    if (Test-Path (Join-Path $imit "videos")) { $folder = $imit }
    elseif (Test-Path (Join-Path $legacyNested "videos")) { $folder = $legacyNested }
    elseif (Test-Path $imit) { $folder = $imit }
    elseif (Test-Path $legacyNested) { $folder = $legacyNested }
    else { $folder = $legacy }
    $videosRoot = Join-Path $folder "videos"
    $imitVideos = Join-Path $videosRoot "imitation"
    $scanRoot = if (Test-Path $imitVideos) { $imitVideos } else { $videosRoot }
    $reels = @("trot_teacher_demo.mp4", "bc_trot_demo.mp4", "dagger_trot_demo.mp4", "valuedice_trot_demo.mp4", "gail_trot_demo.mp4")
    $opened = 0
    # Prefer: videos/imitation/<task>/<kind>_ep####_iter######_stamp/*.mp4
    if (Test-Path $scanRoot) {
        $runs = @(Get-ChildItem -Path $scanRoot -Directory -Recurse -ErrorAction SilentlyContinue |
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
        # Copy imitation video tree (shared logs/imitation), not the RL walk folder.
        $imitRel = if ($folder -eq $imit) { "logs/imitation" } elseif ($folder -eq $legacyNested) { "$LogDirRel/imitation" } else { $LogDirRel }
        $copied = Copy-VideoTree $imitRel
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
        Write-Host "デモ mp4 が見つかりません: $scanRoot"
    } else {
        Write-Host "デモ $opened 本を開きました（videos/imitation/<タスク>/<学習回>/ 以下）。"
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
        Write-Host "模倣学習（トロット教師 → BC / DAgger / ValueDICE）。"
        Write-Host "保存先は logs\imitation\ です（logs\*_curriculum_walk_easy と並列。PPO とは別）。"
        Select-Algo
        $script:stage = "walk"
        $script:easyWalk = $true
        $script:catchLevel = "static"
        $script:curriculumArgs = @("--curriculum", "walk", "--easy-walk")
        Set-LogDirFromStage
        Write-Host "模倣保存先: $((Get-ImitationFolder).Replace((Get-Location).Path + '\', ''))"
        Write-Host "（PPO 保存先: $($script:logDir -replace '/', '\')）"
        Write-Host "   1  BC を実行（推奨・初回。デモの有無を選べます）"
        Write-Host "   2  DAgger を実行（対話的模倣。classic / gated）"
        Write-Host "   3  ValueDICE を実行（ν＋PPO。必要なら BC で温める）"
        Write-Host "   4  既存の重みでデモを再生（bc / dagger / valuedice）"
        Write-Host "   5  教師デモだけ録画（立つ/座る/ジャンプ/前後/旋回/停止）"
        Write-Host "   6  既存のデモ動画だけ開く（学習なし）"
        $bcPick = Read-Host "模倣学習 [1]"
        if ([string]::IsNullOrWhiteSpace($bcPick)) { $bcPick = "1" }
        $logDir = $script:logDir
        if ($bcPick -eq "2") {
            $demoArgs = Ask-TeacherDemoSave
            $mode = Ask-DaggerMode
            $rounds = Ask-DaggerRounds -Default 8
            $initArgs = @()
            $bcPath = (Get-ImitationWeightPath "bc_trot.pt")
            if (Test-Path $bcPath) {
                Write-Host "既存 bc_trot.pt があります。DAgger の初期方策に使いますか？"
                Write-Host "   1  使う（推奨）"
                Write-Host "   2  使わない（教師シードから）"
                $useBc = Read-Host "初期方策 [1]"
                if ($useBc -ne "2") {
                    $initArgs = @("--init-from", $bcPath)
                    Write-Host "BC を初期方策にします: $bcPath"
                }
            }
            Write-Host "DAgger ($mode) rounds=$rounds を実行します。"
            & $python -u -m pick_and_carry.train_dagger --easy-walk --num-envs 32 --seconds 30 --rounds $rounds --mode $mode --log-dir $logDir @algoArgs @armArgs @demoArgs @initArgs
            $code = $LASTEXITCODE
            if ($code -eq 0) {
                Open-Mlflow
                if ($demoArgs -notcontains "none") { Open-BcDemos $logDir }
                Write-Host "続けてメニュー 2 → 歩行 → 簡単 → 初期化 1（保存DAgger→PPO）で続けられます。"
            }
        } elseif ($bcPick -eq "3") {
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
                Write-Host "続けてメニュー 2 → 歩行 → 簡単 → 初期化 2（既存重み）で PPO できます。歩行改善なら BC→PPO（初期化 1）推奨。"
            }
        } elseif ($bcPick -eq "4") {
            $daggerPath = (Get-ImitationWeightPath "dagger_trot.pt")
            $gailPath = (Get-ImitationWeightPath "valuedice_trot.pt")
            if (-not (Test-Path $gailPath)) {
                $gailPath = (Get-ImitationWeightPath "gail_trot.pt")
            }
            if (Test-Path $daggerPath) {
                & $python -u -m pick_and_carry.train_dagger --easy-walk --demo-only bc --log-dir $logDir @algoArgs @armArgs
            } elseif (Test-Path $gailPath) {
                & $python -u -m pick_and_carry.train_gail --easy-walk --demo-only bc --log-dir $logDir @algoArgs @armArgs
            } else {
                & $python -u -m pick_and_carry.train_bc --easy-walk --demo-only bc --log-dir $logDir @algoArgs @armArgs
            }
            $code = $LASTEXITCODE
            if ($code -eq 0) { Open-BcDemos $logDir }
        } elseif ($bcPick -eq "5") {
            & $python -u -m pick_and_carry.train_bc --easy-walk --demo-only teacher --save-demos teacher --log-dir $logDir @algoArgs @armArgs
            $code = $LASTEXITCODE
            if ($code -eq 0) { Open-BcDemos $logDir }
        } elseif ($bcPick -eq "6") {
            $daggerPath = (Get-ImitationWeightPath "dagger_trot.pt")
            $gailPath = (Get-ImitationWeightPath "valuedice_trot.pt")
            if (-not (Test-Path $gailPath)) {
                $gailPath = (Get-ImitationWeightPath "gail_trot.pt")
            }
            if (Test-Path $daggerPath) {
                & $python -u -m pick_and_carry.train_dagger --easy-walk --demo-only all --save-demos all --log-dir $logDir @algoArgs @armArgs
            } elseif (Test-Path $gailPath) {
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
                Write-Host "続けてメニュー 2 → 歩行 → 簡単 → 初期化 1（保存BC→PPO）で続けられます。"
            }
        }
    }
    "2" {
        Write-Host ""
        Write-Host "段階と学習手法を選んで学習を始めます。終わると MLflow、学習曲線、再生動画を開きます。"
        Write-Host "簡単歩行なら推奨は 保存DAgger/BC→PPO（メニューで初期化 1）。"
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
            $bcPath = (Get-ImitationWeightPath "bc_trot.pt")
            Write-Host "BC 完了。PPO は $bcPath から iteration 0 で開始します（既存の ppo_*.pt は使いません）。"
            & $python -u -m pick_and_carry.train_ppo --init-from $bcPath --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
        } elseif ($script:runDaggerFirst) {
            Write-Host "トロット教師つき DAgger を実行してから PPO します…"
            $demoArgs = Ask-TeacherDemoSave
            $mode = Ask-DaggerMode
            $rounds = Ask-DaggerRounds -Default 8
            $daggerEnvs = [Math]::Min(32, $numEnvs)
            $initArgs = @()
            $bcPath = (Get-ImitationWeightPath "bc_trot.pt")
            if (Test-Path $bcPath) {
                $initArgs = @("--init-from", $bcPath)
                Write-Host "既存 BC を DAgger 初期方策に使います: $bcPath"
            }
            & $python -u -m pick_and_carry.train_dagger --easy-walk --num-envs $daggerEnvs --seconds 30 --rounds $rounds --mode $mode --log-dir $logDir @algoArgs @armArgs @demoArgs @initArgs
            $code = $LASTEXITCODE
            if ($code -ne 0) { break }
            if ($demoArgs -notcontains "none") { Open-BcDemos $logDir }
            $daggerPath = (Get-ImitationWeightPath "dagger_trot.pt")
            Write-Host "DAgger 完了。PPO は $daggerPath から iteration 0 で開始します（既存の ppo_*.pt は使いません）。"
            & $python -u -m pick_and_carry.train_ppo --init-from $daggerPath --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
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
            $gailPath = (Get-ImitationWeightPath "valuedice_trot.pt")
            Write-Host "ValueDICE 完了。PPO は $gailPath から iteration 0 で開始します（既存の ppo_*.pt は使いません）。"
            & $python -u -m pick_and_carry.train_ppo --init-from $gailPath --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs
        } else {
            $noAutoArgs = @()
            if ($script:noAutoInit) { $noAutoArgs = @("--no-auto-init-from") }
            if ($script:initFromArgs.Count -gt 0) {
                Write-Host "暖気起動: --init-from のみ（--resume は付けません）。"
                & $python -u -m pick_and_carry.train_ppo --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs @initFromArgs @noAutoArgs
            } else {
                & $python -u -m pick_and_carry.train_ppo --resume latest --num-envs $numEnvs --iterations $trainIters --num-steps 128 --save-interval 20 --log-dir $logDir @algoArgs @armArgs @terrainArgs @controlArgs @visionArgs @curriculumArgs @noAutoArgs
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
            $bcPath = Get-ImitationWeightPath "bc_trot.pt"
            $daggerPath = Get-ImitationWeightPath "dagger_trot.pt"
            $gailPath = Get-ImitationWeightPath "valuedice_trot.pt"
            if (-not (Test-Path $gailPath)) { $gailPath = Get-ImitationWeightPath "gail_trot.pt" }
            Write-Host "   1  PPO の最新重みを再生（iteration $script:activeIter）"
            Write-Host "   2  模倣の dagger / bc / valuedice 重みを再生"
            Write-Host "   3  教師トロットのデモを録画して開く"
            Write-Host "   4  歩行 PPO の途中経過を並べた比較動画（追加学習なし）"
            $playPick = Read-Host "手法 [1]"
            if ([string]::IsNullOrWhiteSpace($playPick)) { $playPick = "1" }
            if ($playPick -eq "2") {
                $imitPath = $null
                if (Test-Path $daggerPath) { $imitPath = $daggerPath }
                elseif (Test-Path $bcPath) { $imitPath = $bcPath }
                elseif (Test-Path $gailPath) { $imitPath = $gailPath }
                if ($null -eq $imitPath) {
                    Write-Host "dagger_trot.pt / bc_trot.pt / valuedice_trot.pt がありません。先にメニュー 1 で模倣学習してください。"
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
            } elseif ($playPick -eq "4") {
                Write-Host "歩行の途中チェックポイントを等間隔に並べます。対象: $logDir （最新 iteration $script:activeIter）"
                $compareName = "PickAndCarry_$($script:algo)_walk_easy_compare.mp4"
                $video = Join-Path $out $compareName
                & $python -u scripts\compare_snapshots.py --log-dir $logDir --output $video --seconds 16 --panels 4 --curriculum walk --easy-walk @algoArgs @armArgs @terrainArgs
                $code = $LASTEXITCODE
                if ($code -eq 0) { Open-LatestPlayVideo $logDir $compareName }
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
        Write-Host "改版スライドで提出動画を作ります（PickAndCarry_学習条件_改版.pptx）。"
        $video = Join-Path $out "PickAndCarry_presentation.mp4"
        $pptx = Join-Path $out "PickAndCarry_学習条件_改版.pptx"
        if (-not (Test-Path $pptx)) {
            Write-Host "改版 PPTX が無いのでスライドを生成します…"
            & $python -u scripts\make_summary_slide.py
            if ($LASTEXITCODE -ne 0) { $code = $LASTEXITCODE; break }
        }
        & $python -u scripts\make_presentation.py --pptx $pptx --videos $out --output $video
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
        Write-Host "同じ段階について、PPO / GRPO / Dr.GRPO / Symmetric PPO の曲線を重ねます。"
        $script:algo = "ppo"
        $script:algoArgs = @("--algo", "ppo")
        Select-Curriculum -ForPlay
        $dirs = @()
        $labels = @()
        foreach ($name in @("ppo", "grpo", "dr_grpo", "sym_ppo")) {
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
    "9" {
        Write-Host ""
        Write-Host " DVC / MinIO（学習ホットパスでは使わない。区切りで共有用）"
        Write-Host "   1  MinIO を起動（Docker Desktop も必要なら起動）"
        Write-Host "   2  初回セットアップ（pip + DVC remote + MinIO 起動）"
        Write-Host "   3  最良チェックポイントを publish → push（metrics で選ぶ・推奨）"
        Write-Host "   4  最新 iteration を publish → push"
        Write-Host "   5  MinIO から pull → restore（段階を選ぶ）"
        Write-Host "   6  DVC の最良重みで推論（pull → resolve → play）"
        Write-Host "   7  状態確認（dvc status）"
        Write-Host "   8  MinIO Console を開く"
        Write-Host "   9  Mender Artifact を作る（DVC best.pt → .mender / OTA 用）"
        Write-Host "   手順書: docs\DVC_MINIO.md / docs\MENDER_OTA.md"
        $dvcPick = Read-Host "DVC / MinIO [3]"
        if ([string]::IsNullOrWhiteSpace($dvcPick)) { $dvcPick = "3" }

        function Invoke-LoadPaiEnv {
            . (Join-Path (Get-Location) "scripts\load_pai_env.ps1")
        }

        switch ($dvcPick) {
            "1" {
                & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path (Get-Location) "scripts\start_minio.ps1")
                $code = $LASTEXITCODE
            }
            "2" {
                & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path (Get-Location) "scripts\setup_dvc_minio.ps1")
                $code = $LASTEXITCODE
            }
            "3" {
                Invoke-LoadPaiEnv
                Select-Algo
                Select-Curriculum -ForPlay
                $logDir = $script:logDir -replace "/", "\"
                Write-Host "最良を publish → push: $logDir"
                & $python -u scripts\dvc_sync.py publish --log-dir $logDir --select best --push
                $code = $LASTEXITCODE
            }
            "4" {
                Invoke-LoadPaiEnv
                Select-Algo
                Select-Curriculum -ForPlay
                $logDir = $script:logDir -replace "/", "\"
                Write-Host "最新を publish → push: $logDir"
                & $python -u scripts\dvc_sync.py publish --log-dir $logDir --select latest --push
                $code = $LASTEXITCODE
            }
            "5" {
                Invoke-LoadPaiEnv
                Select-Algo
                Select-Curriculum -ForPlay
                $logDir = $script:logDir -replace "/", "\"
                Write-Host "pull → restore 先: $logDir"
                & $python -u scripts\dvc_sync.py pull
                $code = $LASTEXITCODE
                if ($code -eq 0) {
                    & $python -u scripts\dvc_sync.py restore --log-dir $logDir
                    $code = $LASTEXITCODE
                }
            }
            "6" {
                Invoke-LoadPaiEnv
                Select-Algo
                Select-Curriculum -ForPlay
                $stageName = Split-Path ($script:logDir -replace "/", "\") -Leaf
                Write-Host "DVC 最良で推論: artifacts\checkpoints\$stageName"
                & $python -u scripts\dvc_sync.py pull
                $code = $LASTEXITCODE
                if ($code -ne 0) { break }
                $resolved = & $python -u scripts\dvc_sync.py resolve --name $stageName 2>&1 | Select-Object -Last 1
                if (-not $resolved -or -not (Test-Path -LiteralPath "$resolved")) {
                    Write-Host "resolve に失敗しました。先にメニュー 9 → 3 で publish してください。"
                    $code = 1
                    break
                }
                Write-Host "使う重み: $resolved"
                $video = Join-Path $out ("PickAndCarry_dvc_" + $stageName + ".mp4")
                & $python -u -m pick_and_carry.train_ppo --play --resume "$resolved" --num-envs 1 --record $video --seconds 20 --log-dir ($script:logDir -replace "/", "\") @algoArgs @armArgs @terrainArgs @curriculumArgs
                $code = $LASTEXITCODE
                if ($code -eq 0 -and (Test-Path $video)) { Start-Process $video }
            }
            "7" {
                Invoke-LoadPaiEnv
                & $python -u scripts\dvc_sync.py status
                $code = $LASTEXITCODE
            }
            "8" {
                Invoke-LoadPaiEnv
                & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path (Get-Location) "scripts\start_minio.ps1") -OpenConsole
                $code = $LASTEXITCODE
            }
            "9" {
                Invoke-LoadPaiEnv
                Select-Algo
                Select-Curriculum -ForPlay
                $stageName = Split-Path ($script:logDir -replace "/", "\") -Leaf
                $deviceType = if ($env:PAI_MENDER_DEVICE_TYPE) { $env:PAI_MENDER_DEVICE_TYPE } else { "go2-z1" }
                Write-Host "Mender Artifact: stage=$stageName device_type=$deviceType"
                Write-Host "（先に 9→3 で best.pt を publish しておくこと）"
                & $python -u scripts\mender_pack_model.py --name $stageName --device-type $deviceType
                $code = $LASTEXITCODE
                if ($code -eq 0) {
                    Write-Host "出力先: artifacts\mender\  → Hosted Mender にアップロードして配信"
                    Write-Host "手順: docs\MENDER_OTA.md"
                }
            }
            default {
                Write-Host "不明な番号です: $dvcPick"
                $code = 1
            }
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
