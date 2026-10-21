chcp.com 65001 > $null
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# launcher/ sits next to the project root in the submission tree.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $ProjectRoot "pick_and_carry"))) {
    $ProjectRoot = $PSScriptRoot
}
Set-Location $ProjectRoot
$Host.UI.RawUI.WindowTitle = "Pick-and-Carry"
$out = Join-Path $ProjectRoot "output"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host ".venv がありません。README の手順で環境を作ってから再実行してください。"
    Read-Host "キーを押すと閉じます"
    exit 1
}

Write-Host ""
Write-Host " Pick-and-Carry（提出用）"
Write-Host " プロジェクト: $(Get-Location)"
Write-Host " 出力: $out"
Write-Host ""
Write-Host "   1  模倣学習（トロット教師 → BC）"
Write-Host "   2  段階学習（歩行 / 到達 / 持ち運び / 箱運び / キャッチ）"
Write-Host "   3  推論（保存した重みを再生）"
Write-Host "   4  箱運びの途中経過を並べた比較動画"
Write-Host "   5  学習曲線を開く"
Write-Host "   6  プレゼン用の動画を作る"
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

function Ask-Int {
    param([string]$Prompt, [int]$Default)
    $raw = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }
    $parsed = 0
    if ([int]::TryParse($raw, [ref]$parsed) -and $parsed -gt 0) { return $parsed }
    Write-Host "正の整数を入れてください。既定の $Default を使います。"
    return $Default
}

function Get-BcPath {
    $candidates = @(
        "logs\imitation\bc_trot.pt",
        "logs\ppo_curriculum_walk_easy\bc_trot.pt",
        "logs\ppo_curriculum_walk\bc_trot.pt"
    )
    foreach ($rel in $candidates) {
        $p = Join-Path (Get-Location) $rel
        if (Test-Path $p) { return $p }
    }
    return $null
}

$choice = Read-Host "番号"
switch ($choice) {
    "1" {
        $logDir = "logs\imitation"
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        $seconds = Ask-Int "教師デモ秒数" 90
        $epochs = Ask-Int "BC epochs" 3000
        $envs = Ask-Int "並列環境数" 32
        Write-Host "トロット教師 → BC を実行します…"
        & $python -u -m pick_and_carry.train_bc --easy-walk --num-envs $envs --seconds $seconds --epochs $epochs --log-dir $logDir --arm z1
        Write-Host "完了。bc_trot.pt は $logDir にあります。"
    }
    "2" {
        Write-Host "段階: 1 歩行(簡単)  2 到達  3 持ち運び  4 箱運び  5 キャッチ"
        $stage = Read-Host "段階 [1]"
        if ([string]::IsNullOrWhiteSpace($stage)) { $stage = "1" }
        $iters = Ask-Int "学習回数" 1000
        $envs = Ask-Int "並列環境数" 64
        $map = @{
            "1" = @("walk", "logs\ppo_curriculum_walk_easy", $true)
            "2" = @("reach", "logs\ppo_curriculum_reach", $false)
            "3" = @("haul", "logs\ppo_curriculum_haul", $false)
            "4" = @("carry", "logs\ppo_z1", $false)
            "5" = @("catch", "logs\ppo_curriculum_catch", $false)
        }
        if (-not $map.ContainsKey($stage)) {
            Write-Host "未知の段階です。"
            break
        }
        $cur = $map[$stage][0]
        $logDir = $map[$stage][1]
        $easy = $map[$stage][2]
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        $args = @(
            "-u", "-m", "pick_and_carry.train_ppo",
            "--num-envs", "$envs", "--iterations", "$iters",
            "--num-steps", "128", "--save-interval", "20",
            "--log-dir", $logDir, "--curriculum", $cur, "--arm", "z1",
            "--resume", "latest"
        )
        if ($easy) {
            $args += "--easy-walk"
            $bc = Get-BcPath
            if ($null -ne $bc) {
                Write-Host "BC 重みから初期化: $bc"
                $args += @("--init-from", $bc)
            }
        }
        Write-Host ("起動: python " + ($args -join " "))
        & $python @args
    }
    "3" {
        Write-Host "段階: 1 歩行(簡単)  2 到達  3 持ち運び  4 箱運び  5 キャッチ"
        $stage = Read-Host "段階 [4]"
        if ([string]::IsNullOrWhiteSpace($stage)) { $stage = "4" }
        $map = @{
            "1" = @("walk", "logs\ppo_curriculum_walk_easy", "PickAndCarry_walk.mp4", $true)
            "2" = @("reach", "logs\ppo_curriculum_reach", "PickAndCarry_reach.mp4", $false)
            "3" = @("haul", "logs\ppo_curriculum_haul", "PickAndCarry_haul.mp4", $false)
            "4" = @("carry", "logs\ppo_z1", "PickAndCarry_z1.mp4", $false)
            "5" = @("catch", "logs\ppo_curriculum_catch", "PickAndCarry_catch.mp4", $false)
        }
        if (-not $map.ContainsKey($stage)) {
            Write-Host "未知の段階です。"
            break
        }
        $cur = $map[$stage][0]
        $logDir = $map[$stage][1]
        $name = $map[$stage][2]
        $easy = $map[$stage][3]
        $record = Join-Path $out $name
        $seconds = Ask-Int "録画秒数" 20
        $play = @(
            "-u", "-m", "pick_and_carry.train_ppo",
            "--play", "--resume", "latest", "--num-envs", "1",
            "--seconds", "$seconds", "--record", $record,
            "--log-dir", $logDir, "--curriculum", $cur, "--arm", "z1"
        )
        if ($easy) { $play += "--easy-walk" }
        & $python @play
        if (Test-Path $record) { Start-Process $record }
    }
    "4" {
        $logDir = "logs\ppo_z1"
        $record = Join-Path $out "PickAndCarry_z1_compare.mp4"
        & $python -u scripts\compare_snapshots.py --log-dir $logDir --seconds 16 --panels 4 --output $record --arm z1
        if (Test-Path $record) { Start-Process $record }
    }
    "5" {
        $html = Join-Path $out "PickAndCarry_curves.html"
        & $python -u scripts\plot_curves.py --log-dir logs\ppo_z1 --output $html
        if (Test-Path $html) { Start-Process $html }
    }
    "6" {
        $pptx = Join-Path $out "PickAndCarry_学習条件.pptx"
        if (-not (Test-Path $pptx)) {
            Write-Host "スライドを先に生成します…"
            & $python -u scripts\make_summary_slide.py
        }
        & $python -u scripts\make_presentation.py --pptx $pptx --videos $out --output (Join-Path $out "PickAndCarry_presentation.mp4")
    }
    default {
        Write-Host "1〜6 を選んでください。"
    }
}

Write-Host ""
Read-Host "キーを押すと閉じます"
