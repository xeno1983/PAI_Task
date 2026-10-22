# Start local MinIO (and Docker Desktop if needed).
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_minio.ps1

param(
    [switch]$OpenConsole
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Test-DockerReady {
    docker info 1>$null 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Wait-DockerReady {
    param([int]$TimeoutSec = 120)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerReady) { return $true }
        Start-Sleep -Seconds 3
    }
    return $false
}

if (-not (Test-Path -LiteralPath (Join-Path $Root ".env"))) {
    Copy-Item -LiteralPath (Join-Path $Root ".env.example") -Destination (Join-Path $Root ".env")
    Write-Host "Created .env from .env.example"
}

. (Join-Path $PSScriptRoot "load_pai_env.ps1")

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker コマンドがありません。Docker Desktop をインストールしてください。"
}

if (-not (Test-DockerReady)) {
    $dockerExe = Join-Path ${env:ProgramFiles} "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $dockerExe)) {
        throw "Docker Desktop が見つかりません: $dockerExe"
    }
    Write-Host "Docker Desktop を起動しています…"
    Start-Process -FilePath $dockerExe | Out-Null
    if (-not (Wait-DockerReady -TimeoutSec 150)) {
        throw "Docker Desktop の起動がタイムアウトしました。手動で起動してから再実行してください。"
    }
    Write-Host "Docker 準備完了。"
}

$composeFile = Join-Path $Root "infra\minio\docker-compose.yml"
Write-Host "MinIO を起動しています…"
docker compose -f $composeFile --env-file ".env" up -d
if ($LASTEXITCODE -ne 0) {
    throw "MinIO の起動に失敗しました (exit $LASTEXITCODE)。"
}

$endpoint = if ($env:PAI_MINIO_ENDPOINT) { $env:PAI_MINIO_ENDPOINT } else { "http://127.0.0.1:9000" }
$console = $endpoint -replace ':9000$', ':9001'
if ($console -eq $endpoint) { $console = "http://127.0.0.1:9001" }

Write-Host ""
Write-Host "MinIO API:     $endpoint"
Write-Host "MinIO Console: $console"
Write-Host "DVC bucket:    $($env:PAI_MINIO_DVC_BUCKET)"
Write-Host "MLflow bucket: $($env:PAI_MINIO_MLFLOW_BUCKET)"

if ($OpenConsole) {
    Start-Process $console
}
