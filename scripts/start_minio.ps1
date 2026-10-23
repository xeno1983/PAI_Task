# Start local MinIO (and Docker Desktop if needed).
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_minio.ps1

param(
    [switch]$OpenConsole
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Test-DockerReady {
    # Native docker stderr becomes terminating under ErrorAction Stop.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        docker info 1>$null 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $prev
    }
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

if (-not (Test-Path -LiteralPath (Join-Path $Root '.env'))) {
    Copy-Item -LiteralPath (Join-Path $Root '.env.example') -Destination (Join-Path $Root '.env')
    Write-Host 'Created .env from .env.example'
}

. (Join-Path $PSScriptRoot 'load_pai_env.ps1')

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'docker command not found. Install Docker Desktop first.'
}

if (-not (Test-DockerReady)) {
    $programFiles = [Environment]::GetFolderPath('ProgramFiles')
    $dockerExe = Join-Path $programFiles 'Docker\Docker\Docker Desktop.exe'
    if (-not (Test-Path -LiteralPath $dockerExe)) {
        throw "Docker Desktop not found: $dockerExe"
    }
    Write-Host 'Starting Docker Desktop...'
    Start-Process -FilePath $dockerExe | Out-Null
    if (-not (Wait-DockerReady -TimeoutSec 150)) {
        throw 'Docker Desktop startup timed out. Start it manually and retry.'
    }
    Write-Host 'Docker is ready.'
}

$composeFile = Join-Path $Root 'infra\minio\docker-compose.yml'
Write-Host 'Starting MinIO...'
# docker compose prints progress on stderr; keep Stop from aborting mid-pull.
$prev = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    docker compose -f $composeFile --env-file '.env' up -d
    $composeExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $prev
}
if ($composeExit -ne 0) {
    throw "MinIO failed to start (exit $composeExit). If images are missing, Docker Hub no longer hosts minio/*; use a local cache or quay.io login."
}

$endpoint = if ($env:PAI_MINIO_ENDPOINT) { $env:PAI_MINIO_ENDPOINT } else { 'http://127.0.0.1:9000' }
$console = $endpoint -replace ':9000$', ':9001'
if ($console -eq $endpoint) { $console = 'http://127.0.0.1:9001' }

Write-Host ''
Write-Host "MinIO API:     $endpoint"
Write-Host "MinIO Console: $console"
Write-Host "DVC bucket:    $($env:PAI_MINIO_DVC_BUCKET)"
Write-Host "MLflow bucket: $($env:PAI_MINIO_MLFLOW_BUCKET)"

if ($OpenConsole) {
    Start-Process $console
}
