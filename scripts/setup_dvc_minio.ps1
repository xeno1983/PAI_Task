# One-shot: start MinIO, install DVC extras, init remote if needed.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_dvc_minio.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env from .env.example"
}

. (Join-Path $PSScriptRoot "load_pai_env.ps1")

Write-Host "Starting MinIO (requires Docker Desktop running)..."
$composeFile = Join-Path $Root "infra\minio\docker-compose.yml"
docker compose -f $composeFile --env-file ".env" up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: MinIO did not start. Start Docker Desktop and re-run this script."
    $global:LASTEXITCODE = 0
}

$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw ".venv is missing. Create the venv from README first."
}

& $python -m pip install -r requirements-dvc.txt

$dvc = Join-Path $Root ".venv\Scripts\dvc.exe"
$config = Join-Path $Root ".dvc\config"
if (-not (Test-Path -LiteralPath $config)) {
    & $dvc init
}

$bucket = if ($env:PAI_MINIO_DVC_BUCKET) { $env:PAI_MINIO_DVC_BUCKET } else { "pai-dvc" }
$endpoint = if ($env:PAI_MINIO_ENDPOINT) { $env:PAI_MINIO_ENDPOINT } else { "http://127.0.0.1:9000" }

& $dvc remote remove minio 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { $global:LASTEXITCODE = 0 }
$remotes = & $dvc remote list
if ($remotes -match "(?m)^minio\b") {
    & $dvc remote modify minio url ("s3://" + $bucket)
} else {
    & $dvc remote add -d minio ("s3://" + $bucket)
}
& $dvc remote modify minio endpointurl $endpoint
& $dvc remote default minio

New-Item -ItemType Directory -Force -Path (Join-Path $Root "artifacts\checkpoints") | Out-Null
Write-Host ""
Write-Host "Ready. Example:"
Write-Host '  . .\scripts\load_pai_env.ps1'
Write-Host '  .\.venv\Scripts\python.exe scripts\dvc_sync.py publish --log-dir logs\ppo_z1'
Write-Host '  .\.venv\Scripts\python.exe scripts\dvc_sync.py push'
Write-Host 'Docs: docs\DVC_MINIO.md'
