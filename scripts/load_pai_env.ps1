# Load .env into this PowerShell session (and child processes).
#   . .\scripts\load_pai_env.ps1

$Root = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $Root ".env"
if (-not (Test-Path $EnvFile)) {
    $EnvFile = Join-Path $Root ".env.example"
    Write-Host "Using .env.example (copy to .env to customize)."
}

Get-Content -LiteralPath $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) { return }
    $parts = $line.Split("=", 2)
    if ($parts.Count -ne 2) { return }
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    Set-Item -Path "Env:$name" -Value $value
}

# Aliases DVC / boto3 expect when only PAI_* is set.
if (-not $env:AWS_ACCESS_KEY_ID -and $env:PAI_MINIO_ACCESS_KEY) {
    $env:AWS_ACCESS_KEY_ID = $env:PAI_MINIO_ACCESS_KEY
}
if (-not $env:AWS_SECRET_ACCESS_KEY -and $env:PAI_MINIO_SECRET_KEY) {
    $env:AWS_SECRET_ACCESS_KEY = $env:PAI_MINIO_SECRET_KEY
}
if (-not $env:MLFLOW_S3_ENDPOINT_URL -and $env:PAI_MINIO_ENDPOINT) {
    $env:MLFLOW_S3_ENDPOINT_URL = $env:PAI_MINIO_ENDPOINT
}
if (-not $env:AWS_DEFAULT_REGION) {
    $env:AWS_DEFAULT_REGION = "us-east-1"
}

Write-Host "MinIO endpoint: $($env:PAI_MINIO_ENDPOINT)"
Write-Host "DVC bucket:     $($env:PAI_MINIO_DVC_BUCKET)"
Write-Host "MLflow bucket:  $($env:PAI_MINIO_MLFLOW_BUCKET)"
