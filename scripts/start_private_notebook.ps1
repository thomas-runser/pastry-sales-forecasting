$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PASTRY_DATA_MODE = "private"
Write-Host "Starting Jupyter with confidential local data (not GitHub data)."
jupyter lab
