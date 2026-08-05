$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
Remove-Item Env:PASTRY_DATA_MODE -ErrorAction SilentlyContinue
Write-Host "Starting Jupyter with public synthetic data."
jupyter lab
