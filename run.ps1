# Task runner for Windows PowerShell (replaces a Makefile).
# Usage:  .\run.ps1 setup | inventory | run | test | lint | clean
param([Parameter(Position = 0)][string]$Task = "help")

$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Invoke-Setup {
    if (-not (Test-Path $Py)) { py -3.11 -m venv (Join-Path $PSScriptRoot ".venv") }
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
    & $Py -m pip install -e $PSScriptRoot
}

function Invoke-Inventory {
    & $Py (Join-Path $PSScriptRoot "scripts\00_inventory_raw.py") --raw-dir (Join-Path $PSScriptRoot "data\raw") --out (Join-Path $PSScriptRoot "docs\data_inventory.md")
}

function Invoke-Run {
    # Runs every numbered script in order: each one regenerates its figures / tables.
    Get-ChildItem (Join-Path $PSScriptRoot "scripts") -Filter "[0-9][0-9]_*.py" | Sort-Object Name | ForEach-Object {
        Write-Host "==> $($_.Name)"
        & $Py $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" }
    }
}

switch ($Task) {
    "setup"     { Invoke-Setup }
    "inventory" { Invoke-Inventory }
    "run"       { Invoke-Run }
    "test"      { & $Py -m pytest }
    "lint"      { & $Py -m ruff check src tests scripts }
    "clean"     {
        Get-ChildItem $PSScriptRoot -Recurse -Directory -Include "__pycache__", ".pytest_cache", ".ruff_cache" |
            Where-Object { $_.FullName -notlike "*\.venv\*" } | Remove-Item -Recurse -Force
    }
    default     { Write-Host "Usage: .\run.ps1 setup | inventory | run | test | lint | clean" }
}
