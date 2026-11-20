# Task runner for Windows PowerShell (replaces a Makefile).
# Usage:  .\run.ps1 setup | inventory | run | notebooks | test | lint | format | clean
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
    # -Filter does not understand [0-9] ranges (it silently matches nothing): use -like instead.
    $Scripts = Get-ChildItem (Join-Path $PSScriptRoot "scripts") -File |
        Where-Object { $_.Name -like "[0-9][0-9]_*.py" } | Sort-Object Name
    if (-not $Scripts) { throw "No numbered scripts found in scripts/" }
    $Scripts | ForEach-Object {
        Write-Host "==> $($_.Name)"
        & $Py $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "$($_.Name) failed" }
    }
}

switch ($Task) {
    "setup"     { Invoke-Setup }
    "inventory" { Invoke-Inventory }
    "run"       { Invoke-Run }
    "notebooks" { & $Py (Join-Path $PSScriptRoot "scripts\run_notebooks.py") }
    "test"      { & $Py -m pytest }
    "lint"      { & $Py -m ruff check src tests scripts; & $Py -m ruff format --check src tests scripts }
    "format"    { & $Py -m ruff format src tests scripts }
    "clean"     {
        Get-ChildItem $PSScriptRoot -Recurse -Directory -Include "__pycache__", ".pytest_cache", ".ruff_cache" |
            Where-Object { $_.FullName -notlike "*\.venv\*" } | Remove-Item -Recurse -Force
    }
    default     { Write-Host "Usage: .\run.ps1 setup | inventory | run | notebooks | test | lint | format | clean" }
}
