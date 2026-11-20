# Task runner for Windows PowerShell (replaces a Makefile).
# Usage:  .\run.ps1 setup | inventory | run | notebooks | test | lint | format | clean
param([Parameter(Position = 0)][string]$Task = "help")

$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Invoke-Py {
    # Run the venv Python and judge it by its exit code only. In Windows PowerShell 5.1, text
    # written to stderr (e.g. a harmless RuntimeWarning) becomes an error record, which would
    # abort the task under $ErrorActionPreference = "Stop".
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Py @args 2>&1 | ForEach-Object { "$_" } }
    finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { throw "python $($args -join ' ') failed (exit code $LASTEXITCODE)" }
}

function Invoke-Setup {
    if (-not (Test-Path $Py)) { py -3.11 -m venv (Join-Path $PSScriptRoot ".venv") }
    Invoke-Py -m pip install --upgrade pip
    Invoke-Py -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
    Invoke-Py -m pip install -e $PSScriptRoot
}

function Invoke-Inventory {
    Invoke-Py (Join-Path $PSScriptRoot "scripts\00_inventory_raw.py") --raw-dir (Join-Path $PSScriptRoot "data\raw") --out (Join-Path $PSScriptRoot "docs\data_inventory.md")
}

function Invoke-Run {
    # Runs every numbered script in order: each one regenerates its figures / tables.
    # -Filter does not understand [0-9] ranges (it silently matches nothing): use -like instead.
    $Scripts = Get-ChildItem (Join-Path $PSScriptRoot "scripts") -File |
        Where-Object { $_.Name -like "[0-9][0-9]_*.py" } | Sort-Object Name
    if (-not $Scripts) { throw "No numbered scripts found in scripts/" }
    foreach ($Script in $Scripts) {
        Write-Host "==> $($Script.Name)"
        Invoke-Py $Script.FullName
    }
}

switch ($Task) {
    "setup"     { Invoke-Setup }
    "inventory" { Invoke-Inventory }
    "run"       { Invoke-Run }
    "notebooks" { Invoke-Py (Join-Path $PSScriptRoot "scripts\run_notebooks.py") }
    "test"      { Invoke-Py -m pytest }
    "lint"      { Invoke-Py -m ruff check src tests scripts; Invoke-Py -m ruff format --check src tests scripts }
    "format"    { Invoke-Py -m ruff format src tests scripts }
    "clean"     {
        Get-ChildItem $PSScriptRoot -Recurse -Directory -Include "__pycache__", ".pytest_cache", ".ruff_cache" |
            Where-Object { $_.FullName -notlike "*\.venv\*" } | Remove-Item -Recurse -Force
    }
    default     { Write-Host "Usage: .\run.ps1 setup | inventory | run | notebooks | test | lint | format | clean" }
}
