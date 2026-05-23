# start.ps1 - Start the AI Video Investigator backend + frontend.
#
# V2.0: Backend uses build_retriever() which selects OpenCLIP
# (ViT-L-14 / laion2b_s32b_b82k) by default. Override per session with
# $env:RETRIEVER_BACKEND = "openai_hf" before invoking this script for
# V1 rollback. Each service runs in its own PowerShell window so logs
# are separated and Ctrl+C stops one without affecting the other.
#
# Usage:
#   .\start.ps1                    # defaults: backend 8000, frontend 4200, open browser
#   .\start.ps1 -NoBrowser         # don't auto-open the frontend
#   .\start.ps1 -BackendPort 8080  # custom ports

[CmdletBinding()]
param(
    [int]    $BackendPort  = 8000,
    [int]    $FrontendPort = 4200,
    [switch] $NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

# --- Pre-flight checks ---
if (-not (Test-Path (Join-Path $Root "venv\Scripts\Activate.ps1"))) {
    Write-Error "Python venv not found at $Root\venv. Create one with: python -m venv venv"
    exit 1
}
if (-not (Test-Path (Join-Path $Root "frontend\package.json"))) {
    Write-Error "Frontend not found at $Root\frontend"
    exit 1
}
if (-not (Test-Path (Join-Path $Root "src\server.py"))) {
    Write-Error "Backend entry point not found at $Root\src\server.py"
    exit 1
}

# Prefer PowerShell 7+ (pwsh), fall back to Windows PowerShell 5.1.
$PwshExe = if (Get-Command pwsh -ErrorAction SilentlyContinue) {
    (Get-Command pwsh).Source
} else {
    (Get-Command powershell).Source
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " AI Video Investigator - V2.0 (OpenCLIP / ViT-L-14 / LAION-2B)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# --- Backend window ---
$BackendCmd = @"
Set-Location '$Root'
.\venv\Scripts\Activate.ps1
`$Host.UI.RawUI.WindowTitle = 'AIVI Backend (FastAPI / V2 OpenCLIP)'
Write-Host '[backend] starting uvicorn src.server:app on http://127.0.0.1:$BackendPort' -ForegroundColor Green
uvicorn src.server:app --host 127.0.0.1 --port $BackendPort
"@
Write-Host "[1/2] Backend  -> http://127.0.0.1:$BackendPort" -ForegroundColor Yellow
Start-Process -FilePath $PwshExe `
    -ArgumentList @("-NoExit", "-Command", $BackendCmd) `
    -WindowStyle Normal

# --- Frontend window ---
$FrontendCmd = @"
Set-Location '$Root\frontend'
`$Host.UI.RawUI.WindowTitle = 'AIVI Frontend (Angular / WP8 GUI)'
Write-Host '[frontend] starting ng serve on http://127.0.0.1:$FrontendPort' -ForegroundColor Green
ng serve --host 127.0.0.1 --port $FrontendPort
"@
Write-Host "[2/2] Frontend -> http://127.0.0.1:$FrontendPort" -ForegroundColor Yellow
Start-Process -FilePath $PwshExe `
    -ArgumentList @("-NoExit", "-Command", $FrontendCmd) `
    -WindowStyle Normal

# --- Tail message ---
Write-Host ""
Write-Host "Both services are starting in separate windows." -ForegroundColor Green
Write-Host ""
Write-Host "  Backend     http://127.0.0.1:$BackendPort"        -ForegroundColor White
Write-Host "  Frontend    http://127.0.0.1:$FrontendPort"       -ForegroundColor White
Write-Host "  API docs    http://127.0.0.1:$BackendPort/docs"   -ForegroundColor White
Write-Host ""
Write-Host "First /api/investigate call loads the V2 model into VRAM (~13s, one-time)." -ForegroundColor DarkGray
Write-Host "Subsequent queries: text-encode ~20ms + FAISS <1ms (+ Claude verification)." -ForegroundColor DarkGray
Write-Host "Stop a service: Ctrl+C in its window, or close the window."                  -ForegroundColor DarkGray
Write-Host ""

# --- Optional browser auto-open ---
if (-not $NoBrowser) {
    Write-Host "Opening browser in 8 seconds (Angular needs ~7s to compile)..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 8
    Start-Process "http://127.0.0.1:$FrontendPort"
}
