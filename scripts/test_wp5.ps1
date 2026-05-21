# =============================================================================
#  WP5 - Reasoner & Router Integration  |  Test Driver
# -----------------------------------------------------------------------------
#  Runs four graduated checks against the AI Video Investigator cascade:
#    1. Reasoner smoke test         -> isolates Claude Haiku 4.5 + JSON contract
#    2. Forced escalation (e2e)     -> proves Reasoner is invoked on AMBIGUOUS
#    3. Skipped escalation (e2e)    -> proves Token Economics on the easy path
#    4. Failure-mode (no API key)   -> proves graceful pre-flight degradation
#
#  Key handling
#  ------------
#  ANTHROPIC_API_KEY is loaded by python-dotenv from .env at Python startup -
#  NOT by PowerShell. The pre-flight (Step 0) verifies the key is visible to
#  Python; individual tests do not re-check it.
#
#  Usage:
#    pwsh -File scripts/test_wp5.ps1            # run all
#    pwsh -File scripts/test_wp5.ps1 -Only 2    # run a single test by number
# =============================================================================

param(
    [int]$Only = 0   # 0 = run all; otherwise 1..4
)

$ErrorActionPreference = "Stop"

# Resolve repo root from this script's location so the driver works regardless
# of the current working directory (academic-grade reproducibility).
$RepoRoot   = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python     = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Main       = Join-Path $RepoRoot "main.py"
$Sample     = Join-Path $RepoRoot "sample_frame.jpg"
$DashCam    = Join-Path $RepoRoot "notebooks\Dash_Cam.mp4"
$DogChase   = Join-Path $RepoRoot "notebooks\Dog_Chase.mp4"
$EnvFile    = Join-Path $RepoRoot ".env"
$EnvBackup  = Join-Path $RepoRoot ".env.wp5test-backup"

Set-Location $RepoRoot

# -----------------------------------------------------------------------------
#  Result tracking - we want a final summary line for the WP5 defense report.
# -----------------------------------------------------------------------------
$Results = [System.Collections.Generic.List[object]]::new()

function Write-Banner($title) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host (" {0}" -f $title)          -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
}

function Add-Result($name, $passed, $note) {
    $Results.Add([pscustomobject]@{ Test = $name; Passed = $passed; Note = $note })
    $tag = if ($passed) { "[PASS]" } else { "[FAIL]" }
    $color = if ($passed) { "Green" } else { "Red" }
    Write-Host "$tag $name - $note" -ForegroundColor $color
}

function Assert-File($path, $purpose) {
    if (-not (Test-Path $path)) {
        throw "Required file missing for $purpose : $path"
    }
}

# -----------------------------------------------------------------------------
#  STEP 0 - Pre-flight: verify Python (via python-dotenv) can see the key.
#  We do this exactly once at startup instead of guarding every test, since
#  the key lives in .env and PowerShell cannot see it.
# -----------------------------------------------------------------------------
function Test-ApiKeyVisibility {
    Write-Banner "STEP 0 - Pre-flight: ANTHROPIC_API_KEY visibility check"

    if (-not (Test-Path $EnvFile)) {
        throw "No .env file found at $EnvFile - create one with ANTHROPIC_API_KEY=<key>"
    }

    # Ask Python whether load_dotenv() exposes a non-empty key. This is the
    # same code path main.py and ClaudeReasoner use, so a green result here
    # guarantees the key is reachable by the rest of the suite.
    $code = "from dotenv import load_dotenv; load_dotenv(); import os; k=os.getenv('ANTHROPIC_API_KEY',''); print('KEY_OK' if k.strip() else 'KEY_MISSING')"
    $out = & $Python -c $code 2>&1 | Out-String

    if ($out -match "KEY_OK") {
        Write-Host "[OK] python-dotenv exposes ANTHROPIC_API_KEY to Python" -ForegroundColor Green
    } else {
        throw "python-dotenv could not find ANTHROPIC_API_KEY in .env. Raw output: $out"
    }
}

# -----------------------------------------------------------------------------
#  TEST 1 - Reasoner smoke test (standalone, ~5s)
#  Verifies the JSON contract end-to-end without FAISS/Router noise.
#  Key is loaded by ClaudeReasoner's own load_dotenv() call - no PS guard.
# -----------------------------------------------------------------------------
function Test-1-ReasonerSmoke {
    Write-Banner "TEST 1 - Reasoner smoke test (standalone)"

    Assert-File $Sample "Test 1 sample frame"

    # Inline Python: instantiate ClaudeReasoner, send one image, dump verdict.
    # Using a here-string keeps the test self-contained (no test fixtures dir).
    $code = @'
import sys, json
sys.path.append("src")
from PIL import Image
from reasoner.claude_engine import ClaudeReasoner, ReasonerVerdict

reasoner = ClaudeReasoner()
verdict = reasoner.verify_event(Image.open("sample_frame.jpg"), "a dog chasing a person")
assert isinstance(verdict, ReasonerVerdict), "verdict has wrong type"
assert isinstance(verdict.event_detected, bool), "event_detected not bool"
assert 0.0 <= verdict.confidence_score <= 1.0, "confidence_score out of range"
assert isinstance(verdict.reasoning, str) and verdict.reasoning, "reasoning empty"

print(json.dumps({
    "event_detected":   verdict.event_detected,
    "confidence_score": verdict.confidence_score,
    "reasoning":        verdict.reasoning[:200],
    "is_verified":      verdict.is_verified,
}, indent=2))
'@

    $out = & $Python -c $code 2>&1 | Out-String
    Write-Host $out

    if ($LASTEXITCODE -eq 0 -and $out -match '"event_detected"') {
        Add-Result "T1-smoke" $true "JSON contract satisfied"
    } else {
        Add-Result "T1-smoke" $false "non-zero exit or malformed output"
    }
}

# -----------------------------------------------------------------------------
#  TEST 2 - Forced escalation (Dash_Cam ambiguous band)
#  Empirically calibrated: the top "two women" score on Dash_Cam is ~0.1914
#  (probed via tau_low=0.99,tau_high=1.0). Thresholds tau_low=0.18 / tau_high=0.25
#  guarantee at least one candidate lands in the AMBIGUOUS band (no
#  IMMEDIATE_MATCH possible since 0.1914 < 0.25), forcing the Reasoner to fire.
# -----------------------------------------------------------------------------
function Test-2-ForcedEscalation {
    Write-Banner "TEST 2 - Forced escalation (Dash_Cam ambiguous band)"

    Assert-File $DashCam "Test 2 video"

    $out = & $Python $Main `
        --video   $DashCam `
        --query   "two women" `
        --tau_low 0.18 `
        --tau_high 0.25 `
        --top_k   5  2>&1 | Out-String
    Write-Host $out

    $escalated = $out -match "Escalating \d+ candidates to Reasoner \(Claude Haiku 4\.5\)"
    $invoked   = $out -match "Reasoner Invoked:\s+True"

    if ($escalated -and $invoked) {
        Add-Result "T2-escalate" $true "Reasoner banner fired AND Reasoner Invoked=True"
    } else {
        Add-Result "T2-escalate" $false "escalation banner=$escalated  invoked=$invoked"
    }
}

# -----------------------------------------------------------------------------
#  TEST 3 - Skipped escalation (Dog_Chase happy path)
#  Per WP4 docs, "dog chase man" on Dog_Chase is a high-confidence baseline.
#  Low tau_high lets CLIP auto-accept all hits. Reasoner MUST NOT fire -
#  this is the literal proof of Token Economics on the easy path.
# -----------------------------------------------------------------------------
function Test-3-SkippedEscalation {
    Write-Banner "TEST 3 - Skipped escalation (Dog_Chase happy path)"

    Assert-File $DogChase "Test 3 video"

    $out = & $Python $Main `
        --video   $DogChase `
        --query   "dog chase man" `
        --tau_low 0.18 `
        --tau_high 0.22 `
        --top_k   10  2>&1 | Out-String
    Write-Host $out

    $noBanner   = -not ($out -match "Escalating \d+ candidates to Reasoner")
    $notInvoked = $out -match "Reasoner Invoked:\s+False"
    $zeroCost   = $out -match 'Estimated Cost:\s+\$0\.0000'

    if ($noBanner -and $notInvoked -and $zeroCost) {
        Add-Result "T3-skip" $true "Reasoner skipped, cost=`$0 (Token Economics OK)"
    } else {
        Add-Result "T3-skip" $false "noBanner=$noBanner notInvoked=$notInvoked zeroCost=$zeroCost"
    }
}

# -----------------------------------------------------------------------------
#  TEST 4 - Failure-mode (no key reachable from Python)
#  Confirms the pre-flight check in main.py degrades gracefully instead of
#  letting a Python traceback escape. Because the key lives in .env (not in
#  PS env), unsetting $env:ANTHROPIC_API_KEY is not enough - python-dotenv
#  would re-load the real key from .env. So we temporarily rename .env to
#  simulate "user forgot the key", then restore it via try/finally.
#
#  Safety: aborts immediately if a stale .env.wp5test-backup already exists
#  (would indicate a prior failed run; manual review required to avoid
#  silently overwriting the user's real .env).
# -----------------------------------------------------------------------------
function Test-4-MissingKey {
    Write-Banner "TEST 4 - Failure-mode (no key reachable from Python)"

    Assert-File $DashCam "Test 4 video"

    if (Test-Path $EnvBackup) {
        Add-Result "T4-no-key" $false "stale $EnvBackup exists - inspect and remove manually before re-running"
        return
    }

    $renamed = $false
    try {
        if (Test-Path $EnvFile) {
            Rename-Item $EnvFile $EnvBackup
            $renamed = $true
        }

        # Use the same empirically-calibrated thresholds as T2 so this run
        # is guaranteed to reach the `if ambiguous:` branch where the
        # missing-key error message lives in main.py.
        $out = & $Python $Main `
            --video   $DashCam `
            --query   "two women" `
            --tau_low 0.18 `
            --tau_high 0.25 `
            --top_k   5  2>&1 | Out-String
        Write-Host $out

        $cleanMsg    = $out -match "CRITICAL ERROR: ANTHROPIC_API_KEY not found"
        $noTraceback = -not ($out -match "Traceback \(most recent call last\)")

        if ($cleanMsg -and $noTraceback) {
            Add-Result "T4-no-key" $true "pre-flight error printed, no traceback"
        } else {
            Add-Result "T4-no-key" $false "cleanMsg=$cleanMsg noTraceback=$noTraceback"
        }
    } finally {
        # ALWAYS restore .env, even if the test or the Python call crashed.
        # This is the contract that makes the rename approach safe.
        if ($renamed -and (Test-Path $EnvBackup)) {
            Rename-Item $EnvBackup $EnvFile
        }
    }
}

# -----------------------------------------------------------------------------
#  Driver
# -----------------------------------------------------------------------------
Test-ApiKeyVisibility

$tests = @{
    1 = ${function:Test-1-ReasonerSmoke}
    2 = ${function:Test-2-ForcedEscalation}
    3 = ${function:Test-3-SkippedEscalation}
    4 = ${function:Test-4-MissingKey}
}

if ($Only -gt 0) {
    if (-not $tests.ContainsKey($Only)) { throw "Unknown test index: $Only (valid: 1..4)" }
    & $tests[$Only]
} else {
    foreach ($k in ($tests.Keys | Sort-Object)) { & $tests[$k] }
}

# -----------------------------------------------------------------------------
#  Final summary - the artifact you paste into the WP5 defense report.
# -----------------------------------------------------------------------------
Write-Banner "WP5 TEST SUMMARY"
$Results | Format-Table -AutoSize

$failed = ($Results | Where-Object { -not $_.Passed }).Count
if ($failed -gt 0) {
    Write-Host "$failed test(s) FAILED" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All tests PASSED" -ForegroundColor Green
}
