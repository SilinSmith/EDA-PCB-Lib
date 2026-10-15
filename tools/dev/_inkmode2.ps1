# -*- coding: utf-8 -*-
# _inkmode2.ps1 -- unconfounded re-run of the ink-criterion arm (user m42009 point 1).
# The first A/B used STAGE1_WIRE_INK=bglab, but that name is ALSO a boolean gate in
# stage1/labels.py:133 (env_on), so the arm silently disabled the weld ink rule too.
# New knob name: STAGE1_WIRE_INK_MODE. Arms here separate the two effects.
#    ikm_base     : nothing (sanity, must equal ab2_base 39.18)
#    ikm_mode     : STAGE1_WIRE_INK_MODE=bglab   (unconfounded ink criterion)
#    ikm_laboff   : STAGE1_WIRE_INK=0            (labels weld ink rule off alone)
# NOTE: pure ASCII only (Windows PowerShell 5.1 parses .ps1 as cp936).
$ErrorActionPreference = "Continue"
$root = "Y:\workspace\EDA"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:FUSION_LLM = "0"
$cases = (Get-Content "$root\work\_cases_w28.txt" -Raw).Trim()
Write-Host ("[ikm] cases = " + $cases + "  FUSION_LLM=" + $env:FUSION_LLM)

function Clean {
    foreach ($k in @("STAGE1_WIRE_INK", "STAGE1_WIRE_INK_MODE", "STAGE1_OCC_LOCAL",
                     "STAGE1_WIRE_BRIDGE", "STAGE1_WIRE_EDGE", "STAGE1_WIRE_COLOUR",
                     "STAGE1_WIRE_UNCONF", "STAGE1_OCC_MODE")) {
        Remove-Item ("Env:\" + $k) -ErrorAction SilentlyContinue
    }
}

function RunArm([string]$name, [string]$cmp) {
    Write-Host ("[ikm] arm " + $name + " start " + (Get-Date -Format "HH:mm:ss"))
    $t0 = Get-Date
    $a = @("-X", "utf8", "eda-pcb\tools\eval\fast.py", "--cases", $cases, "--no-vlm",
           "--out", $name, "--save", $name)
    if ($cmp -ne "") { $a += @("--cmp", $cmp) }
    & python @a 2>&1 | Tee-Object -FilePath "$root\work\_ikm_$name.log"
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host ("[ikm] arm " + $name + " done in " + $dt + "s")
}

Clean
RunArm "ikm_base" "ab2_base"

Clean; $env:STAGE1_WIRE_INK_MODE = "bglab"
RunArm "ikm_mode" "ab2_base"

Clean; $env:STAGE1_WIRE_INK = "0"
RunArm "ikm_laboff" "ab2_base"

Clean
Write-Host "[ikm] all done"
