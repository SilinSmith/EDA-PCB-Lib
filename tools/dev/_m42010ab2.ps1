# -*- coding: utf-8 -*-
# _m42010ab2.ps1 -- user m42009 end-to-end A/B, ONE deterministic rubric.
# arms: base | ink_lab | ink_lab_loc | bridge | occ_keep | occ_big
# FUSION_LLM=0 so the fusion LLM round cannot inject noise into the comparison.
# NOTE: pure ASCII only (Windows PowerShell 5.1 parses .ps1 as cp936).
$ErrorActionPreference = "Continue"
$root = "Y:\workspace\EDA"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:FUSION_LLM = "0"
$cases = (Get-Content "$root\work\_cases_w28.txt" -Raw).Trim()
Write-Host ("[ab2] cases = " + $cases + "  FUSION_LLM=" + $env:FUSION_LLM)

function Clean {
    foreach ($k in @("STAGE1_WIRE_INK", "STAGE1_OCC_LOCAL", "STAGE1_WIRE_BRIDGE",
                     "STAGE1_WIRE_EDGE", "STAGE1_WIRE_COLOUR", "STAGE1_WIRE_UNCONF",
                     "STAGE1_OCC_MODE")) {
        Remove-Item ("Env:\" + $k) -ErrorAction SilentlyContinue
    }
}

function RunArm([string]$name, [string]$cmp) {
    Write-Host ("[ab2] arm " + $name + " start " + (Get-Date -Format "HH:mm:ss"))
    $t0 = Get-Date
    $a = @("-X", "utf8", "eda-pcb\tools\eval\fast.py", "--cases", $cases, "--no-vlm",
           "--out", $name, "--save", $name)
    if ($cmp -ne "") { $a += @("--cmp", $cmp) }
    & python @a 2>&1 | Tee-Object -FilePath "$root\work\_ab2_$name.log"
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host ("[ab2] arm " + $name + " done in " + $dt + "s")
}

Clean
RunArm "ab2_base" ""

Clean; $env:STAGE1_WIRE_INK = "bglab"
RunArm "ab2_inklab" "ab2_base"

Clean; $env:STAGE1_WIRE_INK = "bglab"; $env:STAGE1_OCC_LOCAL = "1"
RunArm "ab2_inkloc" "ab2_base"

Clean; $env:STAGE1_WIRE_BRIDGE = "1"
RunArm "ab2_bridge" "ab2_base"

Clean; $env:STAGE1_OCC_MODE = "keep"
RunArm "ab2_occkeep" "ab2_base"

Clean; $env:STAGE1_OCC_MODE = "big"
RunArm "ab2_occbig" "ab2_base"

Clean
Write-Host "[ab2] all done"
