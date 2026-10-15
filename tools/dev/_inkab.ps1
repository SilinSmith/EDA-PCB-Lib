# -*- coding: utf-8 -*-
# _inkab.ps1 -- background-transition ink A/B + box-bridge A/B (user m42009)
# NOTE: keep this file pure ASCII (Windows PowerShell 5.1 reads .ps1 as ANSI/cp936).
$ErrorActionPreference = "Continue"
$root = "Y:\workspace\EDA"
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$cases = (Get-Content "$root\work\_cases_w28.txt" -Raw).Trim()
Write-Host ("[inkab] cases = " + $cases)

function Clean([string[]]$keep) {
    foreach ($k in @("STAGE1_WIRE_INK", "STAGE1_OCC_LOCAL", "STAGE1_WIRE_BRIDGE",
                     "STAGE1_WIRE_EDGE", "STAGE1_WIRE_COLOUR", "STAGE1_WIRE_UNCONF")) {
        if ($keep -notcontains $k) { Remove-Item ("Env:\" + $k) -ErrorAction SilentlyContinue }
    }
}

function RunArm([string]$name, [string]$cmp) {
    Write-Host ("[inkab] arm " + $name + " start " + (Get-Date -Format "HH:mm:ss"))
    $t0 = Get-Date
    $args = @("-X", "utf8", "eda-pcb\tools\eval\fast.py", "--cases", $cases, "--no-vlm",
              "--out", $name, "--save", $name)
    if ($cmp -ne "") { $args += @("--cmp", $cmp) }
    & python @args 2>&1 | Tee-Object -FilePath "$root\work\_inkab_$name.log"
    $dt = [int]((Get-Date) - $t0).TotalSeconds
    Write-Host ("[inkab] arm " + $name + " done in " + $dt + "s at " + (Get-Date -Format "HH:mm:ss"))
}

# arm 1: baseline (delivery defaults: adaptive grayscale ink, no bridge)
Clean @()
RunArm "ink_base" ""

# arm 2: background-distance (Lab) ink mask instead of grayscale adaptive
Clean @("STAGE1_WIRE_INK")
$env:STAGE1_WIRE_INK = "bglab"
RunArm "ink_lab" "ink_base"

# arm 3: + local paper-colour fill of occluded boxes (no constant-colour seams)
Clean @("STAGE1_WIRE_INK", "STAGE1_OCC_LOCAL")
$env:STAGE1_WIRE_INK = "bglab"
$env:STAGE1_OCC_LOCAL = "1"
RunArm "ink_lab_loc" "ink_base"

# arm 4: box-bridge only (cross-validated in-box wires, base ink)
Clean @("STAGE1_WIRE_BRIDGE")
$env:STAGE1_WIRE_BRIDGE = "1"
RunArm "ink_bridge" "ink_base"

# arm 5: bridge + background-distance ink
Clean @("STAGE1_WIRE_INK", "STAGE1_WIRE_BRIDGE")
$env:STAGE1_WIRE_INK = "bglab"
$env:STAGE1_WIRE_BRIDGE = "1"
RunArm "ink_lab_bridge" "ink_base"

Write-Host "[inkab] all done"
