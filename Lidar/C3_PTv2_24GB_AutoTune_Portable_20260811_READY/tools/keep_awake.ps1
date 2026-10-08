param(
    [Parameter(Mandatory = $true)]
    [int]$OwnerPid,
    [Parameter(Mandatory = $true)]
    [string]$Root,
    [ValidateSet("setup", "training")]
    [string]$Mode = "training"
)

$ErrorActionPreference = "Stop"
$StatePath = Join-Path $Root ("output\keep_awake_{0}.json" -f $Mode)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StatePath) | Out-Null

if (-not ("PortableC3.PowerState" -as [type])) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
namespace PortableC3 {
    public static class PowerState {
        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern uint SetThreadExecutionState(uint esFlags);
    }
}
"@
}

$Continuous = [Convert]::ToUInt32("80000000", 16)
$SystemRequired = [uint32]0x00000001
$DisplayRequired = [uint32]0x00000002
$Required = $Continuous -bor $SystemRequired -bor $DisplayRequired

function Write-State {
    param([string]$Status, [string]$Message)
    @{
        status = $Status
        mode = $Mode
        owner_pid = $OwnerPid
        guard_pid = $PID
        updated_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
        display_required = $true
        system_required = $true
        message = $Message
    } | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

try {
    Write-State -Status "active" -Message "Display and system sleep are blocked while the owner process is active."
    while (Get-Process -Id $OwnerPid -ErrorAction SilentlyContinue) {
        $Result = [PortableC3.PowerState]::SetThreadExecutionState($Required)
        if ($Result -eq 0) {
            throw "SetThreadExecutionState failed."
        }
        Start-Sleep -Seconds 20
    }
} finally {
    [void][PortableC3.PowerState]::SetThreadExecutionState($Continuous)
    Write-State -Status "inactive" -Message "Owner process ended; normal Windows power policy was restored."
}
