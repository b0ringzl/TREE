param(
    [Parameter(Mandatory = $true)]
    [string]$Root,
    [Parameter(Mandatory = $true)]
    [int]$TargetEpoch,
    [Parameter(Mandatory = $true)]
    [int]$TrainerPid,
    [Parameter(Mandatory = $true)]
    [int]$WrapperPid
)

$ErrorActionPreference = "Stop"
$LauncherPath = Join-Path $Root "output\c3_launcher_state.json"
$RequestPath = Join-Path $Root "output\pause_after_epoch.json"
$LogPath = Join-Path $Root "output\pause_after_epoch.log"
$Launcher = Get-Content -LiteralPath $LauncherPath -Raw | ConvertFrom-Json
$Seed = [int]$Launcher.active_seed
$Run = Join-Path $Root ("output\runs\seed_{0}" -f $Seed)
$HistoryPath = Join-Path $Run "training_history.json"
$CheckpointPath = Join-Path $Run "last.pt"
$RunStatePath = Join-Path $Run "run_state.json"
$LivePath = Join-Path $Run "live_progress.json"

function Write-Status {
    param([string]$Message)
    $Line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $Line
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
}

function Set-StateValue {
    param(
        [string]$Path,
        [scriptblock]$Update
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $State = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    & $Update $State
    $State | ConvertTo-Json | Set-Content -LiteralPath $Path -Encoding UTF8
}

try {
    $Trainer = Get-CimInstance Win32_Process -Filter "ProcessId=$TrainerPid"
    if (
        -not $Trainer -or
        -not $Trainer.CommandLine.Contains("train_b5_ptv2.py") -or
        -not $Trainer.CommandLine.Contains("c3_ptv2_repeats")
    ) {
        throw "Trainer PID $TrainerPid does not match the active C3 process."
    }

    $Request = [ordered]@{
        status = "waiting"
        requested_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
        target_epoch = $TargetEpoch
        trainer_pid = $TrainerPid
        wrapper_pid = $WrapperPid
        seed = $Seed
        message = "Pause after epoch $TargetEpoch checkpoint is complete"
    }
    $Request | ConvertTo-Json | Set-Content -LiteralPath $RequestPath -Encoding UTF8
    Write-Status "Waiting to pause seed $Seed after epoch $TargetEpoch."

    while ($true) {
        if (-not (Get-Process -Id $TrainerPid -ErrorAction SilentlyContinue)) {
            throw "Trainer exited before epoch $TargetEpoch was safely recorded."
        }

        $LastEpoch = 0
        if (Test-Path -LiteralPath $HistoryPath) {
            try {
                $HistoryText = [System.IO.File]::ReadAllText($HistoryPath)
                $History = @($HistoryText | ConvertFrom-Json)
                if ($History.Count -gt 0) {
                    $LastEpoch = [int]$History[-1].epoch
                }
            } catch {
            }
        }

        if ($LastEpoch -ge $TargetEpoch) {
            if (-not (Test-Path -LiteralPath $CheckpointPath)) {
                throw "Epoch $LastEpoch was recorded but last.pt is missing."
            }
            $Checkpoint = Get-Item -LiteralPath $CheckpointPath
            if ($Checkpoint.Length -lt 10000000) {
                throw "last.pt is unexpectedly small: $($Checkpoint.Length) bytes."
            }

            $Trainer = Get-CimInstance Win32_Process -Filter "ProcessId=$TrainerPid"
            if (
                -not $Trainer -or
                -not $Trainer.CommandLine.Contains("train_b5_ptv2.py") -or
                -not $Trainer.CommandLine.Contains("c3_ptv2_repeats")
            ) {
                throw "Trainer PID changed before the requested stop."
            }
            Stop-Process -Id $TrainerPid -Force
            Write-Status "Stopped trainer PID $TrainerPid after saved epoch $LastEpoch."

            $Deadline = (Get-Date).AddSeconds(30)
            while (
                (Get-Process -Id $WrapperPid -ErrorAction SilentlyContinue) -and
                (Get-Date) -lt $Deadline
            ) {
                Start-Sleep -Milliseconds 500
            }
            if (Get-Process -Id $WrapperPid -ErrorAction SilentlyContinue) {
                $Wrapper = Get-CimInstance Win32_Process -Filter "ProcessId=$WrapperPid"
                if ($Wrapper -and $Wrapper.CommandLine.Contains("run_c3_local.ps1")) {
                    Stop-Process -Id $WrapperPid -Force
                    Write-Status "Stopped wrapper PID $WrapperPid after its exit timeout."
                }
            }

            $Now = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
            Set-StateValue -Path $LauncherPath -Update {
                param($State)
                $State.status = "paused"
                $State.updated_at = $Now
                $State.exit_code = 0
                $State.message = "Paused by user after epoch $LastEpoch; rerun the local starter to resume"
            }
            Set-StateValue -Path $RunStatePath -Update {
                param($State)
                $State.status = "paused"
                $State.updated_at = $Now
            }
            Set-StateValue -Path $LivePath -Update {
                param($State)
                $State.status = "paused"
                $State.updated_at = $Now
                $State.phase = "paused"
                $State.phase_eta_seconds = $null
            }

            $Request["status"] = "complete"
            $Request["completed_at"] = $Now
            $Request["completed_epoch"] = $LastEpoch
            $Request["checkpoint_path"] = $CheckpointPath
            $Request | ConvertTo-Json | Set-Content -LiteralPath $RequestPath -Encoding UTF8
            Write-Status "Pause completed at epoch $LastEpoch."
            exit 0
        }

        Start-Sleep -Seconds 2
    }
} catch {
    $Now = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    @{
        status = "failed"
        updated_at = $Now
        target_epoch = $TargetEpoch
        trainer_pid = $TrainerPid
        wrapper_pid = $WrapperPid
        message = $_.Exception.Message
    } | ConvertTo-Json | Set-Content -LiteralPath $RequestPath -Encoding UTF8
    Write-Status "Pause watcher failed: $($_.Exception.Message)"
    exit 1
}
