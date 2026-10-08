param([switch]$OpenBrowser)
# Both Windows entry points use the same checked official service.
& (Join-Path $PSScriptRoot 'start-live.ps1') -OpenBrowser:$OpenBrowser
