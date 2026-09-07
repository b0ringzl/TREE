param([switch]$NoBrowser)
& (Join-Path $PSScriptRoot 'start-demo.ps1') -NoBrowser:$NoBrowser
