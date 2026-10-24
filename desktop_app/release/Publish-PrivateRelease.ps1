param(
    [Parameter(Mandatory=$true)][string]$Version,
    [Parameter(Mandatory=$true)][string]$BundleDir,
    [string]$Repository = 'b0ringzl/SpaceScan-Trees-Desktop-Private',
    [switch]$Publish
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($Version -notmatch '^v\d{4}\.\d{2}\.\d{2}\.\d+$') { throw 'Invalid version' }
$gh = Get-Command gh -ErrorAction Stop
$bundle = (Resolve-Path -LiteralPath $BundleDir).Path
$releaseManifest = Join-Path $bundle "SpaceScan-Trees-Desktop-$Version.release.json"
$manifest = Get-Content -LiteralPath $releaseManifest -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.version -ne $Version) { throw 'Release manifest version mismatch' }

# A Git Credential Manager token can be reused without writing it into a file
# or command argument. Its value is never printed; an existing GH_TOKEN wins.
$temporaryToken = $false
if (-not $env:GH_TOKEN) {
    $credential = "protocol=https`nhost=github.com`n`n" | git credential fill 2>$null
    $password = @($credential | Where-Object { $_ -match '^password=' })
    if ($password.Count -ne 1) { throw 'No GitHub credential; run gh auth login.' }
    $env:GH_TOKEN = $password[0].Substring(9)
    $credential = $null
    $password = $null
    $temporaryToken = $true
}
try {
    $private = & $gh.Source repo view $Repository --json isPrivate --jq '.isPrivate'
    if ($LASTEXITCODE -ne 0 -or $private.Trim() -ne 'true') {
        throw "Refusing to publish to a non-private repository: $Repository"
    }
    $assets = @($releaseManifest)
    foreach ($part in $manifest.parts) {
        $file = Join-Path $bundle ([string]$part.name)
        if ((Get-Item -LiteralPath $file).Length -ne [long]$part.bytes) {
            throw "Part size mismatch: $($part.name)"
        }
        $assets += $file
    }
    # The full dirty-tree source snapshot is a local rollback artifact only.
    # It can contain machine-specific paths and is never uploaded to GitHub.
    foreach ($name in @('Install-SpaceScanTrees.ps1','Install-FromGitHub.ps1',
                       'Switch-SpaceScanTreesVersion.ps1')) {
        $file = Join-Path $bundle $name
        if (-not (Test-Path -LiteralPath $file)) { throw "Release asset absent: $name" }
        $assets += $file
    }
    $existing = & $gh.Source api "repos/$Repository/releases/tags/$Version" 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $gh.Source release create $Version --repo $Repository --target main `
            --draft --prerelease --title "SpaceScan Trees Desktop $Version (private evaluation)" `
            --notes 'Private Windows evaluation build. Verify GPU, WebView2, inference and cloud login on each new PC. Prior versions remain available for rollback.'
        if ($LASTEXITCODE -ne 0) { throw 'Unable to create draft release' }
    } else {
        $release = $existing | ConvertFrom-Json
        if (-not $release.draft) { throw 'Release is already published; refusing to mutate assets' }
    }

    foreach ($file in $assets) {
        $name = [IO.Path]::GetFileName($file)
        $remote = & $gh.Source api "repos/$Repository/releases/tags/$Version"
        if ($LASTEXITCODE -ne 0) { throw 'Cannot read release asset list' }
        $release = $remote | ConvertFrom-Json
        $found = @($release.assets | Where-Object { $_.name -eq $name })
        if ($found.Count -eq 0) {
            Write-Host "Uploading $name"
            & $gh.Source release upload $Version $file --repo $Repository
            if ($LASTEXITCODE -ne 0) { throw "Upload failed: $name" }
        }
    }
    $remote = & $gh.Source api "repos/$Repository/releases/tags/$Version"
    if ($LASTEXITCODE -ne 0) { throw 'Cannot verify uploaded assets' }
    $release = $remote | ConvertFrom-Json
    foreach ($file in $assets) {
        $name = [IO.Path]::GetFileName($file)
        $found = @($release.assets | Where-Object { $_.name -eq $name })
        if ($found.Count -ne 1 -or [long]$found[0].size -ne (Get-Item -LiteralPath $file).Length) {
            throw "Remote asset missing or size mismatch: $name"
        }
        $expected = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $found[0].digest -or $found[0].digest -ne "sha256:$expected") {
            throw "Remote SHA-256 digest absent or mismatched: $name"
        }
    }
    if ($Publish) {
        & $gh.Source release edit $Version --repo $Repository --draft=false
        if ($LASTEXITCODE -ne 0) { throw 'Asset verification passed but publish failed' }
        Write-Host "PRIVATE_RELEASE_PUBLISHED $Repository $Version"
    } else {
        Write-Host "PRIVATE_RELEASE_DRAFT_VERIFIED $Repository $Version"
    }
} finally {
    if ($temporaryToken) { Remove-Item Env:GH_TOKEN -ErrorAction SilentlyContinue }
}
