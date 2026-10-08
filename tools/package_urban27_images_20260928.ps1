param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$package = Join-Path $workspace 'deliverables\香港市區常見27樹種_現有圖片副本_20260928_v2'
$officialUrl = 'https://www.greening.gov.hk/tc/resource-centre/plant-species-knowledge/common-tree-in-urban-environment-of-hong-kong/index.html'

$species = @(
    @{ cn='柱狀南洋杉'; latin='Araucaria columnaris'; aliases=@() },
    @{ cn='假檳榔'; latin='Archontophoenix alexandrae'; aliases=@() },
    @{ cn='秋楓'; latin='Bischofia javanica'; aliases=@() },
    @{ cn='串錢柳'; latin='Callistemon viminalis'; aliases=@() },
    @{ cn='木麻黃'; latin='Casuarina equisetifolia'; aliases=@() },
    @{ cn='陰香'; latin='Cinnamomum burmannii'; aliases=@() },
    @{ cn='樟'; latin='Cinnamomum camphora'; aliases=@() },
    @{ cn='水石榕'; latin='Elaeocarpus hainanensis'; aliases=@() },
    @{ cn='垂葉榕'; latin='Ficus benjamina'; aliases=@() },
    @{ cn='細葉榕'; latin='Ficus microcarpa'; aliases=@() },
    @{ cn='銀樺'; latin='Grevillea robusta'; aliases=@() },
    @{ cn='藍花楹'; latin='Jacaranda mimosifolia'; aliases=@() },
    @{ cn='大花紫薇'; latin='Lagerstroemia speciosa'; aliases=@() },
    @{ cn='蒲葵'; latin='Livistona chinensis'; aliases=@() },
    @{ cn='荷花玉蘭'; latin='Magnolia grandiflora'; aliases=@() },
    @{ cn='白楸'; latin='Mallotus paniculatus'; aliases=@() },
    @{ cn='白千層'; latin='Melaleuca cajuputi'; aliases=@('Melaleuca cajuputi subsp. cumingiana') },
    @{ cn='樂昌含笑'; latin='Michelia chapensis'; aliases=@() },
    @{ cn='大王椰子'; latin='Roystonea regia'; aliases=@() },
    @{ cn='鵝掌柴'; latin='Schefflera heptaphylla'; aliases=@() },
    @{ cn='火焰樹'; latin='Spathodea campanulata'; aliases=@() },
    @{ cn='假蘋婆'; latin='Sterculia lanceolata'; aliases=@() },
    @{ cn='黃鐘木'; latin='Tabebuia chrysantha'; aliases=@() },
    @{ cn='紅花風鈴木'; latin='Tabebuia rosea'; aliases=@() },
    @{ cn='欖仁'; latin='Terminalia catappa'; aliases=@() },
    @{ cn='小葉欖仁'; latin='Terminalia mantaly'; aliases=@() },
    @{ cn='二喬木蘭'; latin='Yulania x soulangeana'; aliases=@('Magnolia x soulangeana') }
)

$byLatin = @{}
foreach ($s in $species) {
    $byLatin[$s.latin] = $s
    foreach ($alias in $s.aliases) { $byLatin[$alias] = $s }
}

$imageExtensions = @('.jpg','.jpeg','.png','.webp','.tif','.tiff','.bmp','.heic')
$candidates = [System.Collections.Generic.List[object]]::new()
$missing = [System.Collections.Generic.List[object]]::new()

function Add-Candidate([object]$item, [string]$sourceSet, [string]$sourcePath, [object]$meta) {
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        $missing.Add([pscustomobject]@{ species=$item.latin; source_set=$sourceSet; source_path=$sourcePath })
        return
    }
    if ([IO.Path]::GetExtension($sourcePath).ToLowerInvariant() -notin $imageExtensions) { return }
    $candidates.Add([pscustomobject]@{
        chinese_name=$item.cn
        scientific_name=$item.latin
        source_set=$sourceSet
        source_path=(Resolve-Path -LiteralPath $sourcePath).Path
        source_page=[string]$meta.source_page
        license=[string]$meta.license
        attribution=[string]$meta.attribution
        source_status=[string]$meta.status
        training_eligible=[string]$meta.training_eligible
    })
}

# Source folders whose taxonomy is expressed in their directory names.
$namedCollections = @(
    @{ name='inat_max150'; root='external_datasets\inat_max150' },
    @{ name='inaturalist_2026-06-07'; root='external_datasets\inaturalist_dated\2026-06-07' },
    @{ name='inat_max150_test'; root='external_datasets\inat_max150_test' },
    @{ name='inat_parent_name_supplement'; root='external_datasets\inat_parent_name_supplement' },
    @{ name='project_dataset'; root='dataset' }
)
foreach ($collection in $namedCollections) {
    $root = Join-Path $workspace $collection.root
    foreach ($s in $species) {
        foreach ($taxonName in @($s.latin) + @($s.aliases)) {
            $taxonDir = Join-Path $root $taxonName
            if (-not (Test-Path -LiteralPath $taxonDir -PathType Container)) { continue }
            $metadataByName = @{}
            $metadataFile = Join-Path $taxonDir 'metadata.jsonl'
            if (Test-Path -LiteralPath $metadataFile -PathType Leaf) {
                foreach ($line in [IO.File]::ReadLines($metadataFile)) {
                    if (-not $line.Trim()) { continue }
                    $row = $line | ConvertFrom-Json
                    $fileName = [IO.Path]::GetFileName([string]$row.local_path)
                    if ($fileName) { $metadataByName[$fileName] = $row }
                }
            }
            foreach ($file in Get-ChildItem -LiteralPath $taxonDir -Recurse -File) {
                if ($file.Extension.ToLowerInvariant() -notin $imageExtensions) { continue }
                $row = $metadataByName[$file.Name]
                $meta = [pscustomobject]@{
                    source_page=if ($row) { $row.inat_uri } else { '' }
                    license=if ($row) { $row.photo_license_code } else { '' }
                    attribution=if ($row) { $row.photo_attribution } else { '' }
                    status=if ($collection.name -eq 'project_dataset') { 'project_folder_label; not independently rechecked' } else { 'source_taxon_label; not independently rechecked' }
                    training_eligible='not assessed by this package'
                }
                Add-Candidate $s $collection.name $file.FullName $meta
            }
        }
    }
}

# Flat collections: only a record's explicitly named species and existing file are used.
$manifests = @(
    @{ name='spacescan_references'; path='external_datasets\spacescan_references_20260910_v1\manifest.json' },
    @{ name='spacescan_whole_tree_references'; path='external_datasets\spacescan_whole_tree_references_20260910_v2\manifest.json' },
    @{ name='herbarium_candidates'; path='external_datasets\herbarium_training_candidates_20260907\curated_manifest.json' },
    @{ name='sstv1_incremental'; path='external_datasets\sstv1_incremental_20260914_v1\curated_manifest.json' },
    @{ name='sstv1_white_supplement'; path='external_datasets\sstv1_white_supplement_20260914_v1\curated_manifest.json' },
    @{ name='focus8_authoritative'; path='external_datasets\focus8_authoritative_20260909_v1\manifest.json' },
    @{ name='palm_confusion'; path='external_datasets\palm_confusion_20260908\curated_manifest.json' },
    @{ name='commons_michelia_pending'; path='outputs\SSTv1_Urban153_Commons_Michelia_20260924_v1\manifest.json' },
    @{ name='commons_yulania_pending'; path='outputs\SSTv1_Urban153_Commons_Yulania_20260924_v1\manifest.json' }
)
foreach ($entry in $manifests) {
    $manifestPath = Join-Path $workspace $entry.path
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { continue }
    $rows = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $manifestDir = Split-Path -Parent $manifestPath
    foreach ($row in $rows) {
        $taxonName = @($row.source_species, $row.species, $row.target_species, $row.label) | Where-Object { $_ } | Select-Object -First 1
        if (-not $taxonName -or -not $byLatin.ContainsKey([string]$taxonName)) { continue }
        $fileRef = @($row.file, $row.local_path) | Where-Object { $_ } | Select-Object -First 1
        if (-not $fileRef) { continue }
        $sourcePath = if ([IO.Path]::IsPathRooted([string]$fileRef)) { [string]$fileRef } else { Join-Path $manifestDir ([string]$fileRef) }
        $meta = [pscustomobject]@{
            source_page=(@($row.source_page,$row.source_url,$row.inat_uri) | Where-Object { $_ } | Select-Object -First 1)
            license=(@($row.license,$row.photo_license_code) | Where-Object { $_ } | Select-Object -First 1)
            attribution=(@($row.attribution,$row.photo_attribution,$row.copyright_holder) | Where-Object { $_ } | Select-Object -First 1)
            status=(@($row.review_status,$row.status,$row.split) | Where-Object { $_ } | Select-Object -First 1)
            training_eligible=if ($null -ne $row.training_eligible) { [string]$row.training_eligible } else { 'not stated' }
        }
        Add-Candidate $byLatin[[string]$taxonName] $entry.name $sourcePath $meta
    }
}

# User-confirmed Jacaranda originals were preserved in a previous run even
# though five of the six Desktop paths no longer exist. Include both originals
# and the separately identified target-tree crops, never as independent trees.
$jacarandaRun = Join-Path $workspace 'outputs\SSTv1_Jacaranda_User_20260922_v1'
foreach ($subdir in @('user_originals','target_crops')) {
    $sourceSet = if ($subdir -eq 'user_originals') { 'user_confirmed_original_copy' } else { 'user_confirmed_target_crop' }
    $status = if ($subdir -eq 'user_originals') { 'user-confirmed target tree; original copy' } else { 'user-confirmed target tree; researcher crop boundary not pixel-reviewed' }
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $jacarandaRun $subdir) -File) {
        $meta = [pscustomobject]@{ source_page=''; license='private project image; redistribution not assessed'; attribution='user-provided'; status=$status; training_eligible='not assessed by this package' }
        Add-Candidate $byLatin['Jacaranda mimosifolia'] $sourceSet $file.FullName $meta
    }
}
$terminaliaPhoto = 'C:\Users\25041725G\Desktop\验证\微信图片_20260916084639_130_2.jpg'
$meta = [pscustomobject]@{ source_page=''; license='user-provided; redistribution not assessed'; attribution='user-provided'; status='user-identified target tree'; training_eligible='not assessed by this package' }
Add-Candidate $byLatin['Terminalia mantaly'] 'user_identified' $terminaliaPhoto $meta

Write-Output ("Species: {0}; candidate image occurrences: {1}; unavailable file references: {2}" -f $species.Count,$candidates.Count,$missing.Count)
if ($DryRun) {
    $candidates | Group-Object chinese_name | Sort-Object Name | ForEach-Object { '{0}: {1}' -f $_.Name,$_.Count }
    return
}
if (Test-Path -LiteralPath $package) { throw "Package already exists; refusing to overwrite: $package" }
New-Item -ItemType Directory -Path $package -Force | Out-Null

$imageIndex = [System.Collections.Generic.List[object]]::new()
$occurrences = [System.Collections.Generic.List[object]]::new()
$summary = [System.Collections.Generic.List[object]]::new()
$hashToSpecies = @{}

for ($i = 0; $i -lt $species.Count; $i++) {
    $s = $species[$i]
    $folderName = '{0:D2}_{1}' -f ($i+1),$s.cn
    $folderPath = Join-Path $package $folderName
    New-Item -ItemType Directory -Path $folderPath | Out-Null
    $byHash = @{}
    $sequence = 0
    foreach ($candidate in ($candidates | Where-Object scientific_name -EQ $s.latin)) {
        $hash = (Get-FileHash -LiteralPath $candidate.source_path -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $byHash.ContainsKey($hash)) {
            $sequence++
            $extension = [IO.Path]::GetExtension($candidate.source_path).ToLowerInvariant()
            $destName = '{0:D5}_{1}{2}' -f $sequence,$candidate.source_set,$extension
            $destPath = Join-Path $folderPath $destName
            Copy-Item -LiteralPath $candidate.source_path -Destination $destPath -ErrorAction Stop
            $copiedHash = (Get-FileHash -LiteralPath $destPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($copiedHash -ne $hash) { throw "SHA-256 verification failed: $destPath" }
            $byHash[$hash] = $destPath
            $imageIndex.Add([pscustomobject]@{
                chinese_name=$s.cn; scientific_name=$s.latin; package_file=$destPath.Substring($package.Length+1)
                first_source_file=$candidate.source_path; first_source_set=$candidate.source_set
                sha256=$hash; bytes=(Get-Item -LiteralPath $destPath).Length
            })
            if (-not $hashToSpecies.ContainsKey($hash)) { $hashToSpecies[$hash] = [System.Collections.Generic.HashSet[string]]::new() }
            [void]$hashToSpecies[$hash].Add($s.latin)
        }
        $occurrences.Add([pscustomobject]@{
            chinese_name=$s.cn; scientific_name=$s.latin
            package_file=$byHash[$hash].Substring($package.Length+1)
            source_file=$candidate.source_path; source_set=$candidate.source_set
            source_page=$candidate.source_page; license=$candidate.license
            attribution=$candidate.attribution; source_status=$candidate.source_status
            training_eligible=$candidate.training_eligible; sha256=$hash
        })
    }
    $summary.Add([pscustomobject]@{ chinese_name=$s.cn; scientific_name=$s.latin; unique_images=$sequence; source_occurrences=($candidates | Where-Object scientific_name -EQ $s.latin).Count })
    Write-Output ('{0}: {1} unique images' -f $s.cn,$sequence)
}

$imageIndex | Export-Csv -LiteralPath (Join-Path $package '圖片索引.csv') -NoTypeInformation -Encoding UTF8
$occurrences | Export-Csv -LiteralPath (Join-Path $package '來源與審核狀態.csv') -NoTypeInformation -Encoding UTF8
$summary | Export-Csv -LiteralPath (Join-Path $package '樹種數量.csv') -NoTypeInformation -Encoding UTF8
$missing | Export-Csv -LiteralPath (Join-Path $package '原清單中缺失的檔案.csv') -NoTypeInformation -Encoding UTF8
$conflicts = foreach ($pair in $hashToSpecies.GetEnumerator()) {
    if ($pair.Value.Count -gt 1) { [pscustomobject]@{ sha256=$pair.Key; species=($pair.Value | Sort-Object) -join ' | ' } }
}
@($conflicts) | Export-Csv -LiteralPath (Join-Path $package '跨樹種相同圖片.csv') -NoTypeInformation -Encoding UTF8

$readme = @"
香港市區常見 27 樹種：本地現有圖片副本

樹種清單：$officialUrl
整理日期：2026-09-28
範圍：現有本地圖片；沒有從網站新增下載圖片，沒有複製模型或訓練產物。
原始檔案保持不變；同一樹種內的完全相同檔案按 SHA-256 去重，來源記錄全部保留在「來源與審核狀態.csv」。
每個樹種資料夾只存圖片；「圖片索引.csv」可查每張副本的原始絕對路徑及 SHA-256。
注意：資料夾分類沿用既有來源標籤、研究清單及使用者指認，不代表全部圖片都經專家複核；待審/隔離圖片不能直接用作訓練。
有來源授權或版權限制的圖片僅供本機整理與審視，勿直接公開分發。授權及出處見「來源與審核狀態.csv」。
未將「Araucaria columnaris(Araucaria heterophylla)」混合資料夾強行歸入柱狀南洋杉，避免錯分。
樹種有空資料夾表示此次可核實的本地圖片來源沒有該樹種圖片。
藍花楹的 6 張使用者確認原圖及 6 張目標樹裁片從既有訓練存檔副本恢復；裁片與其原圖不是獨立樹木樣本。
"@
Set-Content -LiteralPath (Join-Path $package '說明.txt') -Value $readme -Encoding UTF8

$totalBytes = ($imageIndex | Measure-Object bytes -Sum).Sum
Write-Output ("DONE: {0} unique images, {1} occurrences, {2:N1} MiB, {3} cross-species duplicate hashes; package: {4}" -f $imageIndex.Count,$occurrences.Count,($totalBytes/1MB),@($conflicts).Count,$package)
