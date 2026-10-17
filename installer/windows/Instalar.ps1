#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'JevQatLocal'),
    [string]$DownloadCache,
    [string]$ModelCache,
    [switch]$DependenciesOnly,
    [switch]$NoLaunch,
    [switch]$NoBrowser,
    [switch]$NoShortcuts
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$OutputEncoding = New-Object Text.UTF8Encoding($false)
[Console]::OutputEncoding = $OutputEncoding
$lock = $null
try {
    if ($env:JEV_SETUP_GATE) {
        $deadline = [DateTime]::UtcNow.AddSeconds(30)
        while (-not (Test-Path -LiteralPath $env:JEV_SETUP_GATE)) {
            if ([DateTime]::UtcNow -ge $deadline) { throw 'No se pudo iniciar el instalador EXE.' }
            Start-Sleep -Milliseconds 50
        }
    }
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'Este instalador requiere Windows x64 (Intel o AMD).'
    }
    $bundle = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'bundle.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($bundle.product -ne 'jev-qat-local-e2b' -or $bundle.schema_version -ne 1) { throw 'Paquete no reconocido.' }
    $InstallDir = [IO.Path]::GetFullPath($InstallDir).TrimEnd('\')
    if ($InstallDir.Length -le 3) { throw 'Elige una carpeta propia para la aplicacion.' }
    $marker = Join-Path $InstallDir '.jev-qat-install.json'
    if (Test-Path -LiteralPath $marker) {
        $owner = Get-Content -LiteralPath $marker -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($owner.product -ne $bundle.product) { throw 'La carpeta pertenece a otra aplicacion.' }
    } else {
        if ((Test-Path -LiteralPath $InstallDir) -and @(Get-ChildItem -LiteralPath $InstallDir -Force).Count -gt 0) {
            throw 'La carpeta no esta vacia y no pertenece a E2B. Elige otra con -InstallDir.'
        }
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
        [IO.File]::WriteAllText($marker, '{"product":"jev-qat-local-e2b"}', (New-Object Text.UTF8Encoding($false)))
    }
    $lock = [IO.File]::Open((Join-Path $InstallDir 'installer.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    $downloads = Join-Path $InstallDir 'downloads'
    New-Item -ItemType Directory -Force -Path $downloads | Out-Null
    $spec = $bundle.assets.python
    if ($spec.sha256 -notmatch '^[a-f0-9]{64}$' -or $spec.url -notlike 'https://www.python.org/ftp/python/*') {
        throw 'Origen o checksum de Python no valido.'
    }
    $archive = Join-Path $downloads $spec.filename
    if (-not (Test-Path -LiteralPath $archive) -and $DownloadCache) {
        $cached = Join-Path $DownloadCache $spec.filename
        if ((Test-Path -LiteralPath $cached) -and (Get-FileHash -LiteralPath $cached -Algorithm SHA256).Hash -eq $spec.sha256) {
            Copy-Item -LiteralPath $cached -Destination $archive
        }
    }
    if ((Test-Path -LiteralPath $archive) -and (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $spec.sha256) {
        Remove-Item -LiteralPath $archive
    }
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Host 'Descargando Python privado (unos 12 MB)...'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $client = New-Object Net.WebClient
        try { $client.DownloadFile($spec.url, "$archive.part") } finally { $client.Dispose() }
        if ((Get-FileHash -LiteralPath "$archive.part" -Algorithm SHA256).Hash -ne $spec.sha256) { throw 'SHA-256 de Python incorrecto.' }
        Move-Item -LiteralPath "$archive.part" -Destination $archive
    }
    $pythonDir = Join-Path $InstallDir 'python'
    $receipt = Join-Path $pythonDir '.archive-sha256'
    if (-not ((Test-Path -LiteralPath $receipt) -and (Get-Content -LiteralPath $receipt -Raw).Trim() -eq $spec.sha256 -and (Test-Path -LiteralPath (Join-Path $pythonDir 'python.exe')))) {
        Write-Host 'Preparando Python aislado...'
        New-Item -ItemType Directory -Force -Path $pythonDir | Out-Null
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [IO.Compression.ZipFile]::OpenRead($archive)
        try {
            foreach ($entry in $zip.Entries) {
                $target = [IO.Path]::GetFullPath((Join-Path $pythonDir $entry.FullName))
                if (-not $target.StartsWith("$pythonDir\", [StringComparison]::OrdinalIgnoreCase)) { throw 'Ruta no valida en el ZIP de Python.' }
                if ($entry.FullName.EndsWith('/')) { New-Item -ItemType Directory -Force -Path $target | Out-Null; continue }
                New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($target)) | Out-Null
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
            }
        } finally { $zip.Dispose() }
        [IO.File]::WriteAllText($receipt, $spec.sha256)
    }
    # Explicit paths keep user/global site-packages and PYTHONPATH isolated.
    $paths = "python314.zip`n.`nLib\site-packages`n..\app`n"
    [IO.File]::WriteAllText((Join-Path $pythonDir 'python314._pth'), $paths)
    $arguments = @((Join-Path $PSScriptRoot 'install.py'), '--bundle', $PSScriptRoot, '--install-dir', $InstallDir)
    if ($DownloadCache) { $arguments += @('--download-cache', $DownloadCache) }
    if ($ModelCache) { $arguments += @('--model-cache', $ModelCache) }
    if ($DependenciesOnly) { $arguments += '--dependencies-only' }
    if ($NoLaunch) { $arguments += '--no-launch' }
    if ($NoBrowser) { $arguments += '--no-browser' }
    if ($NoShortcuts) { $arguments += '--no-shortcuts' }
    & (Join-Path $pythonDir 'python.exe') @arguments
    if ($LASTEXITCODE -ne 0) { throw "La instalacion termino con codigo $LASTEXITCODE. Puedes repetirla para reanudar el modelo." }
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    if ($null -ne $lock) { $lock.Dispose() }
}
