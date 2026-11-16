[CmdletBinding()]
param(
    [string]$RuntimeDir = $env:SVSG_RUNTIME_DIR,
    [string]$Python = $env:SVSG_PYTHON
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\")).Path
if ([string]::IsNullOrWhiteSpace($RuntimeDir)) {
    # 看门狗通常把当前目录设为 Trae 运行目录；保留该目录即可读取其 .env。
    $currentDir = (Get-Location).Path
    if ((Test-Path (Join-Path $currentDir ".env")) -or
        (Test-Path (Join-Path $currentDir "watchdog-3002.log"))) {
        $RuntimeDir = $currentDir
    }
    else {
        $RuntimeDir = $repoRoot
    }
}
$RuntimeDir = (Resolve-Path $RuntimeDir).Path
if ([string]::IsNullOrWhiteSpace($Python)) {
    $candidates = @(
        (Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LocalAppData "Programs\Python\Python311\python.exe")
    )
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($Python)) {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
}

# 让运行目录继续提供 .env/日志，同时始终从 git 仓库加载源码。
$pathEntries = @($repoRoot)
if (-not [string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $pathEntries += $env:PYTHONPATH
}
$env:PYTHONPATH = $pathEntries -join [IO.Path]::PathSeparator

Push-Location $RuntimeDir
try {
    # 运行目录可能保留迁移前的 svsg/ 副本；runpy 前显式插入仓库路径，
    # 避免 Python 的 cwd 优先级把旧代码加载进来。
    $env:SVSG_REPO_ROOT = $repoRoot
    & $Python -c "import os, runpy, sys; sys.path.insert(0, os.environ['SVSG_REPO_ROOT']); runpy.run_module('svsg', run_name='__main__')"
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
