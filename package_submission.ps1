# package_submission.ps1
# Creates a clean submission zip excluding ARC files, debug scripts, and scratch files.

$submissionDir = "..\NetMF_submission"
$zipPath = "..\NetMF_vpatel34_hw1.zip"

# Clean up previous attempt
if (Test-Path $submissionDir) { Remove-Item $submissionDir -Recurse -Force }
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

New-Item -ItemType Directory -Path $submissionDir | Out-Null
New-Item -ItemType Directory -Path "$submissionDir\results" | Out-Null
New-Item -ItemType Directory -Path "$submissionDir\data" | Out-Null

# ── Core code ───────────────────────────────────────────────────────────────
$codeFiles = @(
    "netmf.py",
    "predict.py",
    "run_experiments.py",
    "analyze_task2.py",
    "analyze_task3.py",
    "report.tex",
    "README.md"
)
foreach ($f in $codeFiles) {
    if (Test-Path $f) {
        Copy-Item $f "$submissionDir\$f"
        Write-Host "  [code]    $f"
    } else {
        Write-Warning "  MISSING:  $f"
    }
}

# ── Data ────────────────────────────────────────────────────────────────────
Write-Host ""
foreach ($f in Get-ChildItem "data\") {
    Copy-Item $f.FullName "$submissionDir\data\"
    Write-Host "  [data]    data\$($f.Name)"
}

# ── Results: logs and embeddings (no ARC slurm logs) ─────────────────────
Write-Host ""
$resultFiles = @(
    # BlogCatalog
    "blogcatalog_T1.log",  "blogcatalog_T1.npy",  "blogcatalog_T1_predict.log",
    "blogcatalog_T10.log", "blogcatalog_T10.npy", "blogcatalog_T10_predict.log",
    # PPI
    "ppi_T1.log",  "ppi_T1.npy",  "ppi_T1_predict.log",
    "ppi_T10.log", "ppi_T10.npy", "ppi_T10_predict.log",
    # Wikipedia
    "wikipedia_T1.log",  "wikipedia_T1.npy",  "wikipedia_T1_predict.log",
    "wikipedia_T10.log", "wikipedia_T10.npy", "wikipedia_T10_predict.log",
    # Flickr (ran on ARC; embeddings + predict logs included for reproducibility)
    "flickr_T1.log",  "flickr_T1.npy",  "flickr_T1_predict.log",
    "flickr_T10.log", "flickr_T10.npy", "flickr_T10_predict.log",
    # Summary tables and plots
    "summary.csv",
    "sparsity_table.csv",
    "scalability_table.csv",
    "scalability_plot.png",
    "svd_spectrum_blogcatalog_T1.png",
    "svd_spectrum_ppi_T1.png",
    "svd_spectrum_wikipedia_T1.png",
    "svd_spectrum_flickr_T1.png"
)
foreach ($f in $resultFiles) {
    $src = "results\$f"
    if (Test-Path $src) {
        Copy-Item $src "$submissionDir\results\$f"
        Write-Host "  [result]  results\$f"
    } else {
        Write-Warning "  MISSING:  results\$f"
    }
}

# ── Zip it up ────────────────────────────────────────────────────────────────
Write-Host ""
Compress-Archive -Path "$submissionDir\*" -DestinationPath $zipPath
$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "Done! Submission zip: $zipPath ($sizeMB MB)"
Write-Host ""
Write-Host "Contents summary:"
Get-ChildItem $submissionDir -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Replace((Resolve-Path $submissionDir).Path + "\", "")
    Write-Host ("  " + $rel)
}
