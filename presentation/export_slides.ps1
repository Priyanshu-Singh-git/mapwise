# Export a .pptx with PowerPoint: one 1920x1080 PNG per slide + a matching .pdf
# Usage: powershell -File export_slides.ps1 [deckBaseName]   (default: deck)
param([string]$Deck = "deck")
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pptx = Join-Path $here ($Deck + ".pptx")
$slides = Join-Path $here ($Deck + "_slides")
New-Item -ItemType Directory -Force $slides | Out-Null
Get-ChildItem $slides -Filter *.png | Remove-Item -Force

$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($pptx, $true, $false, $false)
    foreach ($s in $pres.Slides) {
        $out = Join-Path $slides ("slide_{0:D2}.png" -f $s.SlideIndex)
        $s.Export($out, "PNG", 1920, 1080)
    }
    $n = $pres.Slides.Count
    $pres.SaveAs((Join-Path $here ($Deck + ".pdf")), 32)  # 32 = ppSaveAsPDF
    $pres.Close()
    Write-Output ("exported {0} slides + {1}.pdf" -f $n, $Deck)
} finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
