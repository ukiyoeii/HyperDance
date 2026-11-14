param(
    [string]$Root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$InputPath = '',
    [string]$OutputPath = '',
    [string]$Title = '街舞骨架数据预览',
    [string]$Subtitle = '真实数据自带标签 · 非模型预测 · 每类一个样本的静态帧'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$previewRoot = Join-Path $Root 'outputs\data_check'
if (-not $InputPath) { $InputPath = Join-Path $previewRoot 'preview_samples.json' }
if (-not $OutputPath) { $OutputPath = Join-Path $previewRoot 'skeleton_contact_sheet.png' }
$preview = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$width = 1280
$height = 1480
$bitmap = [System.Drawing.Bitmap]::new($width, $height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$bg = [System.Drawing.ColorTranslator]::FromHtml('#0a101a')
$card = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml('#121e2c'))
$text = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml('#edf4ff'))
$muted = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml('#9bafc8'))
$joint = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml('#effffb'))
$upper = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml('#64dec6'), 3)
$lower = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml('#a59cff'), 3)
$titleFont = [System.Drawing.Font]::new('Microsoft YaHei', 22)
$cardFont = [System.Drawing.Font]::new('Segoe UI', 12)
$smallFont = [System.Drawing.Font]::new('Microsoft YaHei', 10)
try {
    $graphics.Clear($bg)
    $graphics.DrawString($Title, $titleFont, $text, 30, 18)
    $graphics.DrawString($Subtitle, $smallFont, $muted, 32, 66)
    for ($i = 0; $i -lt $preview.clips.Count; $i++) {
        $clip = $preview.clips[$i]
        $x = 28 + ($i % 4) * 312
        $y = 110 + [math]::Floor($i / 4) * 326
        $graphics.FillRectangle($card, $x, $y, 296, 308)
        $label = if ($clip.source -eq 'AIST++') { $clip.title.Split('/')[0].Trim() } else { $clip.title }
        $graphics.DrawString($label, $cardFont, $text, $x + 14, $y + 12)
        $graphics.DrawString("$($clip.source) | $($clip.view)", $smallFont, $muted, $x + 14, $y + 39)
        $frameIndex = 0
        while ($frameIndex + 1 -lt $clip.times.Count -and $clip.times[$frameIndex + 1] -le 1.5) { $frameIndex++ }
        $points = $clip.frames[$frameIndex]
        $scale = 232
        $originX = $x + 32
        $originY = $y + 68
        foreach ($edge in $preview.edges) {
            $a = $points[$edge[0]]
            $b = $points[$edge[1]]
            if ($a[2] -lt 0.2 -or $b[2] -lt 0.2) { continue }
            $pen = if ($edge[0] -ge 11) { $lower } else { $upper }
            $graphics.DrawLine($pen, [single]($originX + $a[0] * $scale), [single]($originY + $a[1] * $scale), [single]($originX + $b[0] * $scale), [single]($originY + $b[1] * $scale))
        }
        foreach ($point in $points) {
            if ($point[2] -lt 0.2) { continue }
            $graphics.FillEllipse($joint, [single]($originX + $point[0] * $scale - 3), [single]($originY + $point[1] * $scale - 3), 6, 6)
        }
    }
    $graphics.DrawString('数据自带标签，不是模型预测。预览 HTML 支持播放、切换类别和拖动时间轴。', $smallFont, $muted, 32, 1434)
    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
    foreach ($resource in @($graphics, $bitmap, $card, $text, $muted, $joint, $upper, $lower, $titleFont, $cardFont, $smallFont)) { $resource.Dispose() }
}
Write-Output $OutputPath
