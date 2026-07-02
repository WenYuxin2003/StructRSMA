$ErrorActionPreference = "Stop"

function RGBColor([int]$r, [int]$g, [int]$b) {
    return $r + ($g -shl 8) + ($b -shl 16)
}

function Add-Text($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h,
    [double]$size = 10, [int]$color = 0, [bool]$bold = $false, [int]$align = 1) {
    $s = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $s.TextFrame.MarginLeft = 2
    $s.TextFrame.MarginRight = 2
    $s.TextFrame.MarginTop = 1
    $s.TextFrame.MarginBottom = 1
    $s.TextFrame.WordWrap = -1
    $s.TextFrame.TextRange.Text = $text
    $s.TextFrame.TextRange.Font.Name = "Arial"
    $s.TextFrame.TextRange.Font.Size = $size
    $s.TextFrame.TextRange.Font.Color.RGB = $color
    if ($bold) { $s.TextFrame.TextRange.Font.Bold = -1 }
    $s.TextFrame.TextRange.ParagraphFormat.Alignment = $align
    return $s
}

function Add-Box($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h,
    [int]$lineColor, [int]$fillColor, [double]$size = 9, [bool]$bold = $false,
    [int]$align = 2, [bool]$round = $true, [double]$lineWeight = 1.2) {
    $shapeType = 5
    if (!$round) { $shapeType = 1 }
    $s = $slide.Shapes.AddShape($shapeType, $x, $y, $w, $h)
    if ($round) {
        try { $s.Adjustments.Item(1) = 0.08 } catch {}
    }
    $s.Fill.Visible = -1
    $s.Fill.ForeColor.RGB = $fillColor
    $s.Line.Visible = -1
    $s.Line.ForeColor.RGB = $lineColor
    $s.Line.Weight = $lineWeight
    $s.TextFrame.MarginLeft = 4
    $s.TextFrame.MarginRight = 4
    $s.TextFrame.MarginTop = 2
    $s.TextFrame.MarginBottom = 2
    $s.TextFrame.WordWrap = -1
    $s.TextFrame.TextRange.Text = $text
    $s.TextFrame.TextRange.Font.Name = "Arial"
    $s.TextFrame.TextRange.Font.Size = $size
    $s.TextFrame.TextRange.Font.Color.RGB = $lineColor
    if ($bold) { $s.TextFrame.TextRange.Font.Bold = -1 }
    $s.TextFrame.TextRange.ParagraphFormat.Alignment = $align
    return $s
}

function Add-Line($slide, [double]$x1, [double]$y1, [double]$x2, [double]$y2,
    [int]$color, [bool]$arrow = $true, [bool]$dash = $false, [double]$weight = 1.1) {
    $l = $slide.Shapes.AddConnector(1, $x1, $y1, $x2, $y2)
    $l.Line.ForeColor.RGB = $color
    $l.Line.Weight = $weight
    if ($arrow) { $l.Line.EndArrowheadStyle = 3 }
    if ($dash) { $l.Line.DashStyle = 4 }
    return $l
}

function Add-Matrix($slide, [double]$x, [double]$y, [int]$rows, [int]$cols,
    [double]$cell, [int]$green, [int[]]$on) {
    for ($r = 0; $r -lt $rows; $r++) {
        for ($c = 0; $c -lt $cols; $c++) {
            $idx = $r * $cols + $c
            $fill = RGBColor 255 255 255
            if ($on -contains $idx) { $fill = $green }
            $sq = $slide.Shapes.AddShape(1, $x + $c * $cell, $y + $r * $cell, $cell, $cell)
            $sq.Fill.ForeColor.RGB = $fill
            $sq.Line.ForeColor.RGB = RGBColor 170 205 170
            $sq.Line.Weight = 0.35
        }
    }
    for ($r = 0; $r -lt 5; $r++) {
        $dot = $slide.Shapes.AddShape(9, $x - 13, $y + 2 + $r * 11, 5, 5)
        $dot.Fill.ForeColor.RGB = RGBColor 24 128 45
        $dot.Line.Visible = 0
    }
    for ($c = 0; $c -lt 5; $c++) {
        $dot = $slide.Shapes.AddShape(9, $x + 4 + $c * 13, $y + $rows * $cell + 12, 5, 5)
        $dot.Fill.ForeColor.RGB = RGBColor 24 128 45
        $dot.Line.Visible = 0
    }
}

function Add-Tokens($slide, [double]$x, [double]$y, [int]$n, [int]$color, [double]$cell = 13) {
    for ($i = 0; $i -lt $n; $i++) {
        $sq = $slide.Shapes.AddShape(1, $x + $i * ($cell + 3), $y, $cell, $cell)
        $sq.Fill.ForeColor.RGB = $color
        $sq.Line.ForeColor.RGB = $color
    }
}

function Add-Molecule($slide, [double]$x, [double]$y, [double]$scale = 1.0) {
    $pts = @(
        @(0, 30, 0), @(25, 12, 0), @(52, 25, 1), @(82, 8, 0), @(108, 26, 2), @(72, 55, 0), @(37, 55, 0)
    )
    $bonds = @(@(0,1), @(1,2), @(2,3), @(3,4), @(2,5), @(5,6), @(6,0))
    foreach ($b in $bonds) {
        $a = $pts[$b[0]]
        $c = $pts[$b[1]]
        Add-Line $slide ($x + $a[0]*$scale) ($y + $a[1]*$scale) ($x + $c[0]*$scale) ($y + $c[1]*$scale) (RGBColor 80 80 80) $false $false (1.3*$scale) | Out-Null
    }
    foreach ($p in $pts) {
        $fill = RGBColor 242 242 242
        if ($p[2] -eq 1) { $fill = RGBColor 65 125 230 }
        if ($p[2] -eq 2) { $fill = RGBColor 220 55 55 }
        $o = $slide.Shapes.AddShape(9, $x + $p[0]*$scale - 7*$scale, $y + $p[1]*$scale - 7*$scale, 14*$scale, 14*$scale)
        $o.Fill.ForeColor.RGB = $fill
        $o.Line.ForeColor.RGB = RGBColor 65 65 65
        $o.Line.Weight = 0.7
    }
}

function Add-RNA($slide, [double]$x, [double]$y, [double]$scale = 1.0, [int]$color) {
    $pts = @(
        @(0,52), @(22,30), @(45,15), @(72,21), @(93,47), @(116,67), @(145,52), @(170,25), @(198,14)
    )
    for ($i = 0; $i -lt $pts.Count - 1; $i++) {
        Add-Line $slide ($x + $pts[$i][0]*$scale) ($y + $pts[$i][1]*$scale) ($x + $pts[$i+1][0]*$scale) ($y + $pts[$i+1][1]*$scale) $color $false $false (3.2*$scale) | Out-Null
    }
    foreach ($i in @(0,2,4,6,8)) {
        $nt = $slide.Shapes.AddShape(9, $x + $pts[$i][0]*$scale - 8*$scale, $y + $pts[$i][1]*$scale - 8*$scale, 16*$scale, 16*$scale)
        $nt.Fill.ForeColor.RGB = RGBColor 230 255 230
        $nt.Line.ForeColor.RGB = $color
        $nt.Line.Weight = 1.4
    }
}

$blue = RGBColor 28 82 180
$cyan = RGBColor 18 145 160
$orange = RGBColor 242 96 18
$red = RGBColor 212 45 45
$green = RGBColor 30 130 45
$purple = RGBColor 94 65 175
$gray = RGBColor 65 65 65
$lightBlue = RGBColor 240 247 255
$lightCyan = RGBColor 236 253 253
$lightOrange = RGBColor 255 245 235
$lightRed = RGBColor 255 238 238
$lightGreen = RGBColor 239 253 239
$lightPurple = RGBColor 247 242 255

try {
    $pp = [Runtime.InteropServices.Marshal]::GetActiveObject("PowerPoint.Application")
} catch {
    $pp = New-Object -ComObject PowerPoint.Application
}
$pp.Visible = -1
if ($pp.Presentations.Count -eq 0) {
    $pres = $pp.Presentations.Add()
} else {
    $pres = $pp.ActivePresentation
}

$pres.PageSetup.SlideWidth = 1500
$pres.PageSetup.SlideHeight = 1025

for ($i = $pres.Slides.Count; $i -ge 1; $i--) {
    $sOld = $pres.Slides.Item($i)
    foreach ($sh in $sOld.Shapes) {
        if ($sh.Name -eq "__STRUCTRSMA_TARGET_FIGURE1_MARKER") {
            $sOld.Delete()
            break
        }
    }
}

$slide = $pres.Slides.Add($pres.Slides.Count + 1, 12)
$marker = Add-Text $slide "" 0 0 1 1 1 0 $false 1
$marker.Name = "__STRUCTRSMA_TARGET_FIGURE1_MARKER"
$marker.Visible = 0

Add-Text $slide "Figure 1. Overall architecture of StructRSMA." 405 6 690 42 26 0 $true 2 | Out-Null

$panelY = 52; $panelH = 805
$xA = 6; $wA = 270
$xB = 288; $wB = 430
$xC = 730; $wC = 380
$xD = 1122; $wD = 372

foreach ($p in @(@($xA,$wA),@($xB,$wB),@($xC,$wC),@($xD,$wD))) {
    Add-Box $slide "" $p[0] $panelY $p[1] $panelH (RGBColor 35 35 35) (RGBColor 255 255 255) 1 $false 2 $true 1.1 | Out-Null
}

Add-Text $slide "A. Inputs and data sources" 28 61 225 24 13 0 $true 1 | Out-Null
Add-Text $slide "B. DeepRSMA multiview backbone preserved" 310 61 380 24 13 0 $true 1 | Out-Null
Add-Text $slide "C. Contact prediction and`nstructural pretraining" 798 61 250 45 14 0 $true 2 | Out-Null
Add-Text $slide "D. Structural Contact Adapter`nfor affinity prediction" 1185 61 250 45 14 0 $true 2 | Out-Null

# A. inputs
Add-Box $slide "PDB RNA-ligand complexes" 17 91 248 44 $green $lightGreen 16 $true 2 $true 1.3 | Out-Null
Add-RNA $slide 48 156 0.63 $green
Add-Molecule $slide 138 153 0.67
Add-Text $slide "RNA 3D structure +`nligand coordinates" 78 245 120 38 10 $gray $false 2 | Out-Null
Add-Line $slide 138 285 138 315 0 $true $false 1.0 | Out-Null
Add-Text $slide "distance < 4 A" 103 306 95 18 10 $gray $false 2 | Out-Null
Add-Matrix $slide 70 337 6 7 11 $green @(2,10,16,25,31)
Add-Text $slide "Contact map`nC in {0,1}^{p x q}`n`nrows = RNA`nnucleotides`ncolumns = ligand`natoms" 160 335 98 105 9 0 $false 1 | Out-Null
Add-Box $slide "Structure supervision`nfrom PDB complexes" 32 465 205 45 $green $lightGreen 10 $true 2 $true 1.2 | Out-Null

Add-Box $slide "R-SIM affinity dataset" 17 545 248 44 $blue $lightBlue 16 $true 2 $true 1.3 | Out-Null
Add-Text $slide "RNA sequence" 42 607 120 18 10 0 $false 1 | Out-Null
Add-Box $slide "A     U     G     C" 42 630 135 28 $blue (RGBColor 247 251 255) 11 $true 2 $true 1.0 | Out-Null
Add-Text $slide "Ligand SMILES" 42 668 120 18 10 0 $false 1 | Out-Null
Add-Box $slide "SMILES: CCO..." 42 691 160 28 $blue (RGBColor 247 251 255) 10 $false 2 $true 1.0 | Out-Null
Add-Text $slide "pKd" 42 732 55 20 13 0 $true 1 | Out-Null
Add-Box $slide "Affinity supervision`nfrom R-SIM" 33 770 205 46 $blue (RGBColor 246 250 255) 10 $true 2 $true 1.1 | Out-Null

# B. backbone
Add-Text $slide "Original DeepRSMA backbone preserved" 393 88 220 18 10 $blue $true 2 | Out-Null
Add-Text $slide "RNA side" 365 111 100 18 11 $blue $true 2 | Out-Null
Add-Text $slide "Ligand side" 557 111 105 18 11 $orange $true 2 | Out-Null

Add-Box $slide "RNA sequence branch`nRNA sequence`nA   U   G   C`n`nRNA sequence encoder`n`nH_RS" 301 136 178 155 $blue $lightBlue 11 $true 2 $true 1.3 | Out-Null
Add-Box $slide "RNA graph branch`nRNA secondary /`ncontact graph`n`nRNA graph encoder`n`nH_RG" 301 310 178 155 $cyan $lightCyan 11 $true 2 $true 1.3 | Out-Null
Add-Box $slide "Molecule sequence branch`nLigand SMILES`nSMILES: CCO...`n`nMolecule sequence`nencoder`nH_MS" 515 136 178 155 $orange $lightOrange 10.5 $true 2 $true 1.3 | Out-Null
Add-Box $slide "Molecule graph branch`nMolecular`ngraph`n`nMolecule graph`nencoder`nH_MG" 515 310 178 155 $red $lightRed 10.5 $true 2 $true 1.3 | Out-Null
Add-Molecule $slide 564 350 0.52
Add-Line $slide 390 291 390 310 $gray $true $false 1.0 | Out-Null
Add-Line $slide 604 291 604 310 $gray $true $false 1.0 | Out-Null
Add-Line $slide 390 465 390 505 0 $true $false 1.1 | Out-Null
Add-Line $slide 604 465 604 505 0 $true $false 1.1 | Out-Null
Add-Line $slide 390 505 372 515 0 $true $false 1.0 | Out-Null
Add-Line $slide 604 505 625 515 0 $true $false 1.0 | Out-Null
Add-Box $slide "Cross-fusion module`nRNA-to-ligand attention`nLigand-to-RNA attention" 302 520 390 70 $purple (RGBColor 250 247 255) 13 $true 2 $true 1.2 | Out-Null
Add-Line $slide 390 590 390 625 $blue $true $false 1.1 | Out-Null
Add-Line $slide 604 590 604 625 $red $true $false 1.1 | Out-Null
Add-Text $slide "Nucleotide embeddings:`nr_1, r_2, ..., r_p" 322 625 135 48 10 $blue $true 2 | Out-Null
Add-Text $slide "Atom embeddings:`nm_1, m_2, ..., m_q" 545 625 135 48 10 $red $true 2 | Out-Null
Add-Tokens $slide 336 678 4 $blue 12
Add-Tokens $slide 557 678 4 $red 12
Add-Line $slide 390 697 390 735 $blue $true $false 1.0 | Out-Null
Add-Line $slide 604 697 604 735 $red $true $false 1.0 | Out-Null
Add-Box $slide "Pooled view vectors for SCA and affinity prediction" 306 735 380 78 (RGBColor 110 110 110) (RGBColor 255 255 255) 9 $false 2 $true 1.0 | Out-Null
Add-Box $slide "h_RS" 324 775 52 27 $blue (RGBColor 248 251 255) 10 $true 2 $false 1.0 | Out-Null
Add-Box $slide "h_RG" 410 775 52 27 $cyan (RGBColor 248 255 255) 10 $true 2 $false 1.0 | Out-Null
Add-Box $slide "h_MS" 520 775 52 27 $orange (RGBColor 255 249 243) 10 $true 2 $false 1.0 | Out-Null
Add-Box $slide "h_MG" 620 775 52 27 $red (RGBColor 255 246 246) 10 $true 2 $false 1.0 | Out-Null

# C. contact pretraining
Add-Box $slide "Stage 1: Contact pretraining" 794 104 210 42 $green $lightGreen 13 $true 2 $true 1.2 | Out-Null
Add-Text $slide "PDB complexes only`nPDB contact labels available" 815 157 170 48 11 $green $true 2 | Out-Null
Add-Text $slide "Nucleotide`nembeddings r_i`n(i = 1,...,p)" 764 215 92 55 10 $blue $true 2 | Out-Null
Add-Text $slide "Atom embeddings m_j`n(j = 1,...,q)" 951 215 112 45 10 $red $true 2 | Out-Null
Add-Tokens $slide 756 276 4 $blue 13
Add-Tokens $slide 965 276 4 $red 13
Add-Line $slide 815 294 856 335 $blue $true $false 1.2 | Out-Null
Add-Line $slide 1005 294 965 335 $red $true $false 1.2 | Out-Null
Add-Box $slide "Contact prediction head`n`n[r_i, m_j, r_i * m_j]`n`nPairwise MLP`n`nscore_ij" 795 335 220 145 $green (RGBColor 248 255 248) 14 $true 2 $true 1.3 | Out-Null
Add-Line $slide 905 480 905 527 0 $true $false 1.1 | Out-Null
Add-Line $slide 966 480 966 527 0 $true $false 1.1 | Out-Null
Add-Text $slide "Predicted contact map`nP in [0,1]^{p x q}" 755 528 125 42 10 0 $false 2 | Out-Null
Add-Text $slide "PDB-derived`ncontact map C`nC in {0,1}^{p x q}" 950 528 120 54 10 0 $false 2 | Out-Null
Add-Matrix $slide 780 594 6 7 11 $green @(3,10,17,24,31)
Add-Matrix $slide 970 594 6 7 11 $green @(0,8,16,24,32)
Add-Line $slide 826 670 892 713 $green $true $true 1.0 | Out-Null
Add-Line $slide 1016 670 947 713 $green $true $true 1.0 | Out-Null
Add-Box $slide "L_contact =`nFocal Loss(P, C)`n(Focal loss)" 855 712 155 70 (RGBColor 95 95 95) (RGBColor 255 255 255) 12 $true 2 $true 1.1 | Out-Null

# D. SCA
Add-Box $slide "Stage 2: Affinity fine-tuning with SCA" 1158 104 290 42 $purple $lightPurple 13 $true 2 $true 1.2 | Out-Null
Add-Text $slide "1) Pooled view vectors from backbone`nH = [h_RS, h_RG, h_MS, h_MG]" 1148 167 310 42 11 $blue $true 1 | Out-Null
Add-Box $slide "" 1144 214 315 42 $blue (RGBColor 252 254 255) 1 $false 2 $true 1.2 | Out-Null
Add-Tokens $slide 1163 227 6 $blue 12
Add-Tokens $slide 1264 227 6 $cyan 12
Add-Tokens $slide 1365 227 5 $orange 12
Add-Tokens $slide 1446 227 3 $red 11
Add-Line $slide 1305 256 1305 316 0 $true $false 1.0 | Out-Null
Add-Text $slide "2) Contact prior inferred  (no contact labels used)" 1148 302 310 22 11 $green $true 1 | Out-Null
Add-Text $slide "Pretrained contact head`npredicts P on R-SIM pairs`n(no contact supervision)`nP in [0,1]^{p x q}" 1148 333 150 72 9 0 $false 1 | Out-Null
Add-Matrix $slide 1174 424 6 6 9 $green @(4,9,16,17,24)
Add-Line $slide 1265 452 1330 452 0 $true $false 1.1 | Out-Null
Add-Box $slide "Contact prior statistics`nc = [density, maxprob,`nrnafocus, atomfocus]" 1342 358 120 72 $green $lightGreen 9 $true 2 $true 1.2 | Out-Null
Add-Line $slide 1400 430 1400 475 0 $true $false 1.0 | Out-Null
Add-Line $slide 1220 500 1220 520 0 $true $false 1.0 | Out-Null
Add-Box $slide "" 1144 520 315 148 $purple $lightPurple 1 $false 2 $true 1.3 | Out-Null
Add-Text $slide "Structural Contact Adapter (SCA)" 1192 529 222 24 14 $purple $true 2 | Out-Null
Add-Box $slide "1. Contact-prior gate`ng = softmax(MLP_gate([H,c]))" 1164 563 275 30 $purple (RGBColor 253 251 255) 9 $true 2 $true 1.0 | Out-Null
Add-Box $slide "2. View-level attention`nH' = Attention(H,g)" 1164 604 275 30 $purple (RGBColor 253 251 255) 9 $true 2 $true 1.0 | Out-Null
Add-Box $slide "3. Residual correction`nDelta y = MLP_res([z_contact,pool(H'),y_base])" 1164 644 275 30 $purple (RGBColor 253 251 255) 8.6 $true 2 $true 1.0 | Out-Null
Add-Text $slide "Base affinity path`n(preserved backbone)" 1147 679 132 32 9 $blue $true 2 | Out-Null
Add-Text $slide "SCA output`n(residual calibration)" 1342 679 132 32 9 $purple $true 2 | Out-Null
Add-Box $slide "Affinity head`ny_base" 1167 714 105 42 $blue (RGBColor 245 249 255) 10 $true 2 $true 1.0 | Out-Null
Add-Box $slide "Residual correction`nDelta y" 1340 714 105 42 $purple (RGBColor 250 247 255) 10 $true 2 $true 1.0 | Out-Null
Add-Line $slide 1220 756 1288 775 0 $true $false 1.0 | Out-Null
Add-Line $slide 1392 756 1318 775 0 $true $false 1.0 | Out-Null
$plus = $slide.Shapes.AddShape(9, 1288, 761, 28, 28)
$plus.Fill.ForeColor.RGB = RGBColor 255 255 255
$plus.Line.ForeColor.RGB = $gray
$plus.Line.Weight = 1.1
$plus.TextFrame.TextRange.Text = "+"
$plus.TextFrame.TextRange.Font.Name = "Arial"
$plus.TextFrame.TextRange.Font.Size = 18
$plus.TextFrame.TextRange.Font.Bold = -1
$plus.TextFrame.TextRange.ParagraphFormat.Alignment = 2
Add-Box $slide "Final prediction:`ny_hat = y_base + Delta y`nFinal pKd" 1182 799 250 48 0 (RGBColor 255 255 255) 10 $true 2 $true 1.1 | Out-Null

# bottom timeline
Add-Box $slide "Stage 1: PDB contact pretraining`nInput: RNA-ligand complex   |   Output: contact map   |   Loss: L_contact" 8 875 675 70 $green $lightGreen 15 $true 2 $true 1.3 | Out-Null
Add-Line $slide 685 910 725 910 0 $true $false 2.0 | Out-Null
Add-Box $slide "Stage 2: R-SIM affinity fine-tuning with SCA (affinity loss only)`nInput: RNA sequence + ligand SMILES   |   Output: pKd   |   Loss: L_affinity`nNo contact labels required" 730 875 764 70 $blue (RGBColor 246 250 255) 13 $true 2 $true 1.3 | Out-Null
Add-Box $slide "" 190 968 1120 38 (RGBColor 150 150 150) (RGBColor 255 255 255) 1 $false 2 $true 1.0 | Out-Null
Add-Line $slide 230 987 292 987 0 $true $false 1.2 | Out-Null
Add-Text $slide "Solid arrows = normal forward propagation" 304 978 250 18 9 $gray $false 1 | Out-Null
Add-Line $slide 585 987 650 987 $gray $true $true 1.2 | Out-Null
Add-Text $slide "Dashed arrows = supervision / loss signal (Stage 1 only)" 662 978 350 18 9 $gray $false 1 | Out-Null
$lp = $slide.Shapes.AddShape(9, 1045, 977, 20, 20)
$lp.Fill.ForeColor.RGB = RGBColor 255 255 255
$lp.Line.ForeColor.RGB = $gray
$lp.Line.Weight = 1.1
$lp.TextFrame.TextRange.Text = "+"
$lp.TextFrame.TextRange.Font.Name = "Arial"
$lp.TextFrame.TextRange.Font.Size = 12
$lp.TextFrame.TextRange.Font.Bold = -1
$lp.TextFrame.TextRange.ParagraphFormat.Alignment = 2
Add-Text $slide "= residual addition (sum)" 1073 978 190 18 9 $gray $false 1 | Out-Null

$outDir = "D:\shiyan\DeepRSMA\DeepRSMA-master\docs\figures"
if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
$pptxPath = Join-Path $outDir "StructRSMA_Figure1_target_editable.pptx"
$pngPath = Join-Path $outDir "StructRSMA_Figure1_target_ppt.png"
$pdfPath = Join-Path $outDir "StructRSMA_Figure1_target_ppt.pdf"

if ([string]::IsNullOrWhiteSpace($pres.Path)) {
    $pres.SaveAs($pptxPath)
} else {
    $pres.SaveCopyAs($pptxPath)
}
$slide.Export($pngPath, "PNG", 3000, 2050) | Out-Null
$slide.Export($pdfPath, "PDF") | Out-Null
$pp.ActiveWindow.View.GotoSlide($slide.SlideIndex)

Write-Output "Added target-style StructRSMA Figure 1 slide to active PowerPoint."
Write-Output "PPTX copy: $pptxPath"
Write-Output "PNG export: $pngPath"
Write-Output "PDF export: $pdfPath"
