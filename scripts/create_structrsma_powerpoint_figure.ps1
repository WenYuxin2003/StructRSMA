$ErrorActionPreference = "Stop"

function RGBColor([int]$r, [int]$g, [int]$b) {
    return $r + ($g -shl 8) + ($b -shl 16)
}

function Add-TextBox($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h,
    [int]$size = 9, [int]$color = 0, [bool]$bold = $false, [int]$align = 1) {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $shape.TextFrame.MarginLeft = 2
    $shape.TextFrame.MarginRight = 2
    $shape.TextFrame.MarginTop = 1
    $shape.TextFrame.MarginBottom = 1
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.TextRange.Text = $text
    $shape.TextFrame.TextRange.Font.Name = "Arial"
    $shape.TextFrame.TextRange.Font.Size = $size
    $shape.TextFrame.TextRange.Font.Color.RGB = $color
    if ($bold) { $shape.TextFrame.TextRange.Font.Bold = -1 }
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $align
    return $shape
}

function Add-Box($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h,
    [int]$lineColor, [int]$fillColor, [int]$size = 8, [bool]$bold = $false, [int]$align = 2,
    [int]$shapeType = 5) {
    $shape = $slide.Shapes.AddShape($shapeType, $x, $y, $w, $h)
    $shape.Fill.Visible = -1
    $shape.Fill.ForeColor.RGB = $fillColor
    $shape.Line.Visible = -1
    $shape.Line.ForeColor.RGB = $lineColor
    $shape.Line.Weight = 1.1
    $shape.TextFrame.MarginLeft = 3
    $shape.TextFrame.MarginRight = 3
    $shape.TextFrame.MarginTop = 2
    $shape.TextFrame.MarginBottom = 2
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.TextRange.Text = $text
    $shape.TextFrame.TextRange.Font.Name = "Arial"
    $shape.TextFrame.TextRange.Font.Size = $size
    $shape.TextFrame.TextRange.Font.Color.RGB = $lineColor
    if ($bold) { $shape.TextFrame.TextRange.Font.Bold = -1 }
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $align
    return $shape
}

function Add-Line($slide, [double]$x1, [double]$y1, [double]$x2, [double]$y2,
    [int]$color, [bool]$arrow = $true, [bool]$dash = $false, [double]$weight = 1.0) {
    $line = $slide.Shapes.AddConnector(1, $x1, $y1, $x2, $y2)
    $line.Line.ForeColor.RGB = $color
    $line.Line.Weight = $weight
    if ($arrow) { $line.Line.EndArrowheadStyle = 3 }
    if ($dash) { $line.Line.DashStyle = 4 }
    return $line
}

function Add-Matrix($slide, [double]$x, [double]$y, [int]$rows, [int]$cols, [double]$cell,
    [int]$lineColor, [int]$fillColor, [int[]]$activeCells) {
    for ($r = 0; $r -lt $rows; $r++) {
        for ($c = 0; $c -lt $cols; $c++) {
            $idx = $r * $cols + $c
            $fc = RGBColor 255 255 255
            if ($activeCells -contains $idx) { $fc = $fillColor }
            $sq = $slide.Shapes.AddShape(1, $x + $c * $cell, $y + $r * $cell, $cell, $cell)
            $sq.Fill.ForeColor.RGB = $fc
            $sq.Line.ForeColor.RGB = (RGBColor 180 205 180)
            $sq.Line.Weight = 0.35
        }
    }
    for ($r = 0; $r -lt 4; $r++) {
        $dot = $slide.Shapes.AddShape(9, $x - 8, $y + $r * 8 + 3, 3.5, 3.5)
        $dot.Fill.ForeColor.RGB = $lineColor
        $dot.Line.Visible = 0
    }
    for ($c = 0; $c -lt 4; $c++) {
        $dot = $slide.Shapes.AddShape(9, $x + $c * 8 + 4, $y + $rows * $cell + 6, 3.5, 3.5)
        $dot.Fill.ForeColor.RGB = $lineColor
        $dot.Line.Visible = 0
    }
}

function Add-TokenBar($slide, [double]$x, [double]$y, [int]$n, [int]$color) {
    for ($i = 0; $i -lt $n; $i++) {
        $sq = $slide.Shapes.AddShape(1, $x + $i * 10, $y, 7, 7)
        $sq.Fill.ForeColor.RGB = $color
        $sq.Line.ForeColor.RGB = $color
    }
}

function Add-MoleculeIcon($slide, [double]$x, [double]$y) {
    $coords = @(
        @(0, 18, 0), @(16, 7, 0), @(32, 16, 1), @(49, 6, 0), @(62, 20, 2), @(42, 33, 0), @(22, 33, 0)
    )
    $bonds = @(@(0,1), @(1,2), @(2,3), @(3,4), @(2,5), @(5,6), @(6,0))
    foreach ($b in $bonds) {
        $a = $coords[$b[0]]
        $c = $coords[$b[1]]
        Add-Line $slide ($x + $a[0]) ($y + $a[1]) ($x + $c[0]) ($y + $c[1]) (RGBColor 90 90 90) $false $false 1.1 | Out-Null
    }
    foreach ($p in $coords) {
        $atomColor = RGBColor 245 245 245
        if ($p[2] -eq 1) { $atomColor = RGBColor 55 125 240 }
        if ($p[2] -eq 2) { $atomColor = RGBColor 230 55 55 }
        $o = $slide.Shapes.AddShape(9, $x + $p[0] - 4, $y + $p[1] - 4, 8, 8)
        $o.Fill.ForeColor.RGB = $atomColor
        $o.Line.ForeColor.RGB = RGBColor 70 70 70
        $o.Line.Weight = 0.6
    }
}

function Add-RNAIcon($slide, [double]$x, [double]$y, [int]$color) {
    $pts = @(
        @(0,30), @(12,18), @(25,11), @(38,15), @(50,29), @(62,39), @(74,32), @(88,17), @(100,12)
    )
    for ($i = 0; $i -lt $pts.Count - 1; $i++) {
        Add-Line $slide ($x + $pts[$i][0]) ($y + $pts[$i][1]) ($x + $pts[$i+1][0]) ($y + $pts[$i+1][1]) $color $false $false 2.1 | Out-Null
    }
    for ($i = 0; $i -lt $pts.Count; $i += 2) {
        $nt = $slide.Shapes.AddShape(9, $x + $pts[$i][0] - 4, $y + $pts[$i][1] - 4, 8, 8)
        $nt.Fill.ForeColor.RGB = RGBColor 230 255 230
        $nt.Line.ForeColor.RGB = $color
        $nt.Line.Weight = 1.0
    }
}

$blue = RGBColor 30 82 180
$cyan = RGBColor 20 145 160
$orange = RGBColor 240 100 20
$red = RGBColor 210 45 45
$green = RGBColor 30 130 45
$purple = RGBColor 100 65 180
$gray = RGBColor 70 70 70
$lightBlue = RGBColor 238 245 255
$lightCyan = RGBColor 235 253 253
$lightOrange = RGBColor 255 246 235
$lightRed = RGBColor 255 239 239
$lightGreen = RGBColor 238 252 238
$lightPurple = RGBColor 246 241 255

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

$pres.PageSetup.SlideWidth = 960
$pres.PageSetup.SlideHeight = 660

# Remove an earlier generated copy if present.
for ($i = $pres.Slides.Count; $i -ge 1; $i--) {
    $slideOld = $pres.Slides.Item($i)
    foreach ($shape in $slideOld.Shapes) {
        if ($shape.Name -eq "__STRUCTRSMA_FIGURE1_MARKER") {
            $slideOld.Delete()
            break
        }
    }
}

$slide = $pres.Slides.Add($pres.Slides.Count + 1, 12)
$marker = Add-TextBox $slide "" 0 0 1 1 1 0 $false 1
$marker.Name = "__STRUCTRSMA_FIGURE1_MARKER"
$marker.Visible = 0

Add-TextBox $slide "Figure 1. Overall architecture of StructRSMA." 230 5 500 26 18 0 $true 2 | Out-Null

$panelY = 40
$panelH = 510
$xA = 8; $wA = 170
$xB = 185; $wB = 270
$xC = 462; $wC = 240
$xD = 710; $wD = 242

Add-Box $slide "" $xA $panelY $wA $panelH (RGBColor 35 35 35) (RGBColor 255 255 255) 1 $false 2 1 | Out-Null
Add-Box $slide "" $xB $panelY $wB $panelH (RGBColor 35 35 35) (RGBColor 255 255 255) 1 $false 2 1 | Out-Null
Add-Box $slide "" $xC $panelY $wC $panelH (RGBColor 35 35 35) (RGBColor 255 255 255) 1 $false 2 1 | Out-Null
Add-Box $slide "" $xD $panelY $wD $panelH (RGBColor 35 35 35) (RGBColor 255 255 255) 1 $false 2 1 | Out-Null

Add-TextBox $slide "A. Inputs and data sources" ($xA + 5) 47 ($wA - 10) 18 10 0 $true 1 | Out-Null
Add-TextBox $slide "B. DeepRSMA multiview backbone preserved" ($xB + 5) 47 ($wB - 10) 18 10 0 $true 1 | Out-Null
Add-TextBox $slide "C. Contact prediction and`nstructural pretraining" ($xC + 5) 47 ($wC - 10) 30 10 0 $true 2 | Out-Null
Add-TextBox $slide "D. Structural Contact Adapter`nfor affinity prediction" ($xD + 5) 47 ($wD - 10) 30 10 0 $true 2 | Out-Null

# Panel A
Add-Box $slide "PDB RNA-ligand complexes" ($xA + 9) 76 ($wA - 18) 26 $green $lightGreen 10 $true 2 | Out-Null
Add-RNAIcon $slide ($xA + 32) 112 $green
Add-MoleculeIcon $slide ($xA + 93) 118
Add-TextBox $slide "RNA 3D structure + ligand coordinates" ($xA + 28) 163 115 25 8 $gray $false 2 | Out-Null
Add-Line $slide ($xA + 85) 189 ($xA + 85) 205 $gray $true $false 0.9 | Out-Null
Add-TextBox $slide "distance < 4 A" ($xA + 48) 197 75 14 8 $gray $false 2 | Out-Null
Add-Matrix $slide ($xA + 45) 217 5 6 8 $green (RGBColor 96 172 96) @(1,8,14,21,27)
Add-TextBox $slide "Contact map C in {0,1}^{p x q}`nrows = RNA nucleotides`ncolumns = ligand atoms" ($xA + 96) 216 68 56 7 $gray $false 1 | Out-Null
Add-Box $slide "Structure supervision`nfrom PDB complexes" ($xA + 18) 296 132 36 $green $lightGreen 8 $true 2 | Out-Null

Add-Box $slide "R-SIM affinity dataset" ($xA + 9) 352 ($wA - 18) 24 $blue $lightBlue 10 $true 2 | Out-Null
Add-TextBox $slide "RNA sequence" ($xA + 26) 384 80 12 8 0 $false 1 | Out-Null
Add-Box $slide "A   U   G   C" ($xA + 28) 399 86 18 $blue (RGBColor 246 250 255) 8 $true 2 | Out-Null
Add-TextBox $slide "Ligand SMILES" ($xA + 26) 421 80 12 8 0 $false 1 | Out-Null
Add-Box $slide "SMILES: CCO..." ($xA + 28) 436 98 18 $blue (RGBColor 246 250 255) 8 $false 2 | Out-Null
Add-TextBox $slide "pKd" ($xA + 28) 459 40 14 10 0 $true 1 | Out-Null
Add-Box $slide "Affinity supervision`nfrom R-SIM" ($xA + 25) 482 122 34 $blue (RGBColor 244 248 255) 8 $true 2 | Out-Null

# Panel B branches
Add-TextBox $slide "Original DeepRSMA backbone preserved" ($xB + 70) 68 140 14 8 $blue $true 2 | Out-Null
Add-TextBox $slide "RNA side" ($xB + 58) 85 75 12 8 $blue $true 2 | Out-Null
Add-TextBox $slide "Ligand side" ($xB + 175) 85 75 12 8 $orange $true 2 | Out-Null

Add-Box $slide "RNA sequence branch`nRNA sequence`nA   U   G   C`nRNA sequence encoder`nH_RS" ($xB + 18) 103 115 92 $blue $lightBlue 8 $true 2 | Out-Null
Add-Box $slide "RNA graph branch`nRNA secondary/contact graph`nRNA graph encoder`nH_RG" ($xB + 18) 204 115 92 $cyan $lightCyan 8 $true 2 | Out-Null
Add-Box $slide "Molecule sequence branch`nLigand SMILES`nSMILES: CCO...`nMolecule sequence encoder`nH_MS" ($xB + 151) 103 115 92 $orange $lightOrange 8 $true 2 | Out-Null
Add-Box $slide "Molecule graph branch`nMolecular graph`nMolecule graph encoder`nH_MG" ($xB + 151) 204 115 92 $red $lightRed 8 $true 2 | Out-Null
Add-MoleculeIcon $slide ($xB + 185) 227

Add-Line $slide ($xB + 76) 195 ($xB + 76) 302 $gray $true $false 0.9 | Out-Null
Add-Line $slide ($xB + 209) 195 ($xB + 209) 302 $gray $true $false 0.9 | Out-Null
Add-Line $slide ($xB + 76) 296 ($xB + 90) 318 $gray $true $false 0.9 | Out-Null
Add-Line $slide ($xB + 209) 296 ($xB + 195) 318 $gray $true $false 0.9 | Out-Null
Add-Box $slide "Cross-fusion module`nRNA-to-ligand attention`nLigand-to-RNA attention" ($xB + 18) 318 248 48 $purple (RGBColor 250 247 255) 9 $true 2 | Out-Null
Add-Line $slide ($xB + 80) 366 ($xB + 80) 386 $blue $true $false 0.9 | Out-Null
Add-Line $slide ($xB + 205) 366 ($xB + 205) 386 $red $true $false 0.9 | Out-Null
Add-TextBox $slide "Token-level embeddings for contact prediction" ($xB + 25) 384 105 23 7 $blue $true 2 | Out-Null
Add-TextBox $slide "Token-level embeddings for contact prediction" ($xB + 150) 384 105 23 7 $red $true 2 | Out-Null
Add-TextBox $slide "Nucleotide embeddings r_i`nr_1, r_2, ..., r_p" ($xB + 28) 407 100 24 7 $blue $false 2 | Out-Null
Add-TextBox $slide "Atom embeddings m_j`nm_1, m_2, ..., m_q" ($xB + 153) 407 100 24 7 $red $false 2 | Out-Null
Add-TokenBar $slide ($xB + 45) 433 5 $blue
Add-TokenBar $slide ($xB + 170) 433 5 $red
Add-Line $slide ($xB + 80) 443 ($xB + 80) 465 $blue $true $false 0.8 | Out-Null
Add-Line $slide ($xB + 205) 443 ($xB + 205) 465 $red $true $false 0.8 | Out-Null
Add-Box $slide "Pooled view vectors for SCA and affinity prediction" ($xB + 22) 464 240 44 (RGBColor 105 105 105) (RGBColor 252 252 252) 7 $false 2 | Out-Null
Add-Box $slide "h_RS" ($xB + 35) 488 42 18 $blue (RGBColor 246 250 255) 8 $true 2 1 | Out-Null
Add-Box $slide "h_RG" ($xB + 93) 488 42 18 $cyan (RGBColor 245 255 255) 8 $true 2 1 | Out-Null
Add-Box $slide "h_MS" ($xB + 151) 488 42 18 $orange (RGBColor 255 248 240) 8 $true 2 1 | Out-Null
Add-Box $slide "h_MG" ($xB + 209) 488 42 18 $red (RGBColor 255 244 244) 8 $true 2 1 | Out-Null

# Panel C
Add-Box $slide "Stage 1: Contact pretraining" ($xC + 58) 82 130 24 $green $lightGreen 9 $true 2 | Out-Null
Add-TextBox $slide "PDB complexes only`nPDB-derived distance-defined contact labels" ($xC + 38) 110 170 30 8 $green $true 2 | Out-Null
Add-TextBox $slide "Nucleotide`nembeddings r_i`n(i = 1,...,p)" ($xC + 28) 151 75 38 7 $blue $true 2 | Out-Null
Add-TextBox $slide "Atom embeddings m_j`n(from graph / cross-fused atom stream)`n(j = 1,...,q)" ($xC + 137) 151 88 46 7 $red $true 2 | Out-Null
Add-TokenBar $slide ($xC + 44) 200 4 $blue
Add-TokenBar $slide ($xC + 150) 200 4 $red
Add-Line $slide ($xC + 65) 210 ($xC + 105) 238 $blue $true $false 1.0 | Out-Null
Add-Line $slide ($xC + 170) 210 ($xC + 132) 238 $red $true $false 1.0 | Out-Null
Add-Box $slide "Contact prediction head`n[r_i, m_j, r_i * m_j]`n`nPairwise MLP`n`nscore_ij" ($xC + 50) 237 145 115 $green (RGBColor 248 255 248) 9 $true 2 | Out-Null
Add-Line $slide ($xC + 105) 352 ($xC + 105) 378 $gray $true $false 0.9 | Out-Null
Add-Line $slide ($xC + 145) 352 ($xC + 145) 378 $gray $true $false 0.9 | Out-Null
Add-TextBox $slide "Predicted contact`nprobability map P`nP in [0,1]^{p x q}" ($xC + 22) 372 95 38 7 $gray $false 2 | Out-Null
Add-TextBox $slide "PDB-derived`ncontact map C`nC in {0,1}^{p x q}" ($xC + 135) 372 88 38 7 $gray $false 2 | Out-Null
Add-Matrix $slide ($xC + 42) 414 5 5 8 $green (RGBColor 96 172 96) @(1,5,12,18,22)
Add-Matrix $slide ($xC + 156) 414 5 5 8 $green (RGBColor 96 172 96) @(0,6,12,18,24)
Add-Line $slide ($xC + 65) 458 ($xC + 105) 476 $green $true $true 0.9 | Out-Null
Add-Line $slide ($xC + 180) 458 ($xC + 138) 476 $green $true $true 0.9 | Out-Null
Add-Box $slide "L_contact =`nFocal Loss(P, C)" ($xC + 70) 474 103 45 (RGBColor 90 90 90) (RGBColor 255 255 255) 10 $true 2 | Out-Null

# Panel D
Add-Box $slide "Stage 2: Affinity fine-tuning with SCA" ($xD + 35) 82 172 24 $purple $lightPurple 9 $true 2 | Out-Null
Add-TextBox $slide "1) Pooled view vectors from backbone`nH = [h_RS, h_RG, h_MS, h_MG]" ($xD + 25) 116 190 30 8 $blue $true 1 | Out-Null
Add-Box $slide "" ($xD + 22) 151 198 28 $blue (RGBColor 252 254 255) 1 $false 2 | Out-Null
Add-TokenBar $slide ($xD + 35) 161 5 $blue
Add-TokenBar $slide ($xD + 85) 161 5 $cyan
Add-TokenBar $slide ($xD + 136) 161 5 $orange
Add-TokenBar $slide ($xD + 188) 161 4 $red
Add-TextBox $slide "2) Contact prior inferred (no contact labels used)" ($xD + 25) 190 194 16 8 $green $true 1 | Out-Null
Add-TextBox $slide "Pretrained contact head predicts P on R-SIM pairs`n(no contact supervision)`nP in [0,1]^{p x q}" ($xD + 24) 211 105 44 7 $gray $false 1 | Out-Null
Add-Matrix $slide ($xD + 38) 260 5 5 7 $green (RGBColor 96 172 96) @(2,6,12,13,20)
Add-Line $slide ($xD + 88) 276 ($xD + 135) 276 $gray $true $false 0.9 | Out-Null
Add-Box $slide "Contact prior statistics`nc = [density, maxprob,`nrnafocus, atomfocus]" ($xD + 140) 234 80 58 $green $lightGreen 7 $true 2 | Out-Null
Add-Line $slide ($xD + 180) 292 ($xD + 180) 315 $gray $true $false 0.9 | Out-Null
Add-Line $slide ($xD + 92) 180 ($xD + 92) 315 $gray $true $false 0.9 | Out-Null
Add-Box $slide "" ($xD + 24) 319 194 110 $purple $lightPurple 1 $false 2 | Out-Null
Add-TextBox $slide "Structural Contact Adapter (SCA)" ($xD + 38) 326 168 16 10 $purple $true 2 | Out-Null
Add-Box $slide "1. Contact-prior gate`ng = softmax(MLP_gate([H,c]))" ($xD + 34) 348 174 24 $purple (RGBColor 252 250 255) 7 $true 2 | Out-Null
Add-Box $slide "2. Contact-gated view attention`nH' = Attention(H,g)" ($xD + 34) 379 174 24 $purple (RGBColor 252 250 255) 7 $true 2 | Out-Null
Add-Box $slide "3. Residual correction`nDelta y = MLP_res([z_contact,pool(H'),y_base])" ($xD + 34) 410 174 25 $purple (RGBColor 252 250 255) 7 $true 2 | Out-Null
Add-TextBox $slide "Base affinity path`n(preserved backbone)" ($xD + 24) 438 80 26 7 $blue $true 2 | Out-Null
Add-Box $slide "Affinity head`ny_base" ($xD + 38) 463 75 32 $blue (RGBColor 244 248 255) 8 $true 2 | Out-Null
Add-TextBox $slide "SCA output`n(residual calibration)" ($xD + 145) 438 80 26 7 $purple $true 2 | Out-Null
Add-Box $slide "Residual correction`nDelta y" ($xD + 146) 463 74 32 $purple (RGBColor 250 247 255) 8 $true 2 | Out-Null
Add-Line $slide ($xD + 76) 495 ($xD + 117) 501 $gray $true $false 0.9 | Out-Null
Add-Line $slide ($xD + 183) 495 ($xD + 140) 501 $gray $true $false 0.9 | Out-Null
$plus = $slide.Shapes.AddShape(9, $xD + 118, 494, 18, 18)
$plus.Fill.ForeColor.RGB = RGBColor 255 255 255
$plus.Line.ForeColor.RGB = $gray
$plus.TextFrame.TextRange.Text = "+"
$plus.TextFrame.TextRange.Font.Name = "Arial"
$plus.TextFrame.TextRange.Font.Size = 11
$plus.TextFrame.TextRange.Font.Bold = -1
$plus.TextFrame.TextRange.ParagraphFormat.Alignment = 2
Add-Box $slide "Final prediction:`ny_hat = y_base + Delta y`nFinal pKd" ($xD + 35) 511 178 36 0 (RGBColor 255 255 255) 8 $true 2 | Out-Null

# Bottom timeline and legend
Add-Box $slide "Stage 1: PDB contact pretraining`nInput: RNA-ligand complex   |   Output: contact map   |   Loss: L_contact" 8 566 430 50 $green $lightGreen 11 $true 2 | Out-Null
Add-Line $slide 440 591 478 591 0 $true $false 1.6 | Out-Null
Add-Box $slide "Stage 2: R-SIM affinity fine-tuning with SCA (affinity loss only)`nInput: RNA sequence + ligand SMILES   |   Output: pKd   |   Loss: L_affinity`nNo contact labels required" 480 566 472 50 $blue (RGBColor 245 248 255) 10 $true 2 | Out-Null
Add-Box $slide "" 120 626 720 22 (RGBColor 145 145 145) (RGBColor 255 255 255) 1 $false 2 | Out-Null
Add-Line $slide 160 637 215 637 0 $true $false 1.2 | Out-Null
Add-TextBox $slide "Solid arrows = normal forward propagation" 222 629 170 15 7 $gray $false 1 | Out-Null
Add-Line $slide 410 637 465 637 $gray $true $true 1.2 | Out-Null
Add-TextBox $slide "Dashed arrows = supervision / loss signal (Stage 1 only)" 472 629 245 15 7 $gray $false 1 | Out-Null
$legendPlus = $slide.Shapes.AddShape(9, 720, 629, 15, 15)
$legendPlus.Fill.ForeColor.RGB = RGBColor 255 255 255
$legendPlus.Line.ForeColor.RGB = $gray
$legendPlus.TextFrame.TextRange.Text = "+"
$legendPlus.TextFrame.TextRange.Font.Bold = -1
$legendPlus.TextFrame.TextRange.Font.Size = 9
$legendPlus.TextFrame.TextRange.ParagraphFormat.Alignment = 2
Add-TextBox $slide "= residual addition (sum)" 738 629 120 15 7 $gray $false 1 | Out-Null

# Save a copy and export the generated slide.
$outDir = "D:\shiyan\DeepRSMA\DeepRSMA-master\docs\figures"
if (!(Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
$pptxPath = Join-Path $outDir "StructRSMA_Figure1_editable.pptx"
$pngPath = Join-Path $outDir "StructRSMA_Figure1_ppt.png"
$pdfPath = Join-Path $outDir "StructRSMA_Figure1_ppt.pdf"

if ([string]::IsNullOrWhiteSpace($pres.Path)) {
    $pres.SaveAs($pptxPath)
} else {
    $pres.SaveCopyAs($pptxPath)
}
$slide.Export($pngPath, "PNG", 1920, 1320) | Out-Null
$slide.Export($pdfPath, "PDF") | Out-Null

$pp.ActiveWindow.View.GotoSlide($slide.SlideIndex)

Write-Output "Added StructRSMA Figure 1 slide to active PowerPoint."
Write-Output "PPTX copy: $pptxPath"
Write-Output "PNG export: $pngPath"
Write-Output "PDF export: $pdfPath"
