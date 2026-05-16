Create a clean, minimalist infographic diagram of a neural network architecture called "ape-MPNN" for molecular property prediction.

Style: Flat design, clean vector illustration. White background. Limited color palette: soft blue (#4A90D9), soft teal (#2ECC96), soft coral (#E8737A), soft amber (#F5A623), and light gray (#E8ECF0). No gradients, no shadows, no textures. Thin clean lines, generous white space, modern sans-serif typography. Think of a high-quality tech blog illustration or a clean Notion/Linear style diagram.

Layout: Horizontal data flow, left to right. Clean rounded rectangles connected by thin arrows.

Components (left to right):

LEFT INPUT SECTION:
- A clean simplified 3D molecular structure icon (ball-and-stick, 4-5 atoms, minimal). Thin gray lines for bonds, small colored circles for atoms.
- Small label below: "SDF分子输入" in clean 10pt gray text
- Sub-label: "原子特征 18d + 键特征 7d + 3D坐标"

ENCODERS SECTION (three parallel paths merging):
Three small clean rounded rectangles stacked vertically, connected by thin lines to a merge point:

Top box: Soft blue fill, white text "Atom Encoder" with a small atom icon (circle with orbits). Label: "18→256d"
Middle box: Soft teal fill, white text "Bond Encoder + RBF" with a small bond icon (two circles connected). Label: "7→256d"
Bottom box: Soft coral fill, white text "3D Geometry" with a small compass/ruler icon. Label: "ACSF + Direction →256d"

A small "+" circle at the merge point, then a single arrow pointing right.

CENTER SECTION - LSTM Message Passing (largest, most prominent):
A tall rounded rectangle in soft blue (#4A90D9), containing 5 smaller rounded rectangles stacked vertically inside it, each representing one LSTM layer. The 5 inner boxes are lighter blue (#A8D4F0) with a small circular arrow icon (↻) in each.

Label to the left of this block: "LSTM消息传递" in clean bold text
Label below: "× 5 layers | 256d hidden"
A thin bracket on the right side with text: "~3.8M params"

Right arrow pointing to the pooling section.

POOLING SECTION:
A medium rounded rectangle containing three small icons in a row:
- A small target/bullseye icon → "Attention"
- A small book/pages icon → "Set2Set"
- A small horizontal bar icon → "Mean Pool"

Label below: "多级注意力池化"
A merge arrow pointing right.

OUTPUT SECTION - Multi-Task Head:
A clean branching structure: one input line splits into 12 thin output lines, each ending in a small colored dot.
- 4 dots in gold (G, U, U0, H) — labeled "R²>0.99"
- 5 dots in blue (zpve, Cv, gap, HOMO, alpha, LUMO) — labeled "R²>0.95"
- 2 dots in green (μ, R²) — labeled "R²>0.85"

Right side label: "12量子化学性质"

TOP: A clean thin horizontal line with centered title "Champion Model · ape-MPNN" in modern sans-serif, dark gray (#333).
BOTTOM-RIGHT: Small badge "5.77M params" in a rounded pill shape, light gray fill.

Overall feel: Like a clean, professional infographic you'd see in a well-designed tech documentation site. Minimalist, geometric, lots of breathing room. No characters, no sparkles, no decoration.
