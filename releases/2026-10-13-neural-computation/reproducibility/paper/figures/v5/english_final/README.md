# English Main Figures

These six figures are English-label versions of the corresponding SVG files in
`../chinese_final/`. They are used by the illustrated English main manuscript.

## What Changed

- Chinese labels were translated using the manuscript's terminology glossary.
- Labels use Arial/Helvetica, with size adjusted only to fit the existing panels.
- The two cohort labels in Figure 2 are shifted clear of the reference line and
  given a white text outline. Data points, interval endpoints, axes, and
  reference lines were not moved.
- The old undefined `B` axis shorthand is spelled out as normalized
  post-rewiring return. `J24` is retained.

All non-text scientific SVG geometry is compared with its Chinese source during
generation. No experiments or statistical analyses were rerun to translate the
figures. Values and the layout of each panel are unchanged.

## Reproduction

The generator is `paper/analysis/translate_v5_figures_en.cjs`. `manifest.json`
records the source and output hashes, label translations, and geometry checks.
SVG files are editable; the 1800-pixel-wide PNG files are used in the PDF.
