# SeoulMMOD project page

[Live page](https://muuty.github.io/SeoulMMOD/). GitHub Pages serves `docs/`
from the `review` branch. The site has one HTML page, no JavaScript, and no
build step. Preview this directory with `python3 -m http.server 8000`.

## Component audit — 2026-11-06

The redesign follows existing academic paper pages. Scientific content comes
from the manuscript, rather than new website summaries or interpretations.

| Component | Reference and observed layout | SeoulMMOD decision and source |
|---|---|---|
| Page structure | [Nerfies](https://nerfies.github.io/), [Mip-NeRF 360](https://jonbarron.info/mipnerf360/): one continuous paper page | One `index.html`; remove the separate implementation page. |
| Title | [Nerfies](https://nerfies.github.io/), [DROID](https://droid-dataset.github.io/): centered full paper title | Exact manuscript title from `main.tex`; remove the added slogan and brand mark. |
| Authors and affiliations | [Nerfies](https://nerfies.github.io/), [Zip-NeRF](https://jonbarron.info/zipnerf/): immediately below the title | Names, order, affiliation numbers, and affiliations from `main.tex`; no invented profile links or institutional logos. |
| Conference | [Zip-NeRF](https://jonbarron.info/zipnerf/), [Mip-NeRF 360](https://jonbarron.info/mipnerf360/): venue near the title | NeurIPS 2026 · Evaluations and Datasets Track. |
| Resource links | [Nerfies](https://nerfies.github.io/), [DROID](https://droid-dataset.github.io/): paper/code/data links grouped near authors | Paper, Code, Dataset once at the top. Remove duplicate resource cards and the setup tutorial; setup stays in the repository README. |
| Overview figure | [MathVista](https://mathvista.github.io/), [MMMU](https://mmmu-benchmark.github.io/): benchmark overview before detailed results | Unmodified manuscript Figure 1 (`figures/intro_od_arc.png`) and its original caption (`sections/introduction.tex`). |
| Abstract | [Nerfies](https://nerfies.github.io/), [DROID](https://droid-dataset.github.io/), [VBench](https://vchitect.github.io/VBench-project/): an explicit abstract section | Exact `abstract.tex` text. Remove rewritten dataset pitches, statistic cards, mode badges, and extra claims. |
| Main results | [MathVista](https://mathvista.github.io/): models and evaluation columns displayed together in a full table | Complete manuscript Table 2: both scales, all 14 baselines in paper order, all three horizons, MAE/RMSE/wMAPE, and original cost columns. Preserve precision, bold, underline, and unavailable entries. Remove selectors and the added Run status column. |
| Multi-mode comparison | [DROID](https://droid-dataset.github.io/), [MathVista](https://mathvista.github.io/): separate experiment sections with their evidence | Manuscript Table 4 and its caption, without new percentage claims or Finding boxes. |
| Cross-year comparison | [DROID](https://droid-dataset.github.io/): OOD results shown as an experiment | Complete manuscript Table 5, including all four seasons, the original change column, and caption. |
| Appendix material | The concise paper-first structure of [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) | Implementation explanations stay in Appendix C.2; sparsity analysis stays in Appendix C.5. Remove the website implementation section and sparsity dashboard. |
| Citation | [Nerfies](https://nerfies.github.io/), [MathVista](https://mathvista.github.io/), [VBench](https://vchitect.github.io/VBench-project/): visible BibTeX at the end | Existing citation shown directly in a code block; remove the collapsed panel and copy-button script. |
| Typography and spacing | [Nerfies](https://nerfies.github.io/): Bulma and its academic template stylesheet | Both stylesheets copied byte-for-byte from the pinned Nerfies source below. No custom CSS, inline styles, animations, color palette, or new component library. |
| Narrow screens | Wide scientific tables need to retain their columns | Standard Bulma table containers allow horizontal scrolling, including keyboard scrolling. The page itself remains within the viewport. |
| Navigation and live-service UI | [WILDS](https://wilds.stanford.edu/) is a broader multi-dataset portal; MMMU also has a live leaderboard | Do not import portal navigation, news feeds, leaderboard controls, run monitoring, or dataset-explorer widgets into this paper page. |
| Footer | [Nerfies](https://nerfies.github.io/): template attribution | A single template/license attribution. Funding remains in the paper acknowledgements. |

Eight primary project pages were inspected through HTML/DOM and desktop browser
captures: Nerfies, DROID, Zip-NeRF, Mip-NeRF 360, MMMU, MathVista, VBench, and
WILDS. The MMMU leaderboard displayed its header during inspection; the rendered
MathVista table was used to inspect a populated benchmark table. Page-specific
video galleries, leaderboards, and decorative elements were not copied.

## Manuscript provenance

- Title/authors: `writing/main.tex`; abstract: `writing/abstract.tex`.
- Figure: `writing/figures/intro_od_arc.png` → `assets/od-flows.png`.
- Table 2: `writing/tables/main_perf_aggregated.tex` (28 model rows, 14 columns).
- Table 4: `writing/tables/crossmode_inputs.tex` (4 rows, 5 columns).
- Table 5: `writing/tables/crossyear.tex` (5 rows, 8 columns).
- Table captions: `writing/sections/experiments.tex`.
- PDF: current camera-ready manuscript, including Appendix C.2 implementation
  details. SHA-256:
  `8602e7904e8197bcf9e3bf07a0d84b6ca493e386193997091f42f766e1985fea`.

Tables reproduce the manuscript's reported values; this website revision does
not recompute metrics or modify the benchmark runner. New experiment results
should be incorporated into the paper and page together after validation.

## Template provenance and license

Nerfies source pinned at
[`657409a62d59a93163872c0e4921cf651b987810`](https://github.com/nerfies/nerfies.github.io/tree/657409a62d59a93163872c0e4921cf651b987810).
`static/css/index.css` and `static/css/bulma.min.css` are unmodified copies from
that revision. The HTML adapts its paper-page structure to SeoulMMOD, removes
unused sections, and adds manuscript tables.

The adapted website template is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), following the
[Nerfies website license](https://github.com/nerfies/nerfies.github.io/blob/657409a62d59a93163872c0e4921cf651b987810/README.md#website-license).
Bulma v0.9.1 is MIT licensed; see `static/css/BULMA-LICENSE`.
These template licenses do not change the licenses of the dataset, paper,
scientific figure, or benchmark code.

## Publication checks

Before publishing, verify every displayed table cell and its bold/underline
formatting against the LaTeX sources; check the abstract and captions verbatim.
Check all local assets, the manuscript PDF hash, desktop and 390/320 px layouts,
keyboard-scrollable tables, and absence of page-level horizontal overflow.
Only `index.html` should remain as an HTML page. Check the deployed commit and
live page after GitHub Pages completes its build.

Validation on 2026-11-06 passed: all 452 table cells and their emphasis match
the LaTeX sources; abstract and captions match verbatim; all local assets load.
The PDF and figure are byte-identical to the manuscript files. Chromium checks
passed with JavaScript disabled at 1440, 1024, 390, and 320 px, without page-level
horizontal overflow or failed requests. Desktop, mobile, main-table, and
cross-year screenshots were visually inspected; table keyboard scrolling passed.
