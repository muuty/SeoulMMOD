# SeoulMMOD project page

Project URL: https://muuty.github.io/SeoulMMOD/
Implementation guide: https://muuty.github.io/SeoulMMOD/implementation.html

Buildless HTML served by GitHub Pages from the `review` branch, `/docs` folder.
Edit this folder on `review` and push; GitHub Pages rebuilds the site. No package
installation, build step, external API, or visitor tracking is required.
Open index.html directly, or serve this folder with `python3 -m http.server 8000`.

## Content and provenance

- Benchmark MAE/RMSE: manuscript Table 2 (`tables/main_perf_aggregated.tex`).
- Multi-mode input comparison: manuscript Table 4 and verified saved metrics.
- Sparsity: `rq_results/sparsity_all_models_district_h24.csv`; model-specific
  evaluation-window counts are explicitly stated on the page.
- Cross-year evaluation: manuscript Table 5 (`tables/crossyear.tex`).
- Figure: `figures/intro_od_arc.png`, copied without changing the scientific image.
- `assets/SeoulMMOD-manuscript.pdf`: current camera-ready draft, including
  Appendix C.2 implementation notes. Replace with the submitted final PDF when ready.
- `results.json` and `results.js` contain matching source values.
- `implementation.html` links to pinned source files: SeoulMMOD cc2cc35,
  MPGCN fb041f3, ODCRN edea71a, ODMixer eff16f1. These links describe the code
  inspected for this documentation, not the latest source on a moving branch.

## Research page references

The layout uses the project's own figure, styling, and text. Pages reviewed
for presentation patterns:

- [Nerfies](https://nerfies.github.io/): paper, code, and data links near the title.
- [DROID](https://droid-dataset.github.io/): dataset access, a quickstart, and results.
- [Zip-NeRF](https://jonbarron.info/zipnerf/): concise explanations beside evidence.
- [CityFlow](https://cityflow-project.github.io/): traffic-research framing and setup links.
- [Waymo Open Dataset](https://waymo.com/open/): clear data and benchmark entry points.
- [RT-2](https://robotics-transformer2.github.io/): approach and evaluation sections.

## Verification

Validate both pages, internal anchors, local assets, result-data consistency,
all six scale/horizon states, all three sparsity slices, citation copying,
and desktop/mobile layout before publishing changes. Site JS is buildless.

The existing public runner still averages per-batch wMAPE ratios. Repairing and
validating that code is separate work; the interactive tables here use MAE/RMSE
and do not imply that the wMAPE issue has been fixed.
