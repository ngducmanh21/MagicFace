# Visual verification and AU scoring

The verification output places the source image, edited image and all 12 AU
intensities in one sheet. It works after inference and can also be called from a
future training validation loop. This repository still does not contain a trainer.

## Set up the AU estimator

Keep LibreFace separate from the project's older diffusion dependencies:

```bash
conda create -n magicface-au python=3.9 -y
conda run -n magicface-au python -m pip install -r requirements-au.txt
conda run -n magicface-au python -c "import sys; print(sys.executable)"
```

Use the executable printed by the last command as `--au_python`. Scoring runs in
a child process and defaults to CPU; `--au_device cuda:0` enables GPU scoring.
LibreFace downloads its pretrained weights on first use. `--au_weights_dir`
selects their cache directory. Installation may require CMake; see the
[official LibreFace installation guide](https://github.com/ihp-lab/LibreFace#-installation).

The adapter aligns each face and calls LibreFace's joint AU intensity head.
It records the installed LibreFace version. The optional requirements pin
LibreFace, MediaPipe's legacy solutions API and Pillow's legacy alignment API.
Use images with one face. If alignment or scoring fails, the report shows the
error and leaves the missing values as `N/A`; it never substitutes zero scores.

## Generate and compare several AU edits

From the MagicFace environment:

```bash
python inference.py \
  --img_path test_images/00381.png \
  --bg_path test_images/00381_bg.png \
  --au_test 'AU4+AU1' \
  --AU_variation '0+0' \
  --AU_variation '2+1' \
  --AU_variation '4+2' \
  --seed 424 --inference_steps 50 \
  --saved_path runs/au_review \
  --verify \
  --au_python /absolute/path/to/magicface-au/bin/python
```

Repeated variations automatically enable the report. Every variant uses the same
initial seed, source and background. AU values can be fractional. For negative
combinations, use `--AU_variation=-2+-1` so argparse reads the whole value.
AU names must be supported and unique, with exactly one finite value per name.

Single-edit commands remain supported. Without `--verify`, a single edit saves
the image as before and does not invoke LibreFace.

Outputs under `runs/au_review/00381_verification/`:

| File | Contents |
| --- | --- |
| `report.html` | Local report, label/text filter, unscored filter, full-size image links |
| `grid_001.png`, ... | Sheets with up to four source/result pairs and 12-AU tables each |
| `scores.csv` | One row per result and AU; opens in spreadsheet software |
| `results.json` | Scores, failures, estimator version, seed and model metadata |
| `manifest.json` | Inference inputs/results for rescoring without regeneration |
| `images/` | Copies of the images used by HTML and PNG |

Copy the report directory to share it. Use a different `--saved_path` for each
experiment; rerunning into the same directory replaces matching output filenames.
`--verification_dir` and `--report_title` customize report placement and title.
Every report now includes a portable `manifest.json` whose image paths refer to
the copied `images/` directory. Original paths are retained in `results.json`.

## Interpret the scores

- **Request** is the AU change sent to MagicFace, in model control units.
- **Source / Result** are measured intensities on LibreFace's **0–5** scale.
- **Change** is `result_intensity - source_intensity`.
- **Unchanged AU drift** averages absolute measured changes for AUs requested at
  zero, including omitted AUs. Lower values mean less unintended AU change.
- Requested non-zero AUs are highlighted; requesting zero is a useful baseline.

The released inference interface allows control values outside the estimator's
intensity range. We do not assume that control units equal intensity units or
that their signs are calibrated. By default, expected change and error are `N/A`.
Once you have established a conversion with your checkpoint/data, pass
`--au_delta_scale SCALE`:

```text
expected_delta = requested_delta * SCALE
absolute_error = abs(measured_delta - expected_delta)
edited_au_mae = mean(absolute_error over requested non-zero AUs)
```

The scale can be signed but must be finite and nonzero. Expected absolute
intensities outside 0–5 are flagged rather than clipped. Scoring failures are
excluded from metrics and counted as unscored. Blank CSV cells and JSON `null`
mean unavailable, while numeric zero is a valid measurement. AU estimates are
model predictions, not ground truth or a complete measure of identity/image quality.
See [LibreFace's output format](https://pypi.org/project/libreface/#output-format)
for the intensity and binary detection distinction.

## Build a report from existing results

A manifest groups any number of sources, checkpoints or edits:

```json
{
  "metadata": {"checkpoint": "step_10000"},
  "cases": [
    {
      "source": "images/source.png",
      "result": "images/edited.png",
      "requested_aus": {"AU4": 2, "AU1": 1},
      "label": "step 10000 / AU4 +2, AU1 +1",
      "seed": 424,
      "inference_steps": 50
    }
  ]
}
```

Paths resolve relative to the manifest file; absolute paths also work.

```bash
python verify_results.py --manifest runs/au_review/00381_verification/manifest.json \
  --output_dir runs/rescored --au_python /absolute/path/to/magicface-au/bin/python
```

This command needs Pillow for rendering, but does not import torch/diffusers in
the parent process. It scores each unique image once per report. The child
process keeps estimator seeding and dependencies separate from a training run.

To preview the layout without generating images or installing an estimator:

```bash
python verify_results.py --manifest examples/verification_preview.json \
  --output_dir runs/layout_preview --au_backend none \
  --report_title 'Layout preview - unchanged images, no AU measurements'
```

The example deliberately repeats each source as its result and labels it as a
layout preview. It does not contain generated results or fabricated measurements.

## Call from training validation

Save the validation images, then invoke the same reporting functions on the main
process only (for distributed training, rank zero):

```python
from mgface.verification import score_with_libreface, write_verification_report

# cases uses the manifest schema above, with absolute source/result paths.
paths = [case[key] for case in cases for key in ("source", "result")]
scores = score_with_libreface(paths, python="/path/to/magicface-au/bin/python")
write_verification_report(
    cases, f"runs/validation/step_{global_step:06d}", scores=scores,
    title=f"Validation / step {global_step}",
    metadata={"global_step": global_step},
)
```

Use a fixed validation set, fixed seeds and separate step directories to compare
checkpoints. Passing `scores=None` produces a visual-only report, explicitly marked
unscored. This reporting API does not change model train/eval state.

## Lightweight checks

```bash
python -m unittest discover -s tests -v
```

These tests use controlled score fixtures to verify metric arithmetic, missing
scores, CLI validation, report exports and the isolated worker protocol. They do
not require diffusion checkpoints, LibreFace weights or a GPU.

## Experiment evidence and FER evaluation

Add `--evidence` to either inference or the standalone reporter (it also enables
the report for a single inference edit). It produces
standalone figures and the numeric tables behind them, linked from `report.html`:

```text
runs/experiment_name/
  report.html
  manifest.json
  results.json
  scores.csv
  summary.json
  run_config.json
  artifact_index.json
  grid_001.png
  images/
  figures/
    fig_au_intensity.{png,svg,pdf}
    fig_au_control_response.{png,svg,pdf}
    fig_au_changes_001.{png,svg,pdf}
    fig_fer_source_confusion.{png,svg,pdf}  # when source labels are available
    fig_fer_result_confusion.{png,svg,pdf}  # when result labels are available
  tables/
    au_summary.csv
    au_control_response.csv
    fer_source_per_class.csv              # when labels are available
    fer_source_confusion_counts.csv
    fer_source_confusion_row_normalized.csv
    fer_result_per_class.csv
    fer_result_confusion_counts.csv
    fer_result_confusion_row_normalized.csv
```

The braces above mean separate PNG, SVG and PDF files. `--figure_formats png`
creates only PNG; PNG previews are always included for the HTML. Matplotlib
(already in `requirements.txt`) is required only when evidence export is enabled.
The regular image/score report still works with Pillow alone.

The AU figures show mean source/result intensities, observed response at each
requested control value, and per-edit changes for all 12 AUs. Heatmaps paginate
after 24 edits. Sample counts, unavailable measurements and missing calibration
remain visible. Error bars are **sample standard deviations**, not confidence
intervals; they are undefined for one sample. Combined AU edits are pooled by
AU/control value, so the response plot is descriptive rather than a causal
estimate of an isolated AU's effect. AU means are computed per paired edit case;
a reused source contributes once per edit. No failed measurement is filled with 0.

`summary.json` records sample counts, failure reasons and metric definitions.
`run_config.json` preserves estimator/version, calibration, figure formats and
the supplied model/checkpoint metadata. `artifact_index.json` lists the current
evidence files and their SHA-256 hashes. It covers exported figures/tables and
summary/config, not model weights. Use a fresh output directory per experiment;
the index and HTML identify the current run's artifacts.

### Add FER labels

MagicFace is an image editor; this exporter does **not** train or run a FER
classifier. Supply its predictions and independently verified labels in each
manifest case using the optional `fer` mapping:

```json
{
  "source": "images/source.png",
  "result": "images/edited.png",
  "requested_aus": {"AU6": 2, "AU12": 2},
  "label": "AU6 +2, AU12 +2",
  "fer": {
    "source_true": "neutral",
    "source_pred": "neutral",
    "result_true": "happy",
    "result_pred": "happy"
  }
}
```

These values demonstrate the schema; replace them with your annotations and
classifier predictions. A requested target emotion is **not** ground truth for
the generated image. Leave unknown fields absent or `null`. Use consistent class
names, e.g. do not mix `happy` and `happiness` for the same class. Include the FER
model/version, checkpoint and evaluation split in manifest `metadata` for provenance.

FER evaluation:

- An image needs both true and predicted labels to enter a confusion matrix.
  Incomplete pairs are counted as missing; AU failures do not prevent FER scoring.
- Repeated image paths are deduplicated separately in source/result evaluation.
  A reused source therefore counts once. Conflicting annotations for the same
  path are rejected, including conflicts across source/result roles.
- Matrices use **rows = true labels, columns = predictions**, with a shared label
  set across source/result. Both counts and row-normalized values are exported.
  A row without true support is `N/A` in the normalized matrix, not 0% recall.
- Accuracy, per-class precision/recall/F1 and macro precision/recall/F1 are
  reported. Macro metrics average classes with **true support > 0**. A supported
  class with no predictions contributes 0 to macro precision/F1; its individual
  precision is blank/null because its denominator is zero.
- Source and result supports can differ. The report shows their sample counts
  separately and does not infer an accuracy improvement from unlike datasets.

Without complete FER labels, the report explicitly says FER is unavailable and
does not generate a fabricated confusion matrix.

### Replot previous results without running models

```bash
python verify_results.py \
  --results_json runs/au_review/00381_verification/results.json \
  --output_dir runs/au_review_evidence \
  --evidence
```

`--results_json` and `--manifest` are mutually exclusive. The former reuses stored
AU scores and copied images; it never invokes LibreFace, regardless of AU backend
options. It keeps the saved calibration unless `--au_delta_scale` is supplied.
The latter builds a report from images and runs the chosen AU backend as before.
The same optional `fer` fields can be added to saved result cases before replotting.

For training validation, enable the same export with
`write_verification_report(..., evidence=True, figure_formats=('png', 'pdf'))`.
Choose a separate directory per step/checkpoint. No model train/eval state changes.

A runnable layout-only preview with explicit missing measurements:

```bash
python verify_results.py --manifest examples/verification_preview.json \
  --output_dir runs/evidence_preview --au_backend none \
  --evidence --figure_formats png \
  --report_title 'Layout preview - unchanged images, no measurements'
```
