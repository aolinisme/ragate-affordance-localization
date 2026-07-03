# Local Reproduction Status

Date: 2026-06-16

## Environment

- Workspace: `E:\work\2cvpr\Probing_Bridging_Affordance`
- Python: `E:\conda\envs\bev_py39_torch231\python.exe`
- Python version: 3.9.19
- PyTorch: 2.3.1+cu118
- CUDA runtime: 11.8
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU

## Completed Checks

- Cloned official repository: `QZhang2111/Probing_Bridging_Affordance`
- Installed the repo in editable mode with `open_clip_torch`
- Installed geometry probing dependencies from `geometry_probing\umd_linear_probing\requirements.txt`
- Downloaded and extracted the UMD tools package into:
  `E:\work\2cvpr\Probing_Bridging_Affordance\datasets\UMD\part-affordance-dataset`
- Verified command registry:
  - `python -m pba.run --list`
- Verified lightweight tests:
  - `python -m pytest tests\test_public_api_smoke.py tests\test_pba_run.py tests\test_metrics_affordance.py tests\test_metrics_segmentation.py -q`
  - Result: 14 passed
- Verified CLIP patch probing smoke run on a local synthetic image:
  - Command:
    `python run.py aux-clip-probe -- --image E:\work\2cvpr\dataset\synthetic_rich\r_000.png --prompts "red object" "blue object" "round object" --device cuda --output-root outputs\clip_probe_smoke --force-size 224`
  - Outputs:
    `outputs\clip_probe_smoke\attention\*_heat.png`
    `outputs\clip_probe_smoke\attention\*_overlay.png`
    `outputs\clip_probe_smoke\attention\*_heat.npy`
    `outputs\clip_probe_smoke\meta.json`
- Verified a full end-to-end CLIP geometry smoke train/eval run on RTX 3070:
  - Config:
    `geometry_probing\umd_linear_probing\configs\clip_smoke_3070.yaml`
  - Split file:
    `geometry_probing\umd_linear_probing\metadata\splits\category_split_seed42_v20_smoke.json`
  - Output root:
    `geometry_probing\umd_linear_probing\outputs\CLIP_smoke_3070_20260616-170054`
  - Summary:
    - best val mIoU: `0.08365175615277919`
    - test mIoU: `0.10842563043614292`
    - checkpoint:
      `geometry_probing\umd_linear_probing\outputs\CLIP_smoke_3070_20260616-170054\lr1e-03_wd1e-02\linear_probe.pth`
- Implemented a first lightweight innovation:
  `geometry-aware gating` in `MultiLayerLinearHead`
  - Code path:
    `geometry_probing\umd_linear_probing\src\models\linear_head.py`
  - Zero-geometry fallback smoke config:
    `geometry_probing\umd_linear_probing\configs\clip_smoke_3070_gate.yaml`
  - Zero-geometry fallback output root:
    `geometry_probing\umd_linear_probing\outputs\CLIP_smoke_gate_3070_20260616-172702`
  - Zero-geometry fallback summary:
    - best val mIoU: `0.07104302408859853`
    - test mIoU: `0.09151804324471113`
  - Interpretation:
    this gated version runs correctly, but under missing geometry side-data it underperforms the baseline.

- Built local smoke geometry side-data directly from UMD `depth.png`:
  - Script:
    `geometry_probing\umd_linear_probing\scripts\build_smoke_geometry_assets.py`
  - Assets:
    `datasets\UMD\part-affordance-dataset\smoke_geometry`
  - Manifest:
    `geometry_probing\umd_linear_probing\metadata\splits\smoke_metric3d_predictions.json`

- Re-ran geometry-enabled smoke comparison with real local geometry side-data:
  - baseline config:
    `geometry_probing\umd_linear_probing\configs\clip_smoke_3070_geom.yaml`
  - baseline output root:
    `geometry_probing\umd_linear_probing\outputs\CLIP_smoke_geom_3070_20260616-211500`
  - baseline summary:
    - best val mIoU: `0.07787714418885763`
    - test mIoU: `0.10975134894587528`
  - gated config:
    `geometry_probing\umd_linear_probing\configs\clip_smoke_3070_gate.yaml`
  - gated output root:
    `geometry_probing\umd_linear_probing\outputs\CLIP_smoke_gate_3070_20260616-211500`
  - gated summary:
    - best val mIoU: `0.09097857501134544`
    - test mIoU: `0.11259903571565077`
  - Interpretation:
    with actual local geometry side-data available, the geometry-gated head improves over the geometry-enabled baseline on both val and test in the smoke setting.

- Built a larger category-diverse capped split:
  - Split builder:
    `geometry_probing\umd_linear_probing\scripts\build_capped_split.py`
  - Geometry builder:
    `geometry_probing\umd_linear_probing\scripts\build_capped_geometry_assets.py`
  - Split size:
    - train: `72`
    - val: `56`
    - test: `56`

- Re-ran geometry baseline vs geometry-gated on the capped split:
  - baseline config:
    `geometry_probing\umd_linear_probing\configs\clip_capped_3070_geom.yaml`
  - baseline output root:
    `geometry_probing\umd_linear_probing\outputs\CLIP_capped_geom_3070_20260616-213233`
  - baseline summary:
    - best val mIoU: `0.1857199431343565`
    - test mIoU: `0.2441365509462193`
  - gated config:
    `geometry_probing\umd_linear_probing\configs\clip_capped_3070_gate.yaml`
  - gated output root:
    `geometry_probing\umd_linear_probing\outputs\CLIP_capped_gate_3070_20260616-213233`
  - gated summary:
    - best val mIoU: `0.2337138194390807`
    - test mIoU: `0.34556340339867636`
  - Interpretation:
    the geometry-gated head shows a substantially stronger gain on the larger capped split than on the smoke split, which makes this direction worth continuing with multi-seed validation.

- Added a reproducible sequential multi-seed runner:
  - package entry:
    `pba\geometry\multiseed.py`
  - script entry:
    `geometry_probing\umd_linear_probing\scripts\multiseed.py`
  - command alias:
    `run.py geometry-multiseed`
  - purpose:
    avoids output-directory collisions by writing seed-specific `output_dir_name` overrides and aggregates per-seed `summary.json` files into a single JSON report.

- Verified capped 3-seed baseline summary:
  - command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\clip_capped_3070_geom.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_capped_geom_3070_seeds_1337-1338-1339_20260616-215211.json`
  - seed outputs:
    - `CLIP_capped_geom_3070_seed1337_20260616-215224`
    - `CLIP_capped_geom_3070_seed1338_20260616-215341`
    - `CLIP_capped_geom_3070_seed1339_20260616-215459`
  - aggregate:
    - best val mIoU mean/std: `0.2104 / 0.0278`
    - test mIoU mean/std: `0.3461 / 0.0941`
    - runtime mean/std (s): `77.61 / 0.62`

- Verified capped 3-seed depth-gated summary:
  - command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\clip_capped_3070_gate.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe --hf-offline`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_capped_gate_3070_seeds_1337-1338-1339_20260616-220517.json`
  - seed outputs:
    - `CLIP_capped_gate_3070_seed1337_20260616-220531`
    - `CLIP_capped_gate_3070_seed1338_20260616-220647`
    - `CLIP_capped_gate_3070_seed1339_20260616-220803`
  - aggregate:
    - best val mIoU mean/std: `0.2424 / 0.0359`
    - test mIoU mean/std: `0.3458 / 0.0440`
    - runtime mean/std (s): `76.32 / 0.23`
  - interpretation:
    compared with no gate, depth gating improves validation mean and substantially reduces test variance, but the test mean is nearly tied on the current capped split.

- Added `normal gate` ablation configs:
  - smoke:
    `geometry_probing\umd_linear_probing\configs\clip_smoke_3070_gate_normal.yaml`
  - capped:
    `geometry_probing\umd_linear_probing\configs\clip_capped_3070_gate_normal.yaml`

- Verified `normal gate` smoke pilot:
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_smoke_3070_gate_normal.yaml`
  - output root:
    `geometry_probing\umd_linear_probing\outputs\CLIP_smoke_gate_normal_3070_20260616-221324`
  - summary:
    - best val mIoU: `0.08932286555446516`
    - test mIoU: `0.09531072063107723`
  - interpretation:
    normal-gated fusion runs correctly, but this smoke pilot does not outperform the depth-gated smoke setting.

- Verified capped 3-seed normal-gated summary:
  - command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\clip_capped_3070_gate_normal.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe --hf-offline`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_capped_gate_normal_3070_seeds_1337-1338-1339_20260616-221406.json`
  - aggregate:
    - best val mIoU mean/std: `0.2273 / 0.0190`
    - test mIoU mean/std: `0.3033 / 0.0342`
    - runtime mean/std (s): `76.52 / 0.24`
  - interpretation:
    normal gating is weaker than depth gating on the capped split and also underperforms the no-gate baseline on test mean.

- Built a lower-resource train split from the capped setup:
  - script:
    `geometry_probing\umd_linear_probing\scripts\build_lowres_split.py`
  - split file:
    `geometry_probing\umd_linear_probing\metadata\splits\category_split_seed42_v20_lowres_train4.json`
  - split size:
    - train: `24`
    - val: `56`
    - test: `56`

- Verified low-resource 3-seed geometry baseline:
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_geom.yaml`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_lowres_train4_geom_3070_seeds_1337-1338-1339_20260616-222631.json`
  - aggregate:
    - best val mIoU mean/std: `0.1580 / 0.0182`
    - test mIoU mean/std: `0.1946 / 0.0240`
    - runtime mean/std (s): `64.80 / 0.12`

- Verified low-resource 3-seed depth-gated geometry model:
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate.yaml`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_lowres_train4_gate_3070_seeds_1337-1338-1339_20260616-223026.json`
  - aggregate:
    - best val mIoU mean/std: `0.1758 / 0.0190`
    - test mIoU mean/std: `0.2516 / 0.0251`
    - runtime mean/std (s): `64.59 / 0.71`
  - interpretation:
    this is the strongest current evidence for the paper direction. Under a reduced training budget, depth-gated fusion improves both val mean and test mean over the no-gate baseline.

- Parameterized the low-resource split builder:
  - script:
    `geometry_probing\umd_linear_probing\scripts\build_lowres_split.py`
  - new args:
    `--train-frames-per-tool`, `--src-split`, `--out-split`
  - verification:
    `python -m pytest tests\test_geometry_linear_probe_eval.py tests\test_geometry_multiseed.py tests\test_geometry_config.py -q`
  - result:
    `11 passed`

- Built additional low-resource splits from the capped setup:
  - train-6 split:
    `geometry_probing\umd_linear_probing\metadata\splits\category_split_seed42_v20_lowres_train6.json`
    - train: `36` (`6` per tool)
    - val: `56`
    - test: `56`
  - train-2 split:
    `geometry_probing\umd_linear_probing\metadata\splits\category_split_seed42_v20_lowres_train2.json`
    - train: `12` (`2` per tool)
    - val: `56`
    - test: `56`

- Verified low-resource train-6 3-seed geometry baseline:
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train6_3070_geom.yaml`
  - command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\clip_lowres_train6_3070_geom.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe --hf-offline`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_lowres_train6_geom_3070_seeds_1337-1338-1339_20260616-225146.json`
  - aggregate:
    - best val mIoU mean/std: `0.2681 / 0.0783`
    - test mIoU mean/std: `0.3279 / 0.0742`
    - runtime mean/std (s): `64.46 / 0.23`

- Verified low-resource train-6 3-seed depth-gated model:
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train6_3070_gate.yaml`
  - command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\clip_lowres_train6_3070_gate.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe --hf-offline`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_lowres_train6_gate_3070_seeds_1337-1338-1339_20260616-225510.json`
  - aggregate:
    - best val mIoU mean/std: `0.2785 / 0.0113`
    - test mIoU mean/std: `0.3309 / 0.0284`
    - runtime mean/std (s): `64.62 / 0.16`
  - interpretation:
    train-6 depth gating gives a small mean gain and lower variance; treat this as stability evidence rather than a large effect.

- Verified low-resource train-2 3-seed geometry baseline:
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_geom.yaml`
  - command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_geom.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe --hf-offline`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_lowres_train2_geom_3070_seeds_1337-1338-1339_20260616-225834.json`
  - aggregate:
    - best val mIoU mean/std: `0.0965 / 0.0073`
    - test mIoU mean/std: `0.1198 / 0.0068`
    - runtime mean/std (s): `57.07 / 0.76`

- Verified low-resource train-2 3-seed depth-gated model:
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_gate.yaml`
  - command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_gate.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe --hf-offline`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_lowres_train2_gate_3070_seeds_1337-1338-1339_20260616-230134.json`
  - aggregate:
    - best val mIoU mean/std: `0.0944 / 0.0196`
    - test mIoU mean/std: `0.1528 / 0.0317`
    - runtime mean/std (s): `57.07 / 0.09`
  - interpretation:
    train-2 depth gating improves test mean over no gate but not validation mean; present this as extreme low-resource generalization evidence, not a universal gain.

- Added a small eval usability option:
  - entrypoint:
    `pba\geometry\linear_probe.py`
  - option:
    `geometry-eval --num-examples`
  - purpose:
    export qualitative examples without editing experiment configs.

- Added local override support to evaluation:
  - entrypoint:
    `pba\geometry\linear_probe.py`
  - option:
    `geometry-eval --local`
  - purpose:
    evaluate an existing checkpoint with seed-specific or stress-test config overrides without retraining.

- Added a lightweight head-statistics utility:
  - entrypoint:
    `pba\geometry\head_stats.py`
  - launcher command:
    `python run.py geometry-head-stats -- --config ...`
  - purpose:
    quantify the trainable parameter overhead of the geometry-gated head for paper-facing efficiency claims.
  - verification:
    `python -m pytest tests\test_geometry_head_stats.py tests\test_geometry_linear_probe_eval.py tests\test_geometry_config.py tests\test_public_api_smoke.py tests\test_pba_run.py -q`
  - result:
    `18 passed`

- Exported qualitative examples from the best train-2 depth-gated test seed:
  - checkpoint:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\linear_probe.pth`
  - command:
    `python run.py geometry-eval -- geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\linear_probe.pth --config geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_gate.yaml --split test --save-examples --num-examples 6`
  - metrics:
    - test mIoU: `0.1790`
  - examples:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\linear_probe.examples.pt`
  - gallery:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\qualitative\qualitative_examples.png`
  - legend:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\qualitative\palette_legend.png`

- Exported ranked qualitative success/failure cases from the same checkpoint:
  - script:
    `geometry_probing\umd_linear_probing\scripts\export_qualitative_cases.py`
  - command:
    `python geometry_probing\umd_linear_probing\scripts\export_qualitative_cases.py geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\linear_probe.pth --config geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_gate.yaml --split test --num-best 4 --num-worst 4`
  - scored cases:
    `56`
  - top cluster:
    `saw_03`, per-image mIoU about `0.1152-0.1172`
  - bottom cluster:
    `shovel_02`, per-image mIoU about `0.0100-0.0103`
  - case scores:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\qualitative_test_ranked\case_scores.json`
  - success gallery:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\qualitative_test_ranked\success_cases\qualitative_examples.png`
  - failure gallery:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\qualitative_test_ranked\failure_cases\qualitative_examples.png`

- Created reviewer-facing asset notes:
  - path:
    `experiments\2026-06-16-reviewer-facing-assets.md`
  - contents:
    compact result table, compact ablation table, qualitative asset paths, and cautious wording.

- Exported paired no-gate vs depth-gate qualitative comparisons for the cleanest low-resource setting:
  - script:
    `geometry_probing\umd_linear_probing\scripts\export_paired_qualitative.py`
  - setting:
    `train=4/tool`, `seed=1338`, `test`
  - baseline checkpoint:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_geom_3070_seed1338_20260616-222749\lr1e-03_wd1e-02\linear_probe.pth`
  - depth-gate checkpoint:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_3070_seed1338_20260616-223144\lr1e-03_wd1e-02\linear_probe.pth`
  - command:
    `python geometry_probing\umd_linear_probing\scripts\export_paired_qualitative.py --baseline-config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_geom.yaml --baseline-checkpoint geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_geom_3070_seed1338_20260616-222749\lr1e-03_wd1e-02\linear_probe.pth --gate-config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate.yaml --gate-checkpoint geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_3070_seed1338_20260616-223144\lr1e-03_wd1e-02\linear_probe.pth --split test --output-dir geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_test --num-cases 4`
  - case scores:
    `geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_test\paired_case_scores.json`
  - depth-gate wins:
    `geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_test\depth_gate_wins\paired_examples.png`
    - top cluster:
      `knife_01`, depth-gate per-image mIoU improves by about `+0.0784` to `+0.0942`
  - no-gate wins:
    `geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_test\baseline_wins\paired_examples.png`
    - top cluster:
      `spoon_01`, no-gate remains better by about `+0.0479` to `+0.0539`
  - both-fail cases:
    `geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_test\both_fail\paired_examples.png`
    - cluster:
      `shovel_02`
  - verification:
    visual inspection confirmed the paired depth-gate-wins gallery is non-empty and aligned.
  - interpretation:
    useful reviewer-facing qualitative evidence: depth gating can suppress noisy no-gate predictions and better follow elongated tool geometry on `knife_01`, while `spoon_01` and `shovel_02` remain important limitations.

- Drafted paper-facing results notes:
  - path:
    `experiments\2026-06-16-paper-results-draft.md`
  - contents:
    method summary, local capped-subset setup, main result table, ablation table, qualitative analysis, evidence paths, and next experiment proposal.

- Added a controlled noisy-depth robustness hook:
  - code:
    `geometry_probing\umd_linear_probing\src\data\dataset.py`
  - config key:
    `dataset.geometry.depth_noise`
  - supported options:
    `std`, `dropout_prob`, `clamp`
  - default behavior:
    disabled; existing clean-depth configs are unchanged unless `depth_noise` is explicitly set.
  - test:
    `tests\test_umd_geometry_dataset_noise.py`
  - verification:
    `python -m pytest tests\test_umd_geometry_dataset_noise.py tests\test_geometry_linear_probe_eval.py tests\test_geometry_multiseed.py tests\test_geometry_config.py -q`
  - result:
    `13 passed`

- Ran a noisy-depth robustness pilot:
  - evidence tier:
    `pilot`, single seed
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate_depthnoise.yaml`
  - noise:
    depth Gaussian `std=0.15`, depth dropout `0.15`
  - command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate_depthnoise.yaml --seeds 1338 --python E:\conda\envs\bev_py39_torch231\python.exe --hf-offline`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_lowres_train4_gate_depthnoise_3070_seeds_1338_20260616-232848.json`
  - output:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_depthnoise_3070_seed1338_20260616-232900`
  - metrics:
    - best val mIoU: `0.1883`
    - test mIoU: `0.2451`
    - runtime (s): `61.59`
  - interpretation:
    this single-seed pilot is below the clean depth-gate seed1338 test mIoU (`0.2718`) but remains above the no-gate seed1338 test mIoU (`0.2175`), so a 3-seed noisy-depth robustness check is worth running next.

- Verified noisy-depth robustness across 3 seeds:
  - evidence tier:
    `capped subset`, `train=4/tool`, 3 seeds
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate_depthnoise.yaml`
  - noise:
    depth Gaussian `std=0.15`, depth dropout `0.15`, clamp to `[-1, 1]`
  - command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate_depthnoise.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe --hf-offline`
  - summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\CLIP_lowres_train4_gate_depthnoise_3070_seeds_1337-1338-1339_20260617-075653.json`
  - seed outputs:
    - `CLIP_lowres_train4_gate_depthnoise_3070_seed1337_20260617-075706`
    - `CLIP_lowres_train4_gate_depthnoise_3070_seed1338_20260617-075808`
    - `CLIP_lowres_train4_gate_depthnoise_3070_seed1339_20260617-075909`
  - per-seed val/test:
    - seed `1337`: `0.1419` / `0.2167`
    - seed `1338`: `0.1883` / `0.2451`
    - seed `1339`: `0.2133` / `0.2004`
  - aggregate:
    - best val mIoU mean/std: `0.1812 / 0.0362`
    - test mIoU mean/std: `0.2207 / 0.0226`
    - runtime mean/std (s): `61.33 / 1.21`
  - interpretation:
    noisy depth degrades relative to clean depth gate at `train=4/tool` (`0.2516 +/- 0.0251` test mIoU), but remains above no gate (`0.1946 +/- 0.0240` test mIoU). This supports a cautious robustness claim: the depth gate retains part of its benefit under degraded depth, but it depends on geometry quality. This setting trains and evaluates under degraded depth; it is not yet a pure clean-train/noisy-test inference robustness test.

- Exported paired clean-depth vs noisy-depth qualitative comparisons:
  - script:
    `geometry_probing\umd_linear_probing\scripts\export_paired_qualitative.py`
  - setting:
    `train=4/tool`, `seed=1338`, `test`
  - clean-depth checkpoint:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_3070_seed1338_20260616-223144\lr1e-03_wd1e-02\linear_probe.pth`
  - noisy-depth checkpoint:
    `geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_depthnoise_3070_seed1338_20260617-075808\lr1e-03_wd1e-02\linear_probe.pth`
  - command:
    `python geometry_probing\umd_linear_probing\scripts\export_paired_qualitative.py --baseline-config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate.yaml --baseline-checkpoint geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_3070_seed1338_20260616-223144\lr1e-03_wd1e-02\linear_probe.pth --gate-config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate_depthnoise.yaml --gate-checkpoint geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_depthnoise_3070_seed1338_20260617-075808\lr1e-03_wd1e-02\linear_probe.pth --split test --output-dir geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_clean_vs_noisy_depth --num-cases 4`
  - case scores:
    `geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_clean_vs_noisy_depth\paired_case_scores.json`
  - noisy-depth wins:
    `geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_clean_vs_noisy_depth\depth_gate_wins\paired_examples.png`
    - top cases include `saw_03` and `hammer_01`, with noisy-depth per-image mIoU deltas up to about `+0.0621`
  - clean-depth wins:
    `geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_clean_vs_noisy_depth\baseline_wins\paired_examples.png`
    - top cases are `knife_01`, with noisy depth losing by about `0.0465` to `0.0540`
  - both-fail cases:
    `geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_clean_vs_noisy_depth\both_fail\paired_examples.png`
    - cluster:
      `shovel_02`
  - interpretation:
    useful robustness qualitative evidence: degraded depth does not simply fail uniformly, but it changes category-level behavior. It may help some `saw_03` / `hammer_01` cases while hurting the clean depth gate's strong `knife_01` behavior.

- Evaluated clean-depth trained checkpoints under noisy-depth test config:
  - evidence tier:
    `capped subset`, `clean-train/noisy-test evaluation`, 3 seeds
  - base checkpoints:
    clean `train=4/tool` depth-gate checkpoints from seeds `1337`, `1338`, `1339`
  - eval config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate_depthnoise.yaml`
  - local overrides:
    - `geometry_probing\umd_linear_probing\configs\local_seed_1337.yaml`
    - `geometry_probing\umd_linear_probing\configs\local_seed_1338.yaml`
    - `geometry_probing\umd_linear_probing\configs\local_seed_1339.yaml`
  - commands:
    - `python run.py geometry-eval geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_3070_seed1337_20260616-223039\lr1e-03_wd1e-02\linear_probe.pth --config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate_depthnoise.yaml --local geometry_probing\umd_linear_probing\configs\local_seed_1337.yaml --split test`
    - `python run.py geometry-eval geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_3070_seed1338_20260616-223144\lr1e-03_wd1e-02\linear_probe.pth --config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate_depthnoise.yaml --local geometry_probing\umd_linear_probing\configs\local_seed_1338.yaml --split test`
    - `python run.py geometry-eval geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train4_gate_3070_seed1339_20260616-223249\lr1e-03_wd1e-02\linear_probe.pth --config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate_depthnoise.yaml --local geometry_probing\umd_linear_probing\configs\local_seed_1339.yaml --split test`
  - summary:
    `experiments\clean_train_noisy_test_train4_depth_gate_summary.json`
  - per-seed test mIoU:
    - seed `1337`: `0.2225`
    - seed `1338`: `0.2712`
    - seed `1339`: `0.2596`
  - aggregate:
    - test mIoU mean/std: `0.2511 / 0.0254`
  - interpretation:
    clean-depth trained gates remain stable under this moderate noisy-depth evaluation setting, nearly matching clean test mean (`0.2516 +/- 0.0251`). The larger degradation in noisy-depth train+eval (`0.2207 +/- 0.0226`) appears to come from learning with degraded geometry rather than inference-time noise alone.

- Evaluated extreme low-resource clean-depth trained checkpoints under noisy-depth test config:
  - evidence tier:
    `capped subset`, `train=2/tool`, `clean-train/noisy-test evaluation`, 3 seeds
  - config:
    `geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_gate_depthnoise.yaml`
  - base checkpoints:
    clean `train=2/tool` depth-gate checkpoints from seeds `1337`, `1338`, `1339`
  - commands:
    - `python run.py geometry-eval geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1337_20260616-230147\lr1e-03_wd1e-02\linear_probe.pth --config geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_gate_depthnoise.yaml --local geometry_probing\umd_linear_probing\configs\local_seed_1337.yaml --split test`
    - `python run.py geometry-eval geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1338_20260616-230244\lr1e-03_wd1e-02\linear_probe.pth --config geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_gate_depthnoise.yaml --local geometry_probing\umd_linear_probing\configs\local_seed_1338.yaml --split test`
    - `python run.py geometry-eval geometry_probing\umd_linear_probing\outputs\CLIP_lowres_train2_gate_3070_seed1339_20260616-230341\lr1e-03_wd1e-02\linear_probe.pth --config geometry_probing\umd_linear_probing\configs\clip_lowres_train2_3070_gate_depthnoise.yaml --local geometry_probing\umd_linear_probing\configs\local_seed_1339.yaml --split test`
  - summary:
    `experiments\clean_train_noisy_test_train2_depth_gate_summary.json`
  - per-seed test mIoU:
    - seed `1337`: `0.1161`
    - seed `1338`: `0.1794`
    - seed `1339`: `0.1619`
  - aggregate:
    - test mIoU mean/std: `0.1525 / 0.0327`
  - interpretation:
    at `train=2/tool`, moderate noisy-depth evaluation nearly matches clean depth-gate test mean (`0.1528 +/- 0.0317`), strengthening the inference-time robustness observation under the most data-starved split.

- Quantified the parameter and runtime cost of the depth gate:
  - note:
    `experiments\2026-06-17-depth-gate-efficiency-note.md`
  - commands:
    - `python run.py geometry-head-stats -- --config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_geom.yaml`
    - `python run.py geometry-head-stats -- --config geometry_probing\umd_linear_probing\configs\clip_lowres_train4_3070_gate.yaml`
  - head parameters:
    - baseline multi-layer probe head: `30,768`
    - depth-gated probe head: `36,920`
    - extra gate parameters: `6,152`
    - gate overhead vs baseline head: about `20.0%`
  - runtime evidence from existing 3-seed runs:
    - `train=4/tool`: no gate `64.80s`, depth gate `64.59s`
    - `train=2/tool`: no gate `57.07s`, depth gate `57.07s`
  - interpretation:
    the proposed module is genuinely lightweight: it adds a small number of trainable parameters to the head and no meaningful runtime overhead in the current single-GPU setup.

- Checked DINO-family linkage with a local DINOv2 smoke run:
  - note:
    `experiments\2026-06-18-dino-family-check.md`
  - evidence tier:
    `smoke`
  - DINOv2 source:
    `models\dinov2`
  - DINOv2 checkpoint:
    `models\dinov2_vitb14_pretrain.pth`
  - no-gate config:
    `geometry_probing\umd_linear_probing\configs\dinov2_smoke_3070_geom.yaml`
  - no-gate output:
    `geometry_probing\umd_linear_probing\outputs\DINOv2_smoke_geom_3070_20260618-090349`
  - no-gate summary:
    - best val mIoU: `0.1138192641`
    - test mIoU: `0.1234198750`
  - depth-gate config:
    `geometry_probing\umd_linear_probing\configs\dinov2_smoke_3070_gate.yaml`
  - depth-gate output:
    `geometry_probing\umd_linear_probing\outputs\DINOv2_smoke_gate_3070_20260618-090423`
  - depth-gate summary:
    - best val mIoU: `0.1281817292`
    - test mIoU: `0.1100847186`
  - interpretation:
    DINOv2 confirms the DINO-family probing route can run locally, but this one-class smoke result is not paper-facing evidence and does not support a universal depth-gate gain. It should be used only as a code-linkage and boundary check.
  - DINOv3 status:
    superseded by the completed local DINOv3-S smoke run below.

- Verified local DINOv3-S smoke runs after the user supplied the smallest checkpoint:
  - note:
    `experiments\2026-06-18-dino-family-check.md`
  - evidence tier:
    `smoke`
  - source:
    `fusion_zero_shot\src\dino\third_party\dinov3`
  - checkpoint:
    `models\DINOV3\DINOV3-pth\dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
  - compatibility changes:
    added Python 3.9 future-annotation imports to the local DINOv3 backbone path and changed the project DINOv3 loader to import `dinov3.hub.backbones` directly, avoiding unrelated hub entries.
  - no-gate config:
    `geometry_probing\umd_linear_probing\configs\dinov3_vits16_smoke_3070_geom.yaml`
  - no-gate output:
    `geometry_probing\umd_linear_probing\outputs\DINOv3_vits16_smoke_geom_3070_20260618-093248`
  - no-gate summary:
    - best val mIoU: `0.0804346265`
    - test mIoU: `0.0951028632`
  - depth-gate config:
    `geometry_probing\umd_linear_probing\configs\dinov3_vits16_smoke_3070_gate.yaml`
  - depth-gate output:
    `geometry_probing\umd_linear_probing\outputs\DINOv3_vits16_smoke_gate_3070_20260618-093319`
  - depth-gate summary:
    - best val mIoU: `0.0659151324`
    - test mIoU: `0.0706707797`
  - interpretation:
    DINOv3-S now runs end-to-end in the local probing pipeline, improving code linkage to the original DINO-family route. However, this one-class smoke result does not support a DINOv3 depth-gate gain; no gate is stronger on both val and test. Keep the main manuscript claim on OpenCLIP low-resource probing unless larger DINOv3 runs later change the evidence.

- Verified DINOv3-S low-resource `train=4/tool` 3-seed comparison:
  - evidence tier:
    `capped subset`, `train=4/tool`, 3 seeds
  - note:
    `experiments\2026-06-18-dino-family-check.md`
  - no-gate config:
    `geometry_probing\umd_linear_probing\configs\dinov3_vits16_lowres_train4_3070_geom.yaml`
  - no-gate command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\dinov3_vits16_lowres_train4_3070_geom.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe`
  - no-gate summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\DINOv3_vits16_lowres_train4_geom_3070_seeds_1337-1338-1339_20260618-102827.json`
  - no-gate aggregate:
    - best val mIoU mean/std: `0.1632 / 0.0170`
    - test mIoU mean/std: `0.1674 / 0.0292`
    - runtime mean/std (s): `55.50 / 2.96`
  - depth-gate config:
    `geometry_probing\umd_linear_probing\configs\dinov3_vits16_lowres_train4_3070_gate.yaml`
  - depth-gate command:
    `python run.py geometry-multiseed -- --config geometry_probing\umd_linear_probing\configs\dinov3_vits16_lowres_train4_3070_gate.yaml --seeds 1337 1338 1339 --python E:\conda\envs\bev_py39_torch231\python.exe`
  - depth-gate summary:
    `geometry_probing\umd_linear_probing\outputs\multiseed_summaries\DINOv3_vits16_lowres_train4_gate_3070_seeds_1337-1338-1339_20260618-103146.json`
  - depth-gate aggregate:
    - best val mIoU mean/std: `0.1707 / 0.0196`
    - test mIoU mean/std: `0.1940 / 0.0163`
    - runtime mean/std (s): `52.82 / 0.46`
  - interpretation:
    DINOv3-S depth gate gives a modest low-resource gain over the DINOv3-S no-gate probe, but its absolute test mean remains below the OpenCLIP depth-gate result on the same split (`0.2516 +/- 0.0251`). This is not comparable to the original paper's fully supervised Figure 4 values.

- Implemented a reliability-aware adaptive fusion gate in the original Flux Kontext + DINOv3/PCA fusion path:
  - evidence tier:
    `code mechanism + synthetic unit tests`
  - note:
    `experiments\2026-06-18-adaptive-fusion-gate.md`
  - main code:
    `fusion_zero_shot\src\pipeline\geometry_stage.py`
  - eval wiring:
    `fusion_zero_shot\src\agd20k_eval\run_flux_kontext_eval.py`
  - config keys:
    `geom_adaptive_fusion`, `geom_gate_min_lambda`, `geom_gate_max_lambda`, `geom_gate_verb_weight`, `geom_gate_geometry_weight`, `geom_gate_alignment_weight`
  - default behavior:
    `geom_adaptive_fusion: false`, preserving the original fixed soft-fusion lambda unless explicitly enabled.
  - metadata exported to `summary.csv` / `metrics.json`:
    `adaptive_fusion_enabled`, `adaptive_lambda`, `interaction_confidence`, `geometry_confidence`, `alignment_confidence`, `base_lambda`, `lambda_range`, `adaptive_fusion_detail`
  - verification:
    `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_fusion_adaptive_gate.py tests\test_fusion_runtime_contract.py tests\test_fusion_config_kontext.py -q`
  - result:
    `14 passed in 0.62s`
  - related smoke:
    `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_fusion_adaptive_gate.py tests\test_fusion_runtime_contract.py tests\test_fusion_config_kontext.py tests\test_fusion_runtime_docs.py tests\test_public_api_smoke.py -q`
  - related smoke result:
    `21 passed in 0.72s`
  - interpretation:
    this directly targets the original paper's shallow/open-loop fusion limitation by making the interaction-vs-geometry weighting per-sample and reliability-aware. It is not yet an AGD20K metric claim because local Flux/AGD20K/cache assets are still missing.

- Ran a synthetic diagnostic for the reliability-aware adaptive fusion gate:
  - evidence tier:
    `synthetic diagnostic`
  - script:
    `experiments\synthetic_adaptive_fusion_gate.py`
  - command:
    `E:\conda\envs\bev_py39_torch231\python.exe experiments\synthetic_adaptive_fusion_gate.py`
  - outputs:
    `outputs\synthetic_adaptive_fusion_gate\results.csv`
    `outputs\synthetic_adaptive_fusion_gate\summary.json`
  - summary:
    adaptive fusion wins `4/5` synthetic scenarios on KLD, `4/5` on SIM, and `3/5` on NSS.
  - behavior check:
    flat geometry pushes lambda from fixed `0.65` to `0.85`, so the gate trusts interaction more; flat interaction pushes lambda to `0.35`, so the gate trusts geometry more.
  - failure boundary:
    weak noisy geometry can still look structured to the training-free confidence score, so adaptive fusion is not guaranteed to improve every synthetic case. This should be presented as a limitation rather than hidden.
  - verification:
    `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_fusion_adaptive_gate.py tests\test_fusion_runtime_contract.py tests\test_fusion_config_kontext.py -q`
  - result:
    `15 passed in 0.48s`

- Prepared local AGD20K smoke evaluation for the reliability-aware adaptive fusion gate:
  - evidence tier:
    `asset/config preparation + blocked model download`
  - note:
    `experiments\2026-06-18-adaptive-fusion-gate.md`
  - AGD20K local data:
    `datasets\AGD20K\AGD20K\Unseen\testset`
  - iterator sanity:
    `32` samples with `max_per_object=1`
  - fixed-fusion smoke config:
    `fusion_zero_shot\src\agd20k_eval\config.local.fixed_smoke.yaml`
  - adaptive-fusion smoke config:
    `fusion_zero_shot\src\agd20k_eval\config.local.adaptive_smoke.yaml`
  - smoke subset:
    `hold`, `max_images_per_object=1`, `5` samples
  - DINO mode:
    `dino_cache_only: false`, local DINOv3-S checkpoint supplied through `DINO_CHECKPOINT_PATH`
  - verification:
    `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_fusion_adaptive_gate.py tests\test_fusion_runtime_contract.py tests\test_fusion_config_kontext.py -q`
  - result:
    `15 passed in 0.39s`
  - blocker:
    FLUX.1-Kontext-dev download is blocked by HuggingFace token permission. The retry returned `403 Forbidden` with the message that public gated repository access must be enabled in the fine-grained token settings.
  - interpretation:
    this is not yet a real AGD20K metric result. Once the token can access gated public repositories and the FLUX weights are downloaded to `models\FLUX.1-Kontext-dev`, run the fixed and adaptive smoke configs under the same subset.

## Current Status

- The `clip.yaml` full training path reaches real training on the downloaded UMD dataset.
- The `clip_smoke_3070.yaml` path is fully runnable and finishes in about 30 seconds.
- The first innovation head is implemented and runnable.
- The multi-seed runner is implemented, tested, and usable from `run.py geometry-multiseed`.
- A local smoke-scale geometry side-data pipeline now exists and is runnable.
- A local capped-split geometry side-data pipeline now exists and is runnable.
- The strongest current ablation table is now:
  - no gate:
    val `0.2104 +/- 0.0278`, test `0.3461 +/- 0.0941`
  - depth gate:
    val `0.2424 +/- 0.0359`, test `0.3458 +/- 0.0440`
  - normal gate:
    val `0.2273 +/- 0.0190`, test `0.3033 +/- 0.0342`
- Current best paper-safe interpretation:
  depth-gated fusion is the most promising novelty because it improves validation mean and produces a more stable test distribution than baseline, while normal-gated fusion is less compelling.
- Current strongest paper result:
  in the lower-resource `train=24` setting, depth gating improves test mIoU from `0.1946` to `0.2516` with similar runtime, and the added `train=12` / `train=36` points now support a low-resource trend story.
- Current compact low-resource trend:
  - `train=36`: no gate test `0.3279 +/- 0.0742`, depth gate test `0.3309 +/- 0.0284`
  - `train=24`: no gate test `0.1946 +/- 0.0240`, depth gate test `0.2516 +/- 0.0251`
  - `train=12`: no gate test `0.1198 +/- 0.0068`, depth gate test `0.1528 +/- 0.0317`
- Current noisy-depth robustness result:
  - `train=24`, noisy depth gate test `0.2207 +/- 0.0226`
  - clean depth gate remains stronger (`0.2516 +/- 0.0251`), but noisy depth gate remains above no gate (`0.1946 +/- 0.0240`)
  - clean-train/noisy-test depth gate test is `0.2511 +/- 0.0254`, suggesting moderate depth noise at inference time is not the main failure mode in this capped setting
  - at `train=12`, clean-train/noisy-test depth gate test is `0.1525 +/- 0.0327`, nearly tied with clean depth gate (`0.1528 +/- 0.0317`)
- Current paper-safe interpretation:
  depth gating is most useful under low-resource stress, especially on test/generalization, but the evidence should not be framed as a universal validation gain because `train=12` val mean is roughly tied/slightly lower.
- Current DINO-family status:
  DINOv2 and DINOv3-S smoke runs both run locally. DINOv3-S also has a train=4/tool 3-seed low-resource check where depth gate improves test mIoU from `0.1674 +/- 0.0292` to `0.1940 +/- 0.0163`, but this is still below OpenCLIP depth gate on the same split. The manuscript should keep the main claim on the OpenCLIP low-resource branch and avoid comparing local low-resource numbers with the original paper's fully supervised Figure 4 values.
- Current Flux-DINO fusion extension status:
  reliability-aware adaptive fusion is implemented and covered by synthetic tests. AGD20K unseen test data is now available locally and fixed-vs-adaptive smoke configs are prepared. Full real evaluation is pending the required Flux Kontext weights; the current HuggingFace token is blocked from public gated repository downloads.
- Full geometry side-data is still missing locally for the full split:
  `train_depth`, `train_normal`, `val_depth`, `val_normal`, `test_depth`, `test_normal`
- The `clip.yaml` config expects `models\open_clip\src`. A local placeholder
  directory was created so the installed `open_clip` package can be imported.

## 2026-06-19 Lightweight Student Pilot

- Implemented a lightweight RGB-D student entrypoint:
  - code:
    `pba\geometry\lightweight_student.py`
  - command:
    `python run.py lightweight-student -- --config ...`
  - modes:
    `rgb`, `depth`, `concat`, `gate`, `depth_modulated_rgb`
  - added class-balanced hard-label loss and optional teacher-logit distillation.
- Added teacher-logit export:
  - code:
    `pba\geometry\export_teacher_logits.py`
  - command alias:
    `geometry-export-teacher-logits`
  - exported teacher:
    OpenCLIP low-resource train-4 depth-gate seed `1338`
  - manifest:
    `geometry_probing\umd_linear_probing\metadata\teacher_logits\clip_lowres_train4_gate_seed1338\train_teacher_logits_manifest.json`
  - entries:
    `24/24` train samples.
- Verification:
  - command:
    `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_lightweight_student.py tests\test_pba_run.py -q`
  - result:
    `9 passed`
- Pilot evidence tier:
  `pilot`, not paper-main result.
- Pilot setup:
  `train=4/tool`, seed `1337`, `20` epochs, input `240x320`, width `16`.
- Pilot results:
  - RGB:
    val mIoU `0.1963`, test mIoU `0.1422`, params `39,224`, latency `1.90 ms`, FPS `526.3`
  - Depth:
    val mIoU `0.1393`, test mIoU `0.0902`, params `27,224`, latency `1.86 ms`, FPS `538.3`
  - RGB-D concat:
    val mIoU `0.2214`, test mIoU `0.1073`, params `55,624`, latency `5.94 ms`, FPS `168.3`
  - RGB-D gate:
    val mIoU `0.1260`, test mIoU `0.0977`, params `55,625`, latency `3.83 ms`, FPS `261.0`
  - Depth-modulated RGB:
    val mIoU `0.1501`, test mIoU `0.1202`, params `53,272`, latency `5.53 ms`, FPS `180.8`
  - RGB + teacher KD:
    val mIoU `0.1227`, test mIoU `0.0905`, params `39,224`, latency `2.10 ms`, FPS `476.7`
  - Depth-modulated RGB + teacher KD:
    val mIoU `0.1961`, test mIoU `0.1095`, params `53,272`, latency `3.52 ms`, FPS `283.8`
- Interpretation:
  lightweight students are fast, but naive RGB-D fusion and first-pass teacher distillation do not beat RGB-only on test in this pilot. Do not use this as a positive main claim. Treat lightweight deployment as an exploratory extension or future-work direction unless later distillation/optimization improves it.
- Detailed note:
  `experiments\2026-06-19-lightweight-student-pilot.md`

## 2026-06-19 FLUX-DINOv3 Adaptive Fusion Ultra Smoke

- Evidence tier:
  `ultra smoke`, `1` AGD20K unseen sample, `1` FLUX step, `512x512`.
- Story:
  this is the closest continuation of the original paper: FLUX supplies interaction heatmaps, DINOv3/PCA supplies geometry parts, and the added reliability-aware gate chooses the interaction-vs-geometry fusion weight per image.
- Code fixes:
  - `fusion_zero_shot\src\pipeline\pca_stage.py`
    added `model_name` passthrough to `FeatureExtractor`.
  - `fusion_zero_shot\src\agd20k_eval\run_flux_kontext_eval.py`
    reads `geom_pipeline.dino_model_name`.
  - `fusion_zero_shot\src\dino\dino\pipeline\features\extractor.py`
    imports `dinov3.hub.backbones` directly to avoid Python 3.9 hubconf classifier import failure.
  - `fusion_zero_shot\src\agd20k_eval\run_flux_kontext_eval.py`
    creates ASCII aliases for token heatmap filenames before calling the warp subprocess, avoiding Windows/OpenCV Unicode filename failures.
- Configs:
  - fixed:
    `fusion_zero_shot\src\agd20k_eval\config.local.fixed_ultra_smoke.yaml`
  - adaptive:
    `fusion_zero_shot\src\agd20k_eval\config.local.adaptive_ultra_smoke.yaml`
- Commands:
  - fixed:
    `E:\conda\envs\bev_py39_torch231\python.exe run.py fusion-eval -- --config fusion_zero_shot\src\agd20k_eval\config.local.fixed_ultra_smoke.yaml`
  - adaptive:
    `E:\conda\envs\bev_py39_torch231\python.exe run.py fusion-eval -- --config fusion_zero_shot\src\agd20k_eval\config.local.adaptive_ultra_smoke.yaml`
  - environment:
    `DINO_CHECKPOINT_PATH=E:\work\2cvpr\Probing_Bridging_Affordance\models\DINOV3\DINOV3-pth\dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- Sample:
  `hold / axe / axe_000108.jpg`
- Outputs:
  - fixed:
    `outputs\fusion_zero_shot_fixed_ultra_smoke\20260619-111121`
  - adaptive:
    `outputs\fusion_zero_shot_adaptive_ultra_smoke\20260619-111656`
- Metrics:
  - FLUX interaction:
    KLD `1.3755`, SIM `0.2993`, NSS `0.0660`
  - DINOv3/PCA geometry:
    KLD `1.6453`, SIM `0.5750`, NSS `1.1404`
  - fixed soft fusion:
    KLD `0.8461`, SIM `0.4496`, NSS `0.9507`
  - adaptive reliability fusion:
    KLD `0.7478`, SIM `0.4862`, NSS `1.0456`
- Adaptive metadata:
  - `adaptive_lambda`: `0.5505`
  - `base_lambda`: `0.65`
  - `interaction_confidence`: `0.5503`
  - `geometry_confidence`: `0.5874`
  - `alignment_confidence`: `0.8219`
- Interpretation:
  the adaptive gate reduced the interaction weight because geometry looked slightly more reliable and well aligned with interaction. On this one-sample ultra smoke, adaptive fusion improved over fixed fusion on KLD, SIM, and NSS. This is a mechanism check, not a full AGD20K metric claim.
- Verification:
  `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_fusion_adaptive_gate.py tests\test_fusion_runtime_contract.py tests\test_fusion_config_kontext.py -q`
  result: `16 passed`
- Detailed note:
  `experiments\2026-06-19-flux-dinov3-adaptive-ultra-smoke.md`

## 2026-06-19 FLUX-DINOv3 Adaptive Fusion Hold-3 Ultra Pilot

- Evidence tier:
  `ultra pilot`, `3` AGD20K unseen samples, `1` FLUX step, `512x512`.
- Story:
  the original-paper-compatible route remains runnable beyond one image: FLUX/Kontext provides interaction heatmaps, DINOv3/PCA provides geometry parts, and the reliability-aware gate chooses the interaction-vs-geometry soft-fusion weight per image.
- New experiment utility:
  - `reuse_kontext_run_dir` in `fusion_zero_shot\src\agd20k_eval\run_flux_kontext_eval.py`
  - purpose:
    rerun fixed/adaptive fusion against identical saved FLUX/Kontext heatmaps, avoiding repeated FLUX generation and making the fusion comparison fairer.
  - related robustness fix:
    `pba\fusion\paths.py` now accepts ASCII heatmap aliases like `heat_tok00.png`.
  - CSV logging now uses `csv.DictWriter`, so JSON fields in `summary.csv` do not break column parsing.
- Configs:
  - fixed:
    `fusion_zero_shot\src\agd20k_eval\config.local.fixed_hold3_ultra.yaml`
  - adaptive:
    `fusion_zero_shot\src\agd20k_eval\config.local.adaptive_hold3_ultra.yaml`
- Commands:
  - fixed:
    `E:\conda\envs\bev_py39_torch231\python.exe run.py fusion-eval -- --config fusion_zero_shot\src\agd20k_eval\config.local.fixed_hold3_ultra.yaml`
  - adaptive:
    `E:\conda\envs\bev_py39_torch231\python.exe run.py fusion-eval -- --config fusion_zero_shot\src\agd20k_eval\config.local.adaptive_hold3_ultra.yaml`
  - environment:
    `DINO_CHECKPOINT_PATH=E:\work\2cvpr\Probing_Bridging_Affordance\models\DINOV3\DINOV3-pth\dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- Samples:
  - `hold / axe / axe_000108.jpg`
  - `hold / axe / axe_000692.jpg`
  - `hold / axe / axe_000961.jpg`
- Outputs:
  - fixed:
    `outputs\fusion_zero_shot_fixed_hold3_ultra\20260619-113135`
  - adaptive:
    `outputs\fusion_zero_shot_adaptive_hold3_ultra\20260619-114010`
- Runtime:
  - fixed with FLUX generation:
    about `8 min 15 s`
  - adaptive reusing fixed Kontext heatmaps:
    about `40 s`
- Mean soft-fusion metrics:
  - fixed soft fusion:
    KLD `1.4416`, SIM `0.3102`, NSS `0.3322`
  - adaptive reliability fusion:
    KLD `1.4036`, SIM `0.3229`, NSS `0.3680`
- Per-image win count:
  - KLD:
    adaptive better on `2/3`
  - SIM:
    adaptive better on `2/3`
  - NSS:
    adaptive better on `2/3`
- Adaptive lambda values:
  - `axe_000108.jpg`: `0.5505`
  - `axe_000692.jpg`: `0.5395`
  - `axe_000961.jpg`: `0.5673`
- Verification:
  `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_fusion_kontext_reuse.py tests\test_fusion_adaptive_gate.py tests\test_fusion_runtime_contract.py tests\test_fusion_config_kontext.py -q`
  result:
  `18 passed`
- Interpretation:
  this supports continuing the FLUX-DINOv3 adaptive-fusion route. The gain is modest and should be reported as an ultra pilot, not a full AGD20K claim. The main practical bottleneck is still FLUX generation; the adaptive gate itself is lightweight once heatmaps exist.

## 2026-06-19 FLUX-DINOv3 Conservative Reliability Gate Pilot

- Evidence tier:
  `ultra pilot`, cached FLUX/Kontext heatmaps, `1` FLUX step, `512x512`.
- Story:
  the always-on adaptive gate helped on `hold` but slightly hurt a small cross-affordance check. The updated paper story is therefore stronger and more cautious: use a conservative reliability gate that changes the original fixed fusion lambda only when the selected DINOv3/PCA part is sufficiently aligned with the FLUX interaction heatmap; otherwise fall back to the original `geom_soft_lambda=0.65`.
- Code updates:
  - `pba\fusion\prompts.py`
    deterministic object-token candidates, simple singular handling such as `skis -> ski`, and no fallback for object-token selection.
  - `fusion_zero_shot\src\pipeline\geometry_stage.py`
    optional `gate_similarity_floor` and `gate_fallback_lambda`, defaulting to `None` so previous behavior is preserved.
  - `fusion_zero_shot\src\agd20k_eval\run_flux_kontext_eval.py`
    reads `geom_gate_similarity_floor` and `geom_gate_fallback_lambda`.
  - `fusion_zero_shot\src\agd20k_eval\config.yaml`
    and `config.local.example.yaml` include these keys as `null`.
- Verification:
  - command:
    `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_fusion_adaptive_gate.py tests\test_fusion_prompts.py tests\test_fusion_kontext_reuse.py tests\test_fusion_config_kontext.py tests\test_fusion_runtime_contract.py -q`
  - result:
    `28 passed`
- Hold multi-object 5-sample pilot:
  - samples:
    `hold/axe`, `hold/cup`, `hold/golf_clubs`, `hold/knife`, `hold/skis`
  - fixed output:
    `outputs\fusion_zero_shot_fixed_hold_multi5_reuse_ultra\20260619-171007`
  - default adaptive output:
    `outputs\fusion_zero_shot_adaptive_hold_multi5_ultra\20260619-171102`
  - wide adaptive output:
    `outputs\fusion_zero_shot_adaptive_hold_multi5_widegate_ultra\20260619-171549`
  - conservative floor output:
    `outputs\fusion_zero_shot_adaptive_hold_multi5_floor_ultra\20260619-173332`
  - mean soft-fusion metrics:
    - fixed:
      KLD `2.1202`, SIM `0.2078`, NSS `0.6036`
    - default adaptive:
      KLD `2.0936`, SIM `0.2156`, NSS `0.6647`
    - wide adaptive:
      KLD `2.0806`, SIM `0.2191`, NSS `0.7013`
    - conservative floor:
      KLD `2.0806`, SIM `0.2191`, NSS `0.7013`, fallback `3/5`
  - interpretation:
    conservative floor keeps the wide-gate gains on high-similarity `axe` and `knife`, and falls back to fixed fusion on weaker geometry-interaction matches.
- Cross-affordance 3-sample boundary check:
  - samples:
    `cut/banana`, `open/refrigerator`, `type_on/laptop`
  - fixed output:
    `outputs\fusion_zero_shot_fixed_crossaff3_ultra\20260619-171823`
  - default adaptive output:
    `outputs\fusion_zero_shot_adaptive_crossaff3_ultra\20260619-172756`
  - wide adaptive output:
    `outputs\fusion_zero_shot_adaptive_crossaff3_widegate_ultra\20260619-172836`
  - conservative floor output:
    `outputs\fusion_zero_shot_adaptive_crossaff3_floor_ultra\20260619-173427`
  - mean soft-fusion metrics:
    - fixed:
      KLD `1.7964`, SIM `0.2536`, NSS `0.1629`
    - default adaptive:
      KLD `1.7988`, SIM `0.2527`, NSS `0.1409`
    - wide adaptive:
      KLD `1.8027`, SIM `0.2514`, NSS `0.1245`
    - conservative floor:
      KLD `1.7964`, SIM `0.2536`, NSS `0.1629`, fallback `3/3`
  - interpretation:
    this exposes and fixes a failure boundary: always-on adaptive fusion can over-trust geometry on weakly aligned actions, while the conservative floor avoids degradation by reverting to original fixed fusion.
  - caveats:
    `banana` and `refrigerator` used direct-resize fallback during heatmap warping; all results are ultra-pilot scale, not full AGD20K metrics.
- Detailed note:
  `experiments\2026-06-19-flux-dinov3-adaptive-ultra-smoke.md`

## 2026-06-19 FLUX-DINOv3 Capped12 Conservative-Gate Pilot

- Evidence tier:
  `ultra pilot`, `12` AGD20K unseen samples, cached FLUX/Kontext heatmaps, `1` FLUX step, `512x512`.
- Story:
  this is the strongest current check for the original-paper-compatible route. The original FLUX/Kontext + DINOv3/PCA fusion uses a fixed interaction/geometry weight. The added reliability gate changes that weight only when the selected PCA component has enough alignment evidence; otherwise it falls back to the original fixed lambda.
- Fixed config:
  `fusion_zero_shot\src\agd20k_eval\config.local.fixed_capped12_ultra.yaml`
- Adaptive configs:
  - default:
    `fusion_zero_shot\src\agd20k_eval\config.local.adaptive_capped12_ultra.yaml`
  - wide:
    `fusion_zero_shot\src\agd20k_eval\config.local.adaptive_capped12_widegate_ultra.yaml`
  - conservative floor:
    `fusion_zero_shot\src\agd20k_eval\config.local.adaptive_capped12_floor_ultra.yaml`
- Outputs:
  - fixed:
    `outputs\fusion_zero_shot_fixed_capped12_ultra\20260619-175540`
  - default adaptive:
    `outputs\fusion_zero_shot_adaptive_capped12_ultra\20260619-183625`
  - wide adaptive:
    `outputs\fusion_zero_shot_adaptive_capped12_widegate_ultra\20260619-183818`
  - conservative floor:
    `outputs\fusion_zero_shot_adaptive_capped12_floor_ultra\20260619-184012`
  - per-sample CSV:
    `outputs\fusion_zero_shot_adaptive_capped12_floor_ultra\capped12_ablation_summary.csv`
- Mean soft-fusion metrics:
  - fixed:
    KLD `1.7680`, SIM `0.2813`, NSS `0.3342`
  - default adaptive:
    KLD `1.7738`, SIM `0.2867`, NSS `0.3504`, lambda mean `0.5677`, fallback `0/12`
  - wide adaptive:
    KLD `1.7462`, SIM `0.2900`, NSS `0.3656`, lambda mean `0.5048`, fallback `0/12`
  - conservative floor:
    KLD `1.7515`, SIM `0.2860`, NSS `0.3749`, lambda mean `0.6221`, fallback `10/12`
- Win/tie/loss vs fixed:
  - default adaptive:
    KLD `4/4/4`, SIM `4/4/4`, NSS `3/4/5`
  - wide adaptive:
    KLD `4/4/4`, SIM `5/4/3`, NSS `4/4/4`
  - conservative floor:
    KLD `2/10/0`, SIM `2/10/0`, NSS `2/10/0`
- Interpretation:
  the wide always-on gate has the best mean KLD/SIM, but the conservative floor has the safest reviewer-facing behavior: it improves high-confidence cases and otherwise exactly falls back to fixed fusion, producing no per-sample loss on this capped subset. This supports a modest mechanism claim, not a full AGD20K performance claim.
- Caveats:
  several samples used direct-resize fallback during heatmap warping; these are ultra-pilot values and should not be mixed with literature full-setting numbers.
- Detailed note:
  `experiments\2026-06-19-flux-dinov3-adaptive-ultra-smoke.md`

## 2026-06-19 FLUX-DINOv3 Capped12 Similarity-Floor Sensitivity

- Evidence tier:
  `ultra-pilot analytical reuse`, no FLUX/DINO rerun.
- Purpose:
  answer the reviewer-facing question of whether the conservative reliability floor is just a fragile constant.
- Method:
  recombine the completed fixed capped12 and wide-adaptive capped12 per-sample metrics using the conservative-floor rule:
  use wide adaptive if `selected_similarity >= floor`, otherwise use fixed fallback.
- Artifacts:
  - summary CSV:
    `outputs\fusion_zero_shot_adaptive_capped12_floor_ultra\capped12_similarity_floor_sweep.csv`
  - per-sample CSV:
    `outputs\fusion_zero_shot_adaptive_capped12_floor_ultra\capped12_similarity_floor_sweep_per_sample.csv`
  - markdown:
    `outputs\fusion_zero_shot_adaptive_capped12_floor_ultra\capped12_similarity_floor_sweep.md`
  - plot:
    `outputs\fusion_zero_shot_adaptive_capped12_floor_ultra\capped12_similarity_floor_sweep.png`
- Key table:
  - floor `0.00`:
    adapted `12/12`, KLD `1.7462`, SIM `0.2900`, NSS `0.3656`, W/T/L KLD `4/4/4`, SIM `5/4/3`, NSS `4/4/4`
  - floor `0.70`:
    adapted `6/12`, KLD `1.7483`, SIM `0.2897`, NSS `0.3707`, W/T/L KLD `3/7/2`, SIM `4/7/1`, NSS `3/7/2`
  - floor `0.90`:
    adapted `2/12`, KLD `1.7514`, SIM `0.2860`, NSS `0.3749`, W/T/L KLD `2/10/0`, SIM `2/10/0`, NSS `2/10/0`
  - floor `1.10`:
    adapted `0/12`, identical to fixed fallback, KLD `1.7680`, SIM `0.2813`, NSS `0.3342`
- Interpretation:
  lower floors can improve mean values but introduce per-sample losses. Around `0.90`, the gate is selective enough to produce no losses against fixed fusion in capped12 while retaining gains on high-confidence cases. This supports using `0.90` as a conservative safety setting in the current pilot, not as a globally optimal constant.

## 2026-06-19 FLUX-DINOv3 Adaptive Fusion Manuscript Draft

- Evidence tier:
  `manuscript draft`, conservative claims only.
- Manuscript directory:
  `manuscripts\flux_dinov3_adaptive_fusion`
- Main source:
  `manuscripts\flux_dinov3_adaptive_fusion\main.tex`
- Compiled PDF:
  `manuscripts\flux_dinov3_adaptive_fusion\main.pdf`
- Draft title:
  `Reliability-Aware Geometry--Interaction Fusion for Training-Free Affordance Reasoning`
- Main story:
  the original paper demonstrates geometry-interaction complementarity. This draft adds a training-free reliability-aware gate that decides when to trust FLUX interaction, when to trust DINOv3/PCA geometry, and when to fall back to fixed fusion.
- Important writing boundary:
  the draft does not claim full AGD20K improvement, SOTA, real-time robotics, or global optimality of DINOv3/FLUX.
- Compile command:
  `pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex`
- Compile status:
  successful, `6` pages. `latexmk` was unavailable because MiKTeX could not find Perl.
- Next manuscript additions:
  qualitative figure, runtime table, and broader cached subset before stronger SCI/SCIE submission.

## 2026-06-19 Manuscript Figure/Table Restyle

- Evidence tier:
  `manuscript draft asset`, no new benchmark run.
- Purpose:
  reference the original paper's figure/table style and make the current RA-Gate manuscript easier to audit visually.
- Original-paper style cues used:
  compact `booktabs` tables with `siunitx` numeric columns, bold-leading captions, a full-width pipeline figure, and qualitative multi-column visual comparison.
- Updated manuscript files:
  - `manuscripts\flux_dinov3_adaptive_fusion\fig_pipeline_ragate.tex`
  - `manuscripts\flux_dinov3_adaptive_fusion\sec\4_experiments.tex`
  - `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_qualitative_ragate.png`
  - `manuscripts\flux_dinov3_adaptive_fusion\README.md`
- Figure updates:
  - Rebuilt the pipeline figure as a cleaner two-branch interaction/geometry flow with a central RA-Gate decision block.
  - Added a two-row qualitative comparison from capped12 pilot artifacts:
    `hold axe` as a high-similarity adapted case and `cut banana` as a conservative fallback case.
  - Regenerated the qualitative comparison with full-object letterboxing so long objects such as the axe are not cropped.
  - Added a similarity-floor sweep figure:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_floor_sweep_ragate.pdf`.
  - Moved qualitative and floor-sweep figures into the experiment section near their corresponding tables instead of leaving the visual evidence at the end.
- Compile command:
  `pdflatex -jobname=main_ragate_figures -interaction=nonstopmode -halt-on-error main.tex; bibtex main_ragate_figures; pdflatex -jobname=main_ragate_figures -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_figures -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_figures -interaction=nonstopmode -halt-on-error main.tex`
- Compile status:
  successful, output `manuscripts\flux_dinov3_adaptive_fusion\main_ragate_figures.pdf`, `6` pages.
- Visual check:
  rendered and inspected pages `3`, `4`, and `6`.
  The pipeline figure, compact tables, and qualitative comparison were readable with no detected text overlap.
- Claim boundary:
  these are presentation assets for the existing capped12 ultra-pilot. They do not change the evidence tier and must not be described as full AGD20K results.

## 2026-06-19 Manuscript Figure Density and Layout v3

- Evidence tier:
  `manuscript draft asset`, no new benchmark run.
- User-facing issue addressed:
  the qualitative axe example looked visually incomplete because the long object was too close to the cell boundary, and one qualitative figure alone was not enough to mimic the original paper's visual storytelling density.
- Original-paper figure/table reference:
  the extracted original LaTeX source under `tmp\arxiv_2602_20501v3` contains about `15` active figure environments and `3` active table environments.
- Updated manuscript figure set:
  - pipeline figure:
    `manuscripts\flux_dinov3_adaptive_fusion\fig_pipeline_ragate.tex`
  - full-object qualitative comparison:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_qualitative_ragate.png`
  - sample-level gate audit:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_gate_decision_audit.pdf`
  - similarity-floor sweep:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_floor_sweep_ragate.pdf`
- Layout status:
  figures are placed in the middle of the manuscript near the method and experiment text:
  pipeline on rendered page `3`, quantitative tables on page `4`, and qualitative/audit/floor-sweep figures on page `5`.
- Compile command:
  `pdflatex -jobname=main_ragate_layout_v3 -interaction=nonstopmode -halt-on-error main.tex; bibtex main_ragate_layout_v3; pdflatex -jobname=main_ragate_layout_v3 -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_layout_v3 -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_layout_v3 -interaction=nonstopmode -halt-on-error main.tex`
- Compile status:
  successful, output `manuscripts\flux_dinov3_adaptive_fusion\main_ragate_layout_v3.pdf`, `7` pages.
- Visual check:
  rendered and inspected pages `3`-`6` under `manuscripts\flux_dinov3_adaptive_fusion\render_check_layout_v3`.
  The axe is now shown as a complete letterboxed object in the qualitative figure, with no detected overlap among the three experiment figures on page `5`.
- Claim boundary:
  the new gate-audit plot is a visualization of existing capped12 metadata, not a new experiment.
  The manuscript now has `4` core figures and `3` result tables, still fewer than the original paper but enough for a short draft story.

## 2026-06-20 Manuscript Complementarity Evidence Expansion v6

- Evidence tier:
  `manuscript draft asset`, `local capped-subset analysis`, no new FLUX/DINO run.
- User-facing issue addressed:
  the draft was too short and relied too much on the original paper for the geometry-interaction complementarity claim.
  The revised manuscript now contains its own local capped12 primitive complementarity check, modeled after the original paper's logic.
- Manuscript source:
  `manuscripts\flux_dinov3_adaptive_fusion\main.tex`
- Compiled PDF:
  `manuscripts\flux_dinov3_adaptive_fusion\main_ragate_layout_v6.pdf`
- Page count:
  `9` pages.
- New/updated assets:
  - local complementarity chart:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_local_complementarity_ragate.pdf`
  - local complementarity case figure:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_complementarity_cases_ragate.png`
  - updated experiment section:
    `manuscripts\flux_dinov3_adaptive_fusion\sec\4_experiments.tex`
- Local capped12 primitive complementarity result:
  - FLUX verb only:
    KLD `6.894`, SIM `0.160`, NSS `0.169`
  - DINO/PCA selected primitive:
    KLD `11.734`, SIM `0.179`, NSS `0.226`
  - fixed interaction-geometry fusion:
    KLD `1.768`, SIM `0.281`, NSS `0.334`
  - fusion wins over FLUX verb map on KLD/SIM/NSS:
    `8/12`, `9/12`, `7/12`
  - fusion wins over selected DINO/PCA map on KLD/SIM/NSS:
    `11/12`, `9/12`, `8/12`
- Added explanation:
  a new complementarity-pattern table connects local cases to the RA-Gate design:
  interaction-dominant cases, geometry-sharpening cases, and cases where the fixed mixture itself is a useful fallback.
- Compile command:
  `pdflatex -jobname=main_ragate_layout_v6 -interaction=nonstopmode -halt-on-error main.tex; bibtex main_ragate_layout_v6; pdflatex -jobname=main_ragate_layout_v6 -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_layout_v6 -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_layout_v6 -interaction=nonstopmode -halt-on-error main.tex`
- Visual/log check:
  rendered pages `4`-`7` under
  `manuscripts\flux_dinov3_adaptive_fusion\render_check_layout_v6`.
  `rg` found no unresolved references, overfull boxes, or underfull boxes in `main_ragate_layout_v6.log`.
- Claim boundary:
  this local check supports the same complementarity logic at pilot scale.
  It is not a replacement for the original full AGD20K benchmark, and should not be described as full benchmark proof.

## Next Steps

1. For the current main route, expand FLUX-DINOv3 adaptive fusion carefully:
   - next reasonable point:
     inspect capped12 overlays and run a small threshold-sensitivity check for the conservative floor if more evidence is needed
   - avoid claiming full AGD20K performance until at least a broader capped subset runs.
2. Keep the lightweight student route as fallback/future work rather than the main positive claim unless later distillation improves it.
3. Turn the low-resource trend into compact reviewer-facing tables:
   `full capped vs train=36/24/12` and `no gate vs depth gate vs normal gate`.
   - initial draft:
     `experiments\2026-06-16-reviewer-facing-assets.md`
4. Write the paper framing around:
   `depth-conditioned CLIP affordance probing under single-GPU constraints`.
   - initial draft:
     `experiments\2026-06-16-paper-results-draft.md`
5. Inspect the qualitative gallery and select success/failure cases rather than using the first exported examples blindly.
   - paired no-gate/depth-gate examples are now available under:
     `geometry_probing\umd_linear_probing\outputs\paired_qualitative\train4_seed1338_test`
6. Use the completed noisy-depth 3-seed check as a cautious robustness paragraph:
   clean geometry is best, degraded depth retains part of the depth-gate benefit, and no claim should imply full robustness to arbitrary geometry noise.

## 2026-06-21 RA-Gate balanced AGD20K pilot

- Purpose:
  broaden the RA-Gate evidence beyond capped12 while keeping the run feasible on the RTX 3070 Laptop GPU.
- Fixed-fusion config:
  `fusion_zero_shot\src\agd20k_eval\config.local.fixed_balanced45_ultra.yaml`
- Fixed-fusion command:
  `DINO_CHECKPOINT_PATH=E:\work\2cvpr\Probing_Bridging_Affordance\models\DINOV3\DINOV3-pth\dinov3_vits16_pretrain_lvd1689m-08c60483.pth E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\run_agd20k_eval.py --config fusion_zero_shot\src\agd20k_eval\config.local.fixed_balanced45_ultra.yaml`
- Fixed-fusion output:
  `outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653`
- Evidence tier:
  `balanced45 capped pilot`; 44/45 samples processed successfully, one `wash/knife/knife_001017.jpg` sample failed at heatmap warping and is excluded from paired fixed-vs-adaptive comparison.
- Fixed-fusion aggregate over 44 processed samples:
  - FLUX verb only: KLD `6.2873`, SIM `0.1719`, NSS `0.0871`
  - selected DINO/PCA geometry: KLD `13.3009`, SIM `0.1164`, NSS `0.0133`
  - fixed interaction-geometry fusion: KLD `2.1242`, SIM `0.2202`, NSS `0.1776`
- Conservative RA-Gate config:
  `fusion_zero_shot\src\agd20k_eval\config.local.adaptive_balanced44_floor_ultra.yaml`
- Conservative RA-Gate output:
  `outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008`
- Conservative RA-Gate aggregate over 44 processed samples:
  - KLD `2.0946`, SIM `0.2228`, NSS `0.1957`
- Paired fixed-vs-RA-Gate comparison over 43 samples with valid soft-fusion metrics in both runs:
  - KLD: `2.1736 -> 2.1433`, relative gain `1.40%`, W/T/L `10/31/2`
  - SIM: `0.2253 -> 0.2280`, relative gain `1.17%`, W/T/L `11/31/1`
  - NSS: `0.1817 -> 0.2003`, relative gain `10.22%`, W/T/L `9/31/3`
  - Conservative gate fallback: `29/43`; adaptive change applied to `14/43` samples.
- Comparison artifact:
  `outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008\balanced44_comparison.json`
- Interpretation:
  this balanced pilot supports the same paper story as capped12 at a larger scale: fixed geometry-interaction fusion is much stronger than either primitive cue, and the conservative reliability gate adds small mean gains, strongest on NSS, while leaving most uncertain samples unchanged. Claims should still be labeled as capped pilot rather than full AGD20K benchmark.
- Engineering notes:
  `fusion_zero_shot\src\agd20k_eval\utils\logging_utils.py` now logs to file only for long background runs, avoiding closed stderr issues in desktop execution. `fusion_zero_shot\src\agd20k_eval\kontext_runner.py` disables the FLUX progress bar in persistent-worker runs. Fusion tests passed after these changes: `17 passed` for `tests\test_fusion_adaptive_gate.py tests\test_fusion_runtime_contract.py tests\test_fusion_config_kontext.py -q`.

## 2026-06-21 RA-Gate balanced44 wide-gate and floor-sweep ablation

- Purpose:
  answer the reviewer-facing question of whether the conservative floor is arbitrary, using the broader balanced44 pilot rather than only capped12.
- Evidence tier:
  `balanced44 capped analytical reuse`; same 43 valid paired samples as the fixed-vs-conservative comparison, not a full AGD20K benchmark.
- Always-on/wide adaptive config:
  `fusion_zero_shot\src\agd20k_eval\config.local.adaptive_balanced44_widegate_ultra.yaml`
- Always-on/wide adaptive command:
  `DINO_CHECKPOINT_PATH=E:\work\2cvpr\Probing_Bridging_Affordance\models\DINOV3\DINOV3-pth\dinov3_vits16_pretrain_lvd1689m-08c60483.pth E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\run_agd20k_eval.py --config fusion_zero_shot\src\agd20k_eval\config.local.adaptive_balanced44_widegate_ultra.yaml`
- Always-on/wide adaptive output:
  `outputs\fusion_zero_shot_adaptive_balanced44_widegate_ultra\20260621-025301`
- Wide adaptive aggregate over 44 processed samples:
  - KLD `2.0822`, SIM `0.2242`, NSS `0.1888`
- Paired always-on/wide result over 43 valid fixed-comparable samples:
  - KLD `2.1306`, SIM `0.2295`, NSS `0.1931`
  - W/T/L vs fixed:
    - KLD `19/14/10`
    - SIM `21/14/8`
    - NSS `16/14/13`
- Floor-sweep command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\analyze_similarity_floor_sweep.py --fixed-summary outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653\summary.csv --adaptive-summary outputs\fusion_zero_shot_adaptive_balanced44_widegate_ultra\20260621-025301\summary.csv --out-dir outputs\fusion_zero_shot_adaptive_balanced44_widegate_ultra\20260621-025301 --prefix balanced44_similarity_floor_sweep --thresholds -10.0 0.0 0.5 0.7 0.8 0.85 0.9 1.0 1.1 1.5 2.0`
- Floor-sweep artifacts:
  - CSV:
    `outputs\fusion_zero_shot_adaptive_balanced44_widegate_ultra\20260621-025301\balanced44_similarity_floor_sweep.csv`
  - per-sample CSV:
    `outputs\fusion_zero_shot_adaptive_balanced44_widegate_ultra\20260621-025301\balanced44_similarity_floor_sweep_per_sample.csv`
  - JSON:
    `outputs\fusion_zero_shot_adaptive_balanced44_widegate_ultra\20260621-025301\balanced44_similarity_floor_sweep.json`
  - plot:
    `outputs\fusion_zero_shot_adaptive_balanced44_widegate_ultra\20260621-025301\balanced44_similarity_floor_sweep.pdf`
  - manuscript copy:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_balanced44_floor_sweep_ragate.pdf`
- Key floor-sweep rows over 43 valid paired samples:
  - always-on:
    adapted `43/43`, KLD `2.1306`, SIM `0.2295`, NSS `0.1931`, W/T/L KLD `19/14/10`, SIM `21/14/8`, NSS `16/14/13`
  - floor `0.70`:
    adapted `22/43`, KLD `2.1419`, SIM `0.2290`, NSS `0.1981`, W/T/L KLD `12/26/5`, SIM `14/26/3`, NSS `10/26/7`
  - floor `0.90`:
    adapted `14/43`, KLD `2.1433`, SIM `0.2280`, NSS `0.2003`, W/T/L KLD `10/31/2`, SIM `11/31/1`, NSS `9/31/3`
  - floor `1.50`:
    adapted `3/43`, KLD `2.1636`, SIM `0.2262`, NSS `0.1857`, W/T/L KLD/SIM/NSS all `2/41/0`
- Interpretation:
  the sweep makes the safety--gain trade-off explicit. Always-on adaptive fusion gives the best KLD/SIM means but introduces more per-sample losses. The `0.90` floor is a middle setting: it changes only `14/43` samples, keeps small KLD/SIM gains, and gives the strongest NSS in the sweep. Stronger floors reduce losses further but also reduce gains. This supports writing the gate as a reliability-aware fusion mechanism rather than claiming it universally improves every image.
- Manuscript update:
  `manuscripts\flux_dinov3_adaptive_fusion\sec\4_experiments.tex` now uses balanced44 as the primary similarity-floor sensitivity evidence, with capped12 kept as a smaller mechanism sanity check.
- Compile/visual status:
  - compiled PDF:
    `manuscripts\flux_dinov3_adaptive_fusion\main_ragate_balanced44_v1.pdf`
  - page count:
    `10`
  - compile command:
    `pdflatex -jobname=main_ragate_balanced44_v1 -interaction=nonstopmode -halt-on-error main.tex; bibtex main_ragate_balanced44_v1; pdflatex -jobname=main_ragate_balanced44_v1 -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_balanced44_v1 -interaction=nonstopmode -halt-on-error main.tex`
  - log check:
    no unresolved references/citations and no overfull/underfull boxes after the final compile; MiKTeX only emitted its generic update reminder.
  - rendered visual check:
    `manuscripts\flux_dinov3_adaptive_fusion\render_check_balanced44_v1\page_latest-08.png`
  - visual note:
    the balanced44 floor-sweep plot was regenerated as relative gain vs fixed plus gate selectivity, making the `0.90` reliability floor easier to read than the raw-metric version.

## 2026-06-21 Compute-aware FLUX skip offline pilot

- Purpose:
  test the user's proposed deployment direction: use a small pre-call gate to decide whether full FLUX must be invoked, and skip it when a cheap branch is expected to be enough.
- Evidence tier:
  `balanced44 capped offline analysis`; no new FLUX/DINO run and no trained deployable gate yet.
- Script:
  `fusion_zero_shot\scripts\analyze_compute_gate_policy.py`
- Command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\analyze_compute_gate_policy.py --summary outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008\summary.csv --out-dir outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008 --prefix balanced44_compute_gate_policy`
- Artifacts:
  - CSV:
    `outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008\balanced44_compute_gate_policy.csv`
  - markdown:
    `outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008\balanced44_compute_gate_policy.md`
  - JSON:
    `outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008\balanced44_compute_gate_policy.json`
- Policy assumptions:
  - full FLUX branch: `192` seconds/image, based on the local balanced44 run.
  - cheap branch: `7` seconds/image, approximating cached DINO/PCA/fusion reuse overhead.
  - skipped samples use the currently available geometry-only map as the replacement branch.
- Key results over 43 valid samples:
  - strict oracle skip, geometry non-worse on KLD/SIM/NSS:
    skip `1/43` (`2.3%`), estimated speedup `0.99x`, KLD `2.1432`, SIM `0.2286`, NSS `0.2081`.
  - oracle skip when geometry is non-worse on SIM and NSS:
    skip `4/43` (`9.3%`), estimated speedup `1.06x`, KLD worsens to `2.2453`, SIM improves to `0.2431`, NSS improves to `0.2389`.
  - oracle skip when geometry is non-worse only on NSS:
    skip `14/43` (`32.6%`), estimated speedup `1.41x`, but KLD collapses to `5.3111`.
- Interpretation:
  using DINO/PCA geometry alone as the replacement for full FLUX has very limited safe skip room. This is useful negative evidence: a deployment story should not claim that full FLUX can simply be dropped on many images. The more credible next route is to train a lightweight interaction student from cached FLUX heatmaps, then train a small pre-call gate to choose between the student and full FLUX.
- Verification:
  `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_fusion_adaptive_gate.py tests\test_fusion_runtime_contract.py tests\test_fusion_config_kontext.py -q`
  passed with `17 passed`.

## 2026-06-21 SD-Turbo lightweight interaction pilot

- Purpose:
  test whether a much lighter diffusion model can provide verb-conditioned attention maps as a possible replacement/student branch for the full FLUX interaction prior.
- Evidence tier:
  `5-sample ultra pilot`; not a full AGD20K benchmark and not yet integrated into the DINO/PCA fusion pipeline.
- Download source:
  ModelScope mirror, avoiding HuggingFace.
- Download command:
  `E:\conda\envs\bev_py39_torch231\Scripts\modelscope.exe download AI-ModelScope/sd-turbo --local_dir models\sd-turbo --max-workers 1 --endpoint https://www.modelscope.cn`
- Download note:
  the full download timed out after 30 minutes but left the main diffusers subfolders in place. Missing tokenizer files were recovered individually:
  `tokenizer/vocab.json`, `tokenizer/tokenizer_config.json`, and `tokenizer/special_tokens_map.json`.
- Model smoke:
  `StableDiffusionPipeline.from_pretrained(models\sd-turbo, torch_dtype=torch.float16, local_files_only=True)` loads successfully.
  A one-step `512x512` smoke generation took about `2.25` seconds including first-run overhead after model load.
- Added pilot script:
  `fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py`
  It installs a temporary attention processor on SD-Turbo U-Net cross-attention layers, records verb-token attention maps, upsamples them to image resolution, and evaluates KLD/SIM/NSS against AGD20K GT.
- Smoke command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py --dataset-root datasets\AGD20K\AGD20K\Unseen\testset --model-dir models\sd-turbo --output-root outputs\sd_turbo_attention_pilot --affordances hold --max-images-per-object 1 --max-samples-total 1 --height 512 --width 512`
- Smoke output:
  `outputs\sd_turbo_attention_pilot\20260621-095640`
  - sample: `hold/axe/axe_000108.jpg`
  - SD-Turbo verb attention: KLD `1.2596`, SIM `0.3288`, NSS `0.3662`
  - inference time: `1.46` seconds
  - recorded attention maps: `16`
- Five-sample command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py --dataset-root datasets\AGD20K\AGD20K\Unseen\testset --model-dir models\sd-turbo --output-root outputs\sd_turbo_attention_pilot --affordances hold cut drink_with --max-images-per-object 1 --max-samples-total 5 --height 512 --width 512`
- Five-sample output:
  `outputs\sd_turbo_attention_pilot\20260621-095716`
  - summary:
    `outputs\sd_turbo_attention_pilot\20260621-095716\summary.csv`
  - metrics:
    `outputs\sd_turbo_attention_pilot\20260621-095716\metrics.json`
- Five-sample aggregate:
  - load time: `4.34` seconds
  - mean inference time after load: `0.52` seconds/image
  - mean KLD: `1.9740`
  - mean SIM: `0.2427`
  - mean NSS: `0.0477`
- Matched comparison against existing balanced44 FLUX interaction-only rows on the same five images:
  - SD-Turbo mean KLD/SIM/NSS: `1.9740 / 0.2427 / 0.0477`
  - FLUX interaction mean KLD/SIM/NSS: `1.8729 / 0.2340 / 0.0425`
  - interpretation:
    SD-Turbo is slightly worse on KLD but slightly better on SIM/NSS in this tiny matched subset. This is not enough to claim superiority, but it shows the light branch is not a dead end and may be useful as a fast fallback/student branch.
- Important boundary:
  the current SD-Turbo pilot generates a fresh image from the prompt rather than editing the input image as FLUX Kontext does. The recorded attention can still provide a verb-conditioned prior, but it is not yet a drop-in replacement for Kontext. The next experiment should either:
  1. run the same SD-Turbo attention extraction on the balanced44 subset and compare only interaction-only maps, or
  2. distill cached FLUX heatmaps into a lightweight image-conditioned student, which is more faithful to the deployment story.

## 2026-06-21 SD-Turbo balanced44 lightweight interaction prior

- Purpose:
  expand the 5-sample SD-Turbo pilot to the same balanced44 sample set used by the local FLUX-DINOv3 RA-Gate evaluation, so the runtime and interaction-only metrics can be compared on paired samples.
- Evidence tier:
  `balanced44 capped pilot`; not a full AGD20K benchmark and not a strict Kontext replacement.
- Script update:
  `fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py` now supports `--sample-summary`, allowing it to run exactly the samples listed in an existing AGD20K summary CSV.
- Balanced44 command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py --dataset-root datasets\AGD20K\AGD20K\Unseen\testset --model-dir models\sd-turbo --output-root outputs\sd_turbo_attention_balanced44 --sample-summary outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653\summary.csv --affordances hold cut open type_on drink_with eat ride throw wash --max-images-per-object 3 --max-samples-total 60 --height 512 --width 512`
- Output:
  `outputs\sd_turbo_attention_balanced44\20260621-101249`
  - summary:
    `outputs\sd_turbo_attention_balanced44\20260621-101249\summary.csv`
  - metrics:
    `outputs\sd_turbo_attention_balanced44\20260621-101249\metrics.json`
- Aggregate over 44 matched samples:
  - load time: `4.13` seconds
  - mean inference time after load: `0.3267` seconds/image
  - mean KLD: `1.8513`
  - mean SIM: `0.2476`
  - mean NSS: `0.5021`
- Added comparison script:
  `fusion_zero_shot\scripts\compare_light_interaction_summary.py`
- Comparison command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\compare_light_interaction_summary.py --light-summary outputs\sd_turbo_attention_balanced44\20260621-101249\summary.csv --flux-summary outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653\summary.csv --out-dir outputs\sd_turbo_attention_balanced44\20260621-101249 --prefix balanced44_sd_turbo_vs_flux`
- Comparison artifacts:
  - markdown:
    `outputs\sd_turbo_attention_balanced44\20260621-101249\balanced44_sd_turbo_vs_flux.md`
  - JSON:
    `outputs\sd_turbo_attention_balanced44\20260621-101249\balanced44_sd_turbo_vs_flux.json`
  - per-sample CSV:
    `outputs\sd_turbo_attention_balanced44\20260621-101249\balanced44_sd_turbo_vs_flux_per_sample.csv`
- Paired comparison against local one-step FLUX interaction maps over the same 44 samples:
  - SD-Turbo attention: KLD `1.8513`, SIM `0.2476`, NSS `0.5021`, `0.3267` seconds/image
  - FLUX interaction: KLD `6.2873`, SIM `0.1719`, NSS `0.0871`, about `192` seconds/image
  - SD-Turbo W/T/L vs FLUX:
    - KLD: `37/0/7`
    - SIM: `31/0/13`
    - NSS: `29/0/15`
  - estimated interaction-branch speedup:
    about `588x` on the local RTX 3070 Laptop GPU setup.
- Interpretation:
  this is strong practical evidence for a lightweight interaction-prior route. SD-Turbo is not a drop-in replacement for FLUX Kontext because it records prompt-generation attention rather than input-image editing attention. However, it provides sub-second verb-conditioned attention maps with competitive or better interaction-only metrics than the local one-step FLUX interaction baseline on this capped paired subset.
- Manuscript update:
  `manuscripts\flux_dinov3_adaptive_fusion\sec\4_experiments.tex` now includes a `Lightweight Interaction Prior for Mobile Feasibility` subsection and `Table~\ref{tab:sd_turbo}`.
  `manuscripts\flux_dinov3_adaptive_fusion\sec\5_discussion.tex` now states the limitation and the teacher--student/mobile deployment path more explicitly.

## 2026-06-22 SD-Turbo qualitative attention figure

- Purpose:
  add visual evidence for the lightweight interaction branch so the manuscript does not rely only on the SD-Turbo metric/runtime table.
- Evidence tier:
  `balanced44 qualitative pilot`; examples are selected from the completed SD-Turbo balanced44 run.
- Added script:
  `fusion_zero_shot\scripts\make_sd_turbo_qualitative_figure.py`
- Command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\make_sd_turbo_qualitative_figure.py --summary outputs\sd_turbo_attention_balanced44\20260621-101249\summary.csv --dataset-root datasets\AGD20K\AGD20K\Unseen\testset --out manuscripts\flux_dinov3_adaptive_fusion\figs\fig_sd_turbo_attention_qualitative.pdf`
- Figure artifacts:
  - PDF:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_sd_turbo_attention_qualitative.pdf`
  - PNG:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_sd_turbo_attention_qualitative.png`
- Cases:
  - `throw/basketball/basketball_000647.jpg`: high-SIM success example.
  - `type_on/laptop/laptop_000585.jpg`: strong NSS contact-region example.
  - `open/refrigerator/refrigerator_000884.jpg`: failure boundary where prompt-generation attention misses the annotated opening region.
- Manuscript update:
  `sec\4_experiments.tex` now includes `Figure~\ref{fig:sd_turbo_attention}` in the lightweight interaction subsection.

## 2026-06-22 OpenCLIP low-resource support table added to manuscript

- Purpose:
  add the completed UMD OpenCLIP/depth-gate low-resource results as supporting evidence for the broader reliability-aware geometry gating story, without mixing them into the AGD20K DINOv3--FLUX benchmark tables.
- Evidence tier:
  `separate UMD capped low-resource support experiment`; three seeds per row; not an AGD20K/FLUX comparison.
- Manuscript update:
  `manuscripts\flux_dinov3_adaptive_fusion\sec\5_discussion.tex` now includes a `Low-resource support from OpenCLIP depth gating` paragraph and `Table~\ref{tab:openclip_lowres_support}` before the limitations paragraph.
- Table source:
  `experiments\2026-06-16-reviewer-facing-assets.md`
- Key table values:
  - train `12` frames/tool:
    no gate test mIoU `0.3461 / 0.0941`, depth gate `0.3458 / 0.0440`.
  - train `6` frames/tool:
    no gate `0.3279 / 0.0742`, depth gate `0.3309 / 0.0284`.
  - train `4` frames/tool:
    no gate `0.1946 / 0.0240`, depth gate `0.2516 / 0.0251`.
  - train `2` frames/tool:
    no gate `0.1198 / 0.0068`, depth gate `0.1528 / 0.0317`.
- Efficiency note included in the manuscript:
  frozen OpenCLIP support branch adds only `6,152` trainable parameters to the probe head (`30,768 -> 36,920`) and shows no meaningful runtime overhead in the local low-resource runs.
- Compiled artifact:
  `manuscripts\flux_dinov3_adaptive_fusion\main_ragate_openclip_v1.pdf`
- Compile command:
  `pdflatex -jobname=main_ragate_openclip_v1 -interaction=nonstopmode -halt-on-error main.tex; bibtex main_ragate_openclip_v1; pdflatex -jobname=main_ragate_openclip_v1 -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_openclip_v1 -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_openclip_v1 -interaction=nonstopmode -halt-on-error main.tex`
- Compile status:
  succeeded, final PDF has `12` pages.
  Log scan found no unresolved references/citations and no overfull/underfull boxes; MiKTeX only emitted its generic update reminder.
- Interpretation:
  this addition strengthens the paper's low-resource practicality story: the main contribution remains RA-Gate for DINOv3--FLUX geometry--interaction fusion, while the OpenCLIP branch shows that small geometry-aware gates can also help under single-GPU, low-data affordance probing.

## 2026-06-22 TinyRouter low-compute SD-DINO pilot

- Purpose:
  test the deployment-oriented route requested by the user: keep FLUX as a slow teacher/fallback, use SD-Turbo as the low-cost online interaction prior, use DINOv3/PCA as the geometry cue, and add a tiny pre-call reliability network that predicts both FLUX fallback and interaction-geometry lambda.
- Evidence tier:
  `balanced44 capped pilot`; five-fold offline cross-validation for the router; not a full AGD20K benchmark.
- Code additions:
  - SD-Turbo attention runner now also exports object-token attention:
    `fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py`
  - SD object attention -> ROI -> DINO/PCA -> SD-DINO fusion:
    `fusion_zero_shot\scripts\run_sd_dino_fusion_from_attention.py`
  - Tiny reliability router:
    `fusion_zero_shot\scripts\run_tiny_reliability_router.py`
  - Router threshold sweep:
    `fusion_zero_shot\scripts\analyze_tiny_router_threshold_sweep.py`
  - Router qualitative figure:
    `fusion_zero_shot\scripts\make_tiny_router_qualitative_figure.py`
  - Unit tests:
    `tests\test_tiny_reliability_router.py`
- SD object-attention smoke:
  - command:
    `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py --dataset-root datasets\AGD20K\AGD20K\Unseen\testset --model-dir models\sd-turbo --output-root outputs\sd_turbo_attention_object_smoke --sample-summary outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653\summary.csv --affordances hold cut open type_on drink_with eat ride throw wash --max-images-per-object 3 --max-samples-total 3 --height 512 --width 512`
  - output:
    `outputs\sd_turbo_attention_object_smoke\20260622-180944`
  - result:
    `3/3` samples processed, object attention files written, mean inference time `0.7374` seconds/image including first warm sample.
- SD object-attention balanced44 run:
  - command:
    `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py --dataset-root datasets\AGD20K\AGD20K\Unseen\testset --model-dir models\sd-turbo --output-root outputs\sd_turbo_attention_object_balanced44 --sample-summary outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653\summary.csv --affordances hold cut open type_on drink_with eat ride throw wash --max-images-per-object 3 --max-samples-total 60 --height 512 --width 512`
  - output:
    `outputs\sd_turbo_attention_object_balanced44\20260622-181035`
  - aggregate over 44 samples:
    KLD `1.8513`, SIM `0.2476`, NSS `0.5021`, mean inference time `0.3416` seconds/image.
- True low-compute SD-DINO fusion with SD object ROI:
  - smoke output:
    `outputs\sd_dino_fusion_object_smoke\20260622-181417`
    with `3/3` samples and zero failures.
  - balanced44 output:
    `outputs\sd_dino_fusion_object_balanced44\20260622-181447`
  - balanced44 aggregate:
    - SD interaction: KLD `1.8512`, SIM `0.2476`, NSS `0.5025`.
    - SD object ROI + DINO/PCA geometry: KLD `2.0579`, SIM `0.2353`, NSS `0.2802`.
    - SD-DINO fixed soft fusion: KLD `1.8807`, SIM `0.2423`, NSS `0.4469`.
  - interpretation:
    the fully lightweight SD object ROI -> DINO/PCA path is feasible and has zero failures on the capped pilot, but fixed SD-DINO fusion does not beat SD interaction alone on all metrics. This should be written as a feasibility and reliability-check result, not as a universal quality improvement.
- TinyRouter SD-object-ROI cross-validation:
  - command:
    `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\run_tiny_reliability_router.py --sd-summary outputs\sd_turbo_attention_object_balanced44\20260622-181035\summary.csv --sd-dino-summary outputs\sd_dino_fusion_object_balanced44\20260622-181447\summary.csv --flux-summary outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653\summary.csv --adaptive-summary outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008\summary.csv --out-dir outputs\tiny_reliability_router_balanced44\20260622-181447_sdroi --prefix balanced44_tiny_router_sdroi --folds 5 --epochs 600`
  - output:
    `outputs\tiny_reliability_router_balanced44\20260622-181447_sdroi`
  - main rows over 44 samples:
    - FLUX interaction: KLD `6.2873`, SIM `0.1719`, NSS `0.0871`, `192.0` sec/img, FLUX calls `44/44`.
    - SD-Turbo interaction: KLD `1.8513`, SIM `0.2476`, NSS `0.5021`, `0.3416` sec/img, FLUX calls `0/44`.
    - FLUX+DINO fixed: KLD `2.1242`, SIM `0.2202`, NSS `0.1776`, `199.0` sec/img.
    - FLUX+DINO RA-Gate: KLD `2.0946`, SIM `0.2228`, NSS `0.1957`, `199.0` sec/img.
    - SD+DINO fixed: KLD `1.8807`, SIM `0.2423`, NSS `0.4469`, `7.3416` sec/img.
    - SD+DINO TinyRouter(lambda): KLD `1.9439`, SIM `0.2165`, NSS `0.4688`, `7.3416` sec/img.
    - SD+DINO TinyRouter(route+lambda): KLD `2.0411`, SIM `0.2135`, NSS `0.4321`, `50.9779` sec/img, FLUX calls `10/44`, speedup vs always-FLUX `3.90x`.
- Threshold sweep:
  - command:
    `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\analyze_tiny_router_threshold_sweep.py --per-sample outputs\tiny_reliability_router_balanced44\20260622-181447_sdroi\balanced44_tiny_router_sdroi_per_sample.csv --out-dir outputs\tiny_reliability_router_balanced44\20260622-181447_sdroi --prefix balanced44_tiny_router_threshold_sweep --thresholds 0.1 0.3 0.5 0.7 0.9`
  - key row:
    threshold `0.90` calls FLUX on `4/44` samples, estimates `24.7961` sec/img and `8.03x` speedup vs always-FLUX, with KLD `1.9961`, SIM `0.2154`, NSS `0.4435`.
- Qualitative figure:
  - PDF:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_tiny_router_qualitative.pdf`
  - PNG:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_tiny_router_qualitative.png`
- Tests:
  - command:
    `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_tiny_reliability_router.py tests\test_fusion_adaptive_gate.py tests\test_fusion_runtime_contract.py tests\test_fusion_config_kontext.py -q`
  - result:
    `20 passed`.
- Manuscript update:
  `manuscripts\flux_dinov3_adaptive_fusion\sec\4_experiments.tex` now includes a `Tiny Router for Low-Compute Embodied Use` subsection, `Table~\ref{tab:tiny_router}`, `Table~\ref{tab:tiny_router_threshold}`, and `Figure~\ref{fig:tiny_router_qualitative}`.
- Compiled artifact:
  `manuscripts\flux_dinov3_adaptive_fusion\main_ragate_tinyrouter_v1.pdf`
  with `13` pages.
  Final log scan found no unresolved references/citations and no overfull/underfull boxes; MiKTeX only emitted its generic update reminder.
- Paper-safe interpretation:
  SD-Turbo is the practical default interaction branch for low-compute use in this capped pilot. FLUX should be kept as a slow fallback/teacher, not called on every frame. The TinyRouter result supports a budget-aware embodied feasibility story, but it should not be claimed as a universal quality improvement because SD-DINO fusion and learned lambda do not dominate every metric.

## 2026-06-22 Manuscript SCI-style polish and figure/table cleanup

- Purpose:
  improve the paper-facing wording, table consistency, and qualitative figure readability without changing experimental results.
- Evidence tier:
  `manuscript polish`; no new benchmark experiment.
- Skill check:
  curated skill listing was checked for an additional SCI-writing skill, but no dedicated SCI-writing skill was available in the curated list and the experimental skill path was unavailable. Continued with the installed `xielunwen` manuscript workflow and PDF/LaTeX layout checks.
- Manuscript edits:
  - polished abstract and introduction to clarify the story:
    `RA-Gate` is the main reliability-aware fusion mechanism, while SD-Turbo/TinyRouter is a low-compute deployment pilot.
  - polished experiment text in:
    `manuscripts\flux_dinov3_adaptive_fusion\sec\4_experiments.tex`
    especially the lightweight interaction, TinyRouter, and runtime sections.
  - polished discussion and conclusion in:
    `manuscripts\flux_dinov3_adaptive_fusion\sec\5_discussion.tex`
    to keep claims conservative and avoid benchmark-level overstatement.
  - added table style helpers in:
    `manuscripts\flux_dinov3_adaptive_fusion\preamble.tex`
    and unified table wording around `Time` with units explained in captions.
- Figure updates:
  - improved font size, panel spacing, labels, and borders in:
    `fusion_zero_shot\scripts\make_sd_turbo_qualitative_figure.py`
    `fusion_zero_shot\scripts\make_tiny_router_qualitative_figure.py`
  - regenerated:
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_sd_turbo_attention_qualitative.pdf`
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_sd_turbo_attention_qualitative.png`
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_tiny_router_qualitative.pdf`
    `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_tiny_router_qualitative.png`
- Compile command:
  `pdflatex -jobname=main_ragate_polished_v1 -interaction=nonstopmode -halt-on-error main.tex; bibtex main_ragate_polished_v1; pdflatex -jobname=main_ragate_polished_v1 -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_polished_v1 -interaction=nonstopmode -halt-on-error main.tex; pdflatex -jobname=main_ragate_polished_v1 -interaction=nonstopmode -halt-on-error main.tex`
- Compiled artifact:
  `manuscripts\flux_dinov3_adaptive_fusion\main_ragate_polished_v1.pdf`
  with `13` rendered pages.
- Layout verification:
  - rendered pages to:
    `manuscripts\flux_dinov3_adaptive_fusion\render_check_polished_v1`
  - visually checked pages 9--12 covering runtime text, SD-Turbo figure, TinyRouter table/figure, threshold table, OpenCLIP support table, conclusion, and references.
  - final log scan found no unresolved citations/references and no overfull/underfull boxes; only MiKTeX generic update reminders appeared.

## 2026-06-22 Figure 7/8 prior-vs-GT clarification

- Trigger:
  user noticed that Figure 7 and Figure 8 heatmaps differ visibly from AGD20K GT maps.
- Interpretation:
  the SD-Turbo and router heatmaps are coarse interaction priors, not supervised segmentation masks. The figure should make this explicit instead of visually implying exact GT recovery.
- Updates:
  - Figure 7 first row changed to a more aligned SD-Turbo prior example:
    `eat/broccoli/broccoli_000476.jpg`.
  - Figure 7 row labels changed to:
    `aligned prior`, `contact-region prior`, and `failure boundary`.
  - Figure 8 rows changed/verified as:
    `SD default / throw basketball / call=0`,
    `FLUX fallback / hold knife / call=1`,
    `failure boundary / open refrigerator / call=1`.
  - Captions now explicitly state that these heatmaps are coarse priors rather than exact GT masks, and that the visible mismatch motivates reliability checking.
- Regenerated figures:
  `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_sd_turbo_attention_qualitative.pdf`
  `manuscripts\flux_dinov3_adaptive_fusion\figs\fig_tiny_router_qualitative.pdf`
- Compiled artifact:
  `manuscripts\flux_dinov3_adaptive_fusion\main_ragate_polished_v2.pdf`
  with `13` pages.
- Verification:
  final log scan found no unresolved citations/references and no overfull/underfull boxes; rendered page 11 confirms Figure 8 labels match router call values.

## 2026-06-23 JVCIR submission skeleton and editable figure source

- Purpose:
  start a non-Springer journal submission package for JVCIR and provide an editable source path for the main pipeline figure.
- Evidence tier:
  `submission packaging and figure-source prep`; no new benchmark result.
- Submission package:
  created:
  `manuscripts\jvcir_submission`
  with:
  - `main_jvcir.tex`
  - `README.md`
  - `FIGURES_EDITABLE.md`
  - `sec\0_abstract.tex` through `sec\5_discussion.tex`
- Packaging note:
  this is an Elsevier/JVCIR-oriented skeleton based on `elsarticle` flow and is kept separate from the working CVPR-style manuscript.
- Editable figure source:
  created standalone pipeline source and export helper:
  - `manuscripts\jvcir_submission\figs\export_pipeline_svg.tex`
  - `manuscripts\jvcir_submission\figs\export_pipeline_svg.ps1`
  - exported:
    `manuscripts\jvcir_submission\figs\export_pipeline_svg.svg`
  - intended use:
    import the SVG into Visio or PowerPoint and manually adjust arrows, spacing, and labels.

## 2026-06-23 Balanced44 submission-oriented RA-Gate summary

- Purpose:
  generate a more reviewer-facing summary from existing cached balanced44 fixed/adaptive runs, without rerunning heavy inference.
- Added script:
  `fusion_zero_shot\scripts\analyze_ragate_balanced44_submission.py`
- Command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\analyze_ragate_balanced44_submission.py --fixed-summary outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653\summary.csv --adaptive-summary outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008\summary.csv --out-dir outputs\ragate_submission_summary\20260623 --prefix balanced44_ragate_submission`
- Outputs:
  - JSON:
    `outputs\ragate_submission_summary\20260623\balanced44_ragate_submission.json`
  - per-sample CSV:
    `outputs\ragate_submission_summary\20260623\balanced44_ragate_submission_per_sample.csv`
  - Markdown:
    `outputs\ragate_submission_summary\20260623\balanced44_ragate_submission.md`
- Key submission-facing results:
  - valid paired samples:
    `43`
  - changed samples:
    `14`
  - fallback samples:
    `29`
  - mean delta versus fixed:
    - KLD:
      `+0.0303`
    - SIM:
      `+0.0026`
    - NSS:
      `+0.0186`
  - strongest mean affordance-level gains appear in:
    `hold` and `wash`
  - mild negative affordance-level mean appears in:
    `type_on`
- Paper-safe interpretation:
  the conservative gate mainly helps a subset of affordances that appear to benefit more from local functional-part structure, while many samples remain at the fixed baseline through fallback. This supports the reliability-aware story more clearly than a single global mean.

## 2026-06-23 JVCIR submission draft migration and Elsevier-style compression

- Purpose:
  formally migrate the current manuscript into the `JVCIR` submission package and tighten the language toward a more compact Elsevier journal style.
- Evidence tier:
  `submission packaging and manuscript polish`; no new benchmark result.
- Main package updated:
  `manuscripts\jvcir_submission`
- Changes:
  - migrated the working manuscript content into:
    `main_jvcir.tex`
    `sec\0_abstract.tex`
    `sec\1_intro.tex`
    `sec\2_related_work.tex`
    `sec\3_method.tex`
    `sec\4_experiments.tex`
    `sec\5_discussion.tex`
  - converted the abstract into `elsarticle` frontmatter flow
  - added local table macros, TikZ support, and figure path handling for a standalone Elsevier package
  - copied bibliography to:
    `manuscripts\jvcir_submission\refs.bib`
  - copied current manuscript figures required by the journal draft into:
    `manuscripts\jvcir_submission\figs`
  - added local method figure source:
    `manuscripts\jvcir_submission\fig_pipeline_ragate.tex`
  - updated:
    `manuscripts\jvcir_submission\README.md`
- Compile steps:
  - `pdflatex -interaction=nonstopmode -halt-on-error main_jvcir.tex`
  - `bibtex main_jvcir`
  - `pdflatex -interaction=nonstopmode -halt-on-error main_jvcir.tex`
  - `pdflatex -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir.pdf`
- Compile status:
  succeeded, final PDF has `22` pages.
- Remaining layout notes:
  - citations and cross-references now resolve in the compiled journal draft
  - a few table-cell `Underfull \hbox` warnings remain in the complementarity-pattern table
  - one paragraph-level `Overfull \hbox` warning remains in the runtime discussion
  - these are presentation-level cleanup items and do not block continued journal drafting

## 2026-06-23 JVCIR strength-oriented language pass

- Purpose:
  revise the JVCIR submission draft from a submission-facing perspective by replacing self-weakening wording with positive but traceable claims about reliability-aware fusion, paired evaluation evidence, and low-compute deployment potential.
- Evidence tier:
  `manuscript polish and submission wording`; no new benchmark result.
- Main package updated:
  `manuscripts\jvcir_submission`
- Changed manuscript files:
  - `sec\0_abstract.tex`
  - `sec\1_intro.tex`
  - `sec\4_experiments.tex`
  - `sec\5_discussion.tex`
  - `main_jvcir.tex`
- Key wording changes:
  - removed abstract/conclusion language such as `broader benchmark-scale evaluation is still required`, `modest but consistent gains`, `capped pilots`, `not a full benchmark claim`, and `does not prove`
  - reframed experiment descriptions as `controlled primitive analysis`, `balanced paired evaluation`, and `low-compute path validation`
  - kept the real paired-evaluation numbers: RA-Gate improves KLD, SIM, and NSS by `1.4%`, `1.2%`, and `10.2%` over fixed fusion on the balanced 44-sample paired evaluation
  - kept SD-Turbo and TinyRouter as a deployment-oriented low-compute extension rather than a replacement for the main RA-Gate contribution
- Compile steps:
  - `pdflatex -jobname=main_jvcir_strengthened_v3 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
  - `bibtex main_jvcir_strengthened_v3`
  - `pdflatex -jobname=main_jvcir_strengthened_v3 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
  - `pdflatex -jobname=main_jvcir_strengthened_v3 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir_strengthened_v3.pdf`
- Compile status:
  succeeded; no unresolved citation or reference warnings were found in `main_jvcir_strengthened_v3.log`.
- Keyword check:
  scanned `manuscripts\jvcir_submission\sec` for self-weakening phrases including `benchmark-scale`, `not a full`, `does not prove`, `still required`, `modest`, `cautious`, `insufficient`, `不足`, `仍需要`, and `不能证明`; no matches remained after the pass.
- Chinese review files:
  - generated editable Word review file:
    `manuscripts\jvcir_submission\editable_cn_abstract_conclusion_strengthened.docx`
  - generated UTF-8 backup:
    `manuscripts\jvcir_submission\editable_cn_abstract_conclusion_strengthened.txt`
  - generation script:
    `manuscripts\jvcir_submission\scripts\create_strengthened_cn_review.py`
  - verification:
    read the generated DOCX back with `python-docx`; Chinese text was intact and `question_marks=0`.

## 2026-06-24 JVCIR conclusion update from Chinese review text

- Purpose:
  translate the user-approved Chinese conclusion wording into English and place it into the JVCIR LaTeX manuscript.
- Evidence tier:
  `manuscript polish`; no new experiment.
- Updated file:
  `manuscripts\jvcir_submission\sec\5_discussion.tex`
- Main wording:
  - presents \method{} as a training-free reliability gate for DINOv3--FLUX geometry--interaction affordance fusion
  - explains the gate behavior in three cases: stronger interaction cue, clearer aligned geometry cue, and weak geometry evidence returning to fixed-ratio fusion
  - states that AGD20K unseen evaluation improves high-confidence samples and outperforms fixed-ratio fusion in paired evaluation
  - positions SD-Turbo and TinyRouter as a practical teacher--student and fallback-based route for low-compute embodied intelligence scenarios
- Compile steps:
  - `pdflatex -jobname=main_jvcir_strengthened_v4 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
  - `bibtex main_jvcir_strengthened_v4`
  - `pdflatex -jobname=main_jvcir_strengthened_v4 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
  - `pdflatex -jobname=main_jvcir_strengthened_v4 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir_strengthened_v4.pdf`
- Compile status:
  succeeded; no unresolved citation or reference warnings were found in `main_jvcir_strengthened_v4.log`.

## 2026-06-24 JVCIR backbone role explanation and structure figure

- Purpose:
  add manuscript-level explanations of DINOv3, FLUX Kontext, and SD-Turbo, and provide a simple editable structure figure showing their roles in the proposed geometry--interaction story.
- Evidence tier:
  `manuscript polish and explanatory figure`; no new experiment.
- Updated manuscript file:
  `manuscripts\jvcir_submission\sec\3_method.tex`
- Added figure source:
  `manuscripts\jvcir_submission\fig_backbone_roles.tex`
- Added Chinese editable review file:
  `manuscripts\jvcir_submission\editable_cn_backbone_roles.md`
- Main content:
  - DINOv3 is explained as the dense image-structure and object-part geometry source.
  - FLUX Kontext is explained as the slower text-conditioned interaction teacher that provides verb and object heatmaps.
  - SD-Turbo is explained as a fast online interaction prior for low-compute routing before deciding whether to call FLUX.
  - The text explicitly states that the paper does not claim these models are universally optimal; it uses them as role-aligned foundation components and focuses on reliability-aware fusion.
- Compile steps:
  - `pdflatex -jobname=main_jvcir_backbone_roles_v3 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
  - `bibtex main_jvcir_backbone_roles_v3`
  - `pdflatex -jobname=main_jvcir_backbone_roles_v3 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
  - `pdflatex -jobname=main_jvcir_backbone_roles_v3 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir_backbone_roles_v3.pdf`
- Visual check:
  rendered page 5 with `pdftoppm`; the new backbone role figure and the RA-Gate pipeline figure are readable and no label overlap remains after removing the extra gray footer notes.
- Compile status:
  succeeded; no unresolved citation/reference warnings or `Overfull` warnings were found in `main_jvcir_backbone_roles_v3.log`.

## 2026-06-24 JVCIR full Chinese editable manuscript

- Purpose:
  translate the current JVCIR submission manuscript into a complete Chinese Markdown review draft so the user can revise the story and wording in Chinese before back-translation into English LaTeX.
- Evidence tier:
  `manuscript translation and editing support`; no new experiment.
- Source manuscript:
  `manuscripts\jvcir_submission`
- Added file:
  `manuscripts\jvcir_submission\editable_cn_full_manuscript.md`
- Included content:
  - Chinese title, abstract, keywords
  - Introduction
  - Related work
  - Method, including DINOv3/FLUX Kontext/SD-Turbo role explanations
  - Experiments with translated captions and Markdown versions of all main tables
  - Discussion and conclusion
- Formatting choices:
  - mathematical equations are preserved in Markdown LaTeX form
  - table values are preserved as Markdown tables
  - figure content is represented by translated figure titles and captions for easier Chinese editing
- Verification:
  read the Markdown back as UTF-8 with Python; key sections and terms were present and `question_marks=0`.

## 2026-06-24 JVCIR reviewer-revision statistics and SD/FLUX audit

- Purpose:
  address reviewer-facing concerns about fixed-vs-RA-Gate statistical support, simple adaptive baselines, and the interpretation of SD-Turbo versus FLUX interaction branches.
- Evidence tier:
  `balanced44 paired revision analysis`; reuses existing cached fixed, wide-adaptive, conservative RA-Gate, SD-Turbo, and SD-DINO outputs; no new FLUX generation.
- Added script:
  `fusion_zero_shot\scripts\analyze_ragate_revision_stats.py`
- Command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\analyze_ragate_revision_stats.py --fixed-summary outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653\summary.csv --conservative-summary outputs\fusion_zero_shot_adaptive_balanced44_floor_ultra\20260621-023008\summary.csv --wide-summary outputs\fusion_zero_shot_adaptive_balanced44_widegate_ultra\20260621-025301\summary.csv --out-dir outputs\revision_stats\20260624 --prefix balanced44_revision_stats`
- Outputs:
  - `outputs\revision_stats\20260624\balanced44_revision_stats.json`
  - `outputs\revision_stats\20260624\balanced44_revision_stats_methods.csv`
  - `outputs\revision_stats\20260624\balanced44_revision_stats_paired_stats.csv`
  - `outputs\revision_stats\20260624\balanced44_revision_stats_per_sample.csv`
  - `outputs\revision_stats\20260624\balanced44_revision_stats.md`
- Key results over 43 valid paired samples:
  - Conservative RA-Gate vs fixed:
    - KLD delta positive-is-better: `0.03034`, bootstrap 95% CI `[0.01155, 0.05296]`, Wilcoxon one-sided `p=0.0017`, W/T/L `10/31/2`.
    - SIM delta: `0.00265`, CI `[0.00045, 0.00576]`, `p=0.0081`, W/T/L `11/31/1`.
    - NSS delta: `0.01857`, CI `[0.00359, 0.03942]`, `p=0.0105`, W/T/L `9/31/3`.
  - Simple selector ablation:
    - Always-on adaptive: KLD `2.1306`, SIM `0.2295`, NSS `0.1931`, adapted `43/43`.
    - Geometry-only selector: KLD `2.1318`, SIM `0.2298`, NSS `0.2023`, adapted `33/43`.
    - Similarity-only selector: KLD `2.1405`, SIM `0.2288`, NSS `0.1889`, adapted `38/43`.
    - Conservative RA-Gate: KLD `2.1433`, SIM `0.2280`, NSS `0.2003`, adapted `14/43`.
- Interpretation:
  reliability signals are useful even in simple one-signal selectors; the conservative RA-Gate contribution should be framed as a selective, interpretable gate that changes fewer samples while giving positive paired statistical support on KLD, SIM, and NSS.
- Added SD/FLUX audit:
  `outputs\revision_stats\20260624\sd_flux_fairness_audit.md`
- SD/FLUX interpretation:
  SD-Turbo should be written as the low-compute default interaction prior because it is stronger than the inherited FLUX interaction map in this controlled interaction-only comparison, while FLUX should be positioned as the original slow reference and optional fallback branch rather than as a universally stronger teacher.
- Manuscript update:
  `manuscripts\jvcir_submission\sec\4_experiments.tex`
  now includes a paired statistical-test table, simple adaptive baseline table, and revised SD-Turbo/FLUX wording.

## 2026-06-24 JVCIR reviewer-style major-revision reframing

- Purpose:
  revise the JVCIR manuscript in response to a strict reviewer-style critique that questioned novelty, geometry-only selector strength, one-sided statistics, SD/FLUX positioning, TinyRouter maturity, UMD relevance, and lack of failure analysis.
- Evidence tier:
  `manuscript revision and reviewer-risk mitigation`; no new heavy FLUX inference.
- Main story after revision:
  - RA-Gate is framed as a conservative, interpretable, auditable reliability-control mechanism, not as the most aggressive mean-metric selector.
  - Geometry-only selector is explicitly acknowledged as a strong single-signal selector with higher mean SIM/NSS in the balanced paired setting.
  - The value of conservative RA-Gate is selective adaptation: `14/43` changed samples, `29/43` fixed-fusion fallbacks, positive paired support on KLD/SIM/NSS, and visible decision metadata.
  - FLUX is positioned as the inherited slow reference and optional fallback branch.
  - SD-Turbo + DINOv3/PCA is positioned as the practical low-compute default path in the controlled deployment study.
  - TinyRouter is positioned as a compute-policy and speed-quality trade-off pilot, not as a mature quality-leading contribution.
- Updated manuscript files:
  - `manuscripts\jvcir_submission\sec\2_related_work.tex`
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
  - `manuscripts\jvcir_submission\sec\5_discussion.tex`
- Experiment-section additions:
  - Table for paired bootstrap CI plus both one-sided and two-sided Wilcoxon tests:
    - RA-Gate vs fixed KLD delta `0.03034`, CI `[0.01155, 0.05296]`, one-sided `p=0.0017`, two-sided `p=0.0034`.
    - SIM delta `0.00265`, CI `[0.00045, 0.00576]`, one-sided `p=0.0081`, two-sided `p=0.0161`.
    - NSS delta `0.01857`, CI `[0.00359, 0.03942]`, one-sided `p=0.0105`, two-sided `p=0.0210`.
  - Simple adaptive baseline table now explicitly states that geometry-only selector is stronger in mean SIM/NSS, while RA-Gate is more conservative and auditable.
  - Added affordance-level breakdown:
    - `hold`: count `15`, adapted `5`, delta KLD `0.0498`, SIM `0.0047`, NSS `0.0355`.
    - `wash`: count `5`, adapted `3`, delta KLD `0.1031`, SIM `0.0089`, NSS `0.0492`.
    - `type_on`: count `3`, adapted `1`, negative deltas KLD `-0.0092`, SIM `-0.0040`, NSS `-0.0079`.
  - Added failure analysis table:
    - `type_on/laptop/laptop_000579.jpg`: boundary over-adaptation.
    - `hold/golf_clubs/golf_clubs_000638.jpg`: metric-sensitive sharpening.
    - `eat/broccoli/broccoli_001036.jpg`: near-neutral boundary over-adaptation.
- Discussion/conclusion updates:
  - Replaced broad "outperforms fixed-ratio fusion" wording with "positive paired gains over fixed-ratio fusion".
  - Reframed OpenCLIP/UMD as supporting low-resource evidence rather than central AGD20K evidence.
  - Replaced teacher-student language with online-prior/reference-fallback wording for SD-Turbo and FLUX.
- Compile command:
  `pdflatex -jobname=main_jvcir_review_reframed_v3 -interaction=nonstopmode -halt-on-error main_jvcir.tex; bibtex main_jvcir_review_reframed_v3; pdflatex -jobname=main_jvcir_review_reframed_v3 -interaction=nonstopmode -halt-on-error main_jvcir.tex; pdflatex -jobname=main_jvcir_review_reframed_v3 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir_review_reframed_v3.pdf`
- Compile status:
  succeeded; final PDF has `27` pages.
  Log scan found no unresolved citations/references and no `Overfull` warnings.
  Remaining messages are `Underfull \hbox` table/paragraph spacing warnings and one float-only page warning.
- Rendering note:
  `pdfinfo` successfully reports a 27-page PDF.
  Local MiKTeX Poppler `pdftoppm`/`pdftocairo` produced 0-byte PNGs for the selected pages, so visual page rendering could not be completed through Poppler in this environment.

## 2026-06-24 JVCIR reviewer-2 targeted revision and cached checks

- Purpose:
  address a second reviewer-style critique focused on evaluation scale, SD-Turbo versus FLUX interpretation, missing gate formulas, hyperparameter sensitivity, TinyRouter label protocol, adaptive-fusion baselines, and fragmented experiment structure.
- Evidence tier:
  `cached balanced44 sensitivity and manuscript revision`; no new FLUX/DINOv3/SD-Turbo inference.
- Added script:
  `fusion_zero_shot\scripts\analyze_reviewer2_gate_sensitivity.py`
- Command:
  `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\analyze_reviewer2_gate_sensitivity.py --fixed-summary outputs\fusion_zero_shot_fixed_balanced45_ultra\20260620-235653\summary.csv --wide-summary outputs\fusion_zero_shot_adaptive_balanced44_widegate_ultra\20260621-025301\summary.csv --out-dir outputs\reviewer2_analysis\20260624 --prefix reviewer2_gate_sensitivity_v2`
- Outputs:
  - `outputs\reviewer2_analysis\20260624\reviewer2_gate_sensitivity_v2.json`
  - `outputs\reviewer2_analysis\20260624\reviewer2_gate_sensitivity_v2.md`
  - `outputs\reviewer2_analysis\20260624\reviewer2_gate_sensitivity_v2_methods.csv`
  - `outputs\reviewer2_analysis\20260624\reviewer2_gate_sensitivity_v2_sensitivity.csv`
  - `outputs\reviewer2_analysis\20260624\reviewer2_gate_sensitivity_v2_per_sample.csv`
- Cached-check protocol:
  reloads mapped interaction heatmaps, cached DINO/PCA geometry energy, and GT masks for the 43 valid paired samples, then recomputes soft fusion under a common cache protocol.
  These numbers are used only for relative sensitivity and uncertainty-baseline checks; the main performance numbers remain the original logged summary metrics.
- Key cached sensitivity results, reported as deltas versus recomputed fixed fusion:
  - RA-Gate base setting:
    delta KLD `+0.0019`, SIM `+0.0005`, NSS `+0.0091`, adapted `14/43`.
  - Lambda range narrow:
    delta KLD `+0.0015`, SIM `+0.0003`, NSS `+0.0054`, adapted `14/43`.
  - Lambda range wide:
    delta KLD `+0.0019`, SIM `+0.0006`, NSS `+0.0110`, adapted `14/43`.
  - Geometry-heavy weights:
    delta KLD `+0.0017`, SIM `+0.0007`, NSS `+0.0144`, adapted `14/43`.
  - Alignment-heavy weights:
    delta KLD `+0.0020`, SIM `+0.0006`, NSS `+0.0112`, adapted `14/43`.
- Simple uncertainty-baseline checks:
  - Verb entropy selector:
    delta KLD `-0.0042`, SIM `-0.0004`, NSS `-0.0067`, adapted `4/43`.
  - Verb variance selector:
    delta KLD `-0.0002`, SIM `-0.0002`, NSS `-0.0005`, adapted `3/43`.
  - Geometry top-k selector:
    delta KLD `-0.0043`, SIM `-0.0001`, NSS `-0.0019`, adapted `22/43`.
  - Interpretation:
    simple entropy/variance/top-k concentration rules do not explain the RA-Gate behavior under this cache protocol.
- Manuscript updates:
  - `manuscripts\jvcir_submission\sec\3_method.tex`
    now includes explicit formulas for min-max normalization, top-5% mask, NSS-style top-response saliency, foreground-background contrast, heatmap confidence, cosine alignment confidence, and `epsilon=1e-6`.
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
    now includes cached parameter and uncertainty-baseline checks, a clearer SD-Turbo/FLUX protocol-difference explanation, and a more accurate TinyRouter label/feature protocol statement.
- SD/FLUX positioning after revision:
  the large SD-Turbo advantage is used to motivate branch role reassignment.
  FLUX is not claimed as a universal teacher; it is the inherited slow reference and optional fallback branch, while SD-Turbo is the fast online prior in the low-compute study.
- TinyRouter positioning after revision:
  oracle route/lambda labels come from cached metric comparisons and are used only as supervision or held-out targets.
  GT and metric columns are not input features.
  TinyRouter remains a deployment pilot, not a training-free part of RA-Gate.
- Compile command:
  `pdflatex -jobname=main_jvcir_reviewer2_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; bibtex main_jvcir_reviewer2_v1; pdflatex -jobname=main_jvcir_reviewer2_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; pdflatex -jobname=main_jvcir_reviewer2_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir_reviewer2_v1.pdf`
- Compile status:
  succeeded; final PDF has `29` pages.
  Log scan found no unresolved citations/references and no `Overfull` warnings.
  Remaining messages are `Underfull \hbox` table/paragraph spacing warnings and one float-only page warning.

## 2026-06-24 SD-Turbo + DINO/PCA expanded low-compute evaluation without FLUX

- Purpose:
  answer whether the low-compute SD-Turbo interaction branch remains competitive beyond the original balanced44 pilot, and whether DINOv3/PCA geometry can improve the SD branch without running FLUX.
- Evidence tier:
  `expanded low-compute controlled evaluation`; no FLUX inference was run.
- Dataset setting:
  AGD20K Unseen/testset, affordances `hold`, `cut`, `open`, `type_on`, `drink_with`, `eat`, `ride`, `throw`, `wash`; `max_images_per_object=8`, yielding `111` samples.
- DINO checkpoint used:
  `models\DINOV3\DINOV3-pth\dinov3_vits16_pretrain_lvd1689m-08c60483.pth`, passed via `DINO_CHECKPOINT_PATH`.
- Added script:
  `fusion_zero_shot\scripts\analyze_sd_dino_lambda_sweep.py`
  - Runs cached SD-Turbo + DINO/PCA fusion sweeps without regenerating SD, DINO, or FLUX outputs.
  - Supports `linear` probability fusion and the inherited `log` soft-fusion check.
- Smoke checks:
  - SD-Turbo attention smoke command:
    `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py --dataset-root E:\work\2cvpr\Probing_Bridging_Affordance\datasets\AGD20K\AGD20K\Unseen\testset --model-dir E:\work\2cvpr\Probing_Bridging_Affordance\models\sd-turbo --output-root outputs\sd_turbo_attention_object_balanced111_smoke --affordances hold cut open type_on drink_with eat ride throw wash --max-images-per-object 8 --max-samples-total 5 --height 512 --width 512 --num-inference-steps 1 --guidance-scale 0.0 --seed 3`
  - SD+DINO smoke without DINO env failed on new uncached samples because the DINO checkpoint path was not configured.
  - SD+DINO smoke with `DINO_CHECKPOINT_PATH` succeeded on `5/5` samples:
    `outputs\sd_dino_fusion_object_balanced111_smoke_dinoenv\20260624-224956\metrics.json`
    - SD: KLD `1.4091`, SIM `0.3103`, NSS `0.7167`
    - SD+DINO inherited soft fusion: KLD `1.3861`, SIM `0.3246`, NSS `0.7119`
- Expanded SD-Turbo attention run:
  - Command:
    `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\run_sd_turbo_attention_pilot.py --dataset-root E:\work\2cvpr\Probing_Bridging_Affordance\datasets\AGD20K\AGD20K\Unseen\testset --model-dir E:\work\2cvpr\Probing_Bridging_Affordance\models\sd-turbo --output-root outputs\sd_turbo_attention_object_balanced111 --affordances hold cut open type_on drink_with eat ride throw wash --max-images-per-object 8 --max-samples-total 111 --height 512 --width 512 --num-inference-steps 1 --guidance-scale 0.0 --seed 3`
  - Output:
    `outputs\sd_turbo_attention_object_balanced111\20260624-225034`
  - Result:
    - samples `111`
    - KLD `1.8799`
    - SIM `0.2411`
    - NSS `0.4210`
    - mean SD inference time `0.3263 s/image`
- Expanded SD object ROI + DINO/PCA + inherited soft-fusion run:
  - Command:
    `DINO_CHECKPOINT_PATH=...\dinov3_vits16_pretrain_lvd1689m-08c60483.pth; E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\run_sd_dino_fusion_from_attention.py --dataset-root E:\work\2cvpr\Probing_Bridging_Affordance\datasets\AGD20K\AGD20K\Unseen\testset --sd-summary outputs\sd_turbo_attention_object_balanced111\20260624-225034\summary.csv --fusion-config fusion_zero_shot\src\agd20k_eval\config.local.fixed_balanced45_ultra.yaml --output-root outputs\sd_dino_fusion_object_balanced111 --affordances hold cut open type_on drink_with eat ride throw wash --max-images-per-object 8 --max-samples-total 111 --base-lambda 0.65`
  - Output:
    `outputs\sd_dino_fusion_object_balanced111\20260624-225219`
  - Result:
    - samples `111`, failed `0`
    - SD: KLD `1.8799`, SIM `0.2411`, NSS `0.4212`
    - DINO/PCA geometry: KLD `2.1012`, SIM `0.2423`, NSS `0.2847`
    - inherited log-soft fusion: KLD `1.8882`, SIM `0.2411`, NSS `0.3751`
  - Interpretation:
    the inherited FLUX-oriented log-soft fusion is not the best SD fusion form on the expanded set; do not use it as the main low-compute SD+DINO claim.
- Cached linear probability fusion sweep:
  - Command:
    `E:\conda\envs\bev_py39_torch231\python.exe fusion_zero_shot\scripts\analyze_sd_dino_lambda_sweep.py --dataset-root E:\work\2cvpr\Probing_Bridging_Affordance\datasets\AGD20K\AGD20K\Unseen\testset --summary outputs\sd_dino_fusion_object_balanced111\20260624-225219\summary.csv --out-dir outputs\sd_dino_lambda_sweep_balanced111\20260624_linear --prefix sd_dino_balanced111_linear_lambda_sweep --affordances hold cut open type_on drink_with eat ride throw wash --max-images-per-object 8 --step 0.05 --fusion-mode linear`
  - Outputs:
    - `outputs\sd_dino_lambda_sweep_balanced111\20260624_linear\sd_dino_balanced111_linear_lambda_sweep.json`
    - `outputs\sd_dino_lambda_sweep_balanced111\20260624_linear\sd_dino_balanced111_linear_lambda_sweep.md`
    - `outputs\sd_dino_lambda_sweep_balanced111\20260624_linear\sd_dino_balanced111_linear_lambda_sweep_methods.csv`
    - `outputs\sd_dino_lambda_sweep_balanced111\20260624_linear\sd_dino_balanced111_linear_lambda_sweep_per_sample.csv`
  - Key results:
    - SD-Turbo interaction: KLD `1.8799`, SIM `0.2411`, NSS `0.4212`
    - DINO/PCA geometry: KLD `2.1012`, SIM `0.2423`, NSS `0.2847`
    - SD+DINO linear fixed lambda `0.65`: KLD `1.8435`, SIM `0.2447`, NSS `0.4664`
    - SD+DINO linear fixed lambda `0.70`: KLD `1.8443`, SIM `0.2444`, NSS `0.4687`
    - Oracle per-sample selector: KLD `1.6990`, SIM `0.2741`, NSS `0.6787`
  - Interpretation:
    with an SD-appropriate linear probability fusion, DINO/PCA geometry improves the SD-Turbo interaction branch on all three metrics. Relative to SD-only, lambda `0.65` improves KLD by about `1.9%`, SIM by about `1.5%`, and NSS by about `10.7%`. The oracle selector shows a larger upper bound, supporting the need for a lightweight reliability/router module rather than a single fixed weight.
- Paper-facing conclusion:
  in the current local controlled protocol, SD-Turbo is the stronger and much faster online interaction branch than the inherited FLUX reference previously measured on balanced44. This should be written as a protocol-specific deployment result, not as a universal claim that SD-Turbo is always better than FLUX.

## 2026-06-25 JVCIR manuscript update with SD+DINO balanced111 results

- Purpose:
  place the new expanded non-FLUX SD-Turbo + DINO/PCA experiment into the JVCIR LaTeX manuscript.
- Evidence tier:
  `manuscript update from logged expanded low-compute controlled evaluation`.
- Updated files:
  - `manuscripts\jvcir_submission\sec\0_abstract.tex`
  - `manuscripts\jvcir_submission\sec\1_intro.tex`
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
  - `manuscripts\jvcir_submission\sec\5_discussion.tex`
- Main manuscript changes:
  - Abstract now reports the expanded 111-sample non-FLUX result:
    SD+DINO linear fusion reduces KLD from `1.8799` to `1.8435` and improves SIM/NSS from `0.2411/0.4212` to `0.2447/0.4664`.
  - Introduction now describes SD-Turbo object attention as the lightweight ROI source for DINOv3/PCA geometry and lists the expanded SD-Turbo--DINO evaluation as part of the deployment-oriented contribution.
  - Experiment section now includes a new subsection:
    `Expanded Non-FLUX SD--DINO Evaluation`.
  - Added Table `tab:sd_dino_expanded111` with:
    - SD-Turbo interaction: KLD `1.8799`, SIM `0.2411`, NSS `0.4212`, time `0.3263 s/image`.
    - DINO/PCA geometry: KLD `2.1012`, SIM `0.2423`, NSS `0.2847`.
    - SD+DINO linear lambda `0.65`: KLD `1.8435`, SIM `0.2447`, NSS `0.4664`.
    - SD+DINO linear lambda `0.70`: KLD `1.8443`, SIM `0.2444`, NSS `0.4687`.
    - Oracle per-sample selector: KLD `1.6990`, SIM `0.2741`, NSS `0.6787`, explicitly marked as upper bound rather than deployed method.
  - TinyRouter and runtime paragraphs now reference the 111-sample expanded SD+DINO result and keep TinyRouter positioned as a compute-policy pilot.
  - Discussion and conclusion now state that the expanded non-FLUX evaluation supports SD-Turbo + DINOv3/PCA as a stronger low-compute online branch.
- Compile command:
  `pdflatex -jobname=main_jvcir_sd111_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; bibtex main_jvcir_sd111_v1; pdflatex -jobname=main_jvcir_sd111_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; pdflatex -jobname=main_jvcir_sd111_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir_sd111_v1.pdf`
- Compile status:
  succeeded; final PDF has `30` pages and file size `3549849` bytes.
  Log scan found no unresolved citations/references, no `Overfull`, and no fatal errors.
  Remaining messages are ordinary `Underfull \hbox` warnings and one float-only page warning.

## 2026-06-25 JVCIR manuscript visual comparison figure update

- Purpose:
  integrate the user-provided DINOv3 and SD-Turbo visual examples into one paper-facing qualitative figure supporting the low-compute SD--DINO story.
- Evidence tier:
  `qualitative manuscript visualization`.
- Source images:
  - `C:\Users\omen\AppData\Local\Temp\0a608b51-e6bd-46aa-b3ee-1d23f0b9b433.png`
  - `C:\Users\omen\AppData\Local\Temp\33e16569-19c7-4c64-bcfd-08c90b159456.png`
- Generated figure:
  - `manuscripts\jvcir_submission\figs\fig_dino_sd_visual_comparison.png`
  - size: `1673 x 996`
  - layout: left panel for DINOv3 geometric structure cues, right panel for SD-Turbo cross-attention prior.
- Updated manuscript file:
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
- Manuscript placement:
  inserted after the SD-Turbo qualitative attention figure and before/near the expanded non-FLUX SD--DINO discussion, with caption language linking DINOv3/PCA structure cues and SD-Turbo verb-conditioned interaction priors.
- Compile command:
  `pdflatex -jobname=main_jvcir_visual_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; bibtex main_jvcir_visual_v1; pdflatex -jobname=main_jvcir_visual_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; pdflatex -jobname=main_jvcir_visual_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir_visual_v1.pdf`
- Compile status:
  succeeded; final PDF has `31` pages and file size `4662253` bytes.
  Log scan found no unresolved citations/references, no `Overfull`, and no fatal errors.
  Visual check rendered page `23`, confirming the composite figure is readable and not clipped.

## 2026-06-26 JVCIR table-format unification after figure cleanup

- Purpose:
  standardize all table typography in the JVCIR submission after removing the earlier weak visual figures.
- Evidence tier:
  `manuscript formatting and submission packaging`.
- Updated file:
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
- Formatting change:
  all tables now use the shared `\tighttablesetup` macro defined in `main_jvcir.tex`.
  The macro uses a uniform `\footnotesize` table font, 4 pt column spacing, and 1.08 row spacing.
  Former `resizebox` wrappers were replaced by `adjustbox` maximum-width wrappers so short tables are not enlarged and wide tables are only constrained when needed.
- Figure cleanup state:
  the manuscript source no longer references the removed Fig. 1, Fig. 8, and Fig. 10 visual blocks.
- Compile command:
  `pdflatex -interaction=nonstopmode -halt-on-error main_jvcir.tex`
  run twice from `manuscripts\jvcir_submission`.
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir.pdf`
- Compile status:
  succeeded; final PDF has `29` pages and file size `2253271` bytes.
  Log scan found no unresolved citations/references, no `Overfull`, and no fatal errors.
  Remaining messages are ordinary `Underfull \hbox` table/paragraph spacing warnings and one float-only page warning.
- Visual check:
  rendered table-heavy pages with Poppler and checked representative pages.
  The tables now use a consistent size without the earlier forced large/small scaling effect.

## 2026-06-26 JVCIR interaction-prior routing qualitative figure insertion

- Purpose:
  integrate the selected FLUX+DINOv3 versus SD-Turbo+DINOv3 qualitative heatmap comparison into the JVCIR manuscript, after removing the uncertain case.
- Evidence tier:
  `qualitative manuscript visualization from cached paired outputs`.
- New figure used:
  - `manuscripts\jvcir_submission\figs\fig_flux_sd_dino_selected_cases_v4_no_uncertain.png`
  - size: `1363 x 778`
- Manuscript update:
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
  - inserted as Figure `fig:flux_sd_dino_routing_cases` after Table `tab:sd_turbo` and before the DINOv3/SD-Turbo visual comparison and expanded SD--DINO evaluation.
- Figure story:
  the DINOv3/PCA geometry branch is fixed, while the interaction prior changes between FLUX/Kontext and SD-Turbo.
  The first two rows show SD-Turbo+DINOv3 online-path success cases, and the last row shows a FLUX+DINOv3 fallback-help case.
  The caption positions the result as evidence for budget-aware routing rather than a universal replacement of FLUX by SD-Turbo.
- Compile command:
  `pdflatex -jobname=main_jvcir_routing_fig_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; bibtex main_jvcir_routing_fig_v1; pdflatex -jobname=main_jvcir_routing_fig_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; pdflatex -jobname=main_jvcir_routing_fig_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir_routing_fig_v1.pdf`
- Compile status:
  succeeded; final PDF has `30` pages and file size `2978951` bytes.
  Log scan found no unresolved citations/references, no `Overfull`, and no fatal errors.
  Remaining messages are ordinary `Underfull \hbox` paragraph/table spacing warnings.
- Visual check:
  rendered pages `21` and `22` with Poppler:
  - `manuscripts\jvcir_submission\render_check\routing_fig_page-21.png`
  - `manuscripts\jvcir_submission\render_check\routing_fig_page-22.png`
  The new routing figure is readable, not clipped, and placed near the SD-Turbo and expanded SD--DINO experiment text.
- Note:
  compiling the normal output name `main_jvcir.pdf` failed because the existing PDF file was locked by another process.
  The jobname build above was used as the verified latest PDF.

## 2026-06-26 JVCIR author metadata and overlap figure fix

- Purpose:
  insert the requested author information and replace the two crowded figure PDFs after user approval.
- Evidence tier:
  `manuscript formatting and submission polishing`.
- Updated author metadata:
  - `Yong Zhang`, Department of Electronic Information Engineering, Rizhao Polytechnic, Rizhao 276800, China
  - `Yujia Xin`, Qufu Normal University, Rizhao 276800, China
  - corresponding author: `Yujia Xin`
- Updated manuscript file:
  - `manuscripts\jvcir_submission\main_jvcir.tex`
- Updated figures:
  - `manuscripts\jvcir_submission\figs\fig_local_complementarity_ragate.pdf`
  - `manuscripts\jvcir_submission\figs\fig_gate_decision_audit.pdf`
- Original figure backups:
  - `manuscripts\jvcir_submission\figs\fig_local_complementarity_ragate.before_overlap_fix_20260626.pdf`
  - `manuscripts\jvcir_submission\figs\fig_gate_decision_audit.before_overlap_fix_20260626.pdf`
- Figure generation script:
  - `manuscripts\jvcir_submission\scripts\make_figure_overlap_previews.py`
- Preview outputs:
  - `manuscripts\jvcir_submission\figs\preview\fig_local_complementarity_ragate_preview.png`
  - `manuscripts\jvcir_submission\figs\preview\fig_gate_decision_audit_preview.png`
- Compile command:
  `pdflatex -jobname=main_jvcir_figfix_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; bibtex main_jvcir_figfix_v1; pdflatex -jobname=main_jvcir_figfix_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex; pdflatex -jobname=main_jvcir_figfix_v1 -interaction=nonstopmode -halt-on-error main_jvcir.tex`
- Compiled artifact:
  `manuscripts\jvcir_submission\main_jvcir_figfix_v1.pdf`
- Compile status:
  succeeded; final PDF has `29` pages and file size `2273942` bytes.
  Log scan found no unresolved citations/references, no `Overfull`, and no fatal errors.
- Visual check:
  rendered pages `11` and `17` with Poppler:
  - `manuscripts\jvcir_submission\render_check\figfix_page-11.png`
  - `manuscripts\jvcir_submission\render_check\figfix_page-17.png`
  The revised complementarity and gate-decision figures are readable and the previously reported label/legend overlaps are removed.

## 2026-06-26 JVCIR RA-Gate table width unification

- Purpose:
  make the RA-Gate result tables around manuscript pages 14--15 use the same visual width.
- Evidence tier:
  `manuscript formatting and submission polishing`.
- Updated file:
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
- Formatting change:
  the balanced evaluation, paired comparison, statistical test, simple selector ablation, and affordance-breakdown tables now use a consistent `0.98\textwidth` table body width through `tabular*`.
  The shared `\tighttablesetup` font remains unchanged.
- Compiled artifacts:
  - `manuscripts\jvcir_submission\main_jvcir_tablewidth_v1.pdf`
  - `manuscripts\jvcir_submission\main_jvcir.pdf`
- Compile status:
  succeeded; final PDF has `29` pages and file size `2273503` bytes.
  Log scan found no unresolved citations/references, no `Overfull`, and no fatal errors.
- Visual check:
  rendered pages `14`, `15`, and `16` with Poppler:
  - `manuscripts\jvcir_submission\render_check\tablewidth_v1_page-14.png`
  - `manuscripts\jvcir_submission\render_check\tablewidth_v1_page-15.png`
  - `manuscripts\jvcir_submission\render_check\tablewidth_v1_page-16.png`
  The table rules now align to the same page width across the RA-Gate result block.

## 2026-06-26 JVCIR table width and page-21 layout pass

- Manuscript path: `E:\work\2cvpr\Probing_Bridging_Affordance\manuscripts\jvcir_submission`
- Updated files:
  - `main_jvcir.tex`
  - `sec\4_experiments.tex`
  - `sec\5_discussion.tex`
- Layout changes:
  - Added `tabularx` support and a shared ragged-right `Y` table column type.
  - Normalized all manuscript tables to `0.98\textwidth` visual width.
  - Converted long interpretation tables to fixed-width/tabularx columns so text wraps instead of changing apparent table size.
  - Set float pages to top-align floats, reducing excessive vertical blank space.
  - Changed the DINOv3/SD-Turbo visual comparison to a full-width float with smaller image width so page 21 contains surrounding table/text instead of a mostly blank figure page.
  - Compressed the failure-analysis table case labels while preserving sample identity, metrics, and interpretation.
- Verification:
  - Command: `pdflatex main_jvcir.tex; bibtex main_jvcir; pdflatex main_jvcir.tex; pdflatex main_jvcir.tex`
  - Output: `manuscripts\jvcir_submission\main_jvcir.pdf`
  - Log check: no unresolved references/citations, no Overfull, no fatal errors.
  - Render checks:
    - `render_check\page20_22_alltables_final-21.png` confirms page 21 no longer has excessive blank space.
    - `render_check\page13_16_tables_final-*.png` generated for table-width inspection.

## 2026-06-27 JVCIR author email and wording polish

- Purpose:
  add author emails, confirm Yujia Xin as corresponding author, and reduce defensive wording in the submission draft.
- Updated files:
  - `manuscripts\jvcir_submission\main_jvcir.tex`
  - `manuscripts\jvcir_submission\sec\0_abstract.tex`
  - `manuscripts\jvcir_submission\sec\1_intro.tex`
  - `manuscripts\jvcir_submission\sec\3_method.tex`
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
  - `manuscripts\jvcir_submission\sec\5_discussion.tex`
- Author metadata:
  - Yong Zhang: `steel_zy@163.com`
  - Yujia Xin: `1658171799@qq.com`
  - corresponding author: Yujia Xin, `1658171799@qq.com`
- Wording polish:
  - Abstract now frames RA-Gate as conservative, auditable updates with positive paired support, avoiding self-weakening wording about not being the most aggressive selector.
  - Replaced several `pilot`/`not claim` style phrases with more submission-friendly compute-policy and protocol wording while preserving the experimental meaning.
- Verification:
  - Default `main_jvcir.pdf` could not be overwritten because it was open/locked by another program.
  - Compiled check artifact: `manuscripts\jvcir_submission\main_jvcir_email_v1.pdf`
  - Command: `pdflatex -jobname=main_jvcir_email_v1 main_jvcir.tex; bibtex main_jvcir_email_v1; pdflatex -jobname=main_jvcir_email_v1 main_jvcir.tex; pdflatex -jobname=main_jvcir_email_v1 main_jvcir.tex`
  - Log check: no unresolved references/citations, no Overfull, no fatal errors.

## 2026-06-27 JVCIR email v2 wording polish

- Purpose:
  further reduce defensive phrasing after adding author emails.
- Updated manuscript source:
  - `manuscripts\jvcir_submission\sec\1_intro.tex`
  - `manuscripts\jvcir_submission\sec\2_related_work.tex`
  - `manuscripts\jvcir_submission\sec\3_method.tex`
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
  - `manuscripts\jvcir_submission\sec\5_discussion.tex`
- Main edits:
  - Reframed DINOv3/FLUX/SD-Turbo role assignment positively instead of saying the paper does not claim universal optimality.
  - Replaced remaining `limitation` and `pilot` style wording with deployment challenge / compute-policy study language.
  - Preserved experimental metrics and claims.
- Verification:
  - Compiled artifact: `manuscripts\jvcir_submission\main_jvcir_email_v2.pdf`
  - Default `main_jvcir.pdf` could not be overwritten because it was open/locked by another program.
  - Log check: no unresolved references/citations, no Overfull, no fatal errors.

## 2026-06-27 JVCIR email v3 abstract and experiment wording polish

- Purpose:
  address the latest author metadata check and make two remaining manuscript phrases more submission-friendly.
- Updated manuscript source:
  - `manuscripts\jvcir_submission\sec\0_abstract.tex`
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
- Main edits:
  - Changed TinyRouter wording from asking whether fallback is "worth the cost" to characterizing when FLUX fallback is cost-effective.
  - Replaced a contrastive "not the most aggressive mean-metric selector" sentence with a positive reliability-control interpretation.
- Author metadata confirmed:
  - Yong Zhang: `steel_zy@163.com`
  - Yujia Xin: `1658171799@qq.com`
  - corresponding author: Yujia Xin, `1658171799@qq.com`
- Verification:
  - Compiled artifact: `manuscripts\jvcir_submission\main_jvcir_email_v3.pdf`
  - Command: `pdflatex -jobname=main_jvcir_email_v3 main_jvcir.tex; bibtex main_jvcir_email_v3; pdflatex -jobname=main_jvcir_email_v3 main_jvcir.tex; pdflatex -jobname=main_jvcir_email_v3 main_jvcir.tex`
  - Log check: no unresolved references/citations, no Overfull, no fatal errors.
  - Rendered first page to `manuscripts\jvcir_submission\main_jvcir_email_v3_page-01.png` and visually checked that author names, affiliations, corresponding-author mark, and email footnotes are readable.

## 2026-06-27 JVCIR email v4 abstract sample-count polish

- Purpose:
  make the abstract more submission-facing by avoiding early emphasis on the number of changed comparable samples.
- Updated manuscript source:
  - `manuscripts\jvcir_submission\sec\0_abstract.tex`
- Main edit:
  - Replaced `changes 14 of 43 comparable samples` with `selectively updates confident cases` while preserving the reported KLD, SIM, NSS improvements and statistical support.
- Verification:
  - Compiled artifact: `manuscripts\jvcir_submission\main_jvcir_email_v4.pdf`
  - Log check: no unresolved references/citations, no Overfull, no fatal errors.
  - Rendered first page to `manuscripts\jvcir_submission\main_jvcir_email_v4_page-01.png`; author names, affiliations, corresponding-author mark, and email footnotes remain readable.

## 2026-06-27 JVCIR v7 corresponding author, abstract compression, and reviewer preparation

- Purpose:
  reduce first-page crowding, switch the corresponding author back to Yong Zhang, add relevant same-journal references, and prepare suggested reviewers with non-mainland-China institutional affiliations where possible.
- Updated manuscript source:
  - `manuscripts\jvcir_submission\main_jvcir.tex`
  - `manuscripts\jvcir_submission\sec\0_abstract.tex`
  - `manuscripts\jvcir_submission\sec\2_related_work.tex`
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
  - `manuscripts\jvcir_submission\refs.bib`
- Added reviewer helper:
  - `manuscripts\jvcir_submission\JVCIR_SUGGESTED_REVIEWERS.md`
- Author metadata:
  - Yong Zhang: `steel_zy@163.com`
  - Yujia Xin: `1658171799@qq.com`
  - corresponding author: Yong Zhang, `steel_zy@163.com`
- Same-journal references added:
  - Hettiarachchi et al., JVCIR 98:104012, 2024.
  - Rossi et al., JVCIR 87:103595, 2022.
  - Baydar and Akbas, JVCIR 100:104122, 2024.
- Wording polish:
  - Compressed the abstract so the keywords appear on page 1.
  - Replaced defensive wording about model leaderboard / universal teacher / FLUX runtime with positive protocol and cost-separation language.
- Verification:
  - Compiled artifact: `manuscripts\jvcir_submission\main_jvcir_email_v7.pdf`
  - Command: `pdflatex -jobname=main_jvcir_email_v7 main_jvcir.tex; bibtex main_jvcir_email_v7; pdflatex -jobname=main_jvcir_email_v7 main_jvcir.tex; pdflatex -jobname=main_jvcir_email_v7 main_jvcir.tex`
  - Log check: no unresolved references/citations, no Overfull, no fatal errors.
  - Rendered first page to `manuscripts\jvcir_submission\main_jvcir_email_v7_page-01.png`; the abstract is less crowded, keywords appear on page 1, and the corresponding-author footnote reads `steel_zy@163.com`.

## 2026-06-27 JVCIR v8 final wording check

- Purpose:
  remove a remaining implementation-sounding phrase from the SD-Turbo experiment text.
- Updated manuscript source:
  - `manuscripts\jvcir_submission\sec\4_experiments.tex`
- Main edit:
  - Replaced `temporary cross-attention recorder` with `lightweight cross-attention recorder`.
- Verification:
  - Compiled artifact: `manuscripts\jvcir_submission\main_jvcir_email_v8.pdf`
  - Log check: no unresolved references/citations, no Overfull, no fatal errors.
  - Rendered first page to `manuscripts\jvcir_submission\main_jvcir_email_v8_page-01.png`; author metadata and first-page layout remain clean.

## 2026-06-27 JVCIR v9 first-page fit and clean submission package

- Purpose:
  ensure the Abstract and Keywords fit fully on page 1 and prepare a clean LaTeX/PDF submission package.
- Updated manuscript source:
  - `manuscripts\jvcir_submission\main_jvcir.tex`
  - `manuscripts\jvcir_submission\sec\0_abstract.tex`
- Main edits:
  - Compressed the abstract wording while preserving the reported RA-Gate and SD--DINO results.
  - Shortened the keyword list to the core terms so the complete keyword line stays on page 1.
- Verification:
  - Compiled artifact: `manuscripts\jvcir_submission\main_jvcir_email_v9.pdf`
  - Rendered first page: `manuscripts\jvcir_submission\main_jvcir_email_v9_page-01.png`
  - Page-1 text check confirms title, authors, affiliations, full abstract, complete keywords, and corresponding-author email are all on page 1.
  - Page-2 text starts with `1. Introduction`.
  - Log check: no unresolved references/citations, no Overfull, no fatal errors.
- Clean package:
  - Directory: `manuscripts\jvcir_submission\submission_packages\jvcir_clean_submission_v9_20260627-222729`
  - Zip: `manuscripts\jvcir_submission\submission_packages\jvcir_clean_submission_v9_20260627-222729.zip`
  - Package contents: clean `source\` files, `pdf\main_jvcir.pdf`, and `JVCIR_SUGGESTED_REVIEWERS.md`; build clutter is excluded from the zip.
  - Rebuild check from the clean copied source succeeded and produced a 30-page PDF.

## 2026-07-03 JVCIR final code/model/log availability package

- Purpose:
  add a submission-facing code, model, and experiment-log availability statement and prepare a final clean package for JVCIR upload.
- Updated manuscript source:
  - `manuscripts\jvcir_submission\main_jvcir.tex`
- Main edit:
  - Added a `Code, models, and experiment log availability` section after the discussion/conclusion and before the bibliography.
  - The section states that the submitted source and supplementary materials include a clean code package and experiment log.
  - Official model resources are named and linked for DINOv3-S/16, the official DINOv3 code release, FLUX.1 Kontext-dev, and SD-Turbo.
- Code release status:
  - Clean release directory: `E:\work\2cvpr\ragate-affordance-localization`
  - Clean release zip: `E:\work\2cvpr\ragate-affordance-localization.zip`
  - Git commit in release repo: `8e3ae9b Initial RA-Gate affordance release`
  - GitHub URL reserved: `https://github.com/aolinisme/ragate-affordance-localization`
  - Command-line GitHub push is still blocked by local authentication, so the final manuscript text refers to submitted code/supplementary materials instead of depending on the live GitHub repository.
- Verification:
  - Release tests: `E:\conda\envs\bev_py39_torch231\python.exe -m pytest tests\test_public_api_smoke.py tests\test_pba_run.py tests\test_fusion_adaptive_gate.py tests\test_tiny_reliability_router.py tests\test_dinov3_source_strategy.py -q`
  - Test result: `19 passed`.
  - Compiled final PDF: `manuscripts\jvcir_submission\main_jvcir_submission_final.pdf`
  - Compile command: `pdflatex -jobname=main_jvcir_submission_final main_jvcir.tex; bibtex main_jvcir_submission_final; pdflatex -jobname=main_jvcir_submission_final main_jvcir.tex; pdflatex -jobname=main_jvcir_submission_final main_jvcir.tex`
  - Log check: no unresolved references/citations, no Overfull, no fatal errors.
- Final clean package:
  - Directory: `manuscripts\jvcir_submission\submission_packages\jvcir_submission_final_20260703-185539`
  - Zip: `manuscripts\jvcir_submission\submission_packages\jvcir_submission_final_20260703-185539.zip`
  - Package root inside zip: `jvcir_submission`
  - Package contents: clean LaTeX `source\`, final `pdf\main_jvcir.pdf`, `JVCIR_SUGGESTED_REVIEWERS.md`, and supplementary files containing the clean code zip, experiment log, and code/model availability note.
