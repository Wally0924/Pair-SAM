# Datasets

This directory holds the **manifest CSVs** that every training and evaluation entry point reads, plus the helper scripts that produced them. No images or labels are redistributed here; obtain each dataset from its official source and arrange it exactly as described below so that the manifests resolve without modification.

## 1. Obtain the datasets

| Dataset | Used for | Archives to download |
|---|---|---|
| [ACDC](https://acdc.vision.ee.ethz.ch/) | Main training and evaluation (fog / rain / snow / night) | `rgb_anon_trainvaltest.zip`, `gt_trainval.zip` |
| [Dark Zurich](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) | Night-time cross-dataset evaluation | `Dark_Zurich_val_anon.zip`, `Dark_Zurich_test_anon_withoutGt.zip` |
| [MUSES](https://muses.vision.ee.ethz.ch/) | Cross-dataset evaluation over weather × time-of-day | `frame_camera`, `reference_frame`, `gt_semantic`, `meta.json` |
| [Cityscapes](https://www.cityscapes-dataset.com/) | Clear-weather pre-training stage | `leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip` |
| [Foggy Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) | Legacy synthetic-fog experiments (optional) | `leftImg8bit_trainvaltest_foggy.zip` |
| [RobotCar Correspondence](https://github.com/brdav/cma) | Cross-season evaluation (optional) | fetched by `download_robotcar.sh` |

All datasets are released under their own licenses; registration is required for ACDC, MUSES, and Cityscapes.

## 2. Expected directory layout

Set one root directory and place every dataset beneath it. The tree below is the layout the manifests encode; the comments name the archive that produces each branch. Two points differ from a plain "unzip in place": ACDC and Dark Zurich archives are extracted **into a folder named after the archive**, and Cityscapes is split into `Images/` and `GT/` sub-folders.

```text
$DATASET_ROOT/
├── ACDC/
│   ├── rgb_anon/                                  # rgb_anon_trainvaltest.zip
│   │   └── {fog,rain,snow,night}/
│   │       ├── {train,val,test}/<seq>/<seq>_frame_XXXXXX_rgb_anon.png
│   │       └── {train,val,test}_ref/<seq>/<seq>_frame_XXXXXX_rgb_ref_anon.png
│   └── gt_trainval/                               # gt_trainval.zip, extracted INTO this folder
│       └── gt/{fog,rain,snow,night}/{train,val}/<seq>/
│           ├── <seq>_frame_XXXXXX_gt_labelTrainIds.png
│           └── <seq>_frame_XXXXXX_gt_invIds.png
│
├── Dark_Zurich/
│   ├── Dark_Zurich_val_anon/                      # Dark_Zurich_val_anon.zip, extracted INTO this folder
│   │   ├── rgb_anon/val/night/<seq>/<seq>_frame_XXXXXX_rgb_anon.png
│   │   ├── rgb_anon/val_ref/day/<seq>_ref/<seq>_frame_XXXXXX_ref_rgb_anon.png
│   │   └── gt/val/night/<seq>/<seq>_frame_XXXXXX_gt_{labelTrainIds,invIds}.png
│   └── Dark_Zurich_test_anon_withoutGt/           # Dark_Zurich_test_anon_withoutGt.zip, extracted INTO this folder
│       └── rgb_anon/{test,test_ref}/...           # same pattern as val; no GT
│
├── MUSES/                                         # official layout, unchanged
│   ├── frame_camera/{train,val,test}/<weather>/<time_of_day>/RECxxxx_frame_xxxxxx_frame_camera.png
│   ├── reference_frame/{train,val,test}/<weather>/<time_of_day>/...
│   ├── gt_semantic/{train,val}/<weather>/<time_of_day>/RECxxxx_frame_xxxxxx_gt_labelTrainIds.png
│   └── meta.json
│
├── Cityscapes/                                    # NOTE: Images/ and GT/ sub-folders
│   ├── Images/leftImg8bit/{train,val,test}/<city>/<city>_xxxxxx_xxxxxx_leftImg8bit.png
│   └── GT/gtFine/{train,val,test}/<city>/<city>_xxxxxx_xxxxxx_gtFine_labelTrainIds.png
│
├── Cityscapes_foggy/                              # optional, legacy manifests only
│   └── leftImg8bit_foggy/{train,val,test}/<city>/<city>_..._leftImg8bit_foggy_beta_{0.005,0.01,0.02}.png
│
└── RobotCar/                                      # optional; created by download_robotcar.sh
    ├── images/
    ├── correspondence_data/
    └── segmented_images/
```

Notes:

- **Cityscapes `*_gtFine_labelTrainIds.png`** is not in the official archive. Generate it with `createTrainIdLabelImgs.py` from [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) after extracting `gtFine_trainvaltest.zip` into `Cityscapes/GT/`.
- ACDC, Dark Zurich, and MUSES ship `labelTrainIds` (and ACDC/Dark Zurich `invIds`) directly.
- `<seq>` stands for a GoPro sequence id such as `GOPR0475`; the file names inside are exactly as released.

## 3. Path placeholders

Manifests store paths with two placeholders instead of absolute locations:

| Placeholder | Meaning | Default |
|---|---|---|
| `${DATASET_ROOT}` | root of the tree above | `~/Datasets` |
| `${REPO_ROOT}` | this repository (used only by legacy manifests that point at feature caches under `Datasets/features_*`) | the repository root, derived from `path_resolver.py` |

```bash
export DATASET_ROOT=/path/to/your/Datasets
# export REPO_ROOT=/path/to/Pair-SAM   # normally unnecessary
```

`path_resolver.py` expands both placeholders. `utils/pair_dataloader.py`, `precompute_features.py`, and `train.py` call it when they load a manifest, so no manual substitution is needed. For your own analysis scripts:

```python
import pandas as pd
from path_resolver import resolve_dataframe

df = resolve_dataframe(pd.read_csv("Datasets/acdc_adverse_ref_rgb_val.csv"))
```

## 4. Manifest CSVs

### Primary manifests (used by the released pipeline)

| File | Rows | Consumer | Notes |
|---|---:|---|---|
| `acdc_adverse_ref_rgb_train.csv` | 1,600 | `train.py` (default `--train_csv`) | ACDC train, four conditions |
| `acdc_adverse_ref_rgb_val.csv` | 406 | `train.py` (default `--val_csv`), `scripts/eval/eval_e1_acdc_val_full.py` | ACDC val; labels public |
| `acdc_adverse_ref_rgb_test.csv` | 2,000 | `scripts/eval/dump_acdc_test_preds.py` | ACDC test; `gt_path` and `invalid_mask` empty (server-held) |
| `cityscapes_m2f_train.csv` / `_val.csv` | 2,975 / 500 | `pretrain_cityscapes.py` | Clear-weather pre-training; `ref_image_path` equals `image_path` |
| `darkzurich_adverse_ref_rgb_val.csv` / `_test.csv` | 50 / 151 | Dark Zurich evaluation / submission | `condition_id` fixed to 3 (night) |
| `muses_cond8_ref_rgb_{train,val,test}.csv` | 1,500 / 250 / 750 | `scripts/run_muses_cond8.sh`, `scripts/eval/dump_muses_preds.py` | Eight-condition MUSES split with extra `weather`, `time_of_day` columns |
| `muses_ref_rgb_{val,test}.csv`, `muses_adverse_ref_rgb_*.csv`, `muses_all_ref_rgb_*.csv` | — | MUSES variants (`--cond-mode` / `--adverse-only` in `make_muses_csv.py`) | Four-condition or all-sample variants of the above |

Common columns:

```text
image_path, ref_image_path, gt_path, condition, condition_id, invalid_mask
```

- `image_path` adverse-weather frame; `ref_image_path` its GNSS-paired clear-weather reference; `gt_path` `labelTrainIds` map (empty for test splits); `invalid_mask` official invalid-region mask where provided.
- `condition_id` for ACDC / Dark Zurich / four-condition MUSES: `0 fog, 1 rain, 2 snow, 3 night`.
- `condition_id` for `muses_cond8_*`: `0 fog-day, 1 rain-day, 2 snow-day, 3 clear-night, 4 fog-night, 5 rain-night, 6 snow-night, 7 clear-day` (the first four keep ACDC semantics). MUSES clear-day frames have no separate reference and use themselves as `ref_image_path`.

### Legacy manifests (earlier Foggy-Cityscapes / GPS experiments; not needed for the FULL model)

| File | Notes |
|---|---|
| `train_all.csv`, `val_all.csv`, `test_all.csv` | Foggy Cityscapes (three beta levels) with `ref_mask_path` = Cityscapes colour GT; columns `image_path, ref_mask_path, gt_path` |
| `train_with_gps.csv`, `val_with_gps.csv`, `test_with_gps.csv` | Same plus Cityscapes vehicle GPS (`lat`, `lon`) and `${REPO_ROOT}/Datasets/features_*` cache paths |
| `foggy_0.02/*.csv` | Single beta = 0.02 subset; `*_cached.csv` variants reference feature caches |
| `acdc_train.csv`, `acdc_val.csv`, `*_with_embeddings.csv` | Early ACDC manifests using colour-label references and cached ViT-H features |

Feature caches referenced through `${REPO_ROOT}` are not distributed (about 60 GB); rebuild them with `segment-anything/precompute_features.py` if you need these manifests.

## 5. Verify your setup

```bash
export DATASET_ROOT=/path/to/your/Datasets
cd segment-anything
python - <<'PY'
import os, sys, pandas as pd
sys.path.insert(0, "../Datasets")
from path_resolver import resolve_dataframe
df = resolve_dataframe(pd.read_csv("../Datasets/acdc_adverse_ref_rgb_val.csv"))
missing = [p for col in ("image_path", "ref_image_path", "gt_path") for p in df[col] if isinstance(p, str) and not os.path.exists(p)]
print(f"{len(df)} rows, {len(missing)} missing files")
print("\n".join(missing[:5]))
PY
```

Zero missing files means the layout matches. `check_data_integrity.py` performs a deeper check (opens every image, compares image/label sizes); pass the manifest with `--csv` and resolve placeholders first if you use it on the raw CSV.

## 6. Helper scripts

| Script | Purpose |
|---|---|
| `path_resolver.py` | Placeholder expansion used by the loaders |
| `make_muses_csv.py` | Builds the MUSES manifests from `meta.json` (`--cond-mode map/cond8`, `--adverse-only`, `--verify-paths`) |
| `generate_csv.py` | Builds the legacy Foggy-Cityscapes manifests |
| `add_gps_to_csv.py` | Appends Cityscapes vehicle GPS (`lat`, `lon`) from `vehicle_sequence.json` |
| `check_data_integrity.py` | Checks that every path in a manifest exists and is readable |
| `download_robotcar.sh` | Downloads RobotCar Correspondence into the CMA/Refign layout |

`generate_csv.py`, `add_gps_to_csv.py`, and `check_data_integrity.py` still carry the original author's absolute paths as defaults; edit them or pass explicit arguments before running. The released manifests were produced with these scripts and do not need to be regenerated.
