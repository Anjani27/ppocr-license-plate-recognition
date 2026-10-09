# Car License Plate Recognition with PaddleOCR: PP-OCRv5 vs PP-OCRv4

Training and inference pipelines that fine-tune **PP-OCRv5_server_rec** and **PP-OCRv4_server_rec** on Indian licence plates and compare them on unseen car photos.

| Notebook | What it does |
|---|---|
| [`PPOCR_License_Plate_Training.ipynb`](PPOCR_License_Plate_Training.ipynb) | Setup, dataset preparation, pretrained-model download, training of v5 and v4, curves, evaluation, export, comparison |
| [`PPOCR_License_Plate_Inference.ipynb`](PPOCR_License_Plate_Inference.ipynb) | Loads both exported models, runs them on full car photos, draws boxes, text and confidence, batch statistics, side-by-side comparison, error analysis, export |

Both notebooks were run top to bottom on a fresh Colab runtime and are saved **with all outputs**. Each code cell prints a `✅ Done` line, and each notebook ends with a run-status cell that checks every expected output.

---

## Results

Validation set: 20 plates the models never saw during training.

| | PP-OCRv5_server_rec | PP-OCRv4_server_rec |
|---|---|---|
| Baseline accuracy (pretrained, no training) | 0.80 (16/20) | 0.80 (16/20) |
| **Fine-tuned accuracy** (exact plate match) | **0.85 (17/20)** | **0.85 (17/20)** |
| Fine-tuned norm_edit_dis (character level) | 0.918 | 0.922 |
| Avg confidence: correct / wrong reads | 0.990 / 0.722 | 0.988 / 0.754 |
| Inference time per image (Colab CPU) | 611 ms | 938 ms |
| Exported model size | 84.9 MB | 90.7 MB |
| GPU memory during training (T4) | 2.6 GB | 1.9 GB |

**Key findings**

- **The models are tied on accuracy.** Both read 17 of 20 plates, fail on the same three, and differ by less than one character in edit distance.
- **Fine-tuning added one plate (+5 points) for each model.** The pretrained models already read printed A–Z / 0–9 well; with 20 validation plates the gain is within noise.
- **Errors come mostly from the data.** Two failures are temporary-registration plates where both models read a printed "TEMP" that the labels leave out; the third plate is unreadable for both models.
- **v5 overfits faster.** It reached ~100% training accuracy while validation stayed at 85%; v4 was still learning at epoch 80.
- **Confidence is a useful review signal.** Correct reads average about 0.99 and wrong ones about 0.72–0.75; a 0.9 threshold would flag two of each model's three errors.
- **Practical choice: v5.** Equal accuracy, smaller model, and about 35% faster per image on CPU (single run on shared hardware, so indicative only).

---

## Pipeline

```
Car photo ──► plate box (from the dataset's XML annotation) ──► crop ──► PP-OCRv5 ──► text + confidence
                                                                    └──► PP-OCRv4 ──► text + confidence
                                                                              │
                                    draw box, both answers and confidences on the photo
```

OCR is a two-stage problem: detection (where is the text?) and recognition (what does it say?). This assignment trains the **recognition** models; plate boxes come from the dataset annotations.

---

## Dataset

**Source:** Kaggle [Indian vehicle dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset) (`saisirishan/indian-vehicle-dataset`): car photos with Pascal VOC XML files giving each plate's bounding box and its text. The training notebook downloads it automatically with `kagglehub`.

**Methodology:** from the `google_images` subset (440 annotations) we built our own recognition dataset:

| Step | Kept | Why |
|---|---|---|
| Remove `- Copy` duplicates, check image and XML | 428 | Duplicates and broken pairs |
| Normalise labels (uppercase, A–Z and 0–9 only) | 428 | One consistent target per plate |
| Valid Indian plate pattern + real state code | 352 | Removes junk labels (e.g. `TERRANO`) |
| Plate ≥ 80×22 px and aspect ratio 2.5–7 | 276 | Readable, single-row plates only |
| Contrast filter, ranked by sharpness (Laplacian variance) | 275 | Keeps the clearest crops |
| One image per plate number | 134 | Prevents the same plate in train and val |
| 100 sharpest, shuffled (seed 42) | **80 train / 20 val** | 80/20 split |

Crops include a small margin (4% horizontal, 12% vertical). Labels use the PPOCR recognition format:

```
train/IN_001.png	MH12AB1234
```

**Licence:** the source dataset is CC BY-NC-ND, so the images are **not** included in this repository; the notebook downloads them from Kaggle.

---

## Training setup

Both models used identical settings, so any difference comes from the model.

| Setting | Value |
|---|---|
| Models | `PP-OCRv5_server_rec`, `PP-OCRv4_server_rec`, from official pretrained weights |
| Dictionary | Each model's original (`ppocrv5_dict.txt`, `ppocr_keys_v1.txt`) |
| Epochs / steps | 80 epochs, 720 steps |
| Batch size | 16 per card (multi-scale sampler) |
| Learning rate | 1e-4, cosine schedule, 2 warm-up epochs |
| Evaluation | Every 25 steps on the validation set; best checkpoint kept |
| Hardware | Google Colab, Tesla T4 GPU |
| Framework | PaddlePaddle 3.2.0 (GPU, CUDA 12.6), PaddleOCR (GitHub `main`) |

---

## How to run

1. Open `PPOCR_License_Plate_Training.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all** (about 45 minutes, mostly the two training runs). Allow Google Drive access when asked. Outputs are saved to `MyDrive/award/` on your Drive.
3. Open `PPOCR_License_Plate_Inference.ipynb` with the **same Google account** and run all. A GPU or CPU runtime both work. It reads the exported models and test photos that step 2 saved to Drive.
4. Check the final **Run status** cell of each notebook: it should print `ALL STEPS COMPLETED`.

**Notes**

- Inference uses PaddleOCR's own predictor (`tools/infer/predict_rec.py`) rather than the `paddleocr` pip package, which pulls in `paddlex`/`modelscope` and clashes with the PyTorch preinstalled on Colab.
- That predictor only accepts some model names, so the inference notebook blanks the `model_name` field in a local copy of the v4 `inference.yml`. Weights and preprocessing are unchanged.

---

## Repository structure

```
├── PPOCR_License_Plate_Training.ipynb    # training pipeline (with outputs)
├── PPOCR_License_Plate_Inference.ipynb   # inference pipeline (with outputs)
├── README.md
└── results/
    ├── training_curves.png       # loss and accuracy per step, v5 vs v4
    ├── eval_summary.csv          # baseline vs fine-tuned metrics
    ├── val_predictions.csv       # every validation plate, both models
    ├── predictions.csv           # inference results on full photos
    ├── summary.csv               # inference summary statistics
    ├── error_analysis.png        # confidence vs plate size, correct vs wrong
    ├── failing_crops.png         # the plates both models got wrong
    └── annotated/                # sample annotated car photos
```

---

## Limitations and future work

- **Small dataset.** With 20 validation plates, each plate is worth 5 points, so differences under about three plates are noise. The other dataset subsets (`State-wise_OLX`, `video_images`, 1,524 more valid labels) would give a much stronger test.
- **Label convention.** Temporary "TEMP" plates were labelled without that word. They should either be excluded or labelled as printed, and both models retrained.
- **Single run per model.** Repeated runs with different seeds would show the variance.
- **Equal epoch budget.** v4 was still improving at epoch 80; choosing epochs per model by validation would compare each at its best.
- **No detector.** Inference uses annotation boxes; a full ANPR system would add a plate detector (for example PP-OCR's detection model).

---

## References

- [PaddleOCR documentation](https://www.paddleocr.ai/main/en/index.html)
- [PaddleOCR on GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Indian vehicle dataset on Kaggle](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset)
