# Real Time Group Face Recognition Using FaceNet
RISHIT JHA | KU2407U181 | 202407020172 | CSE-D

## Delivery status
The source and reports are prepared. No group photographs were supplied. A trained group model, actual facial-recognition metrics, webcam screenshots, and a genuine group dataset therefore cannot be included. Six synthetic-vector software checks were executed successfully; these do not establish recognition quality. The webcam and pretrained-encoder integration must be run on the demo laptop. The code does not train FaceNet from scratch: it freezes a pretrained FaceNet-style InceptionResnetV1 encoder and trains a small identity classifier. The paper's original architecture and embedding size are different from this implementation.

## Install (Windows, Python 3.11 recommended)
Open a terminal in this folder:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python Group_x_1.py selftest
```

Linux/macOS: use `python3.11 -m venv .venv`, then `source .venv/bin/activate`.
The pinned stack is intended for Python 3.10/3.11. It was not installed end-to-end in the authoring environment. Use a clean environment to avoid Torch/NumPy conflicts. First encoder use downloads pretrained weights (~107 MB); internet is required then. After caching weights, local inference can run offline. GPU is optional; `--device cpu` forces CPU. If camera 0 fails, try `--camera 1`. A desktop GUI session is needed for webcam windows.

## Collect your dataset
Use only participants who agree to this project. Repeat these commands for all four members, replacing the name. Keep names exactly the same across splits. These are separate recording sessions: change the background/lighting or come back later before validation and test. Do not randomly split adjacent video frames.

```bat
python Group_x_1.py capture --name "Rishit Jha" --split train --count 60
python Group_x_1.py capture --name "Rishit Jha" --split validate --count 20
python Group_x_1.py capture --name "Rishit Jha" --split test --count 20
```

Repeat for the other three actual members. The program saves only frames containing exactly one confident face. Change pose, glasses if normally worn, distance and lighting between frames. Q stops early. It appends new files rather than overwriting previous sessions. The 60/20/20 plan gives 100 images per member and 400 member images total; these are targets, not measured counts.

Collect consenting nonmembers too:

```bat
python Group_x_1.py capture --name unidentified --split validate --count 40
python Group_x_1.py capture --name unidentified --split test --count 40
```

Use at least two nonmembers for validation and two DIFFERENT nonmembers for test (20 frames each). The code cannot verify that identities or capture sessions are disjoint; manage this yourself. Never put unidentified in train. With the suggested counts: 240 train + 120 validation + 120 test = 480 images, including unknown people. Exact file duplicates across splits are rejected automatically; near duplicates are not.

## Train, test and demonstrate

```bat
python Group_x_1.py train
python Group_x_1.py evaluate
python Group_x_1.py live
python Group_x_1.py results-report
```

Training selects the classifier epoch using known-member validation loss. It then selects probability/distance rejection thresholds using known and unknown validation images. The test set is never used to fit or tune the model. `--epochs 150` is the default ceiling; early stopping uses patience 15. Training only the 512-to-K classifier makes this suitable for a small enrollment dataset. No data augmentation is applied.

In the webcam window, green labels indicate accepted members; red labels say `unidentified`. No detected face produces no identity label. Q quits. S saves an actual annotated screenshot in outputs. Softmax scores are not calibrated confidence probabilities. This is a classroom demonstration, not access control; it has no liveness or spoof detection.

## Generated files
- `outputs/classifier.npz`: trained linear head and member centroids.
- `outputs/model.json`: class order, selected thresholds, counts and chosen epoch.
- `outputs/history.json` and `training_curves.png`: real training history.
- `outputs/metrics.json`: per-class precision/recall/F1, open-set accuracy and rejection rates.
- `outputs/predictions.csv`: every accepted test image and its prediction.
- `outputs/confusion_matrix.csv` and `.png`: row/column order is members followed by unidentified.
- `outputs/sample_predictions.png`: first six actual test-image predictions.
- `outputs/*_rejected.json`: detection/decoding failures, excluded from recognition metrics.
- `outputs/live_*.png`: screenshots saved manually with S.
- `outputs/Measured_Results.docx`: measured-results appendix generated after evaluation.

Count rejected images as detection failures separately. The reported dataset-processing time includes image loading/detection/embedding, excludes model initialization, and is not webcam FPS. Live FPS is an approximate processing-loop measurement.

## Submission
The attached assignment specifies face recognition, a four-person group, a 3–4-page double-column Part I review and one runnable code file. The separate report template recommends 25–35 pages. Both report formats are supplied. The PDF says Semester V while the generic cover says VI; the project report uses Semester V. Its 2025 deadline conflicts with academic year 2026–27: check the actual LMS deadline.

1. Fill the actual group number, three missing members and faculty details in the editable reports. Sign certificates/declarations only after review; no signature or faculty approval has been invented.
2. Collect the real dataset and run all commands above. Review incorrect predictions and detection failures.
3. Insert the generated results/graphs and at least five real test cases into Chapters 8–9, or append `Measured_Results.docx`. Include live screenshots. The current reports explicitly mark these results unmeasured.
4. Change `Group_x_1.py` to your actual group number and rename the short review `Group_<number>_report_1.pdf`.
5. Compress code, requirements, README, real data/train, data/validate, data/test and outputs into `Group_<number>_code.zip`. Include cached encoder weights if your demo laptop will have no internet and the weights are not already cached there.
6. Review and understand the implementation, acknowledge assistance as required by your course, and submit one copy per group.

; DeepFace uses strong alignment; DeepID learns identity features; FaceNet supports compact distance-based representations.
- A confident classifier can still accept an unknown person. Thresholds reduce this risk, but do not eliminate it.
