# Real Time Group Face Recognition Using FaceNet

A Python webcam application that recognizes enrolled people and labels rejected matches as `unidentified`. It supports single-person enrollment and recognition of multiple enrolled members.

The project uses MTCNN for face detection and a frozen, pretrained InceptionResnetV1 encoder from `facenet-pytorch` to extract 512-dimensional face embeddings. It does not train FaceNet from scratch.

## Team

| Member | Student ID | Enrollment number |
|---|---|---|
| Neel Jeswani | KU2407U145 | 202407020140 |
| Bhavishya Sharma | KU2407U032 | 202407020032 |
| Priyanshu Mahapatra | KU2407U175 | 202407020166 |
| Ved Soni | KU2407U211 | 202407020201 |

B.Tech CSE, Semester V, Section D — Karnavati University.

## Working demonstration

The local webcam demonstration displayed the enrolled label `Rishit Jha`. Two screenshots recorded displayed processing rates of 12.2 and 12.5 FPS. These are individual observations, not an average benchmark or an accuracy measurement. Held-out recognition accuracy and unknown-person rejection rates have not yet been reported.

Report authors and enrolled identities are separate: adding a name to this README does not enroll that person in the model.

## Run the saved model on Windows

Requirements: Python 3.11, a webcam, and a desktop session. Internet is needed to install dependencies and download pretrained encoder weights on first use. A GPU is optional.

Download and extract the repository, then open PowerShell or the VS Code terminal in the folder containing `Group_x_1.py` and `requirements.txt`.

Before running, confirm that both saved-model files exist:

- `outputs/model.json`
- `outputs/classifier.npz`

These must come from the same successful training run. Raw training photographs are not required to run the saved model.

Run each command on its own line:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe Group_x_1.py live
```

No environment activation is required. After the initial setup, start the application with only:

```powershell
.\.venv311\Scripts\python.exe Group_x_1.py live
```

Do not run `train` just to view the existing demo: training replaces the saved model. The saved model recognizes its enrolled identities; new users must enroll their own faces to be recognized by name.

### Webcam controls

- **Q**: close the webcam window.
- **S**: save an annotated screenshot in `outputs`.
- Green rectangles and names indicate accepted matches.
- Red rectangles with `unidentified` indicate rejected matches.

If the default camera does not open:

```powershell
.\.venv311\Scripts\python.exe Group_x_1.py live --camera 1
```

To force CPU processing:

```powershell
.\.venv311\Scripts\python.exe Group_x_1.py live --device cpu
```

## Enroll yourself or add members

Back up `outputs/model.json` and `outputs/classifier.npz` before retraining. Keep existing members' training and validation images if you want to retain those members when training a new model.

Replace `Your Name` below with the person's actual name. Use the same spelling in every split.

```powershell
.\.venv311\Scripts\python.exe Group_x_1.py capture --name "Your Name" --split train --count 60
.\.venv311\Scripts\python.exe Group_x_1.py capture --name "Your Name" --split validate --count 20
.\.venv311\Scripts\python.exe Group_x_1.py capture --name "Your Name" --split test --count 20
```

Record train, validation and test images in separate sessions with varied lighting, poses and backgrounds. Do not reuse the same photos across splits. These counts are suggested targets, not measured dataset totals.

Capture consenting nonmembers for rejection calibration and evaluation:

```powershell
.\.venv311\Scripts\python.exe Group_x_1.py capture --name unidentified --split validate --count 40
.\.venv311\Scripts\python.exe Group_x_1.py capture --name unidentified --split test --count 40
```

Use different unknown people for validation and testing where possible. Never place `unidentified` images in `data/train`. Validation must contain every enrolled identity plus `unidentified`. The program requires at least five usable training images per identity; more varied images are recommended.

### Import existing photos

You can import JPG/PNG photos instead of recording everyone on the same webcam:

```powershell
.\.venv311\Scripts\python.exe Group_x_1.py import-images --name "Neel Jeswani" --split train --source "C:\FacePhotos\Neel\train"
.\.venv311\Scripts\python.exe Group_x_1.py import-images --name "Neel Jeswani" --split validate --source "C:\FacePhotos\Neel\validate"
.\.venv311\Scripts\python.exe Group_x_1.py import-images --name "Neel Jeswani" --split test --source "C:\FacePhotos\Neel\test"
```

Replace the paths with real folders on your computer. Use separate photos for each split and repeat for each person. Face extraction occurs during training or evaluation; importing an image does not guarantee that its face will be usable.

## Train and evaluate

After preparing the dataset:

```powershell
.\.venv311\Scripts\python.exe Group_x_1.py train
.\.venv311\Scripts\python.exe Group_x_1.py evaluate
.\.venv311\Scripts\python.exe Group_x_1.py live
.\.venv311\Scripts\python.exe Group_x_1.py results-report
```

- **One enrolled identity:** builds a normalized mean face template and calibrates a distance threshold using known and unknown validation images. No classifier epochs are needed.
- **Two or more identities:** trains a small linear classifier on frozen embeddings, selects its checkpoint using known-member validation loss, and calibrates probability and distance thresholds.
- **Evaluation:** uses a separate test set. Do not use test results to tune thresholds.
- **Results report:** requires evaluation outputs from the current saved model.

An optional software check can be run with:

```powershell
.\.venv311\Scripts\python.exe Group_x_1.py selftest
```

Synthetic checks test software logic; they do not measure recognition accuracy.

## Output files

All generated files are stored in `outputs`.

| File | Purpose |
|---|---|
| `model.json` | Enrolled labels, operating mode, thresholds and model metadata |
| `classifier.npz` | Saved recognition arrays, including face centroids |
| `history.json` | Classifier training history; empty in single-person mode |
| `validation_distances.png` | Single-person validation distances and threshold |
| `training_curves.png` | Multi-person classifier learning curves |
| `metrics.json` | Evaluation metrics |
| `predictions.csv` | Predictions for usable test images, including rejected identity matches |
| `confusion_matrix.csv` and `confusion_matrix.png` | Evaluation confusion matrix |
| `sample_predictions.png` | Example test predictions |
| `*_rejected.json` | Image decoding or face-extraction failures |
| `live_*.png` | Screenshots saved with S |
| `Measured_Results.docx` | Results document generated after evaluation |

Files appear only after the corresponding command succeeds. Count face-extraction failures separately from recognition errors. Live processing FPS differs from dataset evaluation timing.

## Troubleshooting

| Problem | Fix |
|---|---|
| Python 3.11 is not found | Install Python 3.11, reopen the terminal, and check `py -3.11 --version`. |
| NumPy installation fails under Python 3.13 | Use the Python 3.11 environment and commands above for the pinned requirements. |
| `model.json` or `classifier.npz` is missing | Obtain both matching files from the trained demo, or prepare the dataset and run `train` successfully. |
| Cannot open `Group_x_1.py` | Check the current folder and filename. Remove any extra space before `.py`. |
| No usable validation images | Capture known-person and unknown-person validation images, then review rejected-image logs. |
| Old error saying two named members are required | Replace the script with the revised version supporting single-person enrollment. |
| Camera fails | Close other camera applications, check camera access, or try `--camera 1`. |

## Files to upload

For the saved-model demo, include `Group_x_1.py`, `requirements.txt`, this README, and the matching `outputs/model.json` and `outputs/classifier.npz` files. Add the project report and demonstration screenshots if desired.

Do not upload `.venv`, `.venv311`, Python caches or credentials. Original face photographs are optional for inference, but a separate test dataset is needed to reproduce evaluation. Obtain participants' permission before publicly sharing their images or face templates.

## Limitations

Recognition depends on lighting, pose, image quality and enrollment diversity. Unknown-person rejection is imperfect. The system has no liveness or spoof detection and is intended for an educational demonstration, not security access control.

The pretrained encoder is provided by the [facenet-pytorch project](https://github.com/timesler/facenet-pytorch). The local classifier or template is fitted to the enrolled dataset; the original FaceNet paper is not reproduced from scratch.
