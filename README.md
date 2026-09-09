# Multi-Object Detection Model

A collection of computer vision notebooks covering **object detection** (multiple objects/classes per image) using two different approaches — **YOLOv8** and **Faster R-CNN** — plus a bonus image classification model.

## 📁 Repository Structure

```
multi_object_detection_model/
├── Untitled29.ipynb                  # Main project: Faster R-CNN multi-object detector (Aquarium dataset)
├── Untitled21.ipynb                  # YOLOv8 object detector (Wildlife dataset)
├── AnimalImagesClassifier.ipynb      # CNN image classifier (bonus/side project, not detection)
├── best.pt                           # Trained YOLOv8 weights (wildlife detector)
└── animal_images_classifier.keras    # Trained Keras CNN weights (animal classifier)
```

> Note: file names come straight from the original Colab notebooks (`Untitled21`, `Untitled29`) and haven't been renamed.

---

## 🐠 1. Faster R-CNN — Aquarium Multi-Object Detection (`Untitled29.ipynb`)

The core project in this repo: a **Faster R-CNN** model (MobileNetV3-Large FPN backbone) trained from scratch on the [Aquarium Dataset](https://www.kaggle.com/datasets/sharansmenon/aquarium-dataset) to detect and localize multiple marine animals in a single image.

**Classes detected (7):** `fish`, `jellyfish`, `penguin`, `puffin`, `shark`, `starfish`, `stingray`

### Pipeline

1. **Data download** — Kaggle Aquarium dataset via `opendatasets`, annotated in COCO format.
2. **Annotation parsing** — builds category/image ID maps from `_annotations.coco.json` and groups annotations per image.
3. **Visualization** — draws ground-truth bounding boxes on sample images with PIL/Matplotlib.
4. **Augmentation** — Albumentations pipeline (resize to 600×600, horizontal/vertical flip, brightness/contrast jitter, color jitter) with COCO-format bounding box handling.
5. **Custom Dataset class** — `AquariumDetection`, built on `pycocotools.COCO`, loads images + bounding boxes for train/valid/test splits.
6. **Model** — `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn` (pretrained backbone) with the box predictor head replaced for 7 custom classes.
7. **Training** — SGD optimizer (lr=0.005, momentum=0.9, weight decay=0.0005), 10 epochs, tracking classifier/box-reg/objectness/RPN losses.
8. **Evaluation** — `torchmetrics.detection.MeanAveragePrecision` computed on the validation set after every epoch.
9. **Inference & visualization** — side-by-side plots of the original image vs. predicted boxes (confidence threshold 0.5) on the test set.
10. **Export** — trained weights saved as `fasterrcnn_aquarium.pth` (downloadable from Colab).

### Results (after 10 epochs)

| Metric | Value |
|---|---|
| mAP @ IoU 0.5:0.95 | 0.331 |
| mAP @ IoU 0.5 | 0.683 |
| mAP @ IoU 0.75 | 0.272 |
| mAR @ 100 detections/image | 0.435 |
| Final training loss (avg) | 0.601 |

### Tech Stack
PyTorch, torchvision (Faster R-CNN), pycocotools, Albumentations, torchmetrics, OpenCV, Matplotlib.

---

## 🦁 2. YOLOv8 — Wildlife Object Detection (`Untitled21.ipynb`)

A second detector built with **Ultralytics YOLOv8** on the [Wildlife Object Detection Dataset (YOLO format)](https://www.kaggle.com/datasets/ankanghosh651/object-detection-wildlife-dataset-yolo-format).

**Classes detected (4):** `buffalo`, `elephant`, `rhino`, `zebra`

### Pipeline

1. Download the dataset from Kaggle via `opendatasets`.
2. Load a pretrained `yolov8n.pt` (nano) checkpoint from Ultralytics.
3. Fine-tune on the wildlife dataset for 20 epochs at 640×640 resolution using `model.train(...)`.
4. Best weights are auto-saved to `runs/detect/train/weights/best.pt` — this is the `best.pt` file included in this repo.

### Results (final epoch, validation set — 150 images / 262 instances)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| All (overall) | 0.977 | 0.898 | 0.961 | 0.821 |
| Buffalo | 0.982 | 0.896 | 0.938 | 0.787 |
| Elephant | 0.946 | 0.904 | 0.973 | 0.807 |
| Rhino | 1.000 | 0.942 | 0.992 | 0.906 |
| Zebra | 0.980 | 0.852 | 0.941 | 0.782 |

Training took ~0.16 hours (20 epochs) on a Tesla T4 GPU. Inference speed: ~3.2 ms/image.

### Tech Stack
Ultralytics YOLOv8, PyTorch (backend).

### Using the trained weights

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("your_image.jpg", conf=0.5)
results[0].show()
```

---

## 🐶 3. Bonus: Animal Image Classifier (`AnimalImagesClassifier.ipynb`)

A simpler side project — a CNN **image classifier** (not object detection) that assigns a single label to an image from a small set of animal classes (`dog`, `cat`, `elephant`/`cow`, `sheep`/background, depending on the training run).

### Pipeline

1. Loads images from local `train`/`test` directories with `keras.utils.image_dataset_from_directory` (128×128, RGB).
2. Normalizes pixel values to [0, 1] and applies data augmentation (random flip, rotation, zoom).
3. A custom CNN: 4 convolutional blocks (32 → 64 → 128 → 256 filters, each with `MaxPooling2D`) → `Flatten` → Dense(128) → Dense(64) → Dense(4, softmax).
4. Trained for 20 epochs with the Adam optimizer and sparse categorical cross-entropy loss.
5. Model saved as `animal_images_classifier.keras` and reloaded for inference on new images via OpenCV.

### Results (final epoch)

| Metric | Value |
|---|---|
| Training accuracy | 0.800 |
| Validation accuracy | 0.744 |

### Tech Stack
TensorFlow / Keras, OpenCV, NumPy.

> ⚠️ This notebook references a local path (`D:\animal Dec`) and hardcoded image files for inference — it will need path updates to run outside the original author's machine.

---

## 🚀 Getting Started

```bash
git clone https://github.com/devanshu6738/multi_object_detection_model.git
cd multi_object_detection_model
pip install ultralytics torch torchvision opencv-python albumentations \
    pycocotools torchmetrics tensorflow opendatasets matplotlib pandas numpy
```

Then open any of the notebooks in Jupyter/Colab. The Faster R-CNN and YOLOv8 notebooks download their datasets automatically via `opendatasets` (a Kaggle account/API key is required).

## 🔧 Possible Improvements

- Consolidate the two detection notebooks into reusable Python scripts/modules instead of Colab notebooks.
- Add a shared `requirements.txt` and remove the hardcoded local path in the classifier notebook.
- Extend training beyond 10–20 epochs and add learning-rate scheduling for the Faster R-CNN model, whose mAP was still improving at the last epoch.
- Add a unified inference script that can run either detector on a given image/video.
- Move the animal classifier into its own repository, since it's a classification task rather than object detection.

## 🙋 Author

**Devanshu** — [@devanshu6738](https://github.com/devanshu6738)
