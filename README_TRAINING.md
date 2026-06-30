Training guide for Plant Disease Detector

This repository includes a training helper script `train.py` to train a Keras model using transfer learning (MobileNetV2).

Required dataset structure:

```
data/
  train/
    Leaf Spot/...
    Rust/...
    Healthy/...
    Powdery Mildew/...
  val/
    Leaf Spot/...
    Rust/...
    Healthy/...
    Powdery Mildew/...
```

Steps to train (recommended):

1. Create and activate a Python virtual environment.

2. Install packages (only if you plan to train locally):

```powershell
pip install tensorflow pillow
```

3. Run the training script:

```powershell
python backend/train.py --data_dir ../data --epochs 15 --batch_size 32 --output backend/model.h5
```

4. After training completes, restart the backend server. The backend will automatically load `backend/model.h5`.

Tips for reaching 90%+ accuracy:
- Use a well-curated labeled dataset (PlantVillage is a common source).
- Use transfer learning and fine-tuning (the provided script does this).
- Apply data augmentation and class balancing.
- Experiment with learning rates, number of unfrozen layers, and more epochs.

If you'd like, I can help prepare dataset download scripts or run training locally if you provide the data or allow me to fetch a public dataset.
