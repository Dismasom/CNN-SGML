# CNN-SGML

Code for the paper: **"Solution-Guided Machine Learning for Physical Field Prediction in Complex Geometries"**.

### Requirements
- Python: 3.10
- PyTorch: 2.5.1
- CUDA: 11.8
- numpy: 1.26.4
- opencv-contrib-python：4.10.0.84
- matplotlib: 3.8.3

## Usage
### Train
```bash
python train_main.py 
```

### Dataset format (Data.rar)
`Data.rar` contains the dataset for the pipe internal flow problem, including `train.npy`, `val.npy`, and `test.npy`.

Each `.npy` file has shape **[N, 256, 768]** (N = number of samples).  
For each sample, the **width dimension (768)** is split into three **256×256** blocks:

- **[0:256]**: `approximate`
- **[256:512]**: `mask`
- **[512:768]**: `target`

Extract the archive and place the files under `./DATA/` (e.g., `./DATA/train.npy`, `./DATA/val.npy`, `./DATA/test.npy`).
