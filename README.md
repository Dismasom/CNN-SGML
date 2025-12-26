# CNN-SGML

Code for the paper: **"Solution-Guided Machine Learning for Physical Field Prediction in Complex Geometries"**.

## Setup
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
### Train
```bash
python train_main.py 
```

### Evaluate
```bash
python eval.py --ckpt path/to/checkpoint
```

## Data
Place it in: `Data/`


}
```

## License
MIT (or see `LICENSE`).
