# Biometrics
Intro to biometrics (facial recongintion) using Python. Based on the tutorial series by Nicholas Renotte.

This project is split into two parts:
- a notebook for training and evaluating the model,
- a verification app to test the model with live inference.

## Prerequisites

- macOS with Python 3.11 available.
- Optional:
  - (for maximum compatibility): `pyenv`, if you want to use the exact project Python from `.python-version`.
  - Docker for running the verification app.

## Notebook setup with venv

Create a local virtual environment and install the notebook dependencies:

```bash
pyenv install 3.11.15
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

If Python 3.11 is already installed, you can skip `pyenv install 3.11.15`.

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name biometrics --display-name "Python (biometrics)"
```

Start JupyterLab and open the notebook:

```bash
jupyter lab notebooks/facial-verification.ipynb
```

In Jupyter or VS Code, select the `Python (biometrics)` kernel before running the notebook.


## Apple Silicon GPU

The default notebook setup runs on CPU, which is the safest path for reproducibility.

To opt in to Apple Metal GPU support:

```bash
source .venv/bin/activate
python -m pip install tensorflow-metal==1.2.0
USE_GPU=1 jupyter lab notebooks/facial-verification.ipynb
```

If TensorFlow crashes or behaves unexpectedly, restart Jupyter without `USE_GPU=1`.

Inside an IDE (kernel launched by the editor, not the shell), set `USE_GPU=1` then restart the kernel: in **PyCharm** via Settings → Languages & Frameworks → Jupyter → managed server → Environment variables; in **VS Code** add `USE_GPU=1` to a `.env` file in the project root.

## Verification app to test models with live inference

- backend: FastAPI
- frontend: React + Vite
- start:

```bash
./app/start.sh
```


### Problems

- Apple Silicon GPU support can be fragile. The stable path is TensorFlow 2.16 on Python 3.11 with CPU by default, and `tensorflow-metal` only when explicitly enabled.
- The original tutorial uses older TensorFlow functionality, so parts of the notebook have been ported to current TensorFlow/Keras behavior.

## References

General biometrics terminology and system background are based on:

- Anil K. Jain, Arun A. Ross, and Karthik Nandakumar. "Introduction to Biometrics." Springer, 2011. https://doi.org/10.1007/978-0-387-77326-1

The Siamese face-verification model and implementation are based on Nicholas Renotte's tutorial and companion code:

- Tutorial playlist: https://www.youtube.com/watch?v=bK_k7eebGgc&list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH
- Original tutorial repository: https://github.com/nicknochnack/FaceRecognition/tree/main

The model idea follows the Siamese network approach from:

- Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. "Siamese Neural Networks for One-shot Image Recognition." ICML Deep Learning Workshop, 2015. https://www.cs.utoronto.ca/~rsalakhu/papers/oneshot1.pdf

The LFW face images used for impostor examples are downloaded through KaggleHub from the Kaggle mirror:

- Kaggle LFW People dataset: https://www.kaggle.com/datasets/atulanandjha/lfwpeople

The original dataset is Labeled Faces in the Wild (LFW), originally hosted by UMass at http://vis-www.cs.umass.edu/lfw/ (the original page may no longer be available). Dataset citation:

- Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. "Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments." Technical Report 07-49, University of Massachusetts, Amherst, 2007.
