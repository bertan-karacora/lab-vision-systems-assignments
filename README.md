# Assignments for Lab Vision Systems

## Links

- [Course website](https://www.ais.uni-bonn.de/SS24/4308_Lab_Vision_Systems.html)

## Assignment topics

1. PyTorch basics, Autograd and Fully Connected Neural Networks
2. Optimization & Convolutional Neural Networks (CNNs)
3. Popular Architectures and Transfer Learning
4. Recurrent Neural Networks (RNNs & LSTM)
5. Autoencoders (AEs), Denoising and Variational AEs
6. Generative Adversarial Networks (GANs)
7. Deep Metric and Similarity Learning
8. Semantic Segmentation and Introduction to Final Project

## Setup

For Python-only projects, as long as any system requirements are only packages I already have installed (or intend to keep), I prefer `venv` over alternatives like `Docker` or `Anaconda`.

### System requirements

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt install -y python3.12 python3.12-venv python-is-python3
```

### Configuring venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Alternative: Python 3.12 as local executable

Assuming you have installed:

- wget
- build-essential

Installing Python 3.12 in `~/.local/`:

```bash
wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz
tar zxfv Python-3.12.2.tgz
rm Python-3.12.2.tgz
cd Python-3.12.2
./configure --prefix="$HOME/.local/" --enable-optimizations --with-lto
make
make install
cd ..
rm -rf Python-3.12.2
```

Every assignment is a Python package which can be installed via:

## Points

| Assignment |  Max  | Points |
| :--------: | :---: | :----: |
|     1      |       |        |
|  **SUM**   |       |        |
