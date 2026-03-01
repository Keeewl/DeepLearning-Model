# DeepLearning-Model
This is a repository for implementing deep learning models.

## Installation
Conda env:
```bash
# Create conda env:
conda create -n dl-model python=3.10 -y
conda activate dl-model

# Install pytorch for NVIDIA GeForce RTX 5090:
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Or you can directly setup conda environment:
conda env create -f environment.yml

# Check env:
python env/check.py
```

## Model list
1. MLP
    ```
    1.1 Fitting sin(x)
    ```

2. CNN

3. RNN -> Attention

4. Transformer

5. Diffusion

6. Conditioning