# LAP

White Matter Tract Segmentation as Multiple Linear Assignment
Problems (LAPs).

Based on the paper:
> Sharmin, Nusrat, Emanuele Olivetti, and Paolo Avesani. “White Matter Tract Segmentation as Multiple Linear Assignment Problems.” Frontiers in Neuroscience 11 (2018). https://www.frontiersin.org/articles/10.3389/fnins.2017.00754.

## Get Started

1. Install [miniconda](https://docs.conda.io/en/main/miniconda.html)
2. Clone this repository
3. Inside of this repository, run:

```sh
conda create -n LAP python=3.10.6 -y
conda activate LAP

pip install -r requirements.txt

# Optional: Cython build
./compile.sh
```

## Usage
Then, you can import and run LAP segmentation from your code:

```py
from .LAP_tract_segmentation import segment_lap

segment_lap(
    "test_tractogram.tck",
    ["train_1_tract.tck", "train_2_tract.tck", ...],
    "test.nii.gz",
    ["train_1.nii.gz", "train_2.nii.gz", ...],
    "test_tract_prediction.tck",
)
```