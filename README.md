# subhkl
Solving crystal orientation from 2D Laue diffraction images

---

## Installation

### Option 1: Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver. If you don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create a virtual environment and install the project:

```bash
uv venv env
source env/bin/activate  # On Windows: env\Scripts\activate
uv pip install -e .
```

### Option 2: Using standard Python venv

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
python -m pip install -e .
```

### Installing test dependencies

```bash
uv pip install -e ".[test]"  # with uv
# or
python -m pip install -e ".[test]"  # with pip
```

### Installing accelerated versions

We use JAX is to enable GPU-accelerated optimization algorithms. Subhkl uses the CPU version of JAX by default, but can be installed
with support for NVIDIA or AMD GPUs.

**Installation options:**

```bash
# CPU-only JAX acceleration (faster, but no GPU)
pip install subhkl

# NVIDIA GPU support (CUDA 12.x)
pip install subhkl[cuda12]

# AMD GPU support (ROCm)
pip install subhkl[rocm]
```

**For development (editable install):**

```bash
# CPU JAX version
uv pip install -e .

# CUDA 12
uv pip install -e ".[cuda12]"

# ROCm (AMD)
uv pip install -e ".[rocm]"
```

## Running with docker

Building:

```
docker build -t subhkl .
```

Running:

```
docker run -it --rm --name=subhkl --gpus all subhkl
```

subhkl will be available for import inside of Python in the container.

## Workflow example (without normalization, for now)

You will need to get the raw mesolite IMAGINE images from GitLab.
Assume that they are stored in the folder `mesolite_202405`.

The script `run_all_imagine.sh` runs the full workflow for a single
image. You can use the following command to apply the script to all the images in
`mesolite_202405`. This will generate a `.mtz` file for each input image.

```bash
for Z in mesolite_202405/*.tif; do run_all_imagine.sh $Z& done
```

To merge the output `.mtz` files, you can use `reciprocalspaceship`. We
will probably add this as a command, but for now the following python code 
works.

```python
import reciprocalspaceship as rs
import os

mtzs = []
for file in os.listdir("mesolite_202405"):
    if os.path.splitext(file)[1] == ".mtz":
        mtzs.append(rs.read_mtz(os.path.join("mesolite_202405", file)))
rs.concat(mtzs).hkl_to_asu().write_mtz("mesolite_202405/meso.mtz")
```

which creates a single `.mtz` file `mesolite_202405/meso.mtz` that contains
all reflections.

## Plotting after the fact

`finder` and `rbf-integrator` draw their unrolled-detector plots while they
run, which costs both rendering time and a lot of disk at the 600 dpi they use.
Two commands rebuild those plots later from the HDF5 files, so a run can be
told to draw nothing and still be looked at afterwards:

```bash
# run the search without drawing anything
python -m subhkl.io.parser finder images.h5 MANDI --output-filename found.h5

# ... and draw it later, from the two HDF5 files
python -m subhkl.io.parser finder-visualize images.h5 found.h5
python -m subhkl.io.parser integrator-visualize images.h5 integrated.h5
```

`images.h5` is the reduced (or merged) image stack the step ran on; the peaks
file supplies the peak centres and the width fitted to each one. Neither
command re-runs the search, so this is quick, and `--dpi` trades resolution for
size. Keep the two HDF5 files -- from a benchmark artifact, for instance -- and
any run can be spot-checked without the raw data.

Each peak is outlined at the size it was actually fitted with, not at a fixed
marker size: a circle of radius `n_sigma * sigma` for the sparse-RBF finder's
isotropic widths, and the corresponding ellipse for the RBF integrator's full
covariance. `--n-sigma` sets which contour that is (2 by default, holding about
86% of a 2D Gaussian's flux). The finders that fit no width are drawn as plain
markers, because there is no size to draw.

## Physics and Conventions

This project uses the **Laue Equation** to relate Miller indices $(h, k, l)$ to the scattering vector $Q$:

$$Q_l = 2\pi R \cdot U \cdot B \cdot \mathbf{h}$$

where:
- **$\mathbf{h}$**: Miller indices vector $\begin{pmatrix} h \\ k \\ l \end{pmatrix}$.
- **$B$**: Reciprocal lattice matrix (Cartesian system). Transforms Miller indices to reciprocal space units ($1/\text{\AA}$ if $2\pi$ is not absorbed).
- **$U$**: Orientation matrix (Sample to Cartesian). Transforms reciprocal lattice to the goniometer/sample frame.
- **$R$**: Goniometer rotation matrix (Lab to Sample). Calculated from goniometer axes and angles using Mantid's `SetGoniometer` convention ($R = R_{\text{omega}} R_{\text{chi}} R_{\text{phi}}$).

The scattering vector $Q$ is defined by the change in wavevector:
$$Q = k_f - k_i = \frac{2\pi}{\lambda} (\hat{k}_f - \hat{k}_i)$$

where $\hat{k}_f$ and $\hat{k}_i$ are unit vectors along the scattered and incident beam directions, respectively.

### Coordinate Systems
- **Lab Frame**: $Z$ is along the incident beam, $Y$ is vertically upward.
- **Sample Frame**: Attached to the innermost goniometer axis.
- **Angles**: $2\theta$ is the scattering angle, $\phi$ is the azimuthal angle.

## Developer Guide

### Running Tests

```bash
pytest -v
```

Tests that need the mesolite dataset are marked `mesolite` and are left out of
that run, because the dataset is downloaded from Zenodo on first use and that
takes hours. Ask for them explicitly:

```bash
pytest -v -m mesolite

# or download only the first few files, which is enough for most of them
MESOLITE_MAX_FILES=1 pytest -v -m mesolite
```

The other markers are `slow` and `integration`; CI runs unit tests with
`-m "not slow and not integration and not mesolite"`.

### Running Linting

```bash
ruff format --check && ruff check
```

To auto-fix formatting issues:

```bash
ruff format
```

### Publishing a Release

The project uses automated publishing to PyPI and GitHub Container Registry when you create a semantic version tag.

**Prerequisites:**

1. **Set up PyPI trusted publishing** (one-time setup):
   - Go to https://pypi.org/manage/account/publishing/
   - Add GitHub as a trusted publisher for your repository
   - Set the workflow filename to `publish.yaml`
   - Set the environment name to `pypi`

2. **Create and push a release tag**:

```bash
# Create a new version tag (e.g., v0.1.0)
git tag v0.1.0

# Push the tag to GitHub
git push origin v0.1.0
```

This will automatically:
- Build and publish the package to PyPI
- Build and push a Docker image to `ghcr.io/zjmorgan/subhkl:v0.1.0` (and `latest`)
