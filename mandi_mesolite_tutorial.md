# Data Reduction on MANDI mesolite data

## Setup

- Get the code by cloning `subhkl` from [here](https://github.com/zjmorgan/subhkl),
making sure to get the most recent branch, and install `subhkl` by running
```commandline
git clone https://github.com/zjmorgan/subhkl
```
```commandline
cd subhkl
```
```commandline
pip install -e .
```
(it needs to be an editable installation)
- You need the following:
  - Nexus file (`.nxs.h5`)
  - Crystallographic parameters
    - `a = 18.39`
    - `b = 56.55`
    - `c = 6.54`
    - `alpha = 90`
    - `beta = 90`
    - `gamma = 90`
    - `space_group = "Fdd2"`
    - `wavelength_min = 2`
    - `wavelength_max = 4.5`

## Running the workflow

For a complete example, see `examples/mandi_multi_run.sh`.

### `reduce`

First, reduce the Nexus event data to a dense image stack:
```commandline
python -m subhkl.io.parser reduce MANDI_11612.nxs.h5 MANDI_11612.reduce.h5 MANDI
```

### `finder`

Find and integrate peaks in the reduced data:
```commandline
python -m subhkl.io.parser finder MANDI_11612.reduce.h5 MANDI
 --output-filename finder_output_11612.h5 
 --finder-algorithm thresholding
 --thresholding-noise-cutoff-quantile 0.99
 --region-growth-minimum-intensity 3.0
 --region-growth-maximum-pixel-radius 12.0
 --peak-minimum-pixels 40
```

### `indexer`

Index the peaks to find the orientation matrix ($U$) and refine lattice parameters:
```commandline
python -m subhkl.io.parser indexer finder_output_11612.h5 indexed_11612.h5 18.39 56.55 6.54 90 90 90 Fdd2 --wavelength-min 2.0 --wavelength-max 4.5
```

### `integrator`

Refine peak positions and integrate intensities using Gaussian fitting:
```commandline
python -m subhkl.io.parser integrator MANDI_11612.reduce.h5 MANDI peak_predictor_11612.h5 integrator_11612.h5 --integration-method gaussian_fit
```
The `gaussian_fit` method refinements peak centers to sub-pixel accuracy, which is essential for high-quality indexing metrics.

#### Time-of-flight gating

On MANDI, `reduce` also keeps each event's time of flight next to the images
(`events/tof`; `--no-events` skips this) and `merge-images` carries the events
into the merged stack. When the images file has events, `integrator` no longer
integrates a reflection on the image of the whole wavelength band: it uses only
the events inside a time window around the reflection's predicted time of
flight and measures the background in time sidebands on the same pixels.

```commandline
python -m subhkl.io.parser integrator scan_master.h5 MANDI peak_predictor.h5 integrator.h5 --tof-gate auto
```

- `--tof-gate auto` (default) measures the window from the data. The events
  near all predicted positions are stacked in `t / t_predicted - 1`; the window
  is the interval holding `--tof-gate-coverage` (default 0.95) of the Bragg
  excess. A common offset, as from a cell-scale or flight-path error, is
  absorbed. If the stack shows no excess the predictions do not line up with
  the events in time and the command stops with an error.
- `--tof-gate lo,hi` sets the window edges relative to the predicted time, for
  example `--tof-gate=-0.01,0.015`.
- `--tof-gate off` integrates the whole band as before.

The sideband background is scaled to the gate by each image's own time
spectrum (`--tof-sideband-measure spectrum`, the default), not by the length
of the intervals in time, because the spectrum is curved and falls off at the
band edges. `--tof-sideband-measure time` assumes a rate flat in time.

To check that a merge's CC½ is Bragg signal, integrate once more with the gate
moved off the reflections, for example `--tof-gate=0.06,0.11`, and merge that:
whatever CC½ remains in a resolution shell is not from the reflections.

Each output row is then one reflection at its own wavelength. `integrator.h5`
records the window used (`tof_gate`), the stacked time profile it was read
from, and per row the gate centre (`peaks/tof`, microseconds) and the expected
background inside the gate (`peaks/tof_background`, counts per pixel).

### `finder-visualize` / `integrator-visualize`

Both `finder` and `rbf-integrator` can draw an unrolled view of the detector
with every peak marked, but only while they run, and at a resolution that is
expensive to keep for many runs. These two commands draw the same plots
afterwards, from the image stack and the peaks file:

```commandline
python -m subhkl.io.parser finder-visualize MANDI_11612.reduce.h5 finder_output_11612.h5 --dpi 150
```

Each peak is drawn at the width the finder fitted it — a circle of radius
`--n-sigma` times that width — so the plot shows not just where the peaks are
but how big the search thought they were, which a fixed marker size cannot.
The integrator's plots come back the same way, at the shape it fitted:

```commandline
python -m subhkl.io.parser integrator-visualize MANDI_11612.reduce.h5 rbf_integrator_11612.h5
```

One image is written per run, named after the run, next to the peaks file
unless `--output-dir` says otherwise.
