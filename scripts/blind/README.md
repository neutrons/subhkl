# Blind reduction prototype

Three scripts that take a merged time-of-flight Laue dataset (images plus
pixel-sorted events) to a unit cell, an orientation and a Laue class without
being told any of them. They are a prototype: one dataset has been through
them, and nothing here is wired into the CLI or the app.

## Steps

1. `blind_cell.py merged.h5 MANDI cell.npz`

   Every event has a full scattering vector. All events are histogrammed in
   the sample frame, counts above the local background are kept, and the 3D
   Fourier transform of that density is searched for peaks. They sit on the
   direct lattice (the transform is the crystal's Patterson function, whose
   origin peak repeats on every lattice vector). No reflection is located
   and nothing is fitted. The basis is then refined on the data, off the
   voxel grid, by maximising the lattice Fourier sum over all its lattice
   points. Because the events carry wavelengths, the cell has its absolute
   scale.

2. `blind_index.py cell.npz merged.h5 MANDI seed.h5`

   Reduces the basis, finds the rotations of the lattice itself (which
   bound the point group), reads the wavelength band off the events, and
   writes a solve-style file with space group `P 1`. That file seeds the
   existing refinement, which supplies detector geometry and per-run
   orientation corrections:

   ```
   python -m subhkl.io.parser solve merged.h5 refined.h5 --refinement band \
       --bootstrap seed.h5 --instrument MANDI --cell a,b,c,alpha,beta,gamma \
       --space-group "<lattice group>" --wavelength-min .. --wavelength-max .. --d-min 2.5
   ```

   Then `peak-predictor` and `integrator` run once with `--space-group "P 1"`.

3. `laue_class.py integrated_p1.h5 seed.h5`

   Symmetry mates have the same d-spacing, so two observations of them at
   the same wavelength share the scattering angle, the incident spectrum
   and the Lorentz factor. Their intensities compare directly, without
   wavelength normalisation. Each lattice rotation is tested on such
   matched pairs against two yardsticks from the same data: the same
   reflection measured twice, and unrelated reflections of the same d.

## Result on the one dataset tried

MANDI, 15 runs, known answer P6₁22 with a = b ≈ 80, c ≈ 238 Å.

| | Blind | Reference |
|---|---|---|
| Cell before refinement | 80.58, 80.58, 240.30 Å; 89.8°, 90.1°, 120.0° | |
| Cell after the existing refinement | 80.63, 80.63, 240.30 Å | 79.85, 79.85, 238.00 Å |
| Orientation, per run | within 0.06° of the reference | |
| Lattice rotations | 12 (622) | |
| Rotations passing the intensity test | 12 of 12, plus inversion | |
| Laue class | 6/mmm | 6/mmm |

Merged with `careless mono` in P622 after one P1 integration to 2.0 Å, the
blind route gives CC½ 0.899 overall, against 0.900 for the route that was
given the cell and space group (integrated to 1.41 Å). By shell (∞–4.0,
4.0–3.0, 3.0–2.5, 2.5–2.2, 2.2–2.0 Å): 0.986, 0.953, 0.775, 0.443, 0.321
blind against 0.982, 0.949, 0.803, 0.628, 0.352. The gap at 2.5–2.2 Å has
not been traced; the two runs differ in prediction limit and in refined
geometry as well as in how the cell was obtained.

The blind cell is about 1% larger than the reference. The reference cell
was supplied by hand, and a 2D Laue solve cannot see its scale; the event
times independently put the scale 0.9% above it.

Timing on one GPU: about 30 s to histogram 9×10⁸ events, under 2 minutes
for step 1 in total.

## Screw axis along c

`axial_screw.py merged.h5 MANDI refined.h5`

The reflections 00l share one detector position and differ only in
wavelength, so the integrator cannot separate them (184 of the 199 axial
rows are censored as unresolvable). But at that spot the time-of-flight
spectrum is the one-dimensional diffraction pattern along c*. Converting
each event near the axis to a fractional index l, and subtracting a
parallel row between the lattice rows as background, gives on this dataset
(l from 32 to 160):

| l mod 6 | Reflections | Net counts | Per reflection |
|---|---|---|---|
| 0 | 21 | 1919 ± 54 | 91 |
| 1 | 21 | 296 ± 37 | 14 |
| 2 | 22 | 285 ± 37 | 13 |
| 3 | 22 | 285 ± 37 | 13 |
| 4 | 21 | 242 ± 37 | 12 |
| 5 | 21 | 262 ± 37 | 13 |

Only l = 6n stands out, by a factor of seven: a 6₁ or 6₅ axis (the two are
enantiomorphs and cannot be told apart from intensities). The other five
classes sit on an equal pedestal that the background row does not remove;
it has not been explained.

## Centring and glide planes from zone sections

`zone_absences.py reduced.h5 MANDI cell.npz`

The same idea one dimension up. Every event is converted to fractional
indices of the conventional cell, and the counts at integer nodes above
the level between nodes are summed by reflection class: all nodes for
centring, nodes in a zone for a glide plane, nodes on an axis for a screw.

Tried on one MANDI run of a garnet (Ia-3d, a = 11.93 Å, band 1–3 Å, 1.3
million events). `blind_cell.py --d-min 0.7 --grid 384 --x-min 4` returns
the primitive cell 10.34 Å, 70.5°/109.5° in a few seconds; its
conventional cell is cubic, 11.93 Å, with two lattice points.

| Class | Nodes | Net counts per node |
|---|---|---|
| all: h+k+l even | 1436 | 307 |
| all: h+k+l odd (I forbids) | 1415 | 0.0 (−52 ± 113 in total) |
| 0kl: both even | 89 | 1142 |
| 0kl: both odd (I allows, glide forbids) | 80 | 17 |
| 0kl: mixed (I forbids) | 156 | 9 |
| hhl: 2h+l = 4n | 79 | 1157 |
| hhl: 2h+l = 4n+2 (I allows, d-glide forbids) | 71 | 20 |
| hhl: 2h+l odd (I forbids) | 147 | 10 |
| h00: h = 4n | 2 | 4284 |
| h00: h = 4n+2 (4₁ forbids) | 4 | 188 |

Body centring and both glide conditions are plain. In the zones a small
pedestal remains in classes that centring already forbids (9–10 per node),
so the glide-forbidden classes at 17–20 per node are at that level. The
axial condition rests on two to four nodes; the 4% found in h = 4n+2 is
significant against counting noise and has not been explained.

## Not done

- The rest of the space group: other axes, glide planes, and a proper
  significance test for the screw (the pedestal above is unexplained).
- A crystal whose point group is lower than its lattice symmetry, more than
  one lattice, or a centred lattice.
- Instruments without event times. There the same transform with |q|
  integrated out (the lattice ladder's zone-axis table) gives the cell
  shape but not its scale.
- The choice of thresholds (background cut, peak dominance, acceptance
  level) is by hand.
