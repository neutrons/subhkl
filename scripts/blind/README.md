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

The blind cell is about 1% larger than the reference. The reference cell
was supplied by hand, and a 2D Laue solve cannot see its scale; the event
times independently put the scale 0.9% above it.

Timing on one GPU: about 30 s to histogram 9×10⁸ events, under 2 minutes
for step 1 in total.

## Not done

- The space group beyond the Laue class. The 6₁ screw axis would show in
  the 00l reflections, but for this long axis 184 of the 199 axial rows are
  censored as unresolvable overlaps.
- A crystal whose point group is lower than its lattice symmetry, more than
  one lattice, or a centred lattice.
- Instruments without event times. There the same transform with |q|
  integrated out (the lattice ladder's zone-axis table) gives the cell
  shape but not its scale.
- The choice of thresholds (background cut, peak dominance, acceptance
  level) is by hand.
