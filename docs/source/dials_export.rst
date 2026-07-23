=============
DIALS export
=============

Every ``subhkl`` command accepts a ``--dials`` (alias ``-dials``) flag. When
set, the command writes its output as a DIALS
`ExperimentList <https://dials.github.io/documentation/data_files.html>`_
(``.expt``, a JSON file) and a
`reflection_table <https://dials.github.io/documentation/data_files.html>`_
(``.refl``, a msgpack file), in addition to its normal HDF5/MTZ output. This
lets subhkl results be loaded, inspected, and compared directly against the
DIALS suite (``dials.image_viewer``, ``dials.reflection_viewer``,
``dials.show``, ``dials.compare_orientation_matrices``, ``dials.scale``, ...).

By default the ``.expt``/``.refl`` pair is named after the command's primary
output file with the extension replaced (``indexed.h5`` →
``indexed.expt`` + ``indexed.refl``). Use ``--dials-prefix STEM`` to choose a
different stem.

.. note::

   The export uses the real ``dxtbx``/``dials`` API, so the DIALS/cctbx toolkit
   must be importable. It is distributed through conda
   (``conda install -c conda-forge dials``). If DIALS is not installed, the
   command still produces its native output and reports a clear error for the
   ``--dials`` step only.

How subhkl maps onto the DIALS model
------------------------------------

subhkl and DIALS share a laboratory frame (``Z`` along the incident beam, ``Y``
up). Because the beam, detector, crystal, and goniometer are all copied out of
that single subhkl frame, the exported experiment is internally consistent and
reproduces subhkl's physics, ``s1 = s0 + S · U · B · h``.

- **Beam** — a ``PolychromaticBeam`` carrying only the incident direction
  (from ``beam/ki_vec``), since subhkl is Laue; the true per-reflection
  wavelength lives in the reflection table's ``wavelength`` column. On dxtbx
  builds without ``PolychromaticBeam`` the export falls back to a monochromatic
  beam at the mean wavelength. Neutron probe is set when supported.
- **Detector** — one dxtbx ``Panel`` per subhkl bank. Geometry comes from
  ``beamlines.json``, but a refined ``detector_calibration`` group in the file
  (written by ``indexer --refine-detector``) overrides the nominal metrology per
  bank, so the exported detector matches the geometry the program actually used.
  Lengths convert from metres to millimetres; the fast axis is columns
  (``uhat``), the slow axis is rows (``vhat``). Instruments built from flat
  panels arranged around the sample (e.g. CG4D, MANDI) map exactly, one dxtbx
  ``Panel`` per bank. dxtbx panels are planar, so if a config uses subhkl's
  cylindrical ``curved`` panel type it is approximated by its tangent plane;
  peak identity is preserved through the stored pixel coordinates.
- **Crystal** — built from ``sample/U`` and ``sample/B``. subhkl's ``B`` uses
  the crystallographic Busing-Levy convention (no ``2π``), so ``A = U · B``
  maps directly onto the dxtbx setting matrix. The space group comes from
  ``sample/space_group``.
- **Goniometer / frames** — each subhkl frame (``peaks/run_index``) becomes a
  separate DIALS ``Experiment`` sharing the same beam/detector/crystal, with the
  frame's rotation baked into the goniometer *setting rotation*. When that
  rotation is reconstructed from raw ``goniometer/axes``/``goniometer/angles``
  (e.g. peak_predictor output, which stores no ``goniometer/R``), the per-axis
  ``goniometer/offsets`` zero-points are folded in, matching the geometry subhkl
  used; a stored ``goniometer/R`` already includes them. Reflections carry
  ``id`` (experiment index) and ``panel`` (bank index).
- **Sample position** — subhkl mounts the sample off the rotation origin, at
  ``s_lab = R_frame · translation`` (``goniometer/translations``), and casts the
  diffracted ray from there. DIALS assumes the sample at the origin, so each
  frame's detector is translated by ``-s_lab`` — a pure translation that leaves
  the beam, crystal, and rotation unchanged and reproduces subhkl's geometry
  exactly. The ``s1`` wavevector is likewise measured from ``xyz - s_lab``. When
  there is no offset the detector is shared across frames.
- **Images / ImageSet** — when the pixel data is available (the ``images`` stack
  from ``reduce``/``merge``, or the input stack a downstream program read), the
  export writes a small self-describing container (``<stem>_images.h5``) and
  attaches an ``ImageSet`` per frame, so the ``.expt`` opens in
  ``dials.image_viewer`` and friends. A bare image stack therefore exports one
  experiment per goniometer setting. If the pixel data or the ImageSet API is
  unavailable, the export falls back to geometry-only (load with
  ``check_format=False``).

Opening the ImageSet requires the container reader to be registered with dxtbx.
subhkl declares it through the ``dxtbx.format`` entry point
(``FormatSubhkl``), so ``pip install -e .`` in the DIALS environment makes it
discoverable. If dxtbx does not pick it up, re-run ``dxtbx.print_format_instances``
or reinstall so the entry point is registered.

Verifying the export in a DIALS environment
-------------------------------------------

The round-trip tests in ``tests/io/test_dials_export.py`` are guarded by
``pytest.importorskip("dxtbx")`` and only run where the DIALS toolkit is
importable, so they are skipped by the ordinary (PyPI/uv) unit run and exercised
by the dedicated ``DIALS export`` CI workflow. To reproduce that environment
locally:

.. code-block:: bash

   # 1. A conda env with the DIALS/dxtbx/cctbx toolkit
   mamba create -y -n ld-subhkl -c conda-forge dials python=3.12
   mamba activate ld-subhkl

   # 2. Install subhkl so the dxtbx.format entry point registers FormatSubhkl.
   #    --no-deps keeps the DIALS stack intact (subhkl's PyPI deps are not needed
   #    to run these tests).
   pip install -e . --no-deps
   pip install pytest pytest-dependency

   # 3. Confirm the reader is discoverable, then run the round-trips.
   python -c "from dxtbx.format.Registry import get_format_class_for; \
       print(get_format_class_for('FormatSubhkl'))"
   MESOLITE_MAX_FILES=0 pytest -v tests/io/test_dials_export.py

If ``get_format_class_for('FormatSubhkl')`` raises ``KeyError``, the entry point
was not picked up — reinstall subhkl in the active env (an editable install must
be re-run after changing the ``dxtbx.format`` entry in ``pyproject.toml``).

Closest DIALS CLI parallel per program
---------------------------------------

+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| subhkl command      | What the ``--dials`` output contains                          | Closest DIALS CLI program(s)                                |
+=====================+===============================================================+=============================================================+
| ``reduce``          | One experiment per goniometer setting, each with an ImageSet  | ``dials.import`` (build experiments from images)            |
|                     | (beam, detector, goniometer); empty reflection table.         |                                                             |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| ``merge_images``    | One experiment per frame of the merged stack, each with an    | ``dials.import`` / ``dials.combine_experiments``            |
|                     | ImageSet; unit cell metadata; empty reflection table.         |                                                             |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| ``finder``          | Observed peaks: ``xyzobs.px.value``, ``intensity.sum``,       | ``dials.find_spots`` (strong.refl)                          |
|                     | ``panel``, per-frame experiments.                             |                                                             |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| ``zone_axis_search``| Experiments with the oriented crystal (``U``/``B``);          | ``dials.index`` orientation search / ``dials.search_beam``  |
|                     | empty reflection table.                                       | for the initial orientation.                                |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| ``indexer``         | Indexed reflections: ``miller_index``, ``wavelength``,        | ``dials.index`` (indexed.expt + indexed.refl);              |
|                     | ``xyzobs.px``, ``intensity`` + crystal model per frame.       | ``dials.refine`` for the geometry refinement it also does.  |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| ``peak_predictor``  | Predicted reflections: ``xyzcal.px``, ``miller_index``,       | ``dials.predict``                                           |
|                     | ``wavelength``, ``predicted`` flag.                           |                                                             |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| ``integrator``      | Integrated reflections: ``intensity.sum``, ``miller_index``,  | ``dials.integrate`` (integrated.refl)                       |
|                     | ``wavelength``, ``s1``.                                        |                                                             |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| ``rbf_integrator``  | Integrated reflections (RBF profile fitting).                 | ``dials.integrate`` (profile-fitting path)                  |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| ``mtz_exporter``    | The indexed/integrated reflections behind the MTZ, as         | ``dials.export format=mtz`` (the inverse: DIALS refl → MTZ) |
|                     | ``.expt``/``.refl``.                                          |                                                             |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+
| ``metrics``         | The reflections of the evaluated file (default stem           | ``dials.rl_png`` / ``dials.report`` (indexing quality);     |
|                     | ``metrics``); residuals remain in ``--csv``.                  | residual analysis in ``dials.refine`` logs.                 |
+---------------------+---------------------------------------------------------------+-------------------------------------------------------------+

Example
-------

.. code-block:: bash

   # Index and also emit indexed.expt + indexed.refl
   subhkl indexer peaks.h5 indexed.h5 --instrument CG4D --nexus raw.nxs.h5 --dials

   # Inspect with DIALS
   dials.show indexed.expt indexed.refl

   # Custom output stem
   subhkl integrator merged.h5 CG4D predicted.h5 integrated.h5 \
       -dials --dials-prefix subhkl_integrated
