"""Static-structure masks: what the detector always sees is not a peak.

Beam-stop shadows, illumination boundaries and instrument-body glow are fixed
in the detector frame, while a Bragg reflection survives at most a frame or
two of sample rotation.  A per-bank low quantile across enough frames
(exposure-normalised) therefore contains the artifacts and none of the crystal -- the frames do not even have
to come from the same sample, only from the same instrument configuration,
which makes the mask an instrument calibration product rather than a
per-dataset one.

Two artifact classes, measured on CG4D l1-mbl (both recurring across ten
runs while every Bragg peak moved):

* an illumination boundary -- the static background fell 2.2 -> 0.6 counts
  over ~25 columns, a gradient 8x the panel's typical column-to-column
  variation.  The smooth background model cannot follow a step, so the
  residual ridge along it is tiled with false atoms.  The boundary sat
  ~190 px inside the panel, out of reach of any border erosion.
* a diffuse static glow at ~2x the ambient rate, carrying dozens of weak
  spurious detections whose positions jitter from run to run -- invisible
  to any peak-level recurrence veto, caught only by a pixel mask.

The mask marks static structure on a *band-pass* of the static map --
scales between the atom footprint and the background window -- because that
is precisely the structure the background model cannot follow and the finder
mistakes for peaks.  A wide smooth halo passes: the background follows it,
and the genuine reflections sitting on it must stay findable.  Steps are
caught by the band-pass gradient, textured glow by the band-pass level, and
the union is dilated so atoms merely touching structure are covered.

The file format is a reduced single-frame stack: ``images`` [n_banks, H, W]
(uint8, 1 = valid, 0 = masked) with ``bank_ids`` alongside, so every tool
that reads reduced files can display a mask, and the finder maps mask frames
onto its input by physical bank rather than by position in the file.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


def estimate_static_mask(
    frames: np.ndarray,
    *,
    smooth_sigma: float = 2.0,
    grad_nmads: float = 8.0,
    texture_factor: float = 0.15,
    wide_sigma: float = 20.0,
    edge_sigma: float = 25.0,
    dilate_px: int = 8,
    static_quantile: float = 25.0,
    grad_min_frac: float = 0.02,
    protect_disks: list | None = None,
    protect_nsigmas: float = 3.5,
) -> np.ndarray:
    """Valid-pixel mask (1 = usable) for one bank from its frame stack.

    Two robustness properties are load-bearing:

    * Frames are normalised to unit mean rate before stacking, so the same
      static feature at a different exposure (or beam current) produces the
      same mask, and frames of mixed exposures may be pooled.  Both criteria
      below are then ratios on a relative rate map rather than counts.
    * The criteria run on a *union of quantile maps* across frames: the low
      quantile (``static_quantile``, default p25) holds what is static in
      nearly every frame, and p75 holds what is static in at least a
      quarter of them -- an illumination boundary that shifts between
      manually-set orientation classes lives only in the latter.  Each map
      carries full multi-frame SNR (per-frame detection with an
      across-frame vote was measured and rejected: single frames at 1-2
      counts/px sit at their own noise floor, and recurring noise votes
      like a subset artifact).  Reflections recurring in a minority of
      frames leak into the high quantile by the same token; certificate
      protection is what makes that affordable, and the measured
      dense-bank cost is +0.5 points.

    ``grad_nmads`` is the boundary criterion: pixels where the static map's
    gradient magnitude exceeds this many MADs of the panel-wide gradient are
    structure, not statistics (the l1-mbl illumination edge measures ~8 MADs
    of margin even at threshold 8).  ``texture_factor`` is the band-pass level
    criterion, relative to the panel's ambient rate; ``wide_sigma`` sets the
    long end of the *level* band (the glow-texture scale); ``edge_sigma``
    sets the long end of the *contrast* band -- the scale the finder's own
    background model can actually follow (its window is max(15,
    5 * max_sigma) px).  They differ because their failure modes differ: a
    longer level band lets plateaus inflate the MAD floor and re-admits
    dense-diffraction texture, while a shorter contrast band opens a blind
    gap -- the l1-mbl backward-panel illumination boundaries (10-90% widths
    ~40 px) vanished from a 10 px band yet still produced false atoms,
    because the finder could not follow them either; their capture
    saturates at 25.  ``dilate_px`` should cover an atom
    footprint (~2 * max_sigma) so that an atom whose tail rests on the
    structure is masked along with it.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError("need a [n_frames, H, W] stack with at least 2 frames")

    scales = frames.mean(axis=(1, 2), keepdims=True)
    rates = frames / np.maximum(scales, 1e-6)
    # Smooth each frame *before* the quantile: at counting rates near one
    # photon per pixel a low quantile of raw integers is zero on both sides
    # of any structure, and the structure vanishes with it.  Smoothing first
    # turns each frame into a local rate estimate, whose low quantile keeps
    # what every frame shows.
    # method="lower": a true order statistic.  The default interpolated
    # percentile blends adjacent order statistics, so a bright reflection
    # present in only some frames leaks a *fraction* of itself into the
    # static map -- and a 400-count peak leaking ten percent still dwarfs
    # any threshold stated in units of a ~2-count ambient.  Measured on
    # l1-mbl forward banks as an exclusion ring around every true peak.
    stack = np.stack([ndimage.gaussian_filter(f, smooth_sigma) for f in rates])

    def _tail_radius(sig: float, height: float, thr: float) -> float:
        """Radius where the peak's smoothed band-pass profile falls below
        ``thr``.

        ``height`` is the peak's measured band-pass value at its own
        center -- already smoothed, already background-subtracted, in the
        same normalised units as ``thr`` -- so the only model left is the
        Gaussian tail of the smoothed profile, whose width is s_eff.
        """
        s_eff_sq = sig**2 + smooth_sigma**2
        if height > thr:
            return float(np.sqrt(2.0 * s_eff_sq * np.log(height / thr)))
        return 0.0

    # The criteria live on a *band-pass* of a static map: structure at
    # scales between the atom footprint (smooth_sigma) and the background
    # window.  What the mask must mark is structure that defeats the
    # background model -- and only that.  A wide static halo, however
    # elevated or statically steep its shoulders, is followed by the
    # windowed background estimate and is where real Bragg peaks live;
    # masking it (as an absolute-gradient or absolute-level criterion does)
    # removed genuine reflections from the centre of the l1-mbl forward
    # banks.  The halo vanishes from the band-pass; what remains is exactly
    # the peak-confusable statics.
    #
    # Two criteria, one per way a static feature defeats the model:
    #
    # *Level* (texture): elevated positive band, the diffuse-glow
    # signature.  Positive lobes only -- false atoms are positive
    # unmodelled structure, and |band| would additionally mask the negative
    # moat the wide subtraction digs around anything bright.  The threshold
    # is the larger of the effect-size criterion (texture_factor x ambient)
    # and a significance floor on the band's own noise (4 MADs, ~2.7
    # sigma): without the floor, an aggressive texture_factor would start
    # masking Poisson speckle on a photon-sparse panel.
    #
    # *Contrast* (boundary): the smoothed gradient magnitude of its own,
    # longer band (edge_sigma -- sharpness is judged against what the
    # finder's background can follow, and plateaus cost nothing there
    # since flat regions have zero gradient), then a MAD significance
    # test.  Smoothing the magnitude consolidates the flanks of any sharp
    # feature at any orientation, so an unbroken ridge masks contiguously.
    # The physical floor (gradient per pixel as a fraction of ambient)
    # guards flat panels; the band > 0 restriction keeps the wide
    # subtraction's negative moat -- whose outer slope is highly
    # significant -- from ringing every bright spot.
    def _criteria(sm: np.ndarray) -> tuple[np.ndarray, float, float]:
        """(un-dilated bad set, ambient, texture threshold) of one map."""
        ambient = max(float(np.median(sm)), 1e-3)
        band = sm - ndimage.gaussian_filter(sm, wide_sigma)
        band_mad = np.median(np.abs(band - np.median(band))) + 1e-9
        thr = max(texture_factor * ambient, 4.0 * band_mad)
        texture = band > thr
        band_e = sm - ndimage.gaussian_filter(sm, edge_sigma)
        gy, gx = np.gradient(band_e)
        edge = ndimage.gaussian_filter(np.hypot(gy, gx), smooth_sigma)
        edge_med = np.median(edge)
        edge_mad = np.median(np.abs(edge - edge_med)) + 1e-9
        boundary = (
            (edge - edge_med > grad_nmads * edge_mad)
            & (edge > grad_min_frac * ambient)
            & (band_e > 0.0)
        )
        return boundary | texture, ambient, thr

    # An artifact need not be static in *every* frame: an illumination
    # boundary that shifts between manually-set orientation classes is
    # present in a quarter or half of the frames and invisible to the low
    # quantile by construction.  The criteria therefore run on a union of
    # quantile maps -- p25 ("static in nearly all frames") and p75
    # ("static in at least a quarter of frames") -- each at full
    # multi-frame SNR.  Per-frame edge detection with an across-frame vote
    # was measured and rejected: at 1-2 counts/px a single frame's edge
    # map sits at its own noise floor, and recurring noise votes at the
    # same rate as subset artifacts.  The high quantile was historically
    # unusable because moving reflections leak into it; certificate
    # protection is what makes it safe now -- and the measured dense-bank
    # cost is +0.5 points.  The protection radius is computed against the
    # smallest of the maps' texture thresholds (the largest radius).
    quantiles = sorted({float(static_quantile), 75.0})
    bad = np.zeros(stack.shape[1:], dtype=bool)
    texture_threshold = np.inf
    band_maps = []
    for q in quantiles:
        sm_q = np.percentile(stack, q, axis=0, method="lower")
        bad_q, _, thr_q = _criteria(sm_q)
        bad |= bad_q
        texture_threshold = min(texture_threshold, thr_q)
        # Kept for the certificates below: a peak's protection radius is
        # read off the very maps the criteria threshold.
        band_maps.append(sm_q - ndimage.gaussian_filter(sm_q, wide_sigma))

    # Certificates exist to protect peaks -- nothing else.  The mask may be
    # as liberal as it likes about admitting static structure, because the
    # only harm masking can do is eat a genuine reflection, and that is
    # exactly what protection covers: an accepted certificate subtracts a
    # disk over precisely where its own smoothed tail exceeds the texture
    # threshold (with a 2*smooth_sigma margin for the frame-to-frame wobble
    # the fitted sigma cannot know), plus the dilation the bad set receives,
    # so a certified footprint never reaches the final mask however bright.
    # The peak's height is *measured*, not modelled: the finder reports no
    # amplitude (that was the hull-era intensity, retired), and what the
    # radius needs is the peak's leak into the quantile maps -- so it is
    # read directly off the same band-pass maps the criteria threshold, at
    # the certificate's own center (a 3x3 max absorbs sub-pixel centers).
    # This also prices quasi-static peaks correctly by construction: an
    # order statistic carries a minority-frame peak at full height or not
    # at all, which no duty-cycle model of a fitted flux could reproduce.
    # Certificates never touch the evidence: an earlier design also cleared
    # the certified footprints out of the frame stack before the quantile,
    # and every failure mode of this estimator's history -- exclusion rings,
    # crater rims, edges dissolving under chains of false certificates --
    # came from that clearing.  Protection is sufficient and its worst case
    # (a false certificate) is one bounded, visible disk.
    #
    # The gate: a certificate is a statement about a *peak*, not about the
    # structure it sits on.  A detection whose underlying static component
    # extends beyond 4x its protected radius is refused: its metrics may be
    # clean -- measured on l1-mbl, detections on the illumination edges
    # carry deviance 20+ and residual/DoF < 2, indistinguishable from faint
    # genuine peaks -- but honouring them would open a chain of disks along
    # the edge.  Components are measured after a morphological closing at
    # the map's own correlation length (2*smooth_sigma): a noisy edge is a
    # chain of fragments -- each innocently compact -- broken at the noise
    # scale, so closing fuses one structure back together, while real
    # separations (adjacent Laue-arc reflections, ~20-40 px apart) stay
    # separate.  The full dilation would be wrong here: at dilate_px=8 it
    # bridged speckle into panel-spanning blobs and the gate refused the
    # forward-bank arcs wholesale.  Measured on l1-mbl the split is bimodal:
    # compact peak leaks reach ~0.8x of the gate at the 90th percentile,
    # edge components sit at 4-12x with extents of 200-500 px.
    accepted: list[tuple[float, float, float, float]] = []
    if protect_disks:
        closed = ndimage.binary_closing(
            bad, iterations=int(np.ceil(2.0 * smooth_sigma))
        )
        lab, n_comp = ndimage.label(closed)
        extents = np.zeros(n_comp + 1)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is not None:
                extents[i] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        H, W = lab.shape
        for r0, c0, sig in protect_disks:
            ri, ci = int(round(r0)), int(round(c0))
            height = max(
                float(
                    band[
                        max(0, ri - 1) : min(H, ri + 2),
                        max(0, ci - 1) : min(W, ci + 2),
                    ].max()
                )
                for band in band_maps
            )
            rad = max(
                protect_nsigmas * sig + 2.0 * smooth_sigma,
                _tail_radius(sig, height, texture_threshold) + 2.0 * smooth_sigma,
            )
            core = int(np.ceil(protect_nsigmas * sig))
            lo_r, hi_r = max(0, ri - core), min(H, ri + core + 1)
            lo_c, hi_c = max(0, ci - core), min(W, ci + core + 1)
            d2 = (np.arange(lo_r, hi_r)[:, None] - r0) ** 2 + (
                np.arange(lo_c, hi_c)[None, :] - c0
            ) ** 2
            comps = np.unique(
                lab[lo_r:hi_r, lo_c:hi_c][d2 <= (protect_nsigmas * sig) ** 2]
            )
            if any(extents[k] > 4.0 * rad for k in comps if k > 0):
                continue
            accepted.append((r0, c0, sig, rad))

    if dilate_px > 0:
        bad = ndimage.binary_dilation(bad, iterations=int(dilate_px))

    # Protection is subtracted *after* dilation so nothing grows back in.
    if accepted:
        rr, cc = np.mgrid[0 : bad.shape[0], 0 : bad.shape[1]]
        protected = np.zeros_like(bad)
        for r0, c0, _sig, rad in accepted:
            protected |= (rr - r0) ** 2 + (cc - c0) ** 2 <= (rad + dilate_px) ** 2
        bad &= ~protected

    return (~bad).astype(np.uint8)


def _read_frames_by_bank(
    paths: list[Path],
) -> tuple[dict[int, list[np.ndarray]], dict[int, int]]:
    """Frames grouped by physical bank, one per distinct orientation.

    The estimator's entire premise is that the sample moves between frames.
    Feed it the same goniometer orientation twice -- the same file listed
    again, or a re-exposure at identical angles -- and the true signal is
    static by construction, so it would be promoted into the mask.  Frames
    are therefore deduplicated per bank: by goniometer angles when the file
    carries them (which also catches re-exposures with fresh counting
    noise), by content hash otherwise (which catches literal duplicates).
    The dropped counts are returned so the caller can say so.
    """
    by_bank: dict[int, list[np.ndarray]] = {}
    provenance: dict[int, list[tuple[int, int]]] = {}
    dropped: Counter = Counter()
    hashes: set = set()
    signatures: dict[int, list[tuple]] = {}
    for file_index, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            images = f["images"]
            bank_ids = np.asarray(f["bank_ids"][()]).astype(int)
            if len(bank_ids) != images.shape[0]:
                raise ValueError(
                    f"{path}: bank_ids ({len(bank_ids)}) does not match "
                    f"images ({images.shape[0]})"
                )
            angles = None
            if (
                "goniometer/angles" in f
                and f["goniometer/angles"].shape[0] == images.shape[0]
            ):
                angles = np.asarray(f["goniometer/angles"][()], dtype=float)
            for i, bank in enumerate(bank_ids):
                frame = np.asarray(images[i], dtype=np.float32)
                digest = (int(bank), hashlib.sha1(frame.tobytes()).hexdigest())
                if digest in hashes:
                    dropped[int(bank)] += 1
                    continue
                ang = None if angles is None else tuple(np.round(angles[i], 4))
                sig = _scene_signature(frame)
                bank_sigs = signatures.setdefault(int(bank), [])
                if any(
                    (ang is None or prev_ang is None or ang == prev_ang)
                    and _same_scene(sig, prev_sig)
                    for prev_ang, prev_sig in bank_sigs
                ):
                    dropped[int(bank)] += 1
                    continue
                hashes.add(digest)
                bank_sigs.append((ang, sig))
                by_bank.setdefault(int(bank), []).append(frame)
                provenance.setdefault(int(bank), []).append((file_index, i))
    return by_bank, dict(dropped), provenance


def _scene_signature(frame: np.ndarray, block: int = 8) -> tuple:
    """Exposure-normalised block means, with the per-block noise scale."""
    H, W = frame.shape
    h, w = H // block, W // block
    scale = max(float(frame.mean()), 1e-6)
    rate = frame[: h * block, : w * block] / scale
    blocks = rate.reshape(h, block, w, block).mean(axis=(1, 3))
    # Poisson sd of a block mean of the *normalised* rate.
    noise = np.sqrt(np.maximum(blocks, 1e-3) / (block * block * scale))
    return blocks, noise


def _same_scene(sig_a: tuple, sig_b: tuple) -> bool:
    """Whether two frames are consistent with re-exposures of one scene.

    Strict on purpose: the cost of a false "duplicate" is silently thrown
    away data, so only agreement everywhere within counting noise counts.
    The max of ~4k standardised block differences between true re-exposures
    sits near 4 sigma; a moved crystal shifts bright peaks by many blocks
    and lights up tens of sigma.
    """
    a, na = sig_a
    b, nb = sig_b
    if a.shape != b.shape:
        return False
    z = np.abs(a - b) / np.sqrt(na**2 + nb**2 + 1e-12)
    return float(z.max()) < 6.0


def _first_instrument(inputs: list) -> str | None:
    """The instrument the first input that knows one records."""
    for path in inputs:
        with h5py.File(path, "r") as f_in:
            recorded = f_in.attrs.get("instrument")
        if isinstance(recorded, bytes):
            recorded = recorded.decode("utf-8")
        if recorded:
            return recorded
    return None


def build_summed_file(inputs: list[str | Path], output: str | Path) -> dict:
    """One frame per physical bank: the sum of its deduplicated frames.

    The companion input to the exoneration: a finder run on this stack sees
    the *pooled* evidence, exactly as the static-map quantile does.  Per-peak
    deviance is additive across frames, so a quasi-static reflection sitting
    just below any single frame's admission level compounds to certification
    here (~n_frames-fold in deviance), and the pooled fit's width knows the
    frame-to-frame wobble the single-frame fits cannot.  Feed the resulting
    peaks file to ``build_mask_file`` as ``pooled_peaks``.

    Frames pass through the same per-bank deduplication as the mask
    estimator, so the pooled statistics describe exactly the evidence the
    mask is built from.  The structural metadata a finder needs
    (instrument/wavelength, sample, goniometer axes) is copied from the
    first input; per-frame goniometer angles are meaningless for a sum and
    are written as zeros, so this file is for peak *metrics* only -- never
    index from it.
    """
    paths = [Path(p) for p in inputs]
    by_bank, duplicates, _ = _read_frames_by_bank(paths)
    if not by_bank:
        raise ValueError("no frames found in the inputs")
    banks = sorted(by_bank)
    sums = np.stack([np.sum(by_bank[b], axis=0) for b in banks])
    n_frames = [len(by_bank[b]) for b in banks]

    instrument = _first_instrument(paths)
    with h5py.File(output, "w") as f:
        f.create_dataset("images", data=sums, compression="gzip", compression_opts=4)
        f["bank_ids"] = np.asarray(banks, dtype=np.int64)
        f.attrs["kind"] = "summed-frames"
        f.attrs["inputs"] = [str(p) for p in paths]
        f.attrs["n_frames"] = np.asarray(n_frames, dtype=np.int64)
        if instrument:
            f.attrs["instrument"] = instrument
        with h5py.File(paths[0], "r") as f_in:
            for key in (
                "instrument/wavelength",
                "sample",
                "goniometer/axes",
                "goniometer/names",
            ):
                if key in f_in:
                    group, _, leaf = key.rpartition("/")
                    dest = f.require_group(group) if group else f
                    f_in.copy(key, dest, name=leaf)
            if "goniometer/angles" in f_in:
                width = f_in["goniometer/angles"].shape[1:]
                f["goniometer/angles"] = np.zeros((len(banks), *width))
    return {
        "banks": banks,
        "n_frames": dict(zip(banks, n_frames)),
        "duplicates_dropped": duplicates,
        "output": str(output),
    }


# chi^2_4 at 95%: the admission level for a four-parameter atom.  The
# exoneration bar sits here, not above it: the finder's reported peaks
# already passed a *calibrated* panel-wide false-alarm control (E[FP] per
# bank), so a higher deviance bar re-litigates evidence the calibration
# settled -- and manufactures a gap.  Masking a quasi-static peak takes
# only ~texture_factor x ambient in the smoothed p25 (~15-20 recorded
# counts), while a deviance-25 certificate takes ~3x that; measured on
# l1-mbl, 340 detections fell in between -- every one with residual
# deviance per DoF < 2 (median 1.11), deviance 20-23, median flux 133 --
# faint genuine reflections masked with no route to exoneration.  84% sat
# in compact blobs the size of their own dilated footprint, not on any
# extended static structure.
CHI2_4_P95 = 9.488


def _confident_peaks_by_frame(
    peaks_path: str | Path,
    deviance_min: float,
    residual_max: float,
) -> dict[int, list[tuple[float, float, float]]]:
    """Per image index: (row, col, sigma) of certified peaks.

    The certificate is the finder's own fit statistics: evidence at least
    at the chi^2_4 admission level (the finder's calibrated false-alarm
    control already governs above it -- see CHI2_4_P95) and a shape the
    atom family explains (residual deviance per DoF near one).  An
    artifact fails one or both and earns no exoneration.  The geometry of
    the protected region is the estimator's business -- it measures the
    peak's height off its own quantile maps (the finder reports no
    amplitude: that was the hull-era intensity, retired with it).
    """
    out: dict[int, list[tuple[float, float, float]]] = {}
    with h5py.File(peaks_path, "r") as f:
        idx = np.asarray(f["peaks/image_index"][()]).astype(int)
        r = np.asarray(f["peaks/pixel_r"][()], dtype=float)
        c = np.asarray(f["peaks/pixel_c"][()], dtype=float)
        sigma = np.asarray(f["peaks/sigma"][()], dtype=float)
        dev = np.asarray(f["peaks/deviance"][()], dtype=float)
        res = np.asarray(f["peaks/residual_deviance"][()], dtype=float)
    confident = (dev > deviance_min) & (res < residual_max)
    for i in np.nonzero(confident)[0]:
        s_i = max(float(sigma[i]), 1.0)
        out.setdefault(int(idx[i]), []).append((float(r[i]), float(c[i]), s_i))
    return out


def _confident_pooled_peaks(
    peaks_path: str | Path,
    deviance_min: float,
    residual_max: float,
    n_by_bank: dict[int, int],
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Per physical bank: certified peaks of a finder run on summed frames.

    Per-frame certification cannot see what the mask sees: the static map
    pools ~n frames of evidence, so a quasi-static reflection just below
    every single frame's admission level is masked with no route to
    exoneration.  A finder run on the per-bank *sum* (see
    ``build_summed_file``) closes the asymmetry -- deviance is additive, so
    the same peak compounds ~n-fold there -- and its certified detections
    are exonerated in every frame of their bank.  No amplitude travels
    with the certificate: the estimator measures each peak's height off
    its own quantile maps, which prices static and quasi-static peaks
    alike without any frame-count rescaling.

    The evidence bar is the same as per-frame -- pooling the evidence is the
    point -- but the *shape* bar must transfer.  Residual deviance per DoF
    is one for a correct model regardless of counts, while its mismatch
    component grows linearly with them: E[res/DoF] ~ 1 + eps^2 * counts.
    Summing n frames multiplies the counts by n, so a bright genuine peak
    whose per-frame residual is a comfortable 1.1 blows past a fixed bar of
    2 on the pooled fit -- goodness of fit gets *stricter* with statistics,
    and the pooled bootstrap was stripping exactly the strong reflections
    it was meant to keep.  The bar therefore scales as
    1 + (residual_max - 1) * n_bank, which preserves the discrimination:
    an artifact's per-frame ~3.5 scales to ~1 + 2.5 n, far above it.

    The bank comes from the finder's own per-peak ``bank`` dataset, so the
    pooled file needs no companion to interpret.
    """
    out: dict[int, list[tuple[float, float, float]]] = {}
    with h5py.File(peaks_path, "r") as f:
        if "bank" not in f:
            raise ValueError(
                f"{peaks_path}: pooled peaks need a per-peak 'bank' dataset "
                "(a finder output has one)"
            )
        bank = np.asarray(f["bank"][()]).astype(int)
        r = np.asarray(f["peaks/pixel_r"][()], dtype=float)
        c = np.asarray(f["peaks/pixel_c"][()], dtype=float)
        sigma = np.asarray(f["peaks/sigma"][()], dtype=float)
        dev = np.asarray(f["peaks/deviance"][()], dtype=float)
        res = np.asarray(f["peaks/residual_deviance"][()], dtype=float)
    n_frames = np.array([n_by_bank.get(int(b), 1) for b in bank], dtype=float)
    residual_bar = 1.0 + (residual_max - 1.0) * np.maximum(n_frames, 1.0)
    confident = (dev > deviance_min) & (res < residual_bar)
    for i in np.nonzero(confident)[0]:
        s_i = max(float(sigma[i]), 1.0)
        out.setdefault(int(bank[i]), []).append((float(r[i]), float(c[i]), s_i))
    return out


def build_mask_file(
    inputs: list[str | Path],
    output: str | Path,
    *,
    peaks: list[str | Path] | None = None,
    pooled_peaks: str | Path | None = None,
    peak_deviance_min: float = CHI2_4_P95,
    peak_residual_max: float = 2.0,
    peak_clear_nsigmas: float = 3.5,
    min_frames: int = 5,
    smooth_sigma: float = 2.0,
    grad_nmads: float = 8.0,
    texture_factor: float = 0.15,
    wide_sigma: float = 20.0,
    edge_sigma: float = 25.0,
    dilate_px: int = 8,
    static_quantile: float = 25.0,
    grad_min_frac: float = 0.02,
) -> dict:
    """Estimate one mask per physical bank across every input file.

    Inputs are reduced/merged stacks (``images`` + ``bank_ids``); they may
    come from different samples.  The cleanest input is a control experiment
    without a sample, where everything is static and nothing needs rescuing.
    When only sample scans exist, certificates *protect* genuine peaks --
    and do nothing else: the mask is deliberately liberal about admitting
    static structure, because the only harm masking can do is eat a
    reflection, and a certified footprint is subtracted from the final mask
    (see ``estimate_static_mask``).  ``peaks`` (finder outputs from an
    unmasked run, paired with ``inputs`` by order) certifies detections by
    their per-frame fit metrics; ``pooled_peaks`` (a finder output from the
    per-bank *summed* stack, see ``build_summed_file``) extends the rescue
    to reflections too faint for any single frame's certificate, whose
    significance compounds across frames there.  A quasi-static reflection
    -- Laue zones from a manually oriented crystal -- is thereby findable
    however many frames it persists through.  A bank with fewer than
    ``min_frames`` frames gets a fully valid mask -- stated in the summary
    rather than silently guessed from statistics too thin to tell a peak
    from a shadow.
    """
    if peaks is not None and len(peaks) != len(inputs):
        raise ValueError(
            f"--peaks must pair with the inputs by order: got {len(peaks)} "
            f"peaks file(s) for {len(inputs)} input(s)"
        )
    by_bank, duplicates, provenance = _read_frames_by_bank([Path(p) for p in inputs])
    if not by_bank:
        raise ValueError("no frames found in the inputs")

    confident: dict[int, dict[int, list]] = {}
    n_exonerated = 0
    if peaks is not None:
        for file_index, peaks_path in enumerate(peaks):
            confident[file_index] = _confident_peaks_by_frame(
                peaks_path,
                peak_deviance_min,
                peak_residual_max,
            )
    pooled: dict[int, list] = {}
    n_exonerated_pooled = 0
    if pooled_peaks is not None:
        pooled = _confident_pooled_peaks(
            pooled_peaks,
            peak_deviance_min,
            peak_residual_max,
            {b: len(fr) for b, fr in by_bank.items()},
        )

    banks = sorted(by_bank)
    masks, thin = [], []
    for bank in banks:
        frames = np.stack(by_bank[bank])
        if frames.shape[0] < min_frames:
            masks.append(np.ones(frames.shape[1:], dtype=np.uint8))
            thin.append(bank)
        else:
            protect = list(pooled.get(bank, []))
            n_exonerated_pooled += len(protect)
            for file_index, image_index in provenance[bank]:
                disks = confident.get(file_index, {}).get(image_index, [])
                n_exonerated += len(disks)
                protect.extend(disks)
            masks.append(
                estimate_static_mask(
                    frames,
                    protect_disks=protect or None,
                    protect_nsigmas=peak_clear_nsigmas,
                    smooth_sigma=smooth_sigma,
                    grad_nmads=grad_nmads,
                    texture_factor=texture_factor,
                    wide_sigma=wide_sigma,
                    edge_sigma=edge_sigma,
                    dilate_px=dilate_px,
                    static_quantile=static_quantile,
                    grad_min_frac=grad_min_frac,
                )
            )

    instrument = _first_instrument(inputs)

    stack = np.stack(masks)
    with h5py.File(output, "w") as f:
        # Mostly-ones uint8 compresses ~100x, and the benchmark harness ships
        # this file in its CI artifact.
        f.create_dataset("images", data=stack, compression="gzip", compression_opts=4)
        f["bank_ids"] = np.asarray(banks, dtype=np.int64)
        f.attrs["kind"] = "static-mask"
        if instrument:
            f.attrs["instrument"] = instrument
        f.attrs["inputs"] = [str(p) for p in inputs]
        f.attrs["min_frames"] = min_frames
        f.attrs["smooth_sigma"] = smooth_sigma
        f.attrs["grad_nmads"] = grad_nmads
        f.attrs["texture_factor"] = texture_factor
        f.attrs["wide_sigma"] = wide_sigma
        f.attrs["edge_sigma"] = edge_sigma
        f.attrs["dilate_px"] = dilate_px
        f.attrs["static_quantile"] = static_quantile
        f.attrs["grad_min_frac"] = grad_min_frac
        # The exoneration provenance: without it, a mask file cannot answer
        # "were peaks passed, and at what bar?" -- the first question asked
        # when a peak turns up masked.
        f.attrs["peaks"] = [str(p) for p in peaks] if peaks else []
        f.attrs["pooled_peaks"] = str(pooled_peaks) if pooled_peaks else ""
        f.attrs["peak_deviance_min"] = peak_deviance_min
        f.attrs["peak_residual_max"] = peak_residual_max
        f.attrs["peak_clear_nsigmas"] = peak_clear_nsigmas
        f.attrs["n_exonerated"] = n_exonerated
        f.attrs["n_exonerated_pooled"] = n_exonerated_pooled

    masked_frac = 1.0 - stack.mean()
    return {
        "banks": banks,
        "n_frames": {b: len(by_bank[b]) for b in banks},
        "thin_banks": thin,
        "n_exonerated": n_exonerated,
        "n_exonerated_pooled": n_exonerated_pooled,
        "duplicates_dropped": duplicates,
        "masked_fraction": float(masked_frac),
        "output": str(output),
    }


def load_mask_for_banks(
    path: str | Path, bank_ids: list[int], shape: tuple[int, int]
) -> np.ndarray:
    """Per-image validity stack for ``bank_ids``, mapped by physical bank.

    A bank absent from the mask file is fully valid: masks may be built from
    a subset of the instrument, and refusing to run would make the mask a
    gate rather than a filter.  A shape mismatch is an error -- it means the
    mask was built for a different reduction geometry, and applying it
    anyway would mask the wrong pixels.
    """
    with h5py.File(path, "r") as f:
        masks = np.asarray(f["images"][()], dtype=np.float32)
        file_banks = np.asarray(f["bank_ids"][()]).astype(int)
    if masks.shape[1:] != tuple(shape):
        raise ValueError(
            f"mask frames are {masks.shape[1:]}, input images are {tuple(shape)}; "
            "the mask was built for a different reduction geometry"
        )
    lookup = {int(b): masks[i] for i, b in enumerate(file_banks)}
    ones = np.ones(shape, dtype=np.float32)
    return np.stack([lookup.get(int(b), ones) for b in bank_ids])
