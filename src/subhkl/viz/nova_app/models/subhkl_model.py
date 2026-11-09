"""Pydantic configuration and nova-galaxy tool execution model for SubHKL."""

from __future__ import annotations

import os
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from subhkl.viz.rainbow_zones import (
    build_rainbow_zone_payload,
    export_static_app_html,
    render_rainbow_html,
)


class InstrumentOptions(str, Enum):
    """Supported ORNL SNS/HFIR single-crystal neutron diffractometers."""

    CG4D = "CG4D"
    MANDI = "MANDI"
    IMAGINE = "IMAGINE"
    TOPAZ = "TOPAZ"


class ProjectionOptions(str, Enum):
    """Detector projection geometry."""

    cylindrical = "cylindrical"
    spherical = "spherical"


class PipelineStageOptions(str, Enum):
    """Coarse NDIP diffraction workflow stages."""

    reduce = "reduce"
    solve = "solve"
    integrate = "integrate"
    merge = "merge"


# Tool ID mapping on NDIP (https://ndip-test.ornl.gov and local galaxy-tools fork)
NDIP_STAGE_TOOL_IDS: dict[str, list[str]] = {
    "reduce": ["imaginex_subhkl_prepare_peaks"],
    "solve": ["imaginex_subhkl_solve", "imaginex_subhkl"],
    "integrate": ["imaginex_subhkl_predict", "imaginex_subhkl_integrate"],
    "merge": ["imaginex_subhkl_metrics"],
}


class SubhklConfig(BaseModel):
    """Validated Pydantic model for SubHKL NOVA configuration and tool inputs.

    Follows the NOVA tutorial Episode 4/5/6 pattern where ``Field(title=...,
    description=...)`` metadata drives ``nova.trame.view.components.InputField``
    labels, hints, and automatic Vuetify validation state.
    """

    galaxy_url: str = Field(
        default_factory=lambda: os.getenv("GALAXY_URL", "https://ndip-test.ornl.gov"),
        title="NDIP / Galaxy URL",
        description="Remote NDIP server (https://ndip-test.ornl.gov) or local Galaxy URL",
    )
    galaxy_key: str = Field(
        default_factory=lambda: os.getenv("GALAXY_API_KEY", ""),
        title="NDIP API Key",
        description="Galaxy API key from User -> Preferences -> Manage API Key",
    )
    instrument: InstrumentOptions = Field(
        default=InstrumentOptions.CG4D,
        title="Instrument",
        description="SNS/HFIR beamline instrument definition",
    )
    space_group: str = Field(
        default="P 21 21 21",
        min_length=1,
        title="Space Group",
        description="Hermann-Mauguin space group symbol",
    )
    a: float = Field(
        default=68.7, gt=1.0, le=1000.0, title="a (Å)", description="Unit cell a (Å)"
    )
    b: float = Field(
        default=69.8, gt=1.0, le=1000.0, title="b (Å)", description="Unit cell b (Å)"
    )
    c: float = Field(
        default=156.9, gt=1.0, le=1000.0, title="c (Å)", description="Unit cell c (Å)"
    )
    alpha: float = Field(
        default=90.0,
        gt=10.0,
        lt=179.0,
        title="α (°)",
        description="Unit cell alpha (deg)",
    )
    beta: float = Field(
        default=90.0,
        gt=10.0,
        lt=179.0,
        title="β (°)",
        description="Unit cell beta (deg)",
    )
    gamma: float = Field(
        default=90.0,
        gt=10.0,
        lt=179.0,
        title="γ (°)",
        description="Unit cell gamma (deg)",
    )
    wavelength_min: float = Field(
        default=2.8,
        gt=0.05,
        le=30.0,
        title="λ min (Å)",
        description="Minimum incident wavelength (Å)",
    )
    wavelength_max: float = Field(
        default=4.5,
        gt=0.05,
        le=30.0,
        title="λ max (Å)",
        description="Maximum incident wavelength (Å)",
    )
    d_min: float = Field(
        default=2.8,
        gt=0.3,
        le=20.0,
        title="d_min (Å)",
        description="High-resolution d-spacing cutoff (Å)",
    )
    n_zones: int = Field(
        default=400,
        ge=10,
        le=2000,
        title="Zones",
        description="Number of zone axes sampled during finder-less solve",
    )
    steps: int = Field(
        default=250,
        ge=10,
        le=5000,
        title="Steps",
        description="Number of refinement steps per stage",
    )
    max_zone_index: int = Field(
        default=3,
        ge=1,
        le=4,
        title="Max Zone Order",
        description="Maximum |u|,|v|,|w| zone axis order to precompute",
    )
    peaks_file: str = Field(
        default="",
        title="Solved / Indexed Peaks (.h5)",
        description="Path to solve.h5 or indexed.h5",
    )
    images_file: str = Field(
        default="",
        title="Detector Images (.h5)",
        description="Optional path to merged.h5 detector images for intensity grounding",
    )

    @model_validator(mode="after")
    def validate_wavelength_bounds(self) -> SubhklConfig:
        if self.wavelength_max <= self.wavelength_min:
            raise ValueError("wavelength_max must be greater than wavelength_min")
        return self


class SubhklGalaxyModel:
    """Business logic for building the Diffraction Parallax payload and running NDIP tools via nova-galaxy."""

    def __init__(
        self,
        peaks_h5: str | Path | None = None,
        images_h5: str | Path | None = None,
        instrument: str = "CG4D",
    ) -> None:
        inst = (
            InstrumentOptions(instrument)
            if instrument in InstrumentOptions.__members__
            else InstrumentOptions.CG4D
        )
        default_peaks = "/home/xvg/subhkl-examples/betalac-7/solve.h5"
        default_images = "/home/xvg/subhkl-examples/betalac-7/merged.h5"
        resolved_peaks = (
            str(peaks_h5)
            if peaks_h5
            else (default_peaks if os.path.isfile(default_peaks) else "")
        )
        resolved_images = (
            str(images_h5)
            if images_h5
            else (default_images if os.path.isfile(default_images) else "")
        )
        self.data = SubhklConfig(
            instrument=inst,
            peaks_file=resolved_peaks,
            images_file=resolved_images,
        )
        self.payload: dict[str, Any] | None = None
        self.frame_index: int | None = None
        self.last_job_result: dict[str, Any] = {}
        self.refresh_payload()

    def refresh_payload(self, frame_index: int | None = None) -> dict[str, Any] | None:
        """Rebuild the Diffraction Parallax & Lattice Ladder payload from current HDF5 files."""
        if frame_index is not None:
            self.frame_index = int(frame_index)
        if not self.data.peaks_file or not os.path.isfile(self.data.peaks_file):
            return self.payload

        img_path = None
        if self.data.images_file and os.path.isfile(self.data.images_file):
            if os.path.dirname(
                os.path.abspath(self.data.images_file)
            ) == os.path.dirname(os.path.abspath(self.data.peaks_file)):
                img_path = self.data.images_file
        self.payload = build_rainbow_zone_payload(
            replay_h5=self.data.peaks_file,
            instrument=self.data.instrument.value,
            space_group=self.data.space_group,
            cell=[
                float(self.data.a),
                float(self.data.b),
                float(self.data.c),
                float(self.data.alpha),
                float(self.data.beta),
                float(self.data.gamma),
            ],
            max_zone_index=self.data.max_zone_index,
            images_h5=img_path,
            frame_index=self.frame_index,
        )
        if self.payload.get("images_h5") and os.path.isfile(
            str(self.payload["images_h5"])
        ):
            self.data.images_file = str(self.payload["images_h5"])
        # Synchronize instrument and unit cell parameters from HDF5 if available
        inst_str = str(self.payload.get("instrument") or "")
        if inst_str in InstrumentOptions.__members__:
            self.data.instrument = InstrumentOptions(inst_str)
        cell = self.payload.get("cell") or []
        if len(cell) == 6 and all(float(v) > 0 for v in cell):
            self.data.a = round(float(cell[0]), 3)
            self.data.b = round(float(cell[1]), 3)
            self.data.c = round(float(cell[2]), 3)
            self.data.alpha = round(float(cell[3]), 2)
            self.data.beta = round(float(cell[4]), 2)
            self.data.gamma = round(float(cell[5]), 2)
        sg = self.payload.get("space_group")
        if sg and sg not in ("Unknown", "P 1"):
            self.data.space_group = str(sg)
        wl_range = self.payload.get("wavelength_range") or []
        if len(wl_range) >= 2 and float(wl_range[1]) > float(wl_range[0]) > 0:
            self.data.wavelength_min = round(float(wl_range[0]), 3)
            self.data.wavelength_max = round(float(wl_range[1]), 3)
        if (
            self.payload.get("artifact_kind") in ("merged", "raw_nexus")
            and len(wl_range) >= 2
        ):
            self.data.d_min = round(0.9 * float(wl_range[0]), 2)
        elif self.payload.get("d_min") and float(self.payload["d_min"]) > 0.3:
            self.data.d_min = round(float(self.payload["d_min"]), 2)
        return self.payload

    def render_html(self) -> str:
        """Render the interactive Diffraction Parallax & Lattice Ladder HTML document."""
        if self.payload is not None:
            return render_rainbow_html(self.payload)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            export_static_app_html(tmp_path, instrument=self.data.instrument.value)
            return Path(tmp_path).read_text(encoding="utf-8")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def build_tool_parameters(self, stage: str = "solve") -> dict[str, Any]:
        """Build tool parameter dictionary matching NDIP imaginex_subhkl_*.xml inputs."""
        if stage == "solve":
            return {
                "instrument": self.data.instrument.value,
                "space_group": self.data.space_group,
                "a": self.data.a,
                "b": self.data.b,
                "c": self.data.c,
                "alpha": self.data.alpha,
                "beta": self.data.beta,
                "gamma": self.data.gamma,
                "wavelength_min": self.data.wavelength_min,
                "wavelength_max": self.data.wavelength_max,
                "d_min": self.data.d_min,
                "n_zones": self.data.n_zones,
                "steps": self.data.steps,
            }
        return {
            "instrument": self.data.instrument.value,
            "space_group": self.data.space_group,
            "wavelength_min": self.data.wavelength_min,
            "wavelength_max": self.data.wavelength_max,
            "d_min": self.data.d_min,
        }

    def run_subhkl_tool(
        self, stage: str = "solve", tool_id: str | None = None
    ) -> dict[str, Any]:
        """Execute an NDIP Galaxy tool using ``nova.galaxy`` (`Connection`, `Tool`, `Parameters`, `Dataset`).

        Follows Episode 3 of the NOVA tutorial:
        1. Connects to ``galaxy_url`` (e.g. ``https://ndip-test.ornl.gov`` or local Galaxy)
        2. Creates and persists a ``Datastore``
        3. Uploads input HDF5 file via ``Dataset.upload(data_store)`` if provided
        4. Runs ``Tool(id=...).run(data_store, params, wait=True)``
        5. Downloads resulting ``solve.h5`` / ``output`` dataset and refreshes the Parallax payload.
        """
        from nova.galaxy import Connection, Dataset, Parameters, Tool

        candidate_tools = (
            [tool_id]
            if tool_id
            else NDIP_STAGE_TOOL_IDS.get(stage, ["imaginex_subhkl_solve"])
        )
        chosen_tool_id = candidate_tools[0]

        conn = Connection(
            galaxy_url=self.data.galaxy_url, galaxy_key=self.data.galaxy_key
        )
        params = Parameters()
        for k, v in self.build_tool_parameters(stage).items():
            params.add_input(name=k, value=v)

        with conn.connect() as galaxy_connection:
            data_store = galaxy_connection.create_data_store(
                name=f"subhkl_{stage}_store"
            )
            data_store.persist()

            input_path = self.data.images_file or self.data.peaks_file
            if input_path and os.path.isfile(input_path):
                input_ds = Dataset(path=input_path, name=os.path.basename(input_path))
                uploaded = input_ds.upload(data_store)
                input_param_name = (
                    "input_images" if stage in ("reduce", "solve") else "input_peaks"
                )
                params.add_input(name=input_param_name, value=uploaded)

            tool = Tool(id=chosen_tool_id)
            outputs = tool.run(data_store, params, wait=True)
            downloaded_path = None
            if outputs is not None:
                for out_name in ("output_solved", "output_peaks", "output"):
                    try:
                        ds = outputs.get_dataset(out_name)
                        target = os.path.join(
                            tempfile.gettempdir(),
                            f"subhkl_nova_{stage}_{int(time.time())}.h5",
                        )
                        ds.download(target)
                        downloaded_path = target
                        break
                    except Exception:
                        continue

            if downloaded_path and os.path.isfile(downloaded_path):
                self.data.peaks_file = downloaded_path
                self.refresh_payload()

            self.last_job_result = {
                "status": "ok",
                "stage": stage,
                "tool_id": chosen_tool_id,
                "downloaded_path": downloaded_path,
            }
            return self.last_job_result
