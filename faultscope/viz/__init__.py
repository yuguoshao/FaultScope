"""PNG visualizations for noise-hotspot diagnostics."""

from faultscope.viz.drawing import VisualizationUnavailableError
from faultscope.viz.repetition import (
    write_repetition_gate_structure_hotspot_map,
    write_repetition_hotspot_heatmap,
)
from faultscope.viz.surface_code import write_rotated_surface_code_spatial_hotspot_map

__all__ = [
    "VisualizationUnavailableError",
    "write_repetition_gate_structure_hotspot_map",
    "write_repetition_hotspot_heatmap",
    "write_rotated_surface_code_spatial_hotspot_map",
]
