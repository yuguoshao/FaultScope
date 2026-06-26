"""Rotated surface-code hotspot visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from faultscope.viz.drawing import (
    _contrast_color,
    _draw_centered_text,
    _draw_diamond,
    _draw_score_scale,
    _draw_top_hotspots,
    _hotspot_color,
    _load_pillow,
    _fonts,
)


def write_rotated_surface_code_spatial_hotspot_map(
    result: Any,
    output_path: str | Path,
    *,
    distance: int,
    highlighted_data: tuple[int, int] | None = None,
    highlighted_check_ids: tuple[str, ...] = (),
) -> Path:
    """Write a spatial hotspot map for a rotated surface-code layout.

    The ``result`` object should expose ``hotspots`` and ``locations``. Data
    locations use tags ``role="data"``, ``row`` and ``col``. Check locations
    use tags ``role="x_check"`` or ``role="z_check"`` plus spatial ``x`` and
    ``y`` coordinates in data-grid units.
    """

    if distance < 3 or distance % 2 != 1:
        raise ValueError("rotated surface-code distance must be an odd integer >= 3")

    Image, ImageDraw, ImageFont = _load_pillow()
    fonts = _fonts(ImageFont)
    data_nodes, check_nodes, max_hotspot = _collect_surface_code_hotspots(result)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    spacing = 112
    grid_width = spacing * (distance - 1)
    image_width = max(1350, 260 + grid_width + 520)
    image_height = max(980, 300 + grid_width)
    image = Image.new("RGB", (image_width, image_height), "#f7f8fa")
    draw = ImageDraw.Draw(image)

    draw.text(
        (42, 30),
        f"Rotated Surface Code d={distance}: Spatial Noise Hotspot Map",
        fill="#17202a",
        font=fonts["title"],
    )
    draw.text(
        (42, 70),
        (
            f"shots={getattr(result, 'shots', '?')}, "
            f"logical failure rate={getattr(result, 'logical_failure_rate', 0.0):.4f}"
        ),
        fill="#34495e",
        font=fonts["regular"],
    )
    draw.text(
        (42, 98),
        "Circles are data-qubit noise locations; diamonds are X/Z check noise locations.",
        fill="#34495e",
        font=fonts["small"],
    )

    origin_x = 130
    origin_y = 170
    def px(x_coord: float) -> int:
        return int(origin_x + x_coord * spacing)

    def py(y_coord: float) -> int:
        return int(origin_y + (distance - 1 - y_coord) * spacing)

    # Data grid links.
    for row in range(distance):
        for col in range(distance):
            x = px(col)
            y = py(row)
            if col + 1 < distance:
                draw.line((x, y, px(col + 1), y), fill="#cbd4dd", width=3)
            if row + 1 < distance:
                draw.line((x, y, x, py(row + 1)), fill="#cbd4dd", width=3)

    # Check nodes are drawn under data nodes so data qubit circles stay legible.
    for node in check_nodes:
        x = px(node["x"])
        y = py(node["y"])
        value = node["hotspot"]
        fill = _hotspot_color(value, max_hotspot)
        outline = "#2864b0" if node["basis"] == "X" else "#20845a"
        radius = 21 + int(15 * value / max_hotspot)
        _draw_diamond(draw, x, y, radius, fill=fill, outline=outline, width=4)
        _draw_centered_text(
            draw,
            node["basis"],
            x,
            y,
            fonts["small"],
            fill=_contrast_color(fill),
        )
        if node["id"] in highlighted_check_ids:
            _draw_diamond(draw, x, y, radius + 10, fill=None, outline="#ff3b30", width=5)

    for row in range(distance):
        for col in range(distance):
            value = data_nodes.get((row, col), 0.0)
            x = px(col)
            y = py(row)
            fill = _hotspot_color(value, max_hotspot)
            radius = 20 + int(18 * value / max_hotspot)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=fill,
                outline="#68316d",
                width=3,
            )
            _draw_centered_text(
                draw,
                f"{row},{col}",
                x,
                y,
                fonts["tiny"],
                fill=_contrast_color(fill),
            )
            if highlighted_data == (row, col):
                draw.ellipse(
                    (x - radius - 9, y - radius - 9, x + radius + 9, y + radius + 9),
                    outline="#00d5ff",
                    width=5,
                )

    legend_x = origin_x + grid_width + 120
    legend_y = 168
    _draw_score_scale(
        draw,
        x0=legend_x,
        y0=legend_y,
        width=420,
        max_value=max_hotspot,
        fonts=fonts,
    )
    draw.text(
        (legend_x, legend_y + 70),
        "Check color outline:",
        fill="#34495e",
        font=fonts["small"],
    )
    _draw_diamond(
        draw,
        legend_x + 20,
        legend_y + 110,
        16,
        fill="#f4d18a",
        outline="#2864b0",
        width=4,
    )
    draw.text(
        (legend_x + 48, legend_y + 98),
        "X stabilizer/check noise",
        fill="#34495e",
        font=fonts["small"],
    )
    _draw_diamond(
        draw,
        legend_x + 20,
        legend_y + 152,
        16,
        fill="#f4d18a",
        outline="#20845a",
        width=4,
    )
    draw.text(
        (legend_x + 48, legend_y + 140),
        "Z stabilizer/check noise",
        fill="#34495e",
        font=fonts["small"],
    )

    _draw_top_hotspots(
        draw,
        result,
        x0=legend_x,
        y0=405,
        width=330,
        fonts=fonts,
        top_k=10,
    )
    draw.text(
        (42, image_height - 80),
        (
            "Spatial view of location-level hotspot scores projected onto "
            f"a d={distance} rotated surface-code layout."
        ),
        fill="#34495e",
        font=fonts["small"],
    )

    image.save(output)
    return output


def _collect_surface_code_hotspots(
    result: Any,
) -> tuple[dict[tuple[int, int], float], list[dict[str, Any]], float]:
    data_nodes: dict[tuple[int, int], float] = {}
    check_nodes: list[dict[str, Any]] = []
    for location_id, hotspot in result.hotspots.items():
        tags = result.locations[location_id].tags
        role = tags.get("role")
        if role == "data":
            key = (int(tags["row"]), int(tags["col"]))
            data_nodes[key] = data_nodes.get(key, 0.0) + float(hotspot)
        elif role in {"x_check", "z_check"}:
            check_nodes.append(
                {
                    "id": location_id,
                    "x": float(tags["x"]),
                    "y": float(tags["y"]),
                    "basis": "X" if role == "x_check" else "Z",
                    "hotspot": float(hotspot),
                }
            )
    max_hotspot = max(
        [0.0]
        + list(data_nodes.values())
        + [node["hotspot"] for node in check_nodes]
    )
    return data_nodes, check_nodes, max_hotspot or 1.0
