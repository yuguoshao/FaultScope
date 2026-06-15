"""PNG visualizations for noise-hotspot diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VisualizationUnavailableError(ImportError):
    """Raised when optional visualization dependencies are unavailable."""


def write_repetition_hotspot_heatmap(
    result: Any,
    output_path: str | Path,
    *,
    distance: int,
    rounds: int,
    highlighted_data: tuple[int, int] | None = None,
    highlighted_measurement: tuple[int, int] | None = None,
) -> Path:
    """Write data-qubit and measurement spatiotemporal hotspot heatmaps."""

    Image, ImageDraw, ImageFont = _load_pillow()
    fonts = _fonts(ImageFont)
    data_heat, measurement_heat, max_hotspot = _collect_repetition_hotspots(
        result,
        distance=distance,
        rounds=rounds,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1500, 860), "#f7f8fa")
    draw = ImageDraw.Draw(image)

    draw.text(
        (42, 30),
        "Forward Noise-aware Stabilizer Simulator: Noise Hotspot Map",
        fill="#17202a",
        font=fonts["title"],
    )
    draw.text(
        (42, 70),
        (
            f"distance={distance}, rounds={rounds}, "
            f"shots={getattr(result, 'shots', '?')}, "
            f"logical failure rate={getattr(result, 'logical_failure_rate', 0.0):.4f}"
        ),
        fill="#34495e",
        font=fonts["regular"],
    )

    _draw_heatmap(
        draw,
        data_heat,
        x0=70,
        y0=145,
        cell=86,
        label_width=70,
        title="Data-qubit noise hotspots",
        x_label="data qubit index",
        palette=((250, 250, 190), (203, 71, 119), (35, 10, 70)),
        fonts=fonts,
        highlighted=highlighted_data,
        highlight_color="#00d5ff",
    )
    _draw_heatmap(
        draw,
        measurement_heat,
        x0=690,
        y0=145,
        cell=86,
        label_width=70,
        title="Measurement noise hotspots",
        x_label="check index",
        palette=((235, 245, 120), (34, 144, 140), (42, 50, 120)),
        fonts=fonts,
        highlighted=highlighted_measurement,
        highlight_color="#ff3b30",
    )
    _draw_top_hotspots(draw, result, x0=1090, y0=150, width=300, fonts=fonts)

    draw.text(
        (42, 805),
        "Hotspot score H_l = |partial J / partial lambda_l|.",
        fill="#34495e",
        font=fonts["regular"],
    )
    image.save(output)
    return output


def write_repetition_gate_structure_hotspot_map(
    result: Any,
    output_path: str | Path,
    *,
    distance: int,
    rounds: int,
    highlighted_data: tuple[int, int] | None = None,
    highlighted_measurement: tuple[int, int] | None = None,
    highlighted_cx: tuple[int, int, str] | None = None,
) -> Path:
    """Write a gate-schedule view with hotspot markers on gate locations."""

    Image, ImageDraw, ImageFont = _load_pillow()
    fonts = _fonts(ImageFont)
    data_hot, measurement_hot, cx_hot = _collect_gate_hotspots(result)
    max_hotspot = max(
        [0.0, *data_hot.values(), *measurement_hot.values(), *cx_hot.values()]
    )
    max_hotspot = max_hotspot or 1.0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2100, 1160), "#f7f8fa")
    draw = ImageDraw.Draw(image)

    draw.text(
        (42, 30),
        "Spatiotemporal Gate-structure Noise Hotspot Map",
        fill="#17202a",
        font=fonts["title"],
    )
    draw.text(
        (42, 70),
        (
            f"distance={distance}, rounds={rounds}, "
            f"shots={getattr(result, 'shots', '?')}, "
            f"logical failure rate={getattr(result, 'logical_failure_rate', 0.0):.4f}"
        ),
        fill="#34495e",
        font=fonts["regular"],
    )
    draw.text(
        (42, 98),
        "Colored markers show H_l on the physical gate schedule.",
        fill="#34495e",
        font=fonts["small"],
    )

    left = 112
    top = 190
    lane_gap = 72
    block_width = 330
    phase = {"idle": 34, "reset": 88, "cx_left": 145, "cx_right": 214, "measure": 286}
    lanes = []
    for index in range(distance):
        lanes.append(("data", index, f"D{index}"))
        if index < distance - 1:
            lanes.append(("ancilla", index, f"A{index}"))
    y_by_lane = {
        lane: top + lane_index * lane_gap
        for lane_index, lane in enumerate(lanes)
    }
    y_data = {index: y_by_lane[("data", index, f"D{index}")] for index in range(distance)}
    y_ancilla = {
        index: y_by_lane[("ancilla", index, f"A{index}")]
        for index in range(distance - 1)
    }
    right_edge = left + rounds * block_width

    for lane in lanes:
        kind, _, label = lane
        y = y_by_lane[lane]
        draw.text(
            (42, y - 12),
            label,
            fill="#23384f" if kind == "data" else "#48627c",
            font=fonts["bold"] if kind == "data" else fonts["regular"],
        )
        draw.line((left - 10, y, right_edge, y), fill="#c7d0d9", width=2)

    for round_index in range(rounds):
        x0 = left + round_index * block_width
        draw.line(
            (x0 - 10, top - 52, x0 - 10, top + (len(lanes) - 1) * lane_gap + 50),
            fill="#d6dce3",
            width=2,
        )
        draw.text((x0 + 115, top - 70), f"round {round_index}", fill="#17202a", font=fonts["bold"])
        for label, phase_key in (
            ("idle", "idle"),
            ("R", "reset"),
            ("CX-L", "cx_left"),
            ("CX-R", "cx_right"),
            ("M", "measure"),
        ):
            x = x0 + phase[phase_key]
            text_width, _ = _text_size(draw, label, fonts["tiny"])
            draw.text(
                (x - text_width / 2, top - 38),
                label,
                fill="#53687d",
                font=fonts["tiny"],
            )

        for data_index in range(distance):
            value = data_hot.get((round_index, data_index), 0.0)
            x = x0 + phase["idle"]
            y = y_data[data_index]
            radius = 7 + int(13 * value / max_hotspot)
            fill = _hotspot_color(value, max_hotspot)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=fill,
                outline="#68316d",
                width=2,
            )
            if highlighted_data == (round_index, data_index):
                draw.ellipse(
                    (
                        x - radius - 6,
                        y - radius - 6,
                        x + radius + 6,
                        y + radius + 6,
                    ),
                    outline="#00d5ff",
                    width=4,
                )

        for check_index in range(distance - 1):
            y_anc = y_ancilla[check_index]
            reset_x = x0 + phase["reset"]
            draw.rounded_rectangle(
                (reset_x - 20, y_anc - 16, reset_x + 20, y_anc + 16),
                radius=5,
                fill="#eef2f6",
                outline="#6b7c8e",
                width=2,
            )
            draw.text(
                (reset_x - 7, y_anc - 9),
                "R",
                fill="#27384a",
                font=fonts["small"],
            )

            for side, data_index, phase_key in (
                ("left", check_index, "cx_left"),
                ("right", check_index + 1, "cx_right"),
            ):
                x = x0 + phase[phase_key]
                y_data_line = y_data[data_index]
                draw.line(
                    (x, min(y_data_line, y_anc), x, max(y_data_line, y_anc)),
                    fill="#263645",
                    width=3,
                )
                draw.ellipse(
                    (x - 6, y_data_line - 6, x + 6, y_data_line + 6),
                    fill="#263645",
                )
                draw.ellipse(
                    (x - 12, y_anc - 12, x + 12, y_anc + 12),
                    outline="#263645",
                    width=3,
                )
                draw.line((x - 8, y_anc, x + 8, y_anc), fill="#263645", width=2)
                draw.line((x, y_anc - 8, x, y_anc + 8), fill="#263645", width=2)

                value = cx_hot.get((round_index, check_index, side), 0.0)
                if value:
                    marker_y = (y_data_line + y_anc) // 2
                    radius = 9 + int(15 * value / max_hotspot)
                    fill = _hotspot_color(value, max_hotspot)
                    draw.ellipse(
                        (
                            x - radius,
                            marker_y - radius,
                            x + radius,
                            marker_y + radius,
                        ),
                        fill=fill,
                        outline="#68316d",
                        width=2,
                    )
                    _draw_centered_text(
                        draw,
                        f"{value:.2f}",
                        x,
                        marker_y,
                        fonts["tiny"],
                        fill="#111111",
                    )
                if highlighted_cx == (round_index, check_index, side):
                    marker_y = (y_data_line + y_anc) // 2
                    draw.rectangle(
                        (x - 30, marker_y - 30, x + 30, marker_y + 30),
                        outline="#ff9f1a",
                        width=4,
                    )

            value = measurement_hot.get((round_index, check_index), 0.0)
            measure_x = x0 + phase["measure"]
            fill = _hotspot_color(value, max_hotspot)
            draw.rounded_rectangle(
                (measure_x - 28, y_anc - 20, measure_x + 28, y_anc + 20),
                radius=6,
                fill=fill,
                outline="#263645",
                width=2,
            )
            _draw_centered_text(
                draw,
                "M",
                measure_x,
                y_anc,
                fonts["small"],
                fill="#111111",
            )
            if highlighted_measurement == (round_index, check_index):
                draw.rounded_rectangle(
                    (measure_x - 36, y_anc - 28, measure_x + 36, y_anc + 28),
                    radius=8,
                    outline="#ff3b30",
                    width=4,
                )

    _draw_score_scale(
        draw,
        x0=42,
        y0=940,
        width=520,
        max_value=max_hotspot,
        fonts=fonts,
    )
    _draw_top_hotspots(draw, result, x0=670, y0=915, width=270, fonts=fonts)
    draw.text(
        (42, 1115),
        "Gate-schedule hotspot view from the same forward score-function estimator.",
        fill="#34495e",
        font=fonts["small"],
    )

    image.save(output)
    return output


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


def _load_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise VisualizationUnavailableError(
            "Pillow is required for hotspot visualization. Install `pillow`."
        ) from exc
    return Image, ImageDraw, ImageFont


def _fonts(ImageFont: Any) -> dict[str, Any]:
    def load(path: str, size: int) -> Any:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()

    return {
        "title": load("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26),
        "bold": load("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20),
        "regular": load("/System/Library/Fonts/Supplemental/Arial.ttf", 16),
        "small": load("/System/Library/Fonts/Supplemental/Arial.ttf", 13),
        "tiny": load("/System/Library/Fonts/Supplemental/Arial.ttf", 11),
    }


def _collect_repetition_hotspots(
    result: Any,
    *,
    distance: int,
    rounds: int,
) -> tuple[list[list[float]], list[list[float]], float]:
    data_heat = [[0.0 for _ in range(distance)] for _ in range(rounds)]
    measurement_heat = [[0.0 for _ in range(distance - 1)] for _ in range(rounds)]
    for location_id, hotspot in result.hotspots.items():
        tags = result.locations[location_id].tags
        if tags.get("operation") == "data_noise":
            data_heat[int(tags["round"])][int(tags["data_index"])] = float(hotspot)
        elif tags.get("operation") == "measurement_noise":
            measurement_heat[int(tags["round"])][int(tags["check"])] = float(hotspot)
    max_hotspot = max(
        [0.0]
        + [value for row in data_heat for value in row]
        + [value for row in measurement_heat for value in row]
    )
    return data_heat, measurement_heat, max_hotspot or 1.0


def _collect_gate_hotspots(
    result: Any,
) -> tuple[
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[int, int, str], float],
]:
    data_hot: dict[tuple[int, int], float] = {}
    measurement_hot: dict[tuple[int, int], float] = {}
    cx_hot: dict[tuple[int, int, str], float] = {}
    for location_id, hotspot in result.hotspots.items():
        tags = result.locations[location_id].tags
        operation = tags.get("operation")
        if operation == "data_noise":
            data_hot[(int(tags["round"]), int(tags["data_index"]))] = float(hotspot)
        elif operation == "measurement_noise":
            measurement_hot[(int(tags["round"]), int(tags["check"]))] = float(hotspot)
        elif operation == "cx_noise":
            cx_hot[(int(tags["round"]), int(tags["check"]), str(tags["side"]))] = (
                float(hotspot)
            )
    return data_hot, measurement_hot, cx_hot


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


def _draw_heatmap(
    draw: Any,
    matrix: list[list[float]],
    *,
    x0: int,
    y0: int,
    cell: int,
    label_width: int,
    title: str,
    x_label: str,
    palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    fonts: dict[str, Any],
    highlighted: tuple[int, int] | None,
    highlight_color: str,
) -> None:
    rows = len(matrix)
    cols = len(matrix[0])
    max_value = max(max(row) for row in matrix) or 1.0
    draw.text((x0, y0 - 40), title, fill="#17202a", font=fonts["bold"])
    for col in range(cols):
        _draw_centered_text(
            draw,
            str(col),
            x0 + label_width + col * cell + cell // 2,
            y0 - 18,
            fonts["regular"],
            fill="#34495e",
        )
    for row in range(rows):
        display_row = rows - 1 - row
        _draw_centered_text(
            draw,
            str(display_row),
            x0 + label_width - 25,
            y0 + row * cell + cell // 2,
            fonts["regular"],
            fill="#34495e",
        )
        for col in range(cols):
            value = matrix[display_row][col]
            color = _ramp(value, max_value, palette)
            x = x0 + label_width + col * cell
            y = y0 + row * cell
            draw.rectangle(
                (x, y, x + cell - 3, y + cell - 3),
                fill=color,
                outline="#ffffff",
            )
            _draw_centered_text(
                draw,
                f"{value:.3f}",
                x + cell // 2,
                y + cell // 2,
                fonts["small"],
                fill=_contrast_color(color),
            )
    if highlighted is not None:
        high_row, high_col = highlighted
        yrow = rows - 1 - high_row
        x = x0 + label_width + high_col * cell
        y = y0 + yrow * cell
        draw.rectangle(
            (x + 4, y + 4, x + cell - 7, y + cell - 7),
            outline=highlight_color,
            width=4,
        )
    scale_y = y0 + rows * cell + 24
    _draw_score_scale(
        draw,
        x0=x0 + label_width,
        y0=scale_y,
        width=cols * cell - 3,
        max_value=max_value,
        fonts=fonts,
        palette=palette,
        label=x_label,
    )


def _draw_top_hotspots(
    draw: Any,
    result: Any,
    *,
    x0: int,
    y0: int,
    width: int,
    fonts: dict[str, Any],
    top_k: int = 8,
) -> None:
    rows = result.top_hotspots(top_k=top_k)
    max_value = max([row.hotspot for row in rows], default=1.0) or 1.0
    draw.text(
        (x0, y0 - 38),
        "Top hotspot locations",
        fill="#17202a",
        font=fonts["bold"],
    )
    for index, row in enumerate(rows):
        y = y0 + index * 44
        draw.text((x0, y), row.location_id, fill="#17202a", font=fonts["tiny"])
        draw.rectangle(
            (x0, y + 16, x0 + width, y + 33),
            fill="#eef1f4",
            outline="#c4ccd6",
        )
        draw.rectangle(
            (x0, y + 16, x0 + int(width * row.hotspot / max_value), y + 33),
            fill=_hotspot_color(row.hotspot, max_value),
        )
        draw.text(
            (x0 + width + 12, y + 16),
            f"{row.hotspot:.4f}",
            fill="#34495e",
            font=fonts["tiny"],
        )


def _draw_score_scale(
    draw: Any,
    *,
    x0: int,
    y0: int,
    width: int,
    max_value: float,
    fonts: dict[str, Any],
    palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None,
    label: str = "|dJ/dlambda|",
) -> None:
    colors = palette or ((254, 235, 172), (241, 111, 92), (108, 36, 117))
    for index in range(width):
        value = max_value * index / max(1, width - 1)
        draw.line((x0 + index, y0, x0 + index, y0 + 14), fill=_ramp(value, max_value, colors))
    draw.rectangle((x0, y0, x0 + width, y0 + 14), outline="#9aa6b2")
    draw.text((x0, y0 + 18), "0", fill="#34495e", font=fonts["tiny"])
    max_text = f"{max_value:.3f}"
    text_width, _ = _text_size(draw, max_text, fonts["tiny"])
    draw.text((x0 + width - text_width, y0 + 18), max_text, fill="#34495e", font=fonts["tiny"])
    _draw_centered_text(draw, label, x0 + width // 2, y0 + 26, fonts["tiny"], fill="#34495e")


def _draw_diamond(
    draw: Any,
    x: int,
    y: int,
    radius: int,
    *,
    fill: tuple[int, int, int] | str | None,
    outline: str,
    width: int,
) -> None:
    points = ((x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y))
    draw.polygon(points, fill=fill, outline=outline)
    if width > 1:
        for offset in range(1, width):
            inner = max(1, radius - offset)
            inner_points = (
                (x, y - inner),
                (x + inner, y),
                (x, y + inner),
                (x - inner, y),
            )
            draw.line((*inner_points, inner_points[0]), fill=outline, width=1)


def _hotspot_color(value: float, max_value: float) -> tuple[int, int, int]:
    return _ramp(
        value,
        max_value,
        ((254, 235, 172), (241, 111, 92), (108, 36, 117)),
    )


def _ramp(
    value: float,
    max_value: float,
    colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> tuple[int, int, int]:
    t = 0.0 if max_value <= 0 else value / max_value
    if t <= 0.5:
        local = t / 0.5
        start, end = colors[0], colors[1]
    else:
        local = (t - 0.5) / 0.5
        start, end = colors[1], colors[2]
    return tuple(
        int(start[index] + (end[index] - start[index]) * max(0.0, min(1.0, local)))
        for index in range(3)
    )


def _contrast_color(color: tuple[int, int, int]) -> str:
    luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    return "#ffffff" if luminance < 125 else "#111111"


def _draw_centered_text(
    draw: Any,
    text: str,
    x: float,
    y: float,
    font: Any,
    *,
    fill: str,
) -> None:
    width, height = _text_size(draw, text, font)
    draw.text((x - width / 2, y - height / 2), text, fill=fill, font=font)


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]
