"""Repetition-code hotspot visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from faultscope.viz.drawing import (
    _draw_centered_text,
    _draw_heatmap,
    _draw_score_scale,
    _draw_top_hotspots,
    _hotspot_color,
    _load_pillow,
    _fonts,
    _text_size,
)


def _validate_repetition_dimensions(distance: int, rounds: int) -> None:
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 1:
        raise ValueError("distance must be a positive integer")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")


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

    _validate_repetition_dimensions(distance, rounds)
    Image, ImageDraw, ImageFont = _load_pillow()
    fonts = _fonts(ImageFont)
    data_heat, measurement_heat, _ = _collect_repetition_hotspots(
        result,
        distance=distance,
        rounds=rounds,
    )

    data_x = 70
    y0 = 145
    cell = 86
    label_width = 70
    panel_gap = 50
    side_panel_gap = 80
    data_panel_width = label_width + max(distance * cell, 180)
    measurement_panel_width = label_width + max((distance - 1) * cell, 180)
    measurement_x = data_x + data_panel_width + panel_gap
    top_hotspots_x = measurement_x + measurement_panel_width + side_panel_gap
    image_width = max(1500, top_hotspots_x + 300 + 100)
    footer_y = max(805, y0 + rounds * cell + 100, 150 + 8 * 44 + 45)
    image_height = max(860, footer_y + 55)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (image_width, image_height), "#f7f8fa")
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
        x0=data_x,
        y0=y0,
        cell=cell,
        label_width=label_width,
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
        x0=measurement_x,
        y0=y0,
        cell=cell,
        label_width=label_width,
        title="Measurement noise hotspots",
        x_label="check index",
        palette=((235, 245, 120), (34, 144, 140), (42, 50, 120)),
        fonts=fonts,
        highlighted=highlighted_measurement,
        highlight_color="#ff3b30",
    )
    _draw_top_hotspots(draw, result, x0=top_hotspots_x, y0=150, width=300, fonts=fonts)

    draw.text(
        (42, footer_y),
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

    _validate_repetition_dimensions(distance, rounds)
    Image, ImageDraw, ImageFont = _load_pillow()
    fonts = _fonts(ImageFont)
    data_hot, measurement_hot, cx_hot = _collect_gate_hotspots(result)
    max_hotspot = max([0.0, *data_hot.values(), *measurement_hot.values(), *cx_hot.values()])
    max_hotspot = max_hotspot or 1.0

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
    y_by_lane = {lane: top + lane_index * lane_gap for lane_index, lane in enumerate(lanes)}
    y_data = {index: y_by_lane[("data", index, f"D{index}")] for index in range(distance)}
    y_ancilla = {index: y_by_lane[("ancilla", index, f"A{index}")] for index in range(distance - 1)}
    right_edge = left + rounds * block_width
    lane_bottom = top + (len(lanes) - 1) * lane_gap
    lower_panel_y = max(915, lane_bottom + 120)
    scale_y = lower_panel_y + 25
    footer_y = max(lower_panel_y + 8 * 44 + 55, scale_y + 80)
    image_width = max(2100, right_edge + 80)
    image_height = max(1160, footer_y + 45)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (image_width, image_height), "#f7f8fa")
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
        y0=scale_y,
        width=520,
        max_value=max_hotspot,
        fonts=fonts,
    )
    _draw_top_hotspots(draw, result, x0=670, y0=lower_panel_y, width=270, fonts=fonts)
    draw.text(
        (42, footer_y),
        "Gate-schedule hotspot view from the same forward score-function estimator.",
        fill="#34495e",
        font=fonts["small"],
    )

    image.save(output)
    return output


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
            cx_hot[(int(tags["round"]), int(tags["check"]), str(tags["side"]))] = float(hotspot)
    return data_hot, measurement_hot, cx_hot
