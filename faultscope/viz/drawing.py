"""Shared drawing helpers for hotspot visualizations."""

from __future__ import annotations

from typing import Any


class VisualizationUnavailableError(ImportError):
    """Raised when optional visualization dependencies are unavailable."""


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
    cols = len(matrix[0]) if matrix else 0
    draw.text((x0, y0 - 40), title, fill="#17202a", font=fonts["bold"])
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
    if cols == 0:
        placeholder_width = 180
        placeholder_height = max(cell, rows * cell)
        draw.rectangle(
            (
                x0 + label_width,
                y0,
                x0 + label_width + placeholder_width,
                y0 + placeholder_height - 3,
            ),
            fill="#eef1f4",
            outline="#c4ccd6",
        )
        _draw_centered_text(
            draw,
            "No checks",
            x0 + label_width + placeholder_width // 2,
            y0 + placeholder_height // 2,
            fonts["regular"],
            fill="#53687d",
        )
        return

    max_value = max((value for row in matrix for value in row), default=0.0) or 1.0
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
    return (
        int(start[0] + (end[0] - start[0]) * max(0.0, min(1.0, local))),
        int(start[1] + (end[1] - start[1]) * max(0.0, min(1.0, local))),
        int(start[2] + (end[2] - start[2]) * max(0.0, min(1.0, local))),
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
