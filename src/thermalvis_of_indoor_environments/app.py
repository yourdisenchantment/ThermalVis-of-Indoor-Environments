# src/thermalvis_of_indoor_environments/app.py

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from scipy.interpolate import griddata


def _make_points(n: float, w: float, s: float, e: float) -> np.ndarray:
    north_point = [0.0, 12.5, n]
    west_point = [12.5, 0.0, w]
    south_point = [25.0, 12.5, s]
    east_point = [12.5, 25.0, e]
    ne_point = [0.0, 25.0, (east_point[2] + north_point[2]) / 2]
    nw_point = [0.0, 0.0, (north_point[2] + west_point[2]) / 2]
    sw_point = [25.0, 0.0, (west_point[2] + south_point[2]) / 2]
    se_point = [25.0, 25.0, (south_point[2] + east_point[2]) / 2]
    return np.array(
        [north_point, west_point, south_point, east_point, ne_point, nw_point, sw_point, se_point],
        dtype=float,
    )


def _interpolate(points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    zc = griddata((x, y), z, (grid_x, grid_y), method="cubic")
    if np.all(np.isnan(zc)):
        zc = griddata((x, y), z, (grid_x, grid_y), method="linear")
    return zc


def run_visualization(
    north: float,
    west: float,
    south: float,
    east: float,
    north2: float | None = None,
    west2: float | None = None,
    south2: float | None = None,
    east2: float | None = None,
    show: bool = True,
    min_gap: float = 0.1,  # минимальный зазор между поверхностями
    alpha: float = 0.4,  # прозрачность поверхностей
    tick_step: float = 0.1,  # шаг делений по оси Z и colorbar (например, 0.1 или 0.05)
) -> tuple[plt.Figure, plt.Axes]:
    # Сетка и область
    grid_x, grid_y = np.mgrid[-1:26:25j, 0:25:25j]
    clip_contour = np.array(
        [(5, -5), (5, 5), (0, 5), (0, 25), (5, 25), (20, 25), (25, 25), (25, 5), (20, 5), (20, -5)]
    )
    clip_path = Path(clip_contour)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    mask = clip_path.contains_points(grid_points).reshape(grid_x.shape)

    # Набор #1
    pts1 = _make_points(north, west, south, east)
    Z1 = _interpolate(pts1, grid_x, grid_y)
    Z1[~mask] = np.nan

    # Набор #2 (опционально)
    has_second = all(v is not None for v in (north2, west2, south2, east2))
    Z2 = None
    if has_second:
        pts2 = _make_points(north2 or 0.0, west2 or 0.0, south2 or 0.0, east2 or 0.0)
        Z2 = _interpolate(pts2, grid_x, grid_y)
        Z2[~mask] = np.nan

    # Общий диапазон по введённым значениям
    inputs = [north, west, south, east]
    if has_second:
        inputs += [float(north2), float(west2), float(south2), float(east2)]
    vmin_in = float(min(inputs))
    vmax_in = float(max(inputs))
    if vmax_in == vmin_in:
        vmax_in = vmin_in + 1e-6

    # Антислипание
    if has_second and Z2 is not None:
        diff = Z2 - Z1
        finite = np.isfinite(diff)
        if np.any(finite):
            min_abs_diff = float(np.nanmin(np.abs(diff[finite])))
            if min_abs_diff < min_gap:
                sign = float(np.sign(np.nanmedian(diff[finite]))) or 1.0
                delta = (min_gap - min_abs_diff) / 2.0
                Z1 = Z1 - delta * sign
                Z2 = Z2 + delta * sign

    # Границы по Z для стен: [floor(min); ceil(max к ближайшей десятой) + 0.1]
    zmin_surface = np.nanmin(Z1) if Z2 is None else float(min(np.nanmin(Z1), np.nanmin(Z2)))
    zmax_surface = np.nanmax(Z1) if Z2 is None else float(max(np.nanmax(Z1), np.nanmax(Z2)))
    z_floor = float(math.floor(min(vmin_in, zmin_surface)))
    z_ceil = float(math.ceil(max(vmax_in, zmax_surface) * 10) / 10 + 0.1)
    z_floor = round(z_floor, 2)
    z_ceil = round(z_ceil, 2)

    # Палитра (как в исходнике)
    colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    def _rgba(Z: np.ndarray) -> np.ndarray:
        norm_grid = (Z - vmin_in) / (vmax_in - vmin_in)
        rgba = cmap(norm_grid)
        rgba[..., 3] = alpha
        return rgba

    rgba1 = _rgba(Z1)
    rgba2 = _rgba(Z2) if (has_second and Z2 is not None) else None

    # Фигура и 3D‑оси
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=40, azim=-140)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Поверхности
    ax.plot_surface(grid_x, grid_y, Z1, facecolors=rgba1, edgecolor="none", shade=False)
    if has_second and Z2 is not None and rgba2 is not None:
        ax.plot_surface(grid_x, grid_y, Z2, facecolors=rgba2, edgecolor="none", shade=False)

    # Подписи к точкам в требуемом порядке: т.1 запад, т.2 север, т.3 юг, т.4 восток
    pts1_ordered = [
        ("т.1 (запад)", [12.5, 0.0, west]),
        ("т.2 (север)", [0.0, 12.5, north]),
        ("т.3 (юг)", [25.0, 12.5, south]),
        ("т.4 (восток)", [12.5, 25.0, east]),
    ]
    ax.scatter(
        [p[1][0] for p in pts1_ordered],
        [p[1][1] for p in pts1_ordered],
        [p[1][2] for p in pts1_ordered],
        c="black",
        marker="o",
        label="Набор 1",
    )
    for label, p in pts1_ordered:
        ax.text(p[0], p[1], p[2], label, fontsize=11, va="bottom", color="black")

    if has_second:
        pts2_ordered = [
            ("т.1 (запад)", [12.5, 0.0, west2]),
            ("т.2 (север)", [0.0, 12.5, north2]),
            ("т.3 (юг)", [25.0, 12.5, south2]),
            ("т.4 (восток)", [12.5, 25.0, east2]),
        ]
        ax.scatter(
            [p[1][0] for p in pts2_ordered],
            [p[1][1] for p in pts2_ordered],
            [p[1][2] for p in pts2_ordered],
            c="dimgray",
            marker="^",
            label="Набор 2",
        )
        for label, p in pts2_ordered:
            ax.text(p[0], p[1], p[2], label, fontsize=11, va="bottom", color="dimgray")

    # Цветовая шкала по общему диапазону входов с настраиваемым шагом
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin_in, vmax_in)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.05)
    step = tick_step if (tick_step and tick_step > 0) else 0.1
    tick_start = math.floor(vmin_in / step) * step
    tick_end = math.ceil(vmax_in / step) * step
    ticks = np.round(np.arange(tick_start, tick_end + step / 2.0, step), 3)
    cbar.set_ticks(ticks)
    cbar.set_label(f"Температура (общий диапазон: {vmin_in:.2f} … {vmax_in:.2f})")

    # Стены помещения
    wall_alpha = 0.3
    wall_color = "gray"
    left_points = np.array([(5, -5), (5, 5), (0, 5), (0, 25), (5, 25)])
    right_points = np.array([(20, 25), (25, 25), (25, 5), (20, 5), (20, -5)])
    arc = np.linspace(0, np.pi, 100)
    arc_x = 12.5 + 7.5 * np.cos(arc)
    arc_y = 25 + 7.5 * np.sin(arc)
    arc_points = np.column_stack((arc_x, arc_y))

    def _extrude_segments(pts: np.ndarray) -> None:
        for i in range(len(pts) - 1):
            xs = np.array([pts[i][0], pts[i + 1][0]])
            ys = np.array([pts[i][1], pts[i + 1][1]])
            X, Z = np.meshgrid(xs, [z_floor, z_ceil])
            Y, _ = np.meshgrid(ys, [z_floor, z_ceil])
            ax.plot_surface(X, Y, Z, color=wall_color, alpha=wall_alpha)

    _extrude_segments(left_points)
    _extrude_segments(right_points)
    _extrude_segments(arc_points)

    for pts in (left_points, right_points):
        for i in range(len(pts) - 1):
            ax.plot(
                [pts[i][0], pts[i + 1][0]],
                [pts[i][1], pts[i + 1][1]],
                [z_floor, z_floor],
                color="black",
                linewidth=1.0,
            )
    ax.plot(arc_x, arc_y, np.full_like(arc_x, z_floor), color="black", linewidth=1.0)

    # Ось Z: настраиваемые деления
    ax.set_zlim(z_floor, z_ceil)
    ax.zaxis.set_major_locator(mticker.MultipleLocator(step))
    minor = step / 2.0 if step > 0 else 0.05
    if minor > 0:
        ax.zaxis.set_minor_locator(mticker.MultipleLocator(minor))
    ax.grid(True, which="both", axis="z", linestyle=":", linewidth=0.5, alpha=0.6)

    if has_second:
        ax.legend(loc="upper left")

    if show:
        plt.show()

    return fig, ax
