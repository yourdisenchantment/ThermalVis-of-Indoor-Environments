# src/thermalvis_of_indoor_environments/app.py

from __future__ import annotations

import math

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.path import Path
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion


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
    min_gap: float = 0.1,  # вертикальный зазор между поверхностями
    alpha: float = 0.4,  # прозрачность поверхностей
    tick_step: float = 0.1,  # ШАГ делений по оси Z И цветовой шкале (0.1, 0.05 и т.п.)
) -> tuple[plt.Figure, plt.Axes]:
    # Сетка и область (узлы совпадают с целыми координатами 0..25)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 25, 51), np.linspace(0, 25, 51), indexing="ij")
    clip_contour = np.array(
        [(5, -5), (5, 5), (0, 5), (0, 25), (5, 25), (20, 25), (25, 25), (25, 5), (20, 5), (20, -5)]
    )
    clip_path = Path(clip_contour)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    mask = clip_path.contains_points(grid_points, radius=-0.49).reshape(grid_x.shape)
    core = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    edge_ring = mask & ~core  # одно-узловое кольцо по периметру

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

    # Общий диапазон по входам (для подписи/информации)
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

    # Границы Z кратно шагу tick_step (и для оси, и для цветовой шкалы)
    zmin_surface = np.nanmin(Z1) if Z2 is None else float(min(np.nanmin(Z1), np.nanmin(Z2)))
    zmax_surface = np.nanmax(Z1) if Z2 is None else float(max(np.nanmax(Z1), np.nanmax(Z2)))
    z_min_raw = float(min(vmin_in, zmin_surface))
    z_max_raw = float(max(vmax_in, zmax_surface))
    step = tick_step if (tick_step and tick_step > 0) else 0.1

    z_floor = math.floor(z_min_raw / step) * step
    z_ceil = math.ceil(z_max_raw / step) * step
    if z_ceil <= z_floor:
        z_ceil = z_floor + step
    # Чуть округляем, чтобы не было 20.599999:
    z_floor = float(np.round(z_floor, 6))
    z_ceil = float(np.round(z_ceil, 6))

    # Палитра (как в исходнике) + прозрачность для NaN
    colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    cmap.set_bad((0, 0, 0, 0))  # NaN - прозрачные

    # Дискретные уровни цвета c заданным шагом + нормализация
    levels = np.round(np.arange(z_floor, z_ceil + step / 2.0, step), 6)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    def _rgba(Z: np.ndarray) -> np.ndarray:
        rgba = cmap(norm(Z))
        # сохраняем прозрачность NaN, альфа - только по валидным значениям
        finite_mask = np.isfinite(Z)
        rgba[finite_mask, 3] = alpha
        return rgba

    rgba1 = _rgba(Z1)
    rgba2 = _rgba(Z2) if (has_second and Z2 is not None) else None

    if "edge_ring" in locals():
        if rgba1 is not None:
            er1 = edge_ring & np.isfinite(Z1)
            rgba1[er1, 3] = np.clip(alpha * 0.25, 0.0, 1.0)  # 25% от общей альфы
        if rgba2 is not None and Z2 is not None:
            er2 = edge_ring & np.isfinite(Z2)
            rgba2[er2, 3] = np.clip(alpha * 0.25, 0.0, 1.0)

    # Фигура
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=40, azim=40)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Поверхности
    ax.plot_surface(grid_x, grid_y, Z1, facecolors=rgba1, edgecolor="none", shade=False)
    if has_second and Z2 is not None and rgba2 is not None:
        ax.plot_surface(grid_x, grid_y, Z2, facecolors=rgba2, edgecolor="none", shade=False)

    # Формат чисел в подписях точек под шаг делений
    val_dec = 2 if step < 0.1 else 1
    val_fmt = f"{{:.{val_dec}f}}"

    # Небольшие компактные плашки
    box1 = dict(
        boxstyle="round,pad=0.15,rounding_size=0.10",
        fc="white",
        lw=0.8,
        alpha=0.75,
    )
    box2 = dict(
        boxstyle="round,pad=0.15,rounding_size=0.10",
        fc="#f3f3f3",
        lw=0.8,
        alpha=0.75,
    )

    # Настройка расположения плашек
    # варианты: "below", "above", "left", "right"
    label_mode1 = "below"
    label_mode2 = "above" if has_second else "below"
    xy_off = 0.8  # смещение по X/Y (в «метрах» плана 0..25)
    z_off = max(0.6 * step, 0.05)  # смещение по Z связано с шагом делений

    def _offset_point(p, mode: str, xy_off: float, z_off: float):
        x, y, z = p
        ha, va = "center", "center"
        if mode == "below":
            z -= z_off
            ha, va = "center", "top"
        elif mode == "above":
            z += z_off
            ha, va = "center", "bottom"
        elif mode == "right":
            x += xy_off
            ha, va = "left", "center"
        elif mode == "left":
            x -= xy_off
            ha, va = "right", "center"
        return (x, y, z), ha, va

    # Подписи к точкам в порядке: т.1 запад, т.2 север, т.3 юг, т.4 восток
    pts1_ordered = [
        (f"т.1 ({val_fmt.format(west)}) Запад", [12.5, 0.0, west]),
        (f"т.2 ({val_fmt.format(north)}) Север", [0.0, 12.5, north]),
        (f"т.3 ({val_fmt.format(south)}) Юг", [25.0, 12.5, south]),
        (f"т.4 ({val_fmt.format(east)}) Восток", [12.5, 25.0, east]),
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
        (tx, ty, tz), ha, va = _offset_point(p, label_mode1, xy_off, z_off)
        txt = ax.text(
            tx,
            ty,
            tz,
            label,
            fontsize=9,
            color="#111111",
            ha=ha,
            va=va,
            bbox=box1,
            zorder=10,
            clip_on=False,
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.8)])

    if has_second:
        pts2_ordered = [
            (f"т.1 ({val_fmt.format(west2)}) Запад", [12.5, 0.0, west2]),
            (f"т.2 ({val_fmt.format(north2)}) Север", [0.0, 12.5, north2]),
            (f"т.3 ({val_fmt.format(south2)}) Юг", [25.0, 12.5, south2]),
            (f"т.4 ({val_fmt.format(east2)}) Восток", [12.5, 25.0, east2]),
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
            (tx, ty, tz), ha, va = _offset_point(p, label_mode2, xy_off, z_off)
            txt = ax.text(
                tx,
                ty,
                tz,
                label,
                fontsize=9,
                color="#222222",
                ha=ha,
                va=va,
                bbox=box2,
                zorder=10,
                clip_on=False,
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.8)])

    # Цветовая шкала: ровно по уровням levels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_clim(z_floor, z_ceil)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.05, boundaries=levels, ticks=levels)
    # Красота формата тиков
    if step < 0.1:
        fmt = mticker.FormatStrFormatter("%.2f")
    else:
        fmt = mticker.FormatStrFormatter("%.1f")
    cbar.formatter = fmt
    cbar.update_ticks()
    cbar.set_label(f"Диапазон: {z_floor:.2f} … {z_ceil:.2f} (шаг {step:g})")

    # Стены помещения
    wall_alpha = 0.1  # оставляем 0.1, как просили
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

    # Ось Z: деления строго по step, без «мелких»
    ax.set_zlim(z_floor, z_ceil)
    ax.zaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.zaxis.set_minor_locator(mticker.NullLocator())
    ax.zaxis.set_major_formatter(fmt)
    ax.grid(True, which="major", axis="z", linestyle=":", linewidth=0.5, alpha=0.6)

    if has_second:
        ax.legend(loc="upper left")

    if show:
        plt.show()

    return fig, ax
