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


def _make_points(n: float, w: float, s: float, e: float):
    north_point = [0.0, 12.5, n]
    west_point = [12.5, 0.0, w]
    south_point = [25.0, 12.5, s]
    east_point = [12.5, 25.0, e]
    ne_point = [0.0, 25.0, (east_point[2] + north_point[2]) / 2]
    nw_point = [0.0, 0.0, (north_point[2] + west_point[2]) / 2]
    sw_point = [25.0, 0.0, (west_point[2] + south_point[2]) / 2]
    se_point = [25.0, 25.0, (south_point[2] + east_point[2]) / 2]
    return np.array(
        [
            north_point,
            west_point,
            south_point,
            east_point,
            ne_point,
            nw_point,
            sw_point,
            se_point,
        ],
        dtype=float,
    )


def _interpolate(points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray):
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
    tick_step: float = 0.1,  # шаг делений по оси Z и цветовой шкале
    grid_res: int = 27,
):
    # Оверскан на одну ячейку с каждой стороны
    grid_x, grid_y = np.meshgrid(
        np.linspace(-1, 26, grid_res),
        np.linspace(-1, 26, grid_res),
        indexing="ij",
    )

    # Полигон помещения
    clip_contour = np.array(
        [
            (5, -5),
            (5, 5),
            (0, 5),
            (0, 25),
            (5, 25),
            (20, 25),
            (25, 25),
            (25, 5),
            (20, 5),
            (20, -5),
        ]
    )
    clip_path = Path(clip_contour)

    # Узлы "внутри или на границе"
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    mask_nodes = clip_path.contains_points(grid_points, radius=1e-12).reshape(grid_x.shape)

    # Интерполяция
    def _interp(n, w, s, e):
        pts = _make_points(n, w, s, e)
        Z = _interpolate(pts, grid_x, grid_y)
        Z[~mask_nodes] = np.nan
        return Z

    Z1 = _interp(north, west, south, east)
    has_second = all(v is not None for v in (north2, west2, south2, east2))
    Z2 = _interp(north2, west2, south2, east2) if has_second else None

    # Общий диапазон входов
    inputs = [north, west, south, east]

    if has_second:
        inputs += [float(north2), float(west2), float(south2), float(east2)]

    vmin_in = float(min(inputs))
    vmax_in = float(max(inputs))

    if vmax_in == vmin_in:
        vmax_in = vmin_in + 1e-6

    if has_second and Z2 is not None:
        diff = Z2 - Z1
        finite = np.isfinite(diff)

        if np.any(finite):
            min_abs = float(np.nanmin(np.abs(diff[finite])))

            if min_abs < min_gap:
                sign = float(np.sign(np.nanmedian(diff[finite]))) or 1.0
                delta = (min_gap - min_abs) / 2.0
                Z1 = Z1 - delta * sign
                Z2 = Z2 + delta * sign

    # Границы Z кратно шагу
    zmin_surface = np.nanmin(Z1) if Z2 is None else float(min(np.nanmin(Z1), np.nanmin(Z2)))
    zmax_surface = np.nanmax(Z1) if Z2 is None else float(max(np.nanmax(Z1), np.nanmax(Z2)))
    z_min_raw = float(min(vmin_in, zmin_surface))
    z_max_raw = float(max(vmax_in, zmax_surface))
    step = tick_step if (tick_step and tick_step > 0) else 0.1

    z_floor = math.floor(z_min_raw / step) * step
    z_ceil = math.ceil(z_max_raw / step) * step

    if z_ceil <= z_floor:
        z_ceil = z_floor + step

    z_floor = float(np.round(z_floor, 6))
    z_ceil = float(np.round(z_ceil, 6))

    # Палитра / дискретная нормализация
    colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    cmap.set_bad((0, 0, 0, 0))
    levels = np.round(np.arange(z_floor, z_ceil + step / 2.0, step), 6)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Пер-лицевые цвета (ячейки), чтобы не рисовать "половинки"
    def make_facecolors(Z: np.ndarray):
        # узлы внутри полигона и валидные
        finite = np.isfinite(Z)
        cell_inside = (
            mask_nodes[:-1, :-1] & mask_nodes[1:, :-1] & mask_nodes[:-1, 1:] & mask_nodes[1:, 1:]
        )
        cell_finite = finite[:-1, :-1] & finite[1:, :-1] & finite[:-1, 1:] & finite[1:, 1:]
        cell_mask = cell_inside & cell_finite  # ячейка валидна, если ВСЕ 4 угла внутри и не NaN

        # безопасное среднее по 4 углам (без предупреждений)
        s00 = Z[:-1, :-1]
        s10 = Z[1:, :-1]
        s01 = Z[:-1, 1:]
        s11 = Z[1:, 1:]
        stack = np.stack([s00, s10, s01, s11], axis=0)
        sum_ = np.nansum(stack, axis=0)
        cnt = np.sum(np.isfinite(stack), axis=0)
        zc = np.divide(sum_, cnt, out=np.full_like(sum_, np.nan), where=cnt > 0)

        fc = cmap(norm(zc))
        fc[..., 3] = alpha

        # полностью скрываем невалидные/внешние ячейки
        fc[~cell_mask, 3] = 0.0

        # мягкий край: приглушаем альфу у ячеек кольца
        ring_nodes = mask_nodes & ~binary_erosion(mask_nodes, structure=np.ones((3, 3), dtype=bool))
        ring_cells = (
            ring_nodes[:-1, :-1] | ring_nodes[1:, :-1] | ring_nodes[:-1, 1:] | ring_nodes[1:, 1:]
        )
        fc[ring_cells & cell_mask, 3] = np.minimum(fc[ring_cells & cell_mask, 3], alpha * 0.35)
        return fc

    fc1 = make_facecolors(Z1)
    fc2 = make_facecolors(Z2) if (has_second and Z2 is not None) else None

    # Фигура/оси
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=40, azim=40)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Поверхности (пер-лицевые цвета)
    surf1 = ax.plot_surface(
        grid_x,
        grid_y,
        Z1,
        facecolors=fc1,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        shade=False,
        rcount=grid_x.shape[0],
        ccount=grid_x.shape[1],
    )
    surf1.set_zsort("min")

    if has_second and Z2 is not None and fc2 is not None:
        surf2 = ax.plot_surface(
            grid_x,
            grid_y,
            Z2,
            facecolors=fc2,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            shade=False,
            rcount=grid_x.shape[0],
            ccount=grid_x.shape[1],
        )
        surf2.set_zsort("min")

    # Подписи точек, как у вас (оставил без изменений, с плашками)
    val_dec = 2 if step < 0.1 else 1
    val_fmt = f"{{:.{val_dec}f}}"
    box1 = {
        "boxstyle": "round,pad=0.15,rounding_size=0.10",
        "fc": "white",
        "lw": 0.8,
        "alpha": 0.75,
    }
    box2 = {
        "boxstyle": "round,pad=0.15,rounding_size=0.10",
        "fc": "#f3f3f3",
        "lw": 0.8,
        "alpha": 0.75,
    }

    label_mode1 = "below"
    label_mode2 = "above" if has_second else "below"
    xy_off = 0.8
    z_off = max(0.6 * step, 0.05)

    def _offset_point(p, mode: str, xy_off: float, z_off: float):
        x, y, z = p
        ha = va = "center"

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

    # Colorbar/оси
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_clim(z_floor, z_ceil)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10, pad=0.05, boundaries=levels, ticks=levels)
    fmt = mticker.FormatStrFormatter("%.2f" if step < 0.1 else "%.1f")
    cbar.formatter = fmt
    cbar.update_ticks()
    cbar.set_label(f"Диапазон: {z_floor:.2f} … {z_ceil:.2f} (шаг {step:g})")

    # Стены/контуры
    wall_alpha = 0.1
    wall_color = "gray"
    left_points = np.array([(5, -5), (5, 5), (0, 5), (0, 25), (5, 25)])
    right_points = np.array([(20, 25), (25, 25), (25, 5), (20, 5), (20, -5)])
    arc = np.linspace(0, np.pi, 100)
    arc_x = 12.5 + 7.5 * np.cos(arc)
    arc_y = 25 + 7.5 * np.sin(arc)
    arc_points = np.column_stack((arc_x, arc_y))

    def _extrude_segments(pts: np.ndarray):
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
