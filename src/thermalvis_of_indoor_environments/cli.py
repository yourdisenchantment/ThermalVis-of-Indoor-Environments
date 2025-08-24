# src/thermalvis_of_indoor_environments/cli.py

from __future__ import annotations

import argparse

try:
    from .app import run_visualization
except ImportError:
    # когда cli.py запускают как скрипт, относительный импорт не сработает
    from thermalvis_of_indoor_environments.app import run_visualization


def _prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not raw and default is not None:
            return float(default)
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("Введите число (можно с точкой или запятой).")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="thermalvis",
        description="Визуализация температурной поверхности по температурам четырёх стен (возможны два набора: день/вечер).",
    )

    g1 = parser.add_argument_group("Набор 1 (обязательный)")
    g1.add_argument("-n", "--north", type=float, help="Температура на северной стене (набор 1)")
    g1.add_argument("-w", "--west", type=float, help="Температура на западной стене (набор 1)")
    g1.add_argument("-s", "--south", type=float, help="Температура на южной стене (набор 1)")
    g1.add_argument("-e", "--east", type=float, help="Температура на восточной стене (набор 1)")

    g2 = parser.add_argument_group("Набор 2 (опционально)")
    g2.add_argument("-N", "--north2", type=float, help="Температура на северной стене (набор 2)")
    g2.add_argument("-W", "--west2", type=float, help="Температура на западной стене (набор 2)")
    g2.add_argument("-S", "--south2", type=float, help="Температура на южной стене (набор 2)")
    g2.add_argument("-E", "--east2", type=float, help="Температура на восточной стене (набор 2)")

    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.1,
        help="Минимальный вертикальный зазор между поверхностями (например, 0.05 или 0.1).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4, help="Прозрачность поверхностей (0..1)."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Не вызывать plt.show() (полезно в CI/сервере)."
    )
    parser.add_argument("--save", type=str, help="Путь для сохранения изображения (PNG/PDF/SVG).")

    args = parser.parse_args()

    # Набор 1 - спрашиваем интерактивно всё, что не передано
    north = (
        args.north
        if args.north is not None
        else _prompt_float("Температура на северной стене (набор 1)")
    )
    west = (
        args.west
        if args.west is not None
        else _prompt_float("Температура на западной стене (набор 1)")
    )
    south = (
        args.south
        if args.south is not None
        else _prompt_float("Температура на южной стене (набор 1)")
    )
    east = (
        args.east
        if args.east is not None
        else _prompt_float("Температура на восточной стене (набор 1)")
    )

    # Набор 2 - если задан хоть один аргумент, спросим недостающие
    has_any_second = any(v is not None for v in (args.north2, args.west2, args.south2, args.east2))
    if has_any_second:
        north2 = (
            args.north2
            if args.north2 is not None
            else _prompt_float("Температура на северной стене (набор 2)")
        )
        west2 = (
            args.west2
            if args.west2 is not None
            else _prompt_float("Температура на западной стене (набор 2)")
        )
        south2 = (
            args.south2
            if args.south2 is not None
            else _prompt_float("Температура на южной стене (набор 2)")
        )
        east2 = (
            args.east2
            if args.east2 is not None
            else _prompt_float("Температура на восточной стене (набор 2)")
        )
    else:
        north2 = west2 = south2 = east2 = None

    fig, _ax = run_visualization(
        north=north,
        west=west,
        south=south,
        east=east,
        north2=north2,
        west2=west2,
        south2=south2,
        east2=east2,
        show=not args.no_show,
        min_gap=args.min_gap,
        alpha=args.alpha,
    )

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
