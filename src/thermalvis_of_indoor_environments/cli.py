# src/thermalvis_of_indoor_environments/cli.py

from __future__ import annotations

import argparse
import sys

from .app import run_visualization


def _readline_any(prompt: str) -> str:
    print(prompt + ": ", end="", flush=True)
    data = sys.stdin.buffer.readline()
    for enc in (getattr(sys.stdin, "encoding", None), "utf-8", "cp1251"):
        if not enc:
            continue
        try:
            return data.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore").strip()


def _prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        raw = _readline_any(
            f"{prompt}" + (f" [{default}]" if default is not None else "")
        )
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("Введите число (можно с точкой или запятой).")


def _prompt_int(
    prompt: str, default: int = 1, choices: tuple[int, ...] = (1, 2)
) -> int:
    while True:
        raw = _readline_any(f"{prompt} [{default}]")
        if raw == "":
            return default
        try:
            v = int(raw)
            if v in choices:
                return v
        except ValueError:
            pass
        print(f"Введите одно из значений: {', '.join(map(str, choices))}.")


def _split_groups(tokens: list[str]) -> list[list[str]]:
    groups: list[list[str]] = [[]]
    for tok in tokens:
        if tok == "\\":
            if groups[-1]:
                groups.append([])
            else:
                # пропускаем повторные разделители
                continue
        else:
            groups[-1].append(tok)
    return [g for g in groups if g]


def _parse_set_args(tokens: list[str]) -> tuple[float, float, float, float]:
    sp = argparse.ArgumentParser(add_help=False)
    sp.add_argument("-n", "--north", type=float, required=True)
    sp.add_argument("-w", "--west", type=float, required=True)
    sp.add_argument("-s", "--south", type=float, required=True)
    sp.add_argument("-e", "--east", type=float, required=True)
    ns = sp.parse_args(tokens)
    return ns.north, ns.west, ns.south, ns.east


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="thermalvis",
        description="Визуализация температурной поверхности по температурам четырёх стен (1-2 набора).",
    )
    # Базовые (не наборные) аргументы
    parser.add_argument(
        "-t", "--type", type=int, choices=[1, 2], help="Количество графиков (1 или 2)."
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.1,
        help="Минимальный вертикальный зазор между поверхностями (например, 0.05 или 0.1).",
    )
    parser.add_argument(
        "--alpha",
        "-alpha",
        type=float,
        default=0.4,
        help="Прозрачность поверхностей (0..1).",
    )
    parser.add_argument(
        "--tick-step",
        type=float,
        default=0.1,
        help="Шаг делений по оси Z и на цветовой шкале (например, 0.1 или 0.05).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Не вызывать plt.show() (полезно в CI/сервере).",
    )
    parser.add_argument(
        "--save", type=str, help="Путь для сохранения изображения (PNG/PDF/SVG)."
    )
    parser.add_argument(
        "--grid-res",
        type=int,
        default=27,
        help="Число узлов по каждой оси (включая оверскан). Минимум 3. Например, 27 (шаг ~1), 51 (шаг ~0.5), 101 (шаг ~0.25).",
    )

    # Парсим только базовые и оставляем "наборные" флаги для отдельного разбора
    args, rest = parser.parse_known_args()

    # РЕЖИМ 1: интерактивный (нет наборных флагов)
    if not rest:
        t = (
            args.type
            if args.type is not None
            else _prompt_int(
                "Сколько графиков строить (1/2)", default=1, choices=(1, 2)
            )
        )

        # Набор 1 - порядок точек: т.1 запад, т.2 север, т.3 юг, т.4 восток
        west = _prompt_float("Введите т.1 (запад)")
        north = _prompt_float("Введите т.2 (север)")
        south = _prompt_float("Введите т.3 (юг)")
        east = _prompt_float("Введите т.4 (восток)")

        north2 = west2 = south2 = east2 = None
        if t == 2:
            west2 = _prompt_float("Введите т.1 (запад) - набор 2")
            north2 = _prompt_float("Введите т.2 (север) - набор 2")
            south2 = _prompt_float("Введите т.3 (юг) - набор 2")
            east2 = _prompt_float("Введите т.4 (восток) - набор 2")

        # Опционально: шаг делений и прозрачность (Enter - оставить по умолчанию)
        tick_step = _prompt_float(
            "Расстояние между делениями (например, 0.1 или 0.05)",
            default=args.tick_step,
        )
        alpha = _prompt_float("Прозрачность поверхностей (0..1)", default=args.alpha)

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
            alpha=alpha,
            tick_step=tick_step,
            grid_res=max(
                3, int(args.grid_res)
            ),  # защита от слишком маленького значения
        )
        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
        return

    # РЕЖИМ 2: CLI с разделителем наборов "\"
    groups = _split_groups(rest)
    t = args.type if args.type is not None else min(2, len(groups)) if groups else 1
    if t not in (1, 2):
        print("Ошибка: допустимые значения для -t/--type: 1 или 2.", file=sys.stderr)
        sys.exit(2)

    if not groups:
        print(
            "Ошибка: передайте набор(ы) значений через -n -w -s -e и разделите группы символом '\\' (если графиков 2).",
            file=sys.stderr,
        )
        sys.exit(2)

    # Парсим наборы по группам
    try:
        n1, w1, s1, e1 = _parse_set_args(groups[0])
        n2 = w2 = s2 = e2 = None
        if t == 2:
            if len(groups) < 2:
                print(
                    "Ошибка: указан -t 2, но передан только один набор. Используйте разделитель '\\' для второго набора.",
                    file=sys.stderr,
                )
                sys.exit(2)
            n2, w2, s2, e2 = _parse_set_args(groups[1])
    except SystemExit:
        # пробрасываем стандартную ошибку argparse
        raise

    fig, _ax = run_visualization(
        north=n1,
        west=w1,
        south=s1,
        east=e1,
        north2=n2,
        west2=w2,
        south2=s2,
        east2=e2,
        show=not args.no_show,
        min_gap=args.min_gap,
        alpha=args.alpha,
        tick_step=args.tick_step,
        grid_res=max(3, int(args.grid_res)),
    )
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
