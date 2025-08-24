# ThermalVis of Indoor Environments

Визуализация 3D‑распределения температуры внутри помещения по температурам четырёх стен. Поддерживаются два набора значений (например, “день” и “вечер”) с:
- общим цветовым диапазоном (удобно сравнивать),
- регулируемой прозрачностью поверхностей,
- гарантированным вертикальным зазором между поверхностями (anti-slip),
- отрисовкой стен и контура пола помещения.

Под капотом: NumPy, SciPy (интерполяция), Matplotlib (3D‑визуализация).

---

## Требования

- Python 3.11–3.12 (рекомендовано; на 3.13 SciPy/Matplotlib могут не иметь готовых колёс).
- Poetry 2.x.
- Графическое окружение для окна Matplotlib. На сервере/CI используйте headless-режим (см. ниже).

Проверка окружения:
```bash
python --version
poetry --version
```

---

## Установка (из исходников)

```bash
git clone <url-вашего-репозитория>
cd thermalvis-of-indoor-environments

# Рекомендуемый Python
poetry env use 3.12

# Установка зависимостей
poetry install
```

---

## Быстрый старт

Запуск из проекта (через Poetry):
- Интерактивно (попросит ввести температуры):
  ```bash
  poetry run thermalvis
  ```
- Один набор значений:
  ```bash
  poetry run thermalvis -n 20.5 -w 20.8 -s 21.1 -e 20.3
  ```
- Два набора (например, “вечер” выше на 0.3):
  ```bash
  poetry run thermalvis \
    -n 20.1 -w 20.9 -s 20.5 -e 20.7 \
    -N 20.4 -W 21.2 -S 20.8 -E 21.0 \
    --min-gap 0.05 --alpha 0.4
  ```
- Сохранить без показа окна (headless):
  - Linux/macOS:
    ```bash
    MPLBACKEND=Agg poetry run thermalvis \
      -n 20.1 -w 20.9 -s 20.5 -e 20.7 \
      -N 20.4 -W 21.2 -S 20.8 -E 21.0 \
      --no-show --save plot.png
    ```
  - Windows (PowerShell):
    ```powershell
    $env:MPLBACKEND="Agg"
    poetry run thermalvis -n 20.1 -w 20.9 -s 20.5 -e 20.7 -N 20.4 -W 21.2 -S 20.8 -E 21.0 --no-show --save plot.png
    ```

Альтернатива: запуск как модуля
```bash
poetry run python -m thermalvis_of_indoor_environments -n 20.5 -w 20.8 -s 21.1 -e 20.3
```

---

## CLI справка

Посмотреть помощь:
```bash
thermalvis --help
# или во время разработки:
poetry run thermalvis --help
```

Доступные опции:
- Набор 1 (обязательный):
  - -n, --north  — северная стена
  - -w, --west   — западная стена
  - -s, --south  — южная стена
  - -e, --east   — восточная стена
- Набор 2 (опционально):
  - -N, --north2
  - -W, --west2
  - -S, --south2
  - -E, --east2
- Визуализация:
  - --min-gap <float>   — минимальный вертикальный зазор между поверхностями (например, 0.05 или 0.1)
  - --alpha <float>     — прозрачность поверхностей (0..1), по умолчанию 0.4
  - --no-show           — не открывать окно (полезно в CI/сервере)
  - --save <path>       — сохранить изображение (PNG/PDF/SVG)

Примечания:
- Если для набора 2 указан хоть один из -N/-W/-S/-E, недостающие значения будут запрошены интерактивно.
- Единицы — градусы (°C), дробные значения через точку: 20.5.

---

## Как это работает

- Вход: температуры в серединах четырёх стен. Угловые точки вычисляются как средние между соседними стенами.
- Интерполяция: регулярная сетка 25×25; метод cubic с фоллбеком на linear.
- Область: значения вне заданного контура помещения маскируются (NaN не рисуются).
- Два набора: обе поверхности нормализуются одной цветовой шкалой (общий vmin/vmax), чтобы сравнение было честным.
- Anti‑slip: если поверхности слишком близко, они симметрично разводятся по Z до зазора min_gap.
- Ось Z и “стены” помещения: диапазон берётся по общим значениям, расширяется до int(min)−1 и int(max)+1.

---

## Структура проекта

```
thermalvis-of-indoor-environments/
├─ pyproject.toml
├─ README.md
├─ src/
│  └─ thermalvis_of_indoor_environments/
│     ├─ __init__.py
│     ├─ __main__.py
│     ├─ app.py        # логика визуализации (поддержка 1–2 наборов)
│     └─ cli.py        # CLI: парсинг аргументов, сохранение, headless и т.п.
└─ tests/              # опционально (pytest)
```

CLI команда thermalvis определяется в pyproject.toml (project.scripts).

---

## Сборка и запуск установленного приложения

Собрать пакет:
```bash
poetry build
# dist/*.whl и *.tar.gz
```

Установить и запустить (варианты):
- В отдельное venv:
  ```bash
  python -m venv .venv-run && . .venv-run/bin/activate  # Windows: .venv-run\Scripts\activate
  pip install dist/thermalvis_of_indoor_environments-*.whl
  thermalvis --help
  ```
- Через pipx (рекомендуется для CLI):
  ```bash
  pipx install dist/thermalvis_of_indoor_environments-*.whl
  thermalvis --help
  ```

---

## Сборка бинарника (exe) через PyInstaller

Добавить dev-зависимость (с ограничением по Python):
```bash
poetry add --group dev 'pyinstaller@^6.15; python_version < "3.15"'
```

Сборка:
- Один файл (one‑file):
  ```bash
  poetry run pyinstaller -F -n thermalvis --console -p src \
    --collect-all matplotlib --collect-all scipy \
    src/thermalvis_of_indoor_environments/cli.py
  ```
- Папка (one‑dir, надёжнее):
  ```bash
  poetry run pyinstaller -D -n thermalvis --console -p src \
    --collect-all matplotlib --collect-all scipy \
    src/thermalvis_of_indoor_environments/cli.py
  ```

Запуск:
- Linux/macOS: ./dist/thermalvis (или ./dist/thermalvis/thermalvis)
- Windows: .\dist\thermalvis.exe (или .\dist\thermalvis\thermalvis.exe)

Замечания:
- Собирать нужно на той же ОС/архитектуре, где запускать.
- С GUI‑бэкендом Matplotlib размер бинарника будет большим (это нормально).
- На сервере/WSL используйте headless (Agg) и --no-show, либо убедитесь, что установлен Tk/Qt.
- При повторной сборке на всякий случай можно добавить флаг `--clean`.

---

## Разработка

- Линт и формат (если используете Ruff):
  ```bash
  poetry run ruff check .
  poetry run ruff check . --fix
  poetry run ruff format
  ```
- Тесты (если используете pytest):
  ```bash
  poetry run pytest -q
  ```

Полезные команды Poetry:
```bash
poetry check
poetry lock
poetry install
poetry show --tree
poetry env use 3.12
```

---

## Частые проблемы

- “Command not found: thermalvis”:
  - Установите пакет или используйте poetry run thermalvis.
  - Проверьте PATH (venv/bin или ~/.local/bin).
- “pyproject.toml changed significantly…”:
  - Выполните: poetry lock, затем poetry install.
- Ошибки SciPy/Matplotlib при установке:
  - Используйте Python 3.11–3.12.
- “No display name and no $DISPLAY”:
  - Запускайте в headless-режиме: установить MPLBACKEND=Agg и флаг --no-show.

---

## Лицензия

MIT
