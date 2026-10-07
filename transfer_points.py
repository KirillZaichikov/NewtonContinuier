import json


def update_json_file(filepath, new_data, new_end_data):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["_Data"] = new_data
    data["end_data"] = new_end_data

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# --------------------------------------------------
# Читаем pointsGon.txt
# --------------------------------------------------

with open("pointsGon.txt", "r", encoding="utf-8") as f:
    raw_data = f.read()

# Если в файле используются буквальные \n
raw_data = raw_data.replace("\\n", "\n")

# nan -> -nan(ind)
raw_data = raw_data.replace("nan", "-nan(ind)")

lines = raw_data.splitlines()

params_list = []
initial1 = []

# 2 параметра
parameters = [[], []]

# 6 фазовых переменных
phase_variables = [[], [], [], [], [], []]


for line in lines:

    line = line.strip()

    if not line:
        continue

    # Разделяем исходную строку по запятым
    tmp = [x.strip() for x in line.split(",")]

    if len(tmp) != 8:
        print("Некорректная строка:", line)
        continue

    # --------------------------------------------------
    # Числовые данные для статистики
    # --------------------------------------------------

    for i in range(2):
        try:
            value = float(tmp[i])

            if value == value:
                parameters[i].append(value)

        except ValueError:
            pass

    for i in range(6):
        try:
            value = float(tmp[i + 2])

            if value == value:
                phase_variables[i].append(value)

        except ValueError:
            pass

    # --------------------------------------------------
    # _Data
    #
    # Только 2 параметра + 6 нулей
    # ЗАПЯТЫХ НЕТ
    # --------------------------------------------------

    params_list.append(
        " ".join([
            tmp[0],
            tmp[1],
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0"
        ]) + "\n"
    )

    # --------------------------------------------------
    # end_data
    #
    # Все 8 значений, но тоже БЕЗ ЗАПЯТЫХ
    # --------------------------------------------------

    initial1.append(" ".join(tmp))


# ==================================================
# СТАТИСТИКА ПАРАМЕТРОВ
# ==================================================

print()
print("=" * 70)
print("ПАРАМЕТРЫ")
print("=" * 70)

for i in range(2):
    if parameters[i]:
        print(
            f"Параметр {i + 1}: "
            f"min = {min(parameters[i]):.15g}, "
            f"max = {max(parameters[i]):.15g}"
        )


# ==================================================
# СТАТИСТИКА ФАЗОВЫХ ПЕРЕМЕННЫХ
# ==================================================

print()
print("=" * 70)
print("ФАЗОВЫЕ ПЕРЕМЕННЫЕ")
print("=" * 70)

for i in range(6):
    if phase_variables[i]:
        print(
            f"Фазовая переменная {i + 1}: "
            f"min = {min(phase_variables[i]):.15g}, "
            f"max = {max(phase_variables[i]):.15g}"
        )


# ==================================================
# РАЗМЕРНОСТЬ СЕТКИ
# ==================================================

param1_values = set(parameters[0])
param2_values = set(parameters[1])

n1 = len(param1_values)
n2 = len(param2_values)

print()
print("=" * 70)
print("РАЗМЕРНОСТЬ СЕТКИ")
print("=" * 70)

print(f"Параметр 1: {n1} значений")
print(f"Параметр 2: {n2} значений")
print(f"Размерность сетки: {n1} x {n2}")
print(f"Количество точек: {len(initial1)}")

if n1 * n2 == len(initial1):
    print("Сетка полная.")
else:
    print(
        f"Внимание: n1 * n2 = {n1 * n2}, "
        f"а точек = {len(initial1)}"
    )


# ==================================================
# ФОРМИРУЕМ ДАННЫЕ
# ==================================================

params = "".join(params_list)
initial = "\n".join(initial1) + "\n"


# ==================================================
# ОБНОВЛЯЕМ JSON
# ==================================================

update_json_file(
    "Lyapunov_CelticStone_6D_flow_30-09-2025_00-00-01.json",
    new_data=params,
    new_end_data=initial
)