import json

def update_json_file(filepath, new_data, new_end_data):
    # Читаем файл
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Обновляем значения по ключам
    # print(new_end_data)
    # data["_data"] = ''
    data["_Data"] = new_data
    data["end_data"] = new_end_data

    # Записываем обратно в файл
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


with open("pointsGon.txt", "r", encoding="utf-8") as f:
    data = f.read()
    end_data = (data.replace(",", "")).replace("nan",'-nan(ind)')
    # print(repr(end_data))
    # end_data = end_data.replace("\\n", "\n")
    # print(repr(end_data))
    # print(end_data[-100:])
    new_data = end_data.split("\\n")[:-1]
    # new_data.sort(key=lambda p: (float((p.split(" "))[0]), float((p.split(" "))[1])))
    # print(*new_data[:10], sep='\n')
    # print(*new_data[-10:], sep='\n')
    # print(end_data)
    params_list = []
    print(len(new_data))
    count = 0
    count1 = 0
    old = float((new_data[0].split(" "))[0])
    # old1 = float((new_data[0].split(" "))[1])
    initial1 = []
    for elements in new_data:
        # if count == 498:
        #     continue

        tmp = elements.split(" ")
        if float(tmp[0])!=old:
            old = float(tmp[0])
            print(count)
            count = 0
        else:
            count += 1

        # if float(tmp[1])!=old1:
        #     old1 = float(tmp[1])
        #     print("sec_param",count1)
        #     count1 = 0
        # else:
        #     count1 += 1

        initial1.append(elements)
        params_list.append(tmp[0] + " " + tmp[1] + " " + "0.0" + "\n")
    print("кол во по energy", count)
    print("кол во по delta", count1)
    params = "".join(params_list)
    print(len(params_list))
    initial = "\n".join(initial1) + "\n"
    print(len(initial1))
    # for e in initial1:
    #     if len(e)!=498:
    #         print(len(e))
    # print(repr(initial))
    # print(params)

update_json_file(
    "Continuation_CelticStone_6D_flow_11-10-2025_1111.json",
    new_data=params,
    new_end_data=initial
)
