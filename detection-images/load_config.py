import os

def load_config():
    file_path=os.path.join(os.path.dirname(__file__),'config.txt')

    type_map = {1: 1, 2: 2, 3: 3, 4: 4, 10: 5, 11: 6, 12: 7, 13: 8}

    file = open(file_path, 'r', encoding='utf-8')

    line = file.readline()
    boundaries = tuple(map(lambda x: tuple(map(float, x.split(','))), line.split(' ')))

    types_limit = {}
    line = file.readline()
    while line:
        line = line.split(',')
        type = type_map[int(line.pop(0))]
        types_limit[type] = tuple([loc[:2] for loc in line])

        line = file.readline()

    file.close()

    return boundaries, types_limit
