ACTIONS = {0: "up", 1: "down", 2: "left", 3: "right"}


def coord_to_pos(size, coords):
    pos = coords[0] * size[0] + coords[1]
    return pos


def pos_to_coord(size, pos):
    return pos // size[0], pos % size[1]


def day_to_five_min(num_days):
    return num_days * 1440 // 5


if __name__ == '__main__':
    print(coord_to_pos((5, 5), (3, 3)))
    print(pos_to_coord((5, 5), coord_to_pos((5, 5), (3, 3))))
