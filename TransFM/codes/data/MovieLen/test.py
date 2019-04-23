def func(signal,current_x = 0, current_y=0, current_direction=2):
    directions = ['E','N','W','S']
    signals = ['L','R','M']
    current_direction = current_direction
    if signals.index(signal) <= 1:
        current_direction = (current_direction + (signals.index(signal) * 2 + 1)) % len(directions)
    else:
        if current_direction % 2 ==0:
            current_x += int((complex(0, 1) ** current_direction).real)
        else:
            current_y += int((complex(0, 1) ** (current_direction-1)).real)
    return current_x, current_y, current_direction


if __name__ == '__main__':
    signals = 'MTMPRPMTMLMRPRMTPLMMTLMRRMP'
    position = [11, 39]
    current_direction = ['E','N','W','S'].index('W')
    for s in signals:
        if s in ['L','R','M']:
            position[0], position[1], current_direction = func(s, position[0], position[1], current_direction)
    print(position,['E','N','W','S'][current_direction ])
