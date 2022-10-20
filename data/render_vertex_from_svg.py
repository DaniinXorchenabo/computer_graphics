
ss = ""


def convert_V_abd_G(data: list[str]) -> list[str]:
    res = []
    for ind, item in enumerate(data):
        if 'V' in item:
            normal, transform, *_ = item.split('V')
            res.append(normal)
            res.append(f"{normal.split()[0]} {transform.strip()}")
        elif 'H' in item:
            normal, transform, *_ = item.split('H')
            res.append(normal)
            res.append(f"{transform.strip()} {normal.split()[-1]}")
        else:
            res.append(item)
    return res


data = [
    [
        [
            float(ii) for ii in jj.strip().split()
        ]
        for jj in convert_V_abd_G(i.replace('M', '').replace('Z', '').split('L'))
    ]
    for i in [
        i.split(' d=')[1].split('stroke=')[0].strip().replace('"', '').replace("'", '')
        for i in ss.split('\n') if ' d=' in i
    ]
]
min_x =  min([(jj[0]) for i in data for jj in i])
max_x =  max([(jj[0]) for i in data for jj in i])
min_y =  min([(jj[1]) for i in data for jj in i])
max_y =  max([(jj[1]) for i in data for jj in i])
center = ((291.5 - 271.5) / 2 + 271.5, 169 )
center = (
    (max_x - min_x) * (center[0] - min_x) / (max_x - min_x),
    (max_y - min_y) * (center[1] - min_y) / (max_y - min_y)
)
data = [
    [
        [x - center[0], y - center[1]]
        for [x, y] in pol[:-1]
    ]
    for pol in data
    if len(pol) == 4 or print("Error! polygon has a no correct len {len(pol)}", pol)
]
max_ = max([abs(ii) for i in data for jj in i for ii in jj] )
mn = 0.9 / max_
data = [[[x*mn, y*mn] for [x, y] in pol] for pol in data if len(pol) == 3 or 1 / 0]
data_for_rust = ',\n'.join([f'''Vertex::new([{', '.join([f'[{x}, {y}]' for [x, y] in pol])}],None, None, None, None, None)''' for pol in data])
