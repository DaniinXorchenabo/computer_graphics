import math


def minus(one, two):
    return [one[0] - two[0], one[1] - two[1], one[2] - two[2], ]


def plus(one, two):
    return [one[0] + two[0], one[1] + two[1], one[2] + two[2], ]


def mul(one, number):
    return [one[0] * number, one[1] * number, one[2] * number]


def normal(points, target_len, base_point):
    v_1_0 = minus(points[1], points[0])

    v_2_0 = minus(points[2], points[0])

    A1 = v_1_0[1] * v_2_0[2] - v_1_0[2] * v_2_0[1]

    B1 = v_1_0[2] * v_2_0[0] - v_1_0[0] * v_2_0[2]

    C1 = v_1_0[0] * v_2_0[1] - v_1_0[1] * v_2_0[0]
    leangh = (A1 * A1 + B1 * B1 + C1 * C1) ** 0.5
    return max(
        plus(mul(mul([A1, B1, C1], target_len), 1/leangh), base_point),
        plus(mul(mul([A1, B1, C1], -target_len), 1 / leangh), base_point),
        key=lambda i: (i[0]*i[0] + i[1]*i[1] + i[2]*i[2])
    )
    # return [A1 * target_len / leangh + base_point[0], B1 * target_len / leangh + base_point[1],
    #         C1 * target_len / leangh + base_point[2]]


triangles = []
radius = 0.115
z = 0.486
center = (0.0, 0.0, 0.5)
level1 = [(math.cos(i) * radius, math.sin(i) * radius, z) for i in
          [i * (360 / 16) * math.pi / 180 for i in range(0, 16)]]
triangles += [(level1[i % 16], level1[(i + 1) % 16], center) for i in range(16)]
z = 0.4
radius = 0.3
level2 = [(math.cos(i) * radius, math.sin(i) * radius, z) for i in
          [((i * (360 / 16)) + 360 / 32) * math.pi / 180 for i in range(0, 16)]]
triangles += [(level1[(i + 0) % 16], level1[(i + 1) % 16], level2[(i) % 16]) for i in range(16)]
triangles += [(level2[i % 16], level2[(i + 1) % 16], level1[(i + 1) % 16]) for i in range(16)]
z = 0.196
radius = 0.46
level3 = [(math.cos(i) * radius, math.sin(i) * radius, z) for i in
          [i * (360 / 16) * math.pi / 180 for i in range(0, 16)]]
triangles += [(level3[i % 16], level3[(i + 1) % 16], level2[i % 16]) for i in range(16)]
triangles += [(level2[i % 16], level2[(i + 1) % 16], level3[(i + 1) % 16]) for i in range(16)]

z = 0.0
radius = 0.949
level4 = [(math.cos(i) * radius, math.sin(i) * radius, z) for i in
          [((i * (360 / 16)) + 360 / 32) * math.pi / 180 for i in range(0, 16)]]
triangles += [(level3[(i + 0) % 16], level3[(i + 1) % 16], level4[(i) % 16]) for i in range(16)]
triangles += [(level4[i % 16], level4[(i + 1) % 16], level3[(i + 1) % 16]) for i in range(16)]

z = -0.196
radius = 0.46
level5 = [(math.cos(i) * radius, math.sin(i) * radius, z) for i in
          [i * (360 / 16) * math.pi / 180 for i in range(0, 16)]]
triangles += [(level5[i % 16], level5[(i + 1) % 16], level4[i % 16]) for i in range(16)]
triangles += [(level4[i % 16], level4[(i + 1) % 16], level5[(i + 1) % 16]) for i in range(16)]

z = -0.4
radius = 0.3
level6 = [(math.cos(i) * radius, math.sin(i) * radius, z) for i in
          [((i * (360 / 16)) + 360 / 32) * math.pi / 180 for i in range(0, 16)]]
tr_6 = [(level5[(i + 0) % 16], level5[(i + 1) % 16], level6[(i) % 16]) for i in range(16)]
triangles += tr_6
triangles += [(level6[i % 16], level6[(i + 1) % 16], level5[(i + 1) % 16]) for i in range(16)]

triangles += [
    (
        plus(mul(minus(median, tr_6[i][ii % 3]), 0.5), tr_6[i][ii % 3]),
        plus(mul(minus(median, tr_6[i][(ii + 1) % 3]), 0.5), tr_6[i][(ii + 1) % 3]),
        point
    )
    for i in range(1, 16, 4) if (point := normal(tr_6[i], 0.4, (median := (
        (tr_6[i][0][0] + tr_6[i][1][0] + tr_6[i][2][0]) / 3,
        (tr_6[i][0][1] + tr_6[i][1][1] + tr_6[i][2][1]) / 3,
        (tr_6[i][0][2] + tr_6[i][1][2] + tr_6[i][2][2]) / 3
    )))) for ii in range(3)
]

radius = 0.115
z = -0.486
level7 = [(math.cos(i) * radius, math.sin(i) * radius, z) for i in
          [i * (360 / 16) * math.pi / 180 for i in range(0, 16)]]
triangles += [(level7[i % 16], level7[(i + 1) % 16], level6[i % 16]) for i in range(16)]
triangles += [(level6[i % 16], level6[(i + 1) % 16], level7[(i + 1) % 16]) for i in range(16)]
center = (0.0, 0.0, -0.5)
triangles += [(level7[i % 16], level7[(i + 1) % 16], center) for i in range(16)]

triangles = [[(round(x, 7), round(y, 7), round(z, 7)) for [x, y, z] in i] for i in triangles]
data_for_rust = ',\n'.join(
    [f"Vertex::new([{', '.join([f'[{x}, {-z}, {y}]' for [x, y, z] in pol])}],None, None, None, None, None)"
     for pol in triangles])

print(data_for_rust)
