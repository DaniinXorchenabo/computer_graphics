import math
from colorsys import hls_to_rgb
from dataclasses import dataclass

rounded = 7

@dataclass
class Vertex:
    x: float
    y: float
    z: float
    radians: float

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        elif item == 3:
            return self.radians
        raise KeyError(item)

    def colors(self):
        r,g,b = hls_to_rgb(
            (self.radians * (180 / math.pi)) % 360 / 360,
            (1 + min(-self.z, 0.6) + 0.4) / 2,
            (1 - ((( self.x) / 2)**2 + ((0 + self.y) / 2)**2) ** 0.5) * 0.9 + 0.1
        )
        return round(r, 3), round(g, 3), round(b, 3), 1.0
@dataclass
class Polygon:
    a: Vertex
    b: Vertex
    c: Vertex
    # def colors(x, y, z):
    #     hls_to_rgb(y * 360, l, s)

    def __getitem__(self, item):
        if item == 0:
            return self.a
        elif item == 1:
            return self.b
        elif item == 2:
            return self.c
        raise KeyError()

    def __repr__(self):
        coords = ', '.join([f'[{round(i.x, rounded)}, {round(-i.z, rounded)}, {round(i.y, rounded)}]'
                            for i in [self.a, self.b, self.c]])
        colors = ', '.join(f'[{r}, {g}, {b}, {a}]' for (r,g,b,a) in
                           [self.a.colors(), self.b.colors(), self.c.colors()])
        return f"Vertex::new([" \
               f"{coords}" \
               f"],None, " \
               f"Some([{colors}])" \
               f", None, None, None)"


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
    res =  Vertex(*max(
        plus(mul(mul([A1, B1, C1], target_len), 1/leangh), base_point),
        plus(mul(mul([A1, B1, C1], -target_len), 1 / leangh), base_point),
        key=lambda i: (i[0]*i[0] + i[1]*i[1] + i[2]*i[2])
    ), 0.0)
    # print(res)
    return  res
    # return [A1 * target_len / leangh + base_point[0], B1 * target_len / leangh + base_point[1],
    #         C1 * target_len / leangh + base_point[2]]


triangles = []
radius = 0.115
z = 0.486
center = Vertex(0.0, 0.0, 0.5, 0.0)
level1 = [Vertex(math.cos(i) * radius, math.sin(i) * radius, z, i) for i in
          [i * (360 / 16) * math.pi / 180 for i in range(0, 16)]]
triangles += [Polygon(level1[i % 16], level1[(i + 1) % 16], center) for i in range(16)]
z = 0.4
radius = 0.3
level2 = [Vertex(math.cos(i) * radius, math.sin(i) * radius, z, i) for i in
          [((i * (360 / 16)) + 360 / 32) * math.pi / 180 for i in range(0, 16)]]
triangles += [Polygon(level1[(i + 1) % 16], level1[(i + 0) % 16],  level2[(i) % 16]) for i in range(16)]
triangles += [Polygon(level2[i % 16], level2[(i + 1) % 16], level1[(i + 1) % 16]) for i in range(16)]
z = 0.196
radius = 0.46
level3 = [Vertex(math.cos(i) * radius, math.sin(i) * radius, z, i) for i in
          [i * (360 / 16) * math.pi / 180 for i in range(0, 16)]]
triangles += [Polygon(level3[i % 16], level3[(i + 1) % 16], level2[i % 16]) for i in range(16)]
triangles += [Polygon( level2[(i + 1) % 16], level2[i % 16], level3[(i + 1) % 16]) for i in range(16)]

z = 0.0
radius = 0.949
level4 = [Vertex(math.cos(i) * radius, math.sin(i) * radius, z, i) for i in
          [((i * (360 / 16)) + 360 / 32) * math.pi / 180 for i in range(0, 16)]]
triangles += [Polygon(level3[(i + 1) % 16], level3[(i + 0) % 16],  level4[(i) % 16]) for i in range(16)]
triangles += [Polygon(level4[i % 16], level4[(i + 1) % 16], level3[(i + 1) % 16]) for i in range(16)]

z = -0.196
radius = 0.46
level5 = [Vertex(math.cos(i) * radius, math.sin(i) * radius, z, i) for i in
          [i * (360 / 16) * math.pi / 180 for i in range(0, 16)]]
triangles += [Polygon(level5[i % 16], level5[(i + 1) % 16], level4[i % 16]) for i in range(16)]
triangles += [Polygon( level4[(i + 1) % 16], level4[i % 16], level5[(i + 1) % 16]) for i in range(16)]

z = -0.4
radius = 0.3
level6 = [Vertex(math.cos(i) * radius, math.sin(i) * radius, z, i) for i in
          [((i * (360 / 16)) + 360 / 32) * math.pi / 180 for i in range(0, 16)]]
tr_6 = [Polygon(level5[(i + 1) % 16], level5[(i + 0) % 16],  level6[(i) % 16]) for i in range(16)]
triangles += tr_6
triangles += [Polygon(level6[i % 16], level6[(i + 1) % 16], level5[(i + 1) % 16]) for i in range(16)]

triangles += [
    Polygon(
        Vertex(*plus(mul(minus(median, tr_6[i][ii % 3]), 0.5), tr_6[i][ii % 3]), 0),
        Vertex(*plus(mul(minus(median, tr_6[i][(ii + 1) % 3]), 0.5), tr_6[i][(ii + 1) % 3]), 0),
        point
    )
    for i in range(1, 16, 4) if (point := normal(tr_6[i], 0.4, (median := (
        (tr_6[i].a.x + tr_6[i].b.x + tr_6[i].c.x) / 3,
        (tr_6[i].a.y + tr_6[i].b.y + tr_6[i].c.y) / 3,
        (tr_6[i].a.z+ tr_6[i].b.z + tr_6[i].c.z) / 3
    )))) for ii in range(3)
]

radius = 0.115
z = -0.486
level7 = [Vertex(math.cos(i) * radius, math.sin(i) * radius, z, i) for i in
          [i * (360 / 16) * math.pi / 180 for i in range(0, 16)]]
triangles += [Polygon(level7[i % 16], level7[(i + 1) % 16], level6[i % 16]) for i in range(16)]
triangles += [Polygon(level6[(i + 1) % 16], level6[i % 16],  level7[(i + 1) % 16]) for i in range(16)]
center = Vertex(0.0, 0.0, -0.5, 0.)
triangles += [Polygon(level7[i % 16], level7[(i + 1) % 16], center) for i in range(16)]

# triangles = [[(round(x, 7), round(y, 7), round(z, 7)) for [x, y, z] in i] for i in triangles]
data_for_rust = ',\n'.join(
    [f"{pol.__repr__()}" for pol in triangles])

print(data_for_rust)
