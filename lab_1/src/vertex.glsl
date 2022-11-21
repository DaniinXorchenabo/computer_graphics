
#version 450

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);
layout (constant_id = 0) const int WIGHT = 64;
layout (constant_id = 1) const int HEIGHT = 64;

layout(location = 0) in vec2[3] position;
layout(location = 3) in vec3[3] move_matrix;
layout(location = 6) in float[3] contour;
layout(location = 9) in vec4[3] contour_colors;
layout(location = 12) in vec4[3] point_colors;

layout(location = 15) out vec4 fragColor;
layout(location = 16) out vec3 contour_size;

layout(location = 19) out float points[3][2] ;
layout(location = 25) out vec4[3] contour_colors_fr;




void main() {
    float c_x = position[gl_VertexIndex % 3].x;
    float c_y = position[gl_VertexIndex % 3].y;

    points[ 0 ][0] = ((position[0].x + 1) / 2) * WIGHT ;
    points[ 0 ][1] = ((1 * position[0].y + 1) / 2) * HEIGHT;
    points[ 1 ][0] = ((position[1].x + 1) / 2) * WIGHT ;
    points[ 1 ][1] = ((1 * position[1].y + 1) / 2) * HEIGHT;
    points[ 2 ][0] = ((position[2].x + 1) / 2) * WIGHT ;
    points[ 2 ][1] = ((1 * position[2].y + 1) / 2) * HEIGHT;

    gl_Position = vec4(c_x, c_y, 0.0, 1.0);
    contour_size = vec3(contour[0], contour[1], contour[2]);
    contour_colors_fr = contour_colors;

    fragColor = point_colors[ gl_VertexIndex % 3 ];

}
