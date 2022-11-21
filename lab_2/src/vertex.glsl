
#version 450

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);
layout (constant_id = 0) const int WIGHT = 64;
layout (constant_id = 1) const int HEIGHT = 64;

layout(location = 0) in vec2[3] position;
// layout(location = 3) in mat3 move_matrix;
layout(location = 4) out vec4 fragColor;
layout(location = 5) out vec3 contour_size;
layout(location = 6) in float[3] contour;
layout(location = 9) in vec4[3] contour_colors;
layout(location = 12) in vec4[3] point_colors;
layout(location = 15) out mat3 points ;
// layout(location = 19) out float points[3][3] ;
layout(location = 19) out vec4[3] contour_colors_fr;
layout(location = 22) in mat4 move_matrix;



void main() {
    // float c_x = position[gl_VertexIndex % 3].x;
    // float c_y = position[gl_VertexIndex % 3].y;


    float x_mn = float(WIGHT) / float(HEIGHT);
    float y_mn = float(HEIGHT) / float(WIGHT);
    x_mn = max(1.0, x_mn);
    y_mn = max(1.0, y_mn);
    // x_mn = 1.0;
    // y_mn = 1.0;

    vec4 pos0 = vec4(position[0].x /x_mn , position[0].y /y_mn , 0.0, 1.0) * move_matrix ;
    vec4 pos1 = vec4(position[1].x / x_mn, position[1].y / y_mn, 0.0, 1.0) * move_matrix ;
    vec4 pos2 = vec4(position[2].x / x_mn, position[2].y /y_mn, 0.0, 1.0) * move_matrix;

    // vec4 pos_m =  gl_VertexIndex % 3 > 1 ? pos2: (gl_VertexIndex % 3 < 1 ? pos0: pos1);
    vec4 pos_m = vec4(position[gl_VertexIndex % 3].x  /x_mn , position[gl_VertexIndex % 3].y / y_mn , 0.0, 1.0)  * move_matrix ;
    // vec3 pos_m = vec3(c_x, c_y, 1.0)  * move_matrix ;
    //
    // vec3 pos0 = vec3(position[0].x, position[0].y, 1.0) * move_matrix ;
    // vec3 pos1 = vec3(position[1].x, position[1].y, 1.0) * move_matrix ;
    // vec3 pos2 =  vec3(position[2].x, position[2].y, 1.0) * move_matrix;

    // c_x = pos_m.x;
    // c_y = pos_m.y;

    // points = [
    //             [((position[0].x + 1) / 2) * WIGHT, ((1 * position[0].y + 1) / 2) * HEIGHT, 1.0],
    //             [((position[1].x + 1) / 2) * WIGHT, ((1 * position[1].y + 1) / 2) * HEIGHT, 1.0],
    //             [((position[2].x + 1) / 2) * WIGHT, ((1 * position[2].y + 1) / 2) * HEIGHT, 1.0]
    //         ];

    points[ 0 ][0] = ((pos0.x + 1) / 2) * WIGHT ;
    points[ 0 ][1] = ((1 * pos0.y + 1) / 2) * HEIGHT;
    points[ 0 ][2] = 1.0;

    points[ 1 ][0] = ((pos1.x + 1) / 2) * WIGHT ;
    points[ 1 ][1] = ((1 * pos1.y + 1) / 2) * HEIGHT;
    points[ 1 ][2] = 1.0;

    points[ 2 ][0] = ((pos2.x + 1) / 2) * WIGHT ;
    points[ 2 ][1] = ((1 * pos2.y + 1) / 2) * HEIGHT;
    points[ 2 ][2] = 1.0;

    // points = points * move_matrix;


    mat3 test_mat;

    // test_mat[0][0] = 1.0;
    // test_mat[0][1] = 0.0;
    // test_mat[0][2] = 0.0;
    //
    // test_mat[1][0] = 0.0;
    // test_mat[1][1] = 1.0;
    // test_mat[1][2] = 0.0;
    //
    // test_mat[2][0] = 0.0;
    // test_mat[2][1] = 0.0;
    // test_mat[2][2] = 1.0;


    // vec3 pos = test_mat * vec3(c_x, c_y, 1.0);

    // gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);
    gl_Position = vec4(pos_m.x, pos_m.y, 0, 1.0);

    // gl_Position = vec4(c_x, c_y, 0.0, 1.0);
    float board_size_mn =  (move_matrix[0][0] + move_matrix[1][1]) / 2;
    contour_size = vec3(contour[0] == 0.0 ? 0.0: (contour[0] == 1.0 ? 1.0 :max( contour[0] * length(pos0.xyz - pos1.xyz) / length(position[0].xy - position[1].xy), 1.4)),
                        contour[1] == 0.0 ? 0.0: (contour[1] == 1.0 ? 1.0 :max( contour[1] * length(pos1.xyz - pos2.xyz) / length(position[1].xy - position[2].xy), 1.4)),
                        contour[2] == 0.0 ? 0.0: (contour[2] == 1.0 ? 1.0 :max( contour[2] * length(pos2.xyz - pos0.xyz) / length(position[2].xy - position[0].xy), 1.4)));
    // // contour_size = vec3(contour[0], contour[1], contour[2]);

    // contour_size = vec3(contour[0] * length(pos0.xyz - pos1.xyz) / length(position[0].xy - position[1].xy),
    //                      contour[1] * length(pos1.xyz - pos2.xyz) / length(position[1].xy - position[2].xy),
    //                     contour[2] * length(pos2.xyz - pos0.xyz) / length(position[2].xy - position[0].xy)
    //                     );


    contour_colors_fr = contour_colors;

    fragColor = point_colors[ gl_VertexIndex % 3 ];


    // if (pos.x == pos_m.x){
    //     fragColor.x = 0.5;
    // }
    // if (pos.y == pos_m.y){
    //     fragColor.y = 0.5;
    // }
    // if (pos.z == pos_m.z){
    //     fragColor.z = 0.5;
    // }

}
