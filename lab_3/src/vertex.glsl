
#version 450

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);
layout (constant_id = 0) const int WIGHT = 64;
layout (constant_id = 1) const int HEIGHT = 64;

layout(location = 0) in mat4 position;
// layout(location = 3) in mat3 move_matrix;
layout(location = 4) out vec4 fragColor;
layout(location = 5) out vec3 contour_size;
layout(location = 6) in float[3] contour;
layout(location = 9) in vec4[3] contour_colors;
layout(location = 12) in vec4[3] point_colors;
layout(location = 15) out mat4 points ;

layout(location = 19) out vec4[3] contour_colors_fr;
layout(location = 22) in mat4 move_matrix;
layout(location = 26) in int projection_flag;



void main() {

    float x_mn_raw = float(WIGHT) / float(HEIGHT);
    float y_mn_raw = float(HEIGHT) / float(WIGHT);

    float x_mn = max(1.0, x_mn_raw);
    float y_mn = max(1.0, y_mn_raw);
    float z_mn = sqrt(x_mn * y_mn);
    z_mn = max(x_mn, y_mn);


    mat4 poses = matrixCompMult(position * move_matrix,  mat4(
        1.0 / x_mn, 1.0 / y_mn, 1.,     1.,
        1.0 / x_mn, 1.0 / y_mn, 1.,     1.,
        1.0 / x_mn, 1.0 / y_mn, 1.,     1.,
        1.0,        1.0,        1.,     1.
    ))  ;

    float fovy = radians(90.) ; // Угол обзора нужен чтобы указать как много объектов попадает на канвас от точки с которой мы смотрим. Не понятно? Тогда проще — чем больше угол мы передадим, тем меньше объекты становятся при удалении. Диапазон углов лучше использовать от 1 до 179.
    float aspect = float(HEIGHT) / float(WIGHT);
    float near = -0.1; //этими параметрами мы подгоняем координаты Z у моделей так, чтобы можно было определить какие модели слишком близко к нам, а какие слишком далеко (Z будет в диапазоне от -1 до 1 после преобразований перспективы), настолько что нам их не нужно рисовать на экране.
    float far = -1000.;
    mat4 perspective_projection = mat4(
        1.0 / tan(fovy / 2.) / aspect, 0.,                      0.,                            0.,
        0.,                             1.0 / tan(fovy / 2.),   0.,                             0.,
        0.,                             0.,                     (far + near) / (far - near),    (-2. * far * near) / (far - near),
        0.,                             0.,                     -1.,                            0.
    );


    mat4 resize_mat =  mat4(
        1. / x_mn,  0.,         0.,         0.,
        0.,         1. / y_mn,  0.,         0.,
        0.,         0.,         1.,         0.,
        0.,         0.,         0.,         1.
    );

    mat4 _pos_m = resize_mat * ( perspective_projection * ((move_matrix * ( position )))) ;

    _pos_m = matrixCompMult(_pos_m , mat4(
        1./_pos_m[0].w / x_mn,  1./_pos_m[0].w / y_mn,  1./_pos_m[0].w, 1./_pos_m[0].w,
        1./_pos_m[1].w / x_mn,  1./_pos_m[1].w / y_mn,  1./_pos_m[1].w, 1./_pos_m[1].w,
        1./_pos_m[2].w / x_mn,  1./_pos_m[2].w / y_mn,  1./_pos_m[2].w, 1./_pos_m[2].w,
        1.,                     1.,                     1.,             1.
    ));

    vec4 pos_m = _pos_m[gl_VertexIndex % 3];

    mat4 projection_mat;
    if (projection_flag == 1){
        projection_mat = mat4(
            0.707,  -0.408, 0., 0.,
            0.,     0.816,  0., 0.,
            -0.707, -0.408, 0., 0.,
            0.,     0.,     0., 1.
        );
    } else {
        projection_mat = mat4(
            1., 0., 0., 0.,
            0., 1., 0., 0.,
            0., 0., 0., 0.,
            0., 0., 0., 1.
        );
    }


    mat4 disp_pos_m =  ( projection_mat *  _pos_m);

    gl_Position = vec4(
        disp_pos_m[gl_VertexIndex % 3].xy,
        atan(_pos_m[gl_VertexIndex % 3].z * 0.01) * 2 / radians(180),
        disp_pos_m[gl_VertexIndex % 3].w
    );


    points[ 0 ][0] = ((disp_pos_m[0].x + 1.0) / 2.0) * float(WIGHT) ;
    points[ 0 ][1] = ((1.0 * disp_pos_m[0].y + 1.0) / 2.0) * float(HEIGHT);
    points[ 0 ][2] = 0.0; //((disp_pos_m[0].z + 1.0) / 2.0) * sqrt(float(WIGHT) * float(HEIGHT));
    points[ 0 ][3] = 0.0;

    points[ 1 ][0] = ((disp_pos_m[1].x + 1.0) / 2.0) * float(WIGHT) ;
    points[ 1 ][1] = ((1.0 * disp_pos_m[1].y + 1.0) / 2.0) * float(HEIGHT);
    points[ 1 ][2] = 0.0; //(disp_pos_m[1].z + 1) / 2.0) * sqrt(float(WIGHT) * float(HEIGHT));
    points[ 1 ][3] = 0.0;

    points[ 2 ][0] = ((disp_pos_m[2].x + 1.0) / 2.0) * float(WIGHT) ;
    points[ 2 ][1] = ((1.0 * disp_pos_m[2].y + 1.0) / 2.0) * float(HEIGHT);
    points[ 2 ][2] = 0.0; //((disp_pos_m[2].z + 1.0) / 2.0) * sqrt(float(WIGHT) * float(HEIGHT));
    points[ 2 ][3] = 0.0;

    points[ 3 ][0] = 0.0;
    points[ 3 ][1] = 0.0;
    points[ 3 ][2] = 0.0;
    points[ 3 ][3] = 1.0;



    float board_size_mn =  (move_matrix[0][0] + move_matrix[1][1]) / 2;
    contour_size = vec3(
        contour[0] == 0.0 ? 0.0: (contour[0] == 1.0 ? 1.0 :max( contour[0] * length(poses[0].xyz - poses[1].xyz) / length(position[0].xy - position[1].xy), 1.4)),
        contour[1] == 0.0 ? 0.0: (contour[1] == 1.0 ? 1.0 :max( contour[1] * length(poses[1].xyz - poses[2].xyz) / length(position[1].xy - position[2].xy), 1.4)),
        contour[2] == 0.0 ? 0.0: (contour[2] == 1.0 ? 1.0 :max( contour[2] * length(poses[2].xyz - poses[0].xyz) / length(position[2].xy - position[0].xy), 1.4))
    );

    contour_colors_fr = contour_colors;

    fragColor = point_colors[ gl_VertexIndex % 3 ] ;


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
