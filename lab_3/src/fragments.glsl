
// language=GLSL
#version 450
layout (constant_id = 0) const int WIGHT = 64;
layout (constant_id = 1) const int HEIGHT = 64;

layout(location = 0) out vec4 f_color;
layout(location = 4) in vec4 fragColor;
layout(location = 5) in vec3 contour_size;
layout(location = 15) in mat4 points;
layout(location = 19) in vec4[3] contour_colors_fr;




void main() {

    // float A1 = points[0][1] - points[1][1];
    // float B1 = points[1][0] - points[0][0];
    // float C1 = points[0][0] * points[1][1] - points[1][0] * points[0][1];
    //
    // float A2 = points[1][1] - points[2][1];
    // float B2 = points[2][0] - points[1][0];
    // float C2 = points[1][0] * points[2][1] - points[2][0] * points[1][1];
    //
    // float A3 = points[2][1] - points[0][1];
    // float B3 = points[0][0] - points[2][0];
    // float C3 = points[2][0] * points[0][1] - points[0][0] * points[2][1];
    //
    // vec4 v_1_0 = points[1] - points[0];
    // vec4 v_2_0 = points[2] - points[0];
    //
    // vec4 v_2_1 = points[2] - points[1];
    // vec4 v_0_1 = points[0] - points[1];
    //
    // vec4 v_0_2 = points[0] - points[2];
    // vec4 v_1_2 = points[1] - points[2];
    //
    // float A1 = v_1_0.y * v_2_0.z - v_1_0.z * v_2_0.y ;
    // float B1 = v_1_0.z * v_2_0.x - v_1_0.x * v_2_0.z;
    // float C1 = v_1_0.x * v_2_0.y - v_1_0.y * v_2_0.x;
    // float D1 = (- A1 * points[0][0] - B1 * points[0][1] - C1 * points[0][2]);
    //
    // float A2 = v_2_1.y * v_0_1.z - v_2_1.z * v_0_1.y;
    // float B2 = v_2_1.z * v_0_1.x - v_2_1.x * v_0_1.z;
    // float C2 = v_2_1.x * v_0_1.y - v_2_1.y * v_0_1.x;
    // float D2 = (- A2 * points[1][0] - B2 * points[1][1] - C2 * points[1][2]);
    //
    // float A3 = v_0_2.y * v_1_2.z - v_0_2.z * v_1_2.y;
    // float B3 = v_0_2.z * v_1_2.x - v_0_2.x * v_1_2.z;
    // float C3 = v_0_2.x * v_1_2.y - v_0_2.y * v_1_2.x;
    // float D3 = (- A2 * points[2][0] - B2 * points[2][1] - C2 * points[2][2]);
    //
    //
    // if (abs( A1 * gl_FragCoord.x   + B1 * gl_FragCoord.y + C1 * gl_FragCoord.z + D1) / sqrt(A1*A1 + B1*B1 + C1*C1) < contour_size.x )
    //     f_color = contour_colors_fr[0];
    //
    // else if (abs( A2 * gl_FragCoord.x   + B2 * gl_FragCoord.y + C2 * gl_FragCoord.z + D2) / sqrt(A2*A2 + B2*B2 + C2*C2) < contour_size.y)
    //     f_color = contour_colors_fr[1];
    //
    // else if (abs( A3 * gl_FragCoord.x   + B3 * gl_FragCoord.y + C3 * gl_FragCoord.z + D3) / sqrt(A3*A3 + B3*B3 + C3*C3) < contour_size.z)
    //     f_color = contour_colors_fr[2];
    //
    // else
    //     f_color = fragColor;


    float A1 = points[0][1] - points[1][1];
    float B1 = points[1][0] - points[0][0];
    float C1 = points[0][0] * points[1][1] - points[1][0] * points[0][1];

    float A2 = points[1][1] - points[2][1];
    float B2 = points[2][0] - points[1][0];
    float C2 = points[1][0] * points[2][1] - points[2][0] * points[1][1];

    float A3 = points[2][1] - points[0][1];
    float B3 = points[0][0] - points[2][0];
    float C3 = points[2][0] * points[0][1] - points[0][0] * points[2][1];

    if (abs( A1 * gl_FragCoord.x   + B1 * gl_FragCoord.y + C1) / sqrt(A1*A1 + B1*B1) < contour_size.x )
        f_color = contour_colors_fr[0];

    else if (abs( A2 * gl_FragCoord.x   + B2 * gl_FragCoord.y + C2) / sqrt(A2*A2 + B2*B2) < contour_size.y)
        f_color = contour_colors_fr[1];

    else if (abs( A3 * gl_FragCoord.x   + B3 * gl_FragCoord.y + C3) / sqrt(A3*A3 + B3*B3) < contour_size.z)
        f_color = contour_colors_fr[2];

    else
        f_color = fragColor;

}"