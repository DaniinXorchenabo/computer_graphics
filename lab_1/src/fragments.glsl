
// language=GLSL
#version 450
layout (constant_id = 0) const int WIGHT = 64;
layout (constant_id = 1) const int HEIGHT = 64;

layout(location = 0) out vec4 f_color;
layout(location = 15) in vec4 fragColor;
layout(location = 16) in vec3 contour_size;

layout(location = 19) in float points[3][2];
layout(location = 25) in vec4[3] contour_colors_fr;


void main() {
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

}
