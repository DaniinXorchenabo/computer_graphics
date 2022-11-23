
// language=GLSL
#version 450
layout (constant_id = 0) const int WIGHT = 64;
layout (constant_id = 1) const int HEIGHT = 64;

layout(location = 0) out vec4 f_color;
layout(location = 4) in vec4 fragColor;
layout(location = 5) in vec3 contour_size;
layout(location = 15) in mat4 points;
layout(location = 19) in vec4[3] contour_colors_fr;
layout(location = 27) in vec4 plane_params;

// layout (depth_any) out float gl_FragDepth;

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

    if (abs( A1 * gl_FragCoord.x   + B1 * gl_FragCoord.y + C1) / sqrt(A1*A1 + B1*B1) < contour_size.x ){
        f_color = contour_colors_fr[0];
        // gl_FragDepth = -1000.;
    }
    else if (abs( A2 * gl_FragCoord.x   + B2 * gl_FragCoord.y + C2) / sqrt(A2*A2 + B2*B2) < contour_size.y){
        f_color = contour_colors_fr[1];
        // gl_FragDepth = -1000.;
    }
    else if (abs( A3 * gl_FragCoord.x   + B3 * gl_FragCoord.y + C3) / sqrt(A3*A3 + B3*B3) < contour_size.z){
        f_color = contour_colors_fr[2];
        // gl_FragDepth = -1000.;
    }
    else {
        vec3 sun_point =  vec3(6.8, 10.7, 20.);
        vec3 cam_pos = vec3(10000., 10000., 1000000.);

        float light_distanse =  plane_params.x   * sun_point.x
                                + plane_params.y * sun_point.y
                                + plane_params.z * sun_point.z
                                + plane_params.w;
        float cam_distanse =    plane_params.x   * cam_pos.x
                                + plane_params.y * cam_pos.y
                                + plane_params.z * cam_pos.z
                                + plane_params.w;

        if (light_distanse * cam_distanse <= 0) {
            f_color = vec4(0., 0., 0., 1.) ;
        } else {
            vec3 to_cam = sun_point - points[3].xyz;
            float mn = abs(plane_params.x * to_cam.x + plane_params.y * to_cam.y +plane_params.z * to_cam.z ) / length(plane_params.xyz) / length(to_cam);
            f_color = vec4(fragColor.xyz * mn, fragColor.a) ;//* abs(plane_params.x * to_cam.x + plane_params.y * to_cam.y +plane_params.z * to_cam.z ) / length(plane_params.xyz) / length(to_cam) ;
//            f_color = vec4(plane_params.xyz / length(plane_params.xyz), 1.) ;


            //            f_color = vec4(points[3].xyz / length(points[3].xyz ), 1.);
            //            f_color = fragColor;
        }
        //        f_color = vec4(1., 1., 1., 1.);
        // float len = length(gl_FragCoord.xyz);
        // f_color = vec4(0., 0., (atan(points[0][2] * 1) * 2 / radians(180)), 0.5);
        // gl_FragDepth = points[0][2];
    }
    // gl_FragDepth = 1. - ((atan(points[0][2] * 1.) * 2. / radians(180)) + 1.) / 2.;
    // f_color = vec4(0., 1., 0., 1.);


}