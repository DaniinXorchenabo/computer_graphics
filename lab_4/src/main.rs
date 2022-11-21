// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! This is the source code of the "Windowing" chapter at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the guide itself.

// use std::default::default;
// use std::mem::zeroed;
// use std::simd::f32x4;
// use std::intrinsics::{cosf32, sinf32};
// use std::ops::ControlFlow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, StateMode};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
};
use vulkano::sync::{self, FenceSignalFuture, FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Event as WinitEvent, Event, MouseButton, MouseScrollDelta, WindowEvent, KeyboardInput, VirtualKeyCode};
use winit::event_loop::{ControlFlow as WinitControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use vulkano::shader::SpecializationConstants;
use vulkano::shader::SpecializationMapEntry;
use std::collections::HashMap;
// use image::error::UnsupportedErrorKind::Format;
// extern crate ndarray;
use vulkano::{format::Format};
use ndarray::{arr2, arr3, Array2, array, Array};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use winit::event::VirtualKeyCode::V;
// use ndarray::{arr2, arr3,  Array2, array, Array};


#[repr(C)]      // `#[repr(C)]` guarantees that the struct has a specific layout
#[derive(Default, Copy, Clone)]
struct MySpecConstants {
    wight: u32,
    height: u32,
}

unsafe impl SpecializationConstants for MySpecConstants {
    fn descriptors() -> &'static [SpecializationMapEntry] {
        static DESCRIPTORS: [SpecializationMapEntry; 2] = [
            SpecializationMapEntry {
                constant_id: 0,
                offset: 0,
                size: 4,
            },
            SpecializationMapEntry {
                constant_id: 1,
                offset: 4,
                size: 4,
            },
        ];

        &DESCRIPTORS
    }
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
pub struct Vertex {
    position: [[f32; 4]; 4],
    // move_matrix: [[f32; 3]; 3],
    move_matrix: [[f32; 4]; 4],
    contour: [f32; 3],
    contour_colors: [[f32; 4]; 3],
    point_colors: [[f32; 4]; 3],
    projection_flag: i32,
}

impl Vertex {
    pub fn new(
        points: [[f32; 3]; 3],
        contour: Option<[f32; 3]>,
        point_colors: Option<[[f32; 4]; 3]>,
        point_color: Option<[f32; 4]>,
        contour_color: Option<[f32; 4]>,
        contour_colors: Option<[[f32; 4]; 3]>,
    ) -> Self {
        // let new_points : [[f32; 4]; 4] = [points[0], points[1], points[2], [0.0, 0.0, 0.0]]
        //     .map(|x| [x[0], x[1], x[2], 0.0]);
        Vertex {
            position: [points[0], points[1], points[2], [0.0, 0.0, 0.0]]
                .map(|x| [x[0], x[1], x[2], 1.0]),
            move_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            // move_matrix: [[1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0 ], [0.0, 0.0, 1.0 ]],
            contour: match contour {
                Some(x) => x,
                None => [1.0, 1.0, 1.0],
            },
            contour_colors: match (contour_colors, contour_color) {
                (Some(x), _) => x,
                (None, Some(y)) => [y.clone(), y.clone(), y],
                (_, _) => [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            },
            point_colors: match (point_colors, point_color) {
                (Some(x), _) => x,
                (None, Some(y)) => [y.clone(), y.clone(), y],
                (_, _) => [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
            },
            projection_flag: 0,
        }
    }
}

#[derive(Default, Clone)]
pub struct Changed {
    move_matrix: bool,
    rotate_angels: bool,
    scale: bool,
    projection_flag: bool,
}

impl Changed {
    pub fn new() -> Self {
        Changed {
            move_matrix: false,
            rotate_angels: false,
            scale: false,
            projection_flag: false,
        }
    }

    pub fn any(&self) -> bool {
        self.move_matrix | self.rotate_angels | self.scale | self.projection_flag
    }
}

#[derive(Default, Clone)]
pub struct Figure {
    polygons: Vec<Vertex>,
    // move_matrix: [[f32; 3]; 3],
    change_matrix: Array2<f32>,

    move_matrix: Array2<f32>,
    rotate_angels: [f32; 3],
    scale: [f32; 3],

    _rotate_matrix: Array2<f32>,
    _changed: Changed,
    projection_mode: i32,
    // _changed: bool
}

impl Figure {
    pub fn new(new_polygons: Vec<Vertex>) -> Self {
        let mut real_polygons = Vec::new();
        // let default_matrix = [[1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0 ], [0.0, 0.0, 1.0 ]];
        // let  default_matrix :Array2<f32> = array![
        //     [1.0, 0.0, 0.0, 1.0],
        //     [0.0, 1.0, 0.0, 1.0],
        //     [0.0, 0.0, 1.0, 1.0],
        //     [0.0, 0.0, 0.0, 1.0]
        // ];
        for vertex in new_polygons {
            // vertex.move_matrix = default_matrix.clone();
            real_polygons.push(vertex.clone());
            real_polygons.push(vertex.clone());
            real_polygons.push(vertex.clone());
        }


        Figure {
            polygons: real_polygons,
            // move_matrix: default_matrix,
            change_matrix: Array::eye(4),
            move_matrix: Array::eye(4),
            rotate_angels: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            _rotate_matrix: Array::eye(4),
            _changed: Changed::new(),
            projection_mode: 0,
            // _changed: false,
        }
    }

    fn get_vertex(&mut self, windows_size: PhysicalSize<u32>) -> Vec<Vertex> {
        if self._changed.any() {
            if self._changed.rotate_angels {
                let (x, y, z) = (self.rotate_angels[0], self.rotate_angels[1], self.rotate_angels[2]);
                let (a, b, c, d, e, f) = (f32::cos(x), f32::sin(x), f32::cos(y), f32::sin(y), f32::cos(z), f32::sin(z));
                self._rotate_matrix = array![
                    [c * e, -c * f, -d, 0.0],
                    [-b * d * e + a * f, b * d * f + a * e, -b * c, 0.0],
                    [a * d * e + b * f, -a * d * f + b * e, a * c, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ];
                // self._rotate_matrix = array![
                //     [1.0, 0.0, 0.0, 0.0],
                //     [0.0, a, -b, 0.0],
                //     [0.0, b, a, 0.0],
                //     [0.0, 0.0, 0.0, 1.0]
                // ];
                // self._rotate_matrix = array![ // ось Х
                //     [1.0, 0.0, 0.0, 0.0],
                //     [0.0, e, -f, 0.0],
                //     [0.0, f, e, 0.0],
                //     [0.0, 0.0, 0.0, 1.0]
                // ];
                // self._rotate_matrix = array![
                //     [e, -f, 0.0, 0.0],
                //     [f, e, 0.0, 0.0],
                //     [0.0, 0.0, 1.0, 0.0],
                //     [0.0, 0.0, 0.0, 1.0]
                // ];
                // self._rotate_matrix = array![
                //     [e, 0.0, -f, 0.0],
                //     [0.0, 1.0, 0.0, 0.0],
                //     [f, 0.0, e, 0.0],
                //     [0.0, 0.0, 0.0, 1.0]
                // ];
            }


            self._changed = Changed::new();
            // self._changed = false;
            self.change_matrix = array![
                [self.scale[0], 0.0, 0.0, 0.0],
                [0.0, self.scale[1], 0.0, 0.0],
                [0.0, 0.0, self.scale[2], 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ];
            // self.change_matrix = array![
            //     [self.scale[0] / 1.0_f32.max(windows_size.width as f32 / windows_size.height as f32), 0.0, 0.0, 0.0],
            //     [0.0, self.scale[1] / 1.0_f32.max(windows_size.height as f32 / windows_size.width as f32), 0.0, 0.0],
            //     [0.0, 0.0, self.scale[2], 0.0],
            //     [0.0, 0.0, 0.0, 1.0]
            // ] ;


            // self.change_matrix = self.change_matrix.clone() * self.move_matrix.clone();

            // self.change_matrix = self.change_matrix.clone() * self._rotate_matrix.clone();
            self.change_matrix = self.change_matrix.dot(&self._rotate_matrix.clone());
            self.change_matrix = self.change_matrix.dot(&self.move_matrix.clone());


            let mut loc_change_matrix: [[f32; 4]; 4] = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ];

            // let mut loc_change_matrix:[[f32; 3]; 3] = [
            //     [1.0, 0.0, 0.0],
            //     [0.0, 1.0, 0.0],
            //     [0.0, 0.0, 1.0]
            // ];
            // change_matrix

            for ((i, j), item) in self.change_matrix.indexed_iter() {
                loc_change_matrix[i][j] = item.clone();
            }


            // let mut res = self.polygons;


            for vertex in &mut self.polygons {
                // vertex.move_matrix = change_matrix.clone()
                vertex.move_matrix = loc_change_matrix.clone();
                vertex.projection_flag = self.projection_mode;
            }
            self.polygons.clone()
            // self.polygons.clone()
        } else {
            self.polygons.clone()
        }
    }
}


//3189 -- 1 1046 -- 2  610 -- 3 409 -- 4  286 -- 5 210 -- 6 140 -- 7
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
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
        (atan(_pos_m[gl_VertexIndex % 3].z * 0.01) * 2 / radians(180)),
        disp_pos_m[gl_VertexIndex % 3].w
    );


    points[ 0 ][0] = ((disp_pos_m[0].x + 1.0) / 2.0) * float(WIGHT) ;
    points[ 0 ][1] = ((1.0 * disp_pos_m[0].y + 1.0) / 2.0) * float(HEIGHT);
    points[ 0 ][2] = _pos_m[gl_VertexIndex % 3].z; //(atan(_pos_m[gl_VertexIndex % 3].z * 0.5) * 2 / radians(180));
    //((disp_pos_m[0].z + 1.0) / 2.0) * sqrt(float(WIGHT) * float(HEIGHT));
    points[ 0 ][3] = 0.0;

    points[ 1 ][0] = ((disp_pos_m[1].x + 1.0) / 2.0) * float(WIGHT) ;
    points[ 1 ][1] = ((1.0 * disp_pos_m[1].y + 1.0) / 2.0) * float(HEIGHT);
    points[ 1 ][2] = _pos_m[gl_VertexIndex % 3].z; //(atan(_pos_m[gl_VertexIndex % 3].z * 0.5) * 2 / radians(180));
     //(disp_pos_m[1].z + 1) / 2.0) * sqrt(float(WIGHT) * float(HEIGHT));
    points[ 1 ][3] = 0.0;

    points[ 2 ][0] = ((disp_pos_m[2].x + 1.0) / 2.0) * float(WIGHT) ;
    points[ 2 ][1] = ((1.0 * disp_pos_m[2].y + 1.0) / 2.0) * float(HEIGHT);
    points[ 2 ][2] = _pos_m[gl_VertexIndex % 3].z; //(atan(_pos_m[gl_VertexIndex % 3].z * 0.5) * 2 / radians(180));
    //((disp_pos_m[2].z + 1.0) / 2.0) * sqrt(float(WIGHT) * float(HEIGHT));
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

}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        // language=GLSL
        src: "
// language=GLSL
#version 450
layout (constant_id = 0) const int WIGHT = 64;
layout (constant_id = 1) const int HEIGHT = 64;

layout(location = 0) out vec4 f_color;
layout(location = 4) in vec4 fragColor;
layout(location = 5) in vec3 contour_size;
layout(location = 15) in mat4 points;
layout(location = 19) in vec4[3] contour_colors_fr;
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
            f_color = fragColor;
            // float len = length(gl_FragCoord.xyz);
            // f_color = vec4(0., 0., (atan(points[0][2] * 1) * 2 / radians(180)), 0.5);
            // gl_FragDepth = points[0][2];
        }
        // gl_FragDepth = 1. - ((atan(points[0][2] * 1.) * 2. / radians(180)) + 1.) / 2.;
        // f_color = vec4(0., 1., 0., 1.);


}"
    }
}

pub fn select_physical_device<'a>(
    instance: &'a Arc<Instance>,
    surface: Arc<Surface<Window>>,
    device_extensions: &DeviceExtensions,
) -> (PhysicalDevice<'a>, QueueFamily<'a>) {
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .expect("no device available");
    (physical_device, queue_family)
}

fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain<Window>>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),  // set the format the same as the swapchain
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
        .unwrap()
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    device: Arc<Device>,
    dimensions__: PhysicalSize<u32>,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(
            device.clone(),
            dimensions,
            // [dimensions__.width, dimensions__.height],
            vulkano::format::Format::D16_UNORM,
        ).unwrap(),
    )
        .unwrap();
    // depthTest;
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
                .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
    windows_size: PhysicalSize<u32>,
) -> Arc<GraphicsPipeline> {
    let consts = MySpecConstants {
        wight: windows_size.width,
        height: windows_size.height,
    };
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), consts.clone())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))

        // .geometry_shader(fs.entry_point("main").unwrap(), ())
        .fragment_shader(fs.entry_point("main").unwrap(), consts)
        // .depth_stencil_state(DepthStencilState{
        //     depth: Some(DepthState{
        //         enable_dynamic: true,
        //         write_enable: StateMode::Dynamic, //StateMode::Fixed(true),
        //         compare_op: StateMode::Dynamic, //StateMode::Fixed(CompareOp::Always)
        //     }),
        //     depth_bounds: None,
        //     stencil: None,
        // })
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
}


fn get_command_buffers(
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::MultipleSubmit,
            )
                .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![
                            Some([1.0, 1.0, 1.0, 1.0].into()),
                            Some(1f32.into()),
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                // .bind_descriptor_sets(
                //     PipelineBindPoint::Graphics,
                //     pipeline.layout().clone(),
                //     0,
                //     set,
                // )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
        .expect("failed to create instance");

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) =
        select_physical_device(&instance, surface.clone(), &device_extensions);

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            enabled_extensions: device_extensions, // new
            ..Default::default()
        },
    )
        .expect("failed to create device");

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let dimensions = surface.window().inner_size();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::color_attachment(),
                composite_alpha,
                ..Default::default()
            },
        )
            .unwrap()
    };

    let render_pass = get_render_pass(device.clone(), swapchain.clone());
    let framebuffers = get_framebuffers(
        &images,
        render_pass.clone(),
        device.clone(),
        surface.window().inner_size(),
    );

    vulkano::impl_vertex!(Vertex, position, move_matrix, contour, contour_colors, point_colors, projection_flag);

    let mut figure1 = Figure::new(
        vec![
            Vertex::new([[0.115, -0.486, 0.0], [0.1062461, -0.486, 0.0440086], [0.0, -0.5, 0.0]],None, Some([[0.89, 0.024, 0.024, 1.0], [0.89, 0.349, 0.024, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[0.1062461, -0.486, 0.0440086], [0.0813173, -0.486, 0.0813173], [0.0, -0.5, 0.0]],None, Some([[0.89, 0.349, 0.024, 1.0], [0.89, 0.674, 0.024, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[0.0813173, -0.486, 0.0813173], [0.0440086, -0.486, 0.1062461], [0.0, -0.5, 0.0]],None, Some([[0.89, 0.674, 0.024, 1.0], [0.782, 0.89, 0.024, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[0.0440086, -0.486, 0.1062461], [0.0, -0.486, 0.115], [0.0, -0.5, 0.0]],None, Some([[0.782, 0.89, 0.024, 1.0], [0.457, 0.89, 0.024, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[0.0, -0.486, 0.115], [-0.0440086, -0.486, 0.1062461], [0.0, -0.5, 0.0]],None, Some([[0.457, 0.89, 0.024, 1.0], [0.132, 0.89, 0.024, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[-0.0440086, -0.486, 0.1062461], [-0.0813173, -0.486, 0.0813173], [0.0, -0.5, 0.0]],None, Some([[0.132, 0.89, 0.024, 1.0], [0.024, 0.89, 0.24, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[-0.0813173, -0.486, 0.0813173], [-0.1062461, -0.486, 0.0440086], [0.0, -0.5, 0.0]],None, Some([[0.024, 0.89, 0.24, 1.0], [0.024, 0.89, 0.565, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[-0.1062461, -0.486, 0.0440086], [-0.115, -0.486, 0.0], [0.0, -0.5, 0.0]],None, Some([[0.024, 0.89, 0.565, 1.0], [0.024, 0.89, 0.89, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[-0.115, -0.486, 0.0], [-0.1062461, -0.486, -0.0440086], [0.0, -0.5, 0.0]],None, Some([[0.024, 0.89, 0.89, 1.0], [0.024, 0.565, 0.89, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[-0.1062461, -0.486, -0.0440086], [-0.0813173, -0.486, -0.0813173], [0.0, -0.5, 0.0]],None, Some([[0.024, 0.565, 0.89, 1.0], [0.024, 0.24, 0.89, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[-0.0813173, -0.486, -0.0813173], [-0.0440086, -0.486, -0.1062461], [0.0, -0.5, 0.0]],None, Some([[0.024, 0.24, 0.89, 1.0], [0.132, 0.024, 0.89, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[-0.0440086, -0.486, -0.1062461], [-0.0, -0.486, -0.115], [0.0, -0.5, 0.0]],None, Some([[0.132, 0.024, 0.89, 1.0], [0.457, 0.024, 0.89, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[-0.0, -0.486, -0.115], [0.0440086, -0.486, -0.1062461], [0.0, -0.5, 0.0]],None, Some([[0.457, 0.024, 0.89, 1.0], [0.782, 0.024, 0.89, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[0.0440086, -0.486, -0.1062461], [0.0813173, -0.486, -0.0813173], [0.0, -0.5, 0.0]],None, Some([[0.782, 0.024, 0.89, 1.0], [0.89, 0.024, 0.674, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[0.0813173, -0.486, -0.0813173], [0.1062461, -0.486, -0.0440086], [0.0, -0.5, 0.0]],None, Some([[0.89, 0.024, 0.674, 1.0], [0.89, 0.024, 0.349, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[0.1062461, -0.486, -0.0440086], [0.115, -0.486, 0.0], [0.0, -0.5, 0.0]],None, Some([[0.89, 0.024, 0.349, 1.0], [0.89, 0.024, 0.024, 1.0], [0.9, 0.0, 0.0, 1.0]]), None, None, None),
            Vertex::new([[0.1062461, -0.486, 0.0440086], [0.115, -0.486, 0.0], [0.2942356, -0.4, 0.0585271]],None, Some([[0.89, 0.349, 0.024, 1.0], [0.89, 0.024, 0.024, 1.0], [0.932, 0.23, 0.068, 1.0]]), None, None, None),
            Vertex::new([[0.0813173, -0.486, 0.0813173], [0.1062461, -0.486, 0.0440086], [0.2494409, -0.4, 0.1666711]],None, Some([[0.89, 0.674, 0.024, 1.0], [0.89, 0.349, 0.024, 1.0], [0.932, 0.554, 0.068, 1.0]]), None, None, None),
            Vertex::new([[0.0440086, -0.486, 0.1062461], [0.0813173, -0.486, 0.0813173], [0.1666711, -0.4, 0.2494409]],None, Some([[0.782, 0.89, 0.024, 1.0], [0.89, 0.674, 0.024, 1.0], [0.932, 0.878, 0.068, 1.0]]), None, None, None),
            Vertex::new([[0.0, -0.486, 0.115], [0.0440086, -0.486, 0.1062461], [0.0585271, -0.4, 0.2942356]],None, Some([[0.457, 0.89, 0.024, 1.0], [0.782, 0.89, 0.024, 1.0], [0.662, 0.932, 0.068, 1.0]]), None, None, None),
            Vertex::new([[-0.0440086, -0.486, 0.1062461], [0.0, -0.486, 0.115], [-0.0585271, -0.4, 0.2942356]],None, Some([[0.132, 0.89, 0.024, 1.0], [0.457, 0.89, 0.024, 1.0], [0.338, 0.932, 0.068, 1.0]]), None, None, None),
            Vertex::new([[-0.0813173, -0.486, 0.0813173], [-0.0440086, -0.486, 0.1062461], [-0.1666711, -0.4, 0.2494409]],None, Some([[0.024, 0.89, 0.24, 1.0], [0.132, 0.89, 0.024, 1.0], [0.068, 0.932, 0.122, 1.0]]), None, None, None),
            Vertex::new([[-0.1062461, -0.486, 0.0440086], [-0.0813173, -0.486, 0.0813173], [-0.2494409, -0.4, 0.1666711]],None, Some([[0.024, 0.89, 0.565, 1.0], [0.024, 0.89, 0.24, 1.0], [0.068, 0.932, 0.446, 1.0]]), None, None, None),
            Vertex::new([[-0.115, -0.486, 0.0], [-0.1062461, -0.486, 0.0440086], [-0.2942356, -0.4, 0.0585271]],None, Some([[0.024, 0.89, 0.89, 1.0], [0.024, 0.89, 0.565, 1.0], [0.068, 0.932, 0.77, 1.0]]), None, None, None),
            Vertex::new([[-0.1062461, -0.486, -0.0440086], [-0.115, -0.486, 0.0], [-0.2942356, -0.4, -0.0585271]],None, Some([[0.024, 0.565, 0.89, 1.0], [0.024, 0.89, 0.89, 1.0], [0.068, 0.77, 0.932, 1.0]]), None, None, None),
            Vertex::new([[-0.0813173, -0.486, -0.0813173], [-0.1062461, -0.486, -0.0440086], [-0.2494409, -0.4, -0.1666711]],None, Some([[0.024, 0.24, 0.89, 1.0], [0.024, 0.565, 0.89, 1.0], [0.068, 0.446, 0.932, 1.0]]), None, None, None),
            Vertex::new([[-0.0440086, -0.486, -0.1062461], [-0.0813173, -0.486, -0.0813173], [-0.1666711, -0.4, -0.2494409]],None, Some([[0.132, 0.024, 0.89, 1.0], [0.024, 0.24, 0.89, 1.0], [0.068, 0.122, 0.932, 1.0]]), None, None, None),
            Vertex::new([[-0.0, -0.486, -0.115], [-0.0440086, -0.486, -0.1062461], [-0.0585271, -0.4, -0.2942356]],None, Some([[0.457, 0.024, 0.89, 1.0], [0.132, 0.024, 0.89, 1.0], [0.338, 0.068, 0.932, 1.0]]), None, None, None),
            Vertex::new([[0.0440086, -0.486, -0.1062461], [-0.0, -0.486, -0.115], [0.0585271, -0.4, -0.2942356]],None, Some([[0.782, 0.024, 0.89, 1.0], [0.457, 0.024, 0.89, 1.0], [0.662, 0.068, 0.932, 1.0]]), None, None, None),
            Vertex::new([[0.0813173, -0.486, -0.0813173], [0.0440086, -0.486, -0.1062461], [0.1666711, -0.4, -0.2494409]],None, Some([[0.89, 0.024, 0.674, 1.0], [0.782, 0.024, 0.89, 1.0], [0.932, 0.068, 0.878, 1.0]]), None, None, None),
            Vertex::new([[0.1062461, -0.486, -0.0440086], [0.0813173, -0.486, -0.0813173], [0.2494409, -0.4, -0.1666711]],None, Some([[0.89, 0.024, 0.349, 1.0], [0.89, 0.024, 0.674, 1.0], [0.932, 0.068, 0.554, 1.0]]), None, None, None),
            Vertex::new([[0.115, -0.486, 0.0], [0.1062461, -0.486, -0.0440086], [0.2942356, -0.4, -0.0585271]],None, Some([[0.89, 0.024, 0.024, 1.0], [0.89, 0.024, 0.349, 1.0], [0.932, 0.068, 0.23, 1.0]]), None, None, None),
            Vertex::new([[0.2942356, -0.4, 0.0585271], [0.2494409, -0.4, 0.1666711], [0.1062461, -0.486, 0.0440086]],None, Some([[0.932, 0.23, 0.068, 1.0], [0.932, 0.554, 0.068, 1.0], [0.89, 0.349, 0.024, 1.0]]), None, None, None),
            Vertex::new([[0.2494409, -0.4, 0.1666711], [0.1666711, -0.4, 0.2494409], [0.0813173, -0.486, 0.0813173]],None, Some([[0.932, 0.554, 0.068, 1.0], [0.932, 0.878, 0.068, 1.0], [0.89, 0.674, 0.024, 1.0]]), None, None, None),
            Vertex::new([[0.1666711, -0.4, 0.2494409], [0.0585271, -0.4, 0.2942356], [0.0440086, -0.486, 0.1062461]],None, Some([[0.932, 0.878, 0.068, 1.0], [0.662, 0.932, 0.068, 1.0], [0.782, 0.89, 0.024, 1.0]]), None, None, None),
            Vertex::new([[0.0585271, -0.4, 0.2942356], [-0.0585271, -0.4, 0.2942356], [0.0, -0.486, 0.115]],None, Some([[0.662, 0.932, 0.068, 1.0], [0.338, 0.932, 0.068, 1.0], [0.457, 0.89, 0.024, 1.0]]), None, None, None),
            Vertex::new([[-0.0585271, -0.4, 0.2942356], [-0.1666711, -0.4, 0.2494409], [-0.0440086, -0.486, 0.1062461]],None, Some([[0.338, 0.932, 0.068, 1.0], [0.068, 0.932, 0.122, 1.0], [0.132, 0.89, 0.024, 1.0]]), None, None, None),
            Vertex::new([[-0.1666711, -0.4, 0.2494409], [-0.2494409, -0.4, 0.1666711], [-0.0813173, -0.486, 0.0813173]],None, Some([[0.068, 0.932, 0.122, 1.0], [0.068, 0.932, 0.446, 1.0], [0.024, 0.89, 0.24, 1.0]]), None, None, None),
            Vertex::new([[-0.2494409, -0.4, 0.1666711], [-0.2942356, -0.4, 0.0585271], [-0.1062461, -0.486, 0.0440086]],None, Some([[0.068, 0.932, 0.446, 1.0], [0.068, 0.932, 0.77, 1.0], [0.024, 0.89, 0.565, 1.0]]), None, None, None),
            Vertex::new([[-0.2942356, -0.4, 0.0585271], [-0.2942356, -0.4, -0.0585271], [-0.115, -0.486, 0.0]],None, Some([[0.068, 0.932, 0.77, 1.0], [0.068, 0.77, 0.932, 1.0], [0.024, 0.89, 0.89, 1.0]]), None, None, None),
            Vertex::new([[-0.2942356, -0.4, -0.0585271], [-0.2494409, -0.4, -0.1666711], [-0.1062461, -0.486, -0.0440086]],None, Some([[0.068, 0.77, 0.932, 1.0], [0.068, 0.446, 0.932, 1.0], [0.024, 0.565, 0.89, 1.0]]), None, None, None),
            Vertex::new([[-0.2494409, -0.4, -0.1666711], [-0.1666711, -0.4, -0.2494409], [-0.0813173, -0.486, -0.0813173]],None, Some([[0.068, 0.446, 0.932, 1.0], [0.068, 0.122, 0.932, 1.0], [0.024, 0.24, 0.89, 1.0]]), None, None, None),
            Vertex::new([[-0.1666711, -0.4, -0.2494409], [-0.0585271, -0.4, -0.2942356], [-0.0440086, -0.486, -0.1062461]],None, Some([[0.068, 0.122, 0.932, 1.0], [0.338, 0.068, 0.932, 1.0], [0.132, 0.024, 0.89, 1.0]]), None, None, None),
            Vertex::new([[-0.0585271, -0.4, -0.2942356], [0.0585271, -0.4, -0.2942356], [-0.0, -0.486, -0.115]],None, Some([[0.338, 0.068, 0.932, 1.0], [0.662, 0.068, 0.932, 1.0], [0.457, 0.024, 0.89, 1.0]]), None, None, None),
            Vertex::new([[0.0585271, -0.4, -0.2942356], [0.1666711, -0.4, -0.2494409], [0.0440086, -0.486, -0.1062461]],None, Some([[0.662, 0.068, 0.932, 1.0], [0.932, 0.068, 0.878, 1.0], [0.782, 0.024, 0.89, 1.0]]), None, None, None),
            Vertex::new([[0.1666711, -0.4, -0.2494409], [0.2494409, -0.4, -0.1666711], [0.0813173, -0.486, -0.0813173]],None, Some([[0.932, 0.068, 0.878, 1.0], [0.932, 0.068, 0.554, 1.0], [0.89, 0.024, 0.674, 1.0]]), None, None, None),
            Vertex::new([[0.2494409, -0.4, -0.1666711], [0.2942356, -0.4, -0.0585271], [0.1062461, -0.486, -0.0440086]],None, Some([[0.932, 0.068, 0.554, 1.0], [0.932, 0.068, 0.23, 1.0], [0.89, 0.024, 0.349, 1.0]]), None, None, None),
            Vertex::new([[0.2942356, -0.4, -0.0585271], [0.2942356, -0.4, 0.0585271], [0.115, -0.486, 0.0]],None, Some([[0.932, 0.068, 0.23, 1.0], [0.932, 0.23, 0.068, 1.0], [0.89, 0.024, 0.024, 1.0]]), None, None, None),
            Vertex::new([[0.46, -0.196, 0.0], [0.4249846, -0.196, 0.1760344], [0.2942356, -0.4, 0.0585271]],None, Some([[0.918, 0.286, 0.286, 1.0], [0.918, 0.523, 0.286, 1.0], [0.932, 0.23, 0.068, 1.0]]), None, None, None),
            Vertex::new([[0.4249846, -0.196, 0.1760344], [0.3252691, -0.196, 0.3252691], [0.2494409, -0.4, 0.1666711]],None, Some([[0.918, 0.523, 0.286, 1.0], [0.918, 0.76, 0.286, 1.0], [0.932, 0.554, 0.068, 1.0]]), None, None, None),
            Vertex::new([[0.3252691, -0.196, 0.3252691], [0.1760344, -0.196, 0.4249846], [0.1666711, -0.4, 0.2494409]],None, Some([[0.918, 0.76, 0.286, 1.0], [0.839, 0.918, 0.286, 1.0], [0.932, 0.878, 0.068, 1.0]]), None, None, None),
            Vertex::new([[0.1760344, -0.196, 0.4249846], [0.0, -0.196, 0.46], [0.0585271, -0.4, 0.2942356]],None, Some([[0.839, 0.918, 0.286, 1.0], [0.602, 0.918, 0.286, 1.0], [0.662, 0.932, 0.068, 1.0]]), None, None, None),
            Vertex::new([[0.0, -0.196, 0.46], [-0.1760344, -0.196, 0.4249846], [-0.0585271, -0.4, 0.2942356]],None, Some([[0.602, 0.918, 0.286, 1.0], [0.365, 0.918, 0.286, 1.0], [0.338, 0.932, 0.068, 1.0]]), None, None, None),
            Vertex::new([[-0.1760344, -0.196, 0.4249846], [-0.3252691, -0.196, 0.3252691], [-0.1666711, -0.4, 0.2494409]],None, Some([[0.365, 0.918, 0.286, 1.0], [0.286, 0.918, 0.444, 1.0], [0.068, 0.932, 0.122, 1.0]]), None, None, None),
            Vertex::new([[-0.3252691, -0.196, 0.3252691], [-0.4249846, -0.196, 0.1760344], [-0.2494409, -0.4, 0.1666711]],None, Some([[0.286, 0.918, 0.444, 1.0], [0.286, 0.918, 0.681, 1.0], [0.068, 0.932, 0.446, 1.0]]), None, None, None),
            Vertex::new([[-0.4249846, -0.196, 0.1760344], [-0.46, -0.196, 0.0], [-0.2942356, -0.4, 0.0585271]],None, Some([[0.286, 0.918, 0.681, 1.0], [0.286, 0.918, 0.918, 1.0], [0.068, 0.932, 0.77, 1.0]]), None, None, None),
            Vertex::new([[-0.46, -0.196, 0.0], [-0.4249846, -0.196, -0.1760344], [-0.2942356, -0.4, -0.0585271]],None, Some([[0.286, 0.918, 0.918, 1.0], [0.286, 0.681, 0.918, 1.0], [0.068, 0.77, 0.932, 1.0]]), None, None, None),
            Vertex::new([[-0.4249846, -0.196, -0.1760344], [-0.3252691, -0.196, -0.3252691], [-0.2494409, -0.4, -0.1666711]],None, Some([[0.286, 0.681, 0.918, 1.0], [0.286, 0.444, 0.918, 1.0], [0.068, 0.446, 0.932, 1.0]]), None, None, None),
            Vertex::new([[-0.3252691, -0.196, -0.3252691], [-0.1760344, -0.196, -0.4249846], [-0.1666711, -0.4, -0.2494409]],None, Some([[0.286, 0.444, 0.918, 1.0], [0.365, 0.286, 0.918, 1.0], [0.068, 0.122, 0.932, 1.0]]), None, None, None),
            Vertex::new([[-0.1760344, -0.196, -0.4249846], [-0.0, -0.196, -0.46], [-0.0585271, -0.4, -0.2942356]],None, Some([[0.365, 0.286, 0.918, 1.0], [0.602, 0.286, 0.918, 1.0], [0.338, 0.068, 0.932, 1.0]]), None, None, None),
            Vertex::new([[-0.0, -0.196, -0.46], [0.1760344, -0.196, -0.4249846], [0.0585271, -0.4, -0.2942356]],None, Some([[0.602, 0.286, 0.918, 1.0], [0.839, 0.286, 0.918, 1.0], [0.662, 0.068, 0.932, 1.0]]), None, None, None),
            Vertex::new([[0.1760344, -0.196, -0.4249846], [0.3252691, -0.196, -0.3252691], [0.1666711, -0.4, -0.2494409]],None, Some([[0.839, 0.286, 0.918, 1.0], [0.918, 0.286, 0.76, 1.0], [0.932, 0.068, 0.878, 1.0]]), None, None, None),
            Vertex::new([[0.3252691, -0.196, -0.3252691], [0.4249846, -0.196, -0.1760344], [0.2494409, -0.4, -0.1666711]],None, Some([[0.918, 0.286, 0.76, 1.0], [0.918, 0.286, 0.523, 1.0], [0.932, 0.068, 0.554, 1.0]]), None, None, None),
            Vertex::new([[0.4249846, -0.196, -0.1760344], [0.46, -0.196, 0.0], [0.2942356, -0.4, -0.0585271]],None, Some([[0.918, 0.286, 0.523, 1.0], [0.918, 0.286, 0.286, 1.0], [0.932, 0.068, 0.23, 1.0]]), None, None, None),
            Vertex::new([[0.2494409, -0.4, 0.1666711], [0.2942356, -0.4, 0.0585271], [0.4249846, -0.196, 0.1760344]],None, Some([[0.932, 0.554, 0.068, 1.0], [0.932, 0.23, 0.068, 1.0], [0.918, 0.523, 0.286, 1.0]]), None, None, None),
            Vertex::new([[0.1666711, -0.4, 0.2494409], [0.2494409, -0.4, 0.1666711], [0.3252691, -0.196, 0.3252691]],None, Some([[0.932, 0.878, 0.068, 1.0], [0.932, 0.554, 0.068, 1.0], [0.918, 0.76, 0.286, 1.0]]), None, None, None),
            Vertex::new([[0.0585271, -0.4, 0.2942356], [0.1666711, -0.4, 0.2494409], [0.1760344, -0.196, 0.4249846]],None, Some([[0.662, 0.932, 0.068, 1.0], [0.932, 0.878, 0.068, 1.0], [0.839, 0.918, 0.286, 1.0]]), None, None, None),
            Vertex::new([[-0.0585271, -0.4, 0.2942356], [0.0585271, -0.4, 0.2942356], [0.0, -0.196, 0.46]],None, Some([[0.338, 0.932, 0.068, 1.0], [0.662, 0.932, 0.068, 1.0], [0.602, 0.918, 0.286, 1.0]]), None, None, None),
            Vertex::new([[-0.1666711, -0.4, 0.2494409], [-0.0585271, -0.4, 0.2942356], [-0.1760344, -0.196, 0.4249846]],None, Some([[0.068, 0.932, 0.122, 1.0], [0.338, 0.932, 0.068, 1.0], [0.365, 0.918, 0.286, 1.0]]), None, None, None),
            Vertex::new([[-0.2494409, -0.4, 0.1666711], [-0.1666711, -0.4, 0.2494409], [-0.3252691, -0.196, 0.3252691]],None, Some([[0.068, 0.932, 0.446, 1.0], [0.068, 0.932, 0.122, 1.0], [0.286, 0.918, 0.444, 1.0]]), None, None, None),
            Vertex::new([[-0.2942356, -0.4, 0.0585271], [-0.2494409, -0.4, 0.1666711], [-0.4249846, -0.196, 0.1760344]],None, Some([[0.068, 0.932, 0.77, 1.0], [0.068, 0.932, 0.446, 1.0], [0.286, 0.918, 0.681, 1.0]]), None, None, None),
            Vertex::new([[-0.2942356, -0.4, -0.0585271], [-0.2942356, -0.4, 0.0585271], [-0.46, -0.196, 0.0]],None, Some([[0.068, 0.77, 0.932, 1.0], [0.068, 0.932, 0.77, 1.0], [0.286, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.2494409, -0.4, -0.1666711], [-0.2942356, -0.4, -0.0585271], [-0.4249846, -0.196, -0.1760344]],None, Some([[0.068, 0.446, 0.932, 1.0], [0.068, 0.77, 0.932, 1.0], [0.286, 0.681, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.1666711, -0.4, -0.2494409], [-0.2494409, -0.4, -0.1666711], [-0.3252691, -0.196, -0.3252691]],None, Some([[0.068, 0.122, 0.932, 1.0], [0.068, 0.446, 0.932, 1.0], [0.286, 0.444, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.0585271, -0.4, -0.2942356], [-0.1666711, -0.4, -0.2494409], [-0.1760344, -0.196, -0.4249846]],None, Some([[0.338, 0.068, 0.932, 1.0], [0.068, 0.122, 0.932, 1.0], [0.365, 0.286, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.0585271, -0.4, -0.2942356], [-0.0585271, -0.4, -0.2942356], [-0.0, -0.196, -0.46]],None, Some([[0.662, 0.068, 0.932, 1.0], [0.338, 0.068, 0.932, 1.0], [0.602, 0.286, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.1666711, -0.4, -0.2494409], [0.0585271, -0.4, -0.2942356], [0.1760344, -0.196, -0.4249846]],None, Some([[0.932, 0.068, 0.878, 1.0], [0.662, 0.068, 0.932, 1.0], [0.839, 0.286, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.2494409, -0.4, -0.1666711], [0.1666711, -0.4, -0.2494409], [0.3252691, -0.196, -0.3252691]],None, Some([[0.932, 0.068, 0.554, 1.0], [0.932, 0.068, 0.878, 1.0], [0.918, 0.286, 0.76, 1.0]]), None, None, None),
            Vertex::new([[0.2942356, -0.4, -0.0585271], [0.2494409, -0.4, -0.1666711], [0.4249846, -0.196, -0.1760344]],None, Some([[0.932, 0.068, 0.23, 1.0], [0.932, 0.068, 0.554, 1.0], [0.918, 0.286, 0.523, 1.0]]), None, None, None),
            Vertex::new([[0.2942356, -0.4, 0.0585271], [0.2942356, -0.4, -0.0585271], [0.46, -0.196, 0.0]],None, Some([[0.932, 0.23, 0.068, 1.0], [0.932, 0.068, 0.23, 1.0], [0.918, 0.286, 0.286, 1.0]]), None, None, None),
            Vertex::new([[0.4249846, -0.196, 0.1760344], [0.46, -0.196, 0.0], [0.9307652, -0.0, 0.1851407]],None, Some([[0.918, 0.523, 0.286, 1.0], [0.918, 0.286, 0.286, 1.0], [0.872, 0.593, 0.528, 1.0]]), None, None, None),
            Vertex::new([[0.3252691, -0.196, 0.3252691], [0.4249846, -0.196, 0.1760344], [0.7890647, -0.0, 0.5272362]],None, Some([[0.918, 0.76, 0.286, 1.0], [0.918, 0.523, 0.286, 1.0], [0.872, 0.721, 0.528, 1.0]]), None, None, None),
            Vertex::new([[0.1760344, -0.196, 0.4249846], [0.3252691, -0.196, 0.3252691], [0.5272362, -0.0, 0.7890647]],None, Some([[0.839, 0.918, 0.286, 1.0], [0.918, 0.76, 0.286, 1.0], [0.872, 0.85, 0.528, 1.0]]), None, None, None),
            Vertex::new([[0.0, -0.196, 0.46], [0.1760344, -0.196, 0.4249846], [0.1851407, -0.0, 0.9307652]],None, Some([[0.602, 0.918, 0.286, 1.0], [0.839, 0.918, 0.286, 1.0], [0.764, 0.872, 0.528, 1.0]]), None, None, None),
            Vertex::new([[-0.1760344, -0.196, 0.4249846], [0.0, -0.196, 0.46], [-0.1851407, -0.0, 0.9307652]],None, Some([[0.365, 0.918, 0.286, 1.0], [0.602, 0.918, 0.286, 1.0], [0.636, 0.872, 0.528, 1.0]]), None, None, None),
            Vertex::new([[-0.3252691, -0.196, 0.3252691], [-0.1760344, -0.196, 0.4249846], [-0.5272362, -0.0, 0.7890647]],None, Some([[0.286, 0.918, 0.444, 1.0], [0.365, 0.918, 0.286, 1.0], [0.528, 0.872, 0.55, 1.0]]), None, None, None),
            Vertex::new([[-0.4249846, -0.196, 0.1760344], [-0.3252691, -0.196, 0.3252691], [-0.7890647, -0.0, 0.5272362]],None, Some([[0.286, 0.918, 0.681, 1.0], [0.286, 0.918, 0.444, 1.0], [0.528, 0.872, 0.679, 1.0]]), None, None, None),
            Vertex::new([[-0.46, -0.196, 0.0], [-0.4249846, -0.196, 0.1760344], [-0.9307652, -0.0, 0.1851407]],None, Some([[0.286, 0.918, 0.918, 1.0], [0.286, 0.918, 0.681, 1.0], [0.528, 0.872, 0.807, 1.0]]), None, None, None),
            Vertex::new([[-0.4249846, -0.196, -0.1760344], [-0.46, -0.196, 0.0], [-0.9307652, -0.0, -0.1851407]],None, Some([[0.286, 0.681, 0.918, 1.0], [0.286, 0.918, 0.918, 1.0], [0.528, 0.807, 0.872, 1.0]]), None, None, None),
            Vertex::new([[-0.3252691, -0.196, -0.3252691], [-0.4249846, -0.196, -0.1760344], [-0.7890647, -0.0, -0.5272362]],None, Some([[0.286, 0.444, 0.918, 1.0], [0.286, 0.681, 0.918, 1.0], [0.528, 0.679, 0.872, 1.0]]), None, None, None),
            Vertex::new([[-0.1760344, -0.196, -0.4249846], [-0.3252691, -0.196, -0.3252691], [-0.5272362, -0.0, -0.7890647]],None, Some([[0.365, 0.286, 0.918, 1.0], [0.286, 0.444, 0.918, 1.0], [0.528, 0.55, 0.872, 1.0]]), None, None, None),
            Vertex::new([[-0.0, -0.196, -0.46], [-0.1760344, -0.196, -0.4249846], [-0.1851407, -0.0, -0.9307652]],None, Some([[0.602, 0.286, 0.918, 1.0], [0.365, 0.286, 0.918, 1.0], [0.636, 0.528, 0.872, 1.0]]), None, None, None),
            Vertex::new([[0.1760344, -0.196, -0.4249846], [-0.0, -0.196, -0.46], [0.1851407, -0.0, -0.9307652]],None, Some([[0.839, 0.286, 0.918, 1.0], [0.602, 0.286, 0.918, 1.0], [0.764, 0.528, 0.872, 1.0]]), None, None, None),
            Vertex::new([[0.3252691, -0.196, -0.3252691], [0.1760344, -0.196, -0.4249846], [0.5272362, -0.0, -0.7890647]],None, Some([[0.918, 0.286, 0.76, 1.0], [0.839, 0.286, 0.918, 1.0], [0.872, 0.528, 0.85, 1.0]]), None, None, None),
            Vertex::new([[0.4249846, -0.196, -0.1760344], [0.3252691, -0.196, -0.3252691], [0.7890647, -0.0, -0.5272362]],None, Some([[0.918, 0.286, 0.523, 1.0], [0.918, 0.286, 0.76, 1.0], [0.872, 0.528, 0.721, 1.0]]), None, None, None),
            Vertex::new([[0.46, -0.196, 0.0], [0.4249846, -0.196, -0.1760344], [0.9307652, -0.0, -0.1851407]],None, Some([[0.918, 0.286, 0.286, 1.0], [0.918, 0.286, 0.523, 1.0], [0.872, 0.528, 0.593, 1.0]]), None, None, None),
            Vertex::new([[0.9307652, -0.0, 0.1851407], [0.7890647, -0.0, 0.5272362], [0.4249846, -0.196, 0.1760344]],None, Some([[0.872, 0.593, 0.528, 1.0], [0.872, 0.721, 0.528, 1.0], [0.918, 0.523, 0.286, 1.0]]), None, None, None),
            Vertex::new([[0.7890647, -0.0, 0.5272362], [0.5272362, -0.0, 0.7890647], [0.3252691, -0.196, 0.3252691]],None, Some([[0.872, 0.721, 0.528, 1.0], [0.872, 0.85, 0.528, 1.0], [0.918, 0.76, 0.286, 1.0]]), None, None, None),
            Vertex::new([[0.5272362, -0.0, 0.7890647], [0.1851407, -0.0, 0.9307652], [0.1760344, -0.196, 0.4249846]],None, Some([[0.872, 0.85, 0.528, 1.0], [0.764, 0.872, 0.528, 1.0], [0.839, 0.918, 0.286, 1.0]]), None, None, None),
            Vertex::new([[0.1851407, -0.0, 0.9307652], [-0.1851407, -0.0, 0.9307652], [0.0, -0.196, 0.46]],None, Some([[0.764, 0.872, 0.528, 1.0], [0.636, 0.872, 0.528, 1.0], [0.602, 0.918, 0.286, 1.0]]), None, None, None),
            Vertex::new([[-0.1851407, -0.0, 0.9307652], [-0.5272362, -0.0, 0.7890647], [-0.1760344, -0.196, 0.4249846]],None, Some([[0.636, 0.872, 0.528, 1.0], [0.528, 0.872, 0.55, 1.0], [0.365, 0.918, 0.286, 1.0]]), None, None, None),
            Vertex::new([[-0.5272362, -0.0, 0.7890647], [-0.7890647, -0.0, 0.5272362], [-0.3252691, -0.196, 0.3252691]],None, Some([[0.528, 0.872, 0.55, 1.0], [0.528, 0.872, 0.679, 1.0], [0.286, 0.918, 0.444, 1.0]]), None, None, None),
            Vertex::new([[-0.7890647, -0.0, 0.5272362], [-0.9307652, -0.0, 0.1851407], [-0.4249846, -0.196, 0.1760344]],None, Some([[0.528, 0.872, 0.679, 1.0], [0.528, 0.872, 0.807, 1.0], [0.286, 0.918, 0.681, 1.0]]), None, None, None),
            Vertex::new([[-0.9307652, -0.0, 0.1851407], [-0.9307652, -0.0, -0.1851407], [-0.46, -0.196, 0.0]],None, Some([[0.528, 0.872, 0.807, 1.0], [0.528, 0.807, 0.872, 1.0], [0.286, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.9307652, -0.0, -0.1851407], [-0.7890647, -0.0, -0.5272362], [-0.4249846, -0.196, -0.1760344]],None, Some([[0.528, 0.807, 0.872, 1.0], [0.528, 0.679, 0.872, 1.0], [0.286, 0.681, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.7890647, -0.0, -0.5272362], [-0.5272362, -0.0, -0.7890647], [-0.3252691, -0.196, -0.3252691]],None, Some([[0.528, 0.679, 0.872, 1.0], [0.528, 0.55, 0.872, 1.0], [0.286, 0.444, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.5272362, -0.0, -0.7890647], [-0.1851407, -0.0, -0.9307652], [-0.1760344, -0.196, -0.4249846]],None, Some([[0.528, 0.55, 0.872, 1.0], [0.636, 0.528, 0.872, 1.0], [0.365, 0.286, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.1851407, -0.0, -0.9307652], [0.1851407, -0.0, -0.9307652], [-0.0, -0.196, -0.46]],None, Some([[0.636, 0.528, 0.872, 1.0], [0.764, 0.528, 0.872, 1.0], [0.602, 0.286, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.1851407, -0.0, -0.9307652], [0.5272362, -0.0, -0.7890647], [0.1760344, -0.196, -0.4249846]],None, Some([[0.764, 0.528, 0.872, 1.0], [0.872, 0.528, 0.85, 1.0], [0.839, 0.286, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.5272362, -0.0, -0.7890647], [0.7890647, -0.0, -0.5272362], [0.3252691, -0.196, -0.3252691]],None, Some([[0.872, 0.528, 0.85, 1.0], [0.872, 0.528, 0.721, 1.0], [0.918, 0.286, 0.76, 1.0]]), None, None, None),
            Vertex::new([[0.7890647, -0.0, -0.5272362], [0.9307652, -0.0, -0.1851407], [0.4249846, -0.196, -0.1760344]],None, Some([[0.872, 0.528, 0.721, 1.0], [0.872, 0.528, 0.593, 1.0], [0.918, 0.286, 0.523, 1.0]]), None, None, None),
            Vertex::new([[0.9307652, -0.0, -0.1851407], [0.9307652, -0.0, 0.1851407], [0.46, -0.196, 0.0]],None, Some([[0.872, 0.528, 0.593, 1.0], [0.872, 0.593, 0.528, 1.0], [0.918, 0.286, 0.286, 1.0]]), None, None, None),
            Vertex::new([[0.46, 0.196, 0.0], [0.4249846, 0.196, 0.1760344], [0.9307652, -0.0, 0.1851407]],None, Some([[0.958, 0.638, 0.638, 1.0], [0.958, 0.758, 0.638, 1.0], [0.872, 0.593, 0.528, 1.0]]), None, None, None),
            Vertex::new([[0.4249846, 0.196, 0.1760344], [0.3252691, 0.196, 0.3252691], [0.7890647, -0.0, 0.5272362]],None, Some([[0.958, 0.758, 0.638, 1.0], [0.958, 0.878, 0.638, 1.0], [0.872, 0.721, 0.528, 1.0]]), None, None, None),
            Vertex::new([[0.3252691, 0.196, 0.3252691], [0.1760344, 0.196, 0.4249846], [0.5272362, -0.0, 0.7890647]],None, Some([[0.958, 0.878, 0.638, 1.0], [0.918, 0.958, 0.638, 1.0], [0.872, 0.85, 0.528, 1.0]]), None, None, None),
            Vertex::new([[0.1760344, 0.196, 0.4249846], [0.0, 0.196, 0.46], [0.1851407, -0.0, 0.9307652]],None, Some([[0.918, 0.958, 0.638, 1.0], [0.798, 0.958, 0.638, 1.0], [0.764, 0.872, 0.528, 1.0]]), None, None, None),
            Vertex::new([[0.0, 0.196, 0.46], [-0.1760344, 0.196, 0.4249846], [-0.1851407, -0.0, 0.9307652]],None, Some([[0.798, 0.958, 0.638, 1.0], [0.678, 0.958, 0.638, 1.0], [0.636, 0.872, 0.528, 1.0]]), None, None, None),
            Vertex::new([[-0.1760344, 0.196, 0.4249846], [-0.3252691, 0.196, 0.3252691], [-0.5272362, -0.0, 0.7890647]],None, Some([[0.678, 0.958, 0.638, 1.0], [0.638, 0.958, 0.718, 1.0], [0.528, 0.872, 0.55, 1.0]]), None, None, None),
            Vertex::new([[-0.3252691, 0.196, 0.3252691], [-0.4249846, 0.196, 0.1760344], [-0.7890647, -0.0, 0.5272362]],None, Some([[0.638, 0.958, 0.718, 1.0], [0.638, 0.958, 0.838, 1.0], [0.528, 0.872, 0.679, 1.0]]), None, None, None),
            Vertex::new([[-0.4249846, 0.196, 0.1760344], [-0.46, 0.196, 0.0], [-0.9307652, -0.0, 0.1851407]],None, Some([[0.638, 0.958, 0.838, 1.0], [0.638, 0.958, 0.958, 1.0], [0.528, 0.872, 0.807, 1.0]]), None, None, None),
            Vertex::new([[-0.46, 0.196, 0.0], [-0.4249846, 0.196, -0.1760344], [-0.9307652, -0.0, -0.1851407]],None, Some([[0.638, 0.958, 0.958, 1.0], [0.638, 0.838, 0.958, 1.0], [0.528, 0.807, 0.872, 1.0]]), None, None, None),
            Vertex::new([[-0.4249846, 0.196, -0.1760344], [-0.3252691, 0.196, -0.3252691], [-0.7890647, -0.0, -0.5272362]],None, Some([[0.638, 0.838, 0.958, 1.0], [0.638, 0.718, 0.958, 1.0], [0.528, 0.679, 0.872, 1.0]]), None, None, None),
            Vertex::new([[-0.3252691, 0.196, -0.3252691], [-0.1760344, 0.196, -0.4249846], [-0.5272362, -0.0, -0.7890647]],None, Some([[0.638, 0.718, 0.958, 1.0], [0.678, 0.638, 0.958, 1.0], [0.528, 0.55, 0.872, 1.0]]), None, None, None),
            Vertex::new([[-0.1760344, 0.196, -0.4249846], [-0.0, 0.196, -0.46], [-0.1851407, -0.0, -0.9307652]],None, Some([[0.678, 0.638, 0.958, 1.0], [0.798, 0.638, 0.958, 1.0], [0.636, 0.528, 0.872, 1.0]]), None, None, None),
            Vertex::new([[-0.0, 0.196, -0.46], [0.1760344, 0.196, -0.4249846], [0.1851407, -0.0, -0.9307652]],None, Some([[0.798, 0.638, 0.958, 1.0], [0.918, 0.638, 0.958, 1.0], [0.764, 0.528, 0.872, 1.0]]), None, None, None),
            Vertex::new([[0.1760344, 0.196, -0.4249846], [0.3252691, 0.196, -0.3252691], [0.5272362, -0.0, -0.7890647]],None, Some([[0.918, 0.638, 0.958, 1.0], [0.958, 0.638, 0.878, 1.0], [0.872, 0.528, 0.85, 1.0]]), None, None, None),
            Vertex::new([[0.3252691, 0.196, -0.3252691], [0.4249846, 0.196, -0.1760344], [0.7890647, -0.0, -0.5272362]],None, Some([[0.958, 0.638, 0.878, 1.0], [0.958, 0.638, 0.758, 1.0], [0.872, 0.528, 0.721, 1.0]]), None, None, None),
            Vertex::new([[0.4249846, 0.196, -0.1760344], [0.46, 0.196, 0.0], [0.9307652, -0.0, -0.1851407]],None, Some([[0.958, 0.638, 0.758, 1.0], [0.958, 0.638, 0.638, 1.0], [0.872, 0.528, 0.593, 1.0]]), None, None, None),
            Vertex::new([[0.7890647, -0.0, 0.5272362], [0.9307652, -0.0, 0.1851407], [0.4249846, 0.196, 0.1760344]],None, Some([[0.872, 0.721, 0.528, 1.0], [0.872, 0.593, 0.528, 1.0], [0.958, 0.758, 0.638, 1.0]]), None, None, None),
            Vertex::new([[0.5272362, -0.0, 0.7890647], [0.7890647, -0.0, 0.5272362], [0.3252691, 0.196, 0.3252691]],None, Some([[0.872, 0.85, 0.528, 1.0], [0.872, 0.721, 0.528, 1.0], [0.958, 0.878, 0.638, 1.0]]), None, None, None),
            Vertex::new([[0.1851407, -0.0, 0.9307652], [0.5272362, -0.0, 0.7890647], [0.1760344, 0.196, 0.4249846]],None, Some([[0.764, 0.872, 0.528, 1.0], [0.872, 0.85, 0.528, 1.0], [0.918, 0.958, 0.638, 1.0]]), None, None, None),
            Vertex::new([[-0.1851407, -0.0, 0.9307652], [0.1851407, -0.0, 0.9307652], [0.0, 0.196, 0.46]],None, Some([[0.636, 0.872, 0.528, 1.0], [0.764, 0.872, 0.528, 1.0], [0.798, 0.958, 0.638, 1.0]]), None, None, None),
            Vertex::new([[-0.5272362, -0.0, 0.7890647], [-0.1851407, -0.0, 0.9307652], [-0.1760344, 0.196, 0.4249846]],None, Some([[0.528, 0.872, 0.55, 1.0], [0.636, 0.872, 0.528, 1.0], [0.678, 0.958, 0.638, 1.0]]), None, None, None),
            Vertex::new([[-0.7890647, -0.0, 0.5272362], [-0.5272362, -0.0, 0.7890647], [-0.3252691, 0.196, 0.3252691]],None, Some([[0.528, 0.872, 0.679, 1.0], [0.528, 0.872, 0.55, 1.0], [0.638, 0.958, 0.718, 1.0]]), None, None, None),
            Vertex::new([[-0.9307652, -0.0, 0.1851407], [-0.7890647, -0.0, 0.5272362], [-0.4249846, 0.196, 0.1760344]],None, Some([[0.528, 0.872, 0.807, 1.0], [0.528, 0.872, 0.679, 1.0], [0.638, 0.958, 0.838, 1.0]]), None, None, None),
            Vertex::new([[-0.9307652, -0.0, -0.1851407], [-0.9307652, -0.0, 0.1851407], [-0.46, 0.196, 0.0]],None, Some([[0.528, 0.807, 0.872, 1.0], [0.528, 0.872, 0.807, 1.0], [0.638, 0.958, 0.958, 1.0]]), None, None, None),
            Vertex::new([[-0.7890647, -0.0, -0.5272362], [-0.9307652, -0.0, -0.1851407], [-0.4249846, 0.196, -0.1760344]],None, Some([[0.528, 0.679, 0.872, 1.0], [0.528, 0.807, 0.872, 1.0], [0.638, 0.838, 0.958, 1.0]]), None, None, None),
            Vertex::new([[-0.5272362, -0.0, -0.7890647], [-0.7890647, -0.0, -0.5272362], [-0.3252691, 0.196, -0.3252691]],None, Some([[0.528, 0.55, 0.872, 1.0], [0.528, 0.679, 0.872, 1.0], [0.638, 0.718, 0.958, 1.0]]), None, None, None),
            Vertex::new([[-0.1851407, -0.0, -0.9307652], [-0.5272362, -0.0, -0.7890647], [-0.1760344, 0.196, -0.4249846]],None, Some([[0.636, 0.528, 0.872, 1.0], [0.528, 0.55, 0.872, 1.0], [0.678, 0.638, 0.958, 1.0]]), None, None, None),
            Vertex::new([[0.1851407, -0.0, -0.9307652], [-0.1851407, -0.0, -0.9307652], [-0.0, 0.196, -0.46]],None, Some([[0.764, 0.528, 0.872, 1.0], [0.636, 0.528, 0.872, 1.0], [0.798, 0.638, 0.958, 1.0]]), None, None, None),
            Vertex::new([[0.5272362, -0.0, -0.7890647], [0.1851407, -0.0, -0.9307652], [0.1760344, 0.196, -0.4249846]],None, Some([[0.872, 0.528, 0.85, 1.0], [0.764, 0.528, 0.872, 1.0], [0.918, 0.638, 0.958, 1.0]]), None, None, None),
            Vertex::new([[0.7890647, -0.0, -0.5272362], [0.5272362, -0.0, -0.7890647], [0.3252691, 0.196, -0.3252691]],None, Some([[0.872, 0.528, 0.721, 1.0], [0.872, 0.528, 0.85, 1.0], [0.958, 0.638, 0.878, 1.0]]), None, None, None),
            Vertex::new([[0.9307652, -0.0, -0.1851407], [0.7890647, -0.0, -0.5272362], [0.4249846, 0.196, -0.1760344]],None, Some([[0.872, 0.528, 0.593, 1.0], [0.872, 0.528, 0.721, 1.0], [0.958, 0.638, 0.758, 1.0]]), None, None, None),
            Vertex::new([[0.9307652, -0.0, 0.1851407], [0.9307652, -0.0, -0.1851407], [0.46, 0.196, 0.0]],None, Some([[0.872, 0.593, 0.528, 1.0], [0.872, 0.528, 0.593, 1.0], [0.958, 0.638, 0.638, 1.0]]), None, None, None),
            Vertex::new([[0.4249846, 0.196, 0.1760344], [0.46, 0.196, 0.0], [0.2942356, 0.4, 0.0585271]],None, Some([[0.958, 0.758, 0.638, 1.0], [0.958, 0.638, 0.638, 1.0], [0.986, 0.846, 0.813, 1.0]]), None, None, None),
            Vertex::new([[0.3252691, 0.196, 0.3252691], [0.4249846, 0.196, 0.1760344], [0.2494409, 0.4, 0.1666711]],None, Some([[0.958, 0.878, 0.638, 1.0], [0.958, 0.758, 0.638, 1.0], [0.986, 0.911, 0.813, 1.0]]), None, None, None),
            Vertex::new([[0.1760344, 0.196, 0.4249846], [0.3252691, 0.196, 0.3252691], [0.1666711, 0.4, 0.2494409]],None, Some([[0.918, 0.958, 0.638, 1.0], [0.958, 0.878, 0.638, 1.0], [0.986, 0.976, 0.813, 1.0]]), None, None, None),
            Vertex::new([[0.0, 0.196, 0.46], [0.1760344, 0.196, 0.4249846], [0.0585271, 0.4, 0.2942356]],None, Some([[0.798, 0.958, 0.638, 1.0], [0.918, 0.958, 0.638, 1.0], [0.932, 0.986, 0.813, 1.0]]), None, None, None),
            Vertex::new([[-0.1760344, 0.196, 0.4249846], [0.0, 0.196, 0.46], [-0.0585271, 0.4, 0.2942356]],None, Some([[0.678, 0.958, 0.638, 1.0], [0.798, 0.958, 0.638, 1.0], [0.868, 0.986, 0.813, 1.0]]), None, None, None),
            Vertex::new([[-0.3252691, 0.196, 0.3252691], [-0.1760344, 0.196, 0.4249846], [-0.1666711, 0.4, 0.2494409]],None, Some([[0.638, 0.958, 0.718, 1.0], [0.678, 0.958, 0.638, 1.0], [0.813, 0.986, 0.824, 1.0]]), None, None, None),
            Vertex::new([[-0.4249846, 0.196, 0.1760344], [-0.3252691, 0.196, 0.3252691], [-0.2494409, 0.4, 0.1666711]],None, Some([[0.638, 0.958, 0.838, 1.0], [0.638, 0.958, 0.718, 1.0], [0.813, 0.986, 0.889, 1.0]]), None, None, None),
            Vertex::new([[-0.46, 0.196, 0.0], [-0.4249846, 0.196, 0.1760344], [-0.2942356, 0.4, 0.0585271]],None, Some([[0.638, 0.958, 0.958, 1.0], [0.638, 0.958, 0.838, 1.0], [0.813, 0.986, 0.954, 1.0]]), None, None, None),
            Vertex::new([[-0.4249846, 0.196, -0.1760344], [-0.46, 0.196, 0.0], [-0.2942356, 0.4, -0.0585271]],None, Some([[0.638, 0.838, 0.958, 1.0], [0.638, 0.958, 0.958, 1.0], [0.813, 0.954, 0.986, 1.0]]), None, None, None),
            Vertex::new([[-0.3252691, 0.196, -0.3252691], [-0.4249846, 0.196, -0.1760344], [-0.2494409, 0.4, -0.1666711]],None, Some([[0.638, 0.718, 0.958, 1.0], [0.638, 0.838, 0.958, 1.0], [0.813, 0.889, 0.986, 1.0]]), None, None, None),
            Vertex::new([[-0.1760344, 0.196, -0.4249846], [-0.3252691, 0.196, -0.3252691], [-0.1666711, 0.4, -0.2494409]],None, Some([[0.678, 0.638, 0.958, 1.0], [0.638, 0.718, 0.958, 1.0], [0.813, 0.824, 0.986, 1.0]]), None, None, None),
            Vertex::new([[-0.0, 0.196, -0.46], [-0.1760344, 0.196, -0.4249846], [-0.0585271, 0.4, -0.2942356]],None, Some([[0.798, 0.638, 0.958, 1.0], [0.678, 0.638, 0.958, 1.0], [0.868, 0.813, 0.986, 1.0]]), None, None, None),
            Vertex::new([[0.1760344, 0.196, -0.4249846], [-0.0, 0.196, -0.46], [0.0585271, 0.4, -0.2942356]],None, Some([[0.918, 0.638, 0.958, 1.0], [0.798, 0.638, 0.958, 1.0], [0.932, 0.813, 0.986, 1.0]]), None, None, None),
            Vertex::new([[0.3252691, 0.196, -0.3252691], [0.1760344, 0.196, -0.4249846], [0.1666711, 0.4, -0.2494409]],None, Some([[0.958, 0.638, 0.878, 1.0], [0.918, 0.638, 0.958, 1.0], [0.986, 0.813, 0.976, 1.0]]), None, None, None),
            Vertex::new([[0.4249846, 0.196, -0.1760344], [0.3252691, 0.196, -0.3252691], [0.2494409, 0.4, -0.1666711]],None, Some([[0.958, 0.638, 0.758, 1.0], [0.958, 0.638, 0.878, 1.0], [0.986, 0.813, 0.911, 1.0]]), None, None, None),
            Vertex::new([[0.46, 0.196, 0.0], [0.4249846, 0.196, -0.1760344], [0.2942356, 0.4, -0.0585271]],None, Some([[0.958, 0.638, 0.638, 1.0], [0.958, 0.638, 0.758, 1.0], [0.986, 0.813, 0.846, 1.0]]), None, None, None),
            Vertex::new([[0.2942356, 0.4, 0.0585271], [0.2494409, 0.4, 0.1666711], [0.4249846, 0.196, 0.1760344]],None, Some([[0.986, 0.846, 0.813, 1.0], [0.986, 0.911, 0.813, 1.0], [0.958, 0.758, 0.638, 1.0]]), None, None, None),
            Vertex::new([[0.2494409, 0.4, 0.1666711], [0.1666711, 0.4, 0.2494409], [0.3252691, 0.196, 0.3252691]],None, Some([[0.986, 0.911, 0.813, 1.0], [0.986, 0.976, 0.813, 1.0], [0.958, 0.878, 0.638, 1.0]]), None, None, None),
            Vertex::new([[0.1666711, 0.4, 0.2494409], [0.0585271, 0.4, 0.2942356], [0.1760344, 0.196, 0.4249846]],None, Some([[0.986, 0.976, 0.813, 1.0], [0.932, 0.986, 0.813, 1.0], [0.918, 0.958, 0.638, 1.0]]), None, None, None),
            Vertex::new([[0.0585271, 0.4, 0.2942356], [-0.0585271, 0.4, 0.2942356], [0.0, 0.196, 0.46]],None, Some([[0.932, 0.986, 0.813, 1.0], [0.868, 0.986, 0.813, 1.0], [0.798, 0.958, 0.638, 1.0]]), None, None, None),
            Vertex::new([[-0.0585271, 0.4, 0.2942356], [-0.1666711, 0.4, 0.2494409], [-0.1760344, 0.196, 0.4249846]],None, Some([[0.868, 0.986, 0.813, 1.0], [0.813, 0.986, 0.824, 1.0], [0.678, 0.958, 0.638, 1.0]]), None, None, None),
            Vertex::new([[-0.1666711, 0.4, 0.2494409], [-0.2494409, 0.4, 0.1666711], [-0.3252691, 0.196, 0.3252691]],None, Some([[0.813, 0.986, 0.824, 1.0], [0.813, 0.986, 0.889, 1.0], [0.638, 0.958, 0.718, 1.0]]), None, None, None),
            Vertex::new([[-0.2494409, 0.4, 0.1666711], [-0.2942356, 0.4, 0.0585271], [-0.4249846, 0.196, 0.1760344]],None, Some([[0.813, 0.986, 0.889, 1.0], [0.813, 0.986, 0.954, 1.0], [0.638, 0.958, 0.838, 1.0]]), None, None, None),
            Vertex::new([[-0.2942356, 0.4, 0.0585271], [-0.2942356, 0.4, -0.0585271], [-0.46, 0.196, 0.0]],None, Some([[0.813, 0.986, 0.954, 1.0], [0.813, 0.954, 0.986, 1.0], [0.638, 0.958, 0.958, 1.0]]), None, None, None),
            Vertex::new([[-0.2942356, 0.4, -0.0585271], [-0.2494409, 0.4, -0.1666711], [-0.4249846, 0.196, -0.1760344]],None, Some([[0.813, 0.954, 0.986, 1.0], [0.813, 0.889, 0.986, 1.0], [0.638, 0.838, 0.958, 1.0]]), None, None, None),
            Vertex::new([[-0.2494409, 0.4, -0.1666711], [-0.1666711, 0.4, -0.2494409], [-0.3252691, 0.196, -0.3252691]],None, Some([[0.813, 0.889, 0.986, 1.0], [0.813, 0.824, 0.986, 1.0], [0.638, 0.718, 0.958, 1.0]]), None, None, None),
            Vertex::new([[-0.1666711, 0.4, -0.2494409], [-0.0585271, 0.4, -0.2942356], [-0.1760344, 0.196, -0.4249846]],None, Some([[0.813, 0.824, 0.986, 1.0], [0.868, 0.813, 0.986, 1.0], [0.678, 0.638, 0.958, 1.0]]), None, None, None),
            Vertex::new([[-0.0585271, 0.4, -0.2942356], [0.0585271, 0.4, -0.2942356], [-0.0, 0.196, -0.46]],None, Some([[0.868, 0.813, 0.986, 1.0], [0.932, 0.813, 0.986, 1.0], [0.798, 0.638, 0.958, 1.0]]), None, None, None),
            Vertex::new([[0.0585271, 0.4, -0.2942356], [0.1666711, 0.4, -0.2494409], [0.1760344, 0.196, -0.4249846]],None, Some([[0.932, 0.813, 0.986, 1.0], [0.986, 0.813, 0.976, 1.0], [0.918, 0.638, 0.958, 1.0]]), None, None, None),
            Vertex::new([[0.1666711, 0.4, -0.2494409], [0.2494409, 0.4, -0.1666711], [0.3252691, 0.196, -0.3252691]],None, Some([[0.986, 0.813, 0.976, 1.0], [0.986, 0.813, 0.911, 1.0], [0.958, 0.638, 0.878, 1.0]]), None, None, None),
            Vertex::new([[0.2494409, 0.4, -0.1666711], [0.2942356, 0.4, -0.0585271], [0.4249846, 0.196, -0.1760344]],None, Some([[0.986, 0.813, 0.911, 1.0], [0.986, 0.813, 0.846, 1.0], [0.958, 0.638, 0.758, 1.0]]), None, None, None),
            Vertex::new([[0.2942356, 0.4, -0.0585271], [0.2942356, 0.4, 0.0585271], [0.46, 0.196, 0.0]],None, Some([[0.986, 0.813, 0.846, 1.0], [0.986, 0.846, 0.813, 1.0], [0.958, 0.638, 0.638, 1.0]]), None, None, None),
            Vertex::new([[0.3292503, 0.23, 0.2739637], [0.3791081, 0.23, 0.1993463], [0.6004535, 0.502142, 0.4012102]],None, Some([[0.964, 0.666, 0.666, 1.0], [0.964, 0.666, 0.666, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.3791081, 0.23, 0.1993463], [0.2913362, 0.332, 0.1946646], [0.6004535, 0.502142, 0.4012102]],None, Some([[0.964, 0.666, 0.666, 1.0], [0.979, 0.753, 0.753, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.2913362, 0.332, 0.1946646], [0.3292503, 0.23, 0.2739637], [0.6004535, 0.502142, 0.4012102]],None, Some([[0.979, 0.753, 0.753, 1.0], [0.964, 0.666, 0.666, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.2739637, 0.23, 0.3292503], [-0.1993463, 0.23, 0.3791081], [-0.4012102, 0.502142, 0.6004535]],None, Some([[0.964, 0.666, 0.666, 1.0], [0.964, 0.666, 0.666, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.1993463, 0.23, 0.3791081], [-0.1946646, 0.332, 0.2913362], [-0.4012102, 0.502142, 0.6004535]],None, Some([[0.964, 0.666, 0.666, 1.0], [0.979, 0.753, 0.753, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.1946646, 0.332, 0.2913362], [-0.2739637, 0.23, 0.3292503], [-0.4012102, 0.502142, 0.6004535]],None, Some([[0.979, 0.753, 0.753, 1.0], [0.964, 0.666, 0.666, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.3292503, 0.23, -0.2739637], [-0.3791081, 0.23, -0.1993463], [-0.6004535, 0.502142, -0.4012102]],None, Some([[0.964, 0.666, 0.666, 1.0], [0.964, 0.666, 0.666, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.3791081, 0.23, -0.1993463], [-0.2913362, 0.332, -0.1946646], [-0.6004535, 0.502142, -0.4012102]],None, Some([[0.964, 0.666, 0.666, 1.0], [0.979, 0.753, 0.753, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[-0.2913362, 0.332, -0.1946646], [-0.3292503, 0.23, -0.2739637], [-0.6004535, 0.502142, -0.4012102]],None, Some([[0.979, 0.753, 0.753, 1.0], [0.964, 0.666, 0.666, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.2739637, 0.23, -0.3292503], [0.1993463, 0.23, -0.3791081], [0.4012102, 0.502142, -0.6004535]],None, Some([[0.964, 0.666, 0.666, 1.0], [0.964, 0.666, 0.666, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.1993463, 0.23, -0.3791081], [0.1946646, 0.332, -0.2913362], [0.4012102, 0.502142, -0.6004535]],None, Some([[0.964, 0.666, 0.666, 1.0], [0.979, 0.753, 0.753, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.1946646, 0.332, -0.2913362], [0.2739637, 0.23, -0.3292503], [0.4012102, 0.502142, -0.6004535]],None, Some([[0.979, 0.753, 0.753, 1.0], [0.964, 0.666, 0.666, 1.0], [0.984, 0.918, 0.918, 1.0]]), None, None, None),
            Vertex::new([[0.115, 0.486, 0.0], [0.1062461, 0.486, 0.0440086], [0.2942356, 0.4, 0.0585271]],None, Some([[0.997, 0.889, 0.889, 1.0], [0.997, 0.929, 0.889, 1.0], [0.986, 0.846, 0.813, 1.0]]), None, None, None),
            Vertex::new([[0.1062461, 0.486, 0.0440086], [0.0813173, 0.486, 0.0813173], [0.2494409, 0.4, 0.1666711]],None, Some([[0.997, 0.929, 0.889, 1.0], [0.997, 0.97, 0.889, 1.0], [0.986, 0.911, 0.813, 1.0]]), None, None, None),
            Vertex::new([[0.0813173, 0.486, 0.0813173], [0.0440086, 0.486, 0.1062461], [0.1666711, 0.4, 0.2494409]],None, Some([[0.997, 0.97, 0.889, 1.0], [0.984, 0.997, 0.889, 1.0], [0.986, 0.976, 0.813, 1.0]]), None, None, None),
            Vertex::new([[0.0440086, 0.486, 0.1062461], [0.0, 0.486, 0.115], [0.0585271, 0.4, 0.2942356]],None, Some([[0.984, 0.997, 0.889, 1.0], [0.943, 0.997, 0.889, 1.0], [0.932, 0.986, 0.813, 1.0]]), None, None, None),
            Vertex::new([[0.0, 0.486, 0.115], [-0.0440086, 0.486, 0.1062461], [-0.0585271, 0.4, 0.2942356]],None, Some([[0.943, 0.997, 0.889, 1.0], [0.902, 0.997, 0.889, 1.0], [0.868, 0.986, 0.813, 1.0]]), None, None, None),
            Vertex::new([[-0.0440086, 0.486, 0.1062461], [-0.0813173, 0.486, 0.0813173], [-0.1666711, 0.4, 0.2494409]],None, Some([[0.902, 0.997, 0.889, 1.0], [0.889, 0.997, 0.916, 1.0], [0.813, 0.986, 0.824, 1.0]]), None, None, None),
            Vertex::new([[-0.0813173, 0.486, 0.0813173], [-0.1062461, 0.486, 0.0440086], [-0.2494409, 0.4, 0.1666711]],None, Some([[0.889, 0.997, 0.916, 1.0], [0.889, 0.997, 0.957, 1.0], [0.813, 0.986, 0.889, 1.0]]), None, None, None),
            Vertex::new([[-0.1062461, 0.486, 0.0440086], [-0.115, 0.486, 0.0], [-0.2942356, 0.4, 0.0585271]],None, Some([[0.889, 0.997, 0.957, 1.0], [0.889, 0.997, 0.997, 1.0], [0.813, 0.986, 0.954, 1.0]]), None, None, None),
            Vertex::new([[-0.115, 0.486, 0.0], [-0.1062461, 0.486, -0.0440086], [-0.2942356, 0.4, -0.0585271]],None, Some([[0.889, 0.997, 0.997, 1.0], [0.889, 0.957, 0.997, 1.0], [0.813, 0.954, 0.986, 1.0]]), None, None, None),
            Vertex::new([[-0.1062461, 0.486, -0.0440086], [-0.0813173, 0.486, -0.0813173], [-0.2494409, 0.4, -0.1666711]],None, Some([[0.889, 0.957, 0.997, 1.0], [0.889, 0.916, 0.997, 1.0], [0.813, 0.889, 0.986, 1.0]]), None, None, None),
            Vertex::new([[-0.0813173, 0.486, -0.0813173], [-0.0440086, 0.486, -0.1062461], [-0.1666711, 0.4, -0.2494409]],None, Some([[0.889, 0.916, 0.997, 1.0], [0.902, 0.889, 0.997, 1.0], [0.813, 0.824, 0.986, 1.0]]), None, None, None),
            Vertex::new([[-0.0440086, 0.486, -0.1062461], [-0.0, 0.486, -0.115], [-0.0585271, 0.4, -0.2942356]],None, Some([[0.902, 0.889, 0.997, 1.0], [0.943, 0.889, 0.997, 1.0], [0.868, 0.813, 0.986, 1.0]]), None, None, None),
            Vertex::new([[-0.0, 0.486, -0.115], [0.0440086, 0.486, -0.1062461], [0.0585271, 0.4, -0.2942356]],None, Some([[0.943, 0.889, 0.997, 1.0], [0.984, 0.889, 0.997, 1.0], [0.932, 0.813, 0.986, 1.0]]), None, None, None),
            Vertex::new([[0.0440086, 0.486, -0.1062461], [0.0813173, 0.486, -0.0813173], [0.1666711, 0.4, -0.2494409]],None, Some([[0.984, 0.889, 0.997, 1.0], [0.997, 0.889, 0.97, 1.0], [0.986, 0.813, 0.976, 1.0]]), None, None, None),
            Vertex::new([[0.0813173, 0.486, -0.0813173], [0.1062461, 0.486, -0.0440086], [0.2494409, 0.4, -0.1666711]],None, Some([[0.997, 0.889, 0.97, 1.0], [0.997, 0.889, 0.929, 1.0], [0.986, 0.813, 0.911, 1.0]]), None, None, None),
            Vertex::new([[0.1062461, 0.486, -0.0440086], [0.115, 0.486, 0.0], [0.2942356, 0.4, -0.0585271]],None, Some([[0.997, 0.889, 0.929, 1.0], [0.997, 0.889, 0.889, 1.0], [0.986, 0.813, 0.846, 1.0]]), None, None, None),
            Vertex::new([[0.2494409, 0.4, 0.1666711], [0.2942356, 0.4, 0.0585271], [0.1062461, 0.486, 0.0440086]],None, Some([[0.986, 0.911, 0.813, 1.0], [0.986, 0.846, 0.813, 1.0], [0.997, 0.929, 0.889, 1.0]]), None, None, None),
            Vertex::new([[0.1666711, 0.4, 0.2494409], [0.2494409, 0.4, 0.1666711], [0.0813173, 0.486, 0.0813173]],None, Some([[0.986, 0.976, 0.813, 1.0], [0.986, 0.911, 0.813, 1.0], [0.997, 0.97, 0.889, 1.0]]), None, None, None),
            Vertex::new([[0.0585271, 0.4, 0.2942356], [0.1666711, 0.4, 0.2494409], [0.0440086, 0.486, 0.1062461]],None, Some([[0.932, 0.986, 0.813, 1.0], [0.986, 0.976, 0.813, 1.0], [0.984, 0.997, 0.889, 1.0]]), None, None, None),
            Vertex::new([[-0.0585271, 0.4, 0.2942356], [0.0585271, 0.4, 0.2942356], [0.0, 0.486, 0.115]],None, Some([[0.868, 0.986, 0.813, 1.0], [0.932, 0.986, 0.813, 1.0], [0.943, 0.997, 0.889, 1.0]]), None, None, None),
            Vertex::new([[-0.1666711, 0.4, 0.2494409], [-0.0585271, 0.4, 0.2942356], [-0.0440086, 0.486, 0.1062461]],None, Some([[0.813, 0.986, 0.824, 1.0], [0.868, 0.986, 0.813, 1.0], [0.902, 0.997, 0.889, 1.0]]), None, None, None),
            Vertex::new([[-0.2494409, 0.4, 0.1666711], [-0.1666711, 0.4, 0.2494409], [-0.0813173, 0.486, 0.0813173]],None, Some([[0.813, 0.986, 0.889, 1.0], [0.813, 0.986, 0.824, 1.0], [0.889, 0.997, 0.916, 1.0]]), None, None, None),
            Vertex::new([[-0.2942356, 0.4, 0.0585271], [-0.2494409, 0.4, 0.1666711], [-0.1062461, 0.486, 0.0440086]],None, Some([[0.813, 0.986, 0.954, 1.0], [0.813, 0.986, 0.889, 1.0], [0.889, 0.997, 0.957, 1.0]]), None, None, None),
            Vertex::new([[-0.2942356, 0.4, -0.0585271], [-0.2942356, 0.4, 0.0585271], [-0.115, 0.486, 0.0]],None, Some([[0.813, 0.954, 0.986, 1.0], [0.813, 0.986, 0.954, 1.0], [0.889, 0.997, 0.997, 1.0]]), None, None, None),
            Vertex::new([[-0.2494409, 0.4, -0.1666711], [-0.2942356, 0.4, -0.0585271], [-0.1062461, 0.486, -0.0440086]],None, Some([[0.813, 0.889, 0.986, 1.0], [0.813, 0.954, 0.986, 1.0], [0.889, 0.957, 0.997, 1.0]]), None, None, None),
            Vertex::new([[-0.1666711, 0.4, -0.2494409], [-0.2494409, 0.4, -0.1666711], [-0.0813173, 0.486, -0.0813173]],None, Some([[0.813, 0.824, 0.986, 1.0], [0.813, 0.889, 0.986, 1.0], [0.889, 0.916, 0.997, 1.0]]), None, None, None),
            Vertex::new([[-0.0585271, 0.4, -0.2942356], [-0.1666711, 0.4, -0.2494409], [-0.0440086, 0.486, -0.1062461]],None, Some([[0.868, 0.813, 0.986, 1.0], [0.813, 0.824, 0.986, 1.0], [0.902, 0.889, 0.997, 1.0]]), None, None, None),
            Vertex::new([[0.0585271, 0.4, -0.2942356], [-0.0585271, 0.4, -0.2942356], [-0.0, 0.486, -0.115]],None, Some([[0.932, 0.813, 0.986, 1.0], [0.868, 0.813, 0.986, 1.0], [0.943, 0.889, 0.997, 1.0]]), None, None, None),
            Vertex::new([[0.1666711, 0.4, -0.2494409], [0.0585271, 0.4, -0.2942356], [0.0440086, 0.486, -0.1062461]],None, Some([[0.986, 0.813, 0.976, 1.0], [0.932, 0.813, 0.986, 1.0], [0.984, 0.889, 0.997, 1.0]]), None, None, None),
            Vertex::new([[0.2494409, 0.4, -0.1666711], [0.1666711, 0.4, -0.2494409], [0.0813173, 0.486, -0.0813173]],None, Some([[0.986, 0.813, 0.911, 1.0], [0.986, 0.813, 0.976, 1.0], [0.997, 0.889, 0.97, 1.0]]), None, None, None),
            Vertex::new([[0.2942356, 0.4, -0.0585271], [0.2494409, 0.4, -0.1666711], [0.1062461, 0.486, -0.0440086]],None, Some([[0.986, 0.813, 0.846, 1.0], [0.986, 0.813, 0.911, 1.0], [0.997, 0.889, 0.929, 1.0]]), None, None, None),
            Vertex::new([[0.2942356, 0.4, 0.0585271], [0.2942356, 0.4, -0.0585271], [0.115, 0.486, 0.0]],None, Some([[0.986, 0.846, 0.813, 1.0], [0.986, 0.813, 0.846, 1.0], [0.997, 0.889, 0.889, 1.0]]), None, None, None),
            Vertex::new([[0.115, 0.486, 0.0], [0.1062461, 0.486, 0.0440086], [0.0, 0.5, 0.0]],None, Some([[0.997, 0.889, 0.889, 1.0], [0.997, 0.929, 0.889, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[0.1062461, 0.486, 0.0440086], [0.0813173, 0.486, 0.0813173], [0.0, 0.5, 0.0]],None, Some([[0.997, 0.929, 0.889, 1.0], [0.997, 0.97, 0.889, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[0.0813173, 0.486, 0.0813173], [0.0440086, 0.486, 0.1062461], [0.0, 0.5, 0.0]],None, Some([[0.997, 0.97, 0.889, 1.0], [0.984, 0.997, 0.889, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[0.0440086, 0.486, 0.1062461], [0.0, 0.486, 0.115], [0.0, 0.5, 0.0]],None, Some([[0.984, 0.997, 0.889, 1.0], [0.943, 0.997, 0.889, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[0.0, 0.486, 0.115], [-0.0440086, 0.486, 0.1062461], [0.0, 0.5, 0.0]],None, Some([[0.943, 0.997, 0.889, 1.0], [0.902, 0.997, 0.889, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[-0.0440086, 0.486, 0.1062461], [-0.0813173, 0.486, 0.0813173], [0.0, 0.5, 0.0]],None, Some([[0.902, 0.997, 0.889, 1.0], [0.889, 0.997, 0.916, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[-0.0813173, 0.486, 0.0813173], [-0.1062461, 0.486, 0.0440086], [0.0, 0.5, 0.0]],None, Some([[0.889, 0.997, 0.916, 1.0], [0.889, 0.997, 0.957, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[-0.1062461, 0.486, 0.0440086], [-0.115, 0.486, 0.0], [0.0, 0.5, 0.0]],None, Some([[0.889, 0.997, 0.957, 1.0], [0.889, 0.997, 0.997, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[-0.115, 0.486, 0.0], [-0.1062461, 0.486, -0.0440086], [0.0, 0.5, 0.0]],None, Some([[0.889, 0.997, 0.997, 1.0], [0.889, 0.957, 0.997, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[-0.1062461, 0.486, -0.0440086], [-0.0813173, 0.486, -0.0813173], [0.0, 0.5, 0.0]],None, Some([[0.889, 0.957, 0.997, 1.0], [0.889, 0.916, 0.997, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[-0.0813173, 0.486, -0.0813173], [-0.0440086, 0.486, -0.1062461], [0.0, 0.5, 0.0]],None, Some([[0.889, 0.916, 0.997, 1.0], [0.902, 0.889, 0.997, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[-0.0440086, 0.486, -0.1062461], [-0.0, 0.486, -0.115], [0.0, 0.5, 0.0]],None, Some([[0.902, 0.889, 0.997, 1.0], [0.943, 0.889, 0.997, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[-0.0, 0.486, -0.115], [0.0440086, 0.486, -0.1062461], [0.0, 0.5, 0.0]],None, Some([[0.943, 0.889, 0.997, 1.0], [0.984, 0.889, 0.997, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[0.0440086, 0.486, -0.1062461], [0.0813173, 0.486, -0.0813173], [0.0, 0.5, 0.0]],None, Some([[0.984, 0.889, 0.997, 1.0], [0.997, 0.889, 0.97, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[0.0813173, 0.486, -0.0813173], [0.1062461, 0.486, -0.0440086], [0.0, 0.5, 0.0]],None, Some([[0.997, 0.889, 0.97, 1.0], [0.997, 0.889, 0.929, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None),
            Vertex::new([[0.1062461, 0.486, -0.0440086], [0.115, 0.486, 0.0], [0.0, 0.5, 0.0]],None, Some([[0.997, 0.889, 0.929, 1.0], [0.997, 0.889, 0.889, 1.0], [1.0, 0.9, 0.9, 1.0]]), None, None, None)

        ]);
    figure1.move_matrix[[3, 2]] += 4.0;
    figure1._changed.move_matrix = true;

    let mut vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        figure1.get_vertex(surface.window().inner_size()).into_iter(),
    )
        .unwrap();

    println!("{}", vertex_buffer.len());

    let vs = vs::load(device.clone()).expect("failed to create shader module");
    let fs = fs::load(device.clone()).expect("failed to create shader module");

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: surface.window().inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
        surface.window().inner_size(),
    );

    let mut command_buffers = get_command_buffers(
        device.clone(),
        queue.clone(),
        pipeline,
        &framebuffers,
        vertex_buffer.clone(),
    );

    // ash::vk::ExtLineRasterizationFnCopy

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    let mut mouse_button_pressed = false;
    let mut mouse_wheel_pressed = false;
    let mut last_mouse_pos: (f64, f64) = (-1.0f64, -1.0f64);
    let mut window_width = surface.window().inner_size().width as f64;
    let mut window_height = surface.window().inner_size().height as f64;
    let mut keyboard_pressed: HashMap<VirtualKeyCode, bool> = HashMap::new();
    let move_rules = HashMap::from([
        (VirtualKeyCode::A, (3, 0, -1)),
        (VirtualKeyCode::D, (3, 0, 1)),
        (VirtualKeyCode::W, (3, 1, -1)),
        (VirtualKeyCode::S, (3, 1, 1)),
        (VirtualKeyCode::Z, (3, 2, -1)),
        (VirtualKeyCode::X, (3, 2, 1)),
    ]);
    let projection_rules: HashMap<VirtualKeyCode, i32> = HashMap::from([
        (VirtualKeyCode::Key1, 1),
    ]);


    event_loop.run(move |event, _, control_flow| match event {
        WinitEvent::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = WinitControlFlow::Exit;
        }
        WinitEvent::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            window_resized = true;
            recreate_swapchain = true;
        }
        WinitEvent::MainEventsCleared => {
            if window_resized || recreate_swapchain {
                recreate_swapchain = false;

                let new_dimensions = surface.window().inner_size();

                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: new_dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };
                swapchain = new_swapchain;
                let new_framebuffers = get_framebuffers(
                    &new_images,
                    render_pass.clone(),
                    device.clone(),
                    surface.window().inner_size(),
                );

                if window_resized {
                    window_resized = false;

                    viewport.dimensions = new_dimensions.into();
                    let new_pipeline = get_pipeline(
                        device.clone(),
                        vs.clone(),
                        fs.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                        surface.window().inner_size(),
                    );
                    vertex_buffer = CpuAccessibleBuffer::from_iter(
                        device.clone(),
                        BufferUsage::vertex_buffer(),
                        false,
                        figure1.get_vertex(surface.window().inner_size()).into_iter(),
                    )
                        .unwrap();
                    command_buffers = get_command_buffers(
                        device.clone(),
                        queue.clone(),
                        new_pipeline,
                        &new_framebuffers,
                        vertex_buffer.clone(),
                    );
                    window_width = surface.window().inner_size().width as f64;
                    window_height = surface.window().inner_size().height as f64;
                }
            }

            let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            // wait for the fence related to this image to finish (normally this would be the oldest fence)
            if let Some(image_fence) = &fences[image_i] {
                image_fence.wait(None).unwrap();
            }

            let previous_future = match fences[previous_fence_i].clone() {
                // Create a NowFuture
                None => {
                    let mut now = sync::now(device.clone());
                    now.cleanup_finished();

                    now.boxed()
                }
                // Use the existing FenceSignalFuture
                Some(fence) => fence.boxed(),
            };

            let future = previous_future
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffers[image_i].clone())
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_i)
                .then_signal_fence_and_flush();

            fences[image_i] = match future {
                Ok(value) => Some(Arc::new(value)),
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    None
                }
                Err(e) => {
                    println!("Failed to flush future : {:?}", e);
                    None
                }
            };

            previous_fence_i = image_i;
        }
        WinitEvent::WindowEvent {
            event: WindowEvent::ReceivedCharacter(code),
            ..
        } => {
            // NdIndex.
            // https://docs.rs/winit/latest/winit/event/enum.WindowEvent.html#variant.ReceivedCharacter
            // println!("+ {:},", code);
            // let mut changes = true;
            // let mn: f32 = 0.05;
            // println!("{:}", figure1.move_matrix);
            // match event {
            //     WinitEvent::WindowEvent {
            //         event: WindowEvent::ReceivedCharacter('a'),
            //         ..
            //     } => {
            //         // let mut z = figure1.move_matrix.slice(s![.., 2]);
            //         // z += 1;
            //         figure1.move_matrix[[3, 0]] -= mn;
            //         figure1._changed.move_matrix = true;
            //     }
            //     WinitEvent::WindowEvent {
            //         event: WindowEvent::ReceivedCharacter('d'),
            //         ..
            //     } => {
            //         figure1.move_matrix[[3, 0]] += mn;
            //         figure1._changed.move_matrix = true;
            //     }
            //     WinitEvent::WindowEvent {
            //         event: WindowEvent::ReceivedCharacter('w'),
            //         ..
            //     } => {
            //         figure1.move_matrix[[3, 1]] -= mn;
            //         figure1._changed.move_matrix = true;
            //     }
            //     WinitEvent::WindowEvent {
            //         event: WindowEvent::ReceivedCharacter('s'),
            //         ..
            //     } => {
            //         figure1.move_matrix[[3, 1]] += mn;
            //         figure1._changed.move_matrix = true;
            //     }
            //     WinitEvent::WindowEvent {
            //         event: WindowEvent::ReceivedCharacter('z'),
            //         ..
            //     } => {
            //         figure1.move_matrix[[3, 2]] -= mn;
            //         figure1._changed.move_matrix = true;
            //     }
            //     WinitEvent::WindowEvent {
            //         event: WindowEvent::ReceivedCharacter('x'),
            //         ..
            //     } => {
            //         figure1.move_matrix[[3, 2]] += mn;
            //         figure1._changed.move_matrix = true;
            //     }
            //     _ => {
            //         changes = false;
            //     }
            // }
            // if changes {
            //     recreate_swapchain = true;
            //     window_resized = true;
            // }
            // vertex_buffer = CpuAccessibleBuffer::from_iter(
            //     device.clone(),
            //     BufferUsage::vertex_buffer(),
            //     false,
            //     figure1.get_vertex(surface.window().inner_size()).into_iter(),
            // )
            //     .unwrap();
        }
        WinitEvent::WindowEvent {
            event: WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    scancode,
                    virtual_keycode: Some(v_code),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            },
            ..
        } => {
            println!("Keycode {:} or {:?} Pressed ", scancode, v_code);
            keyboard_pressed.insert(v_code, true);
            let mn = 0.01;
            let mut changes = false;
            for (keycode, (i, j, sign)) in &move_rules {
                match keyboard_pressed.get(keycode) {
                    Some(true) => {
                        figure1.move_matrix[[i.clone(), j.clone()]] += mn * sign.clone() as f32;
                        figure1._changed.move_matrix = true;
                        changes = true;
                    }
                    _ => {}
                }
            }
            println!("{}", figure1.move_matrix);
            let mut no_projections = true;
            for (keycode, projection_code) in &projection_rules {
                match keyboard_pressed.get(keycode) {
                    Some(true) => {
                        if figure1.projection_mode != projection_code.clone() {
                            figure1.projection_mode = projection_code.clone();
                            figure1._changed.projection_flag = true;
                            changes = true;
                            no_projections = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            // if no_projections &&  figure1.projection_mode != 0{
            //     figure1.projection_mode = 0;
            //     figure1._changed.projection_flag = true;
            //     changes = true;
            // }

            if changes {
                recreate_swapchain = true;
                window_resized = true;
            }
        }
        WinitEvent::WindowEvent {
            event: WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    scancode,
                    virtual_keycode: Some(v_code),
                    state: ElementState::Released,
                    ..
                },
                ..
            },
            ..
        } => {
            println!("Keycode {:} or {:?} Released", scancode, v_code);
            keyboard_pressed.insert(v_code, false);

            let mut changes = false;
            let mut no_projections = true;
            for (keycode, projection_code) in &projection_rules {
                match keyboard_pressed.get(keycode) {
                    Some(false) => {
                        if figure1.projection_mode == projection_code.clone() {
                            figure1.projection_mode = 0;
                            figure1._changed.projection_flag = true;
                            changes = true;
                            no_projections = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if changes {
                recreate_swapchain = true;
                window_resized = true;
            }
        }

        WinitEvent::WindowEvent {
            event: WindowEvent::MouseWheel {
                delta, ..
            },
            ..
        } => {
            let mn = 0.1;
            if mouse_wheel_pressed {
                match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        figure1.rotate_angels[2] += (x + y) * mn;
                        figure1._changed.rotate_angels = true;
                    }
                    _ => {}
                }
            } else {
                match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        println!("* x={:}, y={:}", x, y);
                        if y > 0.0 {
                            figure1.scale[0] *= 1.0 + y / 10.0;
                            figure1.scale[1] *= 1.0 + y / 10.0;
                            figure1.scale[2] *= 1.0 + y / 10.0;
                        } else if y < 0.0 {
                            figure1.scale[0] /= 1.0 - y / 10.0;
                            figure1.scale[1] /= 1.0 - y / 10.0;
                            figure1.scale[2] /= 1.0 - y / 10.0;
                        }
                        // figure1.move_matrix[0][0] *= 1.1;
                        figure1._changed.scale = true;
                    }
                    _ => {}
                }
            }

            if figure1._changed.any() {
                recreate_swapchain = true;
                window_resized = true;

                // vertex_buffer = CpuAccessibleBuffer::from_iter(
                //     device.clone(),
                //     BufferUsage::vertex_buffer(),
                //     false,
                //     figure1.get_vertex(surface.window().inner_size()).into_iter(),
                // )
                //     .unwrap();
            }
        }
        WinitEvent::WindowEvent {
            event: WindowEvent::MouseInput { button, state: ElementState::Pressed, .. }, ..
        } => {
            println!("mouse button is {:?} Pressed", button);
            match button {
                MouseButton::Left => {
                    mouse_button_pressed = true;

                    // figure1.rotate_angels[2] -= 0.1;
                    // figure1._changed.rotate_angels = true;
                }
                MouseButton::Right => {

                    // figure1.rotate_angels[2] += 0.1;
                    // figure1._changed.rotate_angels = true;
                }
                MouseButton::Middle => {
                    mouse_wheel_pressed = true;
                }
                _ => {}
            }

            if figure1._changed.any() {
                recreate_swapchain = true;
                window_resized = true;

                // vertex_buffer = CpuAccessibleBuffer::from_iter(
                //     device.clone(),
                //     BufferUsage::vertex_buffer(),
                //     false,
                //     figure1.get_vertex(surface.window().inner_size()).into_iter(),
                // )
                //     .unwrap();
            }
        }
        WinitEvent::WindowEvent {
            event: WindowEvent::MouseInput { button, state: ElementState::Released, .. }, ..
        } => {
            println!("mouse button is {:?} Released", button);
            match button {
                MouseButton::Left => {
                    mouse_button_pressed = false;
                    last_mouse_pos = (-1.0f64, -1.0f64);
                    // figure1.rotate_angels[2] -= 0.1;
                    // figure1._changed.rotate_angels = true;
                }
                MouseButton::Right => {

                    // figure1.rotate_angels[2] += 0.1;
                    // figure1._changed.rotate_angels = true;
                }
                MouseButton::Middle => {
                    mouse_wheel_pressed = false;
                }
                _ => {}
            }
        }
        WinitEvent::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. }, ..
        } => {
            if mouse_button_pressed {
                if 0f64 <= position.x && position.x <= window_width
                    && 0f64 <= position.y && position.y <= window_height {
                    let sensitivity = 0.01;

                    if last_mouse_pos != (-1.0f64, -1.0f64) {
                        println!("pos vec({:}, {:}) || <{:?}>", position.x - last_mouse_pos.0, position.y - last_mouse_pos.1, position);
                        figure1.rotate_angels[1] -= ((position.x - last_mouse_pos.0) * sensitivity) as f32;
                        figure1.rotate_angels[0] -= ((position.y - last_mouse_pos.1) * sensitivity) as f32;
                        figure1._changed.rotate_angels = true;
                        if figure1._changed.any() {
                            recreate_swapchain = true;
                            window_resized = true;
                        }
                    }
                    last_mouse_pos.0 = position.x;
                    last_mouse_pos.1 = position.y;
                } else {
                    if last_mouse_pos != (-1.0f64, -1.0f64) {
                        last_mouse_pos = (-1.0f64, -1.0f64);
                    }
                }
            }
        }
        // WinitEvent::WindowEvent {
        //     event: WindowEvent::Re, ..
        // } => {
        //     let mn = 0.1;
        //     let mut changes = false;
        //     for (keycode, (i, j, sign)) in &move_rules {
        //         match keyboard_pressed.get(keycode) {
        //             Some(true) => {
        //                 figure1.move_matrix[[i, j]] += mn * sign;
        //                 figure1._changed.move_matrix = true;
        //                 changes = true;
        //             }
        //             _ => {}
        //         }
        //     }
        //
        //     if changes {
        //         recreate_swapchain = true;
        //         window_resized = true;
        //     }
        // }
        // WinitEvent::WindowEvent {
        //     event: WindowEvent::CursorMoved { modifiers: winit::ModifiersState{}, ..}, ..
        // } => {
        //
        // }
        _ => (),
    });
}
/*
0.1253 0.1771
 */