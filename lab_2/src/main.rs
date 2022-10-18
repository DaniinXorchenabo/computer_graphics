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
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
};
use vulkano::sync::{self, FenceSignalFuture, FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::PhysicalSize;
use winit::event::{Event as WinitEvent, Event, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow as WinitControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use vulkano::shader::SpecializationConstants;
use vulkano::shader::SpecializationMapEntry;

// extern crate ndarray;

use ndarray::{arr2, arr3,  Array2, array, Array};
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
    position: [[f32; 2]; 3],
    // move_matrix: [[f32; 3]; 3],
    move_matrix: [[f32; 4]; 4],
    contour: [f32; 3],
    contour_colors: [[f32; 4]; 3],
    point_colors: [[f32; 4]; 3],
}

impl Vertex {
    pub fn new(
        points: [[f32; 2]; 3],
        contour: Option<[f32; 3]>,
        point_colors: Option<[[f32; 4]; 3]>,
        point_color: Option<[f32; 4]>,
        contour_color: Option<[f32; 4]>,
        contour_colors: Option<[[f32; 4]; 3]>,
    ) -> Self {
        Vertex {
            position: points,
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
                (_, _) => [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            },
            point_colors: match (point_colors, point_color) {
                (Some(x), _) => x,
                (None, Some(y)) => [y.clone(), y.clone(), y],
                (_, _) => [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
            },
        }
    }
}

#[derive(Default, Clone)]
pub struct Changed {
    move_matrix: bool,
    rotate_angels: bool,
    scale: bool,
}

impl Changed {
    pub fn new() -> Self {
        Changed {
            move_matrix: false,
            rotate_angels: false,
            scale: false,
        }
    }

    pub fn any(&self) -> bool {
        self.move_matrix | self.rotate_angels | self.scale
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
    // _changed: bool
}

impl Figure {
    pub fn new(new_polygons: Vec<Vertex>) -> Self {
        let mut real_polygons = Vec::new();
        let default_matrix = [[1.0, 0.0, 0.0 ], [0.0, 1.0, 0.0 ], [0.0, 0.0, 1.0 ]];
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
            move_matrix:  Array::eye(4),
            rotate_angels: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            _rotate_matrix:  Array::eye(4),
            _changed: Changed::new(),
            // _changed: false,
        }
    }

    fn get_vertex(&mut self) -> Vec<Vertex> {
        if self._changed.any() {
            if self._changed.rotate_angels {
                let (x, y, z) = (self.rotate_angels[0], self.rotate_angels[1], self.rotate_angels[2]);
                let (a, b, c, d, e, f) = (f32::cos(x), f32::sin(x), f32::cos(y), f32::sin(y), f32::cos(z), f32::sin(z));
                // self._rotate_matrix = array![
                //     [c * e, -c * f, -d, 0.0],
                //     [-b * d * e + a * f, b * d * f + a * e, -b * c, 0.0],
                //     [a * d * e + b * f, -a * d * f + b * e, a * c, 0.0],
                //     [0.0, 0.0, 0.0, 1.0]
                // ];
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
                self._rotate_matrix = array![
                    [e, -f, 0.0, 0.0],
                    [f, e, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ];
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
            ] ;


            self.change_matrix = self.change_matrix.dot(&self.move_matrix);
            // self.change_matrix = self.change_matrix.clone() * self.move_matrix.clone();

            // self.change_matrix = self.change_matrix.clone() * self._rotate_matrix.clone();
            self.change_matrix = self.change_matrix.dot(&self._rotate_matrix);

            let mut loc_change_matrix:[[f32; 4]; 4] = [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0]
            ];
                // let mut loc_change_matrix:[[f32; 3]; 3] = [
                //     [1.0, 0.0, 0.0],
                //     [0.0, 1.0, 0.0],
                //     [0.0, 0.0, 1.0]
                // ];
            // change_matrix

            for ((i, j), item) in self.change_matrix.indexed_iter(){
                loc_change_matrix[i][j] = item.clone();
            }


            let mut res = self.polygons.clone();



            for vertex in &mut res {
                // vertex.move_matrix = change_matrix.clone()
                vertex.move_matrix = loc_change_matrix.clone()
            }
            res
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
    float c_x = position[gl_VertexIndex % 3].x;
    float c_y = position[gl_VertexIndex % 3].y;

    vec4 pos_m = vec4(c_x, c_y, 0.0, 1.0)  * move_matrix ;

    vec4 pos0 = vec4(position[0].x, position[0].y, 0.0, 1.0) * move_matrix ;
    vec4 pos1 = vec4(position[1].x, position[1].y, 0.0, 1.0) * move_matrix ;
    vec4 pos2 = vec4(position[2].x, position[2].y, 0.0, 1.0) * move_matrix;

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
    contour_size = vec3(contour[0] == 0.0 ? 0.0: max( contour[0] * length(pos0.xyz - pos1.xyz) / length(position[0].xy - position[1].xy), 1.4),
                        contour[1] == 0.0 ? 0.0: max( contour[1] * length(pos1.xyz - pos2.xyz) / length(position[1].xy - position[2].xy), 1.4),
                        contour[2] == 0.0 ? 0.0: max( contour[2] * length(pos2.xyz - pos0.xyz) / length(position[2].xy - position[0].xy), 1.4));
    // contour_size = vec3(contour[0], contour[1], contour[2]);
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
layout(location = 15) in mat3 points;
layout(location = 19) in vec4[3] contour_colors_fr;




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
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
        .unwrap()
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
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
                        clear_values: vec![Some([1.0, 1.0, 1.0, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
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
    let framebuffers = get_framebuffers(&images, render_pass.clone());

    vulkano::impl_vertex!(Vertex, position, move_matrix, contour, contour_colors, point_colors);

    let mut figure1 = Figure::new(
        vec![
            Vertex::new(
                [
                    [-0.5, -0.5],
                    [-0.6, -0.1],
                    [-0.0, 0.0]
                ],
                Some([10.0, 20.0, 0.0]), None, Some([1.0, 0.0, 0.0, 1.0]), None, None),
            Vertex::new(
                [
                    [0.5, 0.5],
                    [0.5, 0.1],
                    [0.1, 0.1],
                ],
                Some([10.0, 10.0, 10.0]), None, None, None, Some([[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])),
            Vertex::new(
                [
                    [-0.5, -0.5],
                    [-0.5, -0.9],
                    [-0.9, -0.9],
                ],
                None, Some([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]), None, None, None),
            Vertex::new(
                [
                    [0.9, -0.5],
                    [0.5, -0.9],
                    [0.9, -0.9],
                ],
                None, Some([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]]), None, None, None),
            Vertex::new(
                [
                    [-0.5, 0.9],
                    [-0.5, 0.5],
                    [-0.9, 0.5]
                ],
                None, None, None, None, None),
        ]);

    let mut vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        figure1.get_vertex().into_iter(),
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
                let new_framebuffers = get_framebuffers(&new_images, render_pass.clone());

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
                    command_buffers = get_command_buffers(
                        device.clone(),
                        queue.clone(),
                        new_pipeline,
                        &new_framebuffers,
                        vertex_buffer.clone(),
                    );
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
            println!("+ {:},", code);
            let mut changes = true;
            let mn: f32 = 0.05;
            match event {
                WinitEvent::WindowEvent {
                    event: WindowEvent::ReceivedCharacter('a'),
                    ..
                } => {
                    // let mut z = figure1.move_matrix.slice(s![.., 2]);
                    // z += 1;
                    figure1.move_matrix[[0, 3]] -=  mn;
                    figure1._changed.move_matrix = true;
                }
                WinitEvent::WindowEvent {
                    event: WindowEvent::ReceivedCharacter('d'),
                    ..
                } => {
                    figure1.move_matrix[[0, 3]] += mn;
                    figure1._changed.move_matrix = true;
                }
                WinitEvent::WindowEvent {
                    event: WindowEvent::ReceivedCharacter('w'),
                    ..
                } => {
                    figure1.move_matrix[[1, 3]] -= mn;
                    figure1._changed.move_matrix = true;
                }
                WinitEvent::WindowEvent {
                    event: WindowEvent::ReceivedCharacter('s'),
                    ..
                } => {
                    figure1.move_matrix[[1, 3]] += mn;
                    figure1._changed.move_matrix = true;
                }
                _ => {
                    changes = false;
                }
            }
            if changes {
                recreate_swapchain = true;
                window_resized = true;

                vertex_buffer = CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::vertex_buffer(),
                    false,
                    figure1.get_vertex().into_iter(),
                )
                    .unwrap();
            }
        }
        WinitEvent::WindowEvent {
            event: WindowEvent::MouseWheel {
                delta, ..
            },
            ..
        } => {
            match delta {
                MouseScrollDelta::LineDelta(x, y) => {
                    println!("* x={:}, y={:}", x, y);
                    if y > 0.0 {
                        figure1.scale[0] *= 1.0 + y / 10.0;
                        figure1.scale[1] *= 1.0 + y / 10.0;
                    } else if y < 0.0 {
                        figure1.scale[0] /= 1.0 - y / 10.0;
                        figure1.scale[1] /= 1.0 - y / 10.0;
                    }
                    // figure1.move_matrix[0][0] *= 1.1;
                    figure1._changed.scale = true;
                }
                _ => {}
            }

            if figure1._changed.any() {
                recreate_swapchain = true;
                window_resized = true;

                vertex_buffer = CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::vertex_buffer(),
                    false,
                    figure1.get_vertex().into_iter(),
                )
                    .unwrap();
            }
        }
        WinitEvent::WindowEvent {
            event: WindowEvent::MouseInput { button, .. }, ..
        } => {
            println!("mouse button is {:?}", button);
            match button {
                MouseButton::Left => {
                    figure1.rotate_angels[2] -= 0.1;
                    figure1._changed.rotate_angels = true;
                }
                MouseButton::Right => {
                    figure1.rotate_angels[2] += 0.1;
                    figure1._changed.rotate_angels = true;
                }
                _ => {}
            }

            if figure1._changed.any() {
                recreate_swapchain = true;
                window_resized = true;

                vertex_buffer = CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::vertex_buffer(),
                    false,
                    figure1.get_vertex().into_iter(),
                )
                    .unwrap();
            }
        }
        _ => (),
    });
}
