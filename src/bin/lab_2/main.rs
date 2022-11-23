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

    fn get_vertex(&mut self, windows_size: PhysicalSize<u32>) -> Vec<Vertex> {
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
            // self.change_matrix = array![
            //     [self.scale[0] / 1.0_f32.max(windows_size.width as f32 / windows_size.height as f32), 0.0, 0.0, 0.0],
            //     [0.0, self.scale[1] / 1.0_f32.max(windows_size.height as f32 / windows_size.width as f32), 0.0, 0.0],
            //     [0.0, 0.0, self.scale[2], 0.0],
            //     [0.0, 0.0, 0.0, 1.0]
            // ] ;

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
        path: "src/bin/lab_2/vertex.glsl",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/lab_2/fragments.glsl",
        // src: ""
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
            Vertex::new([[-0.025945945945945948, -0.5416216216216216], [-0.02756756756756757, -0.2545945945945946], [0.0891891891891892, -0.5383783783783784]],None, None, None, None, None),
            Vertex::new([[0.1848648648648649, -0.5172972972972973], [0.0891891891891892, -0.5383783783783784], [-0.02756756756756757, -0.2545945945945946]],None, None, None, None, None),
            Vertex::new([[-0.02756756756756757, -0.25621621621621626], [0.1848648648648649, -0.5172972972972973], [0.28702702702702704, -0.48000000000000004]],None, None, None, None, None),
            Vertex::new([[-0.02756756756756757, -0.25621621621621626], [0.28702702702702704, -0.48000000000000004], [0.3713513513513514, -0.43621621621621626]],None, None, None, None, None),
            Vertex::new([[-0.02756756756756757, -0.25621621621621626], [0.3713513513513514, -0.43621621621621626], [0.43459459459459465, -0.3875675675675676]],None, None, None, None, None),
            Vertex::new([[0.43459459459459465, -0.3875675675675676], [0.4897297297297298, -0.3275675675675676], [-0.02756756756756757, -0.25621621621621626]],None, None, None, None, None),
            Vertex::new([[-0.02756756756756757, -0.25621621621621626], [0.4897297297297298, -0.3275675675675676], [0.5335135135135135, -0.2724324324324325]],None, None, None, None, None),
            Vertex::new([[-0.02756756756756757, -0.25621621621621626], [0.5335135135135135, -0.2724324324324325], [0.5740540540540541, -0.20594594594594595]],None, None, None, None, None),
            Vertex::new([[0.5740540540540541, -0.20594594594594595], [-0.02756756756756757, -0.25621621621621626], [0.5108108108108108, -0.19621621621621624]],None, None, None, None, None),
            Vertex::new([[0.5108108108108108, -0.19621621621621624], [-0.02756756756756757, -0.25621621621621626], [0.4394594594594595, -0.1864864864864865]],None, None, None, None, None),
            Vertex::new([[0.43783783783783786, -0.1864864864864865], [-0.02756756756756757, -0.25621621621621626], [0.36162162162162165, -0.17513513513513515]],None, None, None, None, None),
            Vertex::new([[0.36162162162162165, -0.17513513513513515], [-0.02756756756756757, -0.25621621621621626], [0.26594594594594595, -0.1637837837837838]],None, None, None, None, None),
            Vertex::new([[0.26594594594594595, -0.1637837837837838], [-0.02756756756756757, -0.25621621621621626], [0.1848648648648649, -0.1572972972972973]],None, None, None, None, None),
            Vertex::new([[0.1848648648648649, -0.1572972972972973], [-0.02756756756756757, -0.25621621621621626], [0.11351351351351352, -0.15405405405405406]],None, None, None, None, None),
            Vertex::new([[0.11351351351351352, -0.15405405405405406], [-0.02756756756756757, -0.25621621621621626], [0.025945945945945948, -0.15081081081081082]],None, None, None, None, None),
            Vertex::new([[0.025945945945945948, -0.15081081081081082], [-0.02756756756756757, -0.25621621621621626], [-0.02756756756756757, -0.1491891891891892]],None, None, None, None, None),
            Vertex::new([[-0.02756756756756757, -0.1491891891891892], [-0.02756756756756757, -0.25621621621621626], [-0.08108108108108109, -0.1491891891891892]],None, None, None, None, None),
            Vertex::new([[-0.08108108108108109, -0.1491891891891892], [-0.02756756756756757, -0.25621621621621626], [-0.16864864864864867, -0.15243243243243246]],None, None, None, None, None),
            Vertex::new([[-0.02756756756756757, -0.25621621621621626], [-0.16864864864864867, -0.15243243243243246], [-0.24162162162162165, -0.1556756756756757]],None, None, None, None, None),
            Vertex::new([[-0.24000000000000002, -0.1556756756756757], [-0.02756756756756757, -0.25621621621621626], [-0.31945945945945947, -0.16216216216216217]],None, None, None, None, None),
            Vertex::new([[-0.31945945945945947, -0.16216216216216217], [-0.02756756756756757, -0.25621621621621626], [-0.41837837837837843, -0.17351351351351352]],None, None, None, None, None),
            Vertex::new([[-0.4167567567567568, -0.17351351351351352], [-0.02756756756756757, -0.25621621621621626], [-0.49621621621621625, -0.1848648648648649]],None, None, None, None, None),
            Vertex::new([[-0.49459459459459465, -0.1848648648648649], [-0.02756756756756757, -0.25621621621621626], [-0.5643243243243243, -0.1945945945945946]],None, None, None, None, None),
            Vertex::new([[-0.5627027027027027, -0.1945945945945946], [-0.02756756756756757, -0.25621621621621626], [-0.6291891891891892, -0.20432432432432435]],None, None, None, None, None),
            Vertex::new([[-0.6291891891891892, -0.20432432432432435], [-0.02756756756756757, -0.25621621621621626], [-0.5886486486486487, -0.2708108108108108]],None, None, None, None, None),
            Vertex::new([[-0.5886486486486487, -0.2708108108108108], [-0.02756756756756757, -0.25621621621621626], [-0.544864864864865, -0.3275675675675676]],None, None, None, None, None),
            Vertex::new([[-0.544864864864865, -0.3275675675675676], [-0.02756756756756757, -0.25621621621621626], [-0.4897297297297298, -0.3875675675675676]],None, None, None, None, None),
            Vertex::new([[-0.4897297297297298, -0.3875675675675676], [-0.02756756756756757, -0.25621621621621626], [-0.4264864864864865, -0.43783783783783786]],None, None, None, None, None),
            Vertex::new([[-0.4264864864864865, -0.43783783783783786], [-0.02756756756756757, -0.25621621621621626], [-0.34216216216216216, -0.48162162162162164]],None, None, None, None, None),
            Vertex::new([[-0.34216216216216216, -0.48162162162162164], [-0.02756756756756757, -0.25621621621621626], [-0.23837837837837839, -0.518918918918919]],None, None, None, None, None),
            Vertex::new([[-0.23837837837837839, -0.518918918918919], [-0.02756756756756757, -0.25621621621621626], [-0.14432432432432435, -0.54]],None, None, None, None, None),
            Vertex::new([[-0.14432432432432435, -0.54], [-0.02756756756756757, -0.25621621621621626], [-0.025945945945945948, -0.5416216216216216]],None, None, None, None, None),
            Vertex::new([[0.7913513513513514, -0.14270270270270272], [0.5724324324324325, -0.20594594594594595], [0.544864864864865, -0.2010810810810811]],None, None, None, None, None),
            Vertex::new([[0.5091891891891892, -0.19621621621621624], [0.544864864864865, -0.2010810810810811], [0.7200000000000001, -0.11351351351351352]],None, None, None, None, None),
            Vertex::new([[0.7183783783783785, -0.11351351351351352], [0.5091891891891892, -0.19621621621621624], [0.48000000000000004, -0.192972972972973]],None, None, None, None, None),
            Vertex::new([[0.7183783783783785, -0.11351351351351352], [0.48000000000000004, -0.192972972972973], [0.6745945945945947, -0.0972972972972973]],None, None, None, None, None),
            Vertex::new([[0.48162162162162164, -0.192972972972973], [0.6745945945945947, -0.0972972972972973], [0.44108108108108113, -0.1864864864864865]],None, None, None, None, None),
            Vertex::new([[0.6745945945945947, -0.0972972972972973], [0.44108108108108113, -0.1864864864864865], [0.6178378378378379, -0.0745945945945946]],None, None, None, None, None),
            Vertex::new([[0.6178378378378379, -0.0745945945945946], [0.44108108108108113, -0.1864864864864865], [0.4102702702702703, -0.18162162162162163]],None, None, None, None, None),
            Vertex::new([[0.6178378378378379, -0.0745945945945946], [0.4102702702702703, -0.18162162162162163], [0.5481081081081082, -0.053513513513513515]],None, None, None, None, None),
            Vertex::new([[0.5481081081081082, -0.053513513513513515], [0.4102702702702703, -0.18162162162162163], [0.36162162162162165, -0.17513513513513515]],None, None, None, None, None),
            Vertex::new([[0.45567567567567574, -0.030810810810810812], [0.36162162162162165, -0.17513513513513515], [0.5481081081081082, -0.053513513513513515]],None, None, None, None, None),
            Vertex::new([[0.45567567567567574, -0.030810810810810812], [0.36162162162162165, -0.17513513513513515], [0.3940540540540541, -0.021081081081081084]],None, None, None, None, None),
            Vertex::new([[0.3940540540540541, -0.021081081081081084], [0.36162162162162165, -0.17513513513513515], [0.26432432432432434, -0.1637837837837838]],None, None, None, None, None),
            Vertex::new([[0.3940540540540541, -0.021081081081081084], [0.26432432432432434, -0.1637837837837838], [0.31783783783783787, -0.011351351351351352]],None, None, None, None, None),
            Vertex::new([[0.31783783783783787, -0.011351351351351352], [0.26432432432432434, -0.1637837837837838], [0.1848648648648649, -0.1572972972972973]],None, None, None, None, None),
            Vertex::new([[0.31783783783783787, -0.011351351351351352], [0.1848648648648649, -0.1572972972972973], [0.23675675675675678, -0.0048648648648648655]],None, None, None, None, None),
            Vertex::new([[0.23675675675675678, -0.0048648648648648655], [0.1848648648648649, -0.1572972972972973], [0.11351351351351352, -0.15405405405405406]],None, None, None, None, None),
            Vertex::new([[0.23675675675675678, -0.0048648648648648655], [0.11351351351351352, -0.15405405405405406], [0.12486486486486488, 0.0016216216216216218]],None, None, None, None, None),
            Vertex::new([[0.12486486486486488, 0.0016216216216216218], [0.11351351351351352, -0.15405405405405406], [0.025945945945945948, -0.15081081081081082]],None, None, None, None, None),
            Vertex::new([[0.12486486486486488, 0.0016216216216216218], [0.025945945945945948, -0.15081081081081082], [0.0372972972972973, 0.0032432432432432435]],None, None, None, None, None),
            Vertex::new([[0.038918918918918924, 0.0048648648648648655], [0.025945945945945948, -0.15081081081081082], [-0.02756756756756757, -0.1491891891891892]],None, None, None, None, None),
            Vertex::new([[0.038918918918918924, 0.0032432432432432435], [-0.02756756756756757, -0.15081081081081082], [-0.025945945945945948, 0.0032432432432432435]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.0032432432432432435], [-0.02756756756756757, -0.15081081081081082], [-0.10216216216216217, 0.0032432432432432435]],None, None, None, None, None),
            Vertex::new([[-0.10216216216216217, 0.0016216216216216218], [-0.02756756756756757, -0.1491891891891892], [-0.08108108108108109, -0.1491891891891892]],None, None, None, None, None),
            Vertex::new([[-0.10216216216216217, 0.0032432432432432435], [-0.08108108108108109, -0.1491891891891892], [-0.19945945945945948, 0.0032432432432432435]],None, None, None, None, None),
            Vertex::new([[-0.19783783783783784, 0.0032432432432432435], [-0.08108108108108109, -0.1491891891891892], [-0.16864864864864867, -0.15243243243243246]],None, None, None, None, None),
            Vertex::new([[-0.19783783783783784, 0.0016216216216216218], [-0.16864864864864867, -0.15243243243243246], [-0.28216216216216217, 0.0016216216216216218]],None, None, None, None, None),
            Vertex::new([[-0.24000000000000002, -0.1556756756756757], [-0.16864864864864867, -0.15243243243243246], [-0.28216216216216217, 0.0016216216216216218]],None, None, None, None, None),
            Vertex::new([[-0.28216216216216217, 0.0016216216216216218], [-0.24000000000000002, -0.1556756756756757], [-0.38270270270270274, -0.0048648648648648655]],None, None, None, None, None),
            Vertex::new([[-0.31945945945945947, -0.16216216216216217], [-0.24000000000000002, -0.1556756756756757], [-0.38270270270270274, -0.0048648648648648655]],None, None, None, None, None),
            Vertex::new([[-0.38270270270270274, -0.0048648648648648655], [-0.31945945945945947, -0.16216216216216217], [-0.47513513513513517, -0.01783783783783784]],None, None, None, None, None),
            Vertex::new([[-0.31945945945945947, -0.16216216216216217], [-0.4167567567567568, -0.17351351351351352], [-0.47513513513513517, -0.01783783783783784]],None, None, None, None, None),
            Vertex::new([[-0.47513513513513517, -0.01783783783783784], [-0.4167567567567568, -0.17351351351351352], [-0.544864864864865, -0.030810810810810812]],None, None, None, None, None),
            Vertex::new([[-0.4167567567567568, -0.17351351351351352], [-0.49459459459459465, -0.1848648648648649], [-0.544864864864865, -0.030810810810810812]],None, None, None, None, None),
            Vertex::new([[-0.49459459459459465, -0.1848648648648649], [-0.544864864864865, -0.030810810810810812], [-0.6454054054054055, -0.051891891891891896]],None, None, None, None, None),
            Vertex::new([[-0.5513513513513514, -0.192972972972973], [-0.49459459459459465, -0.1848648648648649], [-0.6454054054054055, -0.051891891891891896]],None, None, None, None, None),
            Vertex::new([[-0.5513513513513514, -0.192972972972973], [-0.6454054054054055, -0.051891891891891896], [-0.7394594594594596, -0.08108108108108109]],None, None, None, None, None),
            Vertex::new([[-0.5902702702702703, -0.19783783783783784], [-0.5513513513513514, -0.192972972972973], [-0.7394594594594596, -0.08108108108108109]],None, None, None, None, None),
            Vertex::new([[-0.5902702702702703, -0.19783783783783784], [-0.7394594594594596, -0.08108108108108109], [-0.8318918918918919, -0.11351351351351352]],None, None, None, None, None),
            Vertex::new([[-0.591891891891892, -0.19783783783783784], [-0.8318918918918919, -0.11351351351351352], [-0.8983783783783784, -0.14270270270270272]],None, None, None, None, None),
            Vertex::new([[-0.8983783783783784, -0.14270270270270272], [-0.591891891891892, -0.19783783783783784], [-0.6097297297297298, -0.2010810810810811]],None, None, None, None, None),
            Vertex::new([[-0.6113513513513514, -0.2010810810810811], [-0.8983783783783784, -0.14270270270270272], [-0.6291891891891892, -0.20432432432432435]],None, None, None, None, None),
            Vertex::new([[0.7881081081081082, -0.14270270270270272], [0.544864864864865, -0.2010810810810811], [0.7200000000000001, -0.11351351351351352]],None, None, None, None, None),
            Vertex::new([[-0.8302702702702703, -0.11351351351351352], [-0.9, -0.14432432432432435], [-0.7021621621621622, 0.02756756756756757]],None, None, None, None, None),
            Vertex::new([[-0.7378378378378379, -0.08108108108108109], [-0.7021621621621622, 0.02756756756756757], [-0.8318918918918919, -0.11351351351351352]],None, None, None, None, None),
            Vertex::new([[-0.7378378378378379, -0.08108108108108109], [-0.7021621621621622, 0.02756756756756757], [-0.6583783783783784, 0.04702702702702703]],None, None, None, None, None),
            Vertex::new([[-0.6454054054054055, -0.051891891891891896], [-0.6583783783783784, 0.04702702702702703], [-0.7362162162162162, -0.08108108108108109]],None, None, None, None, None),
            Vertex::new([[-0.5724324324324325, 0.06486486486486487], [-0.6454054054054055, -0.051891891891891896], [-0.6583783783783784, 0.04702702702702703]],None, None, None, None, None),
            Vertex::new([[-0.5432432432432432, -0.030810810810810812], [-0.5724324324324325, 0.06486486486486487], [-0.6454054054054055, -0.051891891891891896]],None, None, None, None, None),
            Vertex::new([[-0.5172972972972973, 0.0745945945945946], [-0.5432432432432432, -0.030810810810810812], [-0.5724324324324325, 0.06486486486486487]],None, None, None, None, None),
            Vertex::new([[-0.47513513513513517, -0.01783783783783784], [-0.5172972972972973, 0.0745945945945946], [-0.5432432432432432, -0.030810810810810812]],None, None, None, None, None),
            Vertex::new([[-0.45081081081081087, 0.08594594594594596], [-0.47513513513513517, -0.01783783783783784], [-0.5172972972972973, 0.0745945945945946]],None, None, None, None, None),
            Vertex::new([[-0.38270270270270274, -0.0048648648648648655], [-0.45081081081081087, 0.08594594594594596], [-0.47513513513513517, -0.01783783783783784]],None, None, None, None, None),
            Vertex::new([[-0.3745945945945946, 0.09243243243243245], [-0.38270270270270274, -0.0048648648648648655], [-0.45081081081081087, 0.08594594594594596]],None, None, None, None, None),
            Vertex::new([[-0.38270270270270274, -0.0048648648648648655], [-0.3745945945945946, 0.09243243243243245], [-0.28216216216216217, 0.0016216216216216218]],None, None, None, None, None),
            Vertex::new([[-0.3745945945945946, 0.09243243243243245], [-0.28216216216216217, 0.0016216216216216218], [-0.21081081081081082, 0.10054054054054055]],None, None, None, None, None),
            Vertex::new([[-0.1491891891891892, 0.0032432432432432435], [-0.21081081081081082, 0.10054054054054055], [-0.28216216216216217, 0.0016216216216216218]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.10378378378378379], [-0.1491891891891892, 0.0032432432432432435], [-0.21081081081081082, 0.10054054054054055]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.0032432432432432435], [-0.025945945945945948, 0.10378378378378379], [-0.1491891891891892, 0.0032432432432432435]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.0032432432432432435], [0.08756756756756758, 0.0016216216216216218], [-0.025945945945945948, 0.10378378378378379]],None, None, None, None, None),
            Vertex::new([[0.08756756756756758, 0.0016216216216216218], [-0.025945945945945948, 0.10378378378378379], [0.16864864864864867, 0.10054054054054055]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.28540540540540543], [-0.025945945945945948, 0.10378378378378379], [0.07135135135135136, 0.10216216216216217]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.28540540540540543], [0.07135135135135136, 0.10216216216216217], [0.16864864864864867, 0.10054054054054055]],None, None, None, None, None),
            Vertex::new([[0.16864864864864867, 0.10054054054054055], [-0.025945945945945948, 0.28540540540540543], [0.2772972972972973, 0.09243243243243245]],None, None, None, None, None),
            Vertex::new([[0.2756756756756757, 0.09243243243243245], [-0.025945945945945948, 0.28540540540540543], [0.38270270270270274, 0.08594594594594596]],None, None, None, None, None),
            Vertex::new([[0.38270270270270274, 0.08594594594594596], [-0.025945945945945948, 0.28540540540540543], [0.47513513513513517, 0.0745945945945946]],None, None, None, None, None),
            Vertex::new([[0.47513513513513517, 0.0745945945945946], [-0.025945945945945948, 0.28540540540540543], [0.4994594594594595, 0.10054054054054055]],None, None, None, None, None),
            Vertex::new([[0.4994594594594595, 0.10054054054054055], [-0.025945945945945948, 0.28540540540540543], [0.5254054054054055, 0.13135135135135137]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.28540540540540543], [0.5254054054054055, 0.13135135135135137], [0.44432432432432434, 0.1945945945945946]],None, None, None, None, None),
            Vertex::new([[0.44432432432432434, 0.1945945945945946], [-0.025945945945945948, 0.28540540540540543], [0.36324324324324325, 0.23513513513513515]],None, None, None, None, None),
            Vertex::new([[0.36324324324324325, 0.23513513513513515], [-0.025945945945945948, 0.28540540540540543], [0.252972972972973, 0.26432432432432434]],None, None, None, None, None),
            Vertex::new([[0.252972972972973, 0.26432432432432434], [-0.025945945945945948, 0.28540540540540543], [0.13945945945945948, 0.2772972972972973]],None, None, None, None, None),
            Vertex::new([[0.08756756756756758, 0.0016216216216216218], [0.16864864864864867, 0.10054054054054055], [0.23675675675675678, -0.0048648648648648655]],None, None, None, None, None),
            Vertex::new([[0.2756756756756757, 0.09243243243243245], [0.23675675675675678, -0.0048648648648648655], [0.16864864864864867, 0.10054054054054055]],None, None, None, None, None),
            Vertex::new([[0.23675675675675678, -0.0048648648648648655], [0.2756756756756757, 0.09243243243243245], [0.31783783783783787, -0.011351351351351352]],None, None, None, None, None),
            Vertex::new([[0.38270270270270274, 0.08594594594594596], [0.31783783783783787, -0.011351351351351352], [0.2756756756756757, 0.09243243243243245]],None, None, None, None, None),
            Vertex::new([[0.31783783783783787, -0.011351351351351352], [0.38270270270270274, 0.08594594594594596], [0.3908108108108108, -0.021081081081081084]],None, None, None, None, None),
            Vertex::new([[0.45567567567567574, -0.030810810810810812], [0.3908108108108108, -0.021081081081081084], [0.38270270270270274, 0.08594594594594596]],None, None, None, None, None),
            Vertex::new([[0.38270270270270274, 0.08594594594594596], [0.45567567567567574, -0.030810810810810812], [0.47513513513513517, 0.0745945945945946]],None, None, None, None, None),
            Vertex::new([[0.47513513513513517, 0.0745945945945946], [0.45567567567567574, -0.030810810810810812], [0.5497297297297298, -0.053513513513513515]],None, None, None, None, None),
            Vertex::new([[0.47513513513513517, 0.0745945945945946], [0.5497297297297298, -0.053513513513513515], [0.5383783783783784, 0.06486486486486487]],None, None, None, None, None),
            Vertex::new([[0.5383783783783784, 0.06486486486486487], [0.5497297297297298, -0.053513513513513515], [0.6194594594594595, -0.0745945945945946]],None, None, None, None, None),
            Vertex::new([[0.5383783783783784, 0.06486486486486487], [0.6194594594594595, -0.0745945945945946], [0.6048648648648649, 0.04864864864864865]],None, None, None, None, None),
            Vertex::new([[0.6194594594594595, -0.0745945945945946], [0.6048648648648649, 0.04864864864864865], [0.6762162162162163, -0.0972972972972973]],None, None, None, None, None),
            Vertex::new([[0.6486486486486487, 0.029189189189189193], [0.6762162162162163, -0.0972972972972973], [0.7183783783783785, -0.11351351351351352]],None, None, None, None, None),
            Vertex::new([[0.6486486486486487, 0.029189189189189193], [0.7183783783783785, -0.11351351351351352], [0.7897297297297298, -0.14270270270270272]],None, None, None, None, None),
            Vertex::new([[0.6048648648648649, 0.04864864864864865], [0.6762162162162163, -0.0972972972972973], [0.6486486486486487, 0.029189189189189193]],None, None, None, None, None),
            Vertex::new([[0.54, 0.06486486486486487], [0.47513513513513517, 0.0745945945945946], [0.4994594594594595, 0.10054054054054055]],None, None, None, None, None),
            Vertex::new([[0.4994594594594595, 0.10054054054054055], [0.54, 0.06486486486486487], [0.5254054054054055, 0.13135135135135137]],None, None, None, None, None),
            Vertex::new([[0.5805405405405406, 0.05513513513513514], [0.54, 0.06486486486486487], [0.5254054054054055, 0.13135135135135137]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.28540540540540543], [-0.13135135135135137, 0.10216216216216217], [-0.025945945945945948, 0.10378378378378379]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.28540540540540543], [-0.13135135135135137, 0.10216216216216217], [-0.21081081081081082, 0.10054054054054055]],None, None, None, None, None),
            Vertex::new([[-0.21081081081081082, 0.10054054054054055], [-0.025945945945945948, 0.28540540540540543], [-0.3745945945945946, 0.09243243243243245]],None, None, None, None, None),
            Vertex::new([[-0.3745945945945946, 0.09243243243243245], [-0.025945945945945948, 0.28540540540540543], [-0.45081081081081087, 0.08432432432432434]],None, None, None, None, None),
            Vertex::new([[-0.45081081081081087, 0.08432432432432434], [-0.025945945945945948, 0.28540540540540543], [-0.5172972972972973, 0.0745945945945946]],None, None, None, None, None),
            Vertex::new([[-0.5172972972972973, 0.0745945945945946], [-0.025945945945945948, 0.28540540540540543], [-0.5464864864864866, 0.10216216216216217]],None, None, None, None, None),
            Vertex::new([[-0.025945945945945948, 0.28540540540540543], [-0.5464864864864866, 0.10216216216216217], [-0.5740540540540541, 0.12972972972972974]],None, None, None, None, None),
            Vertex::new([[-0.5740540540540541, 0.12972972972972974], [-0.025945945945945948, 0.28540540540540543], [-0.47675675675675677, 0.19621621621621624]],None, None, None, None, None),
            Vertex::new([[-0.47675675675675677, 0.19621621621621624], [-0.025945945945945948, 0.28540540540540543], [-0.3908108108108108, 0.23513513513513515]],None, None, None, None, None),
            Vertex::new([[-0.3908108108108108, 0.23513513513513515], [-0.025945945945945948, 0.28540540540540543], [-0.2740540540540541, 0.26432432432432434]],None, None, None, None, None),
            Vertex::new([[-0.5464864864864866, 0.10216216216216217], [-0.5172972972972973, 0.0745945945945946], [-0.5740540540540541, 0.06486486486486487]],None, None, None, None, None),
            Vertex::new([[-0.5464864864864866, 0.10216216216216217], [-0.5740540540540541, 0.06486486486486487], [-0.5740540540540541, 0.12972972972972974]],None, None, None, None, None),
            Vertex::new([[-0.5740540540540541, 0.12972972972972974], [-0.5740540540540541, 0.06486486486486487], [-0.6421621621621623, 0.05027027027027028]],None, None, None, None, None),
            Vertex::new([[-0.3275675675675676, 0.3648648648648649], [-0.2513513513513514, 0.26594594594594595], [-0.19621621621621624, 0.2708108108108108]],None, None, None, None, None),
            Vertex::new([[-0.14270270270270272, 0.2772972972972973], [-0.19621621621621624, 0.2708108108108108], [-0.3145945945945946, 0.38108108108108113]],None, None, None, None, None),
            Vertex::new([[-0.3145945945945946, 0.38108108108108113], [-0.19621621621621624, 0.2708108108108108], [-0.3275675675675676, 0.3648648648648649]],None, None, None, None, None),
            Vertex::new([[-0.36324324324324325, 0.3924324324324325], [-0.3275675675675676, 0.3648648648648649], [-0.3145945945945946, 0.38108108108108113]],None, None, None, None, None),
            Vertex::new([[0.25783783783783787, 0.37945945945945947], [0.07945945945945947, 0.27891891891891896], [0.13945945945945948, 0.27891891891891896]],None, None, None, None, None),
            Vertex::new([[0.192972972972973, 0.2708108108108108], [0.13945945945945948, 0.27891891891891896], [0.2708108108108108, 0.3648648648648649]],None, None, None, None, None),
            Vertex::new([[0.2708108108108108, 0.3648648648648649], [0.13945945945945948, 0.27891891891891896], [0.25783783783783787, 0.37945945945945947]],None, None, None, None, None),
            Vertex::new([[0.3064864864864865, 0.3908108108108108], [0.2708108108108108, 0.3648648648648649], [0.25783783783783787, 0.37945945945945947]],None, None, None, None, None)
       ]);

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

                // vertex_buffer = CpuAccessibleBuffer::from_iter(
                //     device.clone(),
                //     BufferUsage::vertex_buffer(),
                //     false,
                //     figure1.get_vertex(surface.window().inner_size()).into_iter(),
                // )
                //     .unwrap();
            }
        }
        _ => (),
    });
}
