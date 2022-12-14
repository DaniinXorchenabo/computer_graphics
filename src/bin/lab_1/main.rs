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
use winit::event::{Event as WinitEvent, WindowEvent};
use winit::event_loop::{ControlFlow as WinitControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use vulkano::shader::SpecializationConstants;
use vulkano::shader::SpecializationMapEntry;

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
    move_matrix: [[f32; 3]; 3],
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
            move_matrix: [[1.0, 0.0, 0.0, ], [0.0, 1.0, 0.0, ], [0.0, 0.0, 1.0, ]],
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

#[derive(Default,  Clone)]
pub struct Figure {
    polygons: Vec<Vertex>,
    move_matrix: [[f32; 3]; 3],
    _changed: bool,
}

impl Figure {
    pub fn new(new_polygons: Vec<Vertex>) -> Self {

        let mut real_polygons = Vec::new();
        let default_matrix = [[1.0, 0.0, 0.0, ], [0.0, 1.0, 0.0, ], [0.0, 0.0, 1.0, ]];
        for vertex in new_polygons {
            // vertex.move_matrix = default_matrix.clone();
            real_polygons.push(vertex.clone());
            real_polygons.push(vertex.clone());
            real_polygons.push(vertex.clone());
        }

        Figure {
            polygons: real_polygons,
            move_matrix: default_matrix,
            _changed: false,
        }
    }

    fn get_vertex(&self) -> Vec<Vertex> {
        if self._changed {
            self.polygons.clone()
        } else {
            self.polygons.clone()
        }
    }
}



//3189 -- 1 1046 -- 2  610 -- 3 409 -- 4  286 -- 5 210 -- 6 140 -- 7
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/bin/lab_1/vertex.glsl",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/bin/lab_1/fragments.glsl",
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

    // points: [[f32; 2]; 3],
    // contour: Option<[f32; 3]>,
    // point_colors: Option<[[f32; 4]; 3]>,
    // point_color: Option<[f32; 4]>,
    // contour_color: Option<[f32; 4]>,
    // contour_colors: Option<[[f32; 4]; 3]>,
    let figure1 = Figure::new(
        vec![
            Vertex::new(
                [
                    [-0.5, -0.5],
                    [-0.6, -0.1],
                    [-0.1, 0.2]
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

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::vertex_buffer(),
        false,
        figure1.get_vertex().into_iter(),
    )
        .unwrap();



    // let vertex1 = Vertex::new(
    //     [
    //         [-0.5, -0.5],
    //         [-0.6, -0.1],
    //         [-0.1, 0.2]
    //     ],
    // );
    // let vertex4 = Vertex::new(
    //     [
    //         [0.5, 0.5],
    //         [0.5, 0.1],
    //         [0.1, 0.1],
    //     ],
    // );
    // let vertex7 = Vertex::new(
    //     [
    //         [-0.5, -0.5],
    //         [-0.5, -0.9],
    //         [-0.9, -0.9],
    //     ],
    // );
    // let vertex10 = Vertex::new(
    //     [
    //         [0.9, -0.5],
    //         [0.5, -0.9],
    //         [0.9, -0.9],
    //     ],
    // );
    // let vertex13 = Vertex::new(
    //     [
    //         [-0.5, 0.9],
    //         [-0.5, 0.5],
    //         [-0.9, 0.5]
    //     ],
    // );
    // let vertex2 = vertex1.clone();
    // let vertex3 = vertex1.clone();
    //
    // let vertex5 = vertex4.clone();
    // let vertex6 = vertex4.clone();
    //
    // let vertex8 = vertex7.clone();
    // let vertex9 = vertex7.clone();
    //
    // let vertex11 = vertex10.clone();
    // let vertex12 = vertex10.clone();
    //
    // let vertex14 = vertex13.clone();
    // let vertex15 = vertex13.clone();
    //
    // let vertex_buffer = CpuAccessibleBuffer::from_iter(
    //     device.clone(),
    //     BufferUsage::vertex_buffer(),
    //     false,
    //     vec![
    //         vertex1, vertex2, vertex3,
    //         vertex4, vertex5, vertex6,
    //         vertex7, vertex8, vertex9,
    //         vertex10, vertex11, vertex12,
    //         vertex13, vertex14, vertex15,
    //     ].into_iter(),
    // )
    //     .unwrap();

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
            event: WindowEvent::ReceivedCharacter(_),
            ..
        } => {
            // https://docs.rs/winit/latest/winit/event/enum.WindowEvent.html#variant.ReceivedCharacter
            println!("+");
        }
        _ => (),
    });
}