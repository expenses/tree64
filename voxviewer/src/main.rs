use dolly::prelude::*;
use egui_winit::egui;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        //backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let surface = instance.create_surface(&window).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0].add_srgb_suffix();

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    config.view_formats = vec![swapchain_format];
    config.present_mode = wgpu::PresentMode::AutoVsync;
    surface.configure(&device, &config);

    log::info!("{:?}\n{:?}", &config, &swapchain_capabilities.formats);

    let pipelines = Pipelines::new(&device, &queue, swapchain_format).await;
    let mut resizables = Resizables::new(size.width, size.height, &device, &pipelines);

    let mut egui_renderer = egui_wgpu::Renderer::new(&device, swapchain_format, None, 1, false);

    let egui_context = egui::Context::default();

    let mut egui_state = egui_winit::State::new(
        egui_context,
        egui::viewport::ViewportId::ROOT,
        &window,
        None,
        None,
        None,
    );

    // Create a smoothed orbit camera
    let mut camera: CameraRig = CameraRig::builder()
        .with(Position::new(glam::Vec3::splat((1 << 5) as f32)))
        .with(YawPitch::new().yaw_degrees(45.0).pitch_degrees(-30.0))
        .with(Smooth::new_rotation(0.5))
        .with(Arm::new(glam::Vec3::Z * 250.0))
        .build();

    let mut left_mouse_down = false;
    let mut right_mouse_down = false;

    let mut settings = Settings::default();
    let mut frame_index = 0;
    let mut accumulated_frame_index = 0;

    let mut materials = [
        Material {
            base_colour: [1.0; 3],
            emission_factor: 0.0,
        },
        Material {
            base_colour: [1.0, 0.0, 0.0],
            emission_factor: 0.0,
        },
    ];

    queue.write_buffer(&pipelines.materials, 0, bytemuck::cast_slice(&materials));

    let window = &window;
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter);

            match event {
                Event::WindowEvent { event, .. } => {
                    let egui_response = egui_state.on_window_event(window, &event);
                    if egui_response.consumed {
                        return;
                    }

                    match event {
                    WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => left_mouse_down = state == ElementState::Pressed,
                    WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => right_mouse_down = state == ElementState::Pressed,
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        resizables = Resizables::new(
                            new_size.width.max(1),
                            new_size.height.max(1),
                            &device,
                            &pipelines,
                        );
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    },
                    WindowEvent::MouseWheel { delta, .. } => {
                        // TODO: this is very WIP.
                        let pixel_delta_divisor = if cfg!(target_arch = "wasm32") {
                            1000.0
                        } else {
                            100.0
                        };
                        camera.driver_mut::<Arm>().offset.z *= 1.0 + match delta {
                            MouseScrollDelta::LineDelta(_, y) => y / 10.0,
                            MouseScrollDelta::PixelDelta(pos) => -pos.y as f32 / pixel_delta_divisor,
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let transform = camera.update(1.0 / 60.0);

                        if !settings.accumulate_samples {
                            accumulated_frame_index = 0;
                        }

                        queue.write_buffer(
                            &pipelines.uniform_buffer,
                            0,
                            bytemuck::bytes_of(&Uniforms {
                                #[rustfmt::skip]
                                v_inv: glam::Mat4::look_to_rh(transform.position.into(), transform.forward(), transform.up()).inverse(),
                                p_inv: glam::Mat4::perspective_rh(
                                    settings.vertical_fov.to_radians(),
                                    config.width as f32 / config.height as f32,
                                    0.0001,
                                    1000.0,
                                ).inverse(),
                                resolution: glam::UVec2::new(config.width, config.height),
                                camera_pos: glam::Vec3::from(transform.position).into(),
                                sun_colour: glam::Vec3::from(settings.sun_colour).into(),
                                sun_direction: glam::Vec3::new(settings.sun_long.to_radians().sin() * settings.sun_lat.to_radians().cos(), settings.sun_lat.to_radians().sin(), settings.sun_long.to_radians().cos() * settings.sun_lat.to_radians().cos()).into(),
                                settings: (settings.enable_shadows as i32) | (settings.accumulate_samples as i32) << 1,
                                frame_index,
                                accumulated_frame_index, num_bounces: settings.num_bounces,
                                cos_sun_apparent_size: settings.sun_apparent_size.to_radians().cos(),
                                background_colour: glam::Vec3::from(settings.background_colour).into(),
                                padding: Default::default()
                            }),
                        );
                        queue.write_buffer(&pipelines.blit_uniform_buffer, 0, bytemuck::bytes_of(&accumulated_frame_index));

                        if settings.accumulate_samples {
                            accumulated_frame_index += 1;
                        }
                        frame_index += 1;

                        let frame = surface
                            .get_current_texture()
                            .expect("Failed to acquire next swap chain texture");
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor {
                                format: Some(swapchain_format),
                                ..Default::default()
                            });
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });
                        let (tessellated, screen_descriptor) = {
                            let raw_input = egui_state.take_egui_input(window);

                            egui_state
                                .egui_ctx()
                                .set_pixels_per_point(window.scale_factor() as _);

                        let egui_output = egui_state.egui_ctx().run(raw_input, |ctx| {
                            egui::Window::new("Controls").show(ctx, |ui| {
                                if ui.button("Make Pretty").clicked() {
                                    settings = Settings::pretty();
                                }

                                ui.checkbox(&mut settings.accumulate_samples, "Accumulate Samples");
                                egui::CollapsingHeader::new("Lighting").default_open(true).show(ui, |ui| {
                                    ui.label("Number of bounces");
                                    ui.add(egui::Slider::new(&mut settings.num_bounces, 0..=5));
                                    ui.checkbox(&mut settings.enable_shadows, "Enable shadows");
                                    egui::CollapsingHeader::new("Sun").default_open(true).show(ui, |ui| {
                                        ui.label("Latitude");
                                        ui.add(egui::Slider::new(&mut settings.sun_lat, 0.0..=90.0));
                                        ui.label("Longitude");
                                        ui.add(egui::Slider::new(&mut settings.sun_long, -180.0..=180.0));
                                        ui.label("Apparent size");
                                        ui.add(egui::Slider::new(&mut settings.sun_apparent_size, 0.0..=90.0));
                                        ui.label("Sun colour");
                                        egui::widgets::color_picker::color_edit_button_rgb(ui, &mut settings.sun_colour);
                                    });
                                    ui.label("Background colour");
                                    egui::widgets::color_picker::color_edit_button_rgb(ui, &mut settings.background_colour);
                                });
                                ui.label("Field of view (vertical)");
                                ui.add(egui::Slider::new(&mut settings.vertical_fov, 1.0..=120.0));
                                if ui.button("Reset Settings").clicked() {
                                    settings = Settings::default();
                                }
                                egui::CollapsingHeader::new("Materials").default_open(true).show(ui, |ui| {
                                    for (i, material) in materials.iter_mut().enumerate() {
                                        let mut changed = false;
                                        ui.label("Base Colour");
                                        changed |= egui::widgets::color_picker::color_edit_button_rgb(ui, &mut material.base_colour).changed();
                                        ui.label("Emission Factor");
                                        changed |= ui.add(egui::Slider::new(&mut material.emission_factor, 0.0..=10_000.0)).changed();
                                        if changed {
                                            queue.write_buffer(&pipelines.materials, (i * std::mem::size_of::<Material>()) as _, bytemuck::bytes_of(&*material))};
                                    }

                                });
                            });
                        });


                            egui_state.handle_platform_output(window, egui_output.platform_output);


                        let tris = egui_state.egui_ctx()
                                    .tessellate(egui_output.shapes, window.scale_factor() as _);

                        for (id, image_delta) in &egui_output.textures_delta.set {
                            egui_renderer
                                        .update_texture(&device, &queue, *id, image_delta);
                                }

                                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                                    pixels_per_point: window.scale_factor() as _,
                                    size_in_pixels: [config.width, config.height]
                                };

                            let command_buffers = egui_renderer
                                            .update_buffers(&device, &queue, &mut encoder, &tris, &screen_descriptor);

                            debug_assert!(command_buffers.is_empty());

                                (tris, screen_descriptor)
                        };

                        {
                            let mut compute_pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

                            compute_pass.set_pipeline(&pipelines.trace);
                            compute_pass.set_bind_group(0, &resizables.trace_bind_groups[accumulated_frame_index as usize % 2], &[]);
                            compute_pass.dispatch_workgroups(
                                ((config.width - 1) / 8) + 1,
                                ((config.height - 1) / 8) + 1,
                                1,
                            );
                        }
                        {
                            let mut rpass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                }).forget_lifetime();
                            rpass.set_pipeline(&pipelines.blit_srgb);
                            rpass.set_bind_group(0, &resizables.blit_bind_groups[accumulated_frame_index as usize % 2], &[]);
                            rpass.draw(0..3, 0..1);

                                        egui_renderer.render(&mut rpass, &tessellated, &screen_descriptor);
                        }

                        queue.submit(Some(encoder.finish()));
                        frame.present();
                    }
                    WindowEvent::CloseRequested => target.exit(),
                    _ => {}
                }},
                Event::DeviceEvent { event, .. } => match event {
                    DeviceEvent::MouseMotion { delta: (x, y) } if left_mouse_down => {
                        camera.driver_mut::<YawPitch>().rotate_yaw_pitch((-x/2.0) as f32, (-y/2.0) as f32);
                    },
                    DeviceEvent::MouseMotion { delta: (x, y) } if right_mouse_down => {
                        let transform = camera.final_transform;
                        // Default strength of 500.0 seems to work well at default v fov (45)
                        let strength_divisor = 500.0 * 45.0;
                        let arm_distance = camera.driver::<Arm>().offset.z / strength_divisor * settings.vertical_fov;
                        camera.driver_mut::<Position>().translate((
                            transform.up::<glam::Vec3>() * y as f32 +
                            transform.right::<glam::Vec3>() * -x as f32
                        ) * arm_distance);
                    }
                    _ => {}
                },
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            };
        })
        .unwrap();
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowAttributes::default();
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowAttributesExtWebSys;
        let canvas = wgpu::web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<wgpu::web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    let window = event_loop.create_window(builder).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}

struct Settings {
    sun_long: f32,
    sun_lat: f32,
    enable_shadows: bool,
    sun_apparent_size: f32,
    accumulate_samples: bool,
    num_bounces: u32,
    background_colour: [f32; 3],
    sun_colour: [f32; 3],
    vertical_fov: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sun_long: -45.0_f32,
            sun_lat: 45.0_f32,
            enable_shadows: false,
            sun_apparent_size: 5.0_f32,
            accumulate_samples: false,
            num_bounces: 0,
            background_colour: [0.01; 3],
            sun_colour: [1.0; 3],
            vertical_fov: 45.0_f32,
        }
    }
}

impl Settings {
    fn pretty() -> Self {
        Self {
            enable_shadows: true,
            num_bounces: 1,
            accumulate_samples: true,
            vertical_fov: 22.5,
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Vec3A {
    inner: glam::Vec3,
    padding: u32,
}

impl From<glam::Vec3> for Vec3A {
    fn from(vec: glam::Vec3) -> Self {
        Self {
            inner: vec.into(),
            padding: 0,
        }
    }
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Uniforms {
    v_inv: glam::Mat4,
    p_inv: glam::Mat4,
    camera_pos: Vec3A,
    sun_direction: Vec3A,
    sun_colour: Vec3A,
    background_colour: Vec3A,
    resolution: glam::UVec2,
    settings: i32,
    frame_index: u32,
    cos_sun_apparent_size: f32,
    accumulated_frame_index: u32,
    num_bounces: u32,
    padding: [u32; 1],
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Material {
    base_colour: [f32; 3],
    emission_factor: f32,
}

struct Pipelines {
    blit_srgb: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    filtering_sampler: wgpu::Sampler,
    non_filtering_sampler: wgpu::Sampler,
    trace: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    dag_data: wgpu::Buffer,
    blit_uniform_buffer: wgpu::Buffer,
    materials: wgpu::Buffer,
    tonemapping_lut: wgpu::Texture,
}

impl Pipelines {
    async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        swapchain_format: wgpu::TextureFormat,
    ) -> Self {
        let uniform_entry = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = |binding, visibility, filtering| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Sampler(if filtering {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }),
            count: None,
        };
        let compute_buffer = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                sampler_entry(2, wgpu::ShaderStages::FRAGMENT, true),
                texture_entry(3, wgpu::ShaderStages::FRAGMENT),
                sampler_entry(4, wgpu::ShaderStages::FRAGMENT, false),
            ],
        });

        let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                compute_buffer(1),
                compute_buffer(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                texture_entry(4, wgpu::ShaderStages::COMPUTE),
                sampler_entry(5, wgpu::ShaderStages::COMPUTE, false),
            ],
        });

        let dag_data = [
            [1, 1, 1, 0, 2, 0, 0, 0],
            [1, 1, 1, 256, 1, 256, 256, 0],
            [1, 1, 1, 257, 1, 257, 257, 0],
            [1, 1, 1, 258, 1, 258, 258, 0],
            [1, 1, 1, 259, 1, 259, 259, 0],
            [1, 1, 1, 260, 1, 260, 260, 0],
        ];

        Self {
            non_filtering_sampler: device.create_sampler(&wgpu::SamplerDescriptor::default()),
            filtering_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            trace: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&trace_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(
                        load_resource_str("shaders/raytrace.wgsl").await.into(),
                    ),
                }),
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Default::default(),
            }),
            blit_srgb: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                label: Default::default(),
                depth_stencil: Default::default(),
                multiview: Default::default(),
                multisample: Default::default(),
                primitive: Default::default(),
                cache: Default::default(),
                vertex: wgpu::VertexState {
                    module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(
                            load_resource_str("shaders/blit_srgb_vs.wgsl").await.into(),
                        ),
                    }),
                    entry_point: Some("VSMain"),
                    compilation_options: Default::default(),
                    buffers: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(
                            load_resource_str("shaders/blit_srgb_ps.wgsl").await.into(),
                        ),
                    }),
                    entry_point: Some("PSMain"),
                    compilation_options: Default::default(),
                    targets: &[Some(swapchain_format.into())],
                }),
            }),
            bgl,
            trace_bgl,
            uniform_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<Uniforms>() as _,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blit_uniform_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<[u32; 4]>() as _,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            dag_data: device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&dag_data),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            materials: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<[Material; 256]>() as _,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            tonemapping_lut: device.create_texture_with_data(
                &queue,
                &wgpu::TextureDescriptor {
                    label: None,
                    mip_level_count: 1,
                    size: wgpu::Extent3d {
                        width: 48,
                        height: 48,
                        depth_or_array_layers: 48,
                    },
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgb9e5Ufloat,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &ddsfile::Dds::read(std::io::Cursor::new(
                    load_resource_bytes("tony_mc_mapface.dds").await,
                ))
                .unwrap()
                .data,
            ),
        }
    }
}

async fn load_resource_bytes(filename: &str) -> Vec<u8> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::fs::read(std::path::Path::new("assets").join(filename)).expect(filename)
    }
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use wgpu::web_sys;
        let window = web_sys::window().unwrap();
        let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(filename))
            .await
            .unwrap();
        let resp: web_sys::Response = resp_value.dyn_into().unwrap();
        let resp_array_buffer = wasm_bindgen_futures::JsFuture::from(resp.array_buffer().unwrap())
            .await
            .unwrap();
        let array = wasm_bindgen_futures::js_sys::Uint8Array::new(&resp_array_buffer);
        array.to_vec()
    }
}

async fn load_resource_str(filename: &str) -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::fs::read_to_string(std::path::Path::new("assets").join(filename)).expect(filename)
    }
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use wgpu::web_sys;
        let window = web_sys::window().unwrap();
        let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(filename))
            .await
            .unwrap();
        let resp: web_sys::Response = resp_value.dyn_into().unwrap();
        let resp_text = wasm_bindgen_futures::JsFuture::from(resp.text().unwrap())
            .await
            .unwrap();
        let text: wasm_bindgen_futures::js_sys::JsString = resp_text.dyn_into().unwrap();
        text.into()
    }
}

struct Resizables {
    blit_bind_groups: [wgpu::BindGroup; 2],
    trace_bind_groups: [wgpu::BindGroup; 2],
}

impl Resizables {
    fn new(width: u32, height: u32, device: &wgpu::Device, pipelines: &Pipelines) -> Self {
        let create_hdr = || {
            device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                label: None,
                mip_level_count: 1,
                sample_count: 1,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let hdr_a = create_hdr();
        let hdr_b = create_hdr();

        let create_trace_bind_group = |a: &wgpu::Texture, b: &wgpu::Texture| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.trace_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pipelines.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pipelines.dag_data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pipelines.materials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &a.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &b.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&pipelines.non_filtering_sampler),
                    },
                ],
            })
        };

        let create_blit_bind_group = |hdr: &wgpu::Texture| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pipelines.blit_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &pipelines.tonemapping_lut.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&pipelines.filtering_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &hdr.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&pipelines.non_filtering_sampler),
                    },
                ],
            })
        };

        Self {
            blit_bind_groups: [
                create_blit_bind_group(&hdr_a),
                create_blit_bind_group(&hdr_b),
            ],
            trace_bind_groups: [
                create_trace_bind_group(&hdr_a, &hdr_b),
                create_trace_bind_group(&hdr_b, &hdr_a),
            ],
        }
    }
}
