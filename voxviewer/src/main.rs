use dolly::prelude::*;
use egui_winit::egui;
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Uniforms {
    v_inv: glam::Mat4,
    p_inv: glam::Mat4,
    camera_pos: glam::Vec3,
    _padding0: u32,
    sun_direction: glam::Vec3,
    _padding1: u32,
    resolution: glam::UVec2,
    _padding2: [u32; 2],
}

struct Pipelines {
    blit_srgb: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    trace: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
}

impl Pipelines {
    async fn new(device: &wgpu::Device, swapchain_format: wgpu::TextureFormat) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        Self {
            sampler,
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
        }
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
    bind_group: wgpu::BindGroup,
    trace_bind_group: wgpu::BindGroup,
}

impl Resizables {
    fn new(width: u32, height: u32, device: &wgpu::Device, pipelines: &Pipelines) -> Self {
        let hdr = device.create_texture(&wgpu::TextureDescriptor {
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
        });

        Self {
            bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &hdr.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                    },
                ],
            }),
            trace_bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipelines.trace_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pipelines.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &hdr.create_view(&Default::default()),
                        ),
                    },
                ],
            }),
        }
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::default();

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

    let pipelines = Pipelines::new(&device, swapchain_format).await;
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
        .with(YawPitch::new().yaw_degrees(45.0).pitch_degrees(-30.0))
        .with(Smooth::new_rotation(0.5))
        .with(Arm::new(glam::Vec3::Z * 250.0))
        .build();

    let mut mouse_down = false;

    let mut sun_long = 1.0_f32;
    let mut sun_lat = 1.0_f32;

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

                    match event {
                    WindowEvent::MouseInput { state, button: MouseButton::Left, .. } if !egui_response.consumed => mouse_down = state == ElementState::Pressed,
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
                    }
                    WindowEvent::RedrawRequested => {
                        let transform = camera.update(1.0 / 60.0);

                        queue.write_buffer(
                            &pipelines.uniform_buffer,
                            0,
                            bytemuck::bytes_of(&Uniforms {
                                #[rustfmt::skip]
                                v_inv: glam::Mat4::look_to_rh(transform.position.into(), transform.forward(), transform.up()).inverse(),
                                p_inv: glam::Mat4::perspective_rh(
                                    45.0_f32.to_radians(),
                                    config.width as f32 / config.height as f32,
                                    0.0001,
                                    1000.0,
                                ).inverse(),
                                resolution: glam::UVec2::new(config.width, config.height),
                                camera_pos: transform.position.into(),
                                sun_direction: glam::Vec3::new(sun_long.sin() * sun_lat.cos(), sun_lat.sin(), sun_long.cos() * sun_lat.cos()),
                                _padding0: Default::default(),
                                _padding1: Default::default(),
                                _padding2: Default::default()
                            }),
                        );

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
                                egui::CollapsingHeader::new("Lighting").default_open(true).show(ui, |ui| {
                                    egui::CollapsingHeader::new("Sun").default_open(true).show(ui, |ui| {
                                        ui.label("Latitude");
                                        ui.add(egui::Slider::new(&mut sun_lat, 0.0..=std::f32::consts::FRAC_PI_2));
                                        ui.label("Longitude");
                                        ui.add(egui::Slider::new(&mut sun_long, -std::f32::consts::PI..=std::f32::consts::PI));
                                    });
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
                            compute_pass.set_bind_group(0, &resizables.trace_bind_group, &[]);
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
                            rpass.set_bind_group(0, &resizables.bind_group, &[]);
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
                    DeviceEvent::MouseMotion { delta: (x, y) } if mouse_down => {
                        camera.driver_mut::<YawPitch>().rotate_yaw_pitch((-x/2.0) as f32, (-y/2.0) as f32);
                    },
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
