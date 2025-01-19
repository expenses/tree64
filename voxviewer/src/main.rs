use dolly::prelude::*;
use egui_winit::egui;
use glam::swizzles::Vec3Swizzles;
use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::PhysicalKey,
    window::{Window, WindowId},
};

const USE_SPIRV_SHADER: bool = false;

fn copy_aligned(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    data: &[u8],
    range: std::ops::Range<usize>,
) {
    let aligned_range = std::ops::Range {
        start: range
            .start
            .saturating_sub(wgpu::COPY_BUFFER_ALIGNMENT as _)
            .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as _),

        end: range.end.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as _),
    };
    if aligned_range.end <= data.len() {
        queue.write_buffer(
            buffer,
            aligned_range.start as _,
            &data[aligned_range.clone()],
        );
    } else {
        let mut aligned_data = vec![0; aligned_range.len()];
        aligned_data[..data.len() - aligned_range.start]
            .copy_from_slice(&data[aligned_range.start..data.len()]);
        queue.write_buffer(buffer, aligned_range.start as _, &aligned_data);
    }
}

struct App<'a> {
    egui_state: egui_winit::State,
    window: &'a Window,
    left_mouse_down: bool,
    right_mouse_down: bool,
    config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    accumulated_frame_index: u32,
    frame_index: u32,
    settings: Settings,
    camera: CameraRig,
    resizables: Resizables,
    pipelines: Pipelines,
    materials: Vec<Material>,
    egui_renderer: egui_wgpu::Renderer,
    tree64: svo_dag::Tree64,
}

impl<'a> App<'a> {
    fn draw_egui(&mut self, raw_input: egui::RawInput) -> egui::FullOutput {
        let Self {
            egui_state,
            settings,
            materials,
            queue,
            pipelines,
            tree64,
            ..
        } = self;

        let mut reset_accumulation = false;

        let output = egui_state.egui_ctx().run(raw_input, |ctx| {
            egui::Window::new("Controls").show(ctx, |ui| {
                egui::CollapsingHeader::new("Edits")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.add_enabled_ui(tree64.edits.can_undo(), |ui| {
                            if ui.button("Reset").clicked() {
                                while tree64.edits.can_undo() {
                                    tree64.edits.undo();
                                }
                                reset_accumulation = true;
                            }
                        });
                        ui.add_enabled_ui(tree64.edits.can_undo(), |ui| {
                            if ui.button("Undo").clicked() {
                                tree64.edits.undo();
                                reset_accumulation = true;
                            }
                        });
                        ui.add_enabled_ui(tree64.edits.can_redo(), |ui| {
                            if ui.button("Redo").clicked() {
                                tree64.edits.redo();
                                reset_accumulation = true;
                            }
                        });
                        if ui.button("Expand").clicked() {
                            reset_accumulation = true;
                            let range_to_upload = tree64.expand();
                            queue.write_buffer(
                                &pipelines.tree_nodes,
                                range_to_upload.start as u64
                                    * std::mem::size_of::<svo_dag::Tree64Node>() as u64,
                                bytemuck::cast_slice(&tree64.nodes.inner[range_to_upload]),
                            );
                        }

                        ui.label("Edit distance");
                        ui.add(egui::Slider::new(&mut settings.edit_distance, 0.0..=1000.0));
                        ui.label("Edit Size");
                        ui.add(egui::Slider::new(&mut settings.edit_size, 0.0..=1000.0));

                        let mut edit = |value| {
                            let position: glam::Vec3 =
                                glam::Vec3::from(self.camera.final_transform.position)
                                    + self.camera.final_transform.forward::<glam::Vec3>()
                                        * settings.edit_distance;

                            let ranges = tree64.modify_nodes_in_box(
                                (position - settings.edit_size / 2.0)
                                    .max(glam::Vec3::ZERO)
                                    .xzy()
                                    .as_uvec3(),
                                (position + settings.edit_size / 2.0).xzy().as_uvec3(),
                                value,
                            );
                            self.accumulated_frame_index = 0;
                            queue.write_buffer(
                                &pipelines.tree_nodes,
                                ranges.nodes.start as u64
                                    * std::mem::size_of::<svo_dag::Tree64Node>() as u64,
                                bytemuck::cast_slice(&tree64.nodes.inner[ranges.nodes]),
                            );
                            copy_aligned(
                                &queue,
                                &pipelines.leaf_data,
                                bytemuck::cast_slice(&tree64.data.inner),
                                ranges.data,
                            );
                        };

                        if ui.button("Delete").clicked() {
                            edit(0);
                        }
                        if ui.button("Create").clicked() {
                            edit(1);
                        }
                    });
                egui::CollapsingHeader::new("Rendering")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.label("Max bounces");
                        reset_accumulation |= ui
                            .add(egui::Slider::new(&mut settings.max_bounces, 0..=10))
                            .changed();
                        reset_accumulation |= ui
                            .checkbox(&mut settings.accumulate_samples, "Accumulate Samples")
                            .changed();
                    });
                egui::CollapsingHeader::new("Lighting")
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::CollapsingHeader::new("Sun")
                            .default_open(true)
                            .show(ui, |ui| {
                                reset_accumulation |= ui
                                    .checkbox(&mut settings.enable_shadows, "Enable shadows")
                                    .changed();
                                ui.label("Latitude");
                                reset_accumulation |= ui
                                    .add(egui::Slider::new(&mut settings.sun_lat, 0.0..=90.0))
                                    .changed();
                                ui.label("Longitude");
                                reset_accumulation |= ui
                                    .add(egui::Slider::new(&mut settings.sun_long, -180.0..=180.0))
                                    .changed();
                                ui.label("Apparent size");
                                reset_accumulation |= ui
                                    .add(egui::Slider::new(
                                        &mut settings.sun_apparent_size,
                                        0.0..=90.0,
                                    ))
                                    .changed();
                                ui.label("Sun Strength");
                                reset_accumulation |= ui
                                    .add(egui::Slider::new(
                                        &mut settings.sun_strength,
                                        0.0..=10_000.0,
                                    ))
                                    .changed();
                                ui.label("Colour");
                                reset_accumulation |=
                                    egui::widgets::color_picker::color_edit_button_rgb(
                                        ui,
                                        &mut settings.sun_colour,
                                    )
                                    .changed();
                            });
                        ui.label("Background colour");
                        reset_accumulation |= egui::widgets::color_picker::color_edit_button_rgb(
                            ui,
                            &mut settings.background_colour,
                        )
                        .changed();
                        ui.label("Background Strength");
                        reset_accumulation |= ui
                            .add(egui::Slider::new(
                                &mut settings.background_strength,
                                0.0..=10_000.0,
                            ))
                            .changed();
                    });
                egui::CollapsingHeader::new("Camera")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.label("Field of view (vertical)");
                        reset_accumulation |= ui
                            .add(egui::Slider::new(&mut settings.vertical_fov, 1.0..=120.0))
                            .changed();
                    });
                egui::CollapsingHeader::new("Settings")
                    .default_open(true)
                    .show(ui, |ui| {
                        if ui.button("Reset").clicked() {
                            *settings = Settings::default();
                            reset_accumulation = true;
                        }
                    });
                #[cfg(not(target_arch = "wasm32"))]
                egui::CollapsingHeader::new("Debugging")
                    .default_open(true)
                    .show(ui, |ui| {
                        reset_accumulation |= ui
                            .checkbox(&mut settings.show_heatmap, "Show Heatmap")
                            .changed();
                    });
                egui::CollapsingHeader::new("Materials").show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (i, material) in materials.iter_mut().enumerate() {
                            let mut changed = false;
                            ui.label("Base Colour");
                            changed |= egui::widgets::color_picker::color_edit_button_rgb(
                                ui,
                                &mut material.base_colour,
                            )
                            .changed();
                            ui.label("Emission Factor");
                            changed |= ui
                                .add(egui::Slider::new(
                                    &mut material.emission_factor,
                                    0.0..=10_000.0,
                                ))
                                .changed();
                            ui.label("Linear Roughness");
                            changed |= ui
                                .add(egui::Slider::new(
                                    &mut material.linear_roughness,
                                    0.000..=1.0,
                                ))
                                .changed();
                            ui.label("Metallic Factor");
                            changed |= ui
                                .add(egui::Slider::new(&mut material.metallic, 0.0..=1.0))
                                .changed();
                            if changed {
                                queue.write_buffer(
                                    &pipelines.materials,
                                    (i * std::mem::size_of::<Material>()) as _,
                                    bytemuck::bytes_of(&*material),
                                );
                                reset_accumulation = true;
                            };
                        }
                    });
                });
            });
        });

        if reset_accumulation {
            self.accumulated_frame_index = 0;
        }

        output
    }

    fn get_egui_render_state(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> (Vec<egui::ClippedPrimitive>, egui_wgpu::ScreenDescriptor) {
        let raw_input = self.egui_state.take_egui_input(self.window);
        let egui_output = self.draw_egui(raw_input);

        let Self {
            egui_state,
            window,
            egui_renderer,
            device,
            queue,
            ..
        } = self;

        egui_state
            .egui_ctx()
            .set_pixels_per_point(window.scale_factor() as _);

        egui_state.handle_platform_output(window, egui_output.platform_output);

        let tris = egui_state
            .egui_ctx()
            .tessellate(egui_output.shapes, window.scale_factor() as _);

        for (id, image_delta) in &egui_output.textures_delta.set {
            egui_renderer.update_texture(device, queue, *id, image_delta);
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            pixels_per_point: window.scale_factor() as _,
            size_in_pixels: [self.config.width, self.config.height],
        };

        let command_buffers =
            egui_renderer.update_buffers(device, queue, encoder, &tris, &screen_descriptor);

        debug_assert!(command_buffers.is_empty());

        (tris, screen_descriptor)
    }

    fn write_uniforms(
        &self,
        transform: dolly::transform::Transform<dolly::handedness::RightHanded>,
    ) {
        let settings = &self.settings;
        let (root_node_index, num_levels) = self.tree64.root_node_index_and_num_levels();

        self.queue.write_buffer(
            &self.pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&Uniforms {
                p_inv: (glam::Mat4::perspective_infinite_reverse_rh(
                    settings.vertical_fov.to_radians(),
                    self.config.width as f32 / self.config.height as f32,
                    0.0001,
                ) * glam::Mat4::look_to_rh(
                    glam::Vec3::ZERO,
                    transform.forward(),
                    transform.up(),
                ))
                .inverse(),
                resolution: glam::UVec2::new(self.config.width, self.config.height),
                camera_pos: glam::Vec3::from(transform.position).into(),
                sun_emission: (glam::Vec3::from(settings.sun_colour) * settings.sun_strength)
                    .into(),
                sun_direction: glam::Vec3::new(
                    settings.sun_long.to_radians().sin() * settings.sun_lat.to_radians().cos(),
                    settings.sun_lat.to_radians().sin(),
                    settings.sun_long.to_radians().cos() * settings.sun_lat.to_radians().cos(),
                )
                .into(),
                settings: (settings.enable_shadows as i32)
                    | (settings.accumulate_samples as i32) << 1
                    | (settings.show_heatmap as i32) << 2,
                frame_index: self.frame_index,
                accumulated_frame_index: self.accumulated_frame_index,
                max_bounces: settings.max_bounces,
                cos_sun_apparent_size: settings.sun_apparent_size.to_radians().cos(),
                background_colour: (glam::Vec3::from(settings.background_colour)
                    * settings.background_strength)
                    .into(),
                tree_scale: num_levels as u32 * 2,
                root_node_index,
                _padding: Default::default(),
            }),
        );
        self.queue.write_buffer(
            &self.pipelines.blit_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.accumulated_frame_index),
        );
    }
}

impl<'a> winit::application::ApplicationHandler for App<'a> {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {}

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.window.request_redraw();
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta: (x, y) } if self.left_mouse_down => {
                self.camera
                    .driver_mut::<YawPitch>()
                    .rotate_yaw_pitch((-x / 2.0) as f32, (-y / 2.0) as f32);

                self.accumulated_frame_index = 0;
            }
            DeviceEvent::MouseMotion { delta: (x, y) } if self.right_mouse_down => {
                let transform = self.camera.final_transform;
                // Default strength of 500.0 seems to work well at default v fov (45)
                let strength_divisor = 500.0 * 45.0;
                let arm_distance = self.camera.driver::<Arm>().offset.z / strength_divisor
                    * self.settings.vertical_fov;
                self.camera.driver_mut::<Position>().translate(
                    (transform.up::<glam::Vec3>() * y as f32
                        + transform.right::<glam::Vec3>() * -x as f32)
                        * arm_distance,
                );

                self.accumulated_frame_index = 0;
            }
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let egui_response = self.egui_state.on_window_event(self.window, &event);
        if egui_response.consumed {
            return;
        }

        let view_format = self.config.view_formats[0];

        match event {
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.left_mouse_down = state == ElementState::Pressed,
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => self.right_mouse_down = state == ElementState::Pressed,
            WindowEvent::Resized(new_size) => {
                self.accumulated_frame_index = 0;
                // Reconfigure the surface with the new size
                self.config.width = new_size.width.max(1);
                self.config.height = new_size.height.max(1);
                self.surface.configure(&self.device, &self.config);
                self.resizables = Resizables::new(
                    new_size.width.max(1),
                    new_size.height.max(1),
                    &self.device,
                    &self.pipelines,
                );
                // On macos the window needs to be redrawn manually after resizing
                self.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // TODO: this is very WIP.
                let pixel_delta_divisor = if cfg!(target_arch = "wasm32") {
                    1000.0
                } else {
                    100.0
                };
                self.camera.driver_mut::<Arm>().offset.z *= 1.0
                    + match delta {
                        MouseScrollDelta::LineDelta(_, y) => y / 10.0,
                        MouseScrollDelta::PixelDelta(pos) => -pos.y as f32 / pixel_delta_divisor,
                    }
            }
            WindowEvent::RedrawRequested => {
                let previous_transform = self.camera.final_transform;

                let transform = self.camera.update(1.0 / 60.0);

                if glam::Vec3::from(previous_transform.position)
                    .distance_squared(transform.position.into())
                    > (0.05 * 0.05)
                {
                    self.accumulated_frame_index = 0;
                }

                let settings = &self.settings;

                if !settings.accumulate_samples || settings.show_heatmap {
                    self.accumulated_frame_index = 0;
                }

                self.write_uniforms(transform);

                if settings.accumulate_samples {
                    self.accumulated_frame_index += 1;
                }
                self.frame_index += 1;

                let frame = self
                    .surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(view_format),
                    ..Default::default()
                });
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let (tessellated, screen_descriptor) = self.get_egui_render_state(&mut encoder);
                let egui_cmd_buf = encoder.finish();

                let (cmd_buf_a, cmd_buf_b) =
                    rayon::join(
                        || {
                            let mut encoder = self.device.create_command_encoder(
                                &wgpu::CommandEncoderDescriptor { label: None },
                            );

                            let mut compute_pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

                            compute_pass.set_pipeline(&self.pipelines.trace);
                            compute_pass.set_bind_group(
                                0,
                                &self.resizables.trace_bind_groups
                                    [self.accumulated_frame_index as usize % 2],
                                &[],
                            );

                            let workgroup_size = 8;

                            compute_pass.dispatch_workgroups(
                                (self.config.width + workgroup_size - 1) / workgroup_size,
                                (self.config.height + workgroup_size - 1) / workgroup_size,
                                1,
                            );

                            drop(compute_pass);

                            encoder.finish()
                        },
                        || {
                            let mut encoder = self.device.create_command_encoder(
                                &wgpu::CommandEncoderDescriptor { label: None },
                            );

                            let mut rpass = encoder
                                .begin_render_pass(&wgpu::RenderPassDescriptor {
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
                                })
                                .forget_lifetime();
                            rpass.set_pipeline(&self.pipelines.blit_srgb);
                            rpass.set_bind_group(
                                0,
                                &self.resizables.blit_bind_groups
                                    [(self.accumulated_frame_index) as usize % 2],
                                &[],
                            );
                            rpass.draw(0..3, 0..1);
                            self.egui_renderer
                                .render(&mut rpass, &tessellated, &screen_descriptor);

                            drop(rpass);

                            encoder.finish()
                        },
                    );

                self.queue.submit([egui_cmd_buf, cmd_buf_a, cmd_buf_b]);
                frame.present();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}

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
                required_features: if USE_SPIRV_SHADER {
                    wgpu::Features::SPIRV_SHADER_PASSTHROUGH
                } else {
                    Default::default()
                },
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

    /*let mut materials = vec![
        Material {
            base_colour: [1.0; 3],
            linear_roughness: 1.0,
            metallic: 0.0,
            ..Default::default()
        };
        256
    ];*/

    let palette = [
        (65, 59, 47, 255),
        (113, 99, 77, 255),
        (146, 129, 95, 255),
        (156, 142, 113, 255),
        (175, 159, 130, 255),
        (174, 154, 116, 255),
        (192, 174, 135, 255),
        (203, 187, 154, 255),
        (104, 86, 60, 255),
        (114, 99, 77, 255),
        (125, 107, 81, 255),
        (134, 119, 95, 255),
        (153, 135, 107, 255),
        (172, 156, 131, 255),
        (189, 174, 152, 255),
        (206, 193, 175, 255),
        (38, 38, 38, 255),
        (85, 76, 76, 255),
        (107, 100, 97, 255),
        (115, 114, 119, 255),
        (148, 138, 138, 255),
        (147, 118, 107, 255),
        (157, 163, 173, 255),
        (192, 191, 198, 255),
        (77, 68, 59, 255),
        (111, 97, 81, 255),
        (134, 120, 101, 255),
        (155, 139, 116, 255),
        (168, 150, 125, 255),
        (178, 162, 136, 255),
        (190, 172, 140, 255),
        (196, 181, 158, 255),
        (85, 80, 72, 255),
        (117, 110, 95, 255),
        (136, 126, 109, 255),
        (148, 141, 123, 255),
        (162, 152, 132, 255),
        (169, 161, 144, 255),
        (176, 170, 151, 255),
        (188, 180, 158, 255),
        (78, 73, 66, 255),
        (95, 90, 82, 255),
        (117, 107, 92, 255),
        (140, 131, 117, 255),
        (172, 157, 133, 255),
        (184, 173, 156, 255),
        (201, 188, 169, 255),
        (210, 200, 186, 255),
        (29, 31, 32, 255),
        (43, 44, 45, 255),
        (53, 54, 56, 255),
        (83, 78, 69, 255),
        (98, 94, 86, 255),
        (119, 110, 98, 255),
        (141, 133, 123, 255),
        (162, 157, 150, 255),
        (23, 25, 25, 255),
        (35, 39, 39, 255),
        (44, 48, 47, 255),
        (51, 57, 56, 255),
        (58, 63, 63, 255),
        (66, 73, 72, 255),
        (104, 106, 105, 255),
        (173, 173, 173, 255),
        (51, 57, 56, 255),
        (62, 71, 70, 255),
        (69, 76, 75, 255),
        (73, 82, 82, 255),
        (79, 86, 85, 255),
        (82, 92, 92, 255),
        (89, 96, 95, 255),
        (102, 110, 110, 255),
        (9, 9, 9, 255),
        (9, 9, 9, 255),
        (42, 38, 34, 255),
        (120, 110, 97, 255),
        (149, 135, 117, 255),
        (165, 153, 135, 255),
        (182, 167, 142, 255),
        (192, 179, 157, 255),
        (85, 72, 58, 255),
        (118, 105, 85, 255),
        (136, 120, 96, 255),
        (151, 130, 100, 255),
        (157, 141, 111, 255),
        (164, 145, 111, 255),
        (175, 159, 133, 255),
        (173, 152, 113, 255),
        (2, 86, 10, 255),
        (18, 100, 39, 255),
        (104, 117, 57, 255),
        (6, 102, 17, 255),
        (113, 91, 25, 255),
        (36, 118, 58, 255),
        (22, 115, 43, 255),
        (167, 131, 50, 255),
        (98, 99, 98, 255),
        (125, 97, 30, 255),
        (8, 65, 116, 255),
        (26, 64, 130, 255),
        (45, 81, 146, 255),
        (12, 75, 135, 255),
        (30, 77, 147, 255),
        (167, 131, 50, 255),
        (44, 41, 29, 255),
        (60, 71, 40, 255),
        (78, 81, 63, 255),
        (86, 81, 42, 255),
        (106, 94, 86, 255),
        (108, 93, 66, 255),
        (124, 115, 89, 255),
        (136, 128, 120, 255),
        (53, 44, 48, 255),
        (83, 71, 75, 255),
        (108, 92, 95, 255),
        (138, 123, 121, 255),
        (164, 157, 157, 255),
        (179, 165, 161, 255),
        (183, 175, 170, 255),
        (206, 194, 188, 255),
        (36, 89, 10, 255),
        (71, 129, 44, 255),
        (94, 144, 65, 255),
        (102, 149, 46, 255),
        (100, 135, 14, 255),
        (137, 172, 78, 255),
        (167, 66, 7, 255),
        (234, 165, 0, 255),
        (2, 2, 1, 255),
        (30, 26, 21, 255),
        (65, 59, 47, 255),
        (108, 97, 77, 255),
        (138, 124, 100, 255),
        (149, 136, 114, 255),
        (164, 149, 120, 255),
        (172, 159, 136, 255),
    ];

    fn srgb_to_linear(value: u8) -> f32 {
        let value = value as f32 / 255.0;

        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    let materials = palette
        .map(|(r, g, b, _)| Material {
            // Boost values for brighter bounces.
            base_colour: [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)],
            linear_roughness: 1.0,
            metallic: 0.0,
            emission_factor: 0.0,
            ..Default::default()
        })
        .to_vec();

    /*
    materials[1] = Material {
        base_colour: [1.0, 0.1, 0.1],
        emission_factor: 0.0,
    };
    materials[31] = Material {
        base_colour: [0.0, 0.0, 1.0],
        emission_factor: 10.0,
    };
    */

    let tree64 = svo_dag::Tree64::deserialize(std::io::Cursor::new(
        load_resource_bytes("sponza.tree64").await,
    ))
    .unwrap();

    queue.write_buffer(&pipelines.materials, 0, bytemuck::cast_slice(&materials));
    queue.write_buffer(
        &pipelines.tree_nodes,
        0,
        bytemuck::cast_slice(&tree64.nodes.inner),
    );

    let leaf_data: &[u8] = bytemuck::cast_slice(&tree64.data.inner);
    copy_aligned(&queue, &pipelines.leaf_data, &leaf_data, 0..leaf_data.len());

    let mut app = App {
        egui_renderer: egui_wgpu::Renderer::new(&device, swapchain_format, None, 1, false),
        egui_state: egui_winit::State::new(
            egui::Context::default(),
            egui::viewport::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        ),
        left_mouse_down: false,
        right_mouse_down: false,
        window: &window,
        resizables: Resizables::new(size.width, size.height, &device, &pipelines),
        config,
        surface,
        device,
        queue,
        accumulated_frame_index: 0,
        frame_index: 0,
        settings: Settings::default(),
        camera: CameraRig::builder()
            .with(Position::new(glam::Vec3::new(350.0, 150.0, 230.0)))
            .with(YawPitch::new().yaw_degrees(90.).pitch_degrees(-25.0))
            .with(Smooth::new_position_rotation(0.25, 0.25))
            .with(Arm::new(glam::Vec3::Z * 175.0))
            .build(),
        pipelines,
        materials,
        tree64,
    };

    event_loop.run_app(&mut app).unwrap()
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
    max_bounces: u32,
    background_colour: [f32; 3],
    sun_colour: [f32; 3],
    sun_strength: f32,
    background_strength: f32,
    vertical_fov: f32,
    show_heatmap: bool,
    edit_distance: f32,
    edit_size: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sun_long: 90.0_f32,
            sun_lat: 45.0_f32,
            enable_shadows: true,
            sun_apparent_size: 1.0_f32,
            accumulate_samples: true,
            max_bounces: 1,
            background_colour: [0.01; 3],
            sun_colour: [1.0; 3],
            sun_strength: 20.0,
            background_strength: 1.0,
            vertical_fov: 45.0_f32,
            show_heatmap: false,
            edit_distance: 10.0,
            edit_size: 10.0,
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
            inner: vec,
            padding: 0,
        }
    }
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Uniforms {
    p_inv: glam::Mat4,
    camera_pos: Vec3A,
    sun_direction: Vec3A,
    sun_emission: Vec3A,
    background_colour: Vec3A,
    resolution: glam::UVec2,
    settings: i32,
    frame_index: u32,
    cos_sun_apparent_size: f32,
    accumulated_frame_index: u32,
    max_bounces: u32,
    tree_scale: u32,
    root_node_index: u32,
    _padding: [u32; 3],
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod, Default)]
#[repr(C)]
struct Material {
    base_colour: [f32; 3],
    emission_factor: f32,
    linear_roughness: f32,
    metallic: f32,
    _padding: [u32; 2],
}

struct Pipelines {
    blit_srgb: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    filtering_sampler: wgpu::Sampler,
    non_filtering_sampler: wgpu::Sampler,
    trace: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    tree_nodes: wgpu::Buffer,
    leaf_data: wgpu::Buffer,
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
                compute_buffer(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                texture_entry(5, wgpu::ShaderStages::COMPUTE),
                sampler_entry(6, wgpu::ShaderStages::COMPUTE, false),
            ],
        });

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
                module: &if USE_SPIRV_SHADER {
                    unsafe {
                        device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                            label: None,
                            source: std::borrow::Cow::Borrowed(bytemuck::cast_slice(
                                &load_resource_bytes("shaders/raytrace.spv").await,
                            )),
                        })
                    }
                } else {
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(
                            load_resource_str("shaders/raytrace.wgsl").await.into(),
                        ),
                    })
                },
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
            tree_nodes: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 20_000_000,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            leaf_data: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 22_000_000,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            materials: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: std::mem::size_of::<[Material; 256]>() as _,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            tonemapping_lut: device.create_texture_with_data(
                queue,
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
                format: wgpu::TextureFormat::Rgba32Float,
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
                        resource: pipelines.tree_nodes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pipelines.leaf_data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pipelines.materials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &a.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(
                            &b.create_view(&Default::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
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
