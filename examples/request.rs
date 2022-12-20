use bevy::render::render_resource::ShaderType;
use bevy::{
    core::FrameCount,
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
            Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, ShaderStages,
        },
        renderer::RenderDevice,
    },
};
use bevy_readback::{
    ComputeRequest, ComputeRequestToken, ReadbackComponent, ReadbackComponentPlugin, ReadbackPlugin,
};

pub const NEXT_FRAME: bool = true;

fn main() {
    let mut app = App::new();

    app.add_plugins(DefaultPlugins);

    // add readback main plugin (in immediate or lazy mode)
    if NEXT_FRAME {
        app.add_plugin(ReadbackPlugin::next_frame());
    } else {
        app.add_plugin(ReadbackPlugin::whenever());
    }
    // add plugin per required request type
    app.add_plugin(ReadbackComponentPlugin::<GpuRequest>::default());

    app.add_system(run_compute_requests);
    app.add_startup_system(setup);
    app.run();
}

// not related, just showing something on the screen
fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());

    // Rectangle
    commands.spawn(SpriteBundle {
        sprite: Sprite {
            color: Color::rgb(0.25, 0.25, 0.75),
            custom_size: Some(Vec2::new(50.0, 100.0)),
            ..default()
        },
        ..default()
    });
}

fn run_compute_requests(
    mut req: ComputeRequest<GpuRequest>,
    mut token: Local<Option<ComputeRequestToken<GpuRequest>>>,
    mut counter: Local<u32>,
    frame: Res<FrameCount>,
) {
    if let Some(t) = token.as_ref() {
        if NEXT_FRAME {
            // blocking request,
            // will complete immediately before proper pipelining is implemented
            // will complete after render is complete with proper pipelining
            match req.get(*t) {
                Some(res) => {
                    info!("[{}] got response: {}", frame.0, res);
                    *token = None;
                    *counter = res;
                }
                None => (),
            }
        } else {
            // non-blocking request,
            // will be F+2 before proper pipelining
            // will be F+? after proper pipelining
            match req.try_get(*t) {
                Some(res) => {
                    info!("[{}] got response: {}", frame.0, res);
                    *token = None;
                    *counter = res;
                }
                None => (),
            }
        }
    }

    if *counter == 0 {
        *counter = 1;
    }

    if token.is_none() {
        *token = Some(req.request(*counter));
        info!("[{}] making request for {}", frame.0, *counter);
    }
}

#[derive(Component)]
pub struct GpuRequest {
    #[allow(dead_code)]
    input_buffer: Buffer,
    output_buffer: Buffer,
    bindgroup: BindGroup,
}

// implement readback trait
impl ReadbackComponent for GpuRequest {
    // input data
    type SourceData = u32;
    // data required in render world
    type RenderData = u32;
    // return type (must implement ShaderType, no runtime size elements)
    type Result = u32;

    // system param for prepare function
    type PrepareParam = SRes<RenderDevice>;

    // cheap extract from main world to render world
    fn extract(data: &Self::SourceData) -> Self::RenderData {
        *data
    }

    // build buffers etc; can use resources for persistent buffers
    fn prepare(
        render_data: Self::RenderData,
        layout: &BindGroupLayout,
        device: &SystemParamItem<Self::PrepareParam>,
    ) -> Self {
        let input_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::UNIFORM,
            contents: &render_data.to_le_bytes(),
        });

        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, // note: COPY_SRC is required for the output buffer
            mapped_at_creation: false,
        });

        let bindgroup = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            input_buffer,
            output_buffer,
            bindgroup,
        }
    }

    // return the bind group created in prepare
    fn bind_group(&self) -> BindGroup {
        self.bindgroup.clone()
    }

    // return a reference to the output buffer to be read back
    fn readback_source(&self) -> Buffer {
        self.output_buffer.clone()
    }

    // compute shader ref
    fn shader() -> bevy::render::render_resource::ShaderRef {
        "double.wgsl".into()
    }

    // entry point
    fn entry_point() -> std::borrow::Cow<'static, str> {
        "double".into()
    }

    // vec of layout entries for the shader, used in pipeline FromWorld at app startup
    fn bind_group_layout_entries() -> Vec<bevy::render::render_resource::BindGroupLayoutEntry> {
        vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: bevy::render::render_resource::BindingType::Buffer {
                    ty: bevy::render::render_resource::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(u32::min_size()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: bevy::render::render_resource::BindingType::Buffer {
                    ty: bevy::render::render_resource::BufferBindingType::Storage {
                        read_only: false,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: Some(u32::min_size()),
                },
                count: None,
            },
        ]
    }
}
