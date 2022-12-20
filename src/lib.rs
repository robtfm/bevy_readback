use std::{
    borrow::Cow,
    marker::PhantomData,
    sync::{
        mpsc::{RecvError, TryRecvError},
        Mutex,
    },
};

use bevy::{
    ecs::system::{SystemParam, SystemParamItem},
    prelude::*,
    render::{
        render_graph::{Node, RenderGraph},
        render_resource::{
            encase::private::{BufferRef, ReadFrom, Reader},
            BindGroup, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer,
            BufferDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, MapMode, PipelineCache, ShaderRef, ShaderType,
            SpecializedComputePipeline, SpecializedComputePipelines,
        },
        renderer::RenderDevice,
        Extract, RenderApp, RenderStage,
    },
    utils::HashMap,
};

// chunk size for data returned from render world to main world via channel
pub const BLOCK_SIZE: usize = 1024;

#[derive(Default)]
pub struct ReadbackPlugin {
    poll_device: bool,
}

impl ReadbackPlugin {
    pub fn whenever() -> Self {
        Self { poll_device: false }
    }

    pub fn next_frame() -> Self {
        Self { poll_device: true }
    }
}

impl Plugin for ReadbackPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ComputeRequestTokenDispenser>();

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_system_to_stage(RenderStage::Cleanup, map_buffers);

        let node = ComputeNode::new(&mut render_app.world);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();
        graph.add_node("readback", node);

        if self.poll_device {
            render_app.add_system_to_stage(RenderStage::Cleanup, poll_device.after(map_buffers));
        }
    }
}

pub struct ReadbackComponentPlugin<T: ReadbackComponent> {
    _p: PhantomData<fn(T)>,
}

impl<T: ReadbackComponent> Default for ReadbackComponentPlugin<T> {
    fn default() -> Self {
        Self {
            _p: Default::default(),
        }
    }
}

impl<T: ReadbackComponent> Plugin for ReadbackComponentPlugin<T> {
    fn build(&self, app: &mut App) {
        app.init_resource::<ComputeRequests<T>>();
        app.init_resource::<ComputeResponses<T>>();
        app.add_system_to_stage(CoreStage::First, clear_requests::<T>);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<RenderComputeRequests<T>>();
        render_app.init_resource::<ReadbackComputePipeline<T>>();
        render_app.init_resource::<SpecializedComputePipelines<ReadbackComputePipeline<T>>>();
        render_app.add_system_to_stage(RenderStage::Extract, extract_requests::<T>);
        render_app.add_system_to_stage(RenderStage::Queue, queue_requests::<T>);
    }
}

pub trait ReadbackComponent: 'static + Component + Send + Sync {
    type SourceData: Send + Sync;
    type RenderData: Send + Sync;
    type Result: Send + Sync + ShaderType + ReadFrom + Default;

    type PrepareParam: SystemParam;

    fn extract(data: &Self::SourceData) -> Self::RenderData;
    fn prepare(
        render_data: Self::RenderData,
        layout: &BindGroupLayout,
        param: &SystemParamItem<Self::PrepareParam>,
    ) -> Self;
    fn bind_group(&self) -> BindGroup;
    fn readback_source(&self) -> Buffer;
    fn shader() -> ShaderRef;
    fn entry_point() -> Cow<'static, str>;
    fn bind_group_layout_entries() -> Vec<BindGroupLayoutEntry>;
}

#[derive(PartialOrd, Ord)]
pub struct ComputeRequestToken<T: ReadbackComponent> {
    id: u32,
    _p: PhantomData<fn(T)>,
}

impl<T: ReadbackComponent> Clone for ComputeRequestToken<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            _p: Default::default(),
        }
    }
}
impl<T: ReadbackComponent> Copy for ComputeRequestToken<T> {}

impl<T: ReadbackComponent> PartialEq for ComputeRequestToken<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: ReadbackComponent> Eq for ComputeRequestToken<T> {}

impl<T: ReadbackComponent> std::hash::Hash for ComputeRequestToken<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self._p.hash(state);
    }
}

impl<T: ReadbackComponent> ComputeRequestToken<T> {
    fn new(id: u32) -> Self {
        Self {
            id,
            _p: PhantomData,
        }
    }
}

#[derive(Resource, Default)]
pub struct ComputeRequestTokenDispenser {
    next: u32,
}

impl ComputeRequestTokenDispenser {
    fn next<T: ReadbackComponent>(&mut self) -> ComputeRequestToken<T> {
        self.next = self.next.wrapping_add(1);
        ComputeRequestToken::new(self.next)
    }
}

#[derive(Resource)]
pub struct ComputeRequests<T: ReadbackComponent> {
    reqs: Vec<(T::SourceData, std::sync::mpsc::SyncSender<[u8; BLOCK_SIZE]>)>,
}

impl<T: ReadbackComponent> Default for ComputeRequests<T> {
    fn default() -> Self {
        Self {
            reqs: Default::default(),
        }
    }
}

#[derive(Resource)]
pub struct ComputeResponses<T: ReadbackComponent> {
    resps: Mutex<HashMap<ComputeRequestToken<T>, std::sync::mpsc::Receiver<[u8; BLOCK_SIZE]>>>,
}

impl<T: ReadbackComponent> Default for ComputeResponses<T> {
    fn default() -> Self {
        Self {
            resps: Default::default(),
        }
    }
}

#[derive(SystemParam)]
pub struct ComputeRequest<'w, 's, T: ReadbackComponent> {
    dispenser: ResMut<'w, ComputeRequestTokenDispenser>,
    requests: NonSendMut<'w, ComputeRequests<T>>,
    responses: NonSendMut<'w, ComputeResponses<T>>,
    #[system_param(ignore)]
    _p: PhantomData<fn(&'s T)>,
}

fn num_blocks<T: ReadbackComponent>() -> usize {
    (T::Result::min_size().get() as f32 / BLOCK_SIZE as f32).ceil() as usize
}

impl<'w, 's, T: ReadbackComponent> ComputeRequest<'w, 's, T> {
    pub fn request(&mut self, data: T::SourceData) -> ComputeRequestToken<T> {
        let token = self.dispenser.next();
        let (sender, receiver) = std::sync::mpsc::sync_channel(num_blocks::<T>());
        self.requests.reqs.push((data, sender));
        self.responses.resps.lock().unwrap().insert(token, receiver);
        token
    }

    pub fn try_get(&mut self, token: ComputeRequestToken<T>) -> Option<T::Result> {
        let mut guard = self.responses.resps.lock().unwrap();
        let Some(receiver) = guard.remove(&token) else { return None; };

        match receiver.try_recv() {
            Ok(chunk) => {
                let mut buffer: Vec<u8> = Vec::with_capacity(T::Result::min_size().get() as usize);
                buffer.extend_from_slice(&chunk);
                for _ in 1..num_blocks::<T>() {
                    let Ok(chunk) = receiver.recv() else {
                        return None
                    };

                    buffer.extend_from_slice(&chunk);
                }

                let mut res = T::Result::default();
                let mut reader = Reader::new::<T::Result>(&buffer, 0).unwrap();
                res.read_from(&mut reader);
                Some(res)
            }
            Err(TryRecvError::Empty) => {
                guard.insert(token, receiver);
                None
            }
            Err(TryRecvError::Disconnected) => None,
        }
    }

    pub fn get(&mut self, token: ComputeRequestToken<T>) -> Option<T::Result> {
        let mut guard = self.responses.resps.lock().unwrap();
        let Some(receiver) = guard.remove(&token) else { return None; };

        match receiver.recv() {
            Ok(chunk) => {
                let mut buffer: Vec<u8> = Vec::with_capacity(T::Result::min_size().get() as usize);
                buffer.extend_from_slice(&chunk);
                for _ in 1..num_blocks::<T>() {
                    let Ok(chunk) = receiver.recv() else {
                        return None
                    };

                    buffer.extend_from_slice(&chunk);
                }

                let mut res = T::Result::default();
                let mut reader = Reader::new::<T::Result>(&buffer, 0).unwrap();
                res.read_from(&mut reader);
                Some(res)
            }
            Err(RecvError) => None,
        }
    }
}

#[derive(Resource)]
pub struct RenderComputeRequests<T: ReadbackComponent> {
    reqs: Vec<(T::RenderData, std::sync::mpsc::SyncSender<[u8; BLOCK_SIZE]>)>,
}

impl<T: ReadbackComponent> Default for RenderComputeRequests<T> {
    fn default() -> Self {
        Self {
            reqs: Default::default(),
        }
    }
}

fn extract_requests<T: ReadbackComponent>(
    reqs: Extract<NonSend<ComputeRequests<T>>>,
    mut render_reqs: NonSendMut<RenderComputeRequests<T>>,
) {
    render_reqs.reqs.clear();
    render_reqs.reqs.extend(
        reqs.reqs
            .iter()
            .map(|(source_data, sender)| (T::extract(source_data), sender.clone())),
    );
}

fn clear_requests<T: ReadbackComponent>(mut reqs: NonSendMut<ComputeRequests<T>>) {
    reqs.reqs.clear();
}

#[derive(Component)]
struct ComputePhaseItem {
    pipeline: CachedComputePipelineId,
    bindgroup: BindGroup,
    source: Buffer,
    dest: Buffer,
    size: u64,
    sender: std::sync::mpsc::SyncSender<[u8; BLOCK_SIZE]>,
}

fn queue_requests<T: ReadbackComponent>(
    mut reqs: ResMut<RenderComputeRequests<T>>,
    device: Res<RenderDevice>,
    mut pipelines: ResMut<SpecializedComputePipelines<ReadbackComputePipeline<T>>>,
    pipeline: Res<ReadbackComputePipeline<T>>,
    mut cache: ResMut<PipelineCache>,
    mut commands: Commands,
    param: SystemParamItem<T::PrepareParam>,
) {
    for (render_data, sender) in reqs.reqs.drain(..) {
        let component = T::prepare(render_data, &pipeline.layout, &param);
        let bindgroup = component.bind_group();
        let source = component.readback_source();

        let pipeline = pipelines.specialize(&mut cache, &pipeline, ());
        let blocky_size = (BLOCK_SIZE * num_blocks::<T>()) as u64;
        let size = T::Result::min_size().get();

        // todo cache buffer
        let dest = device.create_buffer(&BufferDescriptor {
            label: None,
            size: blocky_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        commands.spawn(ComputePhaseItem {
            pipeline,
            bindgroup,
            source,
            dest,
            size,
            sender: sender.clone(),
        });
    }
}

#[derive(Resource)]
pub struct ReadbackComputePipeline<T: ReadbackComponent> {
    layout: BindGroupLayout,
    shader: Handle<Shader>,
    _p: PhantomData<fn(T)>,
}

impl<T: ReadbackComponent> FromWorld for ReadbackComputePipeline<T> {
    fn from_world(world: &mut World) -> Self {
        let layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    entries: &T::bind_group_layout_entries(),
                    label: Some(std::any::type_name::<T>()),
                });

        Self {
            layout,
            shader: match T::shader() {
                ShaderRef::Default => panic!(),
                ShaderRef::Handle(h) => h,
                ShaderRef::Path(p) => world.resource::<AssetServer>().load(p),
            },
            _p: Default::default(),
        }
    }
}

impl<T: ReadbackComponent> SpecializedComputePipeline for ReadbackComputePipeline<T> {
    type Key = ();

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![self.layout.clone()]),
            shader: self.shader.clone(),
            shader_defs: Vec::default(),
            entry_point: T::entry_point(),
        }
    }
}

struct ComputeNode {
    query: QueryState<&'static ComputePhaseItem>,
}

impl ComputeNode {
    fn new(world: &mut World) -> Self {
        Self {
            query: world.query(),
        }
    }
}

impl Node for ComputeNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();

        for item in self.query.iter_manual(world) {
            let mut pass = render_context
                .command_encoder
                .begin_compute_pass(&ComputePassDescriptor { label: None });
            let Some(pipeline) = pipeline_cache.get_compute_pipeline(item.pipeline) else { continue };

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &item.bindgroup, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            drop(pass);

            render_context.command_encoder.copy_buffer_to_buffer(
                &item.source,
                0,
                &item.dest,
                0,
                item.size,
            );
        }

        Ok(())
    }
}

fn map_buffers(query: Query<&ComputePhaseItem>) {
    for item in &query {
        let sender = item.sender.clone();
        let output_buffer = item.dest.clone();
        let size = item.size;
        item.dest
            .slice(..)
            .map_async(MapMode::Read, move |res| match res {
                Ok(_) => {
                    let view = output_buffer.slice(..).get_mapped_range();
                    let mut sent = 0;
                    while (sent as u64) < size {
                        let data = view.read::<BLOCK_SIZE>(sent);
                        if let Err(e) = sender.send(*data) {
                            warn!("failed to send complete notice: {}", e);
                        }
                        sent += BLOCK_SIZE;
                    }
                }
                Err(e) => warn!("failed to map buffer: {}", e),
            });
    }
}

fn poll_device(device: Res<RenderDevice>) {
    device.poll(wgpu::Maintain::Wait);
}
