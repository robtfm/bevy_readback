use std::{
    borrow::Cow,
    marker::PhantomData,
    sync::{
        mpsc::{sync_channel, Receiver, SyncSender, TryRecvError},
        Arc, RwLock,
    },
};

use bevy::{
    ecs::system::{SystemParam, SystemParamItem},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_graph::{Node, RenderGraph},
        render_resource::{
            encase::private::{BufferRef, ReadFrom, Reader},
            BindGroup, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer,
            BufferDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, MapMode, PipelineCache, ShaderRef, ShaderSize, ShaderType,
            SpecializedComputePipeline, SpecializedComputePipelines,
        },
        renderer::RenderDevice,
        Extract, RenderApp, RenderStage,
    },
    utils::HashMap,
};

// chunk size for reading from buffer
pub const BLOCK_SIZE: usize = 1024;

// specify whether to force buffer map for next frame
#[derive(Resource)]
pub struct ReadbackPollDevice(bool);

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
        render_app.insert_resource(ReadbackPollDevice(self.poll_device));

        let node = ComputeNode::new(&mut render_app.world);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();
        graph.add_node("readback", node);

        render_app.add_system_to_stage(RenderStage::Cleanup, map_buffers);
        render_app.add_system_to_stage(RenderStage::Cleanup, poll_device.after(map_buffers));
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
        app.init_non_send_resource::<ComputeResponses<T>>();
        app.init_resource::<BufferPool<T>>();
        app.add_plugin(ExtractResourcePlugin::<BufferPool<T>>::default());
        app.add_system_to_stage(CoreStage::First, cleanup::<T>);

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GpuComputeRequests<T>>();
        render_app.init_resource::<ReadbackComputePipeline<T>>();
        render_app.init_resource::<SpecializedComputePipelines<ReadbackComputePipeline<T>>>();
        render_app.add_system_to_stage(RenderStage::Extract, extract_requests::<T>);
        render_app.add_system_to_stage(RenderStage::Queue, queue_requests::<T>);
    }
}

pub trait ReadbackComponent: 'static + Component + Send + Sync {
    type SourceData: Send + Sync;
    type RenderData: Send + Sync;
    type Result: Send + Sync + ShaderType + ShaderSize + ReadFrom + Default;

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

enum BufferState {
    InUse,
    Free,
}

struct BufferWithState {
    buffer: Buffer,
    state: BufferState,
}

#[derive(Resource, ExtractResource)]
pub struct BufferPool<T: ReadbackComponent> {
    buffers: Vec<Arc<RwLock<BufferWithState>>>,
    _p: PhantomData<fn(T)>,
}

impl<T: ReadbackComponent> Default for BufferPool<T> {
    fn default() -> Self {
        Self {
            buffers: Default::default(),
            _p: Default::default(),
        }
    }
}

impl<T: ReadbackComponent> Clone for BufferPool<T> {
    fn clone(&self) -> Self {
        Self {
            buffers: self.buffers.clone(),
            _p: self._p,
        }
    }
}

#[derive(Resource)]
pub struct ComputeRequests<T: ReadbackComponent> {
    reqs: Vec<(T::SourceData, usize, SyncSender<bool>)>,
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
    resps: HashMap<ComputeRequestToken<T>, (usize, Receiver<bool>)>,
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
    requests: ResMut<'w, ComputeRequests<T>>,
    responses: NonSendMut<'w, ComputeResponses<T>>,
    pool: ResMut<'w, BufferPool<T>>,
    render_device: Res<'w, RenderDevice>,
    #[system_param(ignore)]
    _p: PhantomData<fn(&'s T)>,
}

trait ConstSize {
    const SIZE: usize;
}

impl<'w, 's, T: ReadbackComponent> ConstSize for ComputeRequest<'w, 's, T> {
    const SIZE: usize = <T::Result as ShaderSize>::SHADER_SIZE.get() as usize;
}

fn num_blocks<T: ReadbackComponent>() -> usize {
    (T::Result::SHADER_SIZE.get() as f32 / BLOCK_SIZE as f32).ceil() as usize
}

pub enum ComputeError {
    NotReady,
    Failed,
}

impl<'w, 's, T: ReadbackComponent> ComputeRequest<'w, 's, T> {
    pub fn request(&mut self, data: T::SourceData) -> ComputeRequestToken<T> {
        let token = self.dispenser.next();
        let next = self
            .pool
            .buffers
            .iter()
            .enumerate()
            .find(|(_, b)| matches!(b.read().unwrap().state, BufferState::Free));

        let ix = match next {
            None => {
                // info!("creating new buffer {}", self.pool.buffers.len());
                let size = (BLOCK_SIZE * num_blocks::<T>()) as u64;
                let buffer = self.render_device.create_buffer(&BufferDescriptor {
                    label: None,
                    size,
                    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let ix = self.pool.buffers.len();
                self.pool
                    .buffers
                    .push(Arc::new(RwLock::new(BufferWithState {
                        buffer,
                        state: BufferState::InUse,
                    })));
                ix
            }
            Some((ix, buffer_with_state)) => {
                buffer_with_state.write().unwrap().state = BufferState::InUse;
                ix
            }
        };

        let (sender, receiver) = sync_channel(1);

        self.requests.reqs.push((data, ix, sender));
        self.responses.resps.insert(token, (ix, receiver));
        token
    }

    pub fn try_get(&mut self, token: ComputeRequestToken<T>) -> Result<T::Result, ComputeError> {
        let Some((ix, receiver)) = self.responses.resps.remove(&token) else {
            return Err(ComputeError::Failed);
        };

        match receiver.try_recv() {
            Ok(true) => (),
            Err(TryRecvError::Empty) => {
                self.responses.resps.insert(token, (ix, receiver));
                return Err(ComputeError::NotReady);
            }
            _ => return Err(ComputeError::Failed),
        }

        Ok(self.read_buffer(ix))
    }

    pub fn get(&mut self, token: ComputeRequestToken<T>) -> Result<T::Result, ComputeError> {
        let Some((ix, receiver)) = self.responses.resps.remove(&token) else {
            return Err(ComputeError::Failed);
        };

        match receiver.recv() {
            Ok(true) => (),
            _ => return Err(ComputeError::Failed),
        }

        Ok(self.read_buffer(ix))
    }

    fn read_buffer(&mut self, ix: usize) -> T::Result {
        let mut buffer_with_state = self.pool.buffers[ix].write().unwrap();

        // have to copy to vec because "https://users.rust-lang.org/t/cant-use-generic-parameters-from-outer-function/62390"
        let size = T::Result::SHADER_SIZE.get() as usize;
        let mut vec = Vec::with_capacity(size);
        let view = buffer_with_state.buffer.slice(..).get_mapped_range();
        while vec.len() < size {
            vec.extend(view.read::<BLOCK_SIZE>(vec.len()));
        }

        let mut res = T::Result::default();
        let mut reader = Reader::new::<T::Result>(&vec, 0).unwrap();
        res.read_from(&mut reader);
        drop(view);
        buffer_with_state.buffer.unmap();

        buffer_with_state.state = BufferState::Free;

        res
    }
}

#[derive(Resource)]
pub struct GpuComputeRequests<T: ReadbackComponent> {
    reqs: Vec<(T::RenderData, usize, SyncSender<bool>)>,
}

impl<T: ReadbackComponent> Default for GpuComputeRequests<T> {
    fn default() -> Self {
        Self {
            reqs: Default::default(),
        }
    }
}

fn extract_requests<T: ReadbackComponent>(
    reqs: Extract<Res<ComputeRequests<T>>>,
    mut render_reqs: ResMut<GpuComputeRequests<T>>,
) {
    render_reqs.reqs.clear();
    render_reqs.reqs.extend(
        reqs.reqs
            .iter()
            .map(|(source_data, ix, sender)| (T::extract(source_data), *ix, sender.clone())),
    );
}

fn cleanup<T: ReadbackComponent>(mut reqs: ResMut<ComputeRequests<T>>) {
    reqs.reqs.clear();
}

#[derive(Component)]
struct ComputePhaseItem {
    pipeline: CachedComputePipelineId,
    bindgroup: BindGroup,
    source: Buffer,
    dest: Arc<RwLock<BufferWithState>>,
    size: u64,
    sender: SyncSender<bool>,
}

fn queue_requests<T: ReadbackComponent>(
    mut reqs: ResMut<GpuComputeRequests<T>>,
    mut pipelines: ResMut<SpecializedComputePipelines<ReadbackComputePipeline<T>>>,
    pipeline: Res<ReadbackComputePipeline<T>>,
    mut cache: ResMut<PipelineCache>,
    mut commands: Commands,
    param: SystemParamItem<T::PrepareParam>,
    pool: Res<BufferPool<T>>,
) {
    for (render_data, ix, sender) in reqs.reqs.drain(..) {
        let component = T::prepare(render_data, &pipeline.layout, &param);
        let bindgroup = component.bind_group();
        let source = component.readback_source();

        let pipeline = pipelines.specialize(&mut cache, &pipeline, ());
        let size = T::Result::min_size().get();

        commands.spawn(ComputePhaseItem {
            pipeline,
            bindgroup,
            source,
            dest: pool.buffers[ix].clone(),
            size,
            sender,
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
                &item.dest.read().unwrap().buffer,
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
        item.dest
            .read()
            .unwrap()
            .buffer
            .slice(..)
            .map_async(MapMode::Read, move |res| {
                sender.send(res.is_ok()).unwrap();
            });
    }
}

fn poll_device(device: Res<RenderDevice>, poll: Res<ReadbackPollDevice>) {
    if poll.0 {
        device.poll(wgpu::Maintain::Wait);
    }
}
