use crate::{DrawVoxelCubesCommands, VoxelModel};
use bevy::prelude::*;
use bevy::render::view::RenderVisibleEntities;
use bevy::render::{render_phase::*, sync_world::MainEntity};
use std::ops::Range;

pub struct VoxelBinnedPhaseItem {
    pub draw_function: DrawFunctionId,
    // An entity from which data will be fetched, including the mesh if
    // applicable.
    pub representative_entity: (Entity, MainEntity),
    // The ranges of instances.
    pub batch_range: Range<u32>,
    // An extra index, which is either a dynamic offset or an index in the
    // indirect parameters list.
    pub extra_index: PhaseItemExtraIndex,
}

impl PhaseItem for VoxelBinnedPhaseItem {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }

    #[inline]
    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index
    }

    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for VoxelBinnedPhaseItem {
    type BinKey = VoxelBinnedPhaseItemBinKey;

    #[inline]
    fn new(
        key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        Self {
            draw_function: key.draw_function,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

// Data that must be identical in order to batch phase items together.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoxelBinnedPhaseItemBinKey {
    pub draw_function: DrawFunctionId,
}

pub fn queue_voxel_phase_items(
    mut render_phrases: ResMut<ViewBinnedRenderPhases<VoxelBinnedPhaseItem>>,
    draw_functions: Res<DrawFunctions<VoxelBinnedPhaseItem>>,
    views: Query<(Entity, &RenderVisibleEntities)>,
) {
    let draw_function = draw_functions.read().id::<DrawVoxelCubesCommands>();

    // Render phases are per-view, so we need to iterate over all views so that
    // the entity appears in them. (In this example, we have only one view, but
    // it's good practice to loop over all views anyway.)
    for (view_entity, view_visible_entities) in &views {
        let render_phrase = render_phrases.entry(view_entity).or_default();
        render_phrase.clear();

        // Find all the custom rendered entities that are visible from this
        // view.
        for &entity in view_visible_entities.get::<With<VoxelModel>>().iter() {
            // Add the custom render item. We use the
            // [`BinnedRenderPhaseType::NonMesh`] type to skip the special
            // handling that Bevy has for meshes (preprocessing, indirect
            // draws, etc.)
            //
            // The asset ID is arbitrary; we simply use [`AssetId::invalid`],
            // but you can use anything you like. Note that the asset ID need
            // not be the ID of a [`Mesh`].
            render_phrase.add(
                VoxelBinnedPhaseItemBinKey { draw_function },
                entity,
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}
