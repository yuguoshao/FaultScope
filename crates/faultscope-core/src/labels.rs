use std::collections::HashMap;

use crate::{NoiseLocation, NoiseModel, TagValue};

/// Dense identifier for a user-facing noise-location label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct LocationId(usize);

impl LocationId {
    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub(crate) struct TagValueId(pub(crate) usize);

impl TagValueId {
    pub(crate) const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HotspotTagKind {
    Round,
    Gate,
    Operation,
}

impl HotspotTagKind {
    const ALL: [Self; 3] = [Self::Round, Self::Gate, Self::Operation];

    const fn index(self) -> usize {
        match self {
            Self::Round => 0,
            Self::Gate => 1,
            Self::Operation => 2,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Round => "round",
            Self::Gate => "gate",
            Self::Operation => "operation",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LocationLabel {
    label: String,
    tags: HashMap<String, TagValue>,
    hotspot_tags: [Option<TagValueId>; 3],
    lexical_rank: usize,
}

/// Boundary catalog retaining user-facing labels outside computation-heavy data.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LocationCatalog {
    locations: Vec<LocationLabel>,
    tag_values: Vec<TagValue>,
}

impl LocationCatalog {
    pub fn len(&self) -> usize {
        self.locations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    pub fn label(&self, location_id: LocationId) -> &str {
        &self.locations[location_id.index()].label
    }

    pub fn tags(&self, location_id: LocationId) -> &HashMap<String, TagValue> {
        &self.locations[location_id.index()].tags
    }

    pub fn find_label(&self, label: &str) -> Option<LocationId> {
        self.locations
            .iter()
            .position(|location| location.label == label)
            .map(LocationId::new)
    }

    pub(crate) fn hotspot_tag(
        &self,
        location_id: LocationId,
        kind: HotspotTagKind,
    ) -> Option<TagValueId> {
        self.locations[location_id.index()].hotspot_tags[kind.index()]
    }

    pub(crate) fn tag_value(&self, tag_value_id: TagValueId) -> &TagValue {
        &self.tag_values[tag_value_id.index()]
    }

    pub(crate) fn tag_value_count(&self) -> usize {
        self.tag_values.len()
    }

    pub(crate) fn lexical_rank(&self, location_id: LocationId) -> usize {
        self.locations[location_id.index()].lexical_rank
    }

    pub(crate) fn materialize_noise_location(
        &self,
        location: &IndexedNoiseLocation,
    ) -> NoiseLocation {
        NoiseLocation {
            id: self.label(location.location_id).to_string(),
            model: location.model.clone(),
            rate: location.rate,
            qubits: location.qubits.clone(),
            tags: self.tags(location.location_id).clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexedNoiseLocation {
    pub location_id: LocationId,
    pub model: NoiseModel,
    pub rate: f64,
    pub qubits: Vec<usize>,
}

impl IndexedNoiseLocation {
    pub(crate) fn from_input(location_id: LocationId, location: NoiseLocation) -> Self {
        Self {
            location_id,
            model: location.model,
            rate: location.rate,
            qubits: location.qubits,
        }
    }
}

pub(crate) struct LocationCatalogBuilder {
    location_ids: HashMap<String, LocationId>,
    locations: Vec<LocationLabel>,
    tag_value_ids: HashMap<TagValue, TagValueId>,
    tag_values: Vec<TagValue>,
}

impl LocationCatalogBuilder {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            location_ids: HashMap::with_capacity(capacity),
            locations: Vec::with_capacity(capacity),
            tag_value_ids: HashMap::new(),
            tag_values: Vec::new(),
        }
    }

    /// Intern a location label. The first tag set wins for repeated labels,
    /// matching the historical DEM hotspot grouping behavior.
    pub(crate) fn intern(&mut self, label: String, tags: HashMap<String, TagValue>) -> LocationId {
        self.intern_with_status(label, tags).0
    }

    pub(crate) fn intern_with_status(
        &mut self,
        label: String,
        tags: HashMap<String, TagValue>,
    ) -> (LocationId, bool) {
        if let Some(location_id) = self.location_ids.get(&label) {
            return (*location_id, false);
        }

        let location_id = LocationId::new(self.locations.len());
        let mut hotspot_tags = [None; 3];
        for kind in HotspotTagKind::ALL {
            if let Some(tag_value) = tags.get(kind.label()) {
                hotspot_tags[kind.index()] = Some(self.intern_tag_value(tag_value.clone()));
            }
        }
        self.location_ids.insert(label.clone(), location_id);
        self.locations.push(LocationLabel {
            label,
            tags,
            hotspot_tags,
            lexical_rank: 0,
        });
        (location_id, true)
    }

    pub(crate) fn finish(mut self) -> LocationCatalog {
        let mut lexical_order = (0..self.locations.len()).collect::<Vec<_>>();
        lexical_order.sort_unstable_by(|left, right| {
            self.locations[*left]
                .label
                .cmp(&self.locations[*right].label)
        });
        for (rank, location_index) in lexical_order.into_iter().enumerate() {
            self.locations[location_index].lexical_rank = rank;
        }
        LocationCatalog {
            locations: self.locations,
            tag_values: self.tag_values,
        }
    }

    fn intern_tag_value(&mut self, value: TagValue) -> TagValueId {
        if let Some(tag_value_id) = self.tag_value_ids.get(&value) {
            return *tag_value_id;
        }
        let tag_value_id = TagValueId(self.tag_values.len());
        self.tag_value_ids.insert(value.clone(), tag_value_id);
        self.tag_values.push(value);
        tag_value_id
    }
}
