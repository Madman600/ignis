//! Expert residency's reservations as **plan lines** computed at load (ADR
//! 0030): the host plan (the pinned expert pool, the n-gram hot rows, staging
//! buffers, and prompt reuse's host retained slots and KV-RAM arena) with its
//! refusal margin, and the VRAM lines — the expert cache,
//! split into eight K-class pools, the prefill staging ring and residency's
//! tables.
//!
//! The VRAM lines compose with [`crate::vram::plan_vram`] rather than
//! extending its line set: the Flash-Next load names its KV pool
//! (`kv_pool_bytes`), so the plan leaves `budget − total` free, and the
//! expert cache takes that rest after the staging ring and the tables. Pure
//! arithmetic over numbers the loader measured or read from the artifact;
//! every refusal is pinned on the CPU.

use super::class::{ExpertCatalog, KClass, Projection, ProjectionId};
use super::policy::PrefetchBudget;

const GIB: u64 = 1024 * 1024 * 1024;

/// What the host plan leaves of the available physical memory unless the
/// operator names another (`--model-host-margin-bytes`): below it, Windows
/// starts paging. Spec flash-next/03 set "6 GB", taken as GiB; 4 GiB since
/// 2026-10-12 (the owner's call): on the 64 GB box with its working desktop,
/// 6 GiB left Flash-Next no room for its KV-RAM arena.
pub const DEFAULT_HOST_MARGIN_BYTES: u64 = 4 * GIB;

/// The smallest VRAM expert cache a load accepts (spec flash-next/03: "12
/// GB", taken as GiB). Below it the hit rate, and with it decode speed,
/// collapses.
pub const EXPERT_CACHE_FLOOR_BYTES: u64 = 12 * GIB;

/// What the host plan is made from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPlanRequest {
    /// Physical memory available at load (`MEMORYSTATUSEX::ullAvailPhys`),
    /// measured before the first pinned allocation.
    pub available_physical_bytes: u64,
    /// Every expert projection, pinned for the life of the load: the
    /// [`ExpertCatalog::total_bytes`] of the artifact.
    pub expert_pool_bytes: u64,
    /// The n-gram table's hot rows held in RAM (spec flash-next/04).
    pub ngram_hot_rows_bytes: u64,
    /// Host staging buffers: the artifact read buffers and the like.
    pub staging_bytes: u64,
    /// Prompt reuse's host retained slots (`--reuse-retained-host`, spec
    /// flash-next/05): one pinned block of a state image per slot.
    pub retained_host_slots_bytes: u64,
    /// Prompt reuse's KV-RAM arena (`--reuse-kv-host-pool-bytes`, spec
    /// flash-next/05): materialized blobs the device gave up.
    pub kv_ram_arena_bytes: u64,
    /// What the plan must leave of `available_physical_bytes`
    /// (`--model-host-margin-bytes`, [`DEFAULT_HOST_MARGIN_BYTES`] unless
    /// named): not a line, the room the lines may not take.
    pub margin_bytes: u64,
}

/// The host plan the load carries out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostPlan {
    pub request: HostPlanRequest,
    pub total_bytes: u64,
    /// What stays available after the plan: at least the request's margin.
    pub left_bytes: u64,
}

impl HostPlan {
    /// `(name, bytes)` in plan order.
    pub fn entries(&self) -> [(&'static str, u64); HOST_PLAN_LINES] {
        host_lines(&self.request)
    }
}

/// The host plan's lines.
pub const HOST_PLAN_LINES: usize = 5;

/// The lines in plan order: what the model needs to run first, prompt
/// reuse's last, so a plan short of room names a reuse line -- the one an
/// operator can shrink without losing the model -- whenever giving up reuse
/// would make it fit.
fn host_lines(request: &HostPlanRequest) -> [(&'static str, u64); HOST_PLAN_LINES] {
    [
        ("expert_pool", request.expert_pool_bytes),
        ("ngram_hot_rows", request.ngram_hot_rows_bytes),
        ("staging", request.staging_bytes),
        ("retained_host_slots", request.retained_host_slots_bytes),
        ("kv_ram_arena", request.kv_ram_arena_bytes),
    ]
}

/// A load the host plan refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostPlanError {
    /// The plan leaves less than the request's margin of the available
    /// physical memory.
    BelowMargin {
        available_bytes: u64,
        planned_bytes: u64,
        margin_bytes: u64,
        /// The first line, in plan order, whose running total passes what
        /// the margin leaves usable.
        crossing_line: &'static str,
        lines: [(&'static str, u64); HOST_PLAN_LINES],
    },
}

impl HostPlanError {
    /// What the plan needed available: its lines and the margin.
    pub fn needed_bytes(&self) -> u64 {
        let Self::BelowMargin { planned_bytes, margin_bytes, .. } = self;
        planned_bytes.saturating_add(*margin_bytes)
    }

    /// How much more available memory the plan needed than there was.
    pub fn short_bytes(&self) -> u64 {
        let Self::BelowMargin { available_bytes, .. } = self;
        self.needed_bytes().saturating_sub(*available_bytes)
    }
}

/// For a person, in the binary units the plan is laid out in: what the load
/// needed, what it had, how far short it fell, and what would make room.
/// The exact bytes are the refusal's log event's
/// (`ignis.runtime.flash_next_host_plan_refused`) and `GET /v1/models`'s.
impl std::fmt::Display for HostPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self::BelowMargin {
            available_bytes,
            planned_bytes,
            margin_bytes,
            crossing_line,
            lines,
        } = self;
        write!(
            f,
            "not enough RAM: the load pins {} and keeps {} free for the system, so it needs {} \
             available, but only {} is ({} short). ",
            human_bytes(*planned_bytes),
            human_bytes(*margin_bytes),
            human_bytes(self.needed_bytes()),
            human_bytes(*available_bytes),
            human_bytes(self.short_bytes()),
        )?;
        match *crossing_line {
            "expert_pool" => write!(f, "Even the experts alone ({}) do not fit: close other applications", human_bytes(lines[0].1)),
            "ngram_hot_rows" => f.write_str(
                "The experts fit, the n-gram hot rows do not: hold fewer (--ngram-hot-bytes) or close other applications",
            ),
            "staging" => f.write_str(
                "The experts and the n-gram hot rows fit, KV-disk's staging buffers do not: \
                 --qwen38flashnext-kv-disk-bytes 0 drops them, or close other applications",
            ),
            "retained_host_slots" => f.write_str(
                "The model fits, prompt reuse's retained host slots do not: fewer of them \
                 (--qwen38flashnext-reuse-retained-host) or close other applications",
            ),
            "kv_ram_arena" => f.write_str(
                "The model and its retained slots fit, the KV-RAM arena does not: a smaller one \
                 (--qwen38flashnext-reuse-kv-host-pool-bytes, 0 turns it off) or close other applications",
            ),
            _ => f.write_str("Close other applications"),
        }?;
        f.write_str(". Plan: ")?;
        for (i, (name, bytes)) in lines.iter().enumerate() {
            write!(f, "{}{} {}", if i == 0 { "" } else { ", " }, line_label(name), human_bytes(*bytes))?;
        }
        write!(f, "; margin {} (--qwen38flashnext-model-host-margin-bytes).", human_bytes(*margin_bytes))
    }
}

impl std::error::Error for HostPlanError {}

/// A plan line as the refusal names it to a person.
fn line_label(name: &str) -> &str {
    match name {
        "expert_pool" => "experts",
        "ngram_hot_rows" => "n-gram hot rows",
        "retained_host_slots" => "retained host slots",
        "kv_ram_arena" => "KV-RAM arena",
        other => other,
    }
}

/// `4 GiB`, `35.2 GiB`, `992 MiB`, `512 KiB`, `100 bytes`: binary units,
/// whole GiB without a decimal, one decimal otherwise, MiB and KiB rounded.
pub fn human_bytes(bytes: u64) -> String {
    const MIB: u64 = 1 << 20;
    const KIB: u64 = 1 << 10;
    if bytes >= GIB {
        if bytes % GIB == 0 {
            format!("{} GiB", bytes / GIB)
        } else {
            format!("{:.1} GiB", bytes as f64 / GIB as f64)
        }
    } else if bytes >= MIB {
        format!("{} MiB", (bytes as f64 / MIB as f64).round() as u64)
    } else if bytes >= KIB {
        format!("{} KiB", (bytes as f64 / KIB as f64).round() as u64)
    } else {
        format!("{bytes} bytes")
    }
}

/// Lay the host plan out, or refuse the load.
pub fn plan_host(request: &HostPlanRequest) -> Result<HostPlan, HostPlanError> {
    let lines = host_lines(request);
    let total = lines.iter().fold(0u64, |sum, (_, b)| sum.saturating_add(*b));
    let usable = request
        .available_physical_bytes
        .saturating_sub(request.margin_bytes);
    if total > usable {
        let mut running = 0u64;
        let crossing_line = lines
            .iter()
            .find(|(_, bytes)| {
                running = running.saturating_add(*bytes);
                running > usable
            })
            .map_or(lines[0].0, |(name, _)| *name);
        return Err(HostPlanError::BelowMargin {
            available_bytes: request.available_physical_bytes,
            planned_bytes: total,
            margin_bytes: request.margin_bytes,
            crossing_line,
            lines,
        });
    }
    Ok(HostPlan {
        request: *request,
        total_bytes: total,
        left_bytes: request.available_physical_bytes - total,
    })
}

/// What the host plan leaves its n-gram hot-row line (`--ngram-hot-bytes
/// auto`, GitHub #306): the available memory past the margin and every
/// other line. `request.ngram_hot_rows_bytes` is not read, so a line no
/// larger than this is one [`plan_host`] accepts.
pub fn ngram_hot_rows_room(request: &HostPlanRequest) -> u64 {
    let others = host_lines(&HostPlanRequest { ngram_hot_rows_bytes: 0, ..*request })
        .iter()
        .fold(0u64, |sum, (_, bytes)| sum.saturating_add(*bytes));
    request
        .available_physical_bytes
        .saturating_sub(request.margin_bytes)
        .saturating_sub(others)
}

/// The sidecar's `expert_traffic`: calibration selections per (layer,
/// expert), layer-major, checked against the catalog it describes. Both the
/// pool split and the warm start read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTraffic {
    experts: u16,
    counts: Vec<u64>,
}

/// Traffic that does not cover the catalog's experts exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrafficMismatch {
    pub expected: usize,
    pub got: usize,
}

impl std::fmt::Display for TrafficMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the expert traffic counts {} experts, the model has {}",
            self.got, self.expected
        )
    }
}

impl std::error::Error for TrafficMismatch {}

impl ExpertTraffic {
    /// `counts` holds `layers × experts` entries of `catalog`, layer-major.
    pub fn new(catalog: &ExpertCatalog, counts: Vec<u64>) -> Result<Self, TrafficMismatch> {
        let expected = usize::from(catalog.layers()) * usize::from(catalog.experts());
        if counts.len() != expected {
            return Err(TrafficMismatch {
                expected,
                got: counts.len(),
            });
        }
        Ok(Self {
            experts: catalog.experts(),
            counts,
        })
    }

    /// `(layer, expert, selections)` for every expert, layer-major.
    fn per_expert(&self) -> impl Iterator<Item = (u16, u16, u64)> + '_ {
        let experts = usize::from(self.experts);
        self.counts
            .iter()
            .enumerate()
            .map(move |(i, &count)| ((i / experts) as u16, (i % experts) as u16, count))
    }
}

/// What the expert cache line is made from.
#[derive(Debug, Clone, Copy)]
pub struct ExpertCacheRequest<'a> {
    /// The VRAM plan's budget ([`crate::vram::VramPlan::budget_bytes`]).
    pub budget_bytes: u64,
    /// Everything the VRAM plan placed, the named KV pool included
    /// ([`crate::vram::VramPlan::total_bytes`]): non-expert weights, the
    /// CUDA context, workspaces, graphs, lane state, KV.
    pub planned_bytes: u64,
    /// [`prefill_staging_ring_bytes`].
    pub staging_ring_bytes: u64,
    /// [`residency_table_bytes`].
    pub table_bytes: u64,
    /// [`EXPERT_CACHE_FLOOR_BYTES`] for a real load.
    pub floor_bytes: u64,
    /// `--vram-allow-expert-cache-below-floor` (ADR 0045): a cache below the floor
    /// is planned anyway, flagged [`ExpertCachePlan::below_floor`] for the
    /// load's warning, instead of refused. The class minimum still refuses.
    pub allow_below_floor: bool,
    /// The artifact's experts: slot bytes and projections per class.
    pub catalog: &'a ExpertCatalog,
    /// The sidecar's calibration traffic over `catalog`'s experts.
    pub traffic: &'a ExpertTraffic,
    /// [`min_slots_per_class`]: what every populated class gets first.
    pub min_slots: u32,
}

/// One class's pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassPool {
    pub slots: u32,
    pub bytes: u64,
}

/// The expert cache line and its pools.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertCachePlan {
    /// The line: budget − planned − staging ring − tables.
    pub cache_bytes: u64,
    /// Indexed like [`KClass::ALL`]. Their sum is at most `cache_bytes`,
    /// short of it by less than one slot of some class.
    pub pools: [ClassPool; KClass::COUNT],
    pub staging_ring_bytes: u64,
    pub table_bytes: u64,
    /// The decode hit rate these pools are expected to reach from the
    /// calibration rates alone: each class one LRU over its slots, a
    /// projection resident with the Che probability of [`plan_expert_cache`].
    /// Printed beside the cache (spec user story 3) to show what a smaller
    /// cache costs. It sees no locality, which real routing has: on the
    /// study's routing at 21.5 GB it reads 73.8% where the replay measures
    /// 94.5%, so read it as a floor and compare plans by it, not tok/s.
    pub expected_hit_rate: f64,
    /// The cache is below the request's floor, planned only because the
    /// operator allowed it (`--vram-allow-expert-cache-below-floor`).
    pub below_floor: bool,
}

impl std::fmt::Display for ExpertCachePlan {
    /// The plan lines as the load prints them, the cache beside its expected
    /// hit rate.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "expert_cache {} bytes (expected hit rate {:.1}% from calibration rates alone, \
             before locality; slots",
            self.cache_bytes,
            self.expected_hit_rate * 100.0
        )?;
        for class in KClass::ALL {
            write!(f, " {} {}", class.as_str(), self.pools[class.index()].slots)?;
        }
        write!(
            f,
            "), prefill_staging {} bytes, residency_tables {} bytes",
            self.staging_ring_bytes, self.table_bytes
        )
    }
}

impl ExpertCachePlan {
    /// Slots per class: the policy model's
    /// [`PolicyConfig::capacity`](super::policy::PolicyConfig::capacity).
    pub fn capacity(&self) -> [u32; KClass::COUNT] {
        std::array::from_fn(|i| self.pools[i].slots)
    }

    /// What the pools hold.
    pub fn pooled_bytes(&self) -> u64 {
        self.pools.iter().map(|p| p.bytes).sum()
    }

    /// `(name, bytes)` of residency's VRAM lines, in plan order, beside the
    /// ones [`crate::vram::VramLines::entries`] prints.
    pub fn entries(&self) -> [(&'static str, u64); 3] {
        [
            ("expert_cache", self.cache_bytes),
            ("prefill_staging", self.staging_ring_bytes),
            ("residency_tables", self.table_bytes),
        ]
    }
}

/// A load the expert cache line refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertCachePlanError {
    /// What the plan leaves is below the floor.
    BelowFloor {
        cache_bytes: u64,
        floor_bytes: u64,
        budget_bytes: u64,
        planned_bytes: u64,
        staging_ring_bytes: u64,
        table_bytes: u64,
    },
    /// What the plan leaves cannot hold every class's minimum: one decode
    /// step's selection and lookahead.
    BelowClassMinimum {
        cache_bytes: u64,
        needed_bytes: u64,
        min_slots: u32,
    },
}

impl std::fmt::Display for ExpertCachePlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (cache_bytes, floor_bytes, budget_bytes, planned_bytes, staging_ring_bytes, table_bytes) =
            match *self {
                Self::BelowFloor {
                    cache_bytes,
                    floor_bytes,
                    budget_bytes,
                    planned_bytes,
                    staging_ring_bytes,
                    table_bytes,
                } => (cache_bytes, floor_bytes, budget_bytes, planned_bytes, staging_ring_bytes, table_bytes),
                Self::BelowClassMinimum {
                    cache_bytes,
                    needed_bytes,
                    min_slots,
                } => {
                    return write!(
                        f,
                        "the VRAM expert cache would get {cache_bytes} bytes, short of the \
                         {needed_bytes} its K-class pools need for {min_slots} slots each, one \
                         decode step's selection and lookahead"
                    );
                }
            };
        write!(
            f,
            "the VRAM expert cache would get {cache_bytes} bytes, below its {floor_bytes}-byte \
             floor: the {budget_bytes}-byte VRAM budget holds {planned_bytes} for the weights, \
             workspaces, graphs and KV pool, {staging_ring_bytes} for the prefill staging ring \
             and {table_bytes} for residency's tables; shrink the KV pool (a smaller \
             --vram-kv-pool-bytes, or a shorter --model-max-context at one lane), the prefill chunk, \
             --vram-headroom-bytes or --reuse-retained-device, or close what holds VRAM on the desktop; \
             --vram-allow-expert-cache-below-floor starts anyway, with decode slower than the floor's"
        )
    }
}

impl std::error::Error for ExpertCachePlanError {}

/// The expert cache line and its split, or a refusal below the floor.
///
/// Every class with projections first gets `min(min_slots, projections)`
/// slots. The rest is split the way **one LRU over all classes** would hold
/// it: each class gets the bytes such an LRU of that size is expected to
/// keep of it, from the calibration selection rates (the Che approximation:
/// a projection selected at rate λ is resident with probability
/// `1 − e^(−λT)`, with `T` set so the expected bytes fill the cache). Whole
/// slots are floored, and what flooring leaves goes one slot at a time to
/// the classes with the largest remainders that still fit.
///
/// Per-class pools are fixed at load, so this is the static split closest
/// to the single LRU the study simulated: splitting by raw byte traffic
/// starves the K = 2 classes, whose many cold experts an LRU keeps far more
/// of than their traffic suggests, and cost 1.5x the simulated three-lane
/// residency on the study's routing (`expert_residency_study_replay.rs`).
pub fn plan_expert_cache(
    request: &ExpertCacheRequest<'_>,
) -> Result<ExpertCachePlan, ExpertCachePlanError> {
    let cache = request
        .budget_bytes
        .saturating_sub(request.planned_bytes)
        .saturating_sub(request.staging_ring_bytes)
        .saturating_sub(request.table_bytes);
    let below_floor = cache < request.floor_bytes;
    if below_floor && !request.allow_below_floor {
        return Err(ExpertCachePlanError::BelowFloor {
            cache_bytes: cache,
            floor_bytes: request.floor_bytes,
            budget_bytes: request.budget_bytes,
            planned_bytes: request.planned_bytes,
            staging_ring_bytes: request.staging_ring_bytes,
            table_bytes: request.table_bytes,
        });
    }

    let catalog = request.catalog;
    let slot_bytes: [u64; KClass::COUNT] = std::array::from_fn(|i| catalog.slot_bytes(KClass::ALL[i]));
    let projections = catalog.class_counts();
    let mut slots: [u64; KClass::COUNT] =
        std::array::from_fn(|i| u64::from(request.min_slots).min(projections[i]));
    let needed: u64 = (0..KClass::COUNT).map(|i| slots[i] * slot_bytes[i]).sum();
    if needed > cache {
        return Err(ExpertCachePlanError::BelowClassMinimum {
            cache_bytes: cache,
            needed_bytes: needed,
            min_slots: request.min_slots,
        });
    }
    let mut left = cache - needed;

    let target = expected_lru_bytes(catalog, request.traffic, left);
    let mut remainder = [0f64; KClass::COUNT];
    for i in 0..KClass::COUNT {
        if slot_bytes[i] == 0 {
            continue;
        }
        let exact = target[i] / slot_bytes[i] as f64;
        let more = (exact.floor() as u64).min(projections[i] - slots[i]);
        slots[i] += more;
        left = left.saturating_sub(more * slot_bytes[i]);
        remainder[i] = exact - more as f64;
    }
    let mut order: Vec<usize> = (0..KClass::COUNT).collect();
    order.sort_by(|&a, &b| remainder[b].total_cmp(&remainder[a]).then(a.cmp(&b)));
    for i in order {
        if slots[i] < projections[i] && slot_bytes[i] > 0 && slot_bytes[i] <= left {
            slots[i] += 1;
            left -= slot_bytes[i];
        }
    }

    let pools = std::array::from_fn(|i| ClassPool {
        slots: slots[i] as u32,
        bytes: slots[i] * slot_bytes[i],
    });
    Ok(ExpertCachePlan {
        cache_bytes: cache,
        pools,
        staging_ring_bytes: request.staging_ring_bytes,
        table_bytes: request.table_bytes,
        expected_hit_rate: expected_hit_rate(catalog, request.traffic, &slots),
        below_floor,
    })
}

/// Every selected projection's `(class, bytes, selections)`, and per class
/// the bytes of the never-selected ones.
fn selected_projections(
    catalog: &ExpertCatalog,
    traffic: &ExpertTraffic,
) -> (Vec<(usize, f64, f64)>, [f64; KClass::COUNT]) {
    let mut selected = Vec::new();
    let mut idle = [0f64; KClass::COUNT];
    for (layer, expert, count) in traffic.per_expert() {
        for projection in Projection::ALL {
            let id = ProjectionId::new(layer, expert, projection);
            let class = catalog.class_of(id).index();
            let bytes = catalog.bytes(id) as f64;
            if count > 0 {
                selected.push((class, bytes, count as f64));
            } else {
                idle[class] += bytes;
            }
        }
    }
    (selected, idle)
}

/// The characteristic time `T` at which `Σ weight · (1 − e^(−rate·T))`
/// reaches `target`, which must lie below the sum of the weights. Bisected
/// on a log scale.
fn che_time(items: &[(f64, f64)], target: f64) -> f64 {
    let held = |t: f64| -> f64 { items.iter().map(|(w, r)| w * -(-r * t).exp_m1()).sum() };
    let (mut lo, mut hi) = (1e-9f64, 1.0f64);
    while held(hi) < target {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = (lo * hi).sqrt();
        if held(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The share of calibration selections that would hit pools of `slots`,
/// each class one LRU over its own slots (the Che approximation per class).
fn expected_hit_rate(catalog: &ExpertCatalog, traffic: &ExpertTraffic, slots: &[u64; KClass::COUNT]) -> f64 {
    let (selected, _) = selected_projections(catalog, traffic);
    let rate_sum: f64 = selected.iter().map(|(_, _, c)| c).sum();
    if rate_sum == 0.0 {
        return 0.0;
    }
    let mut hits = 0.0;
    for class in 0..KClass::COUNT {
        // Weight 1 per projection: the pool holds `slots` of them.
        let items: Vec<(f64, f64)> = selected
            .iter()
            .filter(|(c, _, _)| *c == class)
            .map(|(_, _, count)| (1.0, count / rate_sum))
            .collect();
        let capacity = slots[class] as f64;
        if capacity >= items.len() as f64 {
            hits += items.iter().map(|(_, r)| r).sum::<f64>();
        } else if capacity > 0.0 {
            let t = che_time(&items, capacity);
            hits += items.iter().map(|(_, r)| r * -(-r * t).exp_m1()).sum::<f64>();
        }
    }
    hits
}

/// Bytes per class that one LRU of `cache` bytes over every projection is
/// expected to hold (the Che approximation, see [`plan_expert_cache`]). A
/// cache larger than every selected projection holds them all, and its rest
/// is shared by the never-selected bytes of each class.
fn expected_lru_bytes(catalog: &ExpertCatalog, traffic: &ExpertTraffic, cache: u64) -> [f64; KClass::COUNT] {
    let (selected, idle) = selected_projections(catalog, traffic);
    let cache = cache as f64;
    let total: f64 = selected.iter().map(|(_, b, _)| b).sum();
    let mut out = [0f64; KClass::COUNT];
    if total <= cache {
        for (class, bytes, _) in &selected {
            out[*class] += bytes;
        }
        let idle_total: f64 = idle.iter().sum();
        if idle_total > 0.0 {
            for i in 0..KClass::COUNT {
                out[i] += (cache - total).min(idle_total) * idle[i] / idle_total;
            }
        }
        return out;
    }
    let rate_sum: f64 = selected.iter().map(|(_, _, c)| c).sum();
    let items: Vec<(f64, f64)> = selected.iter().map(|(_, b, c)| (*b, c / rate_sum)).collect();
    let t = che_time(&items, cache);
    for (class, bytes, count) in &selected {
        out[*class] += bytes * -(-(count / rate_sum) * t).exp_m1();
    }
    out
}

/// The slots a class needs so that no decode step can be refused and its
/// prefetches still find room: every lane's selection and every lane's
/// lookahead, as if all of them fell in that one class.
pub fn min_slots_per_class(lanes: u32, top_k: u32, prefetch_width: u32) -> u32 {
    lanes * (top_k + prefetch_width)
}

/// The decode prefetch budget that served Flash-Next best at one lane, at
/// the 262K x 3-lane default (GitHub #306,
/// `docs/findings/2026-10-06-flash-next-decode-round.md`): 0.67x the
/// 1.75 MB the earlier rule gave every step. Less prefetch is then still
/// copying at the next layer's join.
pub const MEASURED_ONE_ROW_PREFETCH_BUDGET_BYTES: u64 = 1_172_500;

/// The same at three lanes: 1.5x the 1.75 MB. Three lanes' misses saturate
/// the link, and a larger budget's prefetches displace demand bytes.
pub const MEASURED_THREE_ROW_PREFETCH_BUDGET_BYTES: u64 = 2_625_000;

/// The default decode prefetch budget: the line through the two measured
/// points (1,172,500 B at one row, 726,250 B more per further row), so it
/// follows the rows each step decodes — the lanes decoding now, or a verify
/// round's columns — not the lanes configured. No floor: at one row a
/// gate/up projection at K >= 3 (1.24-1.65 MB) is never prefetched, which
/// served faster than the old floor of one such projection.
pub fn default_prefetch_budget() -> PrefetchBudget {
    let per_row = (MEASURED_THREE_ROW_PREFETCH_BUDGET_BYTES - MEASURED_ONE_ROW_PREFETCH_BUDGET_BYTES) / 2;
    PrefetchBudget { one_row_bytes: MEASURED_ONE_ROW_PREFETCH_BUDGET_BYTES, per_row_bytes: per_row }
}

/// The prefill staging ring: two layers, each as large as the heaviest
/// layer's projections — the most one chunk can touch in a layer — so the
/// layer being computed and the layer being filled never overflow it.
pub fn prefill_staging_ring_bytes(catalog: &ExpertCatalog) -> u64 {
    let heaviest = (0..catalog.layers())
        .map(|layer| catalog.layer_bytes(layer))
        .max()
        .unwrap_or(0);
    2 * heaviest
}

/// Per projection, what the leaf's residency keeps (`kernel/src/residency.cu`):
/// its slot-table entry as the expert kernels read it (16 bytes), its state
/// (class, K, host offset, slot, flags: 15 bytes) and the LRU entry of the
/// slot it may hold (owner and stamp: 12 bytes), rounded up.
pub const TABLE_BYTES_PER_PROJECTION: u64 = 48;

/// Residency's device tables, an upper bound of the leaf's own line
/// (`ignis_residency_plan_bytes`'s `tables`, which a cuda load reserves): the
/// per-projection tables counted for as many slots as there are projections,
/// so it does not depend on the cache split it is subtracted before; two
/// copy-job lists and the ring's key lists of one layer each; the lookahead
/// ranking of a chunk; and the scalars and allocation rounding.
pub fn residency_table_bytes(layers: u64, experts: u64, max_tokens: u64, lookahead_width: u64) -> u64 {
    let keys = layers * experts * 2;
    keys * TABLE_BYTES_PER_PROJECTION + 2 * (2 * experts) * (24 + 4) + max_tokens * lookahead_width * 4 + 16 * 1024
}

/// The warm start's order: both projections of every selected expert,
/// hottest first by calibration selections, ties in canonical key order.
/// Never-selected experts are not warmed.
pub fn warm_start_order(traffic: &ExpertTraffic) -> Vec<ProjectionId> {
    let mut experts: Vec<(u64, u16, u16)> = traffic
        .per_expert()
        .filter(|&(_, _, count)| count > 0)
        .map(|(layer, expert, count)| (count, layer, expert))
        .collect();
    experts.sort_by(|a, b| b.0.cmp(&a.0).then((a.1, a.2).cmp(&(b.1, b.2))));
    experts
        .into_iter()
        .flat_map(|(_, layer, expert)| {
            Projection::ALL.map(|projection| ProjectionId::new(layer, expert, projection))
        })
        .collect()
}
