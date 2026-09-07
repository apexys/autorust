//! `schedule/memory.py` + `runtime/support/memory.py` port — TLSF suballocator
//! and the graph-level memory planner.

use std::collections::{HashMap, HashSet};

use crate::dtype::ConstVal;
use crate::uop::arg::Arg;
use crate::uop::ctx::UOpCtx;
use crate::uop::ops::Ops;
use crate::uop::sint::SVal;
use crate::uop::UopId;

// ---------- TLSFAllocator (runtime/support/memory.py:14-106) ----------

fn round_up(x: i64, align: i64) -> i64 {
    if align <= 0 {
        return x;
    }
    ((x + align - 1) / align) * align
}

fn bit_length(x: i64) -> u32 {
    if x <= 0 {
        0
    } else {
        (64 - x.leading_zeros()).max(1)
    }
}

/// Two-Level Segregated Fit allocator (pure logic port).
pub struct TlsfAllocator {
    size: i64,
    base: i64,
    block_size: i64,
    l2_cnt: u32,
    /// lv1 -> lv2 -> list of block starts
    storage: Vec<Vec<Vec<i64>>>,
    lv1_entries: Vec<i64>,
    /// start -> (size, next, prev, is_free)
    blocks: HashMap<i64, (i64, Option<i64>, Option<i64>, bool)>,
}

impl TlsfAllocator {
    pub fn new(size: i64, base: i64, block_size: i64, lv2_cnt: i64) -> Self {
        let l2_cnt = bit_length(lv2_cnt);
        let n = bit_length(size) as usize + 1;
        let mut a = TlsfAllocator {
            size,
            base,
            block_size,
            l2_cnt,
            storage: vec![vec![Vec::new(); 1 << l2_cnt]; n],
            lv1_entries: vec![0; n],
            blocks: HashMap::new(),
        };
        a.blocks.insert(0, (size, None, None, true));
        if size > 0 {
            a.insert_block(0, size, None);
        }
        a
    }

    fn lv1(&self, size: i64) -> usize {
        bit_length(size) as usize
    }

    fn lv2(&self, size: i64) -> usize {
        let bl = bit_length(size);
        let denom = 1i64 << bl.saturating_sub(self.l2_cnt).max(0);
        ((size - (1i64 << (bl - 1))) / denom) as usize
    }

    fn insert_block(&mut self, start: i64, size: i64, prev: Option<i64>) {
        let prev = prev.or_else(|| self.blocks.get(&start).and_then(|b| b.2));
        let (l1, l2) = (self.lv1(size), self.lv2(size));
        self.storage[l1][l2].push(start);
        self.lv1_entries[l1] += 1;
        self.blocks.insert(start, (size, Some(start + size), prev, true));
    }

    fn remove_block(&mut self, start: i64, size: i64, prev: Option<i64>) {
        let prev = prev.or_else(|| self.blocks.get(&start).and_then(|b| b.2));
        let (l1, l2) = (self.lv1(size), self.lv2(size));
        if let Some(pos) = self.storage[l1][l2].iter().position(|&s| s == start) {
            self.storage[l1][l2].remove(pos);
        }
        self.lv1_entries[l1] -= 1;
        self.blocks.insert(start, (size, Some(start + size), prev, false));
    }

    fn split_block(&mut self, start: i64, size: i64, new_size: i64) {
        let nxt = self.blocks[&start].1;
        assert!(self.blocks[&start].3, "TGERR: block must be free");
        self.remove_block(start, size, None);
        self.insert_block(start, new_size, None);
        self.insert_block(start + new_size, size - new_size, Some(start));
        if let Some(n) = nxt {
            if let Some(b) = self.blocks.get_mut(&n) {
                let (bsize, bnext, _, bfree) = *b;
                *b = (bsize, bnext, Some(start + new_size), bfree);
            }
        }
    }

    fn merge_right(&mut self, start: i64) {
        let (mut size, mut nxt, _, is_free) = self.blocks[&start];
        assert!(is_free, "TGERR: block must be free");
        while let Some(n) = nxt {
            let Some(&(bsize, bnext, _, bfree)) = self.blocks.get(&n).copied().as_ref() else { break };
            if !is_free {
                break;
            }
            self.remove_block(start, size, None);
            self.remove_block(n, bsize, None);
            size += bsize;
            self.insert_block(start, size, None);
            assert_eq!(self.blocks[&start].1, bnext, "TGERR: merge mismatch");
            self.blocks.remove(&n);
            nxt = bnext;
        }
        if let Some(n) = nxt {
            if let Some(b) = self.blocks.get_mut(&n) {
                let (bsize, bnext, _, bfree) = *b;
                *b = (bsize, bnext, Some(start), bfree);
            }
        }
    }

    fn merge_block(&mut self, start: i64) {
        let mut start = start;
        loop {
            let prev = self.blocks[&start].2;
            match prev {
                Some(x) if self.blocks.get(&x).map(|b| b.3) == Some(true) => start = x,
                _ => break,
            }
        }
        self.merge_right(start);
    }

    pub fn alloc(&mut self, req_size: i64, align: i64) -> i64 {
        let req_size = self.block_size.max(req_size);
        let mut size = self.block_size.max(req_size + align - 1);
        size = round_up(size, 1i64 << bit_length(size).saturating_sub(self.l2_cnt).max(0));
        let sz_l1 = self.lv1(size);
        let sz_l2 = self.lv2(size);
        for l1 in sz_l1..self.storage.len() {
            if self.lv1_entries[l1] == 0 {
                continue;
            }
            let l2_start = if l1 == self.lv1(size) { sz_l2 } else { 0 };
            for l2 in l2_start..(1usize << self.l2_cnt) {
                if self.storage[l1][l2].is_empty() {
                    continue;
                }
                let mut start = self.storage[l1][l2][0];
                let mut nsize = self.blocks[&start].0;
                assert!(nsize >= size, "TGERR: block must be larger");
                // split for alignment
                let new_start = round_up(start, align);
                if new_start != start {
                    self.split_block(start, nsize, new_start - start);
                    start = new_start;
                    nsize = self.blocks[&new_start].0;
                }
                if nsize > req_size {
                    self.split_block(start, nsize, req_size);
                }
                self.remove_block(start, req_size, None);
                return start + self.base;
            }
        }
        panic!("TGERR: Can't allocate {req_size} bytes");
    }

    pub fn free(&mut self, start: i64) {
        let s = start - self.base;
        let size = self.blocks[&s].0;
        self.insert_block(s, size, None);
        self.merge_block(s);
    }
}

// ---------- memory_plan_rewrite (schedule/memory.py:28) ----------

/// `_collect_bufs`.
fn collect_bufs(ctx: &UOpCtx, u: UopId, out: &mut Vec<UopId>) {
    match ctx.op_of(u) {
        Ops::Buffer => out.push(u),
        Ops::Mselect | Ops::Mstack => {
            for &s in ctx.srcs(u) {
                collect_bufs(ctx, s, out);
            }
        }
        _ => {}
    }
}

/// `_can_plan` — not held, device not DISK/TINYFS/CL/WEBGPU.
fn can_plan(ctx: &mut UOpCtx, b: UopId, held: &[UopId]) -> bool {
    if held.contains(&b) {
        return false;
    }
    match crate::schedule::indexing::device_of(ctx, b) {
        Some(crate::uop::arg::DeviceSpec::Single(d)) => !(d.starts_with("DISK") || d.starts_with("TINYFS") || d.starts_with("CL") || d.starts_with("WEBGPU")),
        Some(crate::uop::arg::DeviceSpec::Multi(v)) => v.iter().all(|d| !(d.starts_with("DISK") || d.starts_with("TINYFS") || d.starts_with("CL") || d.starts_with("WEBGPU"))),
        None => false,
    }
}

/// `memory_plan_rewrite` — suballocate internal buffers into per-lane arenas.
pub fn memory_plan_rewrite(ctx: &mut UOpCtx, linear: UopId, held_bufs: &[UopId]) -> UopId {
    let srcs = ctx.srcs(linear).to_vec();
    let mut first_appearance: HashMap<UopId, i64> = HashMap::new();
    let mut last_appearance: HashMap<UopId, i64> = HashMap::new();
    let mut order: Vec<UopId> = Vec::new();
    let mut copy_bufs: HashSet<UopId> = HashSet::new();
    for (i, &si) in srcs.iter().enumerate() {
        let call_srcs = ctx.srcs(si).to_vec();
        let mut si_bufs = Vec::new();
        for &s in call_srcs.get(1..).unwrap_or(&[]) {
            collect_bufs(ctx, s, &mut si_bufs);
        }
        si_bufs.retain(|&b| can_plan(ctx, b, held_bufs));
        for b in si_bufs {
            if !first_appearance.contains_key(&b) {
                first_appearance.insert(b, i as i64);
                order.push(b);
            }
            last_appearance.insert(b, i as i64);
            if ctx.op_of(call_srcs[0]) == Ops::Copy {
                copy_bufs.insert(b);
            }
        }
    }
    if first_appearance.is_empty() {
        return linear;
    }

    // separate copy and compute buffers into different lanes
    let lane_of = |ctx: &mut UOpCtx, b: UopId| -> (String, i64) {
        let is_copy = copy_bufs.contains(&b);
        let dev = match crate::schedule::indexing::device_of(ctx, b) {
            Some(crate::uop::arg::DeviceSpec::Single(s)) => s,
            Some(crate::uop::arg::DeviceSpec::Multi(v)) => v.into_iter().next().unwrap_or_default(),
            None => String::new(),
        };
        (dev, if is_copy { 1 } else { 0 })
    };
    let mut buf_hold: HashMap<UopId, i64> = HashMap::new();
    for &b in &order {
        if copy_bufs.contains(&b) {
            buf_hold.insert(b, last_appearance[&b] - first_appearance[&b] + 1);
        }
    }

    let block_size: i64 = 256;
    let mut nbytes: HashMap<UopId, i64> = HashMap::new();
    for &b in &order {
        let bytes = ctx.max_numel(b) as i64 * ctx.dt_of(b).itemsize() as i64;
        nbytes.insert(b, round_up(bytes, block_size));
    }
    // build sorted open/close events
    let mut events: Vec<(i64, bool, UopId)> = Vec::new();
    for &b in &order {
        events.push((first_appearance[&b], true, b));
        events.push((last_appearance[&b] + 1 + buf_hold.get(&b).copied().unwrap_or(0), false, b));
    }
    events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let total_memory: i64 = nbytes.values().sum::<i64>() * 2;

    let mut offsets: HashMap<UopId, i64> = HashMap::new();
    let mut allocators: HashMap<(String, i64), (i64, TlsfAllocator)> = HashMap::new();
    for (_, is_open, buf) in events {
        let key = lane_of(ctx, buf);
        let alloc = allocators
            .entry(key.clone())
            .or_insert_with(|| (0, TlsfAllocator::new(total_memory, 0, block_size, 32)));
        if is_open {
            let off = alloc.1.alloc(nbytes[&buf], 1);
            offsets.insert(buf, off);
            let peak = off + ctx.max_numel(buf) as i64 * ctx.dt_of(buf).itemsize() as i64;
            alloc.0 = alloc.0.max(peak);
        } else {
            alloc.1.free(offsets[&buf]);
        }
    }
    let mut arena_sizes: HashMap<(String, i64), i64> = HashMap::new();
    for (k, (peak, _)) in allocators {
        arena_sizes.insert(k, round_up(peak, block_size));
    }

    // build replace_map: each buffer becomes a SHRINK/BITCAST into a shared arena
    let mut arenas: HashMap<(String, i64), UopId> = HashMap::new();
    let mut replace_map: HashMap<UopId, UopId> = HashMap::new();
    for &b in &order {
        let key = lane_of(ctx, b);
        let sz = arena_sizes[&key];
        let arena = *arenas.entry(key).or_insert_with(|| {
            let dev_s = match crate::schedule::indexing::device_of(ctx, b) {
                Some(crate::uop::arg::DeviceSpec::Single(s)) => s,
                _ => String::new(),
            };
            crate::tensor::creation::new_buffer(ctx, &dev_s, SVal::I(sz as i128), crate::dtype::INT8, None)
        });
        let off = offsets[&b];
        let nbytes_b = ctx.max_numel(b) as i64 * ctx.dt_of(b).itemsize() as i64;
        let view = crate::tensor::movement::shrink(
            ctx,
            arena,
            &[(SVal::I(off as i128), SVal::I(nbytes_b as i128))],
        );
        let bc = ctx.bitcast(view, ctx.dt_of(b));
        replace_map.insert(b, bc);
    }

    crate::uop::upat::substitute(ctx, linear, replace_map, true, true)
}

const _: () = {
    // INT8 must exist
    let _ = crate::dtype::INT8;
};

#[allow(dead_code)]
fn unused_const_val(_: ConstVal, _: Arg) {}
