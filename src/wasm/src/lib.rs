#![feature(box_syntax, core_intrinsics, wasm_simd)]
#![allow(non_snake_case)]

use core::arch::wasm32::*;
use rand::prelude::*;
use rand_pcg::Pcg32;
use wasm_bindgen::prelude::*;

pub struct Context<const DIMS: usize> {
    /// Number of nodes
    pub n: usize,
    /// Gradient vector
    pub g: Vec<f32>,
    /// Hessian matrix
    pub H: Vec<f32>,
    /// matrix of desired distances between pairs of nodes
    pub D: Vec<f32>,
    pub G: Vec<f32>,
    rng: Pcg32,
    min_d: f32,
    // snap_grid_size: f32,
    // snap_strength: f32,
    max_h: f32,
    /// Scratch buffer used during step size computation
    Hd: [Vec<f32>; DIMS],
    /// Scratch buffer used to hold distances between all nodes
    distances: Vec<f32>,
    /// Holds square root of the sums of all distances squared for all dimensions
    summed_distances: Vec<f32>,
}

impl<const DIMS: usize> Context<DIMS> {
    pub fn new(D: Vec<f32>, G: Vec<f32>, node_count: usize) -> Self {
        let mut g: Vec<f32> = Vec::with_capacity(DIMS * node_count);
        let mut H: Vec<f32> = Vec::with_capacity(DIMS * node_count * node_count);
        let mut distances: Vec<f32> = Vec::with_capacity(DIMS * node_count * node_count);

        unsafe {
            g.set_len(DIMS * node_count);
            H.set_len(DIMS * node_count * node_count);
            distances.set_len(DIMS * node_count * node_count);
        };
        let mut Hd: [Vec<f32>; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        for i in 0..DIMS {
            unsafe {
                Hd.as_mut_ptr().add(i).write(vec![0.; node_count]);
            }
        }

        let mut summed_distances: Vec<f32> = vec![0.; node_count * node_count];
        // This is an optimization to facilitate faster displacment checking.  We set values where u=v to non-zero values so that they aren't treated as needing displacement.
        for u in 0..node_count {
            for v in 0..node_count {
                if u != v {
                    continue;
                }

                unsafe {
                    *summed_distances.get_unchecked_mut(u * node_count + v) = 10000.;
                }
            }
        }

        let mut min_d = D
            .iter()
            .copied()
            .filter(|&x| !x.is_nan() && !x.is_infinite() && x > 0.)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f32::MAX);
        if min_d == f32::MAX {
            min_d = 1.;
        }

        Context {
            n: node_count,
            g,
            G,
            H,
            D,
            rng: Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7),
            min_d,
            // snap_grid_size: 100.,
            // snap_strength: 1000.,
            max_h: 0.,
            Hd,
            distances,
            summed_distances,
        }
    }

    // #[inline(never)]
    pub fn offset_dir(&mut self) -> [f32; DIMS] {
        let mut u: [f32; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut l = 0.;
        for i in 0..DIMS {
            let x: f32 = self.rng.gen_range(0.01, 1.) - 0.5;
            u[i] = x;
            l += x * x;
        }
        l = l.sqrt();
        for x in &mut u {
            *x *= self.min_d / l;
        }
        u
    }

    fn get_H_mut(&mut self, i: usize, u: usize, v: usize) -> &mut f32 {
        let n = self.n;
        let dim_size = self.n * n;
        if cfg!(debug_assertions) {
            &mut self.H[dim_size * i + (u * n) + v]
        } else {
            unsafe { self.H.get_unchecked_mut(dim_size * i + (u * n) + v) }
        }
    }

    fn set_H(&mut self, i: usize, u: usize, v: usize, val: f32) {
        let n = self.n;
        let dim_size = self.n * n;
        if cfg!(debug_assertions) {
            self.H[dim_size * i + (u * n) + v] = val;
        } else {
            unsafe {
                *self.H.get_unchecked_mut(dim_size * i + (u * n) + v) = val;
            }
        }
    }

    /// Returns `true` if any displacements were applied
    fn apply_displacements(&mut self, x: &mut [f32]) -> bool {
        let n = self.n;
        let mut did_apply = false;

        for u in 0..n {
            for v in 0..n {
                if u == v {
                    continue;
                }

                let summed_distance_squared = if cfg!(debug_assertions) {
                    self.summed_distances[u * n + v]
                } else {
                    unsafe { *self.summed_distances.get_unchecked(u * n + v) }
                };
                if summed_distance_squared > 0.000000001 {
                    continue;
                }

                did_apply = true;

                let rd = self.offset_dir();

                for i in 0..DIMS {
                    if cfg!(debug_assertions) {
                        x[i * n + v] += rd[i];
                    } else {
                        unsafe {
                            *x.get_unchecked_mut(i * n + v) += rd[i];
                        }
                    }
                }
            }
        }

        did_apply
    }

    // #[inline(never)]
    fn compute_distances(&mut self, x: &mut [f32]) -> bool {
        let n = self.n;

        let chunk_count = (n - (n % 4)) / 4;

        // This is a set of flags to facilitate efficient SIMD.  If any of the contained elements are non-zero, then displacements are needed
        let mut needs_displace = false;
        let mut needs_to_apply_displacements = unsafe { f32x4_splat(0.) };
        let displacement_threshold = unsafe { f32x4_splat(0.000000001) };

        for i in 0..DIMS {
            for u in 0..n {
                unsafe {
                    let u_vector =
                        v32x4_load_splat(x.get_unchecked(i * n + u) as *const f32 as *const u32);

                    for v_chunk_ix in 0..chunk_count {
                        let v_vector =
                            v128_load(x.as_ptr().add(i * n + v_chunk_ix * 4) as *const v128);

                        let distances = f32x4_sub(u_vector, v_vector);
                        let out_ix = (i * n * n) + (u * n) + v_chunk_ix * 4;

                        let distances_out_ptr =
                            self.distances.as_mut_ptr().add(out_ix) as *mut v128;
                        v128_store(distances_out_ptr, distances);

                        let distances_squared = f32x4_mul(distances, distances);

                        let summed_distances_squared_ptr = self
                            .summed_distances
                            .as_mut_ptr()
                            .add(u * n + v_chunk_ix * 4)
                            as *mut v128;
                        let summed_distances_squared_v = v128_load(summed_distances_squared_ptr);
                        let summed_distances_squared_v =
                            f32x4_add(distances_squared, summed_distances_squared_v);

                        // sqrt it on the last iteration
                        if i == DIMS - 1 {
                            let sqrted = f32x4_sqrt(summed_distances_squared_v);
                            v128_store(summed_distances_squared_ptr, sqrted);

                            // check here if we need to apply displacements
                            let any_under_displacement_threshold =
                                f32x4_lt(sqrted, displacement_threshold);
                            needs_to_apply_displacements = f32x4_max(
                                needs_to_apply_displacements,
                                any_under_displacement_threshold,
                            );
                        } else {
                            v128_store(summed_distances_squared_ptr, summed_distances_squared_v);
                        }
                    }

                    // Multiply the last partial chunk manually
                    for v in (chunk_count * 4)..n {
                        let out_ix = (i * n * n) + (u * n) + v;

                        let distance = *x.get_unchecked(i * n + u) - *x.get_unchecked(i * n + v);
                        let distance_squared = distance * distance;
                        *self.distances.get_unchecked_mut(out_ix) = distance;
                        *self.summed_distances.get_unchecked_mut(u * n + v) += distance_squared;
                        if i == DIMS - 1 {
                            let sqrtd = self.summed_distances.get_unchecked_mut(u * n + v).sqrt();
                            *self.summed_distances.get_unchecked_mut(u * n + v) = sqrtd;

                            if sqrtd < 0.000000001 {
                                needs_displace = true;
                            }
                        }
                    }
                }
            }
        }

        unsafe {
            needs_displace
                || f32x4_extract_lane::<0>(needs_to_apply_displacements) != 0.
                || f32x4_extract_lane::<1>(needs_to_apply_displacements) != 0.
                || f32x4_extract_lane::<2>(needs_to_apply_displacements) != 0.
                || f32x4_extract_lane::<3>(needs_to_apply_displacements) != 0.
        }
    }

    pub fn compute(&mut self, x: &mut [f32]) {
        // Pre-compute distances and distances squared between all nodes
        loop {
            let needs_displacement = self.compute_distances(x);
            if !needs_displacement {
                break;
            }

            let did_apply = self.apply_displacements(x);
            if !did_apply {
                break;
            }
        }

        let mut max_h: f32 = 0.;
        let n = self.n;

        self.g.fill(0.);

        // across all nodes u
        for u in 0..self.n {
            // Hessian diagonal
            let mut Huu: [f32; DIMS] = [0.; DIMS];

            // across all nodes v
            for v in 0..self.n {
                if u == v {
                    continue;
                }

                let distance = if cfg!(debug_assertions) {
                    let val = self.summed_distances[u * n + v];
                    self.summed_distances[u * n + v] = 0.;
                    val
                } else {
                    unsafe {
                        let ptr = self.summed_distances.get_unchecked_mut(u * n + v);
                        let val = *ptr;
                        *ptr = 0.;
                        val
                    }
                };

                let distance_squared = distance * distance;
                let ideal_distance = if cfg!(debug_assertions) {
                    self.D[u * n + v]
                } else {
                    unsafe { *self.D.get_unchecked(u * n + v) }
                };

                if cfg!(debug_assertions) {
                    if ideal_distance == 0. {
                        panic!("ideal_distance={}; u={}, v={}", ideal_distance, u, v);
                    }
                }

                // weights are passed via G matrix.
                // weight > 1 means not immediately connected
                // small weights (<<1) are used for group dummy nodes
                let mut weight = if cfg!(debug_assertions) {
                    self.G[u * n + v]
                } else {
                    unsafe { *self.G.get_unchecked(u * n + v) }
                };

                // ignore long range attractions for nodes not immediately connected (P-stress)
                if weight > 1. && distance > ideal_distance || !ideal_distance.is_finite() {
                    for i in 0..DIMS {
                        self.set_H(i, u, v, 0.);
                    }
                    continue;
                }

                // weight > 1 was just an indicator - this is an arcane interface,
                // but we are trying to be economical storing and passing node pair info
                if weight > 1. {
                    weight = 1.;
                }

                let ideal_distance_squared = ideal_distance * ideal_distance;
                let gs =
                    2. * weight * (distance - ideal_distance) / (ideal_distance_squared * distance);
                let distance_cubed = distance_squared * distance;
                let hs = 2. * -weight / (ideal_distance_squared * distance_cubed);
                if cfg!(debug_assertions) {
                    if !gs.is_finite() {
                        if !distance_squared.is_normal() {
                            panic!(
                                "bad distance squared: {}, u={}, v={}",
                                distance_squared, u, v
                            );
                        } else if !weight.is_finite() {
                            panic!("bad weight: {}", weight);
                        } else if !distance.is_normal() {
                            panic!();
                        } else if !ideal_distance.is_normal() {
                            panic!();
                        }

                        panic!();
                    }
                }

                for i in 0..DIMS {
                    let distance = if cfg!(debug_assertions) {
                        self.distances[(i * n * n) + u * n + v]
                    } else {
                        unsafe { *self.distances.get_unchecked((i * n * n) + u * n + v) }
                    };

                    self.g[i * n + u] += distance * gs;
                    let idk = hs
                        * (2. * distance_cubed
                            + ideal_distance * (distance * distance - distance_squared));
                    self.set_H(i, u, v, idk);
                    Huu[i] -= idk;
                }
            }
            for i in 0..DIMS {
                self.set_H(i, u, u, Huu[i]);
                max_h = max_h.max(Huu[i]);
            }
        }

        // Grid snap forces: TODO

        // used by locks
        self.max_h = max_h;
    }

    pub fn apply_lock(&mut self, u: usize, p: [f32; DIMS], x_i_u: [f32; DIMS]) {
        let n = self.n;
        for i in 0..DIMS {
            *self.get_H_mut(i, u, u) += self.max_h;
            if cfg!(debug_assertions) {
                self.g[i * n + u] -= self.max_h * (p[i] - x_i_u[i]);
            } else {
                unsafe {
                    *self.g.get_unchecked_mut(i * n + u) -= self.max_h * (p[i] - x_i_u[i]);
                }
            }
        }
    }

    fn dot_prod(a: &[f32], b: &[f32]) -> f32 {
        let mut x = 0.;
        debug_assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            x += unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) };
        }
        x
    }

    /// result r = matrix m * vector v
    fn right_multiply<'a>(m: impl Iterator<Item = &'a [f32]>, v: &[f32], r: &mut [f32]) {
        for (i, mn) in m.enumerate() {
            if cfg!(debug_assertions) {
                r[i] = Self::dot_prod(mn, v);
            } else {
                unsafe { *r.get_unchecked_mut(i) = Self::dot_prod(mn, v) };
            }
        }
    }

    /// Computes the optimal step size to take in direction d using the derivative information in this.g and this.H
    /// returns the scalar multiplier to apply to d to get the optimal step
    ///
    /// Only computes step size for `this.g`; computing step size with `this.e` is not implemented.
    pub fn compute_step_size(&mut self) -> f32 {
        let mut numerator = 0.;
        let mut denominator = 0.;
        let H_dim_size = self.n * self.n;
        let n = self.n;

        for (i, gn) in self.g.chunks_exact(n).enumerate() {
            numerator += Self::dot_prod(gn, gn);
            Self::right_multiply(
                self.H[(i * H_dim_size)..(i * H_dim_size + H_dim_size)].chunks_exact(n),
                gn,
                &mut self.Hd[i],
            );
            denominator += Self::dot_prod(gn, &self.Hd[i]);
        }

        if denominator == 0. || !denominator.is_finite() {
            return 0.;
        }
        return 1. * numerator / denominator;
    }
}

#[wasm_bindgen]
pub fn create_derivative_computer_ctx(
    dimensions: usize,
    node_count: usize,
    D: Vec<f32>,
    G: Vec<f32>,
) -> usize {
    if cfg!(debug_assertions) {
        console_error_panic_hook::set_once();
    }

    if dimensions == 2 {
        debug_assert_eq!(D.len(), node_count * node_count);
        if !G.is_empty() {
            debug_assert_eq!(G.len(), node_count * node_count);
        }

        let ctx: Context<2> = Context::new(
            D,
            if G.is_empty() {
                vec![1.; node_count * node_count]
            } else {
                G
            },
            node_count,
        );
        Box::into_raw(box ctx) as _
    } else if dimensions == 3 {
        debug_assert_eq!(D.len(), node_count * node_count);
        if !G.is_empty() {
            debug_assert_eq!(G.len(), node_count * node_count);
        }

        let ctx: Context<3> = Context::new(
            D,
            if G.is_empty() {
                vec![1.; node_count * node_count]
            } else {
                G
            },
            node_count,
        );
        Box::into_raw(box ctx) as _
    } else {
        if cfg!(debug_assertions) {
            unimplemented!();
        } else {
            unsafe { std::intrinsics::unreachable() }
        }
    }
}

#[wasm_bindgen]
pub fn compute_2d(ctx_ptr: *mut Context<2>, mut x: Vec<f32>) -> Vec<f32> {
    let ctx: &mut Context<2> = unsafe { &mut *ctx_ptr };
    ctx.compute(&mut x);
    x
}

#[wasm_bindgen]
pub fn compute_3d(ctx_ptr: *mut Context<3>, mut x: Vec<f32>) -> Vec<f32> {
    let ctx: &mut Context<3> = unsafe { &mut *ctx_ptr };
    ctx.compute(&mut x);
    x
}

#[wasm_bindgen]
pub fn apply_lock_2d(
    ctx_ptr: *mut Context<2>,
    u: usize,
    p_0: f32,
    p_1: f32,
    x_0_u: f32,
    x_1_u: f32,
) {
    let ctx = unsafe { &mut *ctx_ptr };
    ctx.apply_lock(u, [p_0, p_1], [x_0_u, x_1_u]);
}

#[wasm_bindgen]
pub fn apply_lock_3d(
    ctx_ptr: *mut Context<3>,
    u: usize,
    p_0: f32,
    p_1: f32,
    p_2: f32,
    x_0_u: f32,
    x_1_u: f32,
    x_2_u: f32,
) {
    let ctx = unsafe { &mut *ctx_ptr };
    ctx.apply_lock(u, [p_0, p_1, p_2], [x_0_u, x_1_u, x_2_u]);
}

#[wasm_bindgen]
pub fn compute_step_size_2d(ctx_ptr: *mut Context<2>) -> f32 {
    let ctx = unsafe { &mut *ctx_ptr };
    ctx.compute_step_size()
}

#[wasm_bindgen]
pub fn compute_step_size_3d(ctx_ptr: *mut Context<3>) -> f32 {
    let ctx = unsafe { &mut *ctx_ptr };
    ctx.compute_step_size()
}

#[wasm_bindgen]
pub fn get_memory() -> JsValue {
    wasm_bindgen::memory()
}

////////////////////////////////////////////////////////////////

// D Getters
#[wasm_bindgen]
pub fn get_D_2d(ctx: *mut Context<2>) -> *mut f32 {
    let ctx = unsafe { &mut *ctx };
    ctx.D.as_mut_ptr()
}

#[wasm_bindgen]
pub fn get_D_3d(ctx: *mut Context<3>) -> *mut f32 {
    let ctx = unsafe { &mut *ctx };
    ctx.D.as_mut_ptr()
}

// g Getters
#[wasm_bindgen]
pub fn get_g_2d(ctx: *mut Context<2>) -> *mut f32 {
    unsafe { (*ctx).g.as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_g_3d(ctx: *mut Context<3>) -> *mut f32 {
    unsafe { (*ctx).g.as_mut_ptr() }
}

#[wasm_bindgen]
pub fn set_G_2d(ctx: *mut Context<2>, new_G: Vec<f32>) {
    let ctx = unsafe { &mut *ctx };
    // assert_eq!(new_G.len(), ctx.n * ctx.n);
    ctx.G = new_G;
}

#[wasm_bindgen]
pub fn set_G_3d(ctx: *mut Context<3>, new_G: Vec<f32>) {
    let ctx = unsafe { &mut *ctx };
    ctx.G = new_G;
}
