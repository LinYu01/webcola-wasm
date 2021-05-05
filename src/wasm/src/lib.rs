#![feature(box_syntax, core_intrinsics)]
#![allow(non_snake_case)]

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
    pub G: Option<Vec<f32>>,
    rng: Pcg32,
    min_d: f32,
    // snap_grid_size: f32,
    // snap_strength: f32,
    max_h: f32,
    /// Scratch buffer used during step size computation
    Hd: [Vec<f32>; DIMS],
}

impl<const DIMS: usize> Context<DIMS> {
    pub fn new(D: Vec<f32>, G: Option<Vec<f32>>, node_count: usize) -> Self {
        let mut g: Vec<f32> = Vec::with_capacity(DIMS * node_count);
        let mut H: Vec<f32> = Vec::with_capacity(DIMS * node_count * node_count);
        unsafe {
            H.set_len(DIMS * node_count * node_count);
            g.set_len(DIMS * node_count);
        };
        let mut Hd: [Vec<f32>; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        for i in 0..DIMS {
            unsafe {
                Hd.as_mut_ptr().add(i).write(vec![0.; node_count]);
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
        }
    }

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

    pub fn compute(&mut self, x: &mut [f32]) {
        // distance vector
        let mut d: [f32; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        // distance vector squared
        let mut d2: [f32; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        // Hessian diagonal
        let mut Huu: [f32; DIMS] = [0.; DIMS];
        let mut max_h: f32 = 0.;
        let n = self.n;

        // across all nodes u
        for u in 0..self.n {
            // zero gradient and hessian diagonals
            for i in 0..DIMS {
                if cfg!(debug_assertions) {
                    Huu[i] = 0.;
                    self.g[i * n + u] = 0.;
                } else {
                    unsafe {
                        *Huu.get_unchecked_mut(i) = 0.;
                        *self.g.get_unchecked_mut(i * n + u) = 0.;
                    }
                }
            }

            // across all nodes v
            for v in 0..self.n {
                if u == v {
                    continue;
                }

                // The following loop computes distance vector and
                // randomly displaces nodes that are at identical positions
                let max_displaces = self.n;
                let mut distance_squared = 0.0f32;
                for _ in 0..max_displaces {
                    distance_squared = 0.;
                    for i in 0..DIMS {
                        let dx = if cfg!(debug_assertions) {
                            x[i * n + u] - x[i * n + v]
                        } else {
                            unsafe { *x.get_unchecked(i * n + u) - *x.get_unchecked(i * n + v) }
                        };
                        let dx2 = dx * dx;
                        d[i] = dx;
                        d2[i] = dx2;
                        distance_squared += dx2;
                    }

                    if std::intrinsics::likely(distance_squared > 0.000000001) {
                        break;
                    }
                    let rd = self.offset_dir();

                    for i in 0..DIMS {
                        if cfg!(debug_assertions) {
                            x[i * n + v] += rd[i];
                        } else {
                            unsafe {
                                *x.get_unchecked_mut(i * n + v) += *rd.get_unchecked(i);
                            }
                        }
                    }
                }
                let distance = distance_squared.sqrt();
                let ideal_distance = if cfg!(debug_assertions) {
                    self.D[u * n + v]
                } else {
                    unsafe { *self.D.get_unchecked(u * n + v) }
                };
                // if ideal_distance == 0. {
                //     panic!("ideal_distance=0; self.D={:?}", self.D);
                // }
                // weights are passed via G matrix.
                // weight > 1 means not immediately connected
                // small weights (<<1) are used for group dummy nodes
                let mut weight = match self.G.as_ref() {
                    Some(G) => {
                        if cfg!(debug_assertions) {
                            G[u * n + v]
                        } else {
                            unsafe { *G.get_unchecked(u * n + v) }
                        }
                    }
                    None => 1.,
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
                        if !weight.is_finite() {
                            panic!("bad weight: {}", weight);
                        } else if !distance.is_finite() {
                            panic!();
                        } else if !ideal_distance.is_finite() {
                            panic!();
                        }

                        panic!();
                    }
                }

                for i in 0..DIMS {
                    self.g[i * n + u] += d[i] * gs;
                    let idk =
                        hs * (2. * distance_cubed + ideal_distance * (d2[i] - distance_squared));
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
        assert_eq!(D.len(), node_count * node_count);
        if !G.is_empty() {
            assert_eq!(G.len(), node_count * node_count);
        }

        let ctx: Context<2> =
            Context::new(D, if G.is_empty() { None } else { Some(G) }, node_count);
        Box::into_raw(box ctx) as _
    } else if dimensions == 3 {
        assert_eq!(D.len(), node_count * node_count);
        if !G.is_empty() {
            assert_eq!(G.len(), node_count * node_count);
        }

        let ctx: Context<3> =
            Context::new(D, if G.is_empty() { None } else { Some(G) }, node_count);
        Box::into_raw(box ctx) as _
    } else {
        unimplemented!();
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
    assert_eq!(new_G.len(), ctx.n * ctx.n);
    ctx.G = Some(new_G);
}

#[wasm_bindgen]
pub fn set_G_3d(ctx: *mut Context<3>, new_G: Vec<f32>) {
    let ctx = unsafe { &mut *ctx };
    ctx.G = Some(new_G);
}
