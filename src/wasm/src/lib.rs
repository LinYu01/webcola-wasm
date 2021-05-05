#![feature(box_syntax)]
#![allow(non_snake_case)]

use rand::prelude::*;
use rand_pcg::Pcg32;
use wasm_bindgen::prelude::*;

pub struct Context<const DIMS: usize> {
    /// Number of nodes
    pub n: usize,
    /// Node positions
    pub x: [Vec<f32>; DIMS],
    /// Gradient vector
    pub g: [Vec<f32>; DIMS],
    /// Hessian matrix
    pub H: [Vec<Vec<f32>>; DIMS],
    /// matrix of desired distances between pairs of nodes
    pub D: Vec<Vec<f32>>,
    pub G: Option<Vec<Vec<f32>>>,
    rng: Pcg32,
    min_d: f32,
    // snap_grid_size: f32,
    // snap_strength: f32,
}

impl<const DIMS: usize> Context<DIMS> {
    pub fn new(x: [Vec<f32>; DIMS], D: Vec<Vec<f32>>, G: Option<Vec<Vec<f32>>>) -> Self {
        let node_count = x[0].len();
        let mut g: [Vec<f32>; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut H: [Vec<Vec<f32>>; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        for i in 0..DIMS {
            unsafe {
                g.as_mut_ptr().add(i).write(vec![0.; node_count]);
                H.as_mut_ptr()
                    .add(i)
                    .write(vec![vec![0.; node_count]; node_count]);
            }
        }

        let mut min_d = D
            .iter()
            .flat_map(|d| d.iter().copied())
            .filter(|x| !x.is_nan() && !x.is_infinite() && *x > 0.)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f32::MAX);
        if min_d == f32::MAX {
            min_d = 1.;
        }

        Context {
            n: node_count,
            x,
            g,
            G,
            H,
            D,
            rng: Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7),
            min_d,
            // snap_grid_size: 100.,
            // snap_strength: 1000.,
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

    pub fn compute(&mut self, x: &mut [&mut [f32]; DIMS]) {
        // distance vector
        let mut d: [f32; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        // distance vector squared
        let mut d2: [f32; DIMS] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        // Hessian diagonal
        let mut Huu: [f32; DIMS] = [0.; DIMS];
        let mut max_h: f32 = 0.;

        // across all nodes u
        for u in 0..self.n {
            // zero gradient and hessian diagonals
            for i in 0..DIMS {
                if cfg!(debug_assertions) {
                    Huu[i] = 0.;
                    self.g[i][u] = 0.;
                } else {
                    unsafe {
                        *Huu.get_unchecked_mut(i) = 0.;
                        *self.g.get_unchecked_mut(i).get_unchecked_mut(u) = 0.;
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
                            x[i][u] - x[i][v]
                        } else {
                            unsafe {
                                *x.get_unchecked(i).get_unchecked(u)
                                    - *x.get_unchecked(i).get_unchecked(v)
                            }
                        };
                        let dx2 = dx * dx;

                        if cfg!(debug_assertions) {
                            d[i] = dx;
                            d2[i] = dx2;
                        } else {
                            unsafe {
                                *d.get_unchecked_mut(i) = dx;
                                *d2.get_unchecked_mut(i) = dx2;
                            }
                        }

                        distance_squared += dx2;
                    }

                    if distance_squared > 0.000000001 {
                        break;
                    }
                    let rd = self.offset_dir();

                    for i in 0..DIMS {
                        x[i][v] += rd[i];
                    }
                }
                let distance = distance_squared.sqrt();
                let ideal_distance = if cfg!(debug_assertions) {
                    self.D[u][v]
                } else {
                    unsafe { *self.D.get_unchecked(u).get_unchecked(v) }
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
                            G[u][v]
                        } else {
                            unsafe { *G.get_unchecked(u).get_unchecked(v) }
                        }
                    }
                    None => 1.,
                };

                // ignore long range attractions for nodes not immediately connected (P-stress)
                if weight > 1. && distance > ideal_distance || !ideal_distance.is_finite() {
                    for i in 0..DIMS {
                        if cfg!(debug_assertions) {
                            self.H[i][u][v] = 0.;
                        } else {
                            unsafe {
                                *self
                                    .H
                                    .get_unchecked_mut(i)
                                    .get_unchecked_mut(u)
                                    .get_unchecked_mut(v) = 0.
                            }
                        }
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
                if !gs.is_finite() {
                    if cfg!(debug_assertions) {
                        if !weight.is_finite() {
                            panic!("bad weight: {}", weight);
                        } else if !distance.is_finite() {
                            panic!();
                        } else if !ideal_distance.is_finite() {
                            panic!();
                        }
                    }
                    panic!();
                }

                for i in 0..DIMS {
                    self.g[i][u] += d[i] * gs;
                    let idk =
                        hs * (2. * distance_cubed + ideal_distance * (d2[i] - distance_squared));
                    self.H[i][u][v] = idk;
                    Huu[i] -= idk;
                }
            }
            for i in 0..DIMS {
                self.H[i][u][u] = Huu[i];
                max_h = max_h.max(Huu[i]);
            }
        }

        // Grid snap forces: TODO

        // Locks: TODO
    }
}

#[wasm_bindgen]
pub fn create_derivative_computer_ctx(
    dimensions: usize,
    node_count: usize,
    x: Vec<f32>,
    D: Vec<f32>,
    G: Vec<f32>,
) -> usize {
    console_error_panic_hook::set_once();

    if dimensions == 2 {
        assert_eq!(x.len(), node_count * 2);
        assert_eq!(D.len(), node_count * node_count);
        if !G.is_empty() {
            assert_eq!(G.len(), node_count * node_count);
        }

        let (x1, x2) = x.split_at(node_count);

        let ctx: Context<2> = Context::new(
            [x1.to_owned(), x2.to_owned()],
            D.chunks_exact(node_count).map(Into::into).collect(),
            if G.is_empty() {
                None
            } else {
                let G = G.chunks_exact(node_count).map(Into::into).collect();
                Some(G)
            },
        );
        Box::into_raw(box ctx) as _
    } else if dimensions == 3 {
        assert_eq!(x.len(), node_count * 3);
        assert_eq!(D.len(), node_count * node_count);
        if !G.is_empty() {
            assert_eq!(G.len(), node_count * node_count);
        }

        let ctx: Context<3> = Context::new(
            [
                x[..node_count].to_owned(),
                x[node_count..node_count * 2].to_owned(),
                x[node_count * 2..node_count * 2].to_owned(),
            ],
            D.chunks_exact(node_count).map(Into::into).collect(),
            if G.is_empty() {
                None
            } else {
                let G = G.chunks_exact(node_count).map(Into::into).collect();
                Some(G)
            },
        );
        Box::into_raw(box ctx) as _
    } else {
        unimplemented!();
    }
}

fn unpack_x<'a>(x: &'a mut [f32], n: usize) -> Vec<&'a mut [f32]> {
    x.chunks_exact_mut(n).map(|x| x).collect()
}

fn pack_x(unpacked_x: &[&mut [f32]]) -> Vec<f32> {
    let mut out = Vec::new();
    for xn in unpacked_x {
        for &x in xn.iter() {
            out.push(x);
        }
    }
    out
}

#[wasm_bindgen]
pub fn compute_2d(ctx_ptr: *mut Context<2>, mut x: Vec<f32>) -> Vec<f32> {
    let ctx: &mut Context<2> = unsafe { &mut *ctx_ptr };
    let unpacked_x = if x.is_empty() {
        ctx.x.iter_mut().map(|xn| xn.as_mut_slice()).collect()
    } else {
        unpack_x(&mut x, ctx.n)
    };
    let mut unpacked_x = unpacked_x.into_iter();
    let unpacked_x_0 = unpacked_x.next().unwrap();
    let unpacked_x_1 = unpacked_x.next().unwrap();
    let ctx: &mut Context<2> = unsafe { &mut *ctx_ptr };
    ctx.compute(&mut [unpacked_x_0, unpacked_x_1]);
    pack_x(&[unpacked_x_0, unpacked_x_1])
}

#[wasm_bindgen]
pub fn compute_3d(ctx_ptr: *mut Context<3>, mut x: Vec<f32>) -> Vec<f32> {
    let ctx: &mut Context<3> = unsafe { &mut *ctx_ptr };
    let unpacked_x = if x.is_empty() {
        ctx.x.iter_mut().map(|xn| xn.as_mut_slice()).collect()
    } else {
        unpack_x(&mut x, ctx.n)
    };
    let mut unpacked_x = unpacked_x.into_iter();
    let unpacked_x_0 = unpacked_x.next().unwrap();
    let unpacked_x_1 = unpacked_x.next().unwrap();
    let unpacked_x_2 = unpacked_x.next().unwrap();
    let ctx: &mut Context<3> = unsafe { &mut *ctx_ptr };
    ctx.compute(&mut [unpacked_x_0, unpacked_x_1, unpacked_x_2]);
    pack_x(&[unpacked_x_0, unpacked_x_1, unpacked_x_2])
}

#[wasm_bindgen]
pub fn get_memory() -> JsValue {
    wasm_bindgen::memory()
}

////////////////////////////////////////////////////////////////

// D Getters
#[wasm_bindgen]
pub fn get_D_2d(ctx: *mut Context<2>) -> Vec<usize> {
    let ctx = unsafe { &mut *ctx };
    let dn = &mut ctx.D;
    dn.iter_mut().map(|dnn| dnn.as_mut_ptr() as usize).collect()
}

#[wasm_bindgen]
pub fn get_D_3d(ctx: *mut Context<3>) -> Vec<usize> {
    let ctx = unsafe { &mut *ctx };
    let dn = &mut ctx.D;
    dn.iter_mut().map(|dnn| dnn.as_mut_ptr() as usize).collect()
}

// g Getters
#[wasm_bindgen]
pub fn get_g_2d_0(ctx: *mut Context<2>) -> *mut f32 {
    unsafe { (*ctx).g[0].as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_g_2d_1(ctx: *mut Context<2>) -> *mut f32 {
    unsafe { (*ctx).g[1].as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_g_3d_0(ctx: *mut Context<3>) -> *mut f32 {
    unsafe { (*ctx).g[0].as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_g_3d_1(ctx: *mut Context<3>) -> *mut f32 {
    unsafe { (*ctx).g[1].as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_g_3d_2(ctx: *mut Context<3>) -> *mut f32 {
    unsafe { (*ctx).g[2].as_mut_ptr() }
}

#[wasm_bindgen]
pub fn set_g_2d(ctx: *mut Context<2>, new_g: Vec<f32>) {
    let ctx = unsafe { &mut *ctx };
    debug_assert_eq!(new_g.len(), ctx.n * ctx.n);
    ctx.G = Some(new_g.chunks_exact(ctx.n).map(Into::into).collect());
}

#[wasm_bindgen]
pub fn set_g_3d(ctx: *mut Context<3>, new_g: Vec<f32>) {
    let ctx = unsafe { &mut *ctx };
    ctx.G = Some(new_g.chunks_exact(ctx.n).map(Into::into).collect());
}

// H Getters
#[wasm_bindgen]
pub fn get_H_2d(ctx: *mut Context<2>, i: usize) -> Vec<usize> {
    let ctx = unsafe { &mut *ctx };
    let hn = &mut ctx.H[i];
    hn.iter_mut().map(|hnn| hnn.as_mut_ptr() as usize).collect()
}

#[wasm_bindgen]
pub fn get_H_3d(ctx: *mut Context<3>, i: usize) -> Vec<usize> {
    let ctx = unsafe { &mut *ctx };
    let hn = &mut ctx.H[i];
    hn.iter_mut().map(|hnn| hnn.as_mut_ptr() as usize).collect()
}

// x Getters
#[wasm_bindgen]
pub fn get_x_2d_0(ctx: *mut Context<2>) -> *mut f32 {
    unsafe { (*ctx).x[0].as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_x_2d_1(ctx: *mut Context<2>) -> *mut f32 {
    unsafe { (*ctx).x[1].as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_x_3d_0(ctx: *mut Context<3>) -> *mut f32 {
    unsafe { (*ctx).x[0].as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_x_3d_1(ctx: *mut Context<3>) -> *mut f32 {
    unsafe { (*ctx).x[1].as_mut_ptr() }
}

#[wasm_bindgen]
pub fn get_x_3d_2(ctx: *mut Context<3>) -> *mut f32 {
    unsafe { (*ctx).x[2].as_mut_ptr() }
}
