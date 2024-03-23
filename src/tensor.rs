use pyo3::prelude::*;
use ndarray::ArrayD;
use numpy::{IxDyn, PyReadonlyArrayDyn};
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, RwLock};

use std::simd::f32x64;
const BLOCK: usize = 64; // f32x64
const BLOCK_Y: usize = 8;
const BLOCK_X: usize = 16;

pub static COUNTER: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone)]
pub struct RawTensor {
    id: usize,
    data: ArrayD<f32>,
    grad: ArrayD<f32>,
    children: Vec<(TensorNode, ArrayD<f32>)>, // (child, weight) but do we need weight?
}

#[derive(Clone)]
pub struct TensorNode(pub Arc<RwLock<RawTensor>>);

impl TensorNode {
    pub fn new(data: ArrayD<f32>) -> Self {
        TensorNode(Arc::new(RwLock::new(RawTensor {
            id: COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            grad: ArrayD::zeros(data.raw_dim()),
            data,
            children: Vec::new(),
        })))
    }

    pub fn matmul(&self, other: &TensorNode) -> Self {
        let self_dim = self.0.read().unwrap().data.raw_dim();
        let other_dim = other.0.read().unwrap().data.raw_dim();
        let dim_0 = self_dim[0];
        let dim_1 = self_dim[1];
        let dim_2 = other_dim[1];

        assert_eq!(self_dim[1], other_dim[0], "Tensor dimensions must match");

        let self_data: Vec<f32> = self.0.read().unwrap().data.clone().into_raw_vec();
        let other_data: Vec<f32> = other.0.read().unwrap().data.clone().into_raw_vec();
        let mut out_data: Vec<f32> = vec![0.0; dim_0 * dim_2];

        // reference:
        // https://github.com/tinygrad/tinygrad/blob/master/extra/gemm/gemm.c
        for y in (0..dim_0).step_by(BLOCK_Y) {
            for x in (0..dim_2).step_by(BLOCK_X * BLOCK) {
                
                let mut acc = vec![f32x64::splat(0.0); BLOCK_Y * BLOCK_X];

                for k in 0..dim_1 {
                    for iy in 0..BLOCK_Y {
                        let ta = f32x64::splat(self_data[(y + iy) * dim_1 + k]);
                        for ix in 0..BLOCK_X {
                            let tb = f32x64::from_slice(&other_data[k * dim_2 + x + ix * BLOCK..k * dim_2 + x + (ix + 1) * BLOCK]);
                            acc[iy * BLOCK_X + ix] += ta * tb;
                        }
                    }
                }

                for iy in 0..BLOCK_Y {
                    for ix in 0..BLOCK_X {
                        let acc_vec = acc[iy * BLOCK_X + ix].as_array().to_vec();
                        let st = (y + iy) * dim_2 + x + ix * BLOCK;
                        out_data.splice(st..st + BLOCK, acc_vec);
                    }
                }

            }
        }

        TensorNode::new(ArrayD::from_shape_vec(IxDyn(&[dim_0, dim_2]), out_data).unwrap())
    }

    pub fn pow(&self, exp: f32) -> Self {
        let data = self.0.read().unwrap().data.clone();
        let out_data = data.mapv(|x| x.powf(exp));
        let out = TensorNode::new(out_data);
        {
            let mut out_inner = out.0.write().unwrap();
            out_inner.children.push((self.clone(), data.mapv(|x| exp * x.powf(exp - 1.0))));
        }
        out
    }

    pub fn backward(&self) {
        let mut topo: Vec<TensorNode> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();

        fn build_topo(v: TensorNode, topo: &mut Vec<TensorNode>, visited: &mut HashSet<usize>) {
            visited.insert(v.0.read().unwrap().id);
            for (child, _) in v.0.read().unwrap().children.iter() {
                if !visited.contains(&child.0.read().unwrap().id) {
                    build_topo(child.clone(), topo, visited);
                }
            }
            topo.push(v);
        }
        build_topo(self.clone(), &mut topo, &mut visited);
        
        let self_dim = self.0.read().unwrap().data.raw_dim();
        self.0.write().unwrap().grad = ArrayD::ones(self_dim);
        for v in topo.iter().rev() {
            let grad = v.0.read().unwrap().grad.clone();
            for (child, weight) in v.0.read().unwrap().children.iter() {
                let child_grad = child.0.read().unwrap().grad.clone();
                child.0.write().unwrap().grad = child_grad + grad.clone() * weight.clone();
            }
        }
    }
}

impl Add<TensorNode> for TensorNode {
    type Output = TensorNode;
    fn add(self, other: TensorNode) -> Self::Output {
        let self_data = self.0.read().unwrap().data.clone();
        let other_data = other.0.read().unwrap().data.clone();

        assert_eq!(
            self_data.raw_dim(),
            other_data.raw_dim(),
            "Tensor dimensions must match"
        );

        let out_data = &self_data + &other_data;
        let out = TensorNode::new(out_data);
        {
            let mut out_inner = out.0.write().unwrap();
            out_inner.children.push((self.clone(), ArrayD::ones(self_data.raw_dim())));
            out_inner.children.push((other.clone(), ArrayD::ones(other_data.raw_dim())));
        }
        out
    }
}

impl Sub<TensorNode> for TensorNode {
    type Output = TensorNode;
    fn sub(self, other: TensorNode) -> Self::Output { self + -other }
}

impl Mul<TensorNode> for TensorNode {
    type Output = TensorNode;
    fn mul(self, other: TensorNode) -> Self::Output {
        let self_data = self.0.read().unwrap().data.clone();
        let other_data = other.0.read().unwrap().data.clone();

        assert_eq!(
            self_data.raw_dim(),
            other_data.raw_dim(),
            "Tensor dimensions must match"
        );

        let out_data = &self_data * &other_data;
        let out = TensorNode::new(out_data);
        {
            let mut out_inner = out.0.write().unwrap();
            out_inner.children.push((self.clone(), other_data.clone()));
            out_inner.children.push((other.clone(), self_data.clone()));
        }
        out
    }
}

impl Div<TensorNode> for TensorNode {
    type Output = TensorNode;
    fn div(self, other: TensorNode) -> Self::Output { self * other.pow(-1.0) }
}

impl Neg for TensorNode {
    type Output = TensorNode;
    fn neg(self) -> Self::Output {
        let data = self.0.read().unwrap().data.clone();
        let out = TensorNode::new(-&data);
        {
            let mut out_inner = out.0.write().unwrap();
            out_inner.children.push((self.clone(), -ArrayD::ones(data.raw_dim())));
        }
        out
    }
}

#[pyclass]
pub struct Tensor { ts: TensorNode, }

#[pymethods]
impl Tensor {
    // TODO: initialize directly from data
    #[new]
    fn new(data: PyReadonlyArrayDyn<f32>) -> Self {
        let data_array = data.as_array().to_owned();
        Tensor { ts: TensorNode::new(data_array) }
    }

    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(Tensor { ts: self.ts.matmul(&other.ts) }) }
    fn matmul(&self, other: &Tensor) -> PyResult<Tensor> { Ok(Tensor { ts: self.ts.matmul(&other.ts) }) }
    
    fn backward(&self) -> PyResult<()> { self.ts.backward(); Ok(()) }

    fn __repr__(&self) -> PyResult<String> { Ok(format!("Tensor({:?})", self.ts.0.read().unwrap().data)) }

    fn __add__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(Tensor { ts: self.ts.clone() + other.ts.clone() }) }
    fn __sub__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(Tensor { ts: self.ts.clone() - other.ts.clone() }) }
    fn __mul__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(Tensor { ts: self.ts.clone() * other.ts.clone() }) }
    fn __truediv__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(Tensor { ts: self.ts.clone() / other.ts.clone() }) }
    
    fn __neg__(&self) -> PyResult<Tensor> { Ok(Tensor { ts: -self.ts.clone() }) }
}