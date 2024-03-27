use pyo3::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{collections::HashSet, ops::{Add, Div, Mul, Neg, Sub}, sync::{Arc, RwLock}};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TensorId(usize);
impl TensorId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
struct RawTensor {
    id: TensorId,
    data: Vec<f32>, // TODO: multiple tensors can share the same data
    grad: Vec<f32>,
    shape: Vec<usize>,
    op: Option<String>,
    ctx: Option<f32>,
    children: Vec<Tensor>,
}

#[pyclass]
#[derive(Clone)]
pub struct Tensor(Arc<RwLock<RawTensor>>);

impl Tensor {
    fn new(data: Vec<f32>, shape: Vec<usize>, op: Option<String>) -> Self {
        let len = data.len();
        Tensor(Arc::new(RwLock::new(RawTensor {
            id: TensorId::new(),
            data,
            grad: vec![0.0; len],
            shape,
            op,
            ctx: None,
            children: vec![],
        })))
    }

    fn modify_grad_(&self) {
        if let Some(op) = &self.0.read().unwrap().op {
            if op == "broadcast" {
                fn fill_grad(idx: usize, grad: &Vec<f32>, shape: &Vec<usize>, chd_idx: usize, chd_grad: &mut Vec<f32>, chd_shape: &Vec<usize>) {
                    if idx == shape.len() { chd_grad[idx] = grad[idx]; return; }

                    for i in 0..shape[idx] {
                        let n_idx = idx * shape[idx] + i;
                        let n_chd_idx = chd_idx * chd_shape[idx] + i % chd_shape[idx];
                        fill_grad(n_idx, grad, shape, n_chd_idx, chd_grad, chd_shape);
                    }
                }

                let self_grad = self.0.read().unwrap().grad.clone();
                let self_shape = self.0.read().unwrap().shape.clone();
                let mut chd_grad = self.0.read().unwrap().children[0].0.read().unwrap().grad.clone();
                let chd_shape = self.0.read().unwrap().children[0].0.read().unwrap().shape.clone();
                
                fill_grad(0, &self_grad, &self_shape, 0, &mut chd_grad, &chd_shape);

                self.0.write().unwrap().children[0].0.write().unwrap().grad = chd_grad;
            }
            else if op == "pow" {
                let self_grad = self.0.read().unwrap().grad.clone();
                let exp = self.0.read().unwrap().ctx.unwrap();
                let chd_data = self.0.read().unwrap().children[0].0.read().unwrap().data.clone();
                let mut chd_grad = self.0.read().unwrap().children[0].0.read().unwrap().grad.clone();
                
                for (i, (g, x)) in self_grad.iter().zip(chd_data.iter()).enumerate() {
                    chd_grad[i] += exp * x.powf(exp - 1.0) * g;
                }

                self.0.write().unwrap().children[0].0.write().unwrap().grad = chd_grad;
            }
            else if op == "relu" {
                let self_grad = self.0.read().unwrap().grad.clone();
                let chd_data = self.0.read().unwrap().children[0].0.read().unwrap().data.clone();
                let mut chd_grad = self.0.read().unwrap().children[0].0.read().unwrap().grad.clone();

                for (i, (g, x)) in self_grad.iter().zip(chd_data.iter()).enumerate() {
                    chd_grad[i] += if *x > 0.0 { *g } else { 0.0 };
                }

                self.0.write().unwrap().children[0].0.write().unwrap().grad = chd_grad;
            }
            else if op == "neg" {
                let self_grad = self.0.read().unwrap().grad.clone();
                let mut chd_grad = self.0.read().unwrap().children[0].0.read().unwrap().grad.clone();

                for (i, g) in self_grad.iter().enumerate() {
                    chd_grad[i] -= g;
                }

                self.0.write().unwrap().children[0].0.write().unwrap().grad = chd_grad;
            }
            else if op == "add" {
                let self_grad = self.0.read().unwrap().grad.clone();
                let mut chd_grad_1 = self.0.read().unwrap().children[0].0.read().unwrap().grad.clone();
                let mut chd_grad_2 = self.0.read().unwrap().children[1].0.read().unwrap().grad.clone();
                
                for (i, g) in self_grad.iter().enumerate() {
                    chd_grad_1[i] += g;
                    chd_grad_2[i] += g;
                }

                self.0.write().unwrap().children[0].0.write().unwrap().grad = chd_grad_1;
                self.0.write().unwrap().children[1].0.write().unwrap().grad = chd_grad_2;
            }
            else if op == "mul" {
                let self_grad = self.0.read().unwrap().grad.clone();
                let chd_data_1 = self.0.read().unwrap().children[0].0.read().unwrap().data.clone();
                let chd_data_2 = self.0.read().unwrap().children[1].0.read().unwrap().data.clone();
                let mut chd_grad_1 = self.0.read().unwrap().children[0].0.read().unwrap().grad.clone();
                let mut chd_grad_2 = self.0.read().unwrap().children[1].0.read().unwrap().grad.clone();

                for (i, g) in self_grad.iter().enumerate() {
                    chd_grad_1[i] += chd_data_2[i] * g;
                    chd_grad_2[i] += chd_data_1[i] * g;
                }

                self.0.write().unwrap().children[0].0.write().unwrap().grad = chd_grad_1;
                self.0.write().unwrap().children[1].0.write().unwrap().grad = chd_grad_2;
            }
            // TODO: need rigorous analysis
            else if op == "matmul" {
                let self_grad = self.0.read().unwrap().grad.clone();
                let chd_data_1 = self.0.read().unwrap().children[0].0.read().unwrap().data.clone();
                let chd_data_2 = self.0.read().unwrap().children[1].0.read().unwrap().data.clone();
                let chd_shape_1 = self.0.read().unwrap().children[0].0.read().unwrap().shape.clone();
                let chd_shape_2 = self.0.read().unwrap().children[1].0.read().unwrap().shape.clone();
                let mut chd_grad_1 = self.0.read().unwrap().children[0].0.read().unwrap().grad.clone();
                let mut chd_grad_2 = self.0.read().unwrap().children[1].0.read().unwrap().grad.clone();

                for i in 0..chd_shape_1[0] {
                    for j in 0..chd_shape_2[1] {
                        for k in 0..chd_shape_1[1] {
                            chd_grad_1[i * chd_shape_1[1] + k] += chd_data_2[k * chd_shape_2[1] + j] * self_grad[i * chd_shape_2[1] + j];
                            chd_grad_2[k * chd_shape_2[1] + j] += chd_data_1[i * chd_shape_1[1] + k] * self_grad[i * chd_shape_2[1] + j];
                        }
                    }
                }

                self.0.write().unwrap().children[0].0.write().unwrap().grad = chd_grad_1;
                self.0.write().unwrap().children[1].0.write().unwrap().grad = chd_grad_2;
            }
            else { panic!("Unknown op: {}", op); }
        }
    }
    fn backward_(&self) {
        let mut topo: Vec<Tensor> = vec![];
        let mut visited: HashSet<TensorId> = HashSet::new();

        fn build_topo(v: Tensor, topo: &mut Vec<Tensor>, visited: &mut HashSet<TensorId>) {
            visited.insert(v.0.read().unwrap().id);
            for child in &v.0.read().unwrap().children {
                if !visited.contains(&child.0.read().unwrap().id) {
                    build_topo(child.clone(), topo, visited);
                }
            }
            topo.push(v);
        }
        build_topo(self.clone(), &mut topo, &mut visited);

        let init_grad: Vec<f32> = vec![1.0; self.0.read().unwrap().data.len()];
        self.0.write().unwrap().grad = init_grad;
        for v in topo.iter().rev() { v.modify_grad_(); }
    }

    fn uniform_(shape: Vec<usize>, low: f32, high: f32, seed: Option<u64>) -> Tensor {
        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let data = (0..shape.iter().product::<usize>()).map(|_| rng.gen_range(low..high)).collect();
        Tensor::new(data, shape, None)
    }

    // https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
    // TODO: need rigorous analysis
    fn kaiming_uniform_(shape: Vec<usize>, a: f32, seed: Option<u64>) -> Tensor {
        let bound = (3.0_f32).sqrt() * (2.0 / (1.0 + a.powf(2.0))) / (shape[1] as f32).sqrt();
        Tensor::uniform_(shape, -bound, bound, seed)
    }

    // https://numpy.org/doc/stable/user/basics.broadcasting.html
    // TODO: currently naive, slow, and memory inefficient implementation
    fn broadcast_(&self, other: Tensor) -> (Tensor, Tensor) {
        let mut self_shape = self.0.read().unwrap().shape.clone();
        let mut other_shape = other.0.read().unwrap().shape.clone();
        
        while self_shape.len() < other_shape.len() {
            self_shape.insert(0, 1);
        }
        while other_shape.len() < self_shape.len() {
            other_shape.insert(0, 1);
        }

        let mut out_shape: Vec<usize> = vec![];
        for (s, o) in self_shape.iter().zip(other_shape.iter()) {
            if s == o { out_shape.push(*s); }
            else if *s == 1 { out_shape.push(*o); }
            else if *o == 1 { out_shape.push(*s); }
            else { panic!("Tensor dimensions must match"); }
        }

        if out_shape == self_shape && out_shape == other_shape {
            return (self.clone(), other.clone());
        }

        let mut self_data_br: Vec<f32> = vec![0.0; out_shape.iter().product::<usize>() as usize];
        let mut other_data_br: Vec<f32> = vec![0.0; out_shape.iter().product::<usize>() as usize];

        fn fill_data(idx: usize, data: &Vec<f32>, shape: &Vec<usize>, out_idx: usize, out_data: &mut Vec<f32>, out_shape: &Vec<usize>) {
            if idx == shape.len() { out_data[out_idx] = data[idx]; return; }

            for i in 0..out_shape[idx] {
                let n_idx = idx * shape[idx] + (i % shape[idx]);
                let n_out_idx = out_idx * out_shape[idx] + i;
                fill_data(n_idx, data, shape, n_out_idx, out_data, out_shape);
            }
        }
        fill_data(0, &self.0.read().unwrap().data, &self_shape, 0, &mut self_data_br, &out_shape);
        fill_data(0, &other.0.read().unwrap().data, &other_shape, 0, &mut other_data_br, &out_shape);
        
        let self_br = Tensor::new(self_data_br, out_shape.clone(), Some("broadcast".to_string()));
        let other_br = Tensor::new(other_data_br, out_shape.clone(), Some("broadcast".to_string()));
        {
            self_br.0.write().unwrap().children.push(self.clone());
            other_br.0.write().unwrap().children.push(other.clone());
        }

        (self_br, other_br)
    }
}

// ******************** unary ops *************************

impl Tensor {
    fn pow_(&self, exp: f32) -> Tensor {
        let data: Vec<f32> = self.0.read().unwrap().data.iter().map(|x| x.powf(exp)).collect();
        let out = Tensor::new(data, self.0.read().unwrap().shape.clone(), Some("pow".to_string()));
        out.0.write().unwrap().ctx = Some(exp);
        {
            out.0.write().unwrap().children.push(self.clone());
        }
        out
    }

    fn relu_(&self) -> Tensor {
        let data: Vec<f32> = self.0.read().unwrap().data.iter().map(|x| x.max(0.0)).collect();
        let out = Tensor::new(data, self.0.read().unwrap().shape.clone(), Some("relu".to_string()));
        {
            out.0.write().unwrap().children.push(self.clone());
        }
        out
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let data: Vec<f32> = self.0.read().unwrap().data.iter().map(|x| -x).collect();
        let out = Tensor::new(data, self.0.read().unwrap().shape.clone(), Some("neg".to_string()));
        {
            out.0.write().unwrap().children.push(self.clone());
        }
        out
    }
}

// ******************** binary ops ************************

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Self::Output {
        let (self_br, other_br) = self.broadcast_(other);
        let data: Vec<f32> = self_br.0.read().unwrap().data.iter().zip(other_br.0.read().unwrap().data.iter()).map(|(x, y)| x + y).collect();
        let out = Tensor::new(data, self_br.0.read().unwrap().shape.clone(), Some("add".to_string()));
        {
            out.0.write().unwrap().children.push(self_br);
            out.0.write().unwrap().children.push(other_br);
        }
        out
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Self::Output { self + (-other) }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Self::Output {
        let (self_br, other_br) = self.broadcast_(other);
        let data: Vec<f32> = self_br.0.read().unwrap().data.iter().zip(other_br.0.read().unwrap().data.iter()).map(|(x, y)| x * y).collect();
        let out = Tensor::new(data, self_br.0.read().unwrap().shape.clone(), Some("mul".to_string()));
        {
            out.0.write().unwrap().children.push(self_br);
            out.0.write().unwrap().children.push(other_br);
        }
        out
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, other: Tensor) -> Self::Output { self * other.pow_(-1.0) }
}

use std::simd::f32x64;
const BLOCK: usize = 64;
const BLOCK_Y: usize = 16;
const BLOCK_X: usize = 16;

impl Tensor {
    // slowwww matmul, numpy is 5.6x faster
    // TODO: support nD tensors
    // https://pytorch.org/docs/stable/generated/torch.matmul.html
    fn matmul_(&self, other: Tensor) -> Tensor {
        let self_read = self.0.read().unwrap();
        let other_read = other.0.read().unwrap();

        let dim_0 = self_read.shape[0];
        let dim_1 = self_read.shape[1];
        let dim_2 = other_read.shape[1];

        assert_eq!(self_read.shape[1], other_read.shape[0], "Tensor dimensions must match");

        let self_data = self_read.data.clone();
        let other_data = other_read.data.clone();
        let mut out_data: Vec<f32> = vec![0.0; dim_0 * dim_2];

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
                        let acc_vec = acc[iy * BLOCK_X + ix].to_array();
                        let st = (y + iy) * dim_2 + x + ix * BLOCK;
                        out_data.splice(st..st + BLOCK, acc_vec);
                    }
                }

            }
        }

        let out = Tensor::new(out_data, vec![dim_0, dim_2], Some("matmul".to_string()));
        {
            out.0.write().unwrap().children.push(self.clone());
            out.0.write().unwrap().children.push(other.clone());
        }
        out
    }
}

// ******************** functional nn ops *****************

impl Tensor {
    fn linear_(&self, weight: Tensor, bias: Option<Tensor>) -> Tensor {
        let mut out =  self.matmul_(weight);
        if let Some(bias) = bias { out = out + bias; }
        out
    }
}

// ******************** python wrapper ********************

// TODO: implement the conversion that can detect the wrong shape
#[derive(FromPyObject)]
enum List<T> { Vector(Vec<List<T>>), Item(T), }
#[pyfunction]
fn process_list(list: List<f32>) -> (Vec<f32>, Vec<usize>) {
    match list {
        List::Vector(v) => {
            let mut data: Vec<f32> = vec![];
            let mut shape: Vec<usize> = vec![];
            shape.push(v.len());
            for item in v {
                let (d, s) = process_list(item);
                data.extend(d);
                if shape.len() == 1 {
                    shape.extend(s);
                }
            }
            (data, shape)
        }
        List::Item(i) => (vec![i as f32], vec![]),
    }
}

#[pymethods]
impl Tensor {
    #[new]
    fn __init__(data: List<f32>) -> PyResult<Self> {
        let (data, shape) = process_list(data);
        Ok(Self::new(data, shape, None))
    }

    // TODO: we need better __repr__
    fn __repr__(&self) -> PyResult<String> { Ok(format!("Tensor(data={:?}, shape={:?})", self.0.read().unwrap().data, self.0.read().unwrap().shape)) }

    fn backward(&self) -> PyResult<()> { Ok(self.backward_()) }
    #[staticmethod]
    fn uniform(shape: Vec<usize>, low: f32, high: f32, seed: Option<u64>) -> PyResult<Tensor> {
        Ok(Tensor::uniform_(shape, low, high, seed))
    }
    #[staticmethod] 
    fn kaiming_uniform(shape: Vec<usize>, a: f32, seed: Option<u64>) -> PyResult<Tensor> {
        Ok(Tensor::kaiming_uniform_(shape, a, seed))
    }

    // unary ops
    fn pow(&self, exp: f32) -> PyResult<Tensor> { Ok(self.pow_(exp)) }
    fn relu(&self) -> PyResult<Tensor> { Ok(self.relu_()) }
    fn __neg__(&self) -> PyResult<Tensor> { Ok(-self.clone()) }

    // binary ops
    fn __add__(&self, other: Tensor) -> PyResult<Tensor> { Ok(self.clone() + other.clone()) }
    fn __sub__(&self, other: Tensor) -> PyResult<Tensor> { Ok(self.clone() - other.clone()) }
    fn __mul__(&self, other: Tensor) -> PyResult<Tensor> { Ok(self.clone() * other.clone()) }
    fn __truediv__(&self, other: Tensor) -> PyResult<Tensor> { Ok(self.clone() / other.clone()) }
    fn __matmul__(&self, other: Tensor) -> PyResult<Tensor> { Ok(self.matmul_(other)) }
    fn matmul(&self, other: Tensor) -> PyResult<Tensor> { Ok(self.matmul_(other)) }
    fn dot(&self, other: Tensor) -> PyResult<Tensor> { Ok(self.matmul_(other)) }
    
    // functional nn ops
    fn linear(&self, weight: Tensor, bias: Option<Tensor>) -> PyResult<Tensor> { Ok(self.linear_(weight, bias)) }
}