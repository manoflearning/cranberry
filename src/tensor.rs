use crate::Ops;
use crate::Storage;

use numpy::PyArray;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::ops::Deref;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, RwLock};

#[derive(Clone)]
struct Context {
    prev: Vec<Tensor>,
    op: Option<Ops>,
    op_ctx: Option<f32>, // curently only for pow
}

impl Context {
    fn new(prev: Vec<Tensor>, op: Option<Ops>, op_ctx: Option<f32>) -> Self {
        Self {
            prev,
            op,
            op_ctx,
        }
    }
    fn walk(&self, now: &Tensor) { 
        if let Some(op) = self.op {
            match op {
                Ops::Broadcast => now.0.storage.read().unwrap().broadcast_back(&mut self.prev[0].0.storage.write().unwrap(), now.0.shape.clone(), self.prev[0].0.shape.clone()),
                Ops::Add => self.prev.iter().for_each(|p| now.0.storage.read().unwrap().add_back(&mut p.0.storage.write().unwrap())),
                Ops::Mul => {
                    // to avoid the thread panic when prev[0] == prev[1]
                    let prev_1 = self.prev[1].0.storage.read().unwrap().clone();
                    now.0.storage.read().unwrap().mul_back(&mut self.prev[0].0.storage.write().unwrap(), &prev_1);
                    let prev_0 = self.prev[0].0.storage.read().unwrap().clone();
                    now.0.storage.read().unwrap().mul_back(&mut self.prev[1].0.storage.write().unwrap(), &prev_0);
                }
                Ops::Matmul => {
                    let dim_0 = self.prev[0].0.shape[0];
                    let dim_1 = self.prev[0].0.shape[1];
                    let dim_2 = self.prev[1].0.shape[1];

                    // to avoid the thread panic when prev[0] == prev[1]
                    let prev_1 = self.prev[1].0.storage.read().unwrap().clone();
                    now.0.storage.read().unwrap().matmul_2d_back_0(
                        &mut self.prev[0].0.storage.write().unwrap(), 
                        &prev_1, 
                        vec![dim_0, dim_1, dim_2]
                    );
                    let prev_0 = self.prev[0].0.storage.read().unwrap().clone();
                    now.0.storage.read().unwrap().matmul_2d_back_1(
                        &prev_0, 
                        &mut self.prev[1].0.storage.write().unwrap(), 
                        vec![dim_0, dim_1, dim_2]
                    );
                }
                Ops::Pow => now.0.storage.read().unwrap().pow_back(&mut self.prev[0].0.storage.write().unwrap(), self.op_ctx.unwrap()),
                Ops::Relu => now.0.storage.read().unwrap().relu_back(&mut self.prev[0].0.storage.write().unwrap()),
                Ops::Exp => now.0.storage.read().unwrap().exp_back(&mut self.prev[0].0.storage.write().unwrap()),
                Ops::Log => now.0.storage.read().unwrap().log_back(&mut self.prev[0].0.storage.write().unwrap()),
                Ops::Neg => now.0.storage.read().unwrap().neg_back(&mut self.prev[0].0.storage.write().unwrap()),
                Ops::Sum => now.0.storage.read().unwrap().sum_back(&mut self.prev[0].0.storage.write().unwrap()),
                Ops::Reshape => {}
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
    storage: Arc<RwLock<Storage>>,
    requires_grad: bool,
    ctx: Option<Context>,
    shape: Vec<usize>,
    // TODO: DType, Device
}

// ********************************************************
// ***************          tensor          ***************
// ********************************************************

// cannot modify tensor directly after it is created
// need Mutex or RwLock to be able to modify the tensor
// https://github.com/huggingface/candle/blob/main/candle-core/src/tensor.rs
#[pyclass]
#[derive(Clone)]
pub struct Tensor(Arc<RawTensor>);

impl Tensor {
    fn new(storage: Storage, shape: Vec<usize>, requires_grad: bool, ctx: Option<Context>) -> Self {
        Tensor(Arc::new(
            RawTensor {
                id: TensorId::new(),
                storage: Arc::new(RwLock::new(storage)),
                requires_grad,
                ctx,
                shape,
            }
        ))
    }

    // TODO: if seed exists, using seed
    fn uniform_(shape: Vec<usize>, low: f32, high: f32, _seed: Option<u64>) -> Tensor {
        let data: Vec<f32> = (0..shape.iter().product()).map(|_| rand::random::<f32>() * (high - low) + low).collect();
        Tensor::new(
            Storage::new(data), 
            shape,
            true,
            None
        )
    }
    // https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
    fn kaiming_uniform_(shape: Vec<usize>, a: f32, seed: Option<u64>) -> Tensor {
        let bound = (3.0 as f32).powf(0.5) 
            * (2.0 / (1.0 + a.powf(2.0))).powf(0.5) 
            / (shape.iter().product::<usize>() as f32 / shape[0] as f32).powf(0.5);
        Tensor::uniform_(shape, -bound, bound, seed)
    }

    pub fn zero_grad_(&self) { self.0.storage.write().unwrap().init_grad_to_zero(); }
    pub fn step_(&self, lr: f32) { self.0.storage.write().unwrap().step(lr); }

    // https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
    fn detach_(&self) -> Tensor {
        Tensor::new(
            self.0.storage.read().unwrap().clone(), 
            self.0.shape.clone(), 
            false, 
            None
        )
    }

    #[inline]
    fn len_(&self) -> usize { self.0.shape.iter().product() }
}

impl PartialEq for Tensor {
    // https://pytorch.org/docs/stable/generated/torch.equal.html
    fn eq(&self, other: &Self) -> bool {
        let mut out = true;
        out &= self.0.storage.read().unwrap().get_data() == other.0.storage.read().unwrap().get_data();
        out &= self.0.shape == other.0.shape;
        out
    }
}

// ********************************************************
// ***************      backward prop       ***************
// ********************************************************

impl Tensor {
    fn backward_(&self) {
        assert!(self.0.requires_grad, "cannot call backward on a tensor that doesn't require gradients");
        assert!(self.len_() == 1, "backward() only supports scalar outputs (1-element tensors) right now");
        assert!(self.0.shape == vec![], "backward() only supports scalar outputs (1-element tensors) right now");

        let mut topo: Vec<Tensor> = vec![];
        let mut visited: HashSet<TensorId> = HashSet::new();

        fn build_topo(v: &Tensor, topo: &mut Vec<Tensor>, visited: &mut HashSet<TensorId>) {
            visited.insert(v.0.id);
            if let Some(v_ctx) = &v.0.ctx {
                for u in v_ctx.prev.iter() {
                    if u.0.requires_grad && !visited.contains(&u.0.id) {
                        build_topo(u, topo, visited);
                    }
                }
            }
            topo.push(v.clone());
        }
        build_topo(self, &mut topo, &mut visited);

        self.0.storage.write().unwrap().init_grad_to_one();
        for v in topo.iter().rev() {
            if let Some(v_ctx) = &v.0.ctx { v_ctx.walk(v); }
        }
    }
}

// ********************************************************
// ***************     broadcasting ops     ***************
// ********************************************************

// TODO: handle broadcasting using storage, not a tensor
impl Tensor {
    #[inline]
    fn broadcast_each_(&self, shape: &Vec<usize>) -> Tensor {
        let out_storage = self.0.storage.read().unwrap().deref().broadcast(self.0.shape.clone(), shape.clone());

        Tensor::new(
            out_storage,
            shape.clone(), 
            self.0.requires_grad, 
        Some(Context::new(
            vec![self.clone()], 
            Some(Ops::Broadcast),
            None,
        )))
    }
    // https://numpy.org/doc/stable/user/basics.broadcasting.html
    fn broadcast_(&self, other: &Tensor) -> (Tensor, Tensor) {
        let mut s_shape = self.0.shape.clone();
        let mut o_shape = other.0.shape.clone();

        while s_shape.len() < o_shape.len() {
            s_shape.insert(0, 1);
        }
        while o_shape.len() < s_shape.len() {
            o_shape.insert(0, 1);
        }

        let mut out_shape: Vec<usize> = vec![];
        for (s, o) in s_shape.iter().zip(o_shape.iter()) {
            if s == o { out_shape.push(*s); }
            else if *s == 1 { out_shape.push(*o); }
            else if *o == 1 { out_shape.push(*s); }
            else { panic!("Tensor dimensions must match"); }
        }

        if out_shape == self.0.shape && out_shape == other.0.shape {
            return (self.clone(), other.clone());
        }
        else if out_shape == self.0.shape {
            return (self.clone(), other.broadcast_each_(&out_shape));
        }
        else if out_shape == other.0.shape {
            return (self.broadcast_each_(&out_shape), other.clone());
        }
        else {
            return (self.broadcast_each_(&out_shape), other.broadcast_each_(&out_shape));
        }
    }
}

// ********************************************************
// ***************        binary ops        ***************
// ********************************************************

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Self::Output {
        let (a, b) = self.broadcast_(&other);
        
        let c_storage = a.0.storage.read().unwrap().deref().add(b.0.storage.read().unwrap().deref());

        Tensor::new(
            c_storage, 
            a.0.shape.clone(), 
            a.0.requires_grad || b.0.requires_grad, 
            Some(Context::new(vec![a.clone(), b.clone()], Some(Ops::Add),None))
        )
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, other: Tensor) -> Self::Output { self + (-other) }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Self::Output {
        let (a, b) = self.broadcast_(&other);
        
        let c_storage = a.0.storage.read().unwrap().deref().mul(b.0.storage.read().unwrap().deref());

        Tensor::new(
            c_storage, 
            a.0.shape.clone(), 
            a.0.requires_grad || b.0.requires_grad, 
            Some(Context::new(vec![a.clone(), b.clone()], Some(Ops::Mul),None))
        )
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, other: Tensor) -> Self::Output { self * other.pow_(-1.0) }
}

impl Tensor {
    fn matmul_2d_(self, other: &Tensor) -> Tensor {
        let dim_0 = self.0.shape[0];
        let dim_1 = self.0.shape[1];
        let dim_2 = other.0.shape[1];

        let out_storage = self.0.storage.read().unwrap().deref().matmul_2d(
            &other.0.storage.read().unwrap().deref(), 
            vec![dim_0, dim_1, dim_2]
        );

        Tensor::new(
            out_storage, 
            vec![dim_0, dim_2], 
            self.0.requires_grad || other.0.requires_grad, 
            Some(Context::new(vec![self.clone(), other.clone()], Some(Ops::Matmul),None))
        )
    }
    // https://pytorch.org/docs/stable/generated/torch.matmul.html
    fn matmul_(self, other: &Tensor) -> Tensor {
        let self_shape = self.0.shape.clone();
        let other_shape = other.0.shape.clone();
        // if both tensors are 1-dimensional, the dot product (scalar) is returned
        if self_shape.len() == 1 && other_shape.len() == 1 {
            assert!(self_shape[0] == other_shape[0], "Tensor dimensions must match");
            let a = self.reshape_(vec![1, self_shape[0]]);
            let b = other.reshape_(vec![other_shape[0], 1]);
            return a.matmul_2d_(&b)
        }
        // if both arguments are 2-dimensional, the matrix-matrix product is returned
        else if self_shape.len() == 2 && other_shape.len() == 2 {
            assert!(self_shape[1] == other_shape[0], "Tensor dimensions must match");
            return self.matmul_2d_(other)
        }
        // if the first argument is 1-dimensional and the second argument is 2-dimensional,
        // a 1 is prepended to its dimension for the purpose of the matrix multiply
        // after the matrix multiply, the prepended dimension is removed
        else if self_shape.len() == 1 && other_shape.len() == 2 {
            assert!(self_shape[0] == other_shape[0], "Tensor dimensions must match");
            let a = self.reshape_(vec![1, self_shape[0]]);
            return a.matmul_2d_(other)
        }
        // If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
        else if self_shape.len() == 2 && other_shape.len() == 1 {
            assert!(self_shape[1] == other_shape[0], "Tensor dimensions must match");
            let b = other.reshape_(vec![other_shape[0], 1]);
            return self.matmul_2d_(&b)
        }
        // if both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned
        // if the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after
        // if the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after
        // the non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable)
        else if self_shape.len() >= 1 && other_shape.len() >= 1 && (self_shape.len() > 2 || other_shape.len() > 2){
            unimplemented!()
        }
        else { panic!("Tensor dimensions must match"); }
    }
}

// ********************************************************
// ***************         unary ops        ***************
// ********************************************************

impl Tensor {
    fn pow_(&self, exp: f32) -> Tensor {
        let out_storage = self.0.storage.read().unwrap().deref().pow(exp);
        Tensor::new(
            out_storage,
            self.0.shape.clone(), 
            self.0.requires_grad, 
            Some(Context::new(vec![self.clone()], Some(Ops::Pow),Some(exp)))
        )
    }
    fn relu_(&self) -> Tensor {
        let out_storage = self.0.storage.read().unwrap().deref().relu();
        Tensor::new(
            out_storage,
            self.0.shape.clone(), 
            self.0.requires_grad, 
            Some(Context::new(vec![self.clone()], Some(Ops::Relu),None))
        )
    }
    fn exp_(&self) -> Tensor {
        let out_storage = self.0.storage.read().unwrap().deref().exp();
        Tensor::new(
            out_storage,
            self.0.shape.clone(),
            self.0.requires_grad,
            Some(Context::new(vec![self.clone()], Some(Ops::Exp), None))
        )
    }
    fn sigmoid_(&self) -> Tensor {
        let one = Tensor::new(Storage::new(vec![1.0]), vec![], false, None);
        one.clone() / (one.clone() + (-self.clone()).exp_())
    }
    // https://pytorch.org/docs/stable/generated/torch.log.html
    fn log_(&self) -> Tensor {
        let out_storage = self.0.storage.read().unwrap().deref().log();
        Tensor::new(
            out_storage,
            self.0.shape.clone(),
            self.0.requires_grad,
            Some(Context::new(vec![self.clone()], Some(Ops::Log), None))
        )
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Self::Output {
        let out_storage = self.0.storage.read().unwrap().deref().neg();
        Tensor::new(
            out_storage,
            self.0.shape.clone(), 
            self.0.requires_grad, 
            Some(Context::new(vec![self.clone()], Some(Ops::Neg),None))
        )
    }
}

// ********************************************************
// ***************        reduce ops        ***************
// ********************************************************

impl Tensor {
    // https://pytorch.org/docs/stable/generated/torch.sum.html
    fn sum_(&self) -> Tensor {
        let out_storage = self.0.storage.read().unwrap().deref().sum();
        Tensor::new(
            out_storage,
            vec![],
            self.0.requires_grad, 
            Some(Context::new(vec![self.clone()], Some(Ops::Sum), None))
        )
    }
    // https://pytorch.org/docs/stable/generated/torch.mean.html
    fn mean_(&self) -> Tensor {
        let num = Tensor::new(
            Storage::new(vec![self.len_() as f32]),
            vec![],
            false,
            None,
        );
        self.sum_() / num
    }
}

// ********************************************************
// ***************       movement ops       ***************
// ********************************************************

impl Tensor {
    // https://pytorch.org/docs/stable/generated/torch.reshape.html
    // TODO: if shape contains -1, then it will be calculated automatically
    // TODO: what if reshape is called on a tensor that requires grad?
    fn reshape_(&self, n_shape: Vec<usize>) -> Tensor {
        assert!(self.len_() == n_shape.iter().product(), "cannot reshape tensor of size {} into shape {:?}", self.len_(), n_shape);

        Tensor {
            0: Arc::new(
                RawTensor {
                    id: TensorId::new(),
                    storage: self.0.storage.clone(),
                    requires_grad: self.0.requires_grad,
                    ctx: Some(Context::new(vec![self.clone()], Some(Ops::Reshape),None)),
                    shape: n_shape,
                }
            )
        }
    }

    // https://pytorch.org/docs/stable/generated/torch.transpose.html
    fn transpose_(&self, dim0: usize, dim1: usize) -> Tensor {
        unimplemented!();

        assert!(dim0 < self.0.shape.len() && dim1 < self.0.shape.len(), "dimension out of range");

        let mut n_shape = self.0.shape.clone();
        n_shape.swap(dim0, dim1);

        self.reshape_(n_shape)
    }
}

// ********************************************************
// ***************     functional nn ops    ***************
// ********************************************************

impl Tensor {
    fn linear_(&self, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        let mut out = self.clone().matmul_(&weight);
        if let Some(bias) = bias { out = out + bias.clone(); }
        out
    }
}

// ********************************************************
// ***************      python wrapper      ***************
// ********************************************************

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
                if shape.len() == 1 { shape.extend(s); }
            }
            (data, shape)
        }
        List::Item(i) => (vec![i as f32], vec![]),
    }
}

// TODO: need high readable print_data
fn print_data(v: &Tensor) -> String {
    fn formulate_string(data: &Vec<f32>, shape: &Vec<usize>, idx: usize, dep: usize) -> String {
        if dep == shape.len() { return format!("{:.4}", data[idx]); }
        let mut out = String::from("[");
        for i in 0..shape[dep] {
            out.push_str(&formulate_string(data, shape, idx * shape[dep] + i, dep + 1));
            if i < shape[dep] - 1 { out.push_str(", "); }
        }
        out.push_str("]");
        out
    }
    formulate_string(&v.0.storage.read().unwrap().get_data(), &v.0.shape, 0, 0)
}

#[pymethods]
impl Tensor {
    #[new]
    fn __init__(data: List<f32>, requires_grad: Option<bool>) -> PyResult<Self> {
        let (data, shape) = process_list(data);
        let requires_grad = requires_grad.unwrap_or(false);
        Ok(Tensor::new(Storage::new(data), shape, requires_grad, None))
    }

    #[staticmethod]
    fn uniform(shape: Vec<usize>, low: f32, high: f32, seed: Option<u64>) -> PyResult<Tensor> { Ok(Tensor::uniform_(shape, low, high, seed)) }
    #[staticmethod] 
    fn kaiming_uniform(shape: Vec<usize>, a: f32, seed: Option<u64>) -> PyResult<Tensor> { Ok(Tensor::kaiming_uniform_(shape, a, seed)) }
    
    fn zero_grad(&self) -> PyResult<()> { Ok(self.zero_grad_()) }
    fn step(&self, lr: f32) -> PyResult<()> { Ok(self.step_(lr)) }

    fn detach(&self) -> PyResult<Tensor> { Ok(self.detach_()) }

    fn __eq__(&self, other: &Tensor) -> PyResult<bool> { Ok(self == other) }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("tensor({}, requires_grad={:?})", print_data(self), self.0.requires_grad))
    }

    fn numpy(&self) -> PyResult<Py<PyArrayDyn<f32>>> {
        let shape = self.0.shape.clone();
        let data = self.0.storage.read().unwrap().get_data().clone();

        Python::with_gil(|py| {
            let array = PyArray::from_vec(py, data).reshape(shape)?.to_dyn();
            Ok(array.to_owned())
        })
    }

    #[getter]
    fn data(&self) -> PyResult<Py<PyArrayDyn<f32>>> {
        let shape = self.0.shape.clone();
        let data = self.0.storage.read().unwrap().get_data().clone();

        Python::with_gil(|py| {
            let array = PyArray::from_vec(py, data).reshape(shape)?.to_dyn();
            Ok(array.to_owned())
        })
    }
    #[getter]
    fn grad(&self) -> PyResult<Py<PyArrayDyn<f32>>> {
        let shape = self.0.shape.clone();
        let grad = self.0.storage.read().unwrap().get_grad().clone();

        Python::with_gil(|py| {
            let array = PyArray::from_vec(py, grad).reshape(shape)?.to_dyn();
            Ok(array.to_owned())
        })
    }
    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> { Ok(self.0.shape.clone()) }
    #[getter]
    fn requires_grad(&self) -> PyResult<bool> { Ok(self.0.requires_grad) }

    // backward prop
    fn backward(&self) -> PyResult<()> { Ok(self.backward_()) }

    // binary ops
    fn __add__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(self.clone() + other.clone()) }
    fn __sub__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(self.clone() - other.clone()) }
    fn __mul__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(self.clone() * other.clone()) }
    fn __truediv__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(self.clone() / other.clone()) }
    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> { Ok(self.clone().matmul_(&other)) }
    fn matmul(&self, other: &Tensor) -> PyResult<Tensor> { Ok(self.clone().matmul_(&other)) }
    fn dot(&self, other: &Tensor) -> PyResult<Tensor> { Ok(self.clone().matmul_(&other)) }

    // unary ops
    fn pow(&self, exp: f32) -> PyResult<Tensor> { Ok(self.pow_(exp)) }
    fn relu(&self) -> PyResult<Tensor> { Ok(self.relu_()) }
    fn exp(&self) -> PyResult<Tensor> { Ok(self.exp_()) }
    fn sigmoid(&self) -> PyResult<Tensor> { Ok(self.sigmoid_()) }
    fn log(&self) -> PyResult<Tensor> { Ok(self.log_()) }
    fn __neg__(&self) -> PyResult<Tensor> { Ok(-self.clone()) }

    // reduce ops
    fn sum(&self) -> PyResult<Tensor> { Ok(self.sum_()) }
    fn mean(&self) -> PyResult<Tensor> { Ok(self.mean_()) }

    // movement ops
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Tensor> {  Ok(self.reshape_(shape)) }
    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Tensor> { Ok(self.transpose_(dim0, dim1)) }

    // functional nn ops
    fn linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> PyResult<Tensor> { Ok(self.linear_(weight, bias)) }
}