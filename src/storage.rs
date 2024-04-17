use std::simd::f32x64;
const BLOCK: usize = 64;

#[derive(Clone)]
#[repr(align(32))]
pub struct Storage {
    data: Vec<f32>,
    grad: Vec<f32>,
}

impl Storage {
    pub fn new(data: Vec<f32>) -> Self {
        Self { grad: vec![0.0; data.len()], data, }
    }

    pub fn init_grad_to_zero(&mut self) { self.grad = vec![0.0; self.data.len()]; }
    pub fn init_grad_to_one(&mut self) { self.grad = vec![1.0; self.data.len()]; }
    pub fn step(&mut self, lr: f32) { for i in 0..self.data.len() { self.data[i] -= lr * self.grad[i]; } }

    pub fn get_data(&self) -> &Vec<f32> { &self.data }
    pub fn get_grad(&self) -> &Vec<f32> { &self.grad }

    // broadcasting ops
    pub fn broadcast(&self, in_shape: Vec<usize>, out_shape: Vec<usize>) -> Storage {
        let mut in_shape = in_shape;
        while in_shape.len() < out_shape.len() { in_shape.insert(0, 1); }

        fn fill_data(dep: usize, in_idx: usize, in_data: &Vec<f32>, in_shape: &Vec<usize>, out_idx: usize, out_data: &mut Vec<f32>, out_shape: &Vec<usize>) {
            if dep == out_shape.len() { out_data[out_idx] = in_data[in_idx]; return; }

            for i in 0..out_shape[dep] {
                let n_in_idx = in_idx * in_shape[dep] + i % in_shape[dep];
                let n_out_idx = out_idx * out_shape[dep] + i;
                fill_data(dep + 1, n_in_idx, in_data, in_shape, n_out_idx, out_data, out_shape);
            }
        }

        let mut out_data = vec![0.0; out_shape.iter().product()];

        fill_data(0, 0, &self.data, &in_shape, 0, &mut out_data, &out_shape);

        Self::new(out_data)
    }
    pub fn broadcast_back(&self, other: &mut Storage, s_shape: Vec<usize>, o_shape: Vec<usize>) {
        let mut o_shape = o_shape;
        while o_shape.len() < s_shape.len() { o_shape.insert(0, 1); }

        fn fill_grad(dep: usize, in_idx: usize, in_grad: &mut Vec<f32>, in_shape: &Vec<usize>, out_idx: usize, out_grad: &Vec<f32>, out_shape: &Vec<usize>) {
            if dep == out_shape.len() { in_grad[in_idx] += out_grad[out_idx]; return; }

            for i in 0..out_shape[dep] {
                let n_in_idx = in_idx * in_shape[dep] + i % in_shape[dep];
                let n_out_idx = out_idx * out_shape[dep] + i;
                fill_grad(dep + 1, n_in_idx, in_grad, in_shape, n_out_idx, out_grad, out_shape);
            }
        }

        fill_grad(0, 0, &mut other.grad, &o_shape, 0, &self.grad, &s_shape);
    }

    // binary ops
    pub fn add(&self, other: &Storage) -> Storage {
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        Self::new(data)
    }
    pub fn add_back(&self, other: &mut Storage) {
        for (i, g) in other.grad.iter_mut().enumerate() {
            *g += self.grad[i];
        }
    }
    pub fn mul(&self, other: &Storage) -> Storage {
        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        Self::new(data)
    }
    pub fn mul_back(&self, o0: &mut Storage, o1: &Storage) {
        for (i, g) in o0.grad.iter_mut().enumerate() {
            *g += self.grad[i] * o1.data[i];
        }
    }
    pub fn matmul_2d(&self, other: &Storage, dim: Vec<usize>) -> Storage {
        // reference: https://github.com/tinygrad/tinygrad/blob/master/extra/gemm/gemm.c
        let self_data = self.data.clone();
        let other_data = other.data.clone();
        let mut out_data = vec![0.0; dim[0] * dim[2]];

        for i in 0..dim[0] { // TODO: need to optimize
            for j in (0..dim[2]).step_by(BLOCK) {

                if j+BLOCK < dim[2] {
                    let mut acc = f32x64::splat(0.0);
                    for k in 0..dim[1] {
                        let ta = f32x64::splat(self_data[i*dim[1]+k]);
                        let tb = f32x64::from_slice(&other_data[k*dim[2]+j .. k*dim[2]+j+BLOCK]);

                        acc += ta * tb;
                    }
                    out_data.splice(i*dim[2]+j..i*dim[2]+j+BLOCK, acc.to_array());
                }
                else { // TODO: need a faster way to handle the edge case
                    for k in 0..dim[1] {
                        for l in j..dim[2] {
                            out_data[i*dim[2]+l] += self_data[i*dim[1]+k] * other_data[k*dim[2]+l];
                        }
                    }
                }

            }
        }

        Storage::new(out_data)
    }
    pub fn matmul_2d_back_0(&self, o0: &mut Storage, o1: &Storage, dim: Vec<usize>) {
        let self_grad = self.grad.clone();
        // let o0_data = o0.data.clone();
        let o1_data = o1.data.clone();
        let mut o0_grad = o0.grad.clone();
        // let mut o1_grad = o1.grad.clone();

        for i in 0..dim[0] {
            for j in 0..dim[2] {
                for k in 0..dim[1] {
                    o0_grad[i*dim[1]+k] += o1_data[k*dim[2]+j] * self_grad[i*dim[2]+j];
                    // o1_grad[k*dim[2]+j] += o0_data[i*dim[1]+k] * self_grad[i*dim[2]+j];
                }
            }
        }

        o0.grad = o0_grad;
        // o1.grad = o1_grad;
    }
    pub fn matmul_2d_back_1(&self, o0: &Storage, o1: &mut Storage, dim: Vec<usize>) {
        let self_grad = self.grad.clone();
        let o0_data = o0.data.clone();
        // let o1_data = o1.data.clone();
        // let mut o0_grad = o0.grad.clone();
        let mut o1_grad = o1.grad.clone();

        for i in 0..dim[0] {
            for j in 0..dim[2] {
                for k in 0..dim[1] {
                    // o0_grad[i*dim[1]+k] += o1_data[k*dim[2]+j] * self_grad[i*dim[2]+j];
                    o1_grad[k*dim[2]+j] += o0_data[i*dim[1]+k] * self_grad[i*dim[2]+j];
                }
            }
        }

        // o0.grad = o0_grad;
        o1.grad = o1_grad;
    }

    // unary ops
    pub fn neg(&self) -> Storage {
        let data = self.data.iter().map(|a| -a).collect();
        Self::new(data)
    }
    pub fn neg_back(&self, other: &mut Storage) {
        for (i, g) in other.grad.iter_mut().enumerate() {
            *g += -self.grad[i];
        }
    }
    pub fn pow(&self, exp: f32) -> Storage {
        let data = self.data.iter().map(|a| a.powf(exp)).collect();
        Self::new(data)
    }
    pub fn pow_back(&self, other: &mut Storage, exp: f32) {
        for (i, g) in other.grad.iter_mut().enumerate() {
            *g += self.grad[i] * exp * other.data[i].powf(exp - 1.0);
        }
    }
    pub fn relu(&self) -> Storage {
        let data = self.data.iter().map(|a| a.max(0.0)).collect();
        Self::new(data)
    }
    pub fn relu_back(&self, other: &mut Storage) {
        for (i, g) in other.grad.iter_mut().enumerate() {
            *g += self.grad[i] * if self.data[i] > 0.0 { 1.0 } else { 0.0 };
        }
    }
    pub fn exp(&self) -> Storage {
        let data = self.data.iter().map(|a| a.exp()).collect();
        Self::new(data)
    }
    pub fn exp_back(&self, other: &mut Storage) {
        for (i, g) in other.grad.iter_mut().enumerate() {
            *g += self.grad[i] * self.data[i];
        }
    }
    pub fn log(&self) -> Storage {
        let data = self.data.iter().map(|a| a.ln()).collect();
        Self::new(data)
    }
    pub fn log_back(&self, other: &mut Storage) {
        for (i, g) in other.grad.iter_mut().enumerate() {
            *g += self.grad[i] / other.data[i];
        }
    }

    // reduce ops
    pub fn sum(&self) -> Storage {
        let data = vec![self.data.iter().sum()];
        Self::new(data)
    }
    pub fn sum_back(&self, other: &mut Storage) {
        for g in other.grad.iter_mut() { *g += self.grad[0]; }
    }
}