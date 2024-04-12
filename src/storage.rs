use std::simd::f32x64;
const BLOCK: usize = 64;

#[derive(Clone)]
#[repr(align(32))]
pub struct Storage {
    data: Vec<f32>,
    // TODO: Option<Vec<f32>>
    grad: Vec<f32>,
    // TODO: DType, Device
}

impl Storage {
    pub fn new(data: Vec<f32>) -> Self {
        Self { grad: vec![0.0; data.len()], data, }
    }

    pub fn init_grad_to_zero(&mut self) { self.grad = vec![0.0; self.data.len()]; }
    pub fn init_grad_to_one(&mut self) { self.grad = vec![1.0; self.data.len()]; }
    pub fn step(&mut self, lr: f32) {
        for i in 0..self.data.len() {
            self.data[i] -= lr * self.grad[i];
        }
    }

    pub fn get_data(&self) -> &Vec<f32> { &self.data }
    pub fn get_grad(&self) -> &Vec<f32> { &self.grad }
    pub fn get_grad_mut(&mut self) -> &mut Vec<f32> { &mut self.grad }

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
        // for (i, g) in o1.grad.iter_mut().enumerate() {
        //     *g += self.grad[i] * o0.data[i];
        // }
    }
    // reference: https://github.com/tinygrad/tinygrad/blob/master/extra/gemm/gemm.c
    pub fn matmul_2d(&self, other: &Storage, dim: Vec<usize>) -> Storage {
        let self_data = self.data.clone();
        let other_data = other.data.clone();
        let mut out_data = vec![0.0; dim[0] * dim[2]];

        for i in 0..dim[0] {
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
    pub fn matmul_2d_back(&self, o0: &mut Storage, o1: &mut Storage, dim: Vec<usize>) {
        let self_grad = self.grad.clone();
        let o0_data = o0.data.clone();
        let o1_data = o1.data.clone();
        let mut o0_grad = o0.grad.clone();
        let mut o1_grad = o1.grad.clone();

        for i in 0..dim[0] {
            for j in 0..dim[2] {
                for k in 0..dim[1] {
                    o0_grad[i*dim[1]+k] += o1_data[k*dim[2]+j] * self_grad[i*dim[2]+j];
                    o1_grad[k*dim[2]+j] += o0_data[i*dim[1]+k] * self_grad[i*dim[2]+j];
                }
            }
        }

        o0.grad = o0_grad;
        o1.grad = o1_grad;
    }

    // unary ops
    pub fn pow(&self, exp: f32) -> Storage {
        let data = self.data.iter().map(|a| a.powf(exp)).collect();
        Self::new(data)
    }
    pub fn pow_back(&self, other: &mut Storage) {
        // TODO: need to check the correctness of the formula
        // other.data[0]^exp = self.data[0]
        // <=> exp = log_{other.data[0]}^{self.data[0]}
        // <=> exp = log(self.data[0]) / log(other.data[0])

        // but using float division can easily cause numerical instability (e.g., nan)
        // one idea is to use ternary search.. prolly there is a better way

        // assure that exp is in [-100, 100]
        let mut l = -100.0;
        let mut r = 100.0;
        let eps = 1e-6; // 1e-6 is arbitrary

        while r - l > eps {
            let mid_0 = (2.0 * l + r) / 3.0;
            let mid_1 = (l + 2.0 * r) / 3.0;
            let mut max_diff_0: f32 = 0.0;
            let mut max_diff_1: f32 = 0.0;

            for i in 0..self.data.len() {
                let diff_0 = (self.data[i] - other.data[i].powf(mid_0)).abs();
                let diff_1 = (self.data[i] - other.data[i].powf(mid_1)).abs();
                max_diff_0 = max_diff_0.max(diff_0);
                max_diff_1 = max_diff_1.max(diff_1);
            }

            if max_diff_0 < max_diff_1 { r = mid_1; }
            else { l = mid_0; }
        }

        let exp = (l + r) / 2.0;
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
    pub fn neg(&self) -> Storage {
        let data = self.data.iter().map(|a| -a).collect();
        Self::new(data)
    }
    pub fn neg_back(&self, other: &mut Storage) {
        for (i, g) in other.grad.iter_mut().enumerate() {
            *g += -self.grad[i];
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