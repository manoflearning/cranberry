use std::simd::{f32x64, f32x16, f32x8, f32x4, f32x2, f32x1};
use numpy::{PyArray, PyArrayDyn};
use pyo3::{pyclass, pymethods, Py, PyResult, Python};

const BLOCK: usize = 64;
const BITS: [usize; 7] = [64, 32, 16, 8, 4, 2, 1];

#[pyclass]
#[derive(Clone)]
#[repr(align(64))] // align to 64 bytes for cache efficiency and simd, but not sure if it's effective
struct Core(Vec<f32>);

impl Core { fn new(data: Vec<f32>) -> Self { Self(data) } }

// TODO: check if it's really efficient
// macro_rules! op_io_simd {
//     ($op:ident, $in_arr:expr, $out_arr:expr, $in_idxs:expr, $out_idxs:expr) => {
//         {
//             let mut i = 0; let mut j = 0;
//             while i < $in_idxs.len() && j < $out_idxs.len() {
//                 let in_offset = $in_idxs[i].0 as usize; let in_len = $in_idxs[i].1 as usize;
//                 let out_offset = $out_idxs[j].0 as usize; let out_len = $out_idxs[j].1 as usize;
//                 let len: usize = in_len.min(out_len);
                
//                 $in_idxs[i].0 += len as i32; $out_idxs[j].0 += len as i32;
//                 $in_idxs[i].1 -= len as i32; $out_idxs[j].1 -= len as i32;
//                 if $in_idxs[i].1 == 0 { i += 1; } if $out_idxs[j].1 == 0 { j += 1; }

//                 while len >= BITS[0] {
//                     $out_arr.splice(out_offset..out_offset+BITS[0],
//                         (f32x64::from_slice(&$in_arr[in_offset..in_offset+BITS[0]]) +
//                         f32x64::from_slice(&$out_arr[out_offset..out_offset+BITS[0]])).to_array()
//                     );
//                     in_offset += BITS[0]; out_offset += BITS[0]; len -= BITS[0];
//                 }
//                 if len & BITS[1] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[1],
//                         (f32x32::from_slice(&$in_arr[in_offset..in_offset+BITS[1]]) +
//                         f32x32::from_slice(&$out_arr[out_offset..out_offset+BITS[1]])).to_array()
//                     );
//                     in_offset += BITS[1]; out_offset += BITS[1]; len -= BITS[1];
//                 }
//                 if len & BITS[2] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[2],
//                         (f32x16::from_slice(&$in_arr[in_offset..in_offset+BITS[2]]) +
//                         f32x16::from_slice(&$out_arr[out_offset..out_offset+BITS[2]])).to_array()
//                     );
//                     in_offset += BITS[2]; out_offset += BITS[2]; len -= BITS[2];
//                 }
//                 if len & BITS[3] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[3],
//                         (f32x8::from_slice(&$in_arr[in_offset..in_offset+BITS[3]]) +
//                         f32x8::from_slice(&$out_arr[out_offset..out_offset+BITS[3]])).to_array()
//                     );
//                     in_offset += BITS[3]; out_offset += BITS[3]; len -= BITS[3];
//                 }
//                 if len & BITS[4] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[4],
//                         (f32x4::from_slice(&$in_arr[in_offset..in_offset+BITS[4]]) +
//                         f32x4::from_slice(&$out_arr[out_offset..out_offset+BITS[4]])).to_array()
//                     );
//                     in_offset += BITS[4]; out_offset += BITS[4]; len -= BITS[4];
//                 }
//                 if len & BITS[5] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[5],
//                         (f32x2::from_slice(&$in_arr[in_offset..in_offset+BITS[5]]) +
//                         f32x2::from_slice(&$out_arr[out_offset..out_offset+BITS[5]])).to_array()
//                     );
//                     in_offset += BITS[5]; out_offset += BITS[5]; len -= BITS[5];
//                 }
//                 if len & BITS[6] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[6],
//                         (f32x1::from_slice(&$in_arr[in_offset..in_offset+BITS[6]]) +
//                         f32x1::from_slice(&$out_arr[out_offset..out_offset+BITS[6]])).to_array()
//                     );
//                     in_offset += BITS[6]; out_offset += BITS[6]; len -= BITS[6];
//                 }
//             }
//             assert!(i == $in_idxs.len() && j == $out_idxs.len());
//         }
//     };
// }
// macro_rules! op_iio_simd {
//     ($op:ident, $in1_arr:expr, $in2_arr:expr, $out_arr:expr, $in1_idxs:expr, $in2_idxs:expr, $out_idxs:expr) => {
//         {
//             let mut i = 0; let mut j = 0; let mut k = 0;
//             while i < $in1_idxs.len() && j < $in2_idxs.len() && k < $out_idxs.len() {
//                 let in1_offset = $in1_idxs[i].0 as usize; let in1_len = $in1_idxs[i].1 as usize;
//                 let in2_offset = $in2_idxs[j].0 as usize; let in2_len = $in2_idxs[j].1 as usize;
//                 let out_offset = $out_idxs[k].0 as usize; let out_len = $out_idxs[k].1 as usize;
//                 let len: usize = in1_len.min(in2_len).min(out_len);
                
//                 $in1_idxs[i].0 += len as i32; $in2_idxs[j].0 += len as i32; $out_idxs[k].0 += len as i32;
//                 $in1_idxs[i].1 -= len as i32; $in2_idxs[j].1 -= len as i32; $out_idxs[k].1 -= len as i32;
//                 if ($in1_idxs[i].1 == 0) { i += 1; } if ($in2_idxs[j].1 == 0) { j += 1; } if ($out_idxs[k].1 == 0) { k += 1; }

//                 while len >= BITS[0] {
//                     $out_arr.splice(out_offset..out_offset+BITS[0],
//                         (f32x64::from_slice(&$in1_arr[in1_offset..in1_offset+BITS[0]]).$op(
//                             f32x64::from_slice(&$in2_arr[in2_offset..in2_offset+BITS[0]])
//                         ) + f32x64::from_slice(&$out_arr[out_offset..out_offset+BITS[0]])).to_array()
//                     );
//                     in1_offset += BITS[0]; in2_offset += BITS[0]; out_offset += BITS[0]; len -= BITS[0];
//                 }
//                 if len & BITS[1] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[1],
//                         (f32x32::from_slice(&$in1_arr[in1_offset..in1_offset+BITS[1]]).$op(
//                             f32x32::from_slice(&$in2_arr[in2_offset..in2_offset+BITS[1]])
//                         ) + f32x32::from_slice(&$out_arr[out_offset..out_offset+BITS[1]])).to_array()
//                     );
//                     in1_offset += BITS[1]; in2_offset += BITS[1]; out_offset += BITS[1]; len -= BITS[1];
//                 }
//                 if len & BITS[2] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[2],
//                         (f32x16::from_slice(&$in1_arr[in1_offset..in1_offset+BITS[2]]).$op(
//                             f32x16::from_slice(&$in2_arr[in2_offset..in2_offset+BITS[2]])
//                         ) + f32x16::from_slice(&$out_arr[out_offset..out_offset+BITS[2]])).to_array()
//                     );
//                     in1_offset += BITS[2]; in2_offset += BITS[2]; out_offset += BITS[2]; len -= BITS[2];
//                 }
//                 if len & BITS[3] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[3],
//                         (f32x8::from_slice(&$in1_arr[in1_offset..in1_offset+BITS[3]]).$op(
//                             f32x8::from_slice(&$in2_arr[in2_offset..in2_offset+BITS[3]])
//                         ) + f32x8::from_slice(&$out_arr[out_offset..out_offset+BITS[3]])).to_array()
//                     );
//                     in1_offset += BITS[3]; in2_offset += BITS[3]; out_offset += BITS[3]; len -= BITS[3];
//                 }
//                 if len & BITS[4] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[4],
//                         (f32x4::from_slice(&$in1_arr[in1_offset..in1_offset+BITS[4]]).$op(
//                             f32x4::from_slice(&$in2_arr[in2_offset..in2_offset+BITS[4]])
//                         ) + f32x4::from_slice(&$out_arr[out_offset..out_offset+BITS[4]])).to_array()
//                     );
//                     in1_offset += BITS[4]; in2_offset += BITS[4]; out_offset += BITS[4]; len -= BITS[4];
//                 }
//                 if len & BITS[5] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[5],
//                         (f32x2::from_slice(&$in1_arr[in1_offset..in1_offset+BITS[5]]).$op(
//                             f32x2::from_slice(&$in2_arr[in2_offset..in2_offset+BITS[5]])
//                         ) + f32x2::from_slice(&$out_arr[out_offset..out_offset+BITS[5]])).to_array()
//                     );
//                     in1_offset += BITS[5]; in2_offset += BITS[5]; out_offset += BITS[5]; len -= BITS[5];
//                 }
//                 if len & BITS[6] != 0 {
//                     $out_arr.splice(out_offset..out_offset+BITS[6],
//                         (f32x1::from_slice(&$in1_arr[in1_offset..in1_offset+BITS[6]]).$op(
//                             f32x1::from_slice(&$in2_arr[in2_offset..in2_offset+BITS[6]])
//                         ) + f32x1::from_slice(&$out_arr[out_offset..out_offset+BITS[6]])).to_array()
//                     );
//                     in1_offset += BITS[6]; in2_offset += BITS[6]; out_offset += BITS[6]; len -= BITS[6];
//                 }
//             }
//             assert!(i == $in1_idxs.len() && j == $in2_idxs.len() && k == $out_idxs.len());
//         }
//     };
// }

macro_rules! op_io_naive {
    ($op:ident, $in_arr:expr, $out_arr:expr, $in_idxs:expr, $out_idxs:expr) => {
        let mut i = 0; let mut j = 0;
        while i < $in_idxs.len() && j < $out_idxs.len() {
            let in_offset = $in_idxs[i].0 as usize; let in_len = $in_idxs[j].1 as usize;
            let out_offset = $out_idxs[j].0 as usize; let out_len = $out_idxs[j].1 as usize;

            let len: usize = in_len.min(out_len);
            $in_idxs[i].0 += len as i32; $in_idxs[i].1 -= len as i32;
            $out_idxs[j].0 += len as i32; $out_idxs[j].1 -= len as i32;
            if $in_idxs[i].1 == 0 { i += 1; } if $out_idxs[j].1 == 0 { j += 1; }

            for _ in 0..len {
                $out_arr[out_offset] = $out_arr[out_offset].$op($in_arr[in_offset]);
                in_offset += 1; out_offset += 1;
            }
        }
        assert!(i == $in_idxs.len() && j == $out_idxs.len());
    };
}

macro_rules! op_io_naive {
    ($op:ident, $in1_arr:expr, $in2_arr:expr, $out_arr:expr, $in1_idxs:expr, $in2_idxs:expr, $out_idxs:expr) => {
        let mut i = 0; let mut j = 0; let mut k = 0;
        while i < $in1_idxs.len() && j < $in2_idxs.len() && k < $out_idxs.len() {
            let in1_offset = $in1_idxs[i].0 as usize; let in1_len = $in1_idxs[j].1 as usize;
            let in2_offset = $in2_idxs[j].0 as usize; let in1_len = $in1_idxs[j].1 as usize;
            let out_offset = $out_idxs[j].0 as usize; let out_len = $out_idxs[j].1 as usize;

            let len: usize = in_len.min(out_len);
            $in_idxs[i].0 += len as i32; $in_idxs[i].1 -= len as i32;
            $out_idxs[j].0 += len as i32; $out_idxs[j].1 -= len as i32;
            if $in_idxs[i].1 == 0 { i += 1; } if $out_idxs[j].1 == 0 { j += 1; }

            for _ in 0..len {
                $out_arr[out_offset] = $out_arr[out_offset].$op($in_arr[in_offset]);
                in_offset += 1; out_offset += 1;
            }
        }
        assert!(i == $in_idxs.len() && j == $out_idxs.len());
    };
}

// the poor man's random number generator
// https://blog.orhun.dev/zero-deps-random-in-rust/
fn random_numbers_u32(seed: u32) -> impl Iterator<Item = u32> {
    let mut random = seed;
    std::iter::repeat_with(move || {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        random
    })
}
fn random_numbers_f32(seed: u32, low: f32, high: f32) -> impl Iterator<Item = f32> {
    // uniform distribution in [low, high]
    // TODO: check if it's really uniform and in [low, high]
    // TODO: check if it's avoid nan/inf
    random_numbers_u32(seed).map(move |x| (x as f32 / u32::MAX as f32) * (high - low) + low)
}

#[pymethods]
impl Core {
    #[new]
    fn __init__(data: Vec<f32>) -> PyResult<Self> { Ok(Self::new(data)) }

    #[staticmethod]
    fn zeros(size: usize) -> PyResult<Self> { Ok(Self::new(vec![0.0; size])) }
    #[staticmethod]
    fn ones(size: usize) -> PyResult<Self> { Ok(Self::new(vec![1.0; size])) }

    #[getter]
    fn data(&self) -> PyResult<Vec<f32>> { Ok(self.data.clone()) }
    fn numpy(&self) -> PyResult<Py<PyArrayDyn<f32>>> {
        Python::with_gil(|py| {
            let array = PyArray::from_vec(py, self.data.clone()).reshape(vec![self.data.len()])?.to_dyn();
            Ok(array.to_owned())
        })
    }

    fn fill(&mut self, x: f32) { for g in self.0.iter_mut() { *g = x; } }

    fn step(&mut self, lr: f32) { for (d, g) in self.data.iter_mut().zip(self.grad.iter()) { *d -= lr * g; } }

    // operations
    // TODO: add, sub, mul, div, pow, exp, log

    // random
    #[staticmethod]
    fn randn(size: usize, seed: u32) -> PyResult<Self> {
        // zero mean, unit variance normal distribution
        // manual normal distribution using box-muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        let mut data = random_numbers_f32(seed, 0.0, 1.0).take(size).collect::<Vec<f32>>();
        for i in (0..size).step_by(2) {
            let u1 = data[i];
            let u2 = data[i+1];
            let r = (-2.0 * u1.ln()).sqrt(); // since u1 is in (0, 1), sqrt(-2 * ln(u1)) is in (0, inf)
            data[i] = r * (2.0 * std::f32::consts::PI * u2).cos();
            data[i+1] = r * (2.0 * std::f32::consts::PI * u2).sin();
        }
        Ok(Self::new(data))
    }
    #[staticmethod]
    fn uniform(size: usize, low: f32, high: f32, seed: u32) -> PyResult<Self> {
        Ok(Self::new(random_numbers_f32(seed, low, high).take(size).map(|x| x).collect()))
    }
}