use std::ops::{Deref, Index, IndexMut, Range, RangeFrom};

#[repr(align(64))] // for SIMD and cache alignment
#[derive(Clone)]
pub struct Data {
    pub(crate) data: Vec<f32>,
}

impl Deref for Data {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl PartialEq for Data {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl FromIterator<f32> for Data {
    fn from_iter<I: IntoIterator<Item=f32>>(iter: I) -> Self {
        Data {
            data: iter.into_iter().collect(),
        }
    }
}

impl Index<usize> for Data {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl IndexMut<usize> for Data {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
impl Index<RangeFrom<usize>> for Data {
    type Output = [f32];
    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        &self.data[index]
    }
}
impl Index<Range<usize>> for Data {
    type Output = [f32];
    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.data[index]
    }
}