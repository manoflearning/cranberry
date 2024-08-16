#[cfg(test)]
mod test_storage {
    use crate::storage_ptr::{
        storage_full, storage_full_vec, storage_neg, storage_relu, storage_sqrt,
    };
    use rand::random;

    const DEVICE: &str = "cpu";
    const VEC_SIZE: usize = 1000;
    const EPSILON: f32 = 1e-6; // poor man's float comparison

    #[test]
    fn test_storage_full() {
        todo!();
    }
    #[test]
    fn test_storage_full_vec() {
        todo!()
    }
    #[test]
    fn test_storage_clone() {
        todo!()
    }
    #[test]
    fn test_storage_drop() {
        todo!()
    }
    #[test]
    fn test_storage_neg() {
        todo!()
    }
    #[test]
    fn test_storage_sqrt() {
        todo!()
    }
    #[test]
    fn test_storage_relu() {
        todo!()
    }
    #[test]
    fn test_storage_exp() {
        todo!()
    }
    #[test]
    fn test_storage_log() {
        todo!()
    }
    #[test]
    fn test_storage_add() {
        todo!()
    }
    #[test]
    fn test_storage_sub() {
        todo!()
    }
    #[test]
    fn test_storage_mul() {
        todo!()
    }
    #[test]
    fn test_storage_div() {
        todo!()
    }
    #[test]
    fn test_storage_sum() {
        todo!()
    }
    #[test]
    fn test_storage_max() {
        todo!()
    }
}
