#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Ops {
    Broadcast,
    Pow,
    Relu,
    Neg,
    Add,
    Mul,
    Matmul,
    Sum,
    Reshape
}