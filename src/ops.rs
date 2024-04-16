#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Ops {
    Broadcast,
    Add,
    Mul,
    Matmul,
    Neg,
    Pow,
    Relu,
    Exp,
    Log,
    Sum,
    Reshape
}