#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Ops {
    Broadcast,
    Add,
    Mul,
    Matmul,
    Pow,
    Relu,
    Exp,
    Log,
    Neg,
    Sum,
    Reshape
}