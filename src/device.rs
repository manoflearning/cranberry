#[derive(Clone, Debug, PartialEq)]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
}

impl Device {
    pub fn from_str(device: &str) -> Device {
        match device {
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            "cuda" => Device::Cuda,
            _ => panic!("Unsupported device {}", device),
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Metal => write!(f, "metal"),
            Device::Cuda => write!(f, "cuda"),
        }
    }
}
