#[derive(PartialEq)]
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
