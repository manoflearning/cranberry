use crate::device::Device;

#[test]
fn device_from_str_variants() {
    assert!(matches!(Device::from_str("cpu"), Device::Cpu));
    assert!(matches!(Device::from_str("metal"), Device::Metal));
    assert!(matches!(Device::from_str("cuda"), Device::Cuda));
}

#[test]
fn device_from_str_invalid_panics() {
    let res = std::panic::catch_unwind(|| {
        let _ = Device::from_str("weird");
    });
    assert!(res.is_err());
}
