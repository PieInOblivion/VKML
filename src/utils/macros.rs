macro_rules! include_shader {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $name))
    };
}
