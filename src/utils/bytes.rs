/// Return a byte slice view of any sized value.
///
/// Safety: This performs a plain reinterpretation of the memory of `T` as bytes.
/// The caller must ensure the value is POD-like (e.g., `#[repr(C)]` push-constant structs
/// containing only integer/floating fields). This helper avoids repeated unsafe blocks
/// across the codebase.
pub fn as_bytes<T: Sized>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts((v as *const T) as *const u8, std::mem::size_of::<T>()) }
}
