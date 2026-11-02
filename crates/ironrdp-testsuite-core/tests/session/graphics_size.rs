#[path = "../../../ironrdp-web/src/graphics_size.rs"]
#[allow(dead_code, unreachable_pub, clippy::unwrap_used)]
mod graphics_size_impl;

use graphics_size_impl::checked_graphics_size;

#[test]
fn accepts_up_to_the_limits() {
    assert!(checked_graphics_size(7680, 4320).is_ok());
    assert!(checked_graphics_size(8192, 8192).is_ok());
    assert!(checked_graphics_size(16384, 4096).is_ok());
}

#[test]
fn refuses_beyond_the_limits() {
    assert!(checked_graphics_size(16385, 1).is_err());
    assert!(checked_graphics_size(1, 16385).is_err());
    assert!(checked_graphics_size(16384, 4097).is_err());
    assert!(checked_graphics_size(32766, 32766).is_err());
}

#[test]
fn refuses_a_zero_axis() {
    assert!(checked_graphics_size(0, 100).is_err());
    assert!(checked_graphics_size(100, 0).is_err());
}
