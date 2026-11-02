//! Output size limits for the EGFX graphics buffer. Kept free of browser types so the
//! testsuite can compile this file directly (`ironrdp-web` only builds for wasm).

use core::num::NonZeroU32;

use anyhow::Context as _;

/// Upper bound on one graphics output axis, in pixels. Matches the EGFX compositor's surface cap.
const MAX_GRAPHICS_DIM: u16 = 16384;

/// Upper bound on the RGBA bytes of the graphics output buffer. Matches the EGFX
/// compositor's memory budget, so the output buffer cannot outgrow what the
/// compositor accepts.
const MAX_GRAPHICS_BYTES: u64 = 256 * 1024 * 1024;

/// Validate the output size a server announces with `ResetGraphics` before the
/// session allocates a buffer for it. The protocol allows up to 32766 pixels per
/// axis, about 4.29 GB of RGBA, which a hostile server could use to abort the
/// WebAssembly instance.
pub(crate) fn checked_graphics_size(width: u16, height: u16) -> anyhow::Result<(NonZeroU32, NonZeroU32)> {
    let non_zero_width = NonZeroU32::new(u32::from(width)).context("EGFX width is zero")?;
    let non_zero_height = NonZeroU32::new(u32::from(height)).context("EGFX height is zero")?;
    let bytes = u64::from(width) * u64::from(height) * 4;
    anyhow::ensure!(
        width <= MAX_GRAPHICS_DIM && height <= MAX_GRAPHICS_DIM && bytes <= MAX_GRAPHICS_BYTES,
        "EGFX output size {width}x{height} exceeds the supported maximum"
    );
    Ok((non_zero_width, non_zero_height))
}
