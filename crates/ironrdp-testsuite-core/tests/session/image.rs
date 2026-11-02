use ironrdp_graphics::image_processing::PixelFormat;
use ironrdp_pdu::geometry::{InclusiveRectangle, Rectangle as _};
use ironrdp_session::image::DecodedImage;

#[test]
fn rgba32_bitmap_with_a_short_payload_is_skipped() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 128, 128);
    let region = InclusiveRectangle {
        left: 0,
        top: 0,
        right: 127,
        bottom: 127,
    };

    let updated = image.apply_rgba32_bitmap(&[0xab; 4], &region).unwrap();

    assert_eq!(updated, InclusiveRectangle::empty());
    assert!(image.data().iter().all(|&byte| byte == 0));
}

#[test]
fn rgba32_bitmap_with_a_matching_payload_is_applied() {
    let mut image = DecodedImage::new(PixelFormat::RgbA32, 16, 16);
    let region = InclusiveRectangle {
        left: 2,
        top: 3,
        right: 5,
        bottom: 4,
    };

    let updated = image.apply_rgba32_bitmap(&[0xcd; 4 * 2 * 4], &region).unwrap();

    assert_eq!(updated, region);
    let stride = image.stride();
    assert_eq!(&image.data()[3 * stride + 2 * 4..3 * stride + 6 * 4], &[0xcd; 16]);
}
