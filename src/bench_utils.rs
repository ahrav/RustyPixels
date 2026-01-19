//! Shared helpers for benchmark drivers.

use crate::{Image, ImageFormat};

pub const BENCH_SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];
pub const BENCH_FORMATS: [ImageFormat; 3] =
    [ImageFormat::Gray, ImageFormat::Rgb, ImageFormat::Rgba];
pub const BENCH_ANGLES: [f32; 7] = [15.0, 30.0, 45.0, 60.0, 90.0, 180.0, 270.0];

pub fn create_test_image(width: usize, height: usize, format: ImageFormat) -> Image {
    let mut img = Image::new(width, height, format);
    let channels = img.channels();
    for y in 0..height {
        let row = img.row_mut(y);
        for x in 0..width {
            for c in 0..channels {
                let val = (x + y * c) as f64 / (width + height) as f64;
                row[x * channels + c] = (val * (u16::MAX as f64)) as u16;
            }
        }
    }
    img
}

pub fn format_to_string(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Gray => "GRAY",
        ImageFormat::GrayAlpha => "GRAY_A",
        ImageFormat::Rgb => "RGB",
        ImageFormat::Rgba => "RGBA",
        _ => "UNKNOWN",
    }
}
