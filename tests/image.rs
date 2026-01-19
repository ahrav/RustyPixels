use std::cmp;

use proptest::prelude::*;
use rusty_pixels::{Color, Image, ImageFormat};

fn expected_gray(r: u16, g: u16, b: u16) -> u16 {
    let r = r as u64;
    let g = g as u64;
    let b = b as u64;
    ((77 * r + 150 * g + 29 * b) >> 8) as u16
}

fn premultiply_sample(sample: u16, alpha: u16) -> u16 {
    if alpha == u16::MAX {
        return sample;
    }
    let blend = alpha as f32 / u16::MAX as f32;
    (sample as f32 * blend) as u16
}

fn fill_pattern(image: &mut Image) {
    let channels = image.channels();
    if channels == 0 {
        return;
    }
    let width = image.width();
    for y in 0..image.height() {
        for x in 0..width {
            let base = ((y * width + x) * channels) as u16;
            let pixel = image.pixel_mut(x, y);
            for c in 0..channels {
                pixel[c] = base + c as u16;
            }
        }
    }
}

fn expected_crop_bounds(
    img_w: usize,
    img_h: usize,
    left: i32,
    top: i32,
    width: usize,
    height: usize,
) -> (usize, usize, usize, usize, bool) {
    let mut left = left;
    let mut top = top;
    let mut width = width;
    let mut height = height;

    if left < 0 {
        width = width.saturating_sub((-left) as usize);
        left = 0;
    }
    if top < 0 {
        height = height.saturating_sub((-top) as usize);
        top = 0;
    }
    if left as usize > img_w || top as usize > img_h {
        return (0, 0, 0, 0, true);
    }

    width = cmp::min(width, img_w.saturating_sub(left as usize));
    height = cmp::min(height, img_h.saturating_sub(top as usize));
    (left as usize, top as usize, width, height, false)
}

#[test]
fn test_image_format_channels_and_alpha() {
    let cases = [
        (ImageFormat::None, 0, false),
        (ImageFormat::Gray, 1, false),
        (ImageFormat::GrayAlpha, 2, true),
        (ImageFormat::Rgb, 3, false),
        (ImageFormat::Rgba, 4, true),
        (ImageFormat::YCbCr, 3, false),
        (ImageFormat::YCbCrA, 4, true),
    ];

    for (format, channels, has_alpha) in cases {
        assert_eq!(format.channel_count(), channels);
        assert_eq!(format.has_alpha(), has_alpha);
    }
}

#[test]
fn test_color_accessors_and_gray() {
    let rgba = Color::new([10, 20, 30, 40], ImageFormat::Rgba);
    assert_eq!(rgba.r(), 10);
    assert_eq!(rgba.g(), 20);
    assert_eq!(rgba.b(), 30);
    assert_eq!(rgba.a(), 40);
    assert_eq!(rgba.gray(), expected_gray(10, 20, 30));

    let gray_alpha = Color::new([111, 222, 333, 444], ImageFormat::GrayAlpha);
    assert_eq!(gray_alpha.r(), 111);
    assert_eq!(gray_alpha.g(), 111);
    assert_eq!(gray_alpha.b(), 111);
    assert_eq!(gray_alpha.y(), 111);
    assert_eq!(gray_alpha.cb(), 111);
    assert_eq!(gray_alpha.cr(), 111);
    assert_eq!(gray_alpha.a(), 222);
    assert_eq!(gray_alpha.gray(), 111);

    let ycbcra = Color::new([500, 600, 700, 800], ImageFormat::YCbCrA);
    assert_eq!(ycbcra.y(), 500);
    assert_eq!(ycbcra.cb(), 600);
    assert_eq!(ycbcra.cr(), 700);
    assert_eq!(ycbcra.a(), 800);
    assert_eq!(ycbcra.gray(), 500);

    let none = Color::new([0, 0, 0, 0], ImageFormat::None);
    assert_eq!(none.gray(), 0);
    assert_eq!(none.a(), u16::MAX);
}

#[test]
#[should_panic]
fn test_color_rgb_panics_on_ycbcr() {
    let color = Color::new([1, 2, 3, 4], ImageFormat::YCbCr);
    let _ = color.r();
}

#[test]
#[should_panic]
fn test_color_y_panics_on_rgb() {
    let color = Color::new([1, 2, 3, 4], ImageFormat::Rgb);
    let _ = color.y();
}

#[test]
fn test_fill_pixel_per_format() {
    let rgba = Color::new([10, 20, 30, 40], ImageFormat::Rgba);
    let mut pixel = vec![0u16; 1];
    Image::fill_pixel(ImageFormat::Gray, &mut pixel, &rgba);
    assert_eq!(pixel, vec![10]);

    let mut pixel = vec![0u16; 2];
    Image::fill_pixel(ImageFormat::GrayAlpha, &mut pixel, &rgba);
    assert_eq!(pixel, vec![10, 40]);

    let mut pixel = vec![0u16; 3];
    Image::fill_pixel(ImageFormat::Rgb, &mut pixel, &rgba);
    assert_eq!(pixel, vec![10, 20, 30]);

    let mut pixel = vec![0u16; 4];
    Image::fill_pixel(ImageFormat::Rgba, &mut pixel, &rgba);
    assert_eq!(pixel, vec![10, 20, 30, 40]);

    let ycbcra = Color::new([1, 2, 3, 4], ImageFormat::YCbCrA);
    let mut pixel = vec![0u16; 3];
    Image::fill_pixel(ImageFormat::YCbCr, &mut pixel, &ycbcra);
    assert_eq!(pixel, vec![1, 2, 3]);

    let mut pixel = vec![0u16; 4];
    Image::fill_pixel(ImageFormat::YCbCrA, &mut pixel, &ycbcra);
    assert_eq!(pixel, vec![1, 2, 3, 4]);
}

#[test]
fn test_fill_u8_sets_pixels() {
    let mut img = Image::new(2, 2, ImageFormat::Rgba);
    img.fill_u8(1, 2, 3, 4);
    let alpha = 4u16 * 257;
    let expected = [
        premultiply_sample(1u16 * 257, alpha),
        premultiply_sample(2u16 * 257, alpha),
        premultiply_sample(3u16 * 257, alpha),
        alpha,
    ];

    for y in 0..2 {
        for x in 0..2 {
            assert_eq!(img.pixel(x, y), expected.as_slice());
        }
    }
}

#[test]
fn test_fill_u8_non_premultiplied() {
    let mut img = Image::with_premultiplied(2, 2, ImageFormat::Rgba, false);
    img.fill_u8(1, 2, 3, 4);
    let expected = [1u16 * 257, 2u16 * 257, 3u16 * 257, 4u16 * 257];

    for y in 0..2 {
        for x in 0..2 {
            assert_eq!(img.pixel(x, y), expected.as_slice());
        }
    }
}

#[test]
fn test_fill_with_color_rgb() {
    let mut img = Image::new(3, 1, ImageFormat::Rgb);
    let color = Color::new([11, 22, 33, 44], ImageFormat::Rgba);
    img.fill_with_color(&color);

    for x in 0..3 {
        assert_eq!(img.pixel(x, 0), &[11, 22, 33]);
    }
}

#[test]
fn test_row_and_pixel_layout() {
    let mut img = Image::new(2, 2, ImageFormat::Rgb);
    {
        let row0 = img.row_mut(0);
        row0.copy_from_slice(&[1, 2, 3, 4, 5, 6]);
    }
    {
        let row1 = img.row_mut(1);
        row1.copy_from_slice(&[7, 8, 9, 10, 11, 12]);
    }

    assert_eq!(img.row(0), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(img.row(1), &[7, 8, 9, 10, 11, 12]);
    assert_eq!(img.pixel(0, 0), &[1, 2, 3]);
    assert_eq!(img.pixel(1, 0), &[4, 5, 6]);
    assert_eq!(img.pixel(0, 1), &[7, 8, 9]);
    assert_eq!(img.pixel(1, 1), &[10, 11, 12]);
}

#[test]
fn test_clone_is_deep_copy() {
    let mut img = Image::new(1, 1, ImageFormat::Rgb);
    img.pixel_mut(0, 0).copy_from_slice(&[1, 2, 3]);

    let mut cloned = img.clone();
    cloned.pixel_mut(0, 0).copy_from_slice(&[4, 5, 6]);

    assert_eq!(img.pixel(0, 0), &[1, 2, 3]);
    assert_eq!(cloned.pixel(0, 0), &[4, 5, 6]);
    assert_eq!(cloned.format(), img.format());
    assert_eq!(cloned.allocation(), img.allocation());
    assert_eq!(cloned.is_premultiplied(), img.is_premultiplied());
}

#[test]
fn test_crop_basic() {
    let mut img = Image::new(4, 3, ImageFormat::Rgba);
    fill_pattern(&mut img);

    let cropped = img.crop(1, 1, 2, 2);
    assert_eq!(cropped.width(), 2);
    assert_eq!(cropped.height(), 2);
    assert_eq!(cropped.format(), ImageFormat::Rgba);

    for y in 0..2 {
        for x in 0..2 {
            assert_eq!(cropped.pixel(x, y), img.pixel(x + 1, y + 1));
        }
    }
}

#[test]
fn test_crop_negative_clamps() {
    let mut img = Image::new(2, 2, ImageFormat::Rgb);
    fill_pattern(&mut img);

    let cropped = img.crop(-1, -1, 2, 2);
    assert_eq!(cropped.width(), 1);
    assert_eq!(cropped.height(), 1);
    assert_eq!(cropped.pixel(0, 0), img.pixel(0, 0));
}

#[test]
fn test_crop_full_returns_clone() {
    let mut img = Image::with_premultiplied(2, 2, ImageFormat::Rgba, false);
    fill_pattern(&mut img);

    let original = img.pixel(0, 0).to_vec();
    let mut cropped = img.crop(0, 0, img.width(), img.height());

    assert_eq!(cropped.width(), img.width());
    assert_eq!(cropped.height(), img.height());
    assert_eq!(cropped.format(), img.format());
    assert_eq!(cropped.is_premultiplied(), img.is_premultiplied());
    assert_eq!(cropped.allocation(), img.allocation());

    cropped
        .pixel_mut(0, 0)
        .copy_from_slice(&[999, 998, 997, 996]);
    assert_eq!(img.pixel(0, 0), original.as_slice());
}

#[test]
fn test_crop_out_of_bounds_empty() {
    let img = Image::new(2, 2, ImageFormat::Rgb);

    let cropped = img.crop(3, 0, 1, 1);
    assert_eq!(cropped.width(), 0);
    assert_eq!(cropped.height(), 0);

    let cropped = img.crop(0, 3, 1, 1);
    assert_eq!(cropped.width(), 0);
    assert_eq!(cropped.height(), 0);
}

proptest! {
    #[test]
    fn prop_crop_matches_reference(
        width in 0usize..8,
        height in 0usize..8,
        left in -8i32..8,
        top in -8i32..8,
        crop_w in 0usize..8,
        crop_h in 0usize..8,
    ) {
        let mut img = Image::new(width, height, ImageFormat::Rgba);
        fill_pattern(&mut img);

        let cropped = img.crop(left, top, crop_w, crop_h);
        let (x0, y0, out_w, out_h, empty) =
            expected_crop_bounds(width, height, left, top, crop_w, crop_h);

        if empty {
            prop_assert_eq!(cropped.width(), 0);
            prop_assert_eq!(cropped.height(), 0);
        } else {
            prop_assert_eq!(cropped.width(), out_w);
            prop_assert_eq!(cropped.height(), out_h);
            prop_assert_eq!(cropped.format(), ImageFormat::Rgba);
            prop_assert_eq!(cropped.is_premultiplied(), img.is_premultiplied());
            prop_assert_eq!(cropped.allocation(), img.allocation());

            for y in 0..out_h {
                for x in 0..out_w {
                    prop_assert_eq!(cropped.pixel(x, y), img.pixel(x0 + x, y0 + y));
                }
            }
        }
    }
}
