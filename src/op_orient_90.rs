//! Fast 90-degree rotation and EXIF orientation correction.
//!
//! This module handles exact 90°/180°/270° rotations using simple coordinate
//! remapping. No interpolation is performed, making these operations very fast
//! compared to arbitrary-angle rotation.

use crate::image::Image;

/// EXIF orientation codes for 90-degree rotations and mirrors.
///
/// Values match the EXIF Orientation tag (0x0112). The enum discriminants
/// correspond directly to EXIF codes 1-8.
///
/// ```text
/// Up (1)           UpMirrored (2)    Down (3)         DownMirrored (4)
/// ┌───────┐        ┌───────┐         ┌───────┐        ┌───────┐
/// │ 1   2 │        │ 2   1 │         │ 4   3 │        │ 3   4 │
/// │       │        │       │         │       │        │       │
/// │ 3   4 │        │ 4   3 │         │ 2   1 │        │ 1   2 │
/// └───────┘        └───────┘         └───────┘        └───────┘
///
/// LeftMirrored (5) Right (6)         RightMirrored (7) Left (8)
/// ┌───────┐        ┌───────┐         ┌───────┐         ┌───────┐
/// │ 1   3 │        │ 3   1 │         │ 4   2 │         │ 2   4 │
/// │       │        │       │         │       │         │       │
/// │ 2   4 │        │ 4   2 │         │ 3   1 │         │ 1   3 │
/// └───────┘        └───────┘         └───────┘         └───────┘
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Orientation90 {
    Up = 1,
    UpMirrored = 2,
    Down = 3,
    DownMirrored = 4,
    LeftMirrored = 5,
    Right = 6,
    RightMirrored = 7,
    Left = 8,
}

/// Fast 90-degree rotation operator using coordinate remapping.
///
/// This operator performs exact 90°/180°/270° rotations and mirror operations
/// by simply remapping pixel coordinates. No interpolation is needed, making
/// this significantly faster than arbitrary-angle rotation (typically 10-50x).
#[derive(Copy, Clone, Debug)]
pub struct OpOrient90Increments {
    orientation: Orientation90,
}

impl Default for OpOrient90Increments {
    fn default() -> Self {
        Self {
            orientation: Orientation90::Up,
        }
    }
}

impl OpOrient90Increments {
    pub fn new(orientation: Orientation90) -> Self {
        Self { orientation }
    }

    pub fn set_orientation(&mut self, orientation: Orientation90) -> &mut Self {
        self.orientation = orientation;
        self
    }

    pub fn get_orientation_from_exif_code(code: usize) -> Option<Orientation90> {
        match code {
            1 => Some(Orientation90::Up),
            2 => Some(Orientation90::UpMirrored),
            3 => Some(Orientation90::Down),
            4 => Some(Orientation90::DownMirrored),
            5 => Some(Orientation90::LeftMirrored),
            6 => Some(Orientation90::Right),
            7 => Some(Orientation90::RightMirrored),
            8 => Some(Orientation90::Left),
            _ => None,
        }
    }

    pub fn apply(&self, original: &Image) -> Image {
        if self.orientation == Orientation90::Up {
            return original.clone();
        }

        let (out_w, out_h) =
            output_dimensions(original.width(), original.height(), self.orientation);
        let mut result =
            Image::with_premultiplied(out_w, out_h, original.format(), original.is_premultiplied());
        self.apply_to_preallocated(original, &mut result);
        result
    }

    pub fn apply_to_preallocated(&self, original: &Image, dst: &mut Image) {
        if self.orientation == Orientation90::Up {
            if original.width() == dst.width()
                && original.height() == dst.height()
                && original.format() == dst.format()
            {
                dst.data.copy_from_slice(&original.data);
            }
            return;
        }

        let channels = original.channels();
        let max_x = dst.width().saturating_sub(1);
        let max_y = dst.height().saturating_sub(1);

        for y in 0..dst.height() {
            for x in 0..dst.width() {
                let (orig_x, orig_y) =
                    get_original_coordinates(x, y, max_x, max_y, self.orientation);
                let src_pixel = original.pixel(orig_x, orig_y);
                let dst_pixel = dst.pixel_mut(x, y);
                dst_pixel.copy_from_slice(&src_pixel[..channels]);
            }
        }
    }
}

fn output_dimensions(width: usize, height: usize, orientation: Orientation90) -> (usize, usize) {
    match orientation {
        Orientation90::Up
        | Orientation90::UpMirrored
        | Orientation90::Down
        | Orientation90::DownMirrored => (width, height),
        _ => (height, width),
    }
}

fn get_original_coordinates(
    x: usize,
    y: usize,
    max_x: usize,
    max_y: usize,
    orientation: Orientation90,
) -> (usize, usize) {
    match orientation {
        Orientation90::Up => (x, y),
        Orientation90::UpMirrored => (max_x - x, y),
        Orientation90::Down => (max_x - x, max_y - y),
        Orientation90::DownMirrored => (x, max_y - y),
        Orientation90::LeftMirrored => (y, x),
        Orientation90::Right => (y, max_x - x),
        Orientation90::RightMirrored => (max_y - y, max_x - x),
        Orientation90::Left => (max_y - y, x),
    }
}
