//! Arbitrary-angle image rotation with bicubic interpolation.
//!
//! # Algorithm Overview
//!
//! 1. **Fast-path detection**: Exact 90/180/270 degree angles delegate to
//!    [`OpOrient90Increments`] for simple coordinate remapping.
//!
//! 2. **Bounds + inverse mapping**: Rotated source corners define output size,
//!    and output pixels are mapped back into source space around the image center
//!    to avoid holes.
//!
//! 3. **Tiled processing**: The output is processed in 64x64 tiles (`K_TILE`)
//!    to improve cache locality.
//!
//! 4. **Scanline clipping**: For each output row, polygon intersection determines
//!    the x-range that maps to valid source pixels, skipping background-only regions.
//!
//! 5. **Interior/border split**: Pixels fully inside the source image use an
//!    optimized path that skips bounds checking. Border pixels use a slower path
//!    with boundary handling.
//!
//! 6. **Bicubic interpolation**: Uses Catmull-Rom-style cubic weights over a
//!    4x4 pixel neighborhood for smooth results.
//!
//! On aarch64 with NEON, the RGBA interior path uses a vectorized kernel; other
//! platforms use the scalar implementation.

use crate::image::{Color, Image, MAX_VALUE, Sample, convert_u8_to_sample};
use crate::op_orient_90::{OpOrient90Increments, Orientation90};

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use core::arch::aarch64::{
    float32x4_t, vcvtq_f32_u32, vcvtq_u32_f32, vdupq_n_f32, vld1_u16, vmaxq_f32, vminq_f32,
    vmlaq_n_f32, vmovl_u16, vmovn_u32, vmulq_n_f32, vst1_u16,
};

const ERROR_RANGE: f32 = 10e-5_f32;

/// Tile size for cache-efficient processing.
///
/// Processing in 64x64 tiles keeps the working set small for the 4x4 kernel
/// while limiting per-tile overhead.
const K_TILE: usize = 64;

/// Halo size for bicubic interpolation at image edges.
///
/// Bicubic interpolation requires a 4×4 pixel neighborhood (2 pixels in each
/// direction from the center). This constant defines the padding needed to
/// handle edge pixels that would otherwise sample outside the image bounds.
const K_HALO: f32 = 2.0;

/// Epsilon for interior pixel detection.
///
/// Pixels within this distance of the source image boundary are treated as
/// border pixels requiring bounds checking. A small epsilon avoids floating
/// point edge cases.
const K_INTERIOR_EPSILON: f32 = 1e-3;

#[derive(Clone, Copy)]
struct SpanTransform {
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    half_width_offset: f32,
    original_width_offset: f32,
    original_height_offset: f32,
}

#[derive(Clone, Copy)]
struct SpanContext<'a> {
    row_offset: usize,
    channels: usize,
    cy: f32,
    transform: &'a SpanTransform,
}

#[derive(Clone, Debug, Default)]
struct RotateScratch {
    row_x_start: Vec<i32>,
    row_x_end: Vec<i32>,
    row_inner_start: Vec<i32>,
    row_inner_end: Vec<i32>,
}

impl RotateScratch {
    fn reserve(&mut self, len: usize) {
        if self.row_x_start.len() < len {
            self.row_x_start.resize(len, -1);
            self.row_x_end.resize(len, -1);
            self.row_inner_start.resize(len, -1);
            self.row_inner_end.resize(len, -1);
        }
    }

    fn prepare(&mut self, len: usize) {
        self.reserve(len);
        self.row_x_start[..len].fill(-1);
        self.row_x_end[..len].fill(-1);
        self.row_inner_start[..len].fill(-1);
        self.row_inner_end[..len].fill(-1);
    }
}

/// Direction of rotation.
///
/// - `Cw`: Clockwise rotation (positive angle rotates top toward right)
/// - `Ccw`: Counter-clockwise rotation (positive angle rotates top toward left)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RotateDirection {
    Cw,
    Ccw,
}

/// Arbitrary-angle image rotation with automatic fast-path optimization.
///
/// # Fast Path
///
/// When the rotation angle is an exact multiple of 90 degrees (0°, 90°, 180°, 270°),
/// the operator automatically delegates to [`OpOrient90Increments`] for maximum
/// performance via simple coordinate remapping.
///
/// # Background Color
///
/// Pixels outside the rotated source image are filled with a background color.
/// Default: white (RGB=65535) with transparent alpha (A=0) for alpha formats,
/// or white with opaque alpha for non-alpha formats. Use [`set_background_color`]
/// to customize.
///
/// [`set_background_color`]: OpRotateImage::set_background_color
#[derive(Clone, Debug)]
pub struct OpRotateImage {
    angle: f32,
    direction: RotateDirection,
    background_samples: [Sample; 4],
    background_externally_set: bool,
    use_op_orient: bool,
    orientation: Orientation90,
    scratch: RotateScratch,
}

impl Default for OpRotateImage {
    fn default() -> Self {
        Self::new()
    }
}

impl OpRotateImage {
    pub fn new() -> Self {
        Self {
            angle: 0.0,
            direction: RotateDirection::Cw,
            background_samples: [MAX_VALUE, MAX_VALUE, MAX_VALUE, 0],
            background_externally_set: false,
            use_op_orient: false,
            orientation: Orientation90::Up,
            scratch: RotateScratch::default(),
        }
    }

    /// Sets the rotation angle and direction.
    ///
    /// The angle is normalized to the range (-180°, 180°]. When the normalized
    /// angle is an exact multiple of 90°, enables the fast path that bypasses
    /// interpolation entirely.
    pub fn set_rotation(&mut self, angle_degrees: f32, direction: RotateDirection) -> &mut Self {
        let mut angle = self.normalize_angle(angle_degrees);

        self.direction = direction;
        if self.direction == RotateDirection::Ccw {
            angle = -angle;
        }

        let angle_as_int = angle as i32;
        if angle_as_int % 90 == 0 {
            self.use_op_orient = true;
            self.orientation = match angle_as_int {
                90 => Orientation90::Right,
                180 | -180 => Orientation90::Down,
                -90 => Orientation90::Left,
                _ => Orientation90::Up,
            };
        } else {
            self.use_op_orient = false;
        }

        angle = angle * std::f32::consts::PI / 180.0;
        self.angle = angle;
        self
    }

    /// Sets the background color for areas outside the rotated image.
    ///
    /// If not called, the default is white with transparent alpha (for alpha
    /// formats) or white with opaque alpha (for non-alpha formats).
    pub fn set_background_color(&mut self, r: u8, g: u8, b: u8, a: u8) -> &mut Self {
        self.background_samples = convert_u8_to_sample(r, g, b, a);
        self.background_externally_set = true;
        self
    }

    /// Reserves internal scratch buffers to avoid per-call allocations.
    ///
    /// Call after computing output dimensions if you want to keep the hot path allocation-free.
    pub fn reserve_scratch(&mut self, output_height: usize) -> &mut Self {
        self.scratch.reserve(output_height);
        self
    }

    pub fn apply(&mut self, img: &Image) -> Image {
        if self.use_op_orient {
            return OpOrient90Increments::new(self.orientation).apply(img);
        }

        if !self.background_externally_set {
            if img.format().has_alpha() {
                self.background_samples[3] = 0;
            } else {
                self.background_samples[3] = MAX_VALUE;
            }
        }

        let background_color = Color::new(self.background_samples, img.format());

        let tf = self
            .get_transformation_matrix(self.angle)
            .expect("Error creating transformation matrix");

        let (new_w, new_h) = self.find_new_image_size(img.width(), img.height(), &tf);
        let mut new_image =
            Image::with_premultiplied(new_w, new_h, img.format(), img.is_premultiplied());
        new_image.fill_with_color(&background_color);

        self.process_pixels(img, &mut new_image, &tf);

        let mod_val = self.angle % (std::f32::consts::PI / 4.0);
        // For 45-degree multiples, rounding can add a 1px border; trim it.
        if mod_val.abs() < ERROR_RANGE && new_image.width() > 2 && new_image.height() > 2 {
            new_image.crop(1, 1, new_image.width() - 2, new_image.height() - 2)
        } else {
            new_image
        }
    }

    pub fn compute_output_dimensions(&self, src: &Image) -> (usize, usize) {
        if self.use_op_orient {
            match self.orientation {
                Orientation90::Right
                | Orientation90::Left
                | Orientation90::LeftMirrored
                | Orientation90::RightMirrored => (src.height(), src.width()),
                _ => (src.width(), src.height()),
            }
        } else {
            let tf = self
                .get_transformation_matrix(self.angle)
                .expect("Error creating transformation matrix");
            self.find_new_image_size(src.width(), src.height(), &tf)
        }
    }

    pub fn apply_to_preallocated(&mut self, src: &Image, dst: &mut Image) {
        if self.use_op_orient {
            OpOrient90Increments::new(self.orientation).apply_to_preallocated(src, dst);
            return;
        }

        if !self.background_externally_set {
            if src.format().has_alpha() {
                self.background_samples[3] = 0;
            } else {
                self.background_samples[3] = MAX_VALUE;
            }
        }

        let background_color = Color::new(self.background_samples, src.format());
        dst.fill_with_color(&background_color);

        let tf = self
            .get_transformation_matrix(self.angle)
            .expect("Error creating transformation matrix");
        self.process_pixels(src, dst, &tf);
    }

    fn normalize_angle(&self, angle_degrees: f32) -> f32 {
        angle_degrees - (angle_degrees / 360.0 - 0.5).ceil() * 360.0
    }

    fn get_transformation_matrix(&self, angle: f32) -> Option<[f32; 9]> {
        let matrix = [
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        ];

        let determinant = (matrix[0] * matrix[4]) - (matrix[1] * matrix[3]);
        if (determinant - 1.0).abs() > ERROR_RANGE {
            return None;
        }
        Some(matrix)
    }

    fn inverse_matrix(&self, tf: &[f32; 9]) -> [f32; 9] {
        let mut inv = *tf;
        inv.swap(1, 3);
        inv
    }

    fn find_new_image_size(&self, width: usize, height: usize, tf: &[f32; 9]) -> (usize, usize) {
        let corners = [
            Point3 { x: 0, y: 0, z: 1 },
            Point3 {
                x: (width as i32) - 1,
                y: 0,
                z: 1,
            },
            Point3 {
                x: (width as i32) - 1,
                y: (height as i32) - 1,
                z: 1,
            },
            Point3 {
                x: 0,
                y: (height as i32) - 1,
                z: 1,
            },
        ];

        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;

        for corner in &corners {
            let new_corner = self.find_new_corner(tf, corner, width, height);
            min_x = min_x.min(new_corner.x);
            min_y = min_y.min(new_corner.y);
            max_x = max_x.max(new_corner.x);
            max_y = max_y.max(new_corner.y);
        }

        let new_w = ((max_x - min_x + 1) as f32).round() as usize;
        let new_h = ((max_y - min_y + 1) as f32).round() as usize;
        (new_w, new_h)
    }

    fn find_new_corner(
        &self,
        tf: &[f32; 9],
        corner: &Point3,
        width: usize,
        height: usize,
    ) -> Point3 {
        let cx = corner.x as f32 - (width as f32) * 0.5;
        let cy = corner.y as f32 - (height as f32) * 0.5;

        let new_x = tf[0] * cx + tf[1] * cy + tf[2];
        let new_y = tf[3] * cx + tf[4] * cy + tf[5];

        Point3 {
            x: round_to_int(new_x),
            y: round_to_int(new_y),
            z: 1,
        }
    }

    fn process_pixels(&mut self, src: &Image, dst: &mut Image, tf: &[f32; 9]) {
        let new_width = dst.width();
        let new_height = dst.height();
        let channels = dst.channels();
        let dst_stride = dst.stride();

        let inv = self.inverse_matrix(tf);

        let m00 = inv[0];
        let m01 = inv[1];
        let m02 = inv[2];
        let m10 = inv[3];
        let m11 = inv[4];
        let m12 = inv[5];

        let half_width_offset = 1.0 - (new_width as f32) / 2.0;
        let half_height_offset = 1.0 - (new_height as f32) / 2.0;
        let original_width_offset = (src.width() as f32) / 2.0;
        let original_height_offset = (src.height() as f32) / 2.0;
        let transform = SpanTransform {
            m00,
            m01,
            m02,
            m10,
            m11,
            m12,
            half_width_offset,
            original_width_offset,
            original_height_offset,
        };

        let src_min_x = -K_HALO;
        let src_min_y = -K_HALO;
        let src_max_x = src.width() as f32 - 1.0 + K_HALO;
        let src_max_y = src.height() as f32 - 1.0 + K_HALO;

        let interior_min_x = 1.0 + K_INTERIOR_EPSILON;
        let interior_min_y = 1.0 + K_INTERIOR_EPSILON;
        let interior_max_x = src.width() as f32 - 2.0 - K_INTERIOR_EPSILON;
        let interior_max_y = src.height() as f32 - 2.0 - K_INTERIOR_EPSILON;

        let to_output = |ox: f32, oy: f32| -> Point2f {
            let cx = ox - original_width_offset;
            let cy = oy - original_height_offset;
            let nx = tf[0] * cx + tf[1] * cy + tf[2];
            let ny = tf[3] * cx + tf[4] * cy + tf[5];
            Point2f {
                x: nx - half_width_offset,
                y: ny - half_height_offset,
            }
        };

        let outer_poly = [
            to_output(src_min_x, src_min_y),
            to_output(src_max_x, src_min_y),
            to_output(src_max_x, src_max_y),
            to_output(src_min_x, src_max_y),
        ];

        let new_width_i = new_width as i32;
        self.scratch.prepare(new_height);
        {
            let RotateScratch {
                row_x_start,
                row_x_end,
                row_inner_start,
                row_inner_end,
            } = &mut self.scratch;
            let row_x_start = &mut row_x_start[..new_height];
            let row_x_end = &mut row_x_end[..new_height];
            let row_inner_start = &mut row_inner_start[..new_height];
            let row_inner_end = &mut row_inner_end[..new_height];

            for new_y in 0..new_height {
                let y = new_y as f32;
                if let Some((x0, x1)) = scanline_range(&outer_poly, y, new_width_i, true) {
                    row_x_start[new_y] = x0;
                    row_x_end[new_y] = x1;
                }
            }

            if interior_max_x >= interior_min_x && interior_max_y >= interior_min_y {
                let inner_poly = [
                    to_output(interior_min_x, interior_min_y),
                    to_output(interior_max_x, interior_min_y),
                    to_output(interior_max_x, interior_max_y),
                    to_output(interior_min_x, interior_max_y),
                ];

                for new_y in 0..new_height {
                    let y = new_y as f32;
                    if let Some((x0, x1)) = scanline_range(&inner_poly, y, new_width_i, false) {
                        row_inner_start[new_y] = x0;
                        row_inner_end[new_y] = x1;
                    }
                }
            }
        }

        let row_x_start = &self.scratch.row_x_start[..new_height];
        let row_x_end = &self.scratch.row_x_end[..new_height];
        let row_inner_start = &self.scratch.row_inner_start[..new_height];
        let row_inner_end = &self.scratch.row_inner_end[..new_height];

        let dst_data = dst.data.as_mut_slice();

        for tile_y in (0..new_height).step_by(K_TILE) {
            let y_end = (tile_y + K_TILE).min(new_height);
            let mut row_offsets = [0usize; K_TILE];
            for (idx, new_y) in (tile_y..y_end).enumerate() {
                row_offsets[idx] = new_y * dst_stride;
            }

            for tile_x in (0..new_width).step_by(K_TILE) {
                let tile_x_end = (tile_x + K_TILE).min(new_width);
                for new_y in tile_y..y_end {
                    let row_start = row_x_start[new_y];
                    if row_start < 0 {
                        continue;
                    }
                    let row_end = row_x_end[new_y];
                    let x_start = tile_x.max(row_start as usize);
                    let x_end = tile_x_end.min((row_end + 1) as usize);
                    if x_start >= x_end {
                        continue;
                    }

                    let row_offset = row_offsets[new_y - tile_y];
                    let cy = new_y as f32 + half_height_offset;
                    let span_context = SpanContext {
                        row_offset,
                        channels,
                        cy,
                        transform: &transform,
                    };

                    let inner_start = row_inner_start[new_y];
                    if inner_start >= 0 {
                        let inner_end = row_inner_end[new_y];
                        let inner_x_start = x_start.max(inner_start as usize);
                        let inner_x_end = x_end.min((inner_end + 1) as usize);
                        if inner_x_start < inner_x_end {
                            self.process_border_span(
                                src,
                                dst_data,
                                x_start..inner_x_start,
                                span_context,
                            );
                            self.process_interior_span(
                                src,
                                dst_data,
                                inner_x_start..inner_x_end,
                                span_context,
                            );
                            self.process_border_span(
                                src,
                                dst_data,
                                inner_x_end..x_end,
                                span_context,
                            );
                        } else {
                            self.process_border_span(src, dst_data, x_start..x_end, span_context);
                        }
                    } else {
                        self.process_border_span(src, dst_data, x_start..x_end, span_context);
                    }
                }
            }
        }
    }

    fn process_border_span(
        &self,
        src: &Image,
        dst_data: &mut [Sample],
        span: std::ops::Range<usize>,
        context: SpanContext<'_>,
    ) {
        let SpanContext {
            row_offset,
            channels,
            cy,
            transform,
        } = context;
        let span_start = span.start;
        let span_end = span.end;
        if span_start >= span_end {
            return;
        }
        let SpanTransform {
            m00,
            m01,
            m02,
            m10,
            m11,
            m12,
            half_width_offset,
            original_width_offset,
            original_height_offset,
        } = *transform;
        let cx_start = span_start as f32 + half_width_offset;
        let mut old_x = m00 * cx_start + m01 * cy + m02 + original_width_offset;
        let mut old_y = m10 * cx_start + m11 * cy + m12 + original_height_offset;
        if channels == 4 {
            let mut dst_ptr = unsafe { dst_data.as_mut_ptr().add(row_offset + span_start * 4) };
            for _ in span_start..span_end {
                self.paint_image_cubic_rgba_ptr(src, old_x, old_y, dst_ptr);
                dst_ptr = unsafe { dst_ptr.add(4) };
                old_x += m00;
                old_y += m10;
            }
            return;
        }

        let mut dst_index = row_offset + span_start * channels;
        for _ in span_start..span_end {
            let dst_pixel = &mut dst_data[dst_index..dst_index + channels];
            self.paint_image_cubic(src, old_x, old_y, dst_pixel);
            dst_index += channels;
            old_x += m00;
            old_y += m10;
        }
    }

    fn process_interior_span(
        &self,
        src: &Image,
        dst_data: &mut [Sample],
        span: std::ops::Range<usize>,
        context: SpanContext<'_>,
    ) {
        let SpanContext {
            row_offset,
            channels,
            cy,
            transform,
        } = context;
        let span_start = span.start;
        let span_end = span.end;
        if span_start >= span_end {
            return;
        }
        let SpanTransform {
            m00,
            m01,
            m02,
            m10,
            m11,
            m12,
            half_width_offset,
            original_width_offset,
            original_height_offset,
        } = *transform;
        let cx_start = span_start as f32 + half_width_offset;
        let mut old_x = m00 * cx_start + m01 * cy + m02 + original_width_offset;
        let mut old_y = m10 * cx_start + m11 * cy + m12 + original_height_offset;
        if channels == 4 {
            let mut dst_ptr = unsafe { dst_data.as_mut_ptr().add(row_offset + span_start * 4) };
            for _ in span_start..span_end {
                self.paint_image_cubic_interior_rgba_ptr(src, old_x, old_y, dst_ptr);
                dst_ptr = unsafe { dst_ptr.add(4) };
                old_x += m00;
                old_y += m10;
            }
            return;
        }

        let mut dst_index = row_offset + span_start * channels;
        for _ in span_start..span_end {
            let dst_pixel = &mut dst_data[dst_index..dst_index + channels];
            self.paint_image_cubic_interior(src, old_x, old_y, dst_pixel);
            dst_index += channels;
            old_x += m00;
            old_y += m10;
        }
    }

    #[inline(always)]
    fn paint_image_cubic_interior_rgba(
        &self,
        src: &Image,
        orig_x: f32,
        orig_y: f32,
        dst_pixel: &mut [Sample],
    ) {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            self.paint_image_cubic_interior_rgba_neon(src, orig_x, orig_y, dst_pixel);
        }

        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            self.paint_image_cubic_interior_rgba_scalar(src, orig_x, orig_y, dst_pixel);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    fn paint_image_cubic_interior_rgba_scalar(
        &self,
        src: &Image,
        orig_x: f32,
        orig_y: f32,
        dst_pixel: &mut [Sample],
    ) {
        let pc_x = orig_x as i32;
        let pc_y = orig_y as i32;

        let wx = get_cubic_weights(orig_x - pc_x as f32);
        let wy = get_cubic_weights(orig_y - pc_y as f32);

        let start_x = pc_x - 1;
        let start_y = pc_y - 1;
        let stride = src.stride();
        let row0 = (start_y as usize) * stride + (start_x as usize) * 4;
        let row1 = row0 + stride;
        let row2 = row1 + stride;
        let row3 = row2 + stride;

        let max_val = MAX_VALUE as f32;
        let data = &src.data;

        let mut results = [0.0_f32; 4];
        for c in 0..4 {
            let col0 = data[row0 + c] as f32 * wx[0]
                + data[row0 + c + 4] as f32 * wx[1]
                + data[row0 + c + 8] as f32 * wx[2]
                + data[row0 + c + 12] as f32 * wx[3];
            let col1 = data[row1 + c] as f32 * wx[0]
                + data[row1 + c + 4] as f32 * wx[1]
                + data[row1 + c + 8] as f32 * wx[2]
                + data[row1 + c + 12] as f32 * wx[3];
            let col2 = data[row2 + c] as f32 * wx[0]
                + data[row2 + c + 4] as f32 * wx[1]
                + data[row2 + c + 8] as f32 * wx[2]
                + data[row2 + c + 12] as f32 * wx[3];
            let col3 = data[row3 + c] as f32 * wx[0]
                + data[row3 + c + 4] as f32 * wx[1]
                + data[row3 + c + 8] as f32 * wx[2]
                + data[row3 + c + 12] as f32 * wx[3];

            let mut final_val = col0 * wy[0] + col1 * wy[1] + col2 * wy[2] + col3 * wy[3];
            if final_val < 0.0 {
                final_val = 0.0;
            }
            if final_val > max_val {
                final_val = max_val;
            }
            results[c] = final_val;
        }

        dst_pixel[0] = results[0] as Sample;
        dst_pixel[1] = results[1] as Sample;
        dst_pixel[2] = results[2] as Sample;
        dst_pixel[3] = results[3] as Sample;
    }

    #[inline(always)]
    fn paint_image_cubic_interior_rgba_ptr(
        &self,
        src: &Image,
        orig_x: f32,
        orig_y: f32,
        dst_ptr: *mut Sample,
    ) {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            self.paint_image_cubic_interior_rgba_neon_ptr(src, orig_x, orig_y, dst_ptr);
        }

        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            self.paint_image_cubic_interior_rgba_scalar_ptr(src, orig_x, orig_y, dst_ptr);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    #[inline(always)]
    fn paint_image_cubic_interior_rgba_scalar_ptr(
        &self,
        src: &Image,
        orig_x: f32,
        orig_y: f32,
        dst_ptr: *mut Sample,
    ) {
        let pc_x = orig_x as i32;
        let pc_y = orig_y as i32;

        let wx = get_cubic_weights(orig_x - pc_x as f32);
        let wy = get_cubic_weights(orig_y - pc_y as f32);

        let start_x = pc_x - 1;
        let start_y = pc_y - 1;
        let stride = src.stride();
        let row0 = (start_y as usize) * stride + (start_x as usize) * 4;
        let row1 = row0 + stride;
        let row2 = row1 + stride;
        let row3 = row2 + stride;

        let max_val = MAX_VALUE as f32;
        let data = &src.data;

        let mut results = [0.0_f32; 4];
        for c in 0..4 {
            let col0 = data[row0 + c] as f32 * wx[0]
                + data[row0 + c + 4] as f32 * wx[1]
                + data[row0 + c + 8] as f32 * wx[2]
                + data[row0 + c + 12] as f32 * wx[3];
            let col1 = data[row1 + c] as f32 * wx[0]
                + data[row1 + c + 4] as f32 * wx[1]
                + data[row1 + c + 8] as f32 * wx[2]
                + data[row1 + c + 12] as f32 * wx[3];
            let col2 = data[row2 + c] as f32 * wx[0]
                + data[row2 + c + 4] as f32 * wx[1]
                + data[row2 + c + 8] as f32 * wx[2]
                + data[row2 + c + 12] as f32 * wx[3];
            let col3 = data[row3 + c] as f32 * wx[0]
                + data[row3 + c + 4] as f32 * wx[1]
                + data[row3 + c + 8] as f32 * wx[2]
                + data[row3 + c + 12] as f32 * wx[3];

            let mut final_val = col0 * wy[0] + col1 * wy[1] + col2 * wy[2] + col3 * wy[3];
            if final_val < 0.0 {
                final_val = 0.0;
            }
            if final_val > max_val {
                final_val = max_val;
            }
            results[c] = final_val;
        }

        unsafe {
            *dst_ptr.add(0) = results[0] as Sample;
            *dst_ptr.add(1) = results[1] as Sample;
            *dst_ptr.add(2) = results[2] as Sample;
            *dst_ptr.add(3) = results[3] as Sample;
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    unsafe fn paint_image_cubic_interior_rgba_neon(
        &self,
        src: &Image,
        orig_x: f32,
        orig_y: f32,
        dst_pixel: &mut [Sample],
    ) {
        let pc_x = orig_x as i32;
        let pc_y = orig_y as i32;

        let wx = get_cubic_weights(orig_x - pc_x as f32);
        let wy = get_cubic_weights(orig_y - pc_y as f32);

        let start_x = pc_x - 1;
        let start_y = pc_y - 1;
        let stride = src.stride();
        let row0 = (start_y as usize) * stride + (start_x as usize) * 4;
        let row1 = row0 + stride;
        let row2 = row1 + stride;
        let row3 = row2 + stride;

        unsafe {
            let data = &src.data;
            let col0 = Self::row_cubic_rgba_neon(data, row0, &wx);
            let col1 = Self::row_cubic_rgba_neon(data, row1, &wx);
            let col2 = Self::row_cubic_rgba_neon(data, row2, &wx);
            let col3 = Self::row_cubic_rgba_neon(data, row3, &wx);

            let mut final_val = vmulq_n_f32(col0, wy[0]);
            final_val = vmlaq_n_f32(final_val, col1, wy[1]);
            final_val = vmlaq_n_f32(final_val, col2, wy[2]);
            final_val = vmlaq_n_f32(final_val, col3, wy[3]);

            let zero = vdupq_n_f32(0.0);
            let max_val = vdupq_n_f32(MAX_VALUE as f32);
            final_val = vmaxq_f32(final_val, zero);
            final_val = vminq_f32(final_val, max_val);

            let out_u32 = vcvtq_u32_f32(final_val);
            let out_u16 = vmovn_u32(out_u32);
            vst1_u16(dst_pixel.as_mut_ptr(), out_u16);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    unsafe fn paint_image_cubic_interior_rgba_neon_ptr(
        &self,
        src: &Image,
        orig_x: f32,
        orig_y: f32,
        dst_ptr: *mut Sample,
    ) {
        let pc_x = orig_x as i32;
        let pc_y = orig_y as i32;

        let wx = get_cubic_weights(orig_x - pc_x as f32);
        let wy = get_cubic_weights(orig_y - pc_y as f32);

        let start_x = pc_x - 1;
        let start_y = pc_y - 1;
        let stride = src.stride();
        let row0 = (start_y as usize) * stride + (start_x as usize) * 4;
        let row1 = row0 + stride;
        let row2 = row1 + stride;
        let row3 = row2 + stride;

        unsafe {
            let data = &src.data;
            let col0 = Self::row_cubic_rgba_neon(data, row0, &wx);
            let col1 = Self::row_cubic_rgba_neon(data, row1, &wx);
            let col2 = Self::row_cubic_rgba_neon(data, row2, &wx);
            let col3 = Self::row_cubic_rgba_neon(data, row3, &wx);

            let mut final_val = vmulq_n_f32(col0, wy[0]);
            final_val = vmlaq_n_f32(final_val, col1, wy[1]);
            final_val = vmlaq_n_f32(final_val, col2, wy[2]);
            final_val = vmlaq_n_f32(final_val, col3, wy[3]);

            let zero = vdupq_n_f32(0.0);
            let max_val = vdupq_n_f32(MAX_VALUE as f32);
            final_val = vmaxq_f32(final_val, zero);
            final_val = vminq_f32(final_val, max_val);

            let out_u32 = vcvtq_u32_f32(final_val);
            let out_u16 = vmovn_u32(out_u32);
            vst1_u16(dst_ptr, out_u16);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    unsafe fn row_cubic_rgba_neon(data: &[Sample], row_base: usize, wx: &[f32; 4]) -> float32x4_t {
        unsafe {
            let p0 = Self::load_rgba_f32x4_neon(data, row_base);
            let p1 = Self::load_rgba_f32x4_neon(data, row_base + 4);
            let p2 = Self::load_rgba_f32x4_neon(data, row_base + 8);
            let p3 = Self::load_rgba_f32x4_neon(data, row_base + 12);

            let mut accum = vmulq_n_f32(p0, wx[0]);
            accum = vmlaq_n_f32(accum, p1, wx[1]);
            accum = vmlaq_n_f32(accum, p2, wx[2]);
            vmlaq_n_f32(accum, p3, wx[3])
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    unsafe fn load_rgba_f32x4_neon(data: &[Sample], index: usize) -> float32x4_t {
        unsafe {
            let samples_u16 = vld1_u16(data.as_ptr().add(index));
            let samples_u32 = vmovl_u16(samples_u16);
            vcvtq_f32_u32(samples_u32)
        }
    }

    #[inline(always)]
    fn paint_image_cubic_interior(
        &self,
        src: &Image,
        orig_x: f32,
        orig_y: f32,
        dst_pixel: &mut [Sample],
    ) {
        let pc_x = orig_x as i32;
        let pc_y = orig_y as i32;

        let wx = get_cubic_weights(orig_x - pc_x as f32);
        let wy = get_cubic_weights(orig_y - pc_y as f32);

        let channels = dst_pixel.len();
        let start_x = pc_x - 1;
        let start_y = pc_y - 1;
        let stride = src.stride();
        let max_val = MAX_VALUE as f32;
        let data = &src.data;

        match channels {
            4 => {
                let row0 = (start_y as usize) * stride + (start_x as usize) * 4;
                let row1 = row0 + stride;
                let row2 = row1 + stride;
                let row3 = row2 + stride;
                let mut results = [0.0_f32; 4];
                for c in 0..4 {
                    let col0 = data[row0 + c] as f32 * wx[0]
                        + data[row0 + c + 4] as f32 * wx[1]
                        + data[row0 + c + 8] as f32 * wx[2]
                        + data[row0 + c + 12] as f32 * wx[3];
                    let col1 = data[row1 + c] as f32 * wx[0]
                        + data[row1 + c + 4] as f32 * wx[1]
                        + data[row1 + c + 8] as f32 * wx[2]
                        + data[row1 + c + 12] as f32 * wx[3];
                    let col2 = data[row2 + c] as f32 * wx[0]
                        + data[row2 + c + 4] as f32 * wx[1]
                        + data[row2 + c + 8] as f32 * wx[2]
                        + data[row2 + c + 12] as f32 * wx[3];
                    let col3 = data[row3 + c] as f32 * wx[0]
                        + data[row3 + c + 4] as f32 * wx[1]
                        + data[row3 + c + 8] as f32 * wx[2]
                        + data[row3 + c + 12] as f32 * wx[3];

                    let mut final_val = col0 * wy[0] + col1 * wy[1] + col2 * wy[2] + col3 * wy[3];
                    if final_val < 0.0 {
                        final_val = 0.0;
                    }
                    if final_val > max_val {
                        final_val = max_val;
                    }
                    results[c] = final_val;
                }
                dst_pixel[0] = results[0] as Sample;
                dst_pixel[1] = results[1] as Sample;
                dst_pixel[2] = results[2] as Sample;
                dst_pixel[3] = results[3] as Sample;
            }
            3 => {
                let row0 = (start_y as usize) * stride + (start_x as usize) * 3;
                let row1 = row0 + stride;
                let row2 = row1 + stride;
                let row3 = row2 + stride;
                let mut results = [0.0_f32; 3];
                for c in 0..3 {
                    let col0 = data[row0 + c] as f32 * wx[0]
                        + data[row0 + c + 3] as f32 * wx[1]
                        + data[row0 + c + 6] as f32 * wx[2]
                        + data[row0 + c + 9] as f32 * wx[3];
                    let col1 = data[row1 + c] as f32 * wx[0]
                        + data[row1 + c + 3] as f32 * wx[1]
                        + data[row1 + c + 6] as f32 * wx[2]
                        + data[row1 + c + 9] as f32 * wx[3];
                    let col2 = data[row2 + c] as f32 * wx[0]
                        + data[row2 + c + 3] as f32 * wx[1]
                        + data[row2 + c + 6] as f32 * wx[2]
                        + data[row2 + c + 9] as f32 * wx[3];
                    let col3 = data[row3 + c] as f32 * wx[0]
                        + data[row3 + c + 3] as f32 * wx[1]
                        + data[row3 + c + 6] as f32 * wx[2]
                        + data[row3 + c + 9] as f32 * wx[3];

                    let mut final_val = col0 * wy[0] + col1 * wy[1] + col2 * wy[2] + col3 * wy[3];
                    if final_val < 0.0 {
                        final_val = 0.0;
                    }
                    if final_val > max_val {
                        final_val = max_val;
                    }
                    results[c] = final_val;
                }
                dst_pixel[0] = results[0] as Sample;
                dst_pixel[1] = results[1] as Sample;
                dst_pixel[2] = results[2] as Sample;
            }
            _ => {
                let stride_channels = channels;
                let row0 = (start_y as usize) * stride + (start_x as usize) * stride_channels;
                let row1 = row0 + stride;
                let row2 = row1 + stride;
                let row3 = row2 + stride;
                for c in 0..channels {
                    let col0 = data[row0 + c] as f32 * wx[0]
                        + data[row0 + c + stride_channels] as f32 * wx[1]
                        + data[row0 + c + 2 * stride_channels] as f32 * wx[2]
                        + data[row0 + c + 3 * stride_channels] as f32 * wx[3];
                    let col1 = data[row1 + c] as f32 * wx[0]
                        + data[row1 + c + stride_channels] as f32 * wx[1]
                        + data[row1 + c + 2 * stride_channels] as f32 * wx[2]
                        + data[row1 + c + 3 * stride_channels] as f32 * wx[3];
                    let col2 = data[row2 + c] as f32 * wx[0]
                        + data[row2 + c + stride_channels] as f32 * wx[1]
                        + data[row2 + c + 2 * stride_channels] as f32 * wx[2]
                        + data[row2 + c + 3 * stride_channels] as f32 * wx[3];
                    let col3 = data[row3 + c] as f32 * wx[0]
                        + data[row3 + c + stride_channels] as f32 * wx[1]
                        + data[row3 + c + 2 * stride_channels] as f32 * wx[2]
                        + data[row3 + c + 3 * stride_channels] as f32 * wx[3];

                    let mut final_val = col0 * wy[0] + col1 * wy[1] + col2 * wy[2] + col3 * wy[3];
                    if final_val < 0.0 {
                        final_val = 0.0;
                    }
                    if final_val > max_val {
                        final_val = max_val;
                    }
                    dst_pixel[c] = final_val as Sample;
                }
            }
        }
    }

    #[inline(always)]
    fn paint_image_cubic_rgba_ptr(
        &self,
        src: &Image,
        orig_x: f32,
        orig_y: f32,
        dst_ptr: *mut Sample,
    ) {
        let pc_x = orig_x.floor() as i32;
        let pc_y = orig_y.floor() as i32;

        let w = src.width() as i32;
        let h = src.height() as i32;
        if pc_x < -2 || pc_x > w || pc_y < -2 || pc_y > h {
            return;
        }

        let start_x = pc_x - 1;
        let start_y = pc_y - 1;
        let in_bounds = start_x >= 0 && start_y >= 0 && start_x + 3 < w && start_y + 3 < h;
        if in_bounds {
            self.paint_image_cubic_interior_rgba_ptr(src, orig_x, orig_y, dst_ptr);
            return;
        }

        let wx = get_cubic_weights(orig_x - pc_x as f32);
        let wy = get_cubic_weights(orig_y - pc_y as f32);
        let max_val = MAX_VALUE as f32;
        let patch = self.get_image_patch(src, pc_x, pc_y);

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let patch_ptr = patch.as_ptr() as *const Sample;
            let patch_data = core::slice::from_raw_parts(patch_ptr, 64);
            let col0 = Self::row_cubic_rgba_neon(patch_data, 0, &wx);
            let col1 = Self::row_cubic_rgba_neon(patch_data, 16, &wx);
            let col2 = Self::row_cubic_rgba_neon(patch_data, 32, &wx);
            let col3 = Self::row_cubic_rgba_neon(patch_data, 48, &wx);

            let mut final_val = vmulq_n_f32(col0, wy[0]);
            final_val = vmlaq_n_f32(final_val, col1, wy[1]);
            final_val = vmlaq_n_f32(final_val, col2, wy[2]);
            final_val = vmlaq_n_f32(final_val, col3, wy[3]);

            let zero = vdupq_n_f32(0.0);
            let max_val = vdupq_n_f32(max_val);
            final_val = vmaxq_f32(final_val, zero);
            final_val = vminq_f32(final_val, max_val);

            let out_u32 = vcvtq_u32_f32(final_val);
            let out_u16 = vmovn_u32(out_u32);
            vst1_u16(dst_ptr, out_u16);
        }

        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            let mut results = [0.0_f32; 4];
            for c in 0..4 {
                let col0 = patch[0][0][c] as f32 * wx[0]
                    + patch[0][1][c] as f32 * wx[1]
                    + patch[0][2][c] as f32 * wx[2]
                    + patch[0][3][c] as f32 * wx[3];
                let col1 = patch[1][0][c] as f32 * wx[0]
                    + patch[1][1][c] as f32 * wx[1]
                    + patch[1][2][c] as f32 * wx[2]
                    + patch[1][3][c] as f32 * wx[3];
                let col2 = patch[2][0][c] as f32 * wx[0]
                    + patch[2][1][c] as f32 * wx[1]
                    + patch[2][2][c] as f32 * wx[2]
                    + patch[2][3][c] as f32 * wx[3];
                let col3 = patch[3][0][c] as f32 * wx[0]
                    + patch[3][1][c] as f32 * wx[1]
                    + patch[3][2][c] as f32 * wx[2]
                    + patch[3][3][c] as f32 * wx[3];

                let mut final_val = col0 * wy[0] + col1 * wy[1] + col2 * wy[2] + col3 * wy[3];
                if final_val < 0.0 {
                    final_val = 0.0;
                }
                if final_val > max_val {
                    final_val = max_val;
                }
                results[c] = final_val;
            }

            unsafe {
                *dst_ptr.add(0) = results[0] as Sample;
                *dst_ptr.add(1) = results[1] as Sample;
                *dst_ptr.add(2) = results[2] as Sample;
                *dst_ptr.add(3) = results[3] as Sample;
            }
        }
    }

    #[inline(always)]
    fn paint_image_cubic(&self, src: &Image, orig_x: f32, orig_y: f32, dst_pixel: &mut [Sample]) {
        let pc_x = orig_x.floor() as i32;
        let pc_y = orig_y.floor() as i32;

        let w = src.width() as i32;
        let h = src.height() as i32;
        if pc_x < -2 || pc_x > w || pc_y < -2 || pc_y > h {
            return;
        }

        let start_x = pc_x - 1;
        let start_y = pc_y - 1;
        let in_bounds = start_x >= 0 && start_y >= 0 && start_x + 3 < w && start_y + 3 < h;
        if in_bounds {
            if dst_pixel.len() == 4 {
                self.paint_image_cubic_interior_rgba(src, orig_x, orig_y, dst_pixel);
            } else {
                self.paint_image_cubic_interior(src, orig_x, orig_y, dst_pixel);
            }
            return;
        }

        let wx = get_cubic_weights(orig_x - pc_x as f32);
        let wy = get_cubic_weights(orig_y - pc_y as f32);
        let max_val = MAX_VALUE as f32;

        if dst_pixel.len() == 4 {
            let patch = self.get_image_patch(src, pc_x, pc_y);
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            unsafe {
                let patch_ptr = patch.as_ptr() as *const Sample;
                let patch_data = core::slice::from_raw_parts(patch_ptr, 64);
                let col0 = Self::row_cubic_rgba_neon(patch_data, 0, &wx);
                let col1 = Self::row_cubic_rgba_neon(patch_data, 16, &wx);
                let col2 = Self::row_cubic_rgba_neon(patch_data, 32, &wx);
                let col3 = Self::row_cubic_rgba_neon(patch_data, 48, &wx);

                let mut final_val = vmulq_n_f32(col0, wy[0]);
                final_val = vmlaq_n_f32(final_val, col1, wy[1]);
                final_val = vmlaq_n_f32(final_val, col2, wy[2]);
                final_val = vmlaq_n_f32(final_val, col3, wy[3]);

                let zero = vdupq_n_f32(0.0);
                let max_val = vdupq_n_f32(max_val);
                final_val = vmaxq_f32(final_val, zero);
                final_val = vminq_f32(final_val, max_val);

                let out_u32 = vcvtq_u32_f32(final_val);
                let out_u16 = vmovn_u32(out_u32);
                vst1_u16(dst_pixel.as_mut_ptr(), out_u16);
                return;
            }

            #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
            {
                for (c, dst_value) in dst_pixel.iter_mut().enumerate() {
                    let col0 = patch[0][0][c] as f32 * wx[0]
                        + patch[0][1][c] as f32 * wx[1]
                        + patch[0][2][c] as f32 * wx[2]
                        + patch[0][3][c] as f32 * wx[3];
                    let col1 = patch[1][0][c] as f32 * wx[0]
                        + patch[1][1][c] as f32 * wx[1]
                        + patch[1][2][c] as f32 * wx[2]
                        + patch[1][3][c] as f32 * wx[3];
                    let col2 = patch[2][0][c] as f32 * wx[0]
                        + patch[2][1][c] as f32 * wx[1]
                        + patch[2][2][c] as f32 * wx[2]
                        + patch[2][3][c] as f32 * wx[3];
                    let col3 = patch[3][0][c] as f32 * wx[0]
                        + patch[3][1][c] as f32 * wx[1]
                        + patch[3][2][c] as f32 * wx[2]
                        + patch[3][3][c] as f32 * wx[3];

                    let mut final_val = col0 * wy[0] + col1 * wy[1] + col2 * wy[2] + col3 * wy[3];
                    if final_val < 0.0 {
                        final_val = 0.0;
                    }
                    if final_val > max_val {
                        final_val = max_val;
                    }
                    *dst_value = final_val as Sample;
                }
                return;
            }
        }

        let patch = self.get_image_patch(src, pc_x, pc_y);

        for (c, dst_value) in dst_pixel.iter_mut().enumerate() {
            let col0 = patch[0][0][c] as f32 * wx[0]
                + patch[0][1][c] as f32 * wx[1]
                + patch[0][2][c] as f32 * wx[2]
                + patch[0][3][c] as f32 * wx[3];
            let col1 = patch[1][0][c] as f32 * wx[0]
                + patch[1][1][c] as f32 * wx[1]
                + patch[1][2][c] as f32 * wx[2]
                + patch[1][3][c] as f32 * wx[3];
            let col2 = patch[2][0][c] as f32 * wx[0]
                + patch[2][1][c] as f32 * wx[1]
                + patch[2][2][c] as f32 * wx[2]
                + patch[2][3][c] as f32 * wx[3];
            let col3 = patch[3][0][c] as f32 * wx[0]
                + patch[3][1][c] as f32 * wx[1]
                + patch[3][2][c] as f32 * wx[2]
                + patch[3][3][c] as f32 * wx[3];

            let mut final_val = col0 * wy[0] + col1 * wy[1] + col2 * wy[2] + col3 * wy[3];
            if final_val < 0.0 {
                final_val = 0.0;
            }
            if final_val > max_val {
                final_val = max_val;
            }
            *dst_value = final_val as Sample;
        }
    }

    #[inline(always)]
    fn get_image_patch(&self, image: &Image, x: i32, y: i32) -> [[[Sample; 4]; 4]; 4] {
        let mut patch = [[[0; 4]; 4]; 4];
        let w = image.width() as i32;
        let h = image.height() as i32;
        let channels = image.channels();
        let stride = image.stride();
        let start_x = x - 1;
        let start_y = y - 1;
        let data = &image.data;
        let bg = self.background_samples;

        if start_x >= 0 && start_y >= 0 && start_x + 3 < w && start_y + 3 < h {
            match channels {
                4 => {
                    for r in 0..4 {
                        let row_index = (start_y + r) as usize * stride + (start_x as usize) * 4;
                        for (c, col) in patch[r as usize].iter_mut().enumerate() {
                            let idx = row_index + c * 4;
                            col[0] = data[idx];
                            col[1] = data[idx + 1];
                            col[2] = data[idx + 2];
                            col[3] = data[idx + 3];
                        }
                    }
                }
                3 => {
                    for r in 0..4 {
                        let row_index = (start_y + r) as usize * stride + (start_x as usize) * 3;
                        for (c, col) in patch[r as usize].iter_mut().enumerate() {
                            let idx = row_index + c * 3;
                            col[0] = data[idx];
                            col[1] = data[idx + 1];
                            col[2] = data[idx + 2];
                        }
                    }
                }
                2 => {
                    for r in 0..4 {
                        let row_index = (start_y + r) as usize * stride + (start_x as usize) * 2;
                        for (c, col) in patch[r as usize].iter_mut().enumerate() {
                            let idx = row_index + c * 2;
                            col[0] = data[idx];
                            col[1] = data[idx + 1];
                        }
                    }
                }
                _ => {
                    for r in 0..4 {
                        let row_index = (start_y + r) as usize * stride + (start_x as usize);
                        for (c, col) in patch[r as usize].iter_mut().enumerate() {
                            let idx = row_index + c;
                            col[0] = data[idx];
                        }
                    }
                }
            }
            return patch;
        }

        match channels {
            4 => {
                for r in 0..4 {
                    let cur_y = start_y + r;
                    let y_ok = cur_y >= 0 && cur_y < h;
                    let row_index = if y_ok {
                        Some(cur_y as usize * stride)
                    } else {
                        None
                    };
                    for c in 0..4 {
                        let cur_x = start_x + c;
                        let x_ok = cur_x >= 0 && cur_x < w;
                        if let (Some(row_base), true) = (row_index, x_ok) {
                            let idx = row_base + cur_x as usize * 4;
                            patch[r as usize][c as usize][0] = data[idx];
                            patch[r as usize][c as usize][1] = data[idx + 1];
                            patch[r as usize][c as usize][2] = data[idx + 2];
                            patch[r as usize][c as usize][3] = data[idx + 3];
                        } else {
                            patch[r as usize][c as usize][0] = bg[0];
                            patch[r as usize][c as usize][1] = bg[1];
                            patch[r as usize][c as usize][2] = bg[2];
                            patch[r as usize][c as usize][3] = bg[3];
                        }
                    }
                }
            }
            3 => {
                for r in 0..4 {
                    let cur_y = start_y + r;
                    let y_ok = cur_y >= 0 && cur_y < h;
                    let row_index = if y_ok {
                        Some(cur_y as usize * stride)
                    } else {
                        None
                    };
                    for c in 0..4 {
                        let cur_x = start_x + c;
                        let x_ok = cur_x >= 0 && cur_x < w;
                        if let (Some(row_base), true) = (row_index, x_ok) {
                            let idx = row_base + cur_x as usize * 3;
                            patch[r as usize][c as usize][0] = data[idx];
                            patch[r as usize][c as usize][1] = data[idx + 1];
                            patch[r as usize][c as usize][2] = data[idx + 2];
                        } else {
                            patch[r as usize][c as usize][0] = bg[0];
                            patch[r as usize][c as usize][1] = bg[1];
                            patch[r as usize][c as usize][2] = bg[2];
                        }
                    }
                }
            }
            2 => {
                for r in 0..4 {
                    let cur_y = start_y + r;
                    let y_ok = cur_y >= 0 && cur_y < h;
                    let row_index = if y_ok {
                        Some(cur_y as usize * stride)
                    } else {
                        None
                    };
                    for c in 0..4 {
                        let cur_x = start_x + c;
                        let x_ok = cur_x >= 0 && cur_x < w;
                        if let (Some(row_base), true) = (row_index, x_ok) {
                            let idx = row_base + cur_x as usize * 2;
                            patch[r as usize][c as usize][0] = data[idx];
                            patch[r as usize][c as usize][1] = data[idx + 1];
                        } else {
                            patch[r as usize][c as usize][0] = bg[0];
                            patch[r as usize][c as usize][1] = bg[1];
                        }
                    }
                }
            }
            _ => {
                for r in 0..4 {
                    let cur_y = start_y + r;
                    let y_ok = cur_y >= 0 && cur_y < h;
                    let row_index = if y_ok {
                        Some(cur_y as usize * stride)
                    } else {
                        None
                    };
                    for c in 0..4 {
                        let cur_x = start_x + c;
                        let x_ok = cur_x >= 0 && cur_x < w;
                        if let (Some(row_base), true) = (row_index, x_ok) {
                            let idx = row_base + cur_x as usize;
                            patch[r as usize][c as usize][0] = data[idx];
                        } else {
                            patch[r as usize][c as usize][0] = bg[0];
                        }
                    }
                }
            }
        }

        patch
    }
}

#[derive(Copy, Clone, Debug)]
struct Point2f {
    x: f32,
    y: f32,
}

#[derive(Copy, Clone, Debug)]
struct Point3 {
    x: i32,
    y: i32,
    #[allow(dead_code)]
    z: i32,
}

fn round_to_int(input: f32) -> i32 {
    input.round() as i32
}

/// Computes the x-range where a horizontal scanline intersects a convex polygon.
///
/// Used to determine which output pixels map to valid source pixels, allowing
/// the algorithm to skip background-only regions at the start/end of each row.
/// Returns `None` if the scanline doesn't intersect the polygon.
fn scanline_range(
    poly: &[Point2f; 4],
    y: f32,
    new_width: i32,
    add_padding: bool,
) -> Option<(i32, i32)> {
    let mut xs = [0.0_f32; 4];
    let mut count = 0_usize;

    for i in 0..poly.len() {
        let p0 = poly[i];
        let p1 = poly[(i + 1) % poly.len()];
        let y0 = p0.y;
        let y1 = p1.y;
        if y0 == y1 {
            continue;
        }
        if y < y0.min(y1) || y > y0.max(y1) {
            continue;
        }
        let t = (y - y0) / (y1 - y0);
        xs[count] = p0.x + t * (p1.x - p0.x);
        count += 1;
    }

    if count == 0 {
        return None;
    }

    let mut min_x = xs[0];
    let mut max_x = xs[0];
    for &x in xs.iter().take(count).skip(1) {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
    }

    let mut x0 = min_x.ceil() as i32;
    let mut x1 = max_x.floor() as i32;
    if add_padding {
        x0 -= 1;
        x1 += 1;
    }

    if x0 < 0 {
        x0 = 0;
    }
    if x1 > new_width - 1 {
        x1 = new_width - 1;
    }
    if x0 <= x1 { Some((x0, x1)) } else { None }
}

/// Computes Catmull-Rom cubic interpolation weights for a fractional position.
///
/// Given `t` in [0, 1), returns weights for the 4 samples in a 1D neighborhood
/// centered at the integer position. The weights sum to 1.0 and produce a smooth
/// C1-continuous interpolation curve.
#[inline(always)]
fn get_cubic_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;

    let w1 = t3 - 2.0 * t2 + 1.0;

    let d2 = 1.0 - t;
    let d2_2 = d2 * d2;
    let w2 = (d2_2 * d2) - 2.0 * d2_2 + 1.0;

    let d3 = 2.0 - t;
    let d3_2 = d3 * d3;
    let w3 = -(d3_2 * d3) + 5.0 * d3_2 - 8.0 * d3 + 4.0;

    let d4 = t + 1.0;
    let d4_2 = d4 * d4;
    let w0 = -(d4_2 * d4) + 5.0 * d4_2 - 8.0 * d4 + 4.0;

    [w0, w1, w2, w3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageFormat};

    fn assert_close_pct(expected: f32, actual: f32, pct: f32) {
        let tol = expected.abs() * (pct / 100.0);
        let diff = (expected - actual).abs();
        assert!(
            diff <= tol,
            "expected {expected}, got {actual}, diff {diff} > tol {tol}"
        );
    }

    #[test]
    fn test_image() {
        let mut img = Image::new_empty();
        assert_eq!(img.format(), ImageFormat::None);
        img.reset(500, 500, ImageFormat::Rgb);
        assert_eq!(img.format(), ImageFormat::Rgb);
        assert_eq!(img.width(), 500);
        assert_eq!(img.height(), 500);
    }

    #[test]
    fn test_transformation() {
        let rotate = OpRotateImage::new();
        let tf = rotate
            .get_transformation_matrix(std::f32::consts::PI)
            .expect("matrix");
        assert_close_pct(-1.0, tf[0], 0.01);
    }

    #[test]
    fn test_normalization() {
        let rotate = OpRotateImage::new();
        let expected = 45.0_f32;
        let actual = 405.0_f32;
        let result = rotate.normalize_angle(actual);
        assert_close_pct(expected, result, 0.01);
    }

    #[test]
    fn test_fast_path_direction_affects_orientation() {
        let mut rotate = OpRotateImage::new();
        rotate.set_rotation(90.0, RotateDirection::Cw);
        assert!(rotate.use_op_orient);
        assert_eq!(rotate.orientation, Orientation90::Right);

        rotate.set_rotation(90.0, RotateDirection::Ccw);
        assert!(rotate.use_op_orient);
        assert_eq!(rotate.orientation, Orientation90::Left);
    }

    #[test]
    fn test_fast_path_disabled_on_non_right_angle() {
        let mut rotate = OpRotateImage::new();
        rotate.set_rotation(90.0, RotateDirection::Cw);
        assert!(rotate.use_op_orient);

        rotate.set_rotation(45.0, RotateDirection::Cw);
        assert!(!rotate.use_op_orient);
    }

    #[test]
    fn test_find_new_image_size() {
        let rotate = OpRotateImage::new();
        let expected_size = (500.0_f32 * std::f32::consts::SQRT_2) as usize;
        let tf = rotate
            .get_transformation_matrix(45.0_f32 * std::f32::consts::PI / 180.0)
            .expect("matrix");

        let (result_x, result_y) = rotate.find_new_image_size(500, 500, &tf);
        assert_eq!(result_x, result_y);
        assert_close_pct(expected_size as f32, result_x as f32, 0.01);
    }

    #[test]
    fn test_find_new_corner() {
        let rotate = OpRotateImage::new();
        let tf = rotate
            .get_transformation_matrix(45.0_f32 * std::f32::consts::PI / 180.0)
            .expect("matrix");
        let corner = Point3 { x: 0, y: 0, z: 1 };
        let output = rotate.find_new_corner(&tf, &corner, 500, 500);
        assert_eq!(output.x, 0);
    }
}
