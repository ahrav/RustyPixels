//! Image rotation library with two rotation strategies.
//!
//! # Two-Path Design
//!
//! - **Fast path** ([`OpOrient90Increments`]): For exact 90°/180°/270° rotations and EXIF
//!   orientation codes. Uses simple coordinate remapping with no interpolation.
//! - **Interpolation path** ([`OpRotateImage`]): For arbitrary angles. Uses bicubic
//!   interpolation with 32x32 tiled processing for cache efficiency.
//!
//! [`OpRotateImage`] automatically delegates to the fast path when the angle is an exact
//! multiple of 90 degrees.
//!
//! # Example
//!
//! ```
//! use rusty_pixels::{Image, ImageFormat, OpRotateImage, RotateDirection};
//!
//! // Create a 100x100 RGB image
//! let src = Image::new(100, 100, ImageFormat::Rgb);
//!
//! // Rotate 45 degrees clockwise (uses bicubic interpolation)
//! let mut rotate = OpRotateImage::new();
//! rotate.set_rotation(45.0, RotateDirection::Cw);
//! let rotated = rotate.apply(&src);
//!
//! // Rotate 90 degrees (uses fast path automatically)
//! rotate.set_rotation(90.0, RotateDirection::Cw);
//! let rotated_90 = rotate.apply(&src);
//! ```

#[doc(hidden)]
pub mod bench_utils;
mod image;
mod op_orient_90;
mod op_rotate_image;

pub use crate::image::{Color, Image, ImageAllocation, ImageFormat};
pub use crate::op_orient_90::{OpOrient90Increments, Orientation90};
pub use crate::op_rotate_image::{OpRotateImage, RotateDirection};
