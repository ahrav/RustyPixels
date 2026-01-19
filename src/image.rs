//! Image representation with row-major pixel storage.
//!
//! # Memory Layout
//!
//! Pixels are stored in a flat `Sample` buffer in row-major order:
//!
//! ```text
//! data[y * stride + x * channels + c]
//! ```
//!
//! where `stride = width * channels`.

use std::cmp;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

/// 16-bit color sample. Higher precision than 8-bit avoids banding during interpolation.
pub type Sample = u16;

/// Maximum sample value (65535).
pub const MAX_VALUE: Sample = u16::MAX;

/// Maximum 8-bit sample value (255).
#[allow(dead_code)]
pub const MAX_VALUE_8_BIT: u8 = u8::MAX;

/// Pixel format describing the number and meaning of channels.
///
/// Channel counts: `None`=0, `Gray`=1, `GrayAlpha`=2, `Rgb`/`YCbCr`=3, `Rgba`/`YCbCrA`=4.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    None,
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
    YCbCr,
    YCbCrA,
}

impl ImageFormat {
    /// Returns the number of channels for this format.
    pub fn channel_count(self) -> usize {
        match self {
            ImageFormat::YCbCr => 3,
            ImageFormat::YCbCrA => 4,
            ImageFormat::None => 0,
            ImageFormat::Gray => 1,
            ImageFormat::GrayAlpha => 2,
            ImageFormat::Rgb => 3,
            ImageFormat::Rgba => 4,
        }
    }

    /// Returns true if this format includes an alpha channel.
    pub fn has_alpha(self) -> bool {
        matches!(
            self,
            ImageFormat::GrayAlpha | ImageFormat::Rgba | ImageFormat::YCbCrA
        )
    }
}

/// Allocation strategy for image pixel buffers.
///
/// The default uses huge pages on macOS/Linux and standard pages elsewhere.
/// `HugePages` is best-effort: it falls back to standard pages if the OS
/// cannot satisfy the request.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageAllocation {
    Standard,
    HugePages,
}

impl Default for ImageAllocation {
    fn default() -> Self {
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            ImageAllocation::HugePages
        } else {
            ImageAllocation::Standard
        }
    }
}

impl ImageAllocation {
    /// Alias for `Default::default()`.
    pub fn platform_default() -> Self {
        Self::default()
    }
}

/// A color value with format-aware channel accessors.
///
/// Stores up to 4 samples and interprets them based on the associated `ImageFormat`.
/// Accessors like `r()`, `g()`, `b()` work for RGB/RGBA/Gray formats, while `y()`,
/// `cb()`, `cr()` work for YCbCr formats. Grayscale values are returned for all
/// RGB channels when the format is `Gray` or `GrayAlpha`.
#[derive(Copy, Clone, Debug)]
pub struct Color {
    samples: [Sample; 4],
    format: ImageFormat,
}

impl Color {
    /// Creates a new color from raw samples.
    pub fn new(samples: [Sample; 4], format: ImageFormat) -> Self {
        Self { samples, format }
    }

    /// Returns the red channel. Panics if not RGB/RGBA/Gray/GrayAlpha.
    pub fn r(&self) -> Sample {
        match self.format {
            ImageFormat::Rgb | ImageFormat::Rgba | ImageFormat::Gray | ImageFormat::GrayAlpha => {
                self.samples[0]
            }
            _ => panic!("Image is not RGB nor Grayscale"),
        }
    }

    /// Returns the green channel. Panics if not RGB/RGBA/Gray/GrayAlpha.
    pub fn g(&self) -> Sample {
        match self.format {
            ImageFormat::Rgb | ImageFormat::Rgba => self.samples[1],
            ImageFormat::Gray | ImageFormat::GrayAlpha => self.samples[0],
            _ => panic!("Image is not RGB nor Grayscale"),
        }
    }

    /// Returns the blue channel. Panics if not RGB/RGBA/Gray/GrayAlpha.
    pub fn b(&self) -> Sample {
        match self.format {
            ImageFormat::Rgb | ImageFormat::Rgba => self.samples[2],
            ImageFormat::Gray | ImageFormat::GrayAlpha => self.samples[0],
            _ => panic!("Image is not RGB nor Grayscale"),
        }
    }

    /// Returns the Y (luma) channel. Panics if not YCbCr/YCbCrA/Gray/GrayAlpha.
    pub fn y(&self) -> Sample {
        match self.format {
            ImageFormat::YCbCr
            | ImageFormat::YCbCrA
            | ImageFormat::Gray
            | ImageFormat::GrayAlpha => self.samples[0],
            _ => panic!("Image is not YCbCr nor Grayscale"),
        }
    }

    /// Returns the Cb (chroma blue) channel. Panics if not YCbCr/YCbCrA/Gray/GrayAlpha.
    pub fn cb(&self) -> Sample {
        match self.format {
            ImageFormat::YCbCr | ImageFormat::YCbCrA => self.samples[1],
            ImageFormat::Gray | ImageFormat::GrayAlpha => self.samples[0],
            _ => panic!("Image is not YCbCr nor Grayscale"),
        }
    }

    /// Returns the Cr (chroma red) channel. Panics if not YCbCr/YCbCrA/Gray/GrayAlpha.
    pub fn cr(&self) -> Sample {
        match self.format {
            ImageFormat::YCbCr | ImageFormat::YCbCrA => self.samples[2],
            ImageFormat::Gray | ImageFormat::GrayAlpha => self.samples[0],
            _ => panic!("Image is not YCbCr nor Grayscale"),
        }
    }

    /// Returns the alpha channel, or MAX_VALUE if the format has no alpha.
    pub fn a(&self) -> Sample {
        match self.format {
            ImageFormat::GrayAlpha => self.samples[1],
            ImageFormat::Rgba | ImageFormat::YCbCrA => self.samples[3],
            _ => MAX_VALUE,
        }
    }

    /// Converts to grayscale using ITU-R BT.601 luma coefficients.
    pub fn gray(&self) -> Sample {
        match self.format {
            ImageFormat::Gray | ImageFormat::GrayAlpha => self.r(),
            ImageFormat::YCbCr | ImageFormat::YCbCrA => self.y(),
            ImageFormat::Rgb | ImageFormat::Rgba => {
                // ITU-R BT.601 luma coefficients (77/256, 150/256, 29/256)
                let red_weight: u64 = 77;
                let green_weight: u64 = 150;
                let blue_weight: u64 = 29;
                let r = self.r() as u64;
                let g = self.g() as u64;
                let b = self.b() as u64;
                ((red_weight * r + green_weight * g + blue_weight * b) >> 8) as Sample
            }
            ImageFormat::None => 0,
        }
    }
}

/// Internal buffer that can be backed by standard pages or OS huge pages.
// TODO: Pre-allocate memory via some sort of allocator, free-list, etc?
pub(crate) enum ImageBuffer {
    Vec(Vec<Sample>),
    Mmap {
        ptr: NonNull<Sample>,
        len: usize,
        bytes: usize,
    },
}

impl ImageBuffer {
    fn empty() -> Self {
        ImageBuffer::Vec(Vec::new())
    }

    fn new(len: usize, allocation: ImageAllocation) -> Self {
        if len == 0 {
            return ImageBuffer::empty();
        }
        match allocation {
            ImageAllocation::Standard => ImageBuffer::Vec(vec![0; len]),
            ImageAllocation::HugePages => {
                try_huge_pages(len).unwrap_or_else(|| ImageBuffer::Vec(vec![0; len]))
            }
        }
    }

    pub(crate) fn as_slice(&self) -> &[Sample] {
        match self {
            ImageBuffer::Vec(data) => data.as_slice(),
            ImageBuffer::Mmap { ptr, len, .. } => unsafe {
                slice::from_raw_parts(ptr.as_ptr(), *len)
            },
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [Sample] {
        match self {
            ImageBuffer::Vec(data) => data.as_mut_slice(),
            ImageBuffer::Mmap { ptr, len, .. } => unsafe {
                slice::from_raw_parts_mut(ptr.as_ptr(), *len)
            },
        }
    }
}

impl Deref for ImageBuffer {
    type Target = [Sample];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for ImageBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl fmt::Debug for ImageBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl Drop for ImageBuffer {
    fn drop(&mut self) {
        if let ImageBuffer::Mmap { ptr, bytes, .. } = self {
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            unsafe {
                libc::munmap(ptr.as_ptr() as *mut libc::c_void, *bytes);
            }
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
const HUGE_PAGE_MIN_BYTES: usize = 2 * 1024 * 1024;

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn align_up(value: usize, alignment: usize) -> Option<usize> {
    if alignment == 0 {
        return None;
    }
    let rem = value % alignment;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - rem)
    }
}

#[cfg(target_os = "linux")]
fn page_size() -> usize {
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if size > 0 { size as usize } else { 4096 }
}

#[cfg(target_os = "macos")]
const VM_FLAGS_SUPERPAGE_SIZE_2MB: libc::c_int = 0x00020000;

#[cfg(target_os = "macos")]
fn try_huge_pages(len: usize) -> Option<ImageBuffer> {
    let byte_len = len.checked_mul(std::mem::size_of::<Sample>())?;
    if byte_len < HUGE_PAGE_MIN_BYTES {
        return None;
    }
    let alloc_bytes = align_up(byte_len, HUGE_PAGE_MIN_BYTES)?;
    let map_ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            alloc_bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            VM_FLAGS_SUPERPAGE_SIZE_2MB,
            0,
        )
    };
    if map_ptr == libc::MAP_FAILED {
        return None;
    }
    let ptr = match NonNull::new(map_ptr as *mut Sample) {
        Some(ptr) => ptr,
        None => {
            unsafe {
                libc::munmap(map_ptr, alloc_bytes);
            }
            return None;
        }
    };
    Some(ImageBuffer::Mmap {
        ptr,
        len,
        bytes: alloc_bytes,
    })
}

#[cfg(target_os = "linux")]
fn try_huge_pages(len: usize) -> Option<ImageBuffer> {
    let byte_len = len.checked_mul(std::mem::size_of::<Sample>())?;
    if byte_len < HUGE_PAGE_MIN_BYTES {
        return None;
    }
    let alloc_bytes = align_up(byte_len, page_size())?;
    let map_ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            alloc_bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if map_ptr == libc::MAP_FAILED {
        return None;
    }
    unsafe {
        libc::madvise(map_ptr, alloc_bytes, libc::MADV_HUGEPAGE);
    }
    let ptr = match NonNull::new(map_ptr as *mut Sample) {
        Some(ptr) => ptr,
        None => {
            unsafe {
                libc::munmap(map_ptr, alloc_bytes);
            }
            return None;
        }
    };
    Some(ImageBuffer::Mmap {
        ptr,
        len,
        bytes: alloc_bytes,
    })
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn try_huge_pages(_len: usize) -> Option<ImageBuffer> {
    None
}

/// A 2D image stored as a flat array of 16-bit samples in row-major order.
///
/// # Memory Layout
///
/// Pixel data is stored contiguously: `data[y * stride + x * channels + c]` where
/// `stride = width * channels`. This layout is cache-friendly for row-wise traversal.
/// Buffer allocation can be configured to use OS huge pages when available.
///
/// # Premultiplied Alpha
///
/// When `is_premultiplied` is true, RGB values are pre-multiplied by alpha. This is
/// required for correct interpolation of semi-transparent pixels during rotation.
/// All constructors default to premultiplied alpha.
#[derive(Debug)]
pub struct Image {
    width: usize,
    height: usize,
    format: ImageFormat,
    is_premultiplied: bool,
    allocation: ImageAllocation,
    pub(crate) data: ImageBuffer,
}

impl Clone for Image {
    fn clone(&self) -> Self {
        let mut cloned = Image::with_premultiplied_and_allocation(
            self.width,
            self.height,
            self.format,
            self.is_premultiplied,
            self.allocation,
        );
        cloned
            .data
            .as_mut_slice()
            .copy_from_slice(self.data.as_slice());
        cloned
    }
}

impl Default for Image {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl Image {
    /// Creates an empty image with no dimensions.
    pub fn new_empty() -> Self {
        Self {
            width: 0,
            height: 0,
            format: ImageFormat::None,
            is_premultiplied: true,
            allocation: ImageAllocation::default(),
            data: ImageBuffer::empty(),
        }
    }

    /// Creates a new image with the given dimensions and format.
    pub fn new(width: usize, height: usize, format: ImageFormat) -> Self {
        Self::with_premultiplied_and_allocation(
            width,
            height,
            format,
            true,
            ImageAllocation::default(),
        )
    }

    /// Creates a new image with a specific allocation strategy.
    pub fn with_allocation(
        width: usize,
        height: usize,
        format: ImageFormat,
        allocation: ImageAllocation,
    ) -> Self {
        Self::with_premultiplied_and_allocation(width, height, format, true, allocation)
    }

    /// Creates a new image with explicit premultiplied alpha setting.
    pub fn with_premultiplied(
        width: usize,
        height: usize,
        format: ImageFormat,
        is_premultiplied: bool,
    ) -> Self {
        Self::with_premultiplied_and_allocation(
            width,
            height,
            format,
            is_premultiplied,
            ImageAllocation::default(),
        )
    }

    /// Creates a new image with all options specified.
    pub fn with_premultiplied_and_allocation(
        width: usize,
        height: usize,
        format: ImageFormat,
        is_premultiplied: bool,
        allocation: ImageAllocation,
    ) -> Self {
        let channels = format.channel_count();
        let len = width.saturating_mul(height).saturating_mul(channels);
        Self {
            width,
            height,
            format,
            is_premultiplied,
            allocation,
            data: ImageBuffer::new(len, allocation),
        }
    }

    /// Resets the image to new dimensions and format, reallocating if needed.
    pub fn reset(&mut self, width: usize, height: usize, format: ImageFormat) {
        let allocation = self.allocation;
        *self = Self::with_premultiplied_and_allocation(width, height, format, true, allocation);
    }

    /// Returns the pixel format.
    pub fn format(&self) -> ImageFormat {
        self.format
    }

    /// Returns the allocation strategy.
    pub fn allocation(&self) -> ImageAllocation {
        self.allocation
    }

    /// Returns true if the format has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.format.has_alpha()
    }

    /// Returns true if RGB values are premultiplied by alpha.
    pub fn is_premultiplied(&self) -> bool {
        self.is_premultiplied
    }

    /// Returns the width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the number of channels per pixel.
    pub fn channels(&self) -> usize {
        self.format.channel_count()
    }

    /// Returns the row stride in samples (width * channels).
    pub fn stride(&self) -> usize {
        self.width * self.channels()
    }

    /// Returns the samples for row `y`.
    pub fn row(&self, y: usize) -> &[Sample] {
        let stride = self.stride();
        let start = y * stride;
        let end = start + stride;
        &self.data[start..end]
    }

    /// Returns mutable samples for row `y`.
    pub fn row_mut(&mut self, y: usize) -> &mut [Sample] {
        let stride = self.stride();
        let start = y * stride;
        let end = start + stride;
        &mut self.data[start..end]
    }

    /// Returns the samples for the pixel at (x, y).
    pub fn pixel(&self, x: usize, y: usize) -> &[Sample] {
        let channels = self.channels();
        let stride = self.stride();
        let start = y * stride + x * channels;
        &self.data[start..start + channels]
    }

    /// Returns mutable samples for the pixel at (x, y).
    pub fn pixel_mut(&mut self, x: usize, y: usize) -> &mut [Sample] {
        let channels = self.channels();
        let stride = self.stride();
        let start = y * stride + x * channels;
        &mut self.data[start..start + channels]
    }

    /// Fills all pixels with the given color.
    pub fn fill_with_color(&mut self, color: &Color) {
        let channels = self.channels();
        let width = self.width;
        let format = self.format;
        if channels == 0 {
            return;
        }
        for y in 0..self.height {
            let row = self.row_mut(y);
            for x in 0..width {
                let pixel = &mut row[x * channels..(x + 1) * channels];
                Self::fill_pixel(format, pixel, color);
            }
        }
    }

    /// Fills all pixels with 8-bit RGBA values.
    pub fn fill_u8(&mut self, r: u8, g: u8, b: u8, a: u8) {
        let samples = convert_u8_to_sample(r, g, b, a);
        let color = Color::new(samples, self.format);
        self.fill_with_color(&color);
    }

    /// Writes a color to a pixel slice based on format.
    pub fn fill_pixel(format: ImageFormat, pixel: &mut [Sample], color: &Color) {
        match format {
            ImageFormat::Gray => {
                pixel[0] = color.r();
            }
            ImageFormat::GrayAlpha => {
                pixel[0] = color.r();
                pixel[1] = color.a();
            }
            ImageFormat::Rgb => {
                pixel[0] = color.r();
                pixel[1] = color.g();
                pixel[2] = color.b();
            }
            ImageFormat::Rgba => {
                pixel[0] = color.r();
                pixel[1] = color.g();
                pixel[2] = color.b();
                pixel[3] = color.a();
            }
            ImageFormat::YCbCr => {
                pixel[0] = color.y();
                pixel[1] = color.cb();
                pixel[2] = color.cr();
            }
            ImageFormat::YCbCrA => {
                pixel[0] = color.y();
                pixel[1] = color.cb();
                pixel[2] = color.cr();
                pixel[3] = color.a();
            }
            ImageFormat::None => {}
        }
    }

    /// Extracts a rectangular region. Negative coords clamp to 0; out-of-bounds returns empty.
    pub fn crop(&self, left: i32, top: i32, width: usize, height: usize) -> Image {
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
        if left as usize > self.width || top as usize > self.height {
            return Image::new_empty();
        }

        width = cmp::min(width, self.width - left as usize);
        height = cmp::min(height, self.height - top as usize);

        if left == 0 && top == 0 && width == self.width && height == self.height {
            return self.clone();
        }

        let mut result = Image::with_premultiplied_and_allocation(
            width,
            height,
            self.format,
            self.is_premultiplied,
            self.allocation,
        );
        let channels = self.channels();
        let in_stride = self.stride();
        let out_stride = result.stride();
        let mut in_index = (top as usize) * in_stride + (left as usize) * channels;
        let mut out_index = 0;
        let span = width * channels;

        for _ in 0..height {
            result.data[out_index..out_index + span]
                .copy_from_slice(&self.data[in_index..in_index + span]);
            in_index += in_stride;
            out_index += out_stride;
        }

        result
    }
}

/// Converts 8-bit RGBA to 16-bit samples by multiplying by 257 (0→0, 255→65535).
pub fn convert_u8_to_sample(r: u8, g: u8, b: u8, a: u8) -> [Sample; 4] {
    [
        (r as Sample) * 257,
        (g as Sample) * 257,
        (b as Sample) * 257,
        (a as Sample) * 257,
    ]
}
