//! Criterion benchmarks for `OpRotateImage`.
//!
//! Focuses on rotation kernel cost by reusing preallocated output buffers.
//! Covers size scaling, format differences, and fast-path (90/180/270 deg) vs
//! bicubic interpolation. For detailed metrics or Linux perf counters, use
//! `cargo run --bin benchmark_op_rotate_image`.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rusty_pixels::{Image, ImageFormat, OpRotateImage, RotateDirection, bench_utils};

fn make_fixture(size: usize, format: ImageFormat, angle: f32) -> (OpRotateImage, Image, Image) {
    let input = bench_utils::create_test_image(size, size, format);
    let mut rotate = OpRotateImage::new();
    rotate.set_rotation(angle, RotateDirection::Cw);
    let (out_w, out_h) = rotate.compute_output_dimensions(&input);
    let premul = input.is_premultiplied();
    let output = Image::with_premultiplied(out_w, out_h, format, premul);
    (rotate, input, output)
}

fn bench_size_scaling_rgb_45deg(c: &mut Criterion) {
    let mut group = c.benchmark_group("size_scaling_rgb_45deg");
    for size in bench_utils::BENCH_SIZES {
        let (mut rotate, input, mut output) = make_fixture(size, ImageFormat::Rgb, 45.0);
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                rotate.apply_to_preallocated(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

fn bench_format_512_45deg(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_512_45deg");
    let size = 512_usize;
    for format in bench_utils::BENCH_FORMATS {
        let (mut rotate, input, mut output) = make_fixture(size, format, 45.0);
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_function(
            BenchmarkId::new(bench_utils::format_to_string(format), size),
            |b| {
                b.iter(|| {
                    rotate.apply_to_preallocated(black_box(&input), black_box(&mut output));
                });
            },
        );
    }
    group.finish();
}

fn bench_angle_512_rgb(c: &mut Criterion) {
    let mut group = c.benchmark_group("angle_512_rgb");
    let size = 512_usize;
    let format = ImageFormat::Rgb;
    for angle in bench_utils::BENCH_ANGLES {
        let label = if (angle % 90.0) == 0.0 {
            "fast"
        } else {
            "interp"
        };
        let (mut rotate, input, mut output) = make_fixture(size, format, angle);
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_function(BenchmarkId::new(label, angle as i32), |b| {
            b.iter(|| {
                rotate.apply_to_preallocated(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

fn bench_fast_path_vs_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_vs_interp");
    let format = ImageFormat::Rgb;
    for size in bench_utils::BENCH_SIZES {
        let (mut rotate_fast, input_fast, mut output_fast) = make_fixture(size, format, 90.0);
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_function(BenchmarkId::new("fast_90", size), |b| {
            b.iter(|| {
                rotate_fast
                    .apply_to_preallocated(black_box(&input_fast), black_box(&mut output_fast));
            });
        });

        let (mut rotate_interp, input_interp, mut output_interp) = make_fixture(size, format, 45.0);
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_function(BenchmarkId::new("interp_45", size), |b| {
            b.iter(|| {
                rotate_interp
                    .apply_to_preallocated(black_box(&input_interp), black_box(&mut output_interp));
            });
        });
    }
    group.finish();
}

fn bench_rgba_45deg(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgba_45deg");
    for size in bench_utils::BENCH_SIZES {
        let (mut rotate, input, mut output) = make_fixture(size, ImageFormat::Rgba, 45.0);
        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                rotate.apply_to_preallocated(black_box(&input), black_box(&mut output));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_size_scaling_rgb_45deg,
    bench_format_512_45deg,
    bench_angle_512_rgb,
    bench_fast_path_vs_interpolation,
    bench_rgba_45deg
);
criterion_main!(benches);
