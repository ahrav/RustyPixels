//! Benchmark suite for `OpRotateImage` rotation performance.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin benchmark_op_rotate_image              # Run all benchmarks
//! cargo run --bin benchmark_op_rotate_image -- --json    # JSON output
//! cargo run --bin benchmark_op_rotate_image -- --filter Size  # Filter by pattern
//! cargo run --bin benchmark_op_rotate_image -- --list-tests   # List available tests
//! ```
//!
//! For Linux Docker setup, perf counters, and huge page checks, see
//! `docs/linux-docker-benchmarks.md`.
//!
//! # Notes
//!
//! - Timings measure the rotation kernel with preallocated input/output buffers.
//! - Hardware counters are Linux-only and best-effort; they require perf_event
//!   permissions and are zeroed on other platforms.
//! - Linux runs also request transparent huge pages when available.
//!
//! # Benchmark Categories
//!
//! - **Size**: Scaling behavior from 256×256 to 4096×4096 (RGB, 45°)
//! - **Format**: Gray vs RGB vs RGBA at 512×512
//! - **Angle**: Various angles including fast-path (90°/180°/270°) vs interpolation
//! - **Fast path vs Interpolation**: Direct comparison at multiple sizes
//! - **RGBA Interpolation**: Specialized 4-channel interpolation performance
//!
//! # Metrics
//!
//! - **Timing**: mean, median, std dev, min, max, p95 (in milliseconds)
//! - **Throughput**: megapixels per second (MP/s)
//! - **Memory**: output image size
//! - **Hardware counters**: available on Linux when perf events are enabled

use rusty_pixels::{Image, ImageFormat, OpRotateImage, RotateDirection, bench_utils};
use std::cell::RefCell;
use std::env;
use std::fs::File;
use std::io::{self, Write};
#[cfg(target_os = "linux")]
use std::os::unix::io::RawFd;
use std::time::{Duration, Instant};
#[cfg(target_os = "linux")]
use std::{fs, path::Path};

/// Results from a single benchmark run.
///
/// Collects timing statistics (mean, median, p95), throughput (MP/s),
/// and optional hardware counters (cache misses, IPC) on Linux.
#[derive(Clone, Debug)]
struct BenchmarkResult {
    test_name: String,
    mean_time_ms: f64,
    median_time_ms: f64,
    standard_deviation: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    p95_time_ms: f64,
    iterations: usize,

    image_width: usize,
    image_height: usize,
    format_name: String,
    rotation_angle: f32,

    peak_memory_bytes: usize,
    output_image_bytes: usize,
    total_allocations: usize,

    l1_cache_misses: usize,
    l2_cache_misses: usize,
    l3_cache_misses: usize,
    tlb_misses: usize,
    branch_misses: usize,
    cache_hit_ratio: f64,

    cpu_cycles: usize,
    instructions: usize,
    stalled_cycles_frontend: usize,
    stalled_cycles_backend: usize,
    stalled_cycles_backend_mem: usize,
    instructions_per_cycle: f64,

    pixels_per_second: f64,
    megapixels_per_second: f64,

    allocation_overhead_ms: f64,
    compute_only_time_ms: f64,
}

impl BenchmarkResult {
    fn new(test_name: &str, iterations: usize) -> Self {
        Self {
            test_name: test_name.to_string(),
            mean_time_ms: 0.0,
            median_time_ms: 0.0,
            standard_deviation: 0.0,
            min_time_ms: 0.0,
            max_time_ms: 0.0,
            p95_time_ms: 0.0,
            iterations,
            image_width: 0,
            image_height: 0,
            format_name: String::new(),
            rotation_angle: 0.0,
            peak_memory_bytes: 0,
            output_image_bytes: 0,
            total_allocations: 0,
            l1_cache_misses: 0,
            l2_cache_misses: 0,
            l3_cache_misses: 0,
            tlb_misses: 0,
            branch_misses: 0,
            cache_hit_ratio: 0.0,
            cpu_cycles: 0,
            instructions: 0,
            stalled_cycles_frontend: 0,
            stalled_cycles_backend: 0,
            stalled_cycles_backend_mem: 0,
            instructions_per_cycle: 0.0,
            pixels_per_second: 0.0,
            megapixels_per_second: 0.0,
            allocation_overhead_ms: 0.0,
            compute_only_time_ms: 0.0,
        }
    }
}

/// Point-in-time snapshot of hardware performance counters.
///
/// Values are cumulative from when counters were enabled.
#[derive(Clone, Copy, Default)]
struct HardwareCounterSnapshot {
    l1_cache_misses: usize,
    l2_cache_misses: usize,
    l3_cache_misses: usize,
    tlb_misses: usize,
    branch_misses: usize,
    cpu_cycles: usize,
    instructions: usize,
    stalled_cycles_frontend: usize,
    stalled_cycles_backend: usize,
    stalled_cycles_backend_mem: usize,
}

/// Linux perf_event file descriptors for hardware performance monitoring.
///
/// Opens counters for cache misses (L1/L2/L3), TLB misses, branch mispredictions,
/// CPU cycles, instructions, and stall cycles. Falls back gracefully when
/// specific counters are unavailable (varies by CPU architecture).
#[cfg(target_os = "linux")]
#[derive(Debug)]
struct HardwareCounters {
    l1_cache_misses: RawFd,
    l2_cache_misses: RawFd,
    l3_cache_misses: RawFd,
    tlb_misses: RawFd,
    branch_misses: RawFd,
    cpu_cycles: RawFd,
    instructions: RawFd,
    stalled_cycles_frontend: RawFd,
    stalled_cycles_backend: RawFd,
    stalled_backend_raw: RawFd,
    stalled_backend_mem: RawFd,
    enabled: bool,
}

#[cfg(target_os = "linux")]
impl HardwareCounters {
    /// Opens available hardware counters. Returns disabled instance if `enable` is false
    /// or no counters can be opened (permission denied, unsupported CPU).
    fn new(enable: bool) -> Self {
        if !enable {
            return Self::disabled();
        }

        let base_attr = PerfEventAttr::new();
        let mut counters = Self::disabled();
        let mut any_open = false;

        let mut cache_attr = base_attr;
        cache_attr.type_ = PERF_TYPE_HW_CACHE;
        cache_attr.config = PERF_COUNT_HW_CACHE_L1D
            | (PERF_COUNT_HW_CACHE_OP_READ << 8)
            | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        counters.l1_cache_misses = perf_event_open(&cache_attr);
        any_open |= counters.l1_cache_misses >= 0;

        let l2_events = [
            "l2d_cache_refill",
            "l2d_cache_lmiss_rd",
            "l2_cache_refill",
            "l2_rqsts.miss",
        ];
        counters.l2_cache_misses = open_pmu_event_by_name_list(&l2_events, &base_attr);
        if counters.l2_cache_misses < 0 {
            cache_attr.config = PERF_COUNT_HW_CACHE_LL
                | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            counters.l2_cache_misses = perf_event_open(&cache_attr);
        }
        any_open |= counters.l2_cache_misses >= 0;

        let l3_events = [
            "l3d_cache_refill",
            "l3_cache_refill",
            "llc_misses",
            "llc-load-misses",
        ];
        counters.l3_cache_misses = open_pmu_event_by_name_list(&l3_events, &base_attr);
        if counters.l3_cache_misses < 0 {
            cache_attr.config = PERF_COUNT_HW_CACHE_NODE
                | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            counters.l3_cache_misses = perf_event_open(&cache_attr);
        }
        any_open |= counters.l3_cache_misses >= 0;

        cache_attr.config = PERF_COUNT_HW_CACHE_DTLB
            | (PERF_COUNT_HW_CACHE_OP_READ << 8)
            | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        counters.tlb_misses = perf_event_open(&cache_attr);
        any_open |= counters.tlb_misses >= 0;

        let mut hw_attr = base_attr;
        hw_attr.type_ = PERF_TYPE_HARDWARE;
        hw_attr.config = PERF_COUNT_HW_BRANCH_MISSES;
        counters.branch_misses = perf_event_open(&hw_attr);
        any_open |= counters.branch_misses >= 0;

        hw_attr.config = PERF_COUNT_HW_CPU_CYCLES;
        counters.cpu_cycles = perf_event_open(&hw_attr);
        any_open |= counters.cpu_cycles >= 0;

        hw_attr.config = PERF_COUNT_HW_INSTRUCTIONS;
        counters.instructions = perf_event_open(&hw_attr);
        any_open |= counters.instructions >= 0;

        hw_attr.config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
        counters.stalled_cycles_frontend = perf_event_open(&hw_attr);
        any_open |= counters.stalled_cycles_frontend >= 0;

        hw_attr.config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
        counters.stalled_cycles_backend = perf_event_open(&hw_attr);
        any_open |= counters.stalled_cycles_backend >= 0;

        counters.stalled_backend_raw = open_pmu_event_by_name("stall_backend", &base_attr);
        counters.stalled_backend_mem = open_pmu_event_by_name("stall_backend_mem", &base_attr);
        any_open |= counters.stalled_backend_raw >= 0 || counters.stalled_backend_mem >= 0;

        counters.enabled = any_open;
        counters
    }

    fn available(&self) -> bool {
        self.enabled
    }

    /// Resets all counters to zero and starts counting.
    fn reset_and_enable(&self) {
        if !self.enabled {
            return;
        }
        reset_and_enable_fd(self.l1_cache_misses);
        reset_and_enable_fd(self.l2_cache_misses);
        reset_and_enable_fd(self.l3_cache_misses);
        reset_and_enable_fd(self.tlb_misses);
        reset_and_enable_fd(self.branch_misses);
        reset_and_enable_fd(self.cpu_cycles);
        reset_and_enable_fd(self.instructions);
        reset_and_enable_fd(self.stalled_cycles_frontend);
        reset_and_enable_fd(self.stalled_cycles_backend);
        reset_and_enable_fd(self.stalled_backend_raw);
        reset_and_enable_fd(self.stalled_backend_mem);
    }

    /// Stops all counters. Values are preserved until next reset.
    fn disable(&self) {
        if !self.enabled {
            return;
        }
        disable_fd(self.l1_cache_misses);
        disable_fd(self.l2_cache_misses);
        disable_fd(self.l3_cache_misses);
        disable_fd(self.tlb_misses);
        disable_fd(self.branch_misses);
        disable_fd(self.cpu_cycles);
        disable_fd(self.instructions);
        disable_fd(self.stalled_cycles_frontend);
        disable_fd(self.stalled_cycles_backend);
        disable_fd(self.stalled_backend_raw);
        disable_fd(self.stalled_backend_mem);
    }

    /// Reads current counter values, applying time-based scaling for multiplexed counters.
    fn read(&self) -> HardwareCounterSnapshot {
        if !self.enabled {
            return HardwareCounterSnapshot::default();
        }

        let mut stalled_backend = read_scaled_counter(self.stalled_cycles_backend);
        if stalled_backend == 0 {
            let raw = read_scaled_counter(self.stalled_backend_raw);
            if raw > 0 {
                stalled_backend = raw;
            }
        }

        HardwareCounterSnapshot {
            l1_cache_misses: read_scaled_counter(self.l1_cache_misses) as usize,
            l2_cache_misses: read_scaled_counter(self.l2_cache_misses) as usize,
            l3_cache_misses: read_scaled_counter(self.l3_cache_misses) as usize,
            tlb_misses: read_scaled_counter(self.tlb_misses) as usize,
            branch_misses: read_scaled_counter(self.branch_misses) as usize,
            cpu_cycles: read_scaled_counter(self.cpu_cycles) as usize,
            instructions: read_scaled_counter(self.instructions) as usize,
            stalled_cycles_frontend: read_scaled_counter(self.stalled_cycles_frontend) as usize,
            stalled_cycles_backend: stalled_backend as usize,
            stalled_cycles_backend_mem: read_scaled_counter(self.stalled_backend_mem) as usize,
        }
    }

    fn disabled() -> Self {
        Self {
            l1_cache_misses: -1,
            l2_cache_misses: -1,
            l3_cache_misses: -1,
            tlb_misses: -1,
            branch_misses: -1,
            cpu_cycles: -1,
            instructions: -1,
            stalled_cycles_frontend: -1,
            stalled_cycles_backend: -1,
            stalled_backend_raw: -1,
            stalled_backend_mem: -1,
            enabled: false,
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for HardwareCounters {
    fn drop(&mut self) {
        close_fd(&mut self.l1_cache_misses);
        close_fd(&mut self.l2_cache_misses);
        close_fd(&mut self.l3_cache_misses);
        close_fd(&mut self.tlb_misses);
        close_fd(&mut self.branch_misses);
        close_fd(&mut self.cpu_cycles);
        close_fd(&mut self.instructions);
        close_fd(&mut self.stalled_cycles_frontend);
        close_fd(&mut self.stalled_cycles_backend);
        close_fd(&mut self.stalled_backend_raw);
        close_fd(&mut self.stalled_backend_mem);
        self.enabled = false;
    }
}

/// Specification for a PMU (Performance Monitoring Unit) event.
///
/// Loaded from /sys/bus/event_source/devices/*/events/* on Linux.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug)]
struct PmuEventSpec {
    type_: i32,
    config: u64,
    config1: u64,
    config2: u64,
}

/// Attribute structure for perf_event_open syscall.
///
/// Matches the kernel's `struct perf_event_attr` layout.
#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Clone, Copy)]
struct PerfEventAttr {
    type_: u32,
    size: u32,
    config: u64,
    sample_period_or_freq: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup_events: u32,
    bp_type: u32,
    config1: u64,
    config2: u64,
}

#[cfg(target_os = "linux")]
impl PerfEventAttr {
    fn new() -> Self {
        let mut attr = Self {
            type_: 0,
            size: std::mem::size_of::<Self>() as u32,
            config: 0,
            sample_period_or_freq: 0,
            sample_type: 0,
            read_format: PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING,
            flags: 0,
            wakeup_events: 0,
            bp_type: 0,
            config1: 0,
            config2: 0,
        };

        attr.flags |= PERF_ATTR_FLAG_DISABLED;
        attr.flags |= PERF_ATTR_FLAG_EXCLUDE_KERNEL;
        attr.flags |= PERF_ATTR_FLAG_EXCLUDE_HV;

        attr
    }
}

// =============================================================================
// Linux perf_event constants
// See: https://man7.org/linux/man-pages/man2/perf_event_open.2.html
// =============================================================================

#[cfg(target_os = "linux")]
const PERF_TYPE_HARDWARE: u32 = 0;
#[cfg(target_os = "linux")]
const PERF_TYPE_HW_CACHE: u32 = 3;

#[cfg(target_os = "linux")]
const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_BRANCH_MISSES: u64 = 5;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_STALLED_CYCLES_FRONTEND: u64 = 7;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_STALLED_CYCLES_BACKEND: u64 = 8;

#[cfg(target_os = "linux")]
const PERF_COUNT_HW_CACHE_L1D: u64 = 0;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_CACHE_LL: u64 = 2;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_CACHE_DTLB: u64 = 3;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_CACHE_NODE: u64 = 6;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_CACHE_OP_READ: u64 = 0;
#[cfg(target_os = "linux")]
const PERF_COUNT_HW_CACHE_RESULT_MISS: u64 = 1;

#[cfg(target_os = "linux")]
const PERF_FORMAT_TOTAL_TIME_ENABLED: u64 = 1 << 1;
#[cfg(target_os = "linux")]
const PERF_FORMAT_TOTAL_TIME_RUNNING: u64 = 1 << 2;

#[cfg(target_os = "linux")]
const PERF_ATTR_FLAG_DISABLED: u64 = 1 << 0;
#[cfg(target_os = "linux")]
const PERF_ATTR_FLAG_EXCLUDE_KERNEL: u64 = 1 << 5;
#[cfg(target_os = "linux")]
const PERF_ATTR_FLAG_EXCLUDE_HV: u64 = 1 << 6;

#[cfg(target_os = "linux")]
const IOC_NRBITS: u64 = 8;
#[cfg(target_os = "linux")]
const IOC_TYPEBITS: u64 = 8;
#[cfg(target_os = "linux")]
const IOC_SIZEBITS: u64 = 14;

#[cfg(target_os = "linux")]
const IOC_NRSHIFT: u64 = 0;
#[cfg(target_os = "linux")]
const IOC_TYPESHIFT: u64 = IOC_NRSHIFT + IOC_NRBITS;
#[cfg(target_os = "linux")]
const IOC_SIZESHIFT: u64 = IOC_TYPESHIFT + IOC_TYPEBITS;
#[cfg(target_os = "linux")]
const IOC_DIRSHIFT: u64 = IOC_SIZESHIFT + IOC_SIZEBITS;
#[cfg(target_os = "linux")]
const IOC_NONE: u64 = 0;

#[cfg(target_os = "linux")]
const fn ioc(dir: u64, type_: u64, nr: u64, size: u64) -> u64 {
    (dir << IOC_DIRSHIFT) | (type_ << IOC_TYPESHIFT) | (nr << IOC_NRSHIFT) | (size << IOC_SIZESHIFT)
}

#[cfg(target_os = "linux")]
const fn io(type_: u64, nr: u64) -> u64 {
    ioc(IOC_NONE, type_, nr, 0)
}

#[cfg(target_os = "linux")]
const PERF_EVENT_IOC_ENABLE: libc::c_ulong = io(b'$' as u64, 0) as libc::c_ulong;
#[cfg(target_os = "linux")]
const PERF_EVENT_IOC_DISABLE: libc::c_ulong = io(b'$' as u64, 1) as libc::c_ulong;
#[cfg(target_os = "linux")]
const PERF_EVENT_IOC_RESET: libc::c_ulong = io(b'$' as u64, 3) as libc::c_ulong;

/// Opens a perf event counter. Returns fd on success, -1 on failure.
#[cfg(target_os = "linux")]
fn perf_event_open(attr: &PerfEventAttr) -> RawFd {
    let ret = unsafe {
        libc::syscall(
            libc::SYS_perf_event_open,
            attr as *const PerfEventAttr,
            0 as libc::c_int,
            -1 as libc::c_int,
            -1 as libc::c_int,
            0 as libc::c_ulong,
        )
    };
    if ret < 0 { -1 } else { ret as RawFd }
}

#[cfg(target_os = "linux")]
fn close_fd(fd: &mut RawFd) {
    if *fd >= 0 {
        unsafe {
            libc::close(*fd);
        }
        *fd = -1;
    }
}

#[cfg(target_os = "linux")]
fn reset_and_enable_fd(fd: RawFd) {
    if fd >= 0 {
        unsafe {
            libc::ioctl(fd, PERF_EVENT_IOC_RESET, 0);
            libc::ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
    }
}

#[cfg(target_os = "linux")]
fn disable_fd(fd: RawFd) {
    if fd >= 0 {
        unsafe {
            libc::ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        }
    }
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct PerfRead {
    value: u64,
    time_enabled: u64,
    time_running: u64,
}

/// Reads a counter value, scaling for time-based multiplexing.
///
/// When multiple counters share limited hardware registers, the kernel
/// multiplexes them. This function extracts (value, time_enabled, time_running)
/// and scales: `value * time_enabled / time_running`.
#[cfg(target_os = "linux")]
fn read_scaled_counter(fd: RawFd) -> u64 {
    if fd < 0 {
        return 0;
    }
    let mut data = PerfRead {
        value: 0,
        time_enabled: 0,
        time_running: 0,
    };
    let bytes = unsafe {
        libc::read(
            fd,
            &mut data as *mut PerfRead as *mut libc::c_void,
            std::mem::size_of::<PerfRead>(),
        )
    };
    if bytes < 0 {
        return 0;
    }
    if bytes == std::mem::size_of::<u64>() as isize {
        return data.value;
    }
    if bytes != std::mem::size_of::<PerfRead>() as isize {
        return 0;
    }

    let mut scaled = data.value as f64;
    if data.time_running > 0 && data.time_enabled > data.time_running {
        scaled = (data.value as f64) * (data.time_enabled as f64) / (data.time_running as f64);
    }
    scaled as u64
}

#[cfg(target_os = "linux")]
fn parse_u64_base0(input: &str) -> Option<u64> {
    let value = input.trim();
    if value.is_empty() {
        return None;
    }
    let (radix, digits) = if let Some(rest) = value.strip_prefix("0x").or(value.strip_prefix("0X"))
    {
        (16, rest)
    } else if value.len() > 1 && value.starts_with('0') {
        (8, &value[1..])
    } else {
        (10, value)
    };
    u64::from_str_radix(digits, radix).ok()
}

#[cfg(target_os = "linux")]
fn parse_pmu_event_spec(spec: &str, out: &mut PmuEventSpec) -> bool {
    let mut has_config = false;
    for token in spec.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let mut iter = token.splitn(2, '=');
        let key = iter.next().unwrap_or("").trim();
        let value = iter.next().unwrap_or("").trim();
        if key.is_empty() || value.is_empty() {
            continue;
        }
        let parsed = match parse_u64_base0(value) {
            Some(parsed) => parsed,
            None => continue,
        };
        match key {
            "event" | "config" => {
                out.config = parsed;
                has_config = true;
            }
            "config1" => out.config1 = parsed,
            "config2" => out.config2 = parsed,
            _ => {}
        }
    }
    has_config
}

/// Loads a PMU event specification by name from sysfs.
///
/// Searches /sys/bus/event_source/devices/*/events/{event_name} for
/// architecture-specific event definitions (e.g., "l2d_cache_refill" on ARM).
#[cfg(target_os = "linux")]
fn load_pmu_event_spec(event_name: &str) -> Option<PmuEventSpec> {
    let devices_path = Path::new("/sys/bus/event_source/devices");
    let entries = fs::read_dir(devices_path).ok()?;
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        let base_path = entry.path();
        let event_path = base_path.join("events").join(event_name);
        let event_spec = match fs::read_to_string(&event_path) {
            Ok(spec) => spec,
            Err(_) => continue,
        };
        let event_spec = event_spec.lines().next().unwrap_or("").trim();
        if event_spec.is_empty() {
            continue;
        }

        let type_path = base_path.join("type");
        let type_str = match fs::read_to_string(&type_path) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let type_id = match type_str.trim().parse::<i32>() {
            Ok(value) => value,
            Err(_) => continue,
        };
        if type_id < 0 {
            continue;
        }

        let mut spec = PmuEventSpec {
            type_: type_id,
            config: 0,
            config1: 0,
            config2: 0,
        };
        if !parse_pmu_event_spec(event_spec, &mut spec) {
            continue;
        }
        return Some(spec);
    }
    None
}

#[cfg(target_os = "linux")]
fn open_pmu_event_by_name(event_name: &str, base_attr: &PerfEventAttr) -> RawFd {
    let spec = match load_pmu_event_spec(event_name) {
        Some(spec) => spec,
        None => return -1,
    };
    let mut attr = *base_attr;
    attr.type_ = spec.type_ as u32;
    attr.config = spec.config;
    attr.config1 = spec.config1;
    attr.config2 = spec.config2;
    perf_event_open(&attr)
}

#[cfg(target_os = "linux")]
fn open_pmu_event_by_name_list(names: &[&str], base_attr: &PerfEventAttr) -> RawFd {
    for name in names {
        let fd = open_pmu_event_by_name(name, base_attr);
        if fd >= 0 {
            return fd;
        }
    }
    -1
}

#[cfg(not(target_os = "linux"))]
#[derive(Debug)]
struct HardwareCounters;

#[cfg(not(target_os = "linux"))]
impl HardwareCounters {
    fn new(_enable: bool) -> Self {
        Self
    }

    fn available(&self) -> bool {
        false
    }

    fn reset_and_enable(&self) {}

    fn disable(&self) {}

    fn read(&self) -> HardwareCounterSnapshot {
        HardwareCounterSnapshot::default()
    }
}

/// Statistical functions for benchmark analysis.
struct StatisticalAnalysis;

impl StatisticalAnalysis {
    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn standard_deviation(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = Self::mean(values);
        let mut variance = 0.0;
        for v in values {
            variance += (v - mean) * (v - mean);
        }
        variance /= (values.len() - 1) as f64;
        variance.sqrt()
    }

    fn median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Computes the p-th percentile using linear interpolation.
    ///
    /// `p` is in [0, 1]. For p95, pass 0.95.
    fn percentile(values: &[f64], p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if p <= 0.0 {
            return sorted[0];
        }
        if p >= 1.0 {
            return sorted[sorted.len() - 1];
        }
        let pos = p * (sorted.len() as f64 - 1.0);
        let i = pos.floor() as usize;
        let frac = pos - i as f64;
        if i + 1 >= sorted.len() {
            return sorted[i];
        }
        sorted[i] * (1.0 - frac) + sorted[i + 1] * frac
    }
}

/// Executes benchmarks with optional hardware counter profiling.
struct BenchmarkRunner {
    enable_hardware_counters: bool,
}

impl BenchmarkRunner {
    fn new() -> Self {
        Self {
            enable_hardware_counters: cfg!(target_os = "linux"),
        }
    }

    fn set_enable_hardware_counters(&mut self, enable: bool) {
        self.enable_hardware_counters = if cfg!(target_os = "linux") {
            enable
        } else {
            false
        };
    }

    fn hardware_counters_enabled(&self) -> bool {
        self.enable_hardware_counters
    }

    /// Runs a benchmark: setup once, warmup iterations, then timed iterations.
    ///
    /// Returns timing statistics only (no hardware counters).
    fn run_benchmark<S, B>(
        &self,
        test_name: &str,
        mut setup_fn: S,
        mut bench_fn: B,
        iterations: usize,
        warmups: usize,
    ) -> BenchmarkResult
    where
        S: FnMut(),
        B: FnMut(),
    {
        let mut result = BenchmarkResult::new(test_name, iterations);
        let mut times = Vec::with_capacity(iterations);

        setup_fn();

        for _ in 0..warmups {
            bench_fn();
        }

        for _ in 0..iterations {
            let start = Instant::now();
            bench_fn();
            let duration = start.elapsed();
            times.push(duration_to_ms(duration));
        }

        result.mean_time_ms = StatisticalAnalysis::mean(&times);
        result.median_time_ms = StatisticalAnalysis::median(&times);
        result.standard_deviation = StatisticalAnalysis::standard_deviation(&times);
        result.min_time_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
        result.max_time_ms = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        result.p95_time_ms = StatisticalAnalysis::percentile(&times, 0.95);

        result
    }

    /// Runs a benchmark with hardware counter profiling on one iteration.
    ///
    /// Flow: setup → warmup → (enable counters → 1 iteration → read counters) →
    /// extra warmup → timed iterations. Hardware counters are only collected
    /// for one iteration to avoid measurement overhead affecting timing.
    fn run_benchmark_with_memory_profiling<S, B>(
        &self,
        test_name: &str,
        mut setup_fn: S,
        mut bench_fn: B,
        iterations: usize,
        warmups: usize,
    ) -> BenchmarkResult
    where
        S: FnMut(),
        B: FnMut(),
    {
        let mut result = BenchmarkResult::new(test_name, iterations);
        let mut times = Vec::with_capacity(iterations);

        setup_fn();

        for _ in 0..warmups {
            bench_fn();
        }

        let counters = HardwareCounters::new(self.enable_hardware_counters);
        let snapshot = if counters.available() {
            counters.reset_and_enable();
            bench_fn();
            counters.disable();
            counters.read()
        } else {
            bench_fn();
            HardwareCounterSnapshot::default()
        };

        result.l1_cache_misses = snapshot.l1_cache_misses;
        result.l2_cache_misses = snapshot.l2_cache_misses;
        result.l3_cache_misses = snapshot.l3_cache_misses;
        result.tlb_misses = snapshot.tlb_misses;
        result.branch_misses = snapshot.branch_misses;
        result.cpu_cycles = snapshot.cpu_cycles;
        result.instructions = snapshot.instructions;
        result.stalled_cycles_frontend = snapshot.stalled_cycles_frontend;
        result.stalled_cycles_backend = snapshot.stalled_cycles_backend;
        result.stalled_cycles_backend_mem = snapshot.stalled_cycles_backend_mem;
        result.instructions_per_cycle = if snapshot.cpu_cycles > 0 {
            snapshot.instructions as f64 / snapshot.cpu_cycles as f64
        } else {
            0.0
        };

        for _ in 0..2 {
            bench_fn();
        }

        for _ in 0..iterations {
            let start = Instant::now();
            bench_fn();
            let duration = start.elapsed();
            times.push(duration_to_ms(duration));
        }

        result.mean_time_ms = StatisticalAnalysis::mean(&times);
        result.median_time_ms = StatisticalAnalysis::median(&times);
        result.standard_deviation = StatisticalAnalysis::standard_deviation(&times);
        result.min_time_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
        result.max_time_ms = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        result.p95_time_ms = StatisticalAnalysis::percentile(&times, 0.95);

        result
    }
}

/// Configuration for a rotation benchmark run.
#[derive(Clone, Copy)]
struct RotationBenchmarkConfig {
    format: ImageFormat,
    angle: f32,
    iterations: usize,
    warmups: usize,
    use_memory_profiling: bool,
}

/// Orchestrates the full benchmark suite with filtering and output formatting.
struct BenchmarkSuite {
    runner: BenchmarkRunner,
    results: Vec<BenchmarkResult>,
    filter: String,
    hardware_counters_enabled: bool,
}

impl BenchmarkSuite {
    fn new() -> Self {
        Self {
            runner: BenchmarkRunner::new(),
            results: Vec::new(),
            filter: String::new(),
            hardware_counters_enabled: cfg!(target_os = "linux"),
        }
    }

    fn set_filter(&mut self, filter: String) {
        self.filter = filter;
    }

    fn set_hardware_counters(&mut self, enable: bool) {
        let effective = if cfg!(target_os = "linux") {
            enable
        } else {
            false
        };
        self.hardware_counters_enabled = effective;
        self.runner.set_enable_hardware_counters(effective);
    }

    fn should_run_test(&self, test_name: &str) -> bool {
        if self.filter.is_empty() {
            return true;
        }
        test_name == self.filter || test_name.contains(&self.filter)
    }

    fn run_all(&mut self, json_output: bool, output_file: Option<String>) -> io::Result<()> {
        if !json_output {
            println!("=== OpRotateImage Benchmark Suite ===\n");
        }

        self.benchmark_by_size(json_output);
        self.benchmark_by_format(json_output);
        self.benchmark_by_angle(json_output);
        self.benchmark_fast_path_vs_interpolation(json_output);
        self.benchmark_rgba_interpolation(json_output);

        if json_output {
            let json = self.to_json();
            if let Some(path) = output_file {
                let mut file = File::create(path)?;
                file.write_all(json.as_bytes())?;
            } else {
                println!("{json}");
            }
        } else {
            self.print_report();
        }

        Ok(())
    }

    fn list_tests(&self) {
        let mut names = Vec::new();
        names.extend(self.test_names_by_size());
        names.extend(self.test_names_by_format());
        names.extend(self.test_names_by_angle());
        names.extend(self.test_names_fast_path_vs_interpolation());
        names.extend(self.test_names_rgba_interpolation());
        names.sort();
        for name in names {
            println!("{name}");
        }
    }

    /// Runs a single rotation benchmark with preallocated output buffer.
    ///
    /// Uses RefCell to share state between setup and benchmark closures
    /// while satisfying the borrow checker.
    fn run_rotation_benchmark(
        &self,
        test_name: &str,
        size: usize,
        config: RotationBenchmarkConfig,
    ) -> (BenchmarkResult, Image) {
        let RotationBenchmarkConfig {
            format,
            angle,
            iterations,
            warmups,
            use_memory_profiling,
        } = config;
        let input = RefCell::new(Image::new_empty());
        let output = RefCell::new(Image::new_empty());
        let rotate_op = RefCell::new(OpRotateImage::new());

        let setup = || {
            *input.borrow_mut() = bench_utils::create_test_image(size, size, format);
            let mut rotate = rotate_op.borrow_mut();
            rotate.set_rotation(angle, RotateDirection::Cw);
            let (out_w, out_h) = rotate.compute_output_dimensions(&input.borrow());
            let premul = input.borrow().is_premultiplied();
            *output.borrow_mut() = Image::with_premultiplied(out_w, out_h, format, premul);
            rotate.reserve_scratch(out_h);
            #[cfg(target_os = "linux")]
            {
                input.borrow().advise_hugepage();
                output.borrow().advise_hugepage();
            }
        };

        let bench = || {
            let input_ref = input.borrow();
            let mut output_ref = output.borrow_mut();
            let mut rotate = rotate_op.borrow_mut();
            rotate.apply_to_preallocated(&input_ref, &mut output_ref);
        };

        let result = if use_memory_profiling {
            self.runner
                .run_benchmark_with_memory_profiling(test_name, setup, bench, iterations, warmups)
        } else {
            self.runner
                .run_benchmark(test_name, setup, bench, iterations, warmups)
        };

        let output_image = output.borrow().clone();
        (result, output_image)
    }

    /// Measures scaling behavior from 256×256 to 4096×4096.
    ///
    /// All tests use RGB format at 45° to isolate size effects.
    fn benchmark_by_size(&mut self, silent: bool) {
        if !silent {
            println!("--- Benchmark: Size Scaling (RGB, 45 degrees) ---");
        }

        let sizes = bench_utils::BENCH_SIZES;
        let angle = 45.0_f32;
        let format = ImageFormat::Rgb;

        for size in sizes {
            let test_name = format!("Size_{}x{}", size, size);
            if !self.should_run_test(&test_name) {
                continue;
            }

            let config = RotationBenchmarkConfig {
                format,
                angle,
                iterations: 50,
                warmups: 2,
                use_memory_profiling: true,
            };
            let (mut result, output_image) = self.run_rotation_benchmark(&test_name, size, config);

            self.fill_result_metrics(&mut result, size, size, format, angle, &output_image);
            self.results.push(result.clone());

            if !silent {
                println!(
                    "  {}x{}: {:.2} ms - {:.1} MP/s",
                    size, size, result.compute_only_time_ms, result.megapixels_per_second
                );
            }
        }

        if !silent {
            println!();
        }
    }

    /// Compares Gray vs RGB vs RGBA performance at fixed 512×512 size.
    ///
    /// Shows per-channel overhead of the interpolation kernel.
    fn benchmark_by_format(&mut self, silent: bool) {
        if !silent {
            println!("--- Benchmark: Pixel Format (512x512, 45 degrees) ---");
        }

        let formats = bench_utils::BENCH_FORMATS;
        let size = 512_usize;
        let angle = 45.0_f32;

        for format in formats {
            let test_name = format!("Format_{}_512x512", bench_utils::format_to_string(format));
            if !self.should_run_test(&test_name) {
                continue;
            }

            let config = RotationBenchmarkConfig {
                format,
                angle,
                iterations: 50,
                warmups: 2,
                use_memory_profiling: true,
            };
            let (mut result, output_image) = self.run_rotation_benchmark(&test_name, size, config);

            self.fill_result_metrics(&mut result, size, size, format, angle, &output_image);
            self.results.push(result.clone());

            if !silent {
                println!(
                    "  {}: {:.2} ms - {:.1} MP/s",
                    bench_utils::format_to_string(format),
                    result.compute_only_time_ms,
                    result.megapixels_per_second
                );
            }
        }

        if !silent {
            println!();
        }
    }

    /// Compares rotation angles including fast-path (90°/180°/270°) vs interpolation.
    ///
    /// Fast-path angles use simple coordinate remapping without interpolation.
    fn benchmark_by_angle(&mut self, silent: bool) {
        if !silent {
            println!("--- Benchmark: Rotation Angles (512x512, RGB) ---");
        }

        let angles = bench_utils::BENCH_ANGLES;
        let size = 512_usize;
        let format = ImageFormat::Rgb;

        for angle in angles {
            let fast = (angle % 90.0) == 0.0;
            let suffix = if fast { "_fast" } else { "_interp" };
            let test_name = format!("Angle_{}deg{}", angle as i32, suffix);
            if !self.should_run_test(&test_name) {
                continue;
            }

            let config = RotationBenchmarkConfig {
                format,
                angle,
                iterations: 50,
                warmups: 2,
                use_memory_profiling: true,
            };
            let (mut result, output_image) = self.run_rotation_benchmark(&test_name, size, config);

            self.fill_result_metrics(&mut result, size, size, format, angle, &output_image);
            self.results.push(result.clone());

            if !silent {
                let tag = if fast { "[fast]" } else { "[cubic]" };
                println!(
                    "  {:>3} deg {}: {:.3} ms - {:.1} MP/s",
                    angle as i32, tag, result.compute_only_time_ms, result.megapixels_per_second
                );
            }
        }

        if !silent {
            println!();
        }
    }

    /// Direct comparison of 90° (fast-path) vs 45° (bicubic) at multiple sizes.
    ///
    /// Demonstrates the performance benefit of the optimized 90° rotation path.
    fn benchmark_fast_path_vs_interpolation(&mut self, silent: bool) {
        if !silent {
            println!("--- Benchmark: Fast Path (90 deg) vs Interpolation (45 deg) ---");
        }

        let sizes = bench_utils::BENCH_SIZES;
        let format = ImageFormat::Rgb;

        for size in sizes {
            let test_name = format!("FastPath_90deg_{}", size);
            if self.should_run_test(&test_name) {
                let config = RotationBenchmarkConfig {
                    format,
                    angle: 90.0,
                    iterations: 50,
                    warmups: 3,
                    use_memory_profiling: false,
                };
                let (mut result, output_image) =
                    self.run_rotation_benchmark(&test_name, size, config);

                self.fill_result_metrics(&mut result, size, size, format, 90.0, &output_image);
                self.results.push(result.clone());

                if !silent {
                    println!(
                        "  {}x{} @ 90 deg: {:.3} ms - {:.1} MP/s",
                        size, size, result.compute_only_time_ms, result.megapixels_per_second
                    );
                }
            }

            let test_name = format!("Interpolation_45deg_{}", size);
            if self.should_run_test(&test_name) {
                let config = RotationBenchmarkConfig {
                    format,
                    angle: 45.0,
                    iterations: 50,
                    warmups: 3,
                    use_memory_profiling: false,
                };
                let (mut result, output_image) =
                    self.run_rotation_benchmark(&test_name, size, config);

                self.fill_result_metrics(&mut result, size, size, format, 45.0, &output_image);
                self.results.push(result.clone());

                if !silent {
                    println!(
                        "  {}x{} @ 45 deg: {:.3} ms - {:.1} MP/s",
                        size, size, result.compute_only_time_ms, result.megapixels_per_second
                    );
                }
            }
        }

        if !silent {
            println!();
        }
    }

    /// RGBA-specific benchmarks to measure the specialized 4-channel path.
    ///
    /// Tests the `paint_image_cubic_interior_rgba` optimization.
    fn benchmark_rgba_interpolation(&mut self, silent: bool) {
        if !silent {
            println!("--- Benchmark: RGBA Interpolation (45 degrees) ---");
        }

        let sizes = bench_utils::BENCH_SIZES;
        let angle = 45.0_f32;
        let format = ImageFormat::Rgba;

        for size in sizes {
            let test_name = format!("Interp_RGBA_45deg_{}", size);
            if !self.should_run_test(&test_name) {
                continue;
            }

            let config = RotationBenchmarkConfig {
                format,
                angle,
                iterations: 50,
                warmups: 2,
                use_memory_profiling: true,
            };
            let (mut result, output_image) = self.run_rotation_benchmark(&test_name, size, config);

            self.fill_result_metrics(&mut result, size, size, format, angle, &output_image);
            self.results.push(result.clone());

            if !silent {
                println!(
                    "  {}x{} RGBA: {:.2} ms - {:.1} MP/s",
                    size, size, result.compute_only_time_ms, result.megapixels_per_second
                );
            }
        }

        if !silent {
            println!();
        }
    }

    /// Populates derived metrics: throughput (MP/s), output size, format name.
    fn fill_result_metrics(
        &self,
        result: &mut BenchmarkResult,
        image_width: usize,
        image_height: usize,
        format: ImageFormat,
        rotation_angle: f32,
        output_image: &Image,
    ) {
        result.image_width = image_width;
        result.image_height = image_height;
        result.format_name = bench_utils::format_to_string(format).to_string();
        result.rotation_angle = rotation_angle;
        result.output_image_bytes = output_image.width()
            * output_image.height()
            * output_image.channels()
            * std::mem::size_of::<u16>();

        result.compute_only_time_ms = result.mean_time_ms;
        result.allocation_overhead_ms = 0.0;

        let total_pixels = (image_width * image_height) as f64;
        let compute_only_ms = result.compute_only_time_ms.max(0.000001);
        result.pixels_per_second = (total_pixels / compute_only_ms) * 1000.0;
        result.megapixels_per_second = result.pixels_per_second / 1_000_000.0;

        if !self.hardware_counters_enabled || !self.runner.hardware_counters_enabled() {
            result.cache_hit_ratio = 0.0;
        }
    }

    fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"results\": [\n");
        for (idx, r) in self.results.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!("      \"testName\": \"{}\",\n", r.test_name));
            out.push_str(&format!("      \"meanTimeMs\": {:.3},\n", r.mean_time_ms));
            out.push_str(&format!(
                "      \"medianTimeMs\": {:.3},\n",
                r.median_time_ms
            ));
            out.push_str(&format!(
                "      \"standardDeviation\": {:.3},\n",
                r.standard_deviation
            ));
            out.push_str(&format!("      \"minTimeMs\": {:.3},\n", r.min_time_ms));
            out.push_str(&format!("      \"maxTimeMs\": {:.3},\n", r.max_time_ms));
            out.push_str(&format!("      \"p95TimeMs\": {:.3},\n", r.p95_time_ms));
            out.push_str(&format!("      \"iterations\": {},\n", r.iterations));
            out.push_str(&format!("      \"imageWidth\": {},\n", r.image_width));
            out.push_str(&format!("      \"imageHeight\": {},\n", r.image_height));
            out.push_str(&format!("      \"formatName\": \"{}\",\n", r.format_name));
            out.push_str(&format!(
                "      \"rotationAngle\": {:.1},\n",
                r.rotation_angle
            ));
            out.push_str(&format!(
                "      \"peakMemoryBytes\": {},\n",
                r.peak_memory_bytes
            ));
            out.push_str(&format!(
                "      \"outputImageBytes\": {},\n",
                r.output_image_bytes
            ));
            out.push_str(&format!(
                "      \"totalAllocations\": {},\n",
                r.total_allocations
            ));
            out.push_str(&format!(
                "      \"l1CacheMisses\": {},\n",
                r.l1_cache_misses
            ));
            out.push_str(&format!(
                "      \"l2CacheMisses\": {},\n",
                r.l2_cache_misses
            ));
            out.push_str(&format!(
                "      \"l3CacheMisses\": {},\n",
                r.l3_cache_misses
            ));
            out.push_str(&format!("      \"tlbMisses\": {},\n", r.tlb_misses));
            out.push_str(&format!("      \"branchMisses\": {},\n", r.branch_misses));
            out.push_str(&format!(
                "      \"cacheHitRatio\": {:.4},\n",
                r.cache_hit_ratio
            ));
            out.push_str(&format!("      \"cpuCycles\": {},\n", r.cpu_cycles));
            out.push_str(&format!("      \"instructions\": {},\n", r.instructions));
            out.push_str(&format!(
                "      \"stalledCyclesFrontend\": {},\n",
                r.stalled_cycles_frontend
            ));
            out.push_str(&format!(
                "      \"stalledCyclesBackend\": {},\n",
                r.stalled_cycles_backend
            ));
            out.push_str(&format!(
                "      \"stalledCyclesBackendMem\": {},\n",
                r.stalled_cycles_backend_mem
            ));
            out.push_str(&format!(
                "      \"instructionsPerCycle\": {:.4},\n",
                r.instructions_per_cycle
            ));
            out.push_str(&format!(
                "      \"pixelsPerSecond\": {:.1},\n",
                r.pixels_per_second
            ));
            out.push_str(&format!(
                "      \"megapixelsPerSecond\": {:.2},\n",
                r.megapixels_per_second
            ));
            out.push_str(&format!(
                "      \"allocationOverheadMs\": {:.3},\n",
                r.allocation_overhead_ms
            ));
            out.push_str(&format!(
                "      \"computeOnlyTimeMs\": {:.3}\n",
                r.compute_only_time_ms
            ));
            out.push_str("    }");
            if idx + 1 < self.results.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]\n");
        out.push_str("}\n");
        out
    }

    fn print_report(&self) {
        println!("================================================================");
        println!("                    DETAILED RESULTS");
        println!("================================================================\n");
        println!(
            "{:<35} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}",
            "Test Name", "Mean (ms)", "Median", "Std Dev", "Min", "Max", "P95", "MP/s"
        );
        println!("{}", "-".repeat(119));
        for r in &self.results {
            println!(
                "{:<35} {:<12.3} {:<12.3} {:<12.3} {:<12.3} {:<12.3} {:<12.3} {:<12.1}",
                r.test_name,
                r.mean_time_ms,
                r.median_time_ms,
                r.standard_deviation,
                r.min_time_ms,
                r.max_time_ms,
                r.p95_time_ms,
                r.megapixels_per_second
            );
        }

        println!();
        println!("--- Hardware Performance Counters (for profiled tests) ---");
        if !self.hardware_counters_enabled {
            println!("  (hardware counters disabled)");
        } else if !cfg!(target_os = "linux") {
            println!("  (hardware counters unavailable on this platform)");
        } else {
            let mut printed = false;
            for r in &self.results {
                if r.cpu_cycles == 0
                    && r.l1_cache_misses == 0
                    && r.l2_cache_misses == 0
                    && r.l3_cache_misses == 0
                    && r.tlb_misses == 0
                    && r.branch_misses == 0
                    && r.stalled_cycles_frontend == 0
                    && r.stalled_cycles_backend == 0
                    && r.stalled_cycles_backend_mem == 0
                {
                    continue;
                }
                printed = true;
                println!("  {}:", r.test_name);
                if r.cpu_cycles > 0 {
                    println!("    CPU Cycles: {}", r.cpu_cycles);
                    println!("    Instructions: {}", r.instructions);
                    println!("    IPC: {:.2}", r.instructions_per_cycle);
                    if r.stalled_cycles_frontend > 0
                        || r.stalled_cycles_backend > 0
                        || r.stalled_cycles_backend_mem > 0
                    {
                        println!(
                            "    Stalled Cycles (Frontend): {}",
                            r.stalled_cycles_frontend
                        );
                        println!("    Stalled Cycles (Backend): {}", r.stalled_cycles_backend);
                        if r.stalled_cycles_backend_mem > 0 {
                            println!(
                                "    Stalled Cycles (Backend Mem): {}",
                                r.stalled_cycles_backend_mem
                            );
                        }
                    }
                }
                if r.l1_cache_misses > 0
                    || r.l2_cache_misses > 0
                    || r.l3_cache_misses > 0
                    || r.tlb_misses > 0
                    || r.branch_misses > 0
                {
                    println!("    L1 Cache Misses: {}", r.l1_cache_misses);
                    println!("    L2 Cache Misses: {}", r.l2_cache_misses);
                    if r.l3_cache_misses > 0 {
                        println!("    L3 Cache Misses: {}", r.l3_cache_misses);
                    }
                    println!("    TLB Misses: {}", r.tlb_misses);
                    println!("    Branch Misses: {}", r.branch_misses);
                }
            }
            if !printed {
                println!("  (hardware counters unavailable or permission denied)");
            }
        }
    }

    fn test_names_by_size(&self) -> Vec<String> {
        let sizes = bench_utils::BENCH_SIZES;
        sizes.iter().map(|s| format!("Size_{}x{}", s, s)).collect()
    }

    fn test_names_by_format(&self) -> Vec<String> {
        let formats = bench_utils::BENCH_FORMATS;
        formats
            .iter()
            .map(|f| format!("Format_{}_512x512", bench_utils::format_to_string(*f)))
            .collect()
    }

    fn test_names_by_angle(&self) -> Vec<String> {
        let angles = bench_utils::BENCH_ANGLES;
        angles
            .iter()
            .map(|angle| {
                let fast = (angle % 90.0) == 0.0;
                let suffix = if fast { "_fast" } else { "_interp" };
                format!("Angle_{}deg{}", *angle as i32, suffix)
            })
            .collect()
    }

    fn test_names_fast_path_vs_interpolation(&self) -> Vec<String> {
        let sizes = bench_utils::BENCH_SIZES;
        let mut names = Vec::new();
        for size in sizes {
            names.push(format!("FastPath_90deg_{}", size));
            names.push(format!("Interpolation_45deg_{}", size));
        }
        names
    }

    fn test_names_rgba_interpolation(&self) -> Vec<String> {
        let sizes = bench_utils::BENCH_SIZES;
        sizes
            .iter()
            .map(|s| format!("Interp_RGBA_45deg_{}", s))
            .collect()
    }
}

fn duration_to_ms(dur: Duration) -> f64 {
    dur.as_secs_f64() * 1000.0
}

fn print_usage(program: &str) {
    eprintln!("Usage: {program} [options]");
    eprintln!("Options:");
    eprintln!("  --json              Output results in JSON format");
    eprintln!("  --output <file>     Write results to file (default: stdout)");
    eprintln!("  --filter <pattern>  Run only tests matching pattern");
    eprintln!("  --list-tests        List all available tests");
    eprintln!("  --disable-hw-counters  Disable hardware counters (Linux only)");
    eprintln!("  --help, -h          Show this help");
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut json_output = false;
    let mut output_file: Option<String> = None;
    let mut filter_pattern: Option<String> = None;
    let mut list_tests = false;
    let mut disable_hw = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => {
                json_output = true;
                i += 1;
            }
            "--output" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --output requires a filename argument");
                    print_usage(&args[0]);
                    std::process::exit(1);
                }
                output_file = Some(args[i + 1].clone());
                i += 2;
            }
            "--filter" => {
                if i + 1 >= args.len() {
                    eprintln!("Error: --filter requires a pattern argument");
                    print_usage(&args[0]);
                    std::process::exit(1);
                }
                filter_pattern = Some(args[i + 1].clone());
                i += 2;
            }
            "--list-tests" => {
                list_tests = true;
                i += 1;
            }
            "--disable-hw-counters" => {
                disable_hw = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage(&args[0]);
                return Ok(());
            }
            other => {
                eprintln!("Unknown option: {other}");
                print_usage(&args[0]);
                std::process::exit(1);
            }
        }
    }

    let mut suite = BenchmarkSuite::new();
    if disable_hw {
        suite.set_hardware_counters(false);
    }

    if list_tests {
        suite.list_tests();
        return Ok(());
    }

    if let Some(filter) = filter_pattern {
        suite.set_filter(filter);
    }

    suite.run_all(json_output, output_file)
}
