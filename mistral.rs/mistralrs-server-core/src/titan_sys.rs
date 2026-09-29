//! System readings for the monitor page: NVML (loaded at runtime, optional) and /proc.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

/// `TITAN_MONITOR_NVML=0` turns NVML off (the GPU panel then reads "unavailable").
pub const NVML_ENV: &str = "TITAN_MONITOR_NVML";
/// `TITAN_MONITOR_GPU=<nvml index>`, default 0.
pub const GPU_INDEX_ENV: &str = "TITAN_MONITOR_GPU";
const NVML_LIB: &[u8] = b"libnvidia-ml.so.1\0";
const RTLD_NOW: c_int = 2;
const NVML_SUCCESS: c_int = 0;
const NVML_TEMPERATURE_GPU: c_uint = 0;
const NVML_CLOCK_GRAPHICS: c_uint = 0;
const NVML_CLOCK_SM: c_uint = 1;
const NVML_CLOCK_MEM: c_uint = 2;
const NVML_PCIE_TX: c_uint = 0;
const NVML_PCIE_RX: c_uint = 1;
const SECTOR_BYTES: u64 = 512;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

type Dev = *mut c_void;

#[repr(C)]
#[derive(Default)]
struct Utilization {
    gpu: c_uint,
    memory: c_uint,
}

#[repr(C)]
#[derive(Default)]
struct Memory {
    total: u64,
    free: u64,
    used: u64,
}

type FnU32 = unsafe extern "C" fn(Dev, *mut c_uint) -> c_int;
type FnArgU32 = unsafe extern "C" fn(Dev, c_uint, *mut c_uint) -> c_int;

pub struct Nvml {
    dev: Dev,
    pub name: String,
    util: Option<unsafe extern "C" fn(Dev, *mut Utilization) -> c_int>,
    mem: Option<unsafe extern "C" fn(Dev, *mut Memory) -> c_int>,
    temp: Option<FnArgU32>,
    power: Option<FnU32>,
    power_limit: Option<FnU32>,
    clock: Option<FnArgU32>,
    max_clock: Option<FnArgU32>,
    link_gen: Option<FnU32>,
    link_width: Option<FnU32>,
    max_link_gen: Option<FnU32>,
    max_link_width: Option<FnU32>,
    pcie_tp: Option<FnArgU32>,
    pstate: Option<FnU32>,
}

// SAFETY: NVML handles are thread-safe; the sampler thread is the only user anyway.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

#[derive(Clone, Debug, Default)]
pub struct GpuSample {
    pub util_pct: Option<u32>,
    pub mem_util_pct: Option<u32>,
    pub vram_used: Option<u64>,
    pub vram_total: Option<u64>,
    pub temp_c: Option<u32>,
    pub power_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub clock_gr_mhz: Option<u32>,
    pub clock_sm_mhz: Option<u32>,
    pub clock_mem_mhz: Option<u32>,
    pub clock_sm_max_mhz: Option<u32>,
    pub pstate: Option<u32>,
    pub link_gen: Option<u32>,
    pub link_width: Option<u32>,
    pub max_link_gen: Option<u32>,
    pub max_link_width: Option<u32>,
    pub pcie_tx_kbs: Option<u32>,
    pub pcie_rx_kbs: Option<u32>,
}

unsafe fn sym<T: Copy>(lib: *mut c_void, name: &[u8]) -> Option<T> {
    let p = dlsym(lib, name.as_ptr().cast());
    if p.is_null() {
        None
    } else {
        Some(std::mem::transmute_copy(&p))
    }
}

impl Nvml {
    pub fn load() -> Result<Self, String> {
        if std::env::var(NVML_ENV).is_ok_and(|v| v == "0") {
            return Err(format!("disabled ({NVML_ENV}=0)"));
        }
        let index: c_uint = std::env::var(GPU_INDEX_ENV)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        unsafe {
            let lib = dlopen(NVML_LIB.as_ptr().cast(), RTLD_NOW);
            if lib.is_null() {
                return Err("libnvidia-ml.so.1 not found".into());
            }
            let init: unsafe extern "C" fn() -> c_int =
                sym(lib, b"nvmlInit_v2\0").ok_or("nvmlInit_v2 missing")?;
            let rc = init();
            if rc != NVML_SUCCESS {
                return Err(format!("nvmlInit_v2 failed ({rc})"));
            }
            let by_index: unsafe extern "C" fn(c_uint, *mut Dev) -> c_int =
                sym(lib, b"nvmlDeviceGetHandleByIndex_v2\0")
                    .ok_or("nvmlDeviceGetHandleByIndex_v2 missing")?;
            let mut dev: Dev = std::ptr::null_mut();
            let rc = by_index(index, &mut dev);
            if rc != NVML_SUCCESS {
                return Err(format!("no NVML device {index} ({rc})"));
            }
            let mut name = String::from("GPU");
            if let Some(get_name) = sym::<unsafe extern "C" fn(Dev, *mut c_char, c_uint) -> c_int>(
                lib,
                b"nvmlDeviceGetName\0",
            ) {
                let mut buf = [0 as c_char; 96];
                if get_name(dev, buf.as_mut_ptr(), buf.len() as c_uint) == NVML_SUCCESS {
                    name = CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned();
                }
            }
            Ok(Self {
                dev,
                name,
                util: sym(lib, b"nvmlDeviceGetUtilizationRates\0"),
                mem: sym(lib, b"nvmlDeviceGetMemoryInfo\0"),
                temp: sym(lib, b"nvmlDeviceGetTemperature\0"),
                power: sym(lib, b"nvmlDeviceGetPowerUsage\0"),
                power_limit: sym(lib, b"nvmlDeviceGetEnforcedPowerLimit\0"),
                clock: sym(lib, b"nvmlDeviceGetClockInfo\0"),
                max_clock: sym(lib, b"nvmlDeviceGetMaxClockInfo\0"),
                link_gen: sym(lib, b"nvmlDeviceGetCurrPcieLinkGeneration\0"),
                link_width: sym(lib, b"nvmlDeviceGetCurrPcieLinkWidth\0"),
                max_link_gen: sym(lib, b"nvmlDeviceGetMaxPcieLinkGeneration\0"),
                max_link_width: sym(lib, b"nvmlDeviceGetMaxPcieLinkWidth\0"),
                pcie_tp: sym(lib, b"nvmlDeviceGetPcieThroughput\0"),
                pstate: sym(lib, b"nvmlDeviceGetPerformanceState\0"),
            })
        }
    }

    fn u32_of(&self, f: Option<FnU32>) -> Option<u32> {
        let mut v: c_uint = 0;
        (unsafe { f?(self.dev, &mut v) } == NVML_SUCCESS).then_some(v)
    }

    fn u32_arg(&self, f: Option<FnArgU32>, arg: c_uint) -> Option<u32> {
        let mut v: c_uint = 0;
        (unsafe { f?(self.dev, arg, &mut v) } == NVML_SUCCESS).then_some(v)
    }

    /// One reading; the PCIe throughput counters each take a ~20 ms sample inside NVML.
    pub fn sample(&self) -> GpuSample {
        let mut s = GpuSample::default();
        if let Some(f) = self.util {
            let mut u = Utilization::default();
            if unsafe { f(self.dev, &mut u) } == NVML_SUCCESS {
                s.util_pct = Some(u.gpu);
                s.mem_util_pct = Some(u.memory);
            }
        }
        if let Some(f) = self.mem {
            let mut m = Memory::default();
            if unsafe { f(self.dev, &mut m) } == NVML_SUCCESS {
                s.vram_used = Some(m.used);
                s.vram_total = Some(m.total);
            }
        }
        s.temp_c = self.u32_arg(self.temp, NVML_TEMPERATURE_GPU);
        s.power_w = self.u32_of(self.power).map(|mw| f64::from(mw) / 1e3);
        s.power_limit_w = self.u32_of(self.power_limit).map(|mw| f64::from(mw) / 1e3);
        s.clock_gr_mhz = self.u32_arg(self.clock, NVML_CLOCK_GRAPHICS);
        s.clock_sm_mhz = self.u32_arg(self.clock, NVML_CLOCK_SM);
        s.clock_mem_mhz = self.u32_arg(self.clock, NVML_CLOCK_MEM);
        s.clock_sm_max_mhz = self.u32_arg(self.max_clock, NVML_CLOCK_SM);
        s.pstate = self.u32_of(self.pstate);
        s.link_gen = self.u32_of(self.link_gen);
        s.link_width = self.u32_of(self.link_width);
        s.max_link_gen = self.u32_of(self.max_link_gen);
        s.max_link_width = self.u32_of(self.max_link_width);
        s.pcie_tx_kbs = self.u32_arg(self.pcie_tp, NVML_PCIE_TX);
        s.pcie_rx_kbs = self.u32_arg(self.pcie_tp, NVML_PCIE_RX);
        s
    }
}

/// Cumulative /proc counters at one moment.
#[derive(Clone, Debug, Default)]
pub struct ProcCounters {
    /// Aggregate jiffies: (busy, total).
    pub cpu: (u64, u64),
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub disk_names: Vec<String>,
    /// This process's storage reads (/proc/self/io read_bytes).
    pub proc_read_bytes: u64,
    pub mem_total: u64,
    pub mem_available: u64,
    pub rss: u64,
    pub rss_anon: u64,
    pub rss_file: u64,
}

fn cpu_jiffies(stat: &str) -> (u64, u64) {
    let Some(line) = stat.lines().find(|l| l.starts_with("cpu ")) else {
        return (0, 0);
    };
    let v: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    let total: u64 = v.iter().take(8).sum();
    let idle = v.get(3).copied().unwrap_or(0) + v.get(4).copied().unwrap_or(0);
    (total.saturating_sub(idle), total)
}

/// Whole NVMe namespaces (`nvme0n1`), else whole SATA/virtio disks.
fn is_whole_disk(name: &str, nvme: bool) -> bool {
    if nvme {
        name.strip_prefix("nvme").is_some_and(|r| {
            let mut parts = r.splitn(2, 'n');
            let a = parts.next().unwrap_or("");
            let b = parts.next().unwrap_or("");
            !a.is_empty()
                && a.bytes().all(|c| c.is_ascii_digit())
                && !b.is_empty()
                && b.bytes().all(|c| c.is_ascii_digit())
        })
    } else {
        (name.starts_with("sd") || name.starts_with("vd"))
            && name[2..].bytes().all(|c| c.is_ascii_lowercase())
            && name.len() > 2
    }
}

fn disks(stats: &str) -> (u64, u64, Vec<String>) {
    let rows: Vec<Vec<&str>> = stats
        .lines()
        .map(|l| l.split_whitespace().collect())
        .filter(|f: &Vec<&str>| f.len() >= 10)
        .collect();
    let any_nvme = rows.iter().any(|f| is_whole_disk(f[2], true));
    let (mut r, mut w, mut names) = (0u64, 0u64, Vec::new());
    for f in rows.iter().filter(|f| is_whole_disk(f[2], any_nvme)) {
        r += f[5].parse::<u64>().unwrap_or(0) * SECTOR_BYTES;
        w += f[9].parse::<u64>().unwrap_or(0) * SECTOR_BYTES;
        names.push(f[2].to_string());
    }
    (r, w, names)
}

fn kv_kib(text: &str, key: &str) -> u64 {
    text.lines()
        .find_map(|l| {
            l.strip_prefix(key)
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        })
        .unwrap_or(0)
        * 1024
}

pub fn read_proc() -> ProcCounters {
    let read = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
    let (disk_read_bytes, disk_write_bytes, disk_names) = disks(&read("/proc/diskstats"));
    let meminfo = read("/proc/meminfo");
    let status = read("/proc/self/status");
    let io = read("/proc/self/io");
    ProcCounters {
        cpu: cpu_jiffies(&read("/proc/stat")),
        disk_read_bytes,
        disk_write_bytes,
        disk_names,
        proc_read_bytes: io
            .lines()
            .find_map(|l| {
                l.strip_prefix("read_bytes:")
                    .and_then(|v| v.trim().parse().ok())
            })
            .unwrap_or(0),
        mem_total: kv_kib(&meminfo, "MemTotal:"),
        mem_available: kv_kib(&meminfo, "MemAvailable:"),
        rss: kv_kib(&status, "VmRSS:"),
        rss_anon: kv_kib(&status, "RssAnon:"),
        rss_file: kv_kib(&status, "RssFile:"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_disks_only() {
        assert!(is_whole_disk("nvme0n1", true));
        assert!(!is_whole_disk("nvme0n1p2", true));
        assert!(is_whole_disk("sda", false));
        assert!(!is_whole_disk("sda1", false));
        let (r, w, n) = disks("259 0 nvme0n1 10 0 8 0 3 0 4 0\n259 1 nvme0n1p1 10 0 8 0 3 0 4 0\n");
        assert_eq!((r, w, n), (8 * 512, 4 * 512, vec!["nvme0n1".to_string()]));
    }

    #[test]
    fn cpu_busy_excludes_idle_and_iowait() {
        assert_eq!(cpu_jiffies("cpu  10 0 5 80 5 0 0 0 0 0\n"), (15, 100));
    }
}
