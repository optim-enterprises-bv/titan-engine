#!/usr/bin/env python3
"""Machine state snapshot for a bench run: uptime/load, free -m, top-5 RSS, GPU clocks/temp/power (NVML via
nvidia-smi), GPU compute apps, and the git revisions of the trees the binary is built from.

    hygiene.py OUT.json [BENCH_PIDS_FILE]

BENCH_PIDS_FILE (optional) lists PIDs the bench itself started; they count as titan processes."""
import json, os, subprocess, sys, time

E = os.path.expanduser("~/titan-engine")


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def git(path, ref="HEAD"):
    r = sh(f"git -C {path} rev-parse --short=12 {ref} 2>/dev/null")
    dirty = sh(f"git -C {path} status --porcelain --untracked-files=no 2>/dev/null | wc -l")
    br = sh(f"git -C {path} rev-parse --abbrev-ref HEAD 2>/dev/null")
    return {"rev": r or None, "branch": br or None, "dirty_files": int(dirty or 0) if dirty.isdigit() else None}


def main():
    out = sys.argv[1]
    own = set()
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        own = {int(x) for x in open(sys.argv[2]).read().split() if x.isdigit()}
    la = open("/proc/loadavg").read().split()
    up = float(open("/proc/uptime").read().split()[0])
    mem = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        mem[k] = int(v.split()[0]) // 1024
    procs = []
    for line in sh("ps -eo pid=,ppid=,rss=,comm=,args= --sort=-rss | head -12").splitlines():
        p = line.split(None, 4)
        if len(p) < 4:
            continue
        pid, ppid, rss, comm = int(p[0]), int(p[1]), int(p[2]) // 1024, p[3]
        args = p[4] if len(p) > 4 else ""
        titan = pid in own or ppid in own or comm in ("kbench", "nsys") or (comm == "mistralrs" and "18530" in args)
        procs.append({"pid": pid, "ppid": ppid, "rss_mib": rss, "comm": comm, "args": args[:160], "titan": titan})
    q = "clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,temperature.gpu,memory.used,pstate,utilization.gpu,clocks_throttle_reasons.active"
    g = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits")
    gpu = dict(zip(q.split(","), [x.strip() for x in g.split(",")])) if g and not g.startswith("ERR") else {"error": g}
    apps = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    snap = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "uptime_h": round(up / 3600, 2),
        "load": [float(x) for x in la[:3]],
        "mem_mib": {"total": mem.get("MemTotal"), "available": mem.get("MemAvailable"), "free": mem.get("MemFree"),
                    "swap_used": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)},
        "free_m": sh("free -m"),
        "top_rss": procs[:5],
        "gpu": gpu,
        "gpu_apps": apps,
        "titan_mistral": sh("systemctl --user is-active titan-mistral"),
        "revs": {
            "mistral.rs": dict(git(os.environ.get("BENCH_SRC", f"{E}/mr-094")), path=os.environ.get("BENCH_SRC", f"{E}/mr-094")),
            # candle is a subtree of the titan-engine repo (upstream a9667ca + fork): last commit touching it
            "candle": {"rev": sh(f"git -C {E} log -1 --format=%h -- candle"), "upstream": "a9667ca",
                       "dirty_files": int(sh(f"git -C {E} status --porcelain --untracked-files=no -- candle | wc -l") or 0)},
            "oxide-kernels": git(f"{E}/oxide-kernels"),
            "titan-engine": git(E),
            "llama.cpp": git(os.path.expanduser("~/ai/llama.cpp")),
        },
    }
    big = [p for p in procs if not p["titan"] and p["rss_mib"] > 4096]
    snap["non_titan_over_4g"] = [f'{p["comm"]}({p["pid"]}) {p["rss_mib"]} MiB' for p in big]
    json.dump(snap, open(out, "w"), indent=1)
    print(f"load {la[:3]}  avail {mem.get('MemAvailable')} MiB  gpu {gpu.get('clocks.sm')} MHz {gpu.get('temperature.gpu')} C "
          f"{gpu.get('power.draw')} W  mem.used {gpu.get('memory.used')} MiB  big non-titan: {snap['non_titan_over_4g'] or 'none'}")


if __name__ == "__main__":
    main()
