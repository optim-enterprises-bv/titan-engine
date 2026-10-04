#!/usr/bin/env python3
"""Release benchmark campaign core (driven by ../campaign.sh).

    camp.py plan                  items, status, time estimate, window split
    camp.py window                run pending items until the window budget is spent (caller: campaign.sh window)
    camp.py run ITEM [ITEM...]    run the named items (inside a window)
    camp.py status                done / pending / unstable per item
    camp.py redo ITEM [ITEM...]   mark items pending again (their JSON moves to results/discarded/)
    camp.py dry                   CPU-only rehearsal: fake engines (lib/fake_server.py), fake GPU, dryrun/ tree

Items (resumable one by one; results/<item>.json is written only when an item completes):
    tune.llama.<m>                llama.cpp sweep: --n-cpu-moe fit per -ub, -fa, -t, MTP draft length (probe cache
                                  in state/, so an interrupted sweep continues where it stopped)
    tune.ours.<m>                 ours: pick among the model's candidate configs (models with >1 candidate)
    perf.<m>.<engine>.<var>.short 3 reps: 8 prompts x N tokens decode, cold short TTFT; peak VRAM / RAM
    perf.<m>.<engine>.<var>.L<k>  3 reps: cold ~k-token prompt: TTFT, prompt tok/s, decode tok/s at that context
    qual.<m>.<engine>.<var>[.pN]  q100 score (lib/quality.py)
Consecutive items with the same server config share one server process."""
import copy, glob, hashlib, json, math, os, re, shlex, shutil, signal, statistics, subprocess, sys, threading, time

HOME = os.path.expanduser("~")
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "lib"))
DRY = os.environ.get("CAMPAIGN_DRY") == "1"
ROOT = os.path.join(R, "dryrun") if DRY else R
RES, STATE, LOGS = (os.path.join(ROOT, d) for d in ("results", "state", "logs"))
for d in (RES, STATE, LOGS):
    os.makedirs(d, exist_ok=True)
CFG = json.load(open(os.path.join(R, "campaign.json")))
PORT = CFG["port"] + (20 if DRY else 0)
X = lambda p: os.path.expanduser(p) if isinstance(p, str) else p
T0 = time.time()
DEADLINE = T0 + float(os.environ.get("CAMPAIGN_BUDGET_MIN", CFG["window_minutes"])) * 60
if DRY:  # fake engines are fast; no long load waits
    CFG.update(load_wait_s=5, max_redo=1)


def log(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(os.path.join(LOGS, "campaign.log"), "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def left():
    return DEADLINE - time.time()


def models(include_optional=True):
    return [m for m in CFG["models"] if include_optional or not m.get("optional")]


def M(mid):
    return next(m for m in CFG["models"] if m["id"] == mid)


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return default


def jsave(p, obj):
    tmp = p + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, p)


def spread(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    med = statistics.median(xs)
    return (max(xs) - min(xs)) / med if med else None


# ------------------------------------------------------------------ items
def lengths(m, optional=False):
    return m["lengths"] + (m.get("optional_lengths", []) if optional else [])


def ours_variants(m):
    return list(m["ours"]["variants"].keys())


def llama_variants(m):
    return m["llama"]["variants"]


ENGINES = [e for e in os.environ.get("CAMPAIGN_ENGINE", "ours,llama").split(",") if e]
ONLY_MODEL = os.environ.get("CAMPAIGN_MODEL", "")


def items(include_optional=True, filtered=True):
    """Ordered so that consecutive items share a server: per model, the tunes, then per engine and variant the
    perf items followed by that variant's quality items."""
    out = []
    for m in models(include_optional):
        mid, opt = m["id"], bool(m.get("optional"))
        out.append({"id": f"tune.llama.{mid}", "kind": "tune-llama", "model": mid, "optional": opt})
        if len(m["ours"].get("candidates", {})) > 1:
            out.append({"id": f"tune.ours.{mid}", "kind": "tune-ours", "model": mid, "optional": opt})
        for eng, vs in (("ours", ours_variants(m)), ("llama", llama_variants(m))):
            qv = vs if m.get("quality_variants") == "all" else vs[:1]
            for v in vs:
                base = f"perf.{mid}.{eng}.{v}"
                out.append({"id": base + ".short", "kind": "short", "model": mid, "engine": eng, "variant": v, "optional": opt})
                for k in lengths(m, optional=True):
                    out.append({"id": f"{base}.L{k}", "kind": "long", "length": k, "model": mid, "engine": eng, "variant": v,
                                "optional": opt or k in m.get("optional_lengths", [])})
                if v in qv:
                    parts = m.get("quality_split", 1)
                    for p in range(parts):
                        sfx = f".p{p + 1}" if parts > 1 else ""
                        out.append({"id": f"qual.{mid}.{eng}.{v}{sfx}", "kind": "quality", "model": mid, "engine": eng,
                                    "variant": v, "part": p, "parts": parts, "optional": opt})
    if filtered:
        out = [it for it in out if (it.get("engine") or it["kind"].split("-")[1]) in ENGINES
               and (not ONLY_MODEL or it["model"] == ONLY_MODEL)]
    return out


def done(item_id):
    return os.path.exists(os.path.join(RES, item_id + ".json"))


def deps_ok(it):
    if it["kind"] in ("tune-llama", "tune-ours"):
        return True
    m = M(it["model"])
    if it["engine"] == "llama":
        return done(f"tune.llama.{m['id']}")
    return len(m["ours"].get("candidates", {})) <= 1 or done(f"tune.ours.{m['id']}")


# ------------------------------------------------------------------ estimates
def rates(m, eng):
    """(load_s, prefill tok/s, decode tok/s): measured values when present, else the config's assumptions."""
    e = m["est"]
    load, pre, dec = e["load_s"][eng], e["prefill"][eng], e["decode"][eng]
    meas = jload(os.path.join(STATE, f"rates.{m['id']}.{eng}.json"), {})
    return meas.get("load_s", load), meas.get("prefill", pre), meas.get("decode", dec)


def tokens(k):
    return CFG["tokens"][k]


def est_item(it):
    m = M(it["model"])
    reps = CFG["reps"]
    if it["kind"] == "tune-llama" and m["llama"].get("seed"):
        L, pre, dec = rates(m, "llama")
        pn, pt = m.get("probe_short", [8, 64])
        ll = m["llama"]
        nth = len(set(ll.get("threads", []) + [ll["seed"].get("threads")]) - {None}) or 1
        fit = (L + tokens("4k") / pre + 30) * (2 + 2 * len(ll.get("mtp", [])))
        meas = (L + 2 * tokens("4k") / pre + pn * pt / dec + 40) * (nth + len(ll.get("mtp", [])))
        return fit + meas
    if it["kind"] == "tune-llama":
        L, pre, dec = rates(m, "llama")
        pn, pt = m.get("probe_short", [8, 64])
        probe_fit = L + tokens("4k") / pre + 20
        probe_meas = L + tokens("4k") / pre + pn * pt / dec + 30
        ll = m["llama"]
        n_fit = (3 * len(ll["ub"]) if ll.get("moe") else len(ll["ub"])) + len(ll["fa"]) - 1
        n_meas = len(ll["ub"]) + (1 if ll.get("moe") else 0) + len(ll["fa"]) - 1 + len(ll.get("threads", [])) + len(ll.get("mtp", []))
        return n_fit * probe_fit + n_meas * probe_meas
    if it["kind"] == "tune-ours":
        L, pre, dec = rates(m, "ours")
        pn, pt = m.get("probe_short", [8, 64])
        return len(m["ours"]["candidates"]) * (L + tokens("4k") / pre + pn * pt / dec + 60)
    eng = it["engine"]
    L, pre, dec = rates(m, eng)
    if it["kind"] == "short":
        n, t = m["short"]
        return 60 + L + 20 + 1.15 * reps * (n * (t / dec + 0.3) + n * 0.4)
    if it["kind"] == "long":
        tk = tokens(it["length"])
        # cold prefills + fixed-prompt decode reps (their prefill is cached where the engine has a prefix cache;
        # budgeted as uncached, gpt-oss on ours has none)
        # cold prefills (reps) + the fixed decode prompt: prefilled once, then prefix/slot-cached (llama.cpp slot
        # cache; ours hybrid prefix cache for qwen35/qwen3next; gpt-oss on ours has none: one more prefill per rep)
        nopc = eng == "ours" and m["family"] == "harmony"
        return 1.15 * ((reps + (reps if nopc else 0)) * tk / pre + reps * (m["long_decode"] / (0.8 * dec) + 3))
    if it["kind"] == "quality":
        per_q = m["est"].get("quality_tokens_per_q", 10)
        nq = 40 if quality_set(m) == "q40" else 100
        return nq / it["parts"] * (per_q / dec + 0.3) + 10
    return 60


# ------------------------------------------------------------------ server configs
def unit_conf():
    env, args = {}, []
    for line in open(X(CFG["ours"]["unit"])):
        line = line.strip().replace("%h", HOME)
        if line.startswith("Environment="):
            k, v = line[len("Environment="):].split("=", 1)
            env[k] = v
        elif line.startswith("ExecStart="):
            args = shlex.split(line[len("ExecStart="):])
    return env, args


def ours_binary():
    if DRY:
        return os.path.join(R, "lib", "fake_server.py")
    return os.path.realpath(unit_conf()[1][0])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ours_config(m, variant):
    """(env dict, argv list) for our engine."""
    gg = X(m["gguf"])
    o = m["ours"]
    if o["base"] == "service":
        env, a = unit_conf()
        argv = [ours_binary(), "--seed", "0"] + a[1:]
        for flag, val in (("-p", str(PORT)), ("-m", os.path.dirname(gg)), ("-f", os.path.basename(gg)),
                          ("--max-seq-len", str(m["ctx"]))):
            if flag in argv:
                argv[argv.index(flag) + 1] = val
            else:
                argv += [flag, val]
        cand = None
    else:
        env = {"LD_LIBRARY_PATH": X(CFG["ours"]["ld_library_path"])}
        argv = [ours_binary(), "--seed", "0", "serve", "--host", "127.0.0.1", "-p", str(PORT), "--paged-attn", "off",
                "--max-seq-len", str(m["ctx"]), "--format", "gguf", "-m", os.path.dirname(gg), "-f", os.path.basename(gg)]
        argv += o.get("extra_args", [])
        cands = o.get("candidates", {"default": {}})
        chosen = jload(os.path.join(RES, f"tune.ours.{m['id']}.json"), {}).get("chosen")
        cand = chosen if chosen in cands else next(iter(cands))
        env.update({k: X(v) for k, v in cands[cand].items()})
    env.update({k: X(v) for k, v in o["variants"].get(variant, {}).items()})
    return env, argv, cand


def llama_argv(m, t):
    """t: dict(ncmoe, ub, fa, threads, mtp, fit)."""
    argv = [X(CFG["llama"]["server"]), "-m", X(m["gguf"]), "-c", str(m["ctx"]), "--port", str(PORT)] + CFG["llama"]["common"]
    if t.get("fit"):
        argv += ["--fit", "on"]
    else:
        argv += ["--fit", "off", "-ngl", "99"]
        if m["llama"].get("moe"):
            argv += ["--n-cpu-moe", str(t["ncmoe"])]
    ub = t.get("ub", 512)
    argv += ["-ub", str(ub), "-b", str(max(ub, 2048)), "-fa", t.get("fa", "on")]
    if t.get("threads"):
        argv += ["-t", str(t["threads"])]
    if t.get("mtp"):
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(t["mtp"])]
    if DRY:
        argv = [os.path.join(R, "lib", "fake_server.py"), "--flavor", "llama", "--model", m["id"]] + argv[1:]
    return argv


def llama_config(m, variant):
    tune = jload(os.path.join(RES, f"tune.llama.{m['id']}.json"))
    if not tune:
        raise RuntimeError(f"tune.llama.{m['id']} has not run")
    t = dict(tune["chosen"])
    if variant == "nomtp" or (variant == "best" and not tune.get("mtp_helps", True)):
        t = dict(tune.get("chosen_nomtp") or t)
        t.pop("mtp", None)
    return {"LD_LIBRARY_PATH": X(CFG["llama"]["ld_library_path"])}, llama_argv(m, t), t


# ------------------------------------------------------------------ machine hygiene
def meminfo():
    mi = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        mi[k] = int(v.split()[0]) // 1024
    return {"total": mi["MemTotal"], "available": mi["MemAvailable"], "swap_used": mi["SwapTotal"] - mi["SwapFree"]}


def gpu_state():
    if DRY:
        return {"dry": True}
    q = "memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,pstate,clocks_throttle_reasons.active"
    s = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits")
    return dict(zip(q.split(","), [x.strip() for x in s.split(",")])) if s and not s.startswith("ERR") else {"error": s}


def snapshot():
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load": [float(x) for x in open("/proc/loadavg").read().split()[:3]],
            "mem_mib": meminfo(), "gpu": gpu_state(),
            "top_cpu": sh("ps -eo pid=,pcpu=,rss=,comm= --sort=-pcpu | head -5").splitlines()}


HEAVY_UNITS = ["restic-backup.service"]
HEAVY_SYS_UNITS = ["dnf-makecache.service", "plocate-updatedb.service", "raid-check.service", "fstrim.service",
                   "logrotate.service", "packagekit.service"]


def heavy_jobs():
    """Heavy jobs running now (user restic, system maintenance)."""
    busy = [u for u in HEAVY_UNITS if sh(f"systemctl --user is-active {u}") in ("active", "activating")]
    busy += [u for u in HEAVY_SYS_UNITS if sh(f"systemctl is-active {u}") in ("active", "activating")]
    return busy


def timers_due(within_s):
    """Timers (user + system) of heavy jobs firing within the next within_s seconds."""
    due = []
    for scope in ("--user", ""):
        out = sh(f"systemctl {scope} list-timers --all --no-pager --output=json", timeout=20)
        try:
            rows = json.loads(out)
        except Exception:  # noqa: BLE001
            continue
        for r in rows:
            nxt = r.get("next")
            unit = r.get("activates", "")
            if unit in HEAVY_UNITS + HEAVY_SYS_UNITS and nxt:
                dt = nxt / 1e6 - time.time()
                if 0 <= dt <= within_s:
                    due.append(f"{unit} in {dt / 60:.0f} min")
    return due


def wait_quiet():
    """Wait (bounded) until the 1-minute load is at most load_gate. Returns the load seen."""
    t_end = time.time() + min(CFG["load_wait_s"], max(0, left() - 300))
    while True:
        la = float(open("/proc/loadavg").read().split()[0])
        if la <= CFG["load_gate"] or time.time() >= t_end:
            return la
        time.sleep(5)


def quiet_with_server(mon):
    """With our server already up, the load average still carries the previous item's own CPU work, so the gate is
    the CPU used by everything else: sampled for 10 s, waiting (bounded) until it is at most the foreign-CPU gate."""
    t_end = time.time() + min(300, CFG["load_wait_s"], max(0, left() - 300))
    while True:
        mon.reset()
        time.sleep(10)
        f = mon.read()["foreign_cpu_cores"]
        if f <= CFG["foreign_cpu_gate_cores"] or time.time() >= t_end:
            return f


class Monitor(threading.Thread):
    """Samples GPU memory, the server cgroup's memory and CPU, the server's RSS, and the machine's total CPU, so a
    sub-item gets peak VRAM / RAM and the CPU time spent by everything that is not the server or this client."""

    def __init__(self, cg, pid):
        super().__init__(daemon=True)
        self.cg, self.pid, self.stop_ev = cg, pid, threading.Event()
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.vram = self.cgmem = self.rss = 0
            self.t0 = time.time()
            self.cpu0 = self._cpu()

    def _cpu(self):
        tot = sum(int(x) for x in open("/proc/stat").readline().split()[1:8]) / os.sysconf("SC_CLK_TCK")
        idle = [int(x) for x in open("/proc/stat").readline().split()[1:8]]
        busy = tot - (idle[3] + idle[4]) / os.sysconf("SC_CLK_TCK")
        srv = 0.0
        try:
            for line in open(f"/sys/fs/cgroup{self.cg}/cpu.stat"):
                if line.startswith("usage_usec"):
                    srv = int(line.split()[1]) / 1e6
        except OSError:
            pass
        me = sum(os.times()[:4])
        return busy, srv, me

    def run(self):
        smi = None
        if not DRY:
            smi = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "500"],
                                   stdout=subprocess.PIPE, text=True)
            threading.Thread(target=self._smi, args=(smi,), daemon=True).start()
        while not self.stop_ev.wait(0.5):
            cm = rs = 0
            try:
                cm = int(open(f"/sys/fs/cgroup{self.cg}/memory.current").read()) >> 20
            except OSError:
                pass
            try:
                for line in open(f"/proc/{self.pid}/status"):
                    if line.startswith("VmRSS"):
                        rs = int(line.split()[1]) >> 10
            except OSError:
                pass
            with self.lock:
                self.cgmem, self.rss = max(self.cgmem, cm), max(self.rss, rs)
        if smi:
            smi.terminate()

    def _smi(self, p):
        for line in p.stdout:
            try:
                v = int(line.strip())
            except ValueError:
                continue
            with self.lock:
                self.vram = max(self.vram, v)

    def read(self):
        with self.lock:
            busy, srv, me = self._cpu()
            dt = max(time.time() - self.t0, 1e-3)
            foreign = ((busy - self.cpu0[0]) - (srv - self.cpu0[1]) - (me - self.cpu0[2])) / dt
            return {"peak_vram_mib": self.vram or None, "peak_cgroup_mib": self.cgmem, "peak_rss_mib": self.rss,
                    "server_cpu_cores": round((srv - self.cpu0[1]) / dt, 2), "foreign_cpu_cores": round(max(foreign, 0), 2),
                    "seconds": round(dt, 1)}


# ------------------------------------------------------------------ server lifecycle
class Server:
    cur = None  # the running server (shared across consecutive items with the same key)

    def __init__(self, key, env, argv, gguf, health):
        self.key, self.env, self.argv, self.gguf, self.health = key, env, argv, gguf, health
        self.unit = f"camp-{hashlib.sha1(key.encode()).hexdigest()[:10]}-{os.getpid()}"
        self.log = os.path.join(LOGS, f"{time.strftime('%m%d-%H%M%S')}-{re.sub(r'[^A-Za-z0-9.]', '-', key)[:80]}.server.log")
        self.up_s = None

    @classmethod
    def get(cls, key, env, argv, gguf, health):
        if cls.cur and cls.cur.key == key:
            return cls.cur, False
        cls.stop_current()
        s = cls(key, env, argv, gguf, health)
        s.start()
        cls.cur = s
        return s, True

    @classmethod
    def stop_current(cls):
        if cls.cur:
            cls.cur.stop()
            cls.cur = None

    def start(self):
        if not DRY:
            for _ in range(30):  # the card must be empty
                used = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split("\n")[0]
                if used.isdigit() and int(used) < 700:
                    break
                time.sleep(2)
            else:
                raise RuntimeError(f"GPU memory in use before start: {used} MiB "
                                   f"({sh('nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader')})")
        if CFG["cold_page_cache"] and not DRY and self.gguf and os.path.exists(self.gguf):
            fd = os.open(self.gguf, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        self.snap_before = snapshot()
        self.load_before = wait_quiet()
        cmd = ["systemd-run", "--user", f"--unit={self.unit}", "--collect", "-q", "-p", f"MemoryMax={CFG['memory_max']}",
               "-p", "MemorySwapMax=0", "-p", f"RuntimeMaxSec={int(max(left(), 60)) + 60}", "-p", "TimeoutStopSec=20",
               "-p", f"StandardOutput=truncate:{self.log}", "-p", f"StandardError=truncate:{self.log}"]
        cmd += [f"--setenv={k}={v}" for k, v in self.env.items()]
        if DRY:
            cmd += ["--setenv=PYTHONUNBUFFERED=1", sys.executable]
        cmd += self.argv
        log(f"start {self.key}: {' '.join(shlex.quote(a) for a in self.argv)}")
        t = time.time()
        subprocess.run(cmd, check=True, timeout=60)
        ok = False
        while time.time() - t < min(900, max(left(), 30)):
            if sh(f"curl -sf -m 2 -o /dev/null -w '%{{http_code}}' localhost:{PORT}{self.health}") == "200":
                ok = True
                break
            if sh(f"systemctl --user is-active {self.unit}") not in ("active", "activating"):
                break
            time.sleep(1)
        self.up_s = time.time() - t
        if not ok:
            tail = sh(f"tail -c 1500 {shlex.quote(self.log)}")
            self.stop()
            raise RuntimeError(f"server did not come up ({self.up_s:.0f}s): {tail[-600:]}")
        self.cg = sh(f"systemctl --user show {self.unit} -p ControlGroup --value")
        self.pid = int(sh(f"systemctl --user show {self.unit} -p MainPID --value") or 0)
        self.mon = Monitor(self.cg, self.pid)
        self.mon.start()
        log(f"up {self.key} after {self.up_s:.0f}s")

    def cgroup_peak(self):
        try:
            return int(open(f"/sys/fs/cgroup{self.cg}/memory.peak").read()) >> 20
        except (OSError, AttributeError, ValueError):
            return None

    def oom_in_log(self):
        s = sh(f"grep -a -c -i -E 'out of memory|OUT_OF_MEMORY|failed to allocate|cudaMalloc failed|panicked' {shlex.quote(self.log)}")
        return s.isdigit() and int(s) > 0

    def stop(self):
        if getattr(self, "mon", None):
            self.mon.stop_ev.set()
        sh(f"systemctl --user stop {self.unit}", timeout=60)  # TERM, then KILL after TimeoutStopSec
        if not DRY:
            for _ in range(30):
                used = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split("\n")[0]
                if used.isdigit() and int(used) < 700:
                    break
                time.sleep(1)
        log(f"stopped {self.key}")


def health_of(eng):
    return "/health" if eng == "llama" or DRY else "/v1/models"


def server_for(m, eng, variant):
    if eng == "ours":
        env, argv, cand = ours_config(m, variant)
        if DRY:
            argv = [argv[0], "--flavor", "ours", "--model", m["id"]] + argv[1:]
        desc = {"candidate": cand, "variant": variant}
    else:
        env, argv, t = llama_config(m, variant)
        desc = {"tuned": t, "variant": variant}
    key = json.dumps([m["id"], eng, env, argv], sort_keys=True)
    s, fresh = Server.get(key, env, argv, X(m["gguf"]), health_of(eng))
    return s, fresh, env, argv, desc


# ------------------------------------------------------------------ measurement items
import client as C  # noqa: E402
import quality as Q  # noqa: E402


def adaptive(fn, key):
    """CFG reps (2) runs of fn(); one more when their spread of `key` exceeds the gate. Returns the runs."""
    runs = [fn() for _ in range(CFG["reps"])]
    sp = spread([r[key] for r in runs])
    if sp is None or sp > CFG["spread_gate"]:
        runs.append(fn())
    return runs


def measure_short(m, s):
    n, t = m["short"]
    reps = adaptive(lambda: C.short_decode(PORT, n, t), "decode_tok_s")
    tt = [C.ttft_short(PORT, n) for _ in range(CFG["reps"])]
    return {"reps": reps, "ttft_reps": tt,
            "metrics": {"decode_tok_s": [r["decode_tok_s"] for r in reps], "ttft_short_ms": [x["ttft_ms"] for x in tt]},
            "gated": ["decode_tok_s"],
            "identical_across_reps": len({r["texts_sha"] for r in reps}) == 1}


def measure_long(m, s, k):
    """Cold prefill: fresh-nonce prompts, max_tokens 1 (TTFT, prompt tok/s). Decode at context: the same fixed prompt
    (fixed nonce) with long_decode tokens, so every rep decodes the identical greedy reply (identical expert routing
    and draft acceptance); its prefill may be prefix-cached, which does not enter the decode rate (measured from the
    first token). 2 reps each, a third when the pair's spread exceeds the gate."""
    cold = adaptive(lambda: C.long_prompt(PORT, k, 1), "prompt_tok_s")
    last = cold[-1]["nonce"]  # the engine's prefix/slot cache holds the last prompt: no extra long prefill
    dec = adaptive(lambda: C.long_prompt(PORT, k, m["long_decode"], nonce=last), "decode_tok_s")
    return {"reps": cold, "decode_reps": dec,
            "metrics": {"prompt_tok_s": [r["prompt_tok_s"] for r in cold], "ttft_s": [r["ttft_s"] for r in cold],
                        "decode_tok_s": [r["decode_tok_s"] for r in dec],
                        "server_prompt_tok_s": [r["server_prompt_tok_s"] for r in cold]},
            "prompt_tokens": [r["prompt_tokens"] for r in cold],
            "gated": ["prompt_tok_s", "decode_tok_s"], "method": "v3 adaptive reps: cold-nonce prefill + fixed-prompt decode"}


def summarize(metrics):
    """median / min / max / spread over all runs; gate_spread = the closest pair's spread when a third run was taken
    (the gate asks that two runs agree within 3%; the median of three is robust to the one that does not)."""
    out = {}
    for k, xs in metrics.items():
        ys = [x for x in xs if x is not None]
        g = spread(ys)
        if len(ys) >= 3:
            g = min(spread([a, b]) for i, a in enumerate(ys) for b in ys[i + 1:])
        out[k] = {"median": statistics.median(ys) if ys else None, "min": min(ys) if ys else None,
                  "max": max(ys) if ys else None, "spread": spread(ys), "gate_spread": g, "n": len(ys)}
    return out


def run_measure(it):
    m = M(it["model"])
    eng = it["engine"]
    attempts = []
    for attempt in range(CFG["max_redo"] + 1):
        s, fresh, env, argv, desc = server_for(m, eng, it["variant"])
        if fresh:  # after the page-cache drop: fault the weights in before anything is timed
            C.warm(PORT)
            C.long_prompt(PORT, "4k", 8)
        before = snapshot()
        if not fresh:
            before["foreign_cpu_pre"] = quiet_with_server(s.mon)
        s.mon.reset()
        t = time.time()
        if it["kind"] == "short":
            r = measure_short(m, s)
        else:
            r = measure_long(m, s, it["length"])
        mon = s.mon.read()
        after = snapshot()
        summ = summarize(r["metrics"])
        reasons = []
        for g in r["gated"]:
            sp = summ[g]["gate_spread"]
            if sp is None or sp > CFG["spread_gate"]:
                reasons.append(f"spread {g} {'n/a' if sp is None else f'{100 * sp:.1f}%'} > {100 * CFG['spread_gate']:.0f}%")
        if fresh and s.load_before > CFG["load_gate"]:
            reasons.append(f"load before {s.load_before:.1f} > {CFG['load_gate']}")
        if not fresh and before["foreign_cpu_pre"] > CFG["foreign_cpu_gate_cores"]:
            reasons.append(f"foreign CPU before {before['foreign_cpu_pre']:.1f} cores > {CFG['foreign_cpu_gate_cores']}")
        if mon["foreign_cpu_cores"] > CFG["foreign_cpu_gate_cores"]:
            reasons.append(f"foreign CPU {mon['foreign_cpu_cores']:.1f} cores > {CFG['foreign_cpu_gate_cores']}")
        if heavy_jobs():
            reasons.append(f"heavy job running: {heavy_jobs()}")
        rec = {"attempt": attempt + 1, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t)),
               "duration_s": round(time.time() - t, 1), "server_fresh": fresh, "server_up_s": s.up_s,
               "summary": summ, "raw": r, "monitor": mon, "cgroup_memory_peak_mib": s.cgroup_peak(),
               "hygiene": {"before": before, "after": after}, "contaminated": reasons}
        attempts.append(rec)
        log(f"{it['id']} attempt {attempt + 1}: " + ", ".join(
            f"{k} {v['median']:.1f} ({100 * (v['spread'] or 0):.1f}%)" for k, v in summ.items() if v["median"] is not None)
            + (f"  REDO: {'; '.join(reasons)}" if reasons else ""))
        if not reasons:
            break
        if left() < est_item(it) + 60:
            break
        time.sleep(20)
    final = attempts[-1]
    return {"item": it["id"], "model": m["id"], "engine": eng, "variant": it["variant"], "kind": it["kind"],
            "length": it.get("length"), "status": "ok" if not final["contaminated"] else "unstable",
            "server": {"env": env, "argv": argv, "config": desc, "log": s.log}, "result": final,
            "discarded_attempts": attempts[:-1]}


def quality_set(m):
    """q40 (fixed seeded subset) for slow models, q100 otherwise; decided once per model so both engines match:
    campaign.json quality_set, else q40 when llama.cpp's tuned decode is under 10 tok/s."""
    if m.get("quality_set"):
        return m["quality_set"]
    t = jload(os.path.join(RES, f"tune.llama.{m['id']}.json"), {})
    d = next((r.get("decode_tok_s") for r in t.get("measured", []) if r.get("t") == t.get("chosen_nomtp")), None)
    return "q40" if d is not None and d < 10 else "q100"


def run_quality(it):
    m = M(it["model"])
    s, fresh, env, argv, desc = server_for(m, it["engine"], it["variant"])
    qset = quality_set(m)
    t = time.time()
    if qset == "q40":
        q = Q.run(PORT, m["family"], subset=Q.q40())
    else:
        n = 100
        q = Q.run(PORT, m["family"], f"{it['part'] * n // it['parts']}:{(it['part'] + 1) * n // it['parts']}")
    log(f"{it['id']}: {q['set']} {q['score']}/{q['n']} in {time.time() - t:.0f}s")
    return {"item": it["id"], "model": m["id"], "engine": it["engine"], "variant": it["variant"], "kind": "quality",
            "status": "ok", "server": {"env": env, "argv": argv, "config": desc}, "result": q}


# ------------------------------------------------------------------ tuning
def probe_cache(mid, eng):
    p = os.path.join(STATE, f"probes.{eng}.{mid}.json")
    return p, jload(p, {})


def short_probe(port, m):
    n, t = m.get("probe_short", [8, 64])
    return C.short_decode(port, n, t)["decode_tok_s"]


def probe_clean(s):
    """Hygiene for a probe measurement: (ok, reason)."""
    f = s.mon.read()["foreign_cpu_cores"]
    busy = heavy_jobs()
    if busy:
        return False, f"heavy job {busy}"
    if f > CFG["foreign_cpu_gate_cores"]:
        return False, f"foreign CPU {f:.1f} cores"
    if s.load_before > CFG["load_gate"]:
        return False, f"load before {s.load_before:.1f}"
    return True, ""


def measure_probe(m, s):
    """4k prompt twice (the first after a page-cache drop pays the page faults; the second is kept) + probe decode."""
    C.warm(PORT)
    C.long_prompt(PORT, "4k", 4)
    s.mon.reset()
    r = C.long_prompt(PORT, "4k", 4)
    return {"prompt_tok_s": r["prompt_tok_s"], "decode_tok_s": short_probe(PORT, m)}


def llama_probe(m, t, what):
    """what: 'fit' (starts, and runs a 4k prompt: llama.cpp allocates the whole KV cache and the compute buffers for
    -c/-ub at load, so an out-of-memory config fails here; the long perf items confirm it) or 'meas' (4k prompt rate,
    second of two, + the model's probe decode, 8x64 by default). Cached per config. A 'meas' probe disturbed by other
    CPU load or a heavy job is repeated (up to twice) and never cached as clean."""
    p, cache = probe_cache(m["id"], "llama")
    key = what + ":" + json.dumps(t, sort_keys=True)
    if key in cache and not cache[key].get("contaminated"):
        return cache[key]
    env = {"LD_LIBRARY_PATH": X(CFG["llama"]["ld_library_path"])}
    argv = llama_argv(m, t)
    for attempt in range(3):
        res = {"t": t, "what": what, "argv": argv, "attempt": attempt + 1}
        try:
            s, _ = Server.get("probe:" + key, env, argv, X(m["gguf"]), "/health")
            res["up_s"] = s.up_s
            if what == "fit":
                r = C.long_prompt(PORT, "4k", 4)
                res.update(ok=r["prompt_tok_s"] is not None and not s.oom_in_log(), prompt_tok_s=r["prompt_tok_s"],
                           prompt_tokens=r["prompt_tokens"])
            else:
                res.update(ok=True, **measure_probe(m, s))
                clean, why = probe_clean(s)
                if not clean:
                    res["contaminated"] = why
            res["peak_vram_mib"] = s.mon.read()["peak_vram_mib"]
        except Exception as e:  # noqa: BLE001  (OOM at load or during the prompt: the config does not fit)
            res.update(ok=False, error=str(e)[-400:])
        finally:
            Server.stop_current()
        if res.get("ok") is False and "error" in res and left() < 120:
            raise TimeoutError("window budget spent during a probe; not caching it")
        if not res.get("contaminated"):
            break
        log(f"probe {m['id']} {what} {t}: contaminated ({res['contaminated']}), repeating")
        if heavy_jobs():
            raise TimeoutError(f"heavy job running {heavy_jobs()}: probe deferred")
        time.sleep(30)
    cache[key] = res
    jsave(p, cache)
    log(f"probe {m['id']} {what} {t}: " + ("ok" if res["ok"] else "FAIL") +
        "".join(f" {k}={res[k]:.1f}" for k in ("prompt_tok_s", "decode_tok_s") if isinstance(res.get(k), float))
        + (f" CONTAMINATED {res['contaminated']}" if res.get("contaminated") else ""))
    return res


def fit_from(m, it, c, fit_cost, span=6):
    """Smallest --n-cpu-moe >= c's that runs config c (a variant that needs more VRAM: -fa off, an MTP draft)."""
    if not m["llama"].get("moe") or c.get("fit"):
        need(it["id"], fit_cost)
        return c if llama_probe(m, c, "fit")["ok"] else None
    for n in range(c["ncmoe"], c["ncmoe"] + span + 1):
        cc = dict(c, ncmoe=n)
        need(it["id"], fit_cost)
        if llama_probe(m, cc, "fit")["ok"]:
            return cc
    return None


def score(r):
    return math.sqrt(r["prompt_tok_s"] * r["decode_tok_s"]) if r.get("ok") and r.get("prompt_tok_s") and r.get("decode_tok_s") else 0


def need(it_id, secs):
    if left() < secs:
        raise TimeoutError(f"{it_id}: {left():.0f}s left, next step needs ~{secs:.0f}s; resumes next window")


def run_tune_llama(it):
    """Pre-declared rule: for each -ub the smallest --n-cpu-moe that runs the model's longest prompt (most expert
    layers on the GPU); llama.cpp's own --fit as one more candidate; pick by sqrt(prompt tok/s x decode tok/s);
    then -fa, then -t by the same score; then the MTP draft length by decode tok/s. Measurements: a 4k prompt and
    the model's probe decode (8 prompts x 64 tokens unless the config says less for slow models)."""
    m = M(it["model"])
    ll = m["llama"]
    L, pre, dec = rates(m, "llama")
    pn, pt = m.get("probe_short", [8, 64])
    fit_cost = L + tokens("4k") / pre + 30
    meas_cost = L + tokens("4k") / pre + pn * pt / dec + 40
    table, cands = [], []
    fa0 = ll["fa"][0]
    for ub in ll["ub"]:
        if ll.get("moe"):
            n = ll["ncmoe_hint"]
            nmax = 99
            need(it["id"], fit_cost)
            r = llama_probe(m, {"ncmoe": n, "ub": ub, "fa": fa0}, "fit")
            if r["ok"]:
                best = n
                while n > 0:
                    need(it["id"], fit_cost)
                    if not llama_probe(m, {"ncmoe": n - 1, "ub": ub, "fa": fa0}, "fit")["ok"]:
                        break
                    n -= 1
                    best = n
            else:
                best = None
                while n < nmax:
                    n += 1
                    need(it["id"], fit_cost)
                    if llama_probe(m, {"ncmoe": n, "ub": ub, "fa": fa0}, "fit")["ok"]:
                        best = n
                        break
            if best is not None:
                cands.append({"ncmoe": best, "ub": ub, "fa": fa0})
        else:
            need(it["id"], fit_cost)
            if llama_probe(m, {"ub": ub, "fa": fa0}, "fit")["ok"]:
                cands.append({"ub": ub, "fa": fa0})
    if ll.get("moe"):
        need(it["id"], fit_cost)
        if llama_probe(m, {"fit": True, "fa": fa0}, "fit")["ok"]:
            cands.append({"fit": True, "fa": fa0})
    if not cands:
        raise RuntimeError("no llama.cpp configuration fits")
    for c in cands:
        need(it["id"], meas_cost)
        table.append(llama_probe(m, c, "meas"))
    chosen = max(table, key=score)["t"]
    for fa in ll["fa"][1:]:
        c = fit_from(m, it, dict(chosen, fa=fa), fit_cost)
        need(it["id"], meas_cost)
        if c:
            r = llama_probe(m, c, "meas")
            table.append(r)
            if score(r) > score(llama_probe(m, chosen, "meas")):
                chosen = c
    base = llama_probe(m, chosen, "meas")
    for th in ll.get("threads", []):
        c = dict(chosen, threads=th)
        need(it["id"], meas_cost)
        r = llama_probe(m, c, "meas")
        table.append(r)
        if score(r) > score(base):
            chosen, base = c, r
    mtp_rows, mtp_best = [], None
    for n in ll.get("mtp", []):
        c = fit_from(m, it, dict(chosen, mtp=n), fit_cost)  # the draft needs its own compute buffer
        if not c:
            continue
        need(it["id"], meas_cost)
        r = llama_probe(m, c, "meas")
        mtp_rows.append(r)
        if r["ok"] and (mtp_best is None or r["decode_tok_s"] > mtp_best["decode_tok_s"]):
            mtp_best = r
    mtp_helps = bool(mtp_best and mtp_best["decode_tok_s"] > base["decode_tok_s"])
    base_nomtp = dict(chosen)
    if mtp_best:
        chosen = dict(mtp_best["t"])
    _, cache = probe_cache(m["id"], "llama")
    jsave(os.path.join(STATE, f"rates.{m['id']}.llama.json"),
          {"load_s": base.get("up_s") or L, "prefill": base["prompt_tok_s"], "decode": base["decode_tok_s"]})
    dirty = [r["t"] for r in table + mtp_rows if r.get("contaminated")]
    return {"item": it["id"], "model": m["id"], "kind": "tune-llama", "status": "ok" if not dirty else "unstable",
            "contaminated_probes": dirty, "chosen": chosen, "chosen_nomtp": base_nomtp,
            "mtp_helps": mtp_helps, "rule": run_tune_llama.__doc__.strip(),
            "argv_chosen": llama_argv(m, chosen), "fit_probes": [v for k, v in cache.items() if k.startswith("fit:")],
            "measured": table + mtp_rows}


def run_tune_llama_seeded(it):
    """Seeded (pre-declared rule, 2026-09-29 restructure): start from the known-good settings in campaign.json
    (llama.seed); --n-cpu-moe = the smallest value in seed-2..seed+2 that runs (searching upward past +2 only if none
    of them fits at this context); then -t over llama.threads (6/8/12 plus the seed's), 1 measurement each, picked by
    sqrt(prompt tok/s at 4k x decode tok/s); then, on MTP GGUFs, the seed's draft length with --n-cpu-moe re-fitted
    for the draft's extra VRAM, kept for the headline if its decode is faster."""
    m = M(it["model"])
    ll = m["llama"]
    seed = dict(ll["seed"])
    L, pre, dec = rates(m, "llama")
    pn, pt = m.get("probe_short", [8, 64])
    fit_cost = L + tokens("4k") / pre + 30
    meas_cost = L + 2 * tokens("4k") / pre + pn * pt / dec + 40
    base = {k: seed[k] for k in ("ub", "fa") if k in seed}
    if ll.get("moe"):
        n0 = seed["ncmoe"]
        best = None
        for n in list(range(max(0, n0 - 2), n0 + 3)) + list(range(n0 + 3, n0 + 11)):
            need(it["id"], fit_cost)
            if llama_probe(m, dict(base, ncmoe=n), "fit")["ok"]:
                best = n
                break
        if best is None:
            raise RuntimeError(f"no --n-cpu-moe in {max(0, n0 - 2)}..{n0 + 10} fits")
        base["ncmoe"] = best
    else:
        need(it["id"], fit_cost)
        if not llama_probe(m, base, "fit")["ok"]:
            raise RuntimeError("the dense model does not fit with the seed settings")
    table = []
    ths = sorted(set(ll.get("threads", []) + ([seed["threads"]] if seed.get("threads") else [])))
    chosen, brow = None, None
    for th in ths or [None]:
        c = dict(base, threads=th) if th else dict(base)
        need(it["id"], meas_cost)
        r = llama_probe(m, c, "meas")
        table.append(r)
        if brow is None or score(r) > score(brow):
            chosen, brow = c, r
    if not score(brow):
        raise RuntimeError("no measured llama.cpp configuration ran")
    base_nomtp = dict(chosen)
    mtp_rows, mtp_best = [], None
    for n in ll.get("mtp", []):
        c = fit_from(m, it, dict(chosen, mtp=n), fit_cost)
        if not c:
            continue
        need(it["id"], meas_cost)
        r = llama_probe(m, c, "meas")
        mtp_rows.append(r)
        if r["ok"] and (mtp_best is None or r["decode_tok_s"] > mtp_best["decode_tok_s"]):
            mtp_best = r
    mtp_helps = bool(mtp_best and mtp_best["decode_tok_s"] > brow["decode_tok_s"])
    if mtp_best:
        chosen = dict(mtp_best["t"])
    _, cache = probe_cache(m["id"], "llama")
    jsave(os.path.join(STATE, f"rates.{m['id']}.llama.json"),
          {"load_s": brow.get("up_s") or L, "prefill": brow["prompt_tok_s"], "decode": brow["decode_tok_s"]})
    dirty = [r["t"] for r in table + mtp_rows if r.get("contaminated")]
    return {"item": it["id"], "model": m["id"], "kind": "tune-llama", "status": "ok" if not dirty else "unstable",
            "contaminated_probes": dirty, "seed": seed, "chosen": chosen, "chosen_nomtp": base_nomtp,
            "mtp_helps": mtp_helps, "rule": run_tune_llama_seeded.__doc__.strip(), "argv_chosen": llama_argv(m, chosen),
            "fit_probes": [v for k, v in cache.items() if k.startswith("fit:")], "measured": table + mtp_rows}


def run_tune_ours(it):
    """Pre-declared rule: each candidate config runs a 4k prompt and 8x64 decode; pick by
    sqrt(prompt tok/s x decode tok/s), the same score as the llama.cpp sweep."""
    m = M(it["model"])
    p, cache = probe_cache(m["id"], "ours")
    L, pre, dec = rates(m, "ours")
    rows = []
    for name, env_c in m["ours"]["candidates"].items():
        if name not in cache or cache[name].get("contaminated"):
            pn, pt = m.get("probe_short", [8, 64])
            need(it["id"], L + tokens("4k") / pre + pn * pt / dec + 60)
            env = {"LD_LIBRARY_PATH": X(CFG["ours"]["ld_library_path"])}
            env.update({k: X(v) for k, v in env_c.items()})
            _, argv, _ = ours_config(m, "best")
            if DRY:
                argv = [argv[0], "--flavor", "ours", "--model", m["id"]] + argv[1:]
            res = {"candidate": name, "env": env}
            try:
                s, _ = Server.get("tune-ours:" + name, env, argv, X(m["gguf"]), health_of("ours"))
                res.update(ok=True, up_s=s.up_s, **measure_probe(m, s))
                clean, why = probe_clean(s)
                if not clean:
                    res["contaminated"] = why
                res["peak_vram_mib"] = s.mon.read()["peak_vram_mib"]
            except Exception as e:  # noqa: BLE001
                res.update(ok=False, error=str(e)[-400:])
            finally:
                Server.stop_current()
            cache[name] = res
            jsave(p, cache)
            log(f"ours candidate {m['id']} {name}: " + ("ok" if res["ok"] else "FAIL " + res.get("error", "")[:200]))
        rows.append(cache[name])
    best = max(rows, key=score)
    if not score(best):
        raise RuntimeError("no candidate config of ours ran")
    jsave(os.path.join(STATE, f"rates.{m['id']}.ours.json"),
          {"load_s": best.get("up_s"), "prefill": best["prompt_tok_s"], "decode": best["decode_tok_s"]})
    return {"item": it["id"], "model": m["id"], "kind": "tune-ours", "status": "ok", "chosen": best["candidate"],
            "rule": run_tune_ours.__doc__.strip(), "measured": rows}


# ------------------------------------------------------------------ provenance
def provenance():
    """Pin per engine: state/pinned.json (ours binary sha256) is created/checked only when ours items run;
    state/pinned-llama.json (llama.cpp rev) when llama items run. A new binary for ours: rm state/pinned.json and redo
    the ours items (campaign.sh redo ...)."""
    ob = ours_binary()
    cur = {"ours_binary": ob, "ours_sha256": sha256(ob) if os.path.exists(ob) else None,
           "llama_server": X(CFG["llama"]["server"]),
           "llama_rev": sh(f"git -C {HOME}/ai/llama.cpp rev-parse --short=12 HEAD"),
           "mistral_rs_rev": sh(f"git -C {X(CFG['ours']['src'])} rev-parse --short=12 HEAD"),
           "oxide_kernels_rev": sh(f"git -C {HOME}/titan-engine/oxide-kernels rev-parse --short=12 HEAD"),
           "titan_engine_rev": sh(f"git -C {HOME}/titan-engine rev-parse --short=12 HEAD"),
           "unit": open(X(CFG["ours"]["unit"])).read(),
           "cpu": sh("lscpu | grep 'Model name'").split(":")[-1].strip(),
           "gpu": "dry-run" if DRY else sh("nvidia-smi --query-gpu=name,memory.total,power.limit,driver_version --format=csv,noheader"),
           "kernel": os.uname().release, "mem_total_mib": meminfo()["total"],
           "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if "llama" in ENGINES:
        if not DRY and not cur["llama_rev"].startswith(CFG["llama"]["expect_rev"]):
            raise SystemExit(f"llama.cpp is at {cur['llama_rev']}, the campaign expects {CFG['llama']['expect_rev']}")
        pl = os.path.join(STATE, "pinned-llama.json")
        if not os.path.exists(pl):
            jsave(pl, cur)
    if "ours" in ENGINES:
        p = os.path.join(STATE, "pinned.json")
        pin = jload(p)
        if pin is None:
            jsave(p, cur)
            return cur
        if pin["ours_sha256"] != cur["ours_sha256"] and os.environ.get("CAMPAIGN_REPIN") != "1":
            raise SystemExit(f"our binary changed since the ours phase started ({pin['ours_sha256'][:12]} -> "
                             f"{(cur['ours_sha256'] or '?')[:12]}). rm state/pinned.json and redo every ours item, "
                             f"or set CAMPAIGN_REPIN=1.")
        return pin
    return cur


# ------------------------------------------------------------------ driver
RUNNERS = {"tune-llama": lambda it: (run_tune_llama_seeded if M(it["model"])["llama"].get("seed") else run_tune_llama)(it), "tune-ours": run_tune_ours, "short": run_measure, "long": run_measure,
           "quality": run_quality}


def run_item(it, prov):
    t = time.time()
    try:
        out = RUNNERS[it["kind"]](it)
    except TimeoutError as e:
        log(f"{it['id']}: {e}")
        return "budget"
    except Exception as e:  # noqa: BLE001
        Server.stop_current()
        if left() < 120:  # the window deadline killed it (server RuntimeMaxSec): not the item's fault
            log(f"{it['id']}: interrupted by the window deadline ({type(e).__name__}); resumes next window")
            return "budget"
        log(f"{it['id']}: FAILED {type(e).__name__}: {str(e)[:500]}")
        fails = jload(os.path.join(STATE, "failures.json"), {})
        fails.setdefault(it["id"], []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "error": str(e)[-1500:]})
        jsave(os.path.join(STATE, "failures.json"), fails)
        return "failed"
    out.update(duration_s=round(time.time() - t, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               provenance={k: prov.get(k) for k in ("ours_sha256", "llama_rev", "mistral_rs_rev", "oxide_kernels_rev")},
               synthetic=DRY)
    jsave(os.path.join(RES, it["id"] + ".json"), out)
    if it["kind"] == "short" and out["status"] == "ok" and it["variant"] in (ours_variants(M(it["model"]))[:1] + ["best", "nomtp"]):
        r = out["result"]  # measured rates feed `plan` (the long items update the prefill rate)
        rp = os.path.join(STATE, f"rates.{it['model']}.{it['engine']}.json")
        jsave(rp, dict(jload(rp, {}), decode=r["summary"]["decode_tok_s"]["median"], load_s=r["server_up_s"] or 30))
    if it["kind"] == "long" and out["status"] == "ok":
        rp = os.path.join(STATE, f"rates.{it['model']}.{it['engine']}.json")
        jsave(rp, dict(jload(rp, {}), prefill=out["result"]["summary"]["prompt_tok_s"]["median"]))
    return "done"


def pending(include_optional):
    fails = jload(os.path.join(STATE, "failures.json"), {})
    return [it for it in items(include_optional) if not done(it["id"]) and len(fails.get(it["id"], [])) < 2]


def cmd_window(args):
    include_opt = os.environ.get("CAMPAIGN_OPTIONAL") == "1"
    prov = provenance()
    log(f"== window start; engines {ENGINES}{' model ' + ONLY_MODEL if ONLY_MODEL else ''}; budget {left() / 60:.0f} min; "
        f"ours {prov['ours_sha256'][:12] if 'ours' in ENGINES and prov['ours_sha256'] else '-'}; llama {prov['llama_rev']}")
    only = set(args)
    ran = 0
    try:
        for it in items(include_opt or bool(only)):
            if only and it["id"] not in only:
                continue
            if done(it["id"]) or not deps_ok(it):
                continue
            fails = jload(os.path.join(STATE, "failures.json"), {})
            if len(fails.get(it["id"], [])) >= 2 and not only:
                continue
            e = est_item(it)
            resumable = it["kind"].startswith("tune")
            if e > left() - 60 and not resumable:
                log(f"{it['id']}: needs ~{e / 60:.1f} min, {left() / 60:.1f} left: stop here")
                break
            if left() < 180:
                break
            busy = heavy_jobs()
            if busy:
                log(f"heavy job running ({busy}): ending the window")
                break
            log(f"-> {it['id']} (est {e / 60:.1f} min, {left() / 60:.1f} left)")
            st = run_item(it, prov)
            ran += 1
            if st == "budget":
                break
    finally:
        Server.stop_current()
    log(f"== window end; {ran} items run; {len(pending(include_opt))} pending")
    import report
    report.main(ROOT)


def cmd_plan(_):
    include_opt = True
    its = items(include_opt)
    rows, win, wmin, cap = [], 1, 0.0, CFG["window_minutes"]
    total = {"req": 0.0, "opt": 0.0}
    for it in its:
        if done(it["id"]):
            continue
        e = est_item(it) / 60
        total["opt" if it["optional"] else "req"] += e
        if it["optional"]:
            rows.append((it["id"], e, "optional"))
            continue
        if wmin + e > cap and wmin > 0:
            win += 1
            wmin = 0.0
        # a resumable tune longer than a window spills into the next one
        while e > cap - wmin and it["kind"].startswith("tune"):
            e_part = cap - wmin
            rows.append((it["id"], e_part, f"W{win} (part)"))
            e -= e_part
            win += 1
            wmin = 0.0
        wmin += e
        rows.append((it["id"], e, f"W{win}"))
    print(f"{'item':48s} {'est min':>8s}  window")
    for r in rows:
        print(f"{r[0]:48s} {r[1]:8.1f}  {r[2]}")
    ndone = sum(done(it['id']) for it in its)
    print(f"\n{ndone}/{len(its)} items done. Remaining: required {total['req']:.0f} min in {win} windows of <= {cap} min "
          f"(+ ~2 min per window for service stop/start); optional {total['opt']:.0f} min more.")
    print("Estimates use campaign.json assumptions until a model's tuning has run, then the measured rates.")


def cmd_precheck(args):
    """Exit 1 if a heavy job runs now or restic fires within the window; print the timer schedule either way."""
    within = float(args[0]) * 60 if args else CFG["window_minutes"] * 60 + 300
    print(sh("systemctl --user list-timers --all --no-pager"))
    print(sh("systemctl list-timers --all --no-pager"))
    busy, due = heavy_jobs(), timers_due(within)
    print(f"heavy jobs running: {busy or 'none'}; heavy timers due within {within / 60:.0f} min: {due or 'none'}")
    block = busy or [d for d in due if d.startswith("restic")]
    if block:
        print(f"NOT QUIET: {block}")
        sys.exit(1)


def cmd_status(_):
    fails = jload(os.path.join(STATE, "failures.json"), {})
    for it in items(True):
        p = os.path.join(RES, it["id"] + ".json")
        st = jload(p, {}).get("status", "done") if os.path.exists(p) else ("FAILED x%d" % len(fails[it["id"]]) if it["id"] in fails else "pending")
        print(f"{it['id']:48s} {st}{'  (optional)' if it['optional'] else ''}")


def cmd_redo(args):
    os.makedirs(os.path.join(RES, "discarded"), exist_ok=True)
    fails = jload(os.path.join(STATE, "failures.json"), {})
    for a in args:
        p = os.path.join(RES, a + ".json")
        if os.path.exists(p):
            shutil.move(p, os.path.join(RES, "discarded", f"{a}.{time.strftime('%m%d-%H%M%S')}.json"))
        fails.pop(a, None)
        if a.startswith("tune."):
            _, eng, mid = a.split(".")
            pc = os.path.join(STATE, f"probes.{eng}.{mid}.json")
            if os.path.exists(pc):
                shutil.move(pc, pc + f".{time.strftime('%m%d-%H%M%S')}")
        print(f"{a}: pending")
    jsave(os.path.join(STATE, "failures.json"), fails)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "plan"
    {"plan": cmd_plan, "window": cmd_window, "run": cmd_window, "status": cmd_status, "redo": cmd_redo,
     "precheck": cmd_precheck}[c](sys.argv[2:])
