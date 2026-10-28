//! Fails the build when `src/format_gen.rs` and the spec that generates it have diverged.
//!
//! The enum lives in source (not `OUT_DIR`) so it is readable and greppable, which means it can go
//! stale. The generator stamps the spec's FNV-1a hash into the file; this script recomputes it and
//! panics with the exact regenerate command when they disagree, so a spec edit can never be silently
//! ignored by a build.

use std::path::{Path, PathBuf};

const SPEC_REL: &str = "../titan-variants/specs/twin-moe-formats.toml";
const GEN_REL: &str = "src/format_gen.rs";

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let spec: PathBuf = root.join(SPEC_REL);
    let generated: PathBuf = root.join(GEN_REL);

    println!("cargo::rerun-if-changed={}", spec.display());
    println!("cargo::rerun-if-changed={}", generated.display());

    // A checkout without the codegen crate beside it (e.g. a vendored copy) is not a failure.
    let Ok(spec_text) = std::fs::read(&spec) else {
        println!("cargo::warning=format_gen.rs staleness check skipped: {} not found", spec.display());
        return;
    };
    let Ok(generated_text) = std::fs::read_to_string(&generated) else {
        println!("cargo::warning=format_gen.rs missing next to cpu.rs; the enum will not compile");
        return;
    };

    let want = format!("spec-fnv1a: {:#018x}", fnv1a(&spec_text));
    if !generated_text.contains(&want) {
        panic!(
            "\n\n{} is stale: the spec {} has changed since it was generated.\n\
             Regenerate it:\n\n    cd {} && titan-variants twin specs/twin-moe-formats.toml \\\n        > ../titan-kernels-oxide/src/format_gen.rs\n\n\
             (expected `{want}` in the generated file)\n",
            Path::new(GEN_REL).display(),
            spec.display(),
            root.parent().unwrap_or(Path::new(".")).join("titan-variants").display(),
        );
    }
}
