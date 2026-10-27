//! titan-variants: one declaration -> pruned kernel variants, entry points and a measured route table.
//!
//! Ports two mechanisms from trymirai/uzu (`crates/uzu-engine/build/`):
//!   * `metal/variant_combinations.rs::constrained_combinations` -- prune the Cartesian product of
//!     variant axes with a predicate evaluated during the search, small domains first.
//!   * `metal/wrapper.rs` + `common/traitgen.rs` -- emit one entry point per surviving combination
//!     and generate the shared trait from the same parsed signature.
//! and adds uzu's `metal/kernel/matmul/qmv/routes.rs` idea: dispatch decisions as a *table* keyed by
//! device/arch/quant/role, with exact-device match, then arch fallback only when the rows agree,
//! then a generic fallback -- instead of an environment variable per knob.
//!
//! Modes:
//!   gen <spec>      emit the kernel region to stdout
//!   verify <spec>   regenerate and diff the generated names against an existing crate's region
//!   prune <spec>    report naive product vs viable combinations (what the pruner saves)
//!   route <spec>    emit a `route()` dispatch fn from [[route.row]] entries

use std::collections::BTreeMap;
use std::fmt::Write as _;

// ---------------------------------------------------------------- spec model

#[derive(Debug, Clone)]
enum Value {
    Bool(bool),
    Str(String),
}

impl Value {
    fn as_str(&self) -> &str {
        match self {
            Value::Bool(_) => "",
            Value::Str(s) => s,
        }
    }
    fn truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Str(s) => !s.is_empty(),
        }
    }
}

#[derive(Debug)]
struct AxisValue {
    /// flattened `axis.field` -> value, plus `axis.name`
    fields: BTreeMap<String, Value>,
}

#[derive(Debug)]
struct Axis {
    name: String,
    values: Vec<AxisValue>,
}

#[derive(Debug)]
struct Kernel {
    args: String,
    call: String,
    entry: String,
    name_template: String,
    begin: String,
    end: String,
    target: Option<String>,
}

#[derive(Debug)]
struct RouteRow {
    device: String,
    arch: String,
    quant: String,
    role: String,
    m: String,
    keys: String,
    kernel: String,
    measured_by: String,
}

#[derive(Debug)]
struct TwinVariant {
    name: String,
    id: u32,
    prefix: String,
    ftz: bool,
    doc: String,
}

#[derive(Debug)]
struct Twin {
    enum_name: String,
    check_file: Option<String>,
}

#[derive(Debug)]
struct Spec {
    kernel: Kernel,
    axes: Vec<Axis>,
    constraints: Vec<String>,
    routes: Vec<RouteRow>,
    twin: Option<(Twin, Vec<TwinVariant>)>,
}

fn parse_spec(text: &str) -> Result<Spec, String> {
    let root: toml::Value = text.parse().map_err(|e: toml::de::Error| e.to_string())?;
    let table = root.as_table().ok_or("spec must be a table")?;

    let empty = toml::value::Table::new();
    let k = table.get("kernel").and_then(|v| v.as_table()).unwrap_or(&empty);
    if !k.is_empty() && !k.contains_key("name_template") {
        return Err("missing kernel.name_template".into());
    }
    let opt = |key: &str| -> String { k.get(key).and_then(|v| v.as_str()).unwrap_or("").to_owned() };
    let kernel = Kernel {
        args: opt("args"),
        call: opt("call"),
        entry: opt("entry"),
        name_template: opt("name_template"),
        begin: opt("begin"),
        end: opt("end"),
        target: k.get("target").and_then(|v| v.as_str()).map(str::to_owned),
    };

    let mut axes = Vec::new();
    let axis_array = table.get("axis").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    for axis in &axis_array {
        let axis = axis.as_table().ok_or("axis must be a table")?;
        let name = axis.get("name").and_then(|v| v.as_str()).ok_or("axis.name")?.to_owned();
        let mut values = Vec::new();
        for value in axis.get("value").and_then(|v| v.as_array()).ok_or("axis.value")? {
            let vt = value.as_table().ok_or("axis.value entry must be a table")?;
            let mut fields = BTreeMap::new();
            for (fk, fv) in vt {
                let v = match fv {
                    toml::Value::Boolean(b) => Value::Bool(*b),
                    toml::Value::String(s) => Value::Str(s.clone()),
                    toml::Value::Integer(i) => Value::Str(i.to_string()),
                    other => return Err(format!("axis {name}.{fk}: unsupported value {other:?}")),
                };
                fields.insert(format!("{name}.{fk}"), v);
            }
            values.push(AxisValue { fields });
        }
        axes.push(Axis { name, values });
    }

    let constraints = table
        .get("constraints")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default();

    let mut routes = Vec::new();
    if let Some(rows) = table.get("route").and_then(|v| v.get("row")).and_then(|v| v.as_array()) {
        for row in rows {
            let t = row.as_table().ok_or("route.row must be a table")?;
            let g = |key: &str| -> Result<String, String> {
                t.get(key)
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
                    .ok_or_else(|| format!("route.row missing {key}"))
            };
            let measured_by = g("measured_by")?;
            if measured_by.is_empty() {
                return Err("route row has no measured_by provenance; refuse to emit unmeasured rows".into());
            }
            routes.push(RouteRow {
                device: g("device")?,
                arch: g("arch")?,
                quant: g("quant")?,
                role: g("role")?,
                m: g("m")?,
                keys: t.get("keys").and_then(|v| v.as_str()).unwrap_or("0..4294967295").to_owned(),
                kernel: g("kernel")?,
                measured_by,
            });
        }
    }

    let mut twin = None;
    if let Some(t) = table.get("twin").and_then(|v| v.as_table()) {
        let enum_name = t.get("enum_name").and_then(|v| v.as_str()).unwrap_or("Format").to_owned();
        let check_file = t.get("check_file").and_then(|v| v.as_str()).map(str::to_owned);
        let mut vars: Vec<TwinVariant> = Vec::new();
        if let Some(arr) = t.get("variant").and_then(|v| v.as_array()) {
            for v in arr {
                let vt = v.as_table().ok_or("twin.variant must be a table")?;
                let name = vt
                    .get("name")
                    .and_then(|x| x.as_str())
                    .ok_or("twin.variant.name is required")?
                    .to_owned();
                vars.push(TwinVariant {
                    id: vt.get("id").and_then(|x| x.as_integer()).unwrap_or(vars.len() as i64) as u32,
                    prefix: vt.get("prefix").and_then(|x| x.as_str()).unwrap_or("").to_owned(),
                    ftz: vt.get("ftz").and_then(|x| x.as_bool()).unwrap_or(false),
                    doc: vt.get("doc").and_then(|x| x.as_str()).unwrap_or("").to_owned(),
                    name,
                });
            }
        }
        twin = Some((Twin { enum_name, check_file }, vars));
    }

    Ok(Spec { kernel, axes, constraints, routes, twin })
}

// ------------------------------------------------------------- expressions

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    True,
    False,
    Eq,
    Ne,
    And,
    Or,
    Not,
    LParen,
    RParen,
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let b: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        match c {
            ' ' | '\t' | '\n' => i += 1,
            '(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '&' if b.get(i + 1) == Some(&'&') => {
                out.push(Tok::And);
                i += 2;
            }
            '|' if b.get(i + 1) == Some(&'|') => {
                out.push(Tok::Or);
                i += 2;
            }
            '=' if b.get(i + 1) == Some(&'=') => {
                out.push(Tok::Eq);
                i += 2;
            }
            '!' if b.get(i + 1) == Some(&'=') => {
                out.push(Tok::Ne);
                i += 2;
            }
            '!' => {
                out.push(Tok::Not);
                i += 1;
            }
            '"' => {
                i += 1;
                let mut s = String::new();
                while i < b.len() && b[i] != '"' {
                    if b[i] == '\\' {
                        i += 1;
                    }
                    s.push(*b.get(i).ok_or("unterminated string")?);
                    i += 1;
                }
                i += 1;
                out.push(Tok::Str(s));
            }
            c if c.is_alphanumeric() || c == '_' || c == '.' => {
                let mut s = String::new();
                while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_' || b[i] == '.') {
                    s.push(b[i]);
                    i += 1;
                }
                out.push(match s.as_str() {
                    "true" => Tok::True,
                    "false" => Tok::False,
                    _ => Tok::Ident(s),
                });
            }
            other => return Err(format!("unexpected char {other:?}")),
        }
    }
    Ok(out)
}

fn validate_expr(src: &str) -> Result<(), String> {
    let toks = lex(src)?;
    let mut depth: i32 = 0;
    for t in &toks {
        match t {
            Tok::LParen => depth += 1,
            Tok::RParen => {
                depth -= 1;
                if depth < 0 {
                    return Err(format!("unbalanced ) in {src:?}"));
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(format!("unbalanced ( in {src:?}"));
    }
    // A constraint must be evaluable: it has to reference at least one identifier.
    if !toks.iter().any(|t| matches!(t, Tok::Ident(_))) {
        return Err(format!("constraint {src:?} references no axis field"));
    }
    Ok(())
}

/// Evaluate a boolean expression (`a.b == "x" && !c.flag`) against the flattened combination.
/// Returns a tri-state: Some(true/false) when decidable, None when a referenced field is unassigned
/// (used by the pruner to keep a partial assignment alive).
fn eval(src: &str, env: &BTreeMap<String, Option<&Value>>) -> Option<bool> {
    struct E<'a> {
        toks: Vec<Tok>,
        pos: usize,
        env: &'a BTreeMap<String, Option<&'a Value>>,
        unknown: bool,
    }
    impl<'a> E<'a> {
        fn peek(&self) -> Option<&Tok> {
            self.toks.get(self.pos)
        }
        fn value_of(&mut self) -> Option<Value> {
            match self.peek().cloned() {
                Some(Tok::True) => {
                    self.pos += 1;
                    Some(Value::Bool(true))
                }
                Some(Tok::False) => {
                    self.pos += 1;
                    Some(Value::Bool(false))
                }
                Some(Tok::Str(s)) => {
                    self.pos += 1;
                    Some(Value::Str(s))
                }
                Some(Tok::Ident(name)) => {
                    self.pos += 1;
                    match self.env.get(&name) {
                        Some(Some(v)) => Some((*v).clone()),
                        Some(None) => {
                            self.unknown = true;
                            None
                        }
                        None => {
                            self.unknown = true;
                            None
                        }
                    }
                }
                _ => {
                    self.unknown = true;
                    None
                }
            }
        }
        fn or_expr(&mut self) -> Option<bool> {
            let mut acc = self.and_expr();
            while self.peek() == Some(&Tok::Or) {
                self.pos += 1;
                let rhs = self.and_expr();
                acc = match (acc, rhs) {
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                };
            }
            acc
        }
        fn and_expr(&mut self) -> Option<bool> {
            let mut acc = self.atom();
            while self.peek() == Some(&Tok::And) {
                self.pos += 1;
                let rhs = self.atom();
                acc = match (acc, rhs) {
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                };
            }
            acc
        }
        fn atom(&mut self) -> Option<bool> {
            if self.peek() == Some(&Tok::Not) {
                self.pos += 1;
                return self.atom().map(|b| !b);
            }
            if self.peek() == Some(&Tok::LParen) {
                self.pos += 1;
                let v = self.or_expr();
                if self.peek() == Some(&Tok::RParen) {
                    self.pos += 1;
                }
                return v;
            }
            let lhs = self.value_of();
            let op = self.peek().cloned();
            match op {
                Some(Tok::Eq) | Some(Tok::Ne) => {
                    self.pos += 1;
                    let rhs = self.value_of();
                    let eq = match (lhs, rhs) {
                        (Some(a), Some(b)) => match (&a, &b) {
                            (Value::Bool(x), Value::Bool(y)) => x == y,
                            (Value::Str(x), Value::Str(y)) => x == y,
                            (Value::Bool(x), Value::Str(y)) => x.to_string() == *y,
                            (Value::Str(x), Value::Bool(y)) => *x == y.to_string(),
                        },
                        _ => return None,
                    };
                    Some(if op == Some(Tok::Ne) { !eq } else { eq })
                }
                _ => lhs.map(|v| v.truthy()),
            }
        }
    }

    let toks = lex(src).ok()?;
    let mut e = E { toks, pos: 0, env, unknown: false };
    let v = e.or_expr();
    if e.unknown {
        None
    } else {
        v
    }
}

// --------------------------------------------------------------- pruner
// Port of uzu `metal/variant_combinations.rs::constrained_combinations`: depth-first over the
// domains, binding the smallest domain first, pruning whole sub-trees the constraints can never
// satisfy, and returning the survivors in Cartesian-product order.

fn constrained_combinations(
    domain_lengths: &[usize],
    mut could_satisfy: impl FnMut(&[Option<usize>], bool) -> bool,
) -> Vec<Box<[usize]>> {
    let complete = domain_lengths.is_empty();
    let mut selection = vec![None; domain_lengths.len()];
    if !could_satisfy(&selection, complete) {
        return Vec::new();
    }
    if complete {
        return vec![Box::new([])];
    }
    let mut axis_order: Vec<_> = (0..domain_lengths.len()).collect();
    axis_order.sort_by_key(|&axis| domain_lengths[axis]);

    let mut combinations = Vec::new();
    let first_axis = axis_order[0];
    let mut stack = vec![(first_axis, 0..domain_lengths[first_axis])];
    while let Some((axis, values)) = stack.last_mut() {
        let axis = *axis;
        let Some(value) = values.next() else {
            selection[axis] = None;
            stack.pop();
            continue;
        };
        selection[axis] = Some(value);
        let next_axis = axis_order.get(stack.len()).copied();
        let viable = could_satisfy(&selection, next_axis.is_none());
        if !viable {
            continue;
        }
        let Some(next_axis) = next_axis else {
            combinations.push(selection.iter().map(|value| value.unwrap()).collect());
            continue;
        };
        stack.push((next_axis, 0..domain_lengths[next_axis]));
    }
    combinations.sort_unstable();
    combinations
}

fn viable(spec: &Spec) -> Vec<Box<[usize]>> {
    let domain_lengths: Vec<usize> = spec.axes.iter().map(|a| a.values.len()).collect();
    constrained_combinations(&domain_lengths, |selection, _complete| {
        let mut env: BTreeMap<String, Option<&Value>> = BTreeMap::new();
        for (axis, choice) in spec.axes.iter().zip(selection) {
            match choice {
                Some(i) => {
                    for (k, v) in &axis.values[*i].fields {
                        env.insert(k.clone(), Some(v));
                    }
                }
                None => {
                    for k in axis.values.iter().flat_map(|v| v.fields.keys()) {
                        env.entry(k.clone()).or_insert(None);
                    }
                }
            }
        }
        spec.constraints.iter().all(|src| eval(src, &env) != Some(false))
    })
}

/// Interpolate `{axis.field}` placeholders until stable, so a value's own text may reference other
/// axes (e.g. `doc = "{fmt.name}: ... ({fmt.prog})"`).
fn interpolate(template: &str, spec: &Spec, combo: &[usize], name: Option<&str>) -> String {
    let mut out = template.to_string();
    if let Some(name) = name {
        out = out.replace("{name}", name);
    }
    out = out.replace("{args}", &spec.kernel.args);
    out = out.replace("{call}", &spec.kernel.call);
    for _ in 0..4 {
        let before = out.clone();
        for (axis, choice) in spec.axes.iter().zip(combo) {
            for (k, v) in &axis.values[*choice].fields {
                let field = k.strip_prefix(&format!("{}.", axis.name)).unwrap();
                out = out.replace(&format!("{{{}.{}}}", axis.name, field), v.as_str());
            }
        }
        if out == before {
            return out;
        }
    }
    out
}

fn name_of(spec: &Spec, combo: &[usize]) -> String {
    let out = interpolate(&spec.kernel.name_template, spec, combo, None);
    if let Some(i) = out.find('{') {
        panic!("unresolved template field in {:?}: {}", spec.kernel.name_template, &out[i..]);
    }
    out
}

fn render_entry(spec: &Spec, combo: &[usize], name: &str) -> String {
    // `{{`/`}}` are the escape used in the specs (they were f-string bodies); emit real braces.
    interpolate(&spec.kernel.entry, spec, combo, Some(name)).replace("{{", "{").replace("}}", "}")
}

// ------------------------------------------------------------------ modes

fn region_names(path: &str, begin: &str, end: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let start = text.find(begin).ok_or_else(|| format!("{path}: no begin marker {begin:?}"))?;
    let rest = &text[start + begin.len()..];
    let stop = rest.find(end).ok_or_else(|| format!("{path}: no end marker {end:?}"))?;
    let mut names = Vec::new();
    for line in rest[..stop].lines() {
        if let Some(idx) = line.find("pub unsafe fn ") {
            let tail = &line[idx + "pub unsafe fn ".len()..];
            let name: String = tail.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if !name.is_empty() {
                names.push(name);
            }
        }
    }
    Ok(names)
}

/// uzu `common/traitgen.rs` adapted to titan: the GPU side enumerates one entry point per
/// (format x variant), the CPU twin dispatches on a runtime `Format`. Generating the enum and its
/// exhaustive helpers from the spec means a new format cannot be added without touching every
/// exhaustive match, which is the compile-time parity property the trait gives uzu.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn emit_twin(e: &str, vars: &[TwinVariant], spec_hash: u64) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by titan-variants twin. Do not edit; edit the spec and re-run:").unwrap();
    writeln!(out, "//   titan-variants twin specs/twin-moe-formats.toml > titan-kernels-oxide/src/format_gen.rs").unwrap();
    writeln!(out, "// spec-fnv1a: {spec_hash:#018x}").unwrap();
    writeln!(out, "#[derive(Clone, Copy, PartialEq, Eq, Debug)]").unwrap();
    writeln!(out, "#[repr(u32)]").unwrap();
    writeln!(out, "pub enum {e} {{").unwrap();
    for v in vars {
        if !v.doc.is_empty() {
            writeln!(out, "    /// {}", v.doc).unwrap();
        }
        writeln!(out, "    {} = {},", v.name, v.id).unwrap();
    }
    writeln!(out, "}}\n").unwrap();

    writeln!(out, "#[allow(dead_code)]").unwrap();
    writeln!(out, "impl {e} {{").unwrap();
    writeln!(
        out,
        "    pub const ALL: &'static [{e}] = &[{}];\n",
        vars.iter().map(|v| format!("{e}::{}", v.name)).collect::<Vec<_>>().join(", ")
    )
    .unwrap();

    writeln!(out, "    pub const fn id(self) -> u32 {{").unwrap();
    writeln!(out, "        self as u32").unwrap();
    writeln!(out, "    }}\n").unwrap();

    writeln!(out, "    /// `None` for an id no format claims: the caller must fail, not guess.").unwrap();
    writeln!(out, "    pub const fn from_id(id: u32) -> Option<{e}> {{").unwrap();
    writeln!(out, "        match id {{").unwrap();
    for v in vars {
        writeln!(out, "            {} => Some({e}::{}),", v.id, v.name).unwrap();
    }
    writeln!(out, "            _ => None,").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}\n").unwrap();

    // Exhaustive: no wildcard, so adding a variant breaks the build here first.
    writeln!(out, "    /// GPU entry-point prefix (`{{prefix}}_moe_b1`, ...).").unwrap();
    writeln!(out, "    pub const fn prefix(self) -> &'static str {{").unwrap();
    writeln!(out, "        match self {{").unwrap();
    for v in vars {
        writeln!(out, "            {e}::{} => {:?},", v.name, v.prefix).unwrap();
    }
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}\n").unwrap();

    writeln!(out, "    /// Float program family: true means flush-to-zero / fast-math (llama.cpp's).").unwrap();
    writeln!(out, "    pub const fn ftz(self) -> bool {{").unwrap();
    writeln!(out, "        match self {{").unwrap();
    for v in vars {
        writeln!(out, "            {e}::{} => {},", v.name, v.ftz).unwrap();
    }
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    out
}

/// Text check of the hand-written CPU twin: which formats have an explicit arm at each
/// `match fmt {` site, and which are silently absorbed by a `_ =>` wildcard.
fn check_twin(path: &str, e: &str, vars: &[TwinVariant]) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|err| format!("{path}: {err}"))?;
    let lines: Vec<&str> = text.lines().collect();
    let mut sites: Vec<(usize, Vec<String>, bool)> = Vec::new(); // (line, arms, has_wildcard)
    for (i, line) in lines.iter().enumerate() {
        if !line.contains("match fmt") {
            continue;
        }
        let mut arms = Vec::new();
        let mut wildcard = false;
        let mut depth = 0i32;
        for (j, l) in lines.iter().enumerate().skip(i) {
            depth += l.matches('{').count() as i32 - l.matches('}').count() as i32;
            if l.trim_start().starts_with("_ =>") {
                wildcard = true;
            }
            for v in vars {
                let needle = format!("{e}::{}", v.name);
                if l.contains(&needle) && j > i && !arms.contains(&v.name) {
                    arms.push(v.name.clone());
                }
            }
            if depth <= 0 && j > i {
                break;
            }
        }
        sites.push((i + 1, arms, wildcard));
    }
    eprintln!("cpu twin: {path}  ({} match {e} sites)", sites.len());
    for (line, arms, wildcard) in &sites {
        let absorbed: Vec<&str> = vars
            .iter()
            .map(|v| v.name.as_str())
            .filter(|n| !arms.iter().any(|a| a == n))
            .collect();
        eprintln!(
            "  line {line:>5}: {} explicit arm(s){}",
            arms.len(),
            if *wildcard { ", `_ =>` wildcard present" } else { ", NO wildcard (exhaustive)" }
        );
        if !absorbed.is_empty() {
            eprintln!("           absorbed by the wildcard: {}", absorbed.join(", "));
        }
    }
    let unreferenced: Vec<&str> = vars
        .iter()
        .map(|v| v.name.as_str())
        .filter(|n| !text.contains(&format!("{e}::{n}")))
        .collect();
    eprintln!("  formats not referenced anywhere in the file: {}", if unreferenced.is_empty() { "none".to_string() } else { unreferenced.join(", ") });
    Ok(())
}

fn region_text(path: &str, begin: &str, end: &str) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let start = text.find(begin).ok_or_else(|| format!("{path}: no begin marker"))? + begin.len();
    let rest = &text[start..];
    let stop = rest.find(end).ok_or_else(|| format!("{path}: no end marker"))?;
    Ok(rest[..stop].to_string())
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or("usage: titan-variants <gen|verify|prune|route> <spec.toml>")?;
    let path = args.next().ok_or("missing spec path")?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let spec = parse_spec(&text)?;
    for c in &spec.constraints {
        validate_expr(c)?;
    }

    if matches!(mode.as_str(), "gen" | "verify" | "prune")
        && (spec.kernel.name_template.is_empty() || spec.kernel.entry.is_empty() || spec.axes.is_empty())
    {
        return Err(format!("{mode}: spec needs [kernel] fields and at least one [[axis]]"));
    }

    match mode.as_str() {
        "gen" => {
            let mut out = String::new();
            for combo in viable(&spec) {
                let name = name_of(&spec, &combo);
                out.push_str(&render_entry(&spec, &combo, &name));
                out.push('\n');
            }
            print!("{out}");
        }
        "prune" => {
            let naive: usize = spec.axes.iter().map(|a| a.values.len()).product();
            let combinations = viable(&spec).len();
            let axes: Vec<String> =
                spec.axes.iter().map(|a| format!("{}={}", a.name, a.values.len())).collect();
            println!("axes:        {}", axes.join(" x "));
            println!("naive:       {naive}");
            println!("viable:      {combinations}");
            println!("pruned:      {} ({:.1}%)", naive - combinations, 100.0 * (naive - combinations) as f64 / naive as f64);
            for c in &spec.constraints {
                println!("constraint:  {c}");
            }
        }
        "verify" => {
            let target = spec.kernel.target.clone().ok_or("spec has no kernel.target")?;
            let target = if std::path::Path::new(&target).is_absolute() {
                target
            } else {
                let specdir = std::path::Path::new(&path).parent().unwrap_or(std::path::Path::new("."));
                specdir.join(&target).to_string_lossy().into_owned()
            };
            let existing = region_names(&target, &spec.kernel.begin, &spec.kernel.end)?;
            let generated: Vec<String> = viable(&spec).iter().map(|c| name_of(&spec, c)).collect();
            let existing_set: std::collections::BTreeSet<&String> = existing.iter().collect();
            let generated_set: std::collections::BTreeSet<&String> = generated.iter().collect();
            let missing: Vec<_> = existing_set.difference(&generated_set).collect();
            let extra: Vec<_> = generated_set.difference(&existing_set).collect();
            println!("target:      {target}");
            println!("existing:    {} entry points in the generated region", existing.len());
            println!("regenerated: {} entry points", generated.len());
            println!("missing:     {}", missing.len());
            for m in missing.iter().take(10) {
                println!("   - {m}");
            }
            println!("extra:       {}", extra.len());
            for e in extra.iter().take(10) {
                println!("   + {e}");
            }
            let existing_text = region_text(&target, &spec.kernel.begin, &spec.kernel.end)?;
            let generated_text: String = viable(&spec)
                .iter()
                .map(|c| render_entry(&spec, c, &name_of(&spec, c)))
                .collect::<Vec<_>>()
                .join("\n");
            let norm = |t: &str| -> Vec<String> {
                t.lines().map(|l| l.trim_end().to_string()).filter(|l| !l.trim().is_empty()).collect()
            };
            let (a, b) = (norm(&existing_text), norm(&generated_text));
            println!("text:        {} lines existing vs {} generated", a.len(), b.len());
            let mut text_diffs = 0;
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                if x != y {
                    text_diffs += 1;
                    if text_diffs <= 3 {
                        println!("   line {} differs:\n     existing:  {x}\n     generated: {y}", i + 1);
                    }
                }
            }
            text_diffs += a.len().abs_diff(b.len());
            println!("text diffs:  {text_diffs}");
            if missing.is_empty() && extra.is_empty() && text_diffs == 0 {
                println!("PASS: the spec reproduces the crate's generated region exactly (names + text)");
            } else {
                println!("FAIL");
                return Err("verification failed".into());
            }
        }
        "twin" => {
            let (twin, vars) = spec.twin.as_ref().ok_or("spec has no [twin] section")?;
            if vars.is_empty() {
                return Err("spec has no [[twin.variant]]".into());
            }
            let hash = fnv1a(text.as_bytes());
            print!("{}", emit_twin(&twin.enum_name, vars, hash));
            if let Some(file) = &twin.check_file {
                let path = if std::path::Path::new(file).is_absolute() {
                    file.clone()
                } else {
                    let specdir = std::path::Path::new(&path).parent().unwrap_or(std::path::Path::new("."));
                    specdir.join(file).to_string_lossy().into_owned()
                };
                check_twin(&path, &twin.enum_name, vars)?;
            }
        }
        "route" => {
            if spec.routes.is_empty() {
                return Err("spec has no [[route.row]]".into());
            }
            let mut out = String::new();
            out.push_str(
                "// GENERATED by titan-variants. Do not edit: rows come from measured A/B runs.\n\
                 // Selection: exact device -> same arch only if every measured row agrees -> None (generic).\n\
                 #[derive(Clone, Copy, PartialEq, Eq, Debug)]\n\
                 pub struct RouteKey<'a> {\n\
                 \x20   pub device: &'a str,\n\
                 \x20   pub arch: &'a str,\n\
                 \x20   pub quant: &'a str,\n\
                 \x20   pub role: &'a str,\n\
                 \x20   pub m: u32,\n\
                 \x20   /// Context depth in keys (KV length) when the kernel runs.\n\
                 \x20   pub keys: u32,\n\
                 }\n\n\
                 pub struct RouteRow {\n\
                 \x20   pub device: &'static str,\n\
                 \x20   pub arch: &'static str,\n\
                 \x20   pub quant: &'static str,\n\
                 \x20   pub role: &'static str,\n\
                 \x20   pub m_lo: u32,\n\
                 \x20   pub m_hi: u32,\n\
                 \x20   pub keys_lo: u32,\n\
                 \x20   pub keys_hi: u32,\n\
                 \x20   pub kernel: &'static str,\n\
                 \x20   pub measured_by: &'static str,\n\
                 }\n\n\
                 pub const ROWS: &[RouteRow] = &[\n",
            );
            for r in &spec.routes {
                let (lo, hi) = parse_m(&r.m)?;
                let (klo, khi) = parse_m(&r.keys)?;
                writeln!(
                    out,
                    "    RouteRow {{ device: {:?}, arch: {:?}, quant: {:?}, role: {:?}, m_lo: {lo}, m_hi: {hi}, keys_lo: {klo}, keys_hi: {khi}, kernel: {:?}, measured_by: {:?} }},",
                    r.device, r.arch, r.quant, r.role, r.kernel, r.measured_by
                )
                .unwrap();
            }
            out.push_str(
                "];\n\n\
                 pub fn route(key: &RouteKey<'_>) -> Option<&'static RouteRow> {\n\
                 \x20   let matches = |r: &&RouteRow| {\n\
                 \x20       r.quant == key.quant\n\
                 \x20           && r.role == key.role\n\
                 \x20           && (r.m_lo..=r.m_hi).contains(&key.m)\n\
                 \x20           && (r.keys_lo..=r.keys_hi).contains(&key.keys)\n\
                 \x20   };\n\
                 \x20   ROWS.iter().filter(matches).find(|r| r.device == key.device).or_else(|| {\n\
                 \x20       // Same GPU generation, but only when every measured row agrees; otherwise\n\
                 \x20       // fall through to the generic kernel rather than guess.\n\
                 \x20       let mut same = ROWS.iter().filter(matches).filter(|r| r.arch == key.arch);\n\
                 \x20       let first = same.next()?;\n\
                 \x20       same.all(|r| r.kernel == first.kernel).then_some(first)\n\
                 \x20   })\n\
                 }\n",
            );
            print!("{out}");
        }
        other => return Err(format!("unknown mode {other}")),
    }
    Ok(())
}

fn parse_m(m: &str) -> Result<(u32, u32), String> {
    match m.split_once("..") {
        Some((a, b)) => Ok((a.parse().map_err(|_| "bad m")?, b.parse().map_err(|_| "bad m")?)),
        None => {
            let v: u32 = m.parse().map_err(|_| "bad m")?;
            Ok((v, v))
        }
    }
}
