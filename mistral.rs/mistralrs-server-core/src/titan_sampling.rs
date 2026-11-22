//! `TITAN_SAMPLING=temp=0.7,top_p=0.8,top_k=20,min_p=0,presence=1.5`: server-side sampling defaults for requests that
//! leave those fields unset (clients such as an agent stack send none, and the GGUF pipeline has no model-card defaults).

use std::sync::OnceLock;

#[derive(Default)]
pub(crate) struct Defaults {
    pub temperature: Option<f64>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f32>,
}

pub(crate) fn defaults() -> &'static Defaults {
    static D: OnceLock<Defaults> = OnceLock::new();
    D.get_or_init(|| {
        let mut d = Defaults::default();
        let Ok(spec) = mistralrs_core::titan_cfg::var("TITAN_SAMPLING") else {
            return d;
        };
        for kv in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let Some((k, v)) = kv.split_once('=') else {
                tracing::warn!("TITAN_SAMPLING: ignoring {kv:?} (want key=value)");
                continue;
            };
            let ok = match k.trim() {
                "temp" | "temperature" => v.trim().parse().map(|x| d.temperature = Some(x)).is_ok(),
                "top_k" => v.trim().parse().map(|x| d.top_k = Some(x)).is_ok(),
                "top_p" => v.trim().parse().map(|x| d.top_p = Some(x)).is_ok(),
                "min_p" => v.trim().parse().map(|x| d.min_p = Some(x)).is_ok(),
                "presence" | "presence_penalty" => v.trim().parse().map(|x| d.presence_penalty = Some(x)).is_ok(),
                _ => false,
            };
            if !ok {
                tracing::warn!("TITAN_SAMPLING: ignoring {kv:?}");
            }
        }
        tracing::info!(
            "titan sampling defaults: temperature {:?}, top_k {:?}, top_p {:?}, min_p {:?}, presence {:?}",
            d.temperature,
            d.top_k,
            d.top_p,
            d.min_p,
            d.presence_penalty
        );
        d
    })
}
