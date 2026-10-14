//! A reload plans the KV pool its cold start planned (GitHub #324). A load
//! after another in the same process reads NVML's free memory with the CUDA
//! context, and the state the kernels set up once per process, still held,
//! and the VRAM plan charges both again as its `cuda_context` and `residual`
//! lines: before #324 the 27B came back from a switch, or from an idle
//! unload, with 8.2% fewer pages.
//!
//! The 27B is loaded through the production load path (`load::prepare_model`,
//! `load::load_model`), shut down, and loaded again in the same process; the
//! second pool must be the first within 1%, which leaves room for the
//! desktop's own drift between the two readings. Its own test binary: the
//! first load must make the process's context. Run under
//! `scripts/gpu-profile.ps1` only (`IGNIS_GPU_PROFILE=1`). Machine-local path:
//! the 27B at [`ARTIFACT_27B`] (or `IGNIS_ARTIFACT_27B`).

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::CudaDevice;
use ignis_core::gpu_profile;

const ARTIFACT_27B: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL_27B: &str = "qwen3.8-27b";

/// The defaults at a context that fits beside the desktop, with idle unload
/// off (ADR 0049): the reload is the test's. The pool takes the rest of the
/// budget at any context.
fn options() -> ignis_server::config::Config {
    let args: Vec<String> = ["--model-max-context", "16384", "--model-idle-unload", "0"].map(String::from).to_vec();
    match ignis_server::config::resolve(&args, |_| None).expect("the options resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => {
            ignis_server::config::Config { model: MODEL_27B.to_owned(), model_named: true, ..config }
        }
        _ => unreachable!("flags without --help are a config"),
    }
}

/// The KV pool's pages of one load of `artifact`, torn down before it
/// returns; `None` when the GPU profile says to skip.
fn pool_pages_of_one_load(artifact: &Path) -> Option<u32> {
    let loaded = match ignis_server::load::prepare_model(&options(), artifact).and_then(ignis_server::load::load_model) {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load {}: {e}", artifact.display()));
            return None;
        }
    };
    let pages = loaded.reservations.as_ref().expect("a text load reserves its pool").kv_pool_pages;
    loaded.model.shut_down();
    drop(loaded);
    Some(pages)
}

#[test]
#[ignore = "GPU profile only: the real 27B artifact, and the card exclusively"]
fn a_reload_plans_the_kv_pool_its_cold_start_planned() {
    let artifact = std::env::var("IGNIS_ARTIFACT_27B").unwrap_or_else(|_| ARTIFACT_27B.to_owned());
    let artifact = Path::new(&artifact);
    if !artifact.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {}", artifact.display())) {
        return;
    }
    assert!(!CudaDevice::context_created(), "the cold load must make this process's context");
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().expect("runtime");
    let _guard = runtime.enter();

    let Some(cold) = pool_pages_of_one_load(artifact) else {
        return;
    };
    assert!(CudaDevice::context_created(), "the cold load made the context, and the teardown kept it");
    assert!(ignis_runtime::qwen27b_loaded(), "the cold load left the 27B's per-process state behind");
    let reload = pool_pages_of_one_load(artifact).expect("the reload loads what the cold start loaded");
    println!("27B KV pool: cold {cold} pages, reload {reload} pages");
    assert!(
        cold.abs_diff(reload) <= cold / 100,
        "the reload planned {reload} pages against the cold start's {cold}: more than 1% apart"
    );
}
