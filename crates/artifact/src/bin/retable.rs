//! `ignis-artifact-retable`: put Qwen's FP8 n-gram table into a packed
//! Flash-Next main container (spec flash-next/08), and optionally re-pin the
//! MTP companion to the result.
//!
//! ```text
//! ignis-artifact-retable --main <in.ninfer> --fp8-source <dir> --out <out.ninfer>
//!                        [--companion <mtp.ninfer> --companion-out <mtp-out.ninfer>]
//! ```
//!
//! `--fp8-source` is a directory holding `Qwen/Qwen3.8-Flash-Next-FP8`'s
//! `model.safetensors.index.json` and the shards that carry the n-gram table
//! (`model-00005` .. `model-00037` at the revision this spec names).

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use ignis_artifact::flash_next::retable::{read_fp8_source, repin_companion, retable};
use ignis_artifact::flash_next::FlashNextGeometry;

fn usage() -> ExitCode {
    eprintln!(
        "usage: ignis-artifact-retable --main <in.ninfer> --fp8-source <dir> --out <out.ninfer> \
         [--companion <mtp.ninfer> --companion-out <mtp-out.ninfer>]"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let (mut main, mut source, mut out, mut companion, mut companion_out) = (None, None, None, None, None);
    while let Some(flag) = args.next() {
        let value = args.next().map(PathBuf::from);
        match flag.as_str() {
            "--main" => main = value,
            "--fp8-source" => source = value,
            "--out" => out = value,
            "--companion" => companion = value,
            "--companion-out" => companion_out = value,
            _ => return usage(),
        }
    }
    let (Some(main), Some(source), Some(out)) = (main, source, out) else { return usage() };
    if companion.is_some() != companion_out.is_some() {
        return usage();
    }
    let geometry = FlashNextGeometry::qwen38_flash_next();
    let started = Instant::now();
    let run = || -> Result<(), ignis_artifact::ArtifactError> {
        let fp8 = read_fp8_source(&source, &geometry)?;
        eprintln!("FP8 table: {} rows in {} shards", fp8.rows(), fp8.shards.len());
        let report = retable(&main, &fp8, &out, &geometry, &mut |line| {
            eprintln!("[{:>5.0}s] {line}", started.elapsed().as_secs_f64())
        })?;
        eprintln!("wrote {} ({} bytes, {} objects)", report.path.display(), report.bytes, report.objects);
        if let (Some(companion), Some(companion_out)) = (&companion, &companion_out) {
            repin_companion(companion, &out, companion_out)?;
            eprintln!("re-pinned {} -> {}", companion.display(), companion_out.display());
        }
        Ok(())
    };
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("retable: {e}");
            ExitCode::FAILURE
        }
    }
}
