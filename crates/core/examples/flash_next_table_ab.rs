//! Spec flash-next/08's A/B: the same text, teacher-forced through two
//! Flash-Next containers that differ only in the n-gram table.
//!
//! ```text
//! flash_next_table_ab score   <model-dir> <corpus-dir> <out-dir>
//! flash_next_table_ab compare <out-dir-a> <out-dir-b>
//! ```
//!
//! `score` tokenizes every `<corpus-dir>/*.txt` with the container's own
//! frontend, cuts it into windows of 2,048 tokens (at most 8 a file), and for
//! every position that has a next token writes `<file>.nll.f32` (the next
//! token's negative log-likelihood) and `<file>.top64.{i32,f32}` (the 64 most
//! probable ids and their log-probabilities), then `<file>.json` with the
//! perplexity. `compare` reads two such directories and prints, per file and
//! overall: both perplexities, the share of positions whose top-1 agrees,
//! and KL(a || b) over a's top 64 (b's log-probability of an id outside its
//! own top 64 floored at its 64th): an approximation, the same order as the
//! converter's top-64 KLD.
//!
//! Needs the GPU to itself and ~38 GB of free RAM (the pinned expert pool):
//! stop the server first.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::frontend::FrontendSet;
use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::Reader;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine};

const WINDOW: usize = 2048;
const MAX_WINDOWS: usize = 8;
const TOP: usize = 64;
const EXPERT_CACHE_BYTES: u64 = 17_000_000_000;

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_i32(path: &Path, values: &[i32]) -> Result<(), String> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn read_i32(path: &Path) -> Result<Vec<i32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn corpus_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("{} holds no .txt files", dir.display()));
    }
    Ok(files)
}

fn score(model_dir: &Path, corpus: &Path, out: &Path) -> Result<(), String> {
    fs::create_dir_all(out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let reader = Reader::open(&model_dir.join(ARTIFACT_FILE_NAME)).map_err(|e| format!("{e:?}"))?;
    let frontend = FrontendSet::from_reader(&reader).map_err(|e| format!("{e:?}"))?;
    let weights_id = reader.identity().weights_id.clone();
    drop(reader);
    let options = EngineOptions {
        decode_lanes: 1,
        max_context_tokens: WINDOW as u32 * 2,
        expert_cache_bytes: EXPERT_CACHE_BYTES,
        capture_graphs: false,
        ..EngineOptions::default()
    };
    let mut engine = FlashNextEngine::load(model_dir, options)?;
    let vocab = engine.vocab();
    let started = Instant::now();
    for file in corpus_files(corpus)? {
        let name = file.file_stem().and_then(|s| s.to_str()).ok_or("a corpus file has no stem")?.to_owned();
        let text = fs::read_to_string(&file).map_err(|e| format!("read {}: {e}", file.display()))?;
        let tokens = frontend.tokenizer().encode(&text).map_err(|e| format!("{e:?}"))?;
        let (mut nll, mut top_ids, mut top_lp) = (Vec::new(), Vec::new(), Vec::new());
        for window in tokens.chunks(WINDOW).take(MAX_WINDOWS).filter(|w| w.len() > 1) {
            engine.span_logits(window, &mut |first, rows| {
                for (r, row) in rows.chunks_exact(vocab).enumerate() {
                    let p = first + r;
                    if p + 1 >= window.len() {
                        continue;
                    }
                    let values: Vec<f32> = row.iter().map(|&b| bf16(b)).collect();
                    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let lse = max + values.iter().map(|v| ((v - max) as f64).exp()).sum::<f64>().ln() as f32;
                    nll.push(lse - values[window[p + 1] as usize]);
                    let mut order: Vec<usize> = (0..vocab).collect();
                    order.select_nth_unstable_by(TOP, |a, b| values[*b].total_cmp(&values[*a]));
                    let mut top: Vec<usize> = order[..TOP].to_vec();
                    top.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
                    top_ids.extend(top.iter().map(|&i| i as i32));
                    top_lp.extend(top.iter().map(|&i| values[i] - lse));
                }
                Ok(())
            })?;
        }
        let mean = nll.iter().map(|&x| f64::from(x)).sum::<f64>() / nll.len().max(1) as f64;
        write_f32(&out.join(format!("{name}.nll.f32")), &nll)?;
        write_i32(&out.join(format!("{name}.top64.i32")), &top_ids)?;
        write_f32(&out.join(format!("{name}.top64.f32")), &top_lp)?;
        let summary = serde_json::json!({"weights_id": weights_id, "positions": nll.len(), "mean_nll": mean, "ppl": mean.exp()});
        fs::write(out.join(format!("{name}.json")), summary.to_string()).map_err(|e| format!("write summary: {e}"))?;
        eprintln!("[{:>5.0}s] {name}: {} positions, ppl {:.4}", started.elapsed().as_secs_f64(), nll.len(), mean.exp());
    }
    Ok(())
}

fn compare(a: &Path, b: &Path) -> Result<(), String> {
    let mut names: Vec<String> = fs::read_dir(a)
        .map_err(|e| format!("read {}: {e}", a.display()))?
        .filter_map(|e| e.ok()?.file_name().to_str()?.strip_suffix(".nll.f32").map(str::to_owned))
        .collect();
    names.sort();
    let (mut total, mut nll_a, mut nll_b, mut agree, mut kl) = (0usize, 0f64, 0f64, 0usize, 0f64);
    println!("{:<14} {:>9} {:>10} {:>10} {:>9} {:>10}", "file", "positions", "ppl a", "ppl b", "top1 =", "KL(a||b)");
    for name in names {
        let (na, nb) = (read_f32(&a.join(format!("{name}.nll.f32")))?, read_f32(&b.join(format!("{name}.nll.f32")))?);
        if na.len() != nb.len() {
            return Err(format!("{name}: {} positions in a, {} in b", na.len(), nb.len()));
        }
        let (ia, la) = (read_i32(&a.join(format!("{name}.top64.i32")))?, read_f32(&a.join(format!("{name}.top64.f32")))?);
        let (ib, lb) = (read_i32(&b.join(format!("{name}.top64.i32")))?, read_f32(&b.join(format!("{name}.top64.f32")))?);
        let (mut f_agree, mut f_kl) = (0usize, 0f64);
        for p in 0..na.len() {
            let (ra, rb) = (p * TOP..(p + 1) * TOP, p * TOP..(p + 1) * TOP);
            if ia[ra.start] == ib[rb.start] {
                f_agree += 1;
            }
            let floor = lb[rb.end - 1];
            for k in ra {
                let lp_b = (rb.start..rb.end).find(|&j| ib[j] == ia[k]).map_or(floor, |j| lb[j]);
                f_kl += f64::from(la[k]).exp() * f64::from(la[k] - lp_b);
            }
        }
        let n = na.len();
        let (sa, sb) = (na.iter().map(|&x| f64::from(x)).sum::<f64>(), nb.iter().map(|&x| f64::from(x)).sum::<f64>());
        println!(
            "{name:<14} {n:>9} {:>10.4} {:>10.4} {:>8.2}% {:>10.5}",
            (sa / n as f64).exp(),
            (sb / n as f64).exp(),
            100.0 * f_agree as f64 / n as f64,
            f_kl / n as f64
        );
        total += n;
        nll_a += sa;
        nll_b += sb;
        agree += f_agree;
        kl += f_kl;
    }
    println!(
        "{:<14} {total:>9} {:>10.4} {:>10.4} {:>8.2}% {:>10.5}",
        "ALL",
        (nll_a / total as f64).exp(),
        (nll_b / total as f64).exp(),
        100.0 * agree as f64 / total as f64,
        kl / total as f64
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["score", model, corpus, out] => score(Path::new(model), Path::new(corpus), Path::new(out)),
        ["compare", a, b] => compare(Path::new(a), Path::new(b)),
        _ => Err("usage: flash_next_table_ab score <model-dir> <corpus-dir> <out-dir> | compare <out-a> <out-b>".into()),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("flash_next_table_ab: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
