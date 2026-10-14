//! The condition budget (spec image-model/05 §The condition budget, spec
//! image-model/04 §Load and plan): the size each condition image of an edit
//! is resized to before it is encoded.
//!
//! Each image first gets diffusers' size for it — 1024² pixels of area at its
//! own aspect ratio, each side a multiple of 32 ([`condition_size`]). A
//! condition prefix holds the prompt's tokens plus, per image, one token per
//! latent cell ([`ImageSize::latent_cells`]: one image-pad slot of the
//! encoder becomes 2 × 2 latent tokens, so a 1024² image is 4,096) — the
//! runner counts it, holding the tokenizer
//! ([`ignis_core::image::ImageRunner::condition_tokens`]):
//!
//! ```text
//! prefix tokens = tokens(prompt as templated) + Σ (W_i / 16) · (H_i / 16)
//! ```
//!
//! When that is over the budget, every image is scaled by one factor — all
//! of them get the same smaller area — until the prefix fits, never below
//! `image.condition_floor_px`² of area before its sides are rounded to 32
//! (the rounding may take an image a little under it: 672 x 384 at a 512
//! floor); one that does not fit even there is refused, naming the budget
//! ([`fit`]).

use ignis_core::image::{ImagePlan, ImageSize};

/// The side of the square whose area a condition image is resized to before
/// the budget: diffusers' `output_resolution`, 1024.
pub const CONDITION_SIDE: u32 = 1024;

/// diffusers' `calculate_dimensions` (`pipelines/qwenimage21/pipeline_qwenimage21.py`,
/// copied there from `qwenimage/pipeline_qwenimage_edit.py`): the size of
/// `area` pixels at `ratio` (width / height), each side rounded to a multiple
/// of 32 the way Python's `round` does, halves to even. Where diffusers
/// would round a side of a sliver to 0, it is 32.
pub fn condition_size(area: f64, ratio: f64) -> ImageSize {
    let width = (area * ratio).sqrt();
    let height = width / ratio;
    let side = |pixels: f64| ((pixels / 32.0).round_ties_even() as u32).max(1) * 32;
    ImageSize::new(side(width), side(height))
}

/// The condition budget a job gets, in condition-prefix tokens: the load's
/// plan, lowered to `image.condition_budget_tokens` when that is set (0 is
/// the plan's own).
pub fn budget(plan: &ImagePlan, configured: u32) -> u64 {
    match configured {
        0 => plan.condition_budget_tokens,
        configured => plan.condition_budget_tokens.min(u64::from(configured)),
    }
}

/// The sizes the budget left an edit's condition images at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fitted {
    /// One per condition image, in order.
    pub sizes: Vec<ImageSize>,
    /// Smaller than [`condition_size`] at 1024²: the response reports them
    /// (`condition_sizes`).
    pub scaled: bool,
}

/// Why an edit's condition images do not fit the budget even at the floor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverBudget {
    /// The prefix's tokens with every image at the floor's area.
    pub tokens: u64,
    /// The condition budget.
    pub budget: u64,
    /// `image.condition_floor_px`.
    pub floor: u32,
}

impl std::fmt::Display for OverBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the prompt and the condition images take {} condition-prefix tokens even with every image scaled down to {}x{} pixels of area, over the condition budget of {} tokens; send fewer or smaller images or a shorter prompt (image.condition_budget_tokens, image.condition_floor_px)",
            self.tokens, self.floor, self.floor, self.budget
        )
    }
}

/// The sizes `sources` (each condition image's own size, in order) are
/// encoded at, `tokens` counting a prefix of the prompt with images of those
/// sizes and `budget` the tokens it may hold: each at 1024² of area when they
/// all fit, else all at the largest common area, at least `floor`²
/// (`image.condition_floor_px`), that fits. The tokens only grow with the
/// area — each side is a rounded multiple of its square root — so the
/// largest fitting area is a bisection over the square's side.
pub fn fit(tokens: impl Fn(&[ImageSize]) -> u64, sources: &[ImageSize], budget: u64, floor: u32) -> Result<Fitted, OverBudget> {
    let sizes_at = |side: u32| -> Vec<ImageSize> {
        let area = f64::from(side) * f64::from(side);
        sources.iter().map(|source| condition_size(area, f64::from(source.width) / f64::from(source.height))).collect()
    };
    let full = sizes_at(CONDITION_SIDE);
    if tokens(&full) <= budget {
        return Ok(Fitted { sizes: full, scaled: false });
    }
    let floor = floor.min(CONDITION_SIDE);
    let at_floor = tokens(&sizes_at(floor));
    if at_floor > budget {
        return Err(OverBudget { tokens: at_floor, budget, floor });
    }
    // `fits` fits and `over` does not.
    let (mut fits, mut over) = (floor, CONDITION_SIDE);
    while over - fits > 1 {
        let side = fits + (over - fits) / 2;
        if tokens(&sizes_at(side)) <= budget {
            fits = side;
        } else {
            over = side;
        }
    }
    Ok(Fitted { sizes: sizes_at(fits), scaled: true })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fake runner's count: `prompt` tokens and the images' cells.
    fn with_prompt(prompt: u64) -> impl Fn(&[ImageSize]) -> u64 {
        move |sizes| prompt + sizes.iter().map(|size| size.latent_cells() as u64).sum::<u64>()
    }

    /// diffusers' numbers, from Python: `calculate_dimensions(1024 * 1024, r)`
    /// for 1:1, 16:9, 9:16, 4:3 and a 2:1000 sliver, and the halves Python
    /// rounds to even.
    #[test]
    fn the_condition_size_is_diffusers_calculate_dimensions() {
        let area = 1024.0 * 1024.0;
        assert_eq!(condition_size(area, 1.0), ImageSize::new(1024, 1024));
        assert_eq!(condition_size(area, 16.0 / 9.0), ImageSize::new(1376, 768));
        assert_eq!(condition_size(area, 9.0 / 16.0), ImageSize::new(768, 1376));
        assert_eq!(condition_size(area, 4.0 / 3.0), ImageSize::new(1184, 896));
        assert_eq!(condition_size(area, 2.0 / 1000.0), ImageSize::new(32, 22912));
        // 80 / 32 = 2.5 rounds to 2, 112 / 32 = 3.5 to 4.
        assert_eq!(condition_size(80.0 * 80.0, 1.0), ImageSize::new(64, 64));
        assert_eq!(condition_size(112.0 * 112.0, 1.0), ImageSize::new(128, 128));
        // A side diffusers would make 0.
        assert_eq!(condition_size(area, 1.0 / 100_000.0).width, 32);
    }

    #[test]
    fn the_budget_is_the_plans_unless_the_config_lowers_it() {
        let plan = ImagePlan { condition_budget_tokens: 10_000, ..ignis_core::image::fake_plan() };
        assert_eq!(budget(&plan, 0), 10_000);
        assert_eq!(budget(&plan, 4_000), 4_000);
        assert_eq!(budget(&plan, 50_000), 10_000, "a value can only lower it");
    }

    /// Within the budget every image is at 1024² of area; over it, all of
    /// them are scaled by one factor to the largest area that fits, and
    /// one more pixel of side would not; below the floor it is refused with
    /// the numbers.
    #[test]
    fn over_the_budget_every_image_is_scaled_by_one_factor_down_to_the_floor() {
        let sources = [ImageSize::new(1920, 1080), ImageSize::new(600, 800), ImageSize::new(500, 500)];
        // 86 x 48, 56 x 74 and 64 x 64 latent cells.
        let exactly = 20 + 4128 + 4144 + 4096;
        let full = fit(with_prompt(20), &sources, exactly, 512).unwrap();
        assert_eq!(full, Fitted { sizes: vec![ImageSize::new(1376, 768), ImageSize::new(896, 1184), ImageSize::new(1024, 1024)], scaled: false });
        assert!(fit(with_prompt(20), &sources, exactly - 1, 512).unwrap().scaled, "one token short");
        let tokens = |sizes: &[ImageSize]| 20 + sizes.iter().map(|s| s.latent_cells() as u64).sum::<u64>();

        let budget = 2 * 4096;
        let scaled = fit(with_prompt(20), &sources, budget, 512).unwrap();
        assert!(scaled.scaled);
        assert!(tokens(&scaled.sizes) <= budget);
        // One common area: each image is its own aspect ratio at it.
        let side = (1..=1024u32)
            .rev()
            .find(|&side| scaled.sizes[2] == condition_size(f64::from(side * side), 1.0) && scaled.sizes[0] == condition_size(f64::from(side * side), 1920.0 / 1080.0))
            .expect("one side gives all three sizes");
        assert!(side >= 512 && side < 1024);
        let bigger = |side: u32| sources.iter().map(|s| condition_size(f64::from(side * side), f64::from(s.width) / f64::from(s.height))).collect::<Vec<_>>();
        let next = (side + 1..=1024).find(|&next| bigger(next) != scaled.sizes).unwrap();
        assert!(tokens(&bigger(next)) > budget, "the next larger sizes would not fit");

        // At the floor, 512 x 512 pixels of area each: 672 x 384, 448 x 576
        // and 512 x 512, as Python computes them.
        let at_floor = bigger(512);
        assert_eq!(at_floor, [ImageSize::new(672, 384), ImageSize::new(448, 576), ImageSize::new(512, 512)]);
        assert_eq!(fit(with_prompt(20), &sources, tokens(&at_floor), 512), Ok(Fitted { sizes: at_floor.clone(), scaled: true }), "the floor itself fits");
        let refused = fit(with_prompt(20), &sources, tokens(&at_floor) - 1, 512).unwrap_err();
        assert_eq!(refused, OverBudget { tokens: tokens(&at_floor), budget: tokens(&at_floor) - 1, floor: 512 });
        assert!(refused.to_string().contains("condition budget of 3059 tokens"), "{refused}");
    }

    /// No condition image (a generation): the prompt alone against the
    /// budget.
    #[test]
    fn a_generation_fits_its_prompt_alone() {
        assert_eq!(fit(with_prompt(30), &[], 100, 512), Ok(Fitted { sizes: Vec::new(), scaled: false }));
        assert_eq!(fit(with_prompt(130), &[], 100, 512), Err(OverBudget { tokens: 130, budget: 100, floor: 512 }));
    }
}
