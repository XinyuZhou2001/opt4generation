# opt4generation

**Generating Data by Climbing a Fixed Landscape** is an interactive research essay about Gaussian perturbations, diffusion, and noise unconditioning through an optimization perspective.

Read the essay at [opt4generation](https://xinyuzhou2001.github.io/opt4generation/).

The essay moves from high-dimensional Gaussian shells to density landscapes and gradient ascent, then examines diversity, local manifold geometry, non-memorizing outcomes, and guided generation. Its three interactive illustrations let readers explore the underlying mechanisms.

## Research

This account is based primarily on two papers by Xinyu Zhou, Jiawei Zhang, and Stephen J. Wright:

1. [Smoothing the Score Function to Enhance Generalization in Diffusion Models](https://arxiv.org/abs/2601.19285).
2. [An Optimization Framework for Denoising Generation](https://arxiv.org/abs/2611.22202).

The essay also cites foundational work on score matching, denoising score matching, NCSN, DDPM, and score-based SDEs. This repository is not a comprehensive literature survey. Related viewpoints or results may appear elsewhere, sometimes under different terminology; consult the papers' related-work sections and the wider literature.

## Interpreting the illustrations

Analytic densities, scores, and weights are calculated from the stated toy models. Their numerical trajectories illustrate particular settings rather than prove general claims about learned generators. The guided comparison includes prescribed off-band errors and chosen numerical scaling: it is a scenario illustration, not a recording of a trained diffusion network or a matched empirical benchmark. The essay explains these distinctions in its illustration note.

Formal guarantees are those established in the papers, under their stated assumptions. The papers' neural-network experiments are separate from the interactive simulations in this site.

## Read locally

The public site is already built and needs no package installation or build service. Open `index.html` in a recent browser, or serve this directory:

```sh
python -m http.server 8000
```

The reading assets and interactive pages use relative links, so the site also works under a GitHub Pages project subdirectory. Mathematical fonts and interactive scripts are bundled locally.

## Publish with GitHub Pages

In the repository settings, select **Pages → Build and deployment → Source: GitHub Actions**.

The workflow in [`.github/workflows/pages.yml`](.github/workflows/pages.yml) runs on pushes to `main` and can also be started manually. It packages the reviewed static files into `_site/`, uploads that directory with the official Pages artifact action, and deploys it through the official Pages deployment action. No additional application build runs in GitHub Actions.

The packaging script uses an explicit public-file list. Git metadata, workflow files, environment files, dependency directories, archives, and private hosting metadata are excluded from the published artifact. If the default branch is renamed, update the workflow's branch filter accordingly.

## Files and notices

- `index.html`: the complete essay.
- `gaussian-landscape.html`, `local-manifold.html`, `guided-shells.html`: standalone interactive views.
- `generating-by-climbing.md`: the editable essay text.
- `assets/`, `interactive/`, `manifold/`, `guidance/`: browser assets and interactive sources.
- `figure-checks.json`: numerical diagnostics for the stated toy examples.
- `licenses/`: bundled third-party notices, including the KaTeX license.

No additional license for the authors' text, illustrations, or research content is granted by this README. Third-party components retain their respective notices.
