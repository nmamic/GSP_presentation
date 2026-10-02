# GSP colloquium presentation — context for Claude Code

## What this folder is
LaTeX Beamer slides for my guest-student-programme (GSP) colloquium talk at JSC
(Forschungszentrum Jülich). Main file: `main.tex`. Compile with `latexmk -pdf main.tex`
(pdfLaTeX). The FZJ Beamer theme is installed globally in my texmf tree — do NOT copy
theme `.sty` files, `fzj.pdf` or `placeholder` into this folder.

## Talk constraints (from the organisers)
- 20 min talk + 5 min discussion, mixed audience (students, JSC staff), streamed/recorded.
- 33/33/33 rule: 1/3 everybody understands (what/why), 1/3 how we solve it,
  1/3 deep dive ("look at this crazy thing"), last 1% free.
- 17 content frames (title slide and \makesection divider slides NOT counted),
  ~70 s per frame. Overlay builds inside one frame are fine and do not count extra.
  Anything beyond 17 goes to backup slides after "Questions?".

## Titles (final — do not change without asking)
- `\title`: "Towards a Foundational Model for Particle-Based Methods"
- `\subtitle`: "Scaling neural operators for large-scale plasma simulations and coupling to
  production codes"

## The current state of the project (this is what the talk reports — NOT the Go20 talk)
The Go20 slides (May 2026) are background only and partly outdated. In particular:
- Go20's "cheap surrogate" motivation is outdated: we now know one model inference step is
  MORE expensive than one PIF step, by construction (the model applies the NUFFT-type
  particle↔Fourier transform 2L times per step for d_v channels; PIF only a few times).
  My supervisor confirmed this was expected.
- The model's possible advantages are open HYPOTHESES (implicit / larger time steps via
  autograd Jacobians, differentiability, field operators that are expensive or unknown,
  interfacing with any particle code). Never present them as results.
Facts that ARE established:
- Distributed training (DDP × particle decomposition) was built by Chelsea John; I made its
  data loading shard-aware (each rank reads only its particles).
- Parallel (sharded) inference in `picND` works (particle decomposition, all_reduce of
  Fourier modes, O(d_v·m) payload independent of N).
- Initial particles: each rank draws its own particles with seed 152 + 100·rank (fix for the
  rising noise floor). The run is NOT meant to be identical to the 1-GPU run, and that does
  not matter — do not add any reproducibility caveat.
- Global-bounds bug found and fixed.
- IPPL ↔ PyTorch coupling WORKS and gives CORRECT physics: pybind11 + DLPack, zero-copy,
  same MPI rank hosts IPPL and the model, Python writes E directly into IPPL's memory,
  no changes to IPPL, model loaded by name at runtime.
- Inference scaling through the IPPL interop layer: 1–512 GPUs, up to 1.6 B particles,
  400 time steps.
- IPPL underlies the production accelerator code OPAL-X (Mayani et al., PASC '26, intro) —
  this justifies "production codes" in the subtitle.
- Nothing is blocked. Do NOT write "blocked" anywhere.
- ADIOS2/openPMD streaming: started. ParaView Catalyst: not done (possible future route:
  VTK can import DLPack tensors without copying).

## Hard rules
- NEVER invent numbers, plots or results. Only values from files in this folder or values I
  give explicitly. Old plotting scripts contained PLACEHOLDER timings
  (443/230/132/75 ms, 110–125 ms) — never use them.
- Never use the phrase "cheap surrogate" and never claim the model is faster than PIC/PIF.
- Do not claim I built the distributed training.
- Do not claim ParaView Catalyst / in situ visualisation works.
- Mark design properties that were not tested as such (e.g. "any Kokkos code").
- Go20 figures only with Sriram's permission, credited ("Muralikrishnan, Go20 2026").
  Figures from papers: redraw and cite ("adapted from").
- Anything uncertain or marked TODO(ask me): ask me, do not guess.

## Key sources (for citations on slides)
- S. Muralikrishnan, Go20 2026 slides (`refs/` if present) — background, terminology.
- Mayani, Fischill, Muralikrishnan, Adelmann, PASC '26, arXiv:2605.05469
  (PIC vs PIF in IPPL; OPAL-X builds on IPPL).
- Muralikrishnan et al., SIAM PP 2024, arXiv:2205.11052 (Alpine mini-apps).
- Mitchell et al., J. Comput. Phys. 396 (2019) (original PIF).
- Herde et al., Poseidon, NeurIPS 2024, arXiv:2405.19101 (pretrained on a SMALL set of
  fluid PDEs: compressible Euler + incompressible Navier–Stokes; generalises to unseen PDEs).
- Grid heating / Debye length: arXiv:2503.05123; Barnes & Chacón arXiv:1910.10833.
- Particle noise ∝ n^(-1/2): arXiv:2506.11320.
- DLPack: https://dmlc.github.io/dlpack/latest/ ; VTK DLPack support:
  https://docs.vtk.org/en/latest/api/python/vtkmodules/vtkmodules.util.dlpack_support.html

## Theme pitfalls (FZJ Beamer theme)
- `\maketitle` and `\makesection` must be OUTSIDE frames (they build their own frame).
- Long subtitle: `\setbeamerfont*{subtitle}{parent=subtitle long}` (9 pt) after `\usetheme`.
- Title page uses the image variant with `\titlegraphic{...placeholder}`;
  `\fzjset{section page=text}`.
- Frame titles are all caps by default.
- FZJ colours (RGB): fzjblue (2,61,107), fzjlightblue (173,189,227), fzjred (235,95,115),
  fzjgreen (185,210,95), fzjyellow (250,235,90), fzjviolet (175,130,185),
  fzjorange (250,180,90). Use these in matplotlib figures (divide by 255).
- Colour code across ALL slides: fzjblue = simulation side (IPPL/PIC/PIF),
  fzjorange = ML side (PyTorch/model), fzjred = problems/bugs only.
- Figures: vector PDF from matplotlib in `figures/`. Simple diagrams in TikZ.
- Nothing may overflow its box or the slide; check every page of the PDF.

## How I want to work
- Explain what you change and why; be rigorous, no hand-waving.
- Push back if something I ask for is illogical or unsupported by the data.
- Prefer showing me how to do things over dumping large blocks of code, unless I
  explicitly ask you to write it.
- After edits to `main.tex`, compile and report the first real error from the log, if any.