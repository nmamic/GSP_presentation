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
- 16–17 content frames (title slide and \makesection divider slides NOT counted),
  i.e. ~70 s per frame.

## Titles (final — do not change without asking)
- `\title`: "Towards a Foundational Model for Particle-Based Methods"
- `\subtitle`: "Scaling Neural Operators for Large-Scale Plasma Simulations and Coupling to
  Production Codes"

## Slide plan (17 content frames)
As built in main.tex (revision 2, 2026-10-01): 19 frames. The plan below is historical; the
current order is: PIC loop · From grids to Fourier modes (old 2+3 merged) · Why surrogates ·
DL for PDEs · NEOPIC · NUFNO model | Where the project stood (old 7+8 merged, with "next steps"
cards) · Training I/O · Parallel inference · What is IPPL? (new) · Coupling to IPPL · Scaling
results (moved after coupling) | Two ways to split the particles · Bounds bug (fixed early on) ·
The rising noise floor (new) · DLPack zero-copy · Why DLPack matters | Status · Acknowledgements.
Schematic figures: figures/make_figures.py (synthetic particles, illustrations only). Real
figures go in figures/ under the names used by the \figorph placeholders in main.tex.
Copy as little as possible from GO20.
### Section 1 — Introduction (everyone understands), frames 1–6
1. Particle–field systems and the PIC loop: init → scatter to mesh → solve on mesh →
   gather → push (draw it like GO20 slide 3).
2. Particle-in-Fourier (PIF): scatter/gather directly in Fourier space (NUFFT), no real-space
   grid; no aliasing, spectral accuracy; cost O(Np·Nm) direct, NUFFT → same order as PIC but
   higher constant (GO20 slide 5 + history slide).
3. Shortfalls of grid-based PIC: aliasing → grid heating / loss of energy conservation;
   cells must resolve the Debye length; statistical noise ~ Nppc^(-1/2).
   Visual: 2D cyclotron PIC vs PIF (GO20 slides 14–15, only with Sriram's permission, credited).
4. Why surrogates: high fidelity needs ~1e11 grid points/modes, ~1e12 particles, ~1e5 steps;
   many-query scenarios out of reach even at exascale (GO20 slide 22).
   NOTE: PIC/PIF DO scale well (GO20 slide 18) — the problem is cost, not scaling. Never
   claim "PIC does not scale".
5. Deep learning for PDEs so far, e.g. Poseidon (Herde et al., NeurIPS 2024): pretrained on
   fluid dynamics (compressible Euler, incompressible Navier–Stokes) data on regular grids,
   vision-transformer architecture. Points to make: grid/image-based input, not particle data;
   not coupled to production simulation codes. Do NOT say Poseidon "is not general" — its
   paper explicitly shows generalisation to unseen PDEs.
6. Our approach (NEOPIC): a neural operator approximates the whole map
   positions/velocities → fields (scatter + solve + gather); the particle push stays classical
   (GO20 slide 23). Model: nonuniform FNO (NUFNO) — FNO whose Fourier transforms are NUDFTs;
   each Fourier block ≈ a learnable PIF scheme in a lifted latent space (GO20 slide 24).
   Data: mini-apps (Landau, two-stream, bump-on-tail, cyclotron), 600 steps, 500k particles;
   input particle positions, output E at particle positions (GO20 slide 25).
   Goals listed by Sriram (not shown by me): not limited by Debye length/CFL,
   resolution invariance, interfaces with any particle code — present as GOALS, not results.

### Section 2 — The project (how), frames 7–12
7. State when I arrived: NUFNO model; DDP + particle-decomposition ("tensor parallel")
   training implemented by Chelsea John; `picND` evaluation against a PIF reference;
   training data from mini-apps.
8. What was missing: (a) inference not parallel — all particles on one GPU;
   (b) redundant reads — training read all particles from HDF5 on every rank, inference
   loaded checkpoint/HDF5 on every rank; (c) no coupling to a simulation code.
9. Training I/O fix: shard particles at HDF5 read time; read volume down, peak memory
   unchanged (set downstream by activations/Fourier layers). Side note on the same frame:
   local batch × shard size b·n = B·N/W is independent of t_p, so more GPUs at fixed W did
   not lower peak memory (22222 MiB measured at tp=2 and tp=8).
10. Parallel inference: particle decomposition, all_reduce of Fourier modes
    (payload O(dv·m), independent of particle count).
11. Scaling results: strong + weak scaling from REAL logs only. Done THROUGH the IPPL
    interop layer: NUFFT model, 400 time steps, 1–512 GPUs, up to 1.6B particles
    (refs/presentation_gsp_IPPL_python_interop.pdf p.13–15).
12. Coupling to IPPL: same MPI rank hosts IPPL (Kokkos View on GPU) and embedded
    Python/PyTorch; pybind11 + DLPack, zero-copy where layout allows. Works end to end:
    Python writes E directly into IPPL memory. Opt-in, header-only, no changes to IPPL.
    Details: refs/presentation_gsp_IPPL_python_interop.pdf (my own deck for the IPPL team).

### Section 3 — Deep dive, frames 13–15
13. Particle decomposition vs domain decomposition (Sriram's terms, GO20 slides 16–17) and
    the bug it exposes: the model was trained/evaluated with particle decomposition (each
    rank: random subset over the whole domain); IPPL uses domain decomposition (each rank:
    one subdomain). Inference `__call__` does not all-reduce global position min/max
    (valid() does) → harmless under particle decomposition (local ≈ global bounds), fatal
    under domain decomposition (each rank rescales its subdomain to the full [0, 2π), ranks
    sum Fourier coefficients in incompatible coordinates). Also: normalisation divides by
    max instead of (max − min). Visual: 1D domain, particles coloured by rank, both cases.
14. Zero-copy Kokkos View → PyTorch via DLPack ("how it works"):
    - DLPack = minimal C struct describing a tensor: data pointer, device, dtype, shape,
      strides, plus a deleter for lifetime management. No copy, no allocator.
    - Peel the types: ParticleAttrib<Vector<double,3>> → Kokkos::View<Vector<double,3>*>
      → shape (N, 3) of double.
    - Zero-copy only if the element type is standard layout with no padding
      (sizeof(Vector<T,D>) == D·sizeof(T)) → reinterpret the pointer; otherwise repack into
      a flat buffer ON THE DEVICE (still no host transfer).
    - Lifetime: the capsule keeps the Kokkos View alive until PyTorch releases it.
    - Teardown order: release Python objects/model before Py_Finalize, Kokkos::finalize,
      MPI_Finalize.
    - Same MPI rank hosts both sides → rank i of the simulation talks to rank i of the model.
    TODO(ask me): status of the return path (PyTorch field → Kokkos View) and how
    GPU stream/execution-space synchronisation is handled — the audience may ask.
15. Why DLPack matters beyond this project (interoperability):
    - Open standard adopted by NumPy, CuPy, PyTorch, TensorFlow, JAX, MXNet, TVM, Paddle,
      mpi4py (per the DLPack docs).
    - Cross-hardware (CPU, CUDA, ROCm, …) and a plain C ABI → usable from any language.
    - Consequence: the C++ export side is generic over Kokkos Views, so any Kokkos-based
      simulation code can hand its data to any DLPack-aware ML framework; the model side is
      not tied to PyTorch. State this as a design property, not as something tested with
      other codes/frameworks.

### Section 4 — Status, frames 16–17
16. Status and outlook: works / in progress / blocked (large jobs on JUWELS Booster
    during software updates) / next steps.
17. Acknowledgements: Sriramkrishnan Muralikrishnan, Chelsea John (the foundation model is
    her PhD thesis — present my work as contributing to it); NEOPIC funded by Helmholtz AI
    (INF funding code ZT-I-PF-5-242).

## Hard rules
- NEVER invent numbers, plots or results. Only values from files in this folder or values
  I give explicitly. Earlier plotting scripts contained PLACEHOLDER timings
  (443/230/132/75 ms, 110–125 ms) — never use them.
- Do not claim the model is faster than PIC/PIF (not measured).
- Never mention ParaView / Catalyst / VTK. ADIOS2/openPMD streaming only on the outlook slide.
- Noise-floor bug (deep dive): inference initial conditions used cp.random.seed(152) on every
  rank → very similar (not exactly identical) particles on all ranks → noise floor rose with the
  GPU count. Fixed with cp.random.seed(152 + 100*rank). Independent seeds per rank, NOT a
  reproduction of the single-GPU sequence.
- The local-vs-global bounds bug was fixed long before the IPPL scaling runs.
- Do not claim I built the distributed training — I optimised its data loading.
- Figures from Sriram's GO20 slides only with his permission, always credited
  ("Muralikrishnan, Go20 2026"). Figures from papers: redraw and cite ("adapted from").
- If something is uncertain, ask me instead of guessing.

## Key sources (for citations on slides)
- S. Muralikrishnan, "Particle-in-Fourier: A Promising Paradigm for Extreme-Scale Plasma
  Simulations and Beyond", Go20 2026 (slides, if in this folder).
- Mayani, Fischill, Muralikrishnan, Adelmann, PASC '26, arXiv:2605.05469 (PIC vs PIF in IPPL).
- Muralikrishnan et al., SIAM PP 2024, arXiv:2205.11052 (Alpine mini-apps).
- Mitchell et al., J. Comput. Phys. 396 (2019) (original PIF).
- Herde et al., Poseidon, NeurIPS 2024, arXiv:2405.19101.
- Grid heating / noise: arXiv:2503.05123; Barnes & Chacón arXiv:1910.10833.
- DLPack: https://dmlc.github.io/dlpack/latest/ (adoption list, C ABI, cross-hardware).
- VTK DLPack support: https://docs.vtk.org/en/latest/api/python/vtkmodules/vtkmodules.util.dlpack_support.html

## Theme pitfalls (FZJ Beamer theme)
- `\maketitle` and `\makesection` must be OUTSIDE frames (they build their own frame).
- Long subtitle: `\setbeamerfont*{subtitle}{parent=subtitle long}` (9 pt) after `\usetheme`.
- Title page uses the image variant with `\titlegraphic{...placeholder}`. Section pages use the
  same image variant (`\fzjset{section page=image}`, decided 2026-10-01) with a redefined
  template in main.tex that prints the section name in the title font.
- Title slide prints date | author | institute on one line — keep `\institute` short.
- Frame titles are all caps by default.
- FZJ colours (RGB): fzjblue (2,61,107), fzjlightblue (173,189,227), fzjred (235,95,115),
  fzjgreen (185,210,95), fzjyellow (250,235,90), fzjviolet (175,130,185),
  fzjorange (250,180,90). Use these in matplotlib figures (divide by 255).
- Colour code across ALL slides: fzjblue = simulation side (IPPL/PIC/PIF),
  fzjorange = ML side (PyTorch/model), fzjred = problems/bugs only.
- Figures: vector PDF from matplotlib, stored in `figures/`. Simple diagrams in TikZ
  (already loaded by the theme). Say what figures to use, they are not uploaded beforehand.

## How I want to work
- Explain what you change and why; be rigorous, no hand-waving.
- Push back if something I ask for is illogical or unsupported by the data.
- Prefer showing me how to do things over dumping large blocks of code, unless I
  explicitly ask you to write it.
- After edits to `main.tex`, compile and report the first real error from the log, if any.