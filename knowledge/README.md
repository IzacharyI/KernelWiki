# Kernel knowledge

- [Operators](operator/README.md): implementations, performance evidence, numerical contracts, tuning and lessons.
- [General knowledge](general/README.md): profiling, optimization, backend programming, hardware and reusable recipes.
- [Kernel source collection](../kernels/README.md): complete implementations with entry points, dependencies, tests and benchmarks.

## Reading a backend page

`<backend>_aiter.md` describes the pinned AITER source snapshot with entry points, parameter names, configuration references and related tests. Symbol lists include support functions; they are not counts of independently launchable kernels. ASM code objects are grouped by family, and may include compiled DSL output.

`<backend>_notes.md` explains selected implementation decisions in detail. `<backend>.md` preserves older GEAK material, including dated measurements and upstream reports. Read the source date before applying a historical recommendation.

## Performance evidence

Keep independently measured results, upstream-reported results and source-only observations distinct. A source change or a tuned configuration does not itself establish a speedup. Every new performance result should name hardware, software/source revisions, workload, layouts/dtypes, timing protocol, baseline, correctness check and evidence location. Rank implementations only for compatible cases.

## Contributing knowledge

Put operator-specific implementation knowledge under `operator/<op>/backend/`; put transferable run-derived lessons under that operator's `learned_knowledge/`. Put methods that apply across operators under `general/`. Add a local navigation link and cite the relevant source or measurement. AITER implementation links should include a commit.

Store reusable code as a complete source bundle under `kernels/<operator>/<backend>/<implementation>/`. Link its implementation ID from the corresponding knowledge page. Include the exact source digest with measured results; historical patch references remain historical evidence, not the current storage format. All repository documentation and comments use English.

The first AITER source pass prioritizes files changed from 2026-07-21 through 2026-10-21. Older GEAK knowledge remains available. There is no scheduled update process or automatic best-kernel promotion.
