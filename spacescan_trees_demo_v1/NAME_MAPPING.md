# Species display names / 樹種雙語名稱

Updated 2026-09-10. All 148 deployed teacher labels have a Latin scientific name and a Traditional Chinese botanical name. `species_names.json` is bundled locally; displaying names does not require a network connection.

Only the SpaceScan Trees preview presentation changes. The raw `species` classifier key, label ordering, model weights and probability scores remain unchanged. API / JSON exports additionally contain `scientific_name`, `chinese_name` and the combined `name` field. Cards, comparison headings and reference image accessible labels consume these same fields.

Most names are drawn from the public Hong Kong Herbarium species index (existing project cache dated 2026-09-07; missing initial letters checked on 2026-09-10). Each entry records a detail-page URL. Supplementary sources are recorded for Areca catechu, Tabebuia impetiginosa and Wodyetia bifurcata. These are nomenclature sources, not evidence for any individual image identification.

- At the user's request on 2026-09-10, the existing Ficus microcarpa / Ficus benjamina group displays as Ficus microcarpa / 榕樹（細葉榕） across results, references and exported display fields. The internal merged classifier key, score aggregation and historical labels are unchanged. This display alias does not establish species-level discrimination from F. benjamina. Original reference taxon attribution remains intact.
- The existing combined Araucaria class remains 南洋杉. No taxonomic split or training relabel is performed.
- Eucalyptus citriodora / E. torelliana use the Chinese names of Corymbia citriodora / C. torelliana while retaining model keys.
- Macaranga tanarius and Melaleuca cajuputi use the existing project's named infraspecific source records; this is presentation mapping, not a new equivalence decision for training samples.
- Bauhinia x blakeana (洋紫荊), B. purpurea (紅花羊蹄甲) and B. variegata (宮粉羊蹄甲) remain distinct outputs. Historical human-label corrections are not changed by this mapping.

Validation: `python -m unittest spacescan_trees_demo_v1.test_preview -v` includes full 148-class coverage, grouped-label preservation, distinct Bauhinia names, reference names and score / raw-key invariance.
