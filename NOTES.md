# Wildlife Analyzer — Ideas, Experiments, Dead Ends

A running log of things we tried, why they failed or were removed, and ideas for future work.

---

## Consensus Pass — History and Current Approach

**Status: Active** (`CONSENSUS_WINDOW = 5` in config.py)

### Goal

After analysis, group consecutive frames with inconsistent species IDs and re-query the API with all frames together. Intended to fix the same eel being identified as "Sharptail Snake Eel", "Goldspotted Snake Eel", and "Blotched Snake Eel" across 5 consecutive seconds.

### Failed Approaches

1. **Window-based clustering** — group all CSV rows within N seconds of each other, send all frames to API for consensus. Too aggressive: collapsed diverse reef scenes (6 different fish in 6 seconds → single wrong ID).
2. **Genus-overlap heuristic** — only trigger consensus when at least two frames in the cluster share the same genus (first word of scientific name). Correctly caught *Myrichthys breviceps* vs *Myrichthys ocellatus* eel confusion. Failed because common species like Brown Chromis (*Chromis*) and French Grunt (*Haemulon*) appear in nearly every second of reef footage, so their genera naturally overlap across frames showing completely different animals.
3. **Low-confidence trigger** — also trigger consensus when all frames in the cluster are below `CONFIDENCE_THRESHOLD`. Made things worse — even more diverse scenes got collapsed incorrectly.
4. **Tight sub-clustering** — only cluster truly consecutive frames (gap ≤ `SNAPSHOT_INTERVAL + 1` seconds) to avoid merging different animals. Helped but didn't solve the root problem.

**Root problem:** Can't reliably distinguish "same animal, confused model" from "different animals in consecutive seconds" using only CSV data and confidence scores. In dense reef footage, common species like Brown Chromis and French Grunt appear in nearly every second, so their genus naturally overlaps across frames that genuinely show different animals.

**What worked in isolation:** Same-genus consensus correctly resolved eel clusters (*Myrichthys*) and grunt clusters (*Haemulon*) — but the signal was too noisy in full-video runs.

### Current Approach (v2 — working)

Use **common-name keyword overlap** to filter which frames get sent:
1. Find the most-repeated genus in the cluster (the anchor)
2. Extract the dominant word from those frames' common names (e.g. "eel" from "Goldspotted Snake Eel", "Sharptail Snake Eel")
3. Send only frames whose common name contains that word — bystander frames (Frogfish, Sea Cucumber) have different type words and are excluded
4. The consensus prompt names the type word explicitly: *"sequential frames of the same eel"*

**Why it works:** Common names carry semantic type information that scientific names obscure across genera. "Spotted snake eel" (*Ophichthus ophis*) and "Goldspotted Snake Eel" (*Myrichthys ocellatus*) share "eel" even though they're different genera — the keyword correctly groups them while excluding a Frogfish in the same time window.

**Tested results:** Correctly resolved eel clusters (`[05:35–05:39]`, `[05:50–05:52]`, `[08:26–08:28]`) and a Blue Tang / Bluehead Wrasse confusion. Bystanders (Frogfish at 06:17, Sea Cucumber at 08:24) correctly excluded. No diverse reef scenes collapsed.

### Remaining Future Ideas
- Use visual similarity (embedding distance) before triggering — frames that look visually similar are more likely the same subject
- Ask the AI a yes/no "same animal?" question before attempting a consensus ID
- Run consensus as a separate optional CLI mode so the user can review clusters before committing

---

## Extract Parallelism

**Goal:** Speed up frame extraction by parallelizing blur scoring.

**What we did:** Read all frames in a SMART_EXTRACT interval into memory, then score them all in parallel using `ThreadPoolExecutor`. The single pool is created once outside the main loop (important — creating a new pool per interval caused ~600% CPU usage and significant overhead for long videos).

**Result:** Modest improvement (~5-10% faster). Frame reading (sequential, disk-bound) is the dominant bottleneck, not blur scoring. True parallelism would require decoding frames off the main thread, which OpenCV doesn't support cleanly.

---

## Curation Best Practices

### How YouTube chapters deduplication works

The chapters builder (`wildlife_analyzer.py:1432–1440`) tracks `prev_species` — the set of species identified in the immediately preceding frame. A species only gets a new chapter line when it was **absent** from that previous set. Result: 20 consecutive frames of the same eel produce exactly **one** chapter entry (the first). A second entry only appears if the animal disappears from frame and then reappears later.

### Practical implications

- **One clean frame per continuous sighting is enough.** Once you have a high-confidence ID, every additional frame of the same animal in the same run adds zero new chapter entries. Extra frames only inflate analysis time and API cost.
- **Delete file AND CSV row together.** If you remove an image from `good/`, also remove its row from the sighting CSV. If the row stays, the frame is considered processed and won't be re-queued. If the file stays but the row is deleted, `analyze` will re-queue and reprocess it on the next run.
- **Don't delete from `blurry/` or `rejected/` before running `tune`.** Those folders are the training signal — `tune` compares score distributions across all three folders to calibrate thresholds. Deleting from them skews the calibration.

### When extra frames are actually worth keeping

- **Gaps in the sighting** — if the animal swam away and came back, the gap breaks the `prev_species` chain and a second entry will be added. Keep the frame that marks the return.
- **Different angle or behavior** — a head-on shot of an eel after several profile shots may produce a better or higher-confidence ID, and is worth keeping as an alternative.
- **ID uncertainty** — if the first frame's ID is low-confidence and a later frame gets 0.95, keep both; the better one will win after common-name normalization.

---

## Ideas Not Yet Tried

- **`sort-tags` CLI mode:** After manually tagging rejected frames with Finder color tags, run a script to move red-tagged files from `good/` to `blurry/` before running `tune`. Enables manual curation that preserves tune's ability to learn thresholds.
- **Visual similarity clustering:** Use image embeddings (CLIP or similar) to group visually similar frames before consensus — more reliable than genus/timestamp heuristics.
- **Batch API mode:** Send multiple frames in a single Gemini batch call to reduce per-frame overhead and latency.
