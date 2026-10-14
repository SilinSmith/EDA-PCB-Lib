# Unified IC and non-IC pin recognition

This integration combines IC v53 and non-IC R21 in the existing
`stage1` → `stage2` → `fusion` pipeline. Both source checkouts were backed up;
the IC checkout was read only throughout the merge. Weights are unchanged.

## Recognition behavior

ICs retain measured color contacts, monochrome frame/stroke evidence, complete
and vertical text crops, literal designators, validated alphanumeric pin
identifiers, and evidence-based duplicate selection. Unread IC numbers are
omitted by default rather than invented from position. The optional name
library contains generic names and suffix grammars and only corrects supported
confusions with the required observed readings and chip context. It contains
no image IDs, chip-specific answers or pin-order tables.
See [the name library guide](../stage2/README.pin-name-library.md).

Non-IC components retain measured color terminals and connection evidence:
actual pin/wire boundaries, directly attached junction centers, short wires,
mixed-color boundaries, supported symmetry and same-image length references.
RGB LED packages remain one four-terminal component with separate external
channels. Gray non-IC count recovery retains existing position estimates; it
does not claim precise black-and-white pin/wire boundaries.

Non-IC designator re-reading remains separate. Complete source crops,
consistent readings, literal prefixes and unambiguous ownership are required.
Small blue capacitor captions use fixed retry views and the existing acceptance
threshold; conflicting readings are not repaired by guessing a number.

Stage one runs IC recovery before non-IC refinement and constructs network
indices after both steps. Stage two retains both designator audit streams after
reassignment, then applies the IC name library without changing selected
physical pins or their network indices.

## Running

Use the existing entry point, for example:

```powershell
python -m submission.run -input INPUT_DIRECTORY -output OUTPUT_DIRECTORY -device cpu -imgsz 1600 -mode off -vlm off -fusion on
```

The optional local PP-OCR dependency is pinned in
[`requirements.pin-name-ocr.txt`](../requirements.pin-name-ocr.txt). The adapter
verifies package version and model hash; inference does not download models.
`auto` keeps the existing path when the provider is unavailable; explicit `on`
reports a missing or invalid provider.

| Setting | Default | Scope |
|---|---|---|
| `STAGE1_MONO_FRAME_PINS` | `1` | IC monochrome frame recovery |
| `STAGE1_NON_IC_PINS` | `1` | Non-IC terminals, RGB grouping and gray counts |
| `STAGE2_DESIGNATOR_OCR` | follows `STAGE2_PIN_NAME_OCR` | IC caption re-reading |
| `STAGE2_NON_IC_CAPTION_OCR` | `auto` | Non-IC designators |
| `STAGE2_PIN_NAME_LIBRARY` | `1` | Supported IC name corrections |
| `STAGE2_BOX_PIN_DROP_UNREAD` | `all` | Omit unobserved IC identifiers |

## Validation and limits

Validation completed on the merged source:

- 948 unit tests passed.
- Fresh image-only inference on all ten official 10GT drawings preserved the
  R21 non-IC component and pin fields. Identity, type and count agreement remains
  229/233; count agreement after physical component matching remains 233/233.
- IC replay on frozen stage-one inputs preserved all component fields, pin
  fields (including points) and indices for 20 color, 10 official and 17
  monochrome drawings. Color replay reused the v53 initial CRNN cache with
  verified box equality and reran supplementary OCR. Official and monochrome
  replay reran both original and supplementary OCR.
- Fresh detector and assembly checks on three representative color drawings
  preserved 270 measured non-IC points and all four RGB external channels.
- Source hashes match the tested version; the IC checkout, official labels
  and weights remain unchanged.

See [the compact validation record](metrics/pin-recognition-integration-20261014.json).
These are field/regression checks, not independently measured coordinate
accuracy or a contest total.

The user supplied new evaluation rules: ordinary net members align at component
level, ICs at pin level, and power/ground by type, value and geometry rather
than automatic keys. The existing local scoring implementation predates these
changes. Historical network scores are not current official scores.

Known output gaps remain: power/ground `value` fields are empty, and eight gray
power/ground references in the first 10GT image are omitted by the existing
evidence gate. Non-IC LED identity/type errors and missed or ambiguous IC
readings remain. References, labels and weights were not changed to hide them.
