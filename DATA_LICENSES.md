# Data licences

| Data | Licence |
|---|---|
| Machine responses, scored (`data/responses_free25*.jsonl`), results (`results/`, `camera_ready_checks/results/`), generated tables and figures | Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ |
| Item index (`data/items25_index.json`): ICAR item identifiers, stem length and answer type, no item text | CC BY 4.0 |
| SAPA ICAR release (`data/sapaICARData18aug2010thru20may2013.csv`) | CC0 1.0, Condon & Revelle, Harvard Dataverse, doi:10.7910/DVN/AD9RVY |

The code is under the MIT licence in `LICENSE`.

## Not included

**ICAR item text and answer keys.** The items were administered from the PsychArchives release of the
ICAR material (Doebler, Revelle, Condon et al., doi:10.23668/psycharchives.22167). That release is
under a Scientific Use Licence, chosen to preserve the integrity of the item material, including the
scoring keys. This repository therefore contains no item stems, options or keys from it. The public
response files carry scored outcomes only: no model output text, option orders or key positions.
Researchers can obtain the items from PsychArchives under its licence; `data/README.md` gives the
file format the administration and audit scripts expect.

**The `psychTools` response matrices** (ICAR-16, BFI, SPI) used in the August 2026 phase are GPL and are
not redistributed; `scripts/prep_data.py` rebuilds them from the package's data files.

No scoring keys are included either, not even the 16 ICAR sample-test keys published with `psychTools`;
`scripts/verify_items.py` reads those from a file you supply (`ICAR16_KEYS`, see `data/README.md`).
