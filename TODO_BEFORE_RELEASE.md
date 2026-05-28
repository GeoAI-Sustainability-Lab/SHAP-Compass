# TODO before tagging the paper-release version

This file lists every placeholder that must be filled in before tagging
the version that accompanies the published paper. Search the repo for
`<...>` to find them all; the table below tells you what each one means
and which file it lives in.

## Placeholders to replace

| Placeholder | Files | What to put |
|---|---|---|
| `<FIRST_AUTHOR_NAME>` | `pyproject.toml` | First author full name (display form). |
| `<FIRST_AUTHOR_EMAIL>` | `pyproject.toml`, `CITATION.cff` | First author contact email. |
| `<FIRST_AUTHOR_FAMILY_NAME>` | `CITATION.cff` | First author family / surname (CFF needs it split). |
| `<FIRST_AUTHOR_GIVEN_NAMES>` | `CITATION.cff` | First author given names. |
| `<FIRST_AUTHOR_ORCID>` | `CITATION.cff` | ORCID id, e.g. `0000-0002-1825-0097`. Leave the URL prefix. |
| `<FIRST_AUTHOR_AFFILIATION>` | `CITATION.cff` | Institution name. |
| `<PAPER_DOI>`, `<VOLUME>`, `<START_PAGE>`, `<END_PAGE>` | `CITATION.cff` (commented out) | Uncomment and fill once the paper is accepted and has a DOI. |

If there are multiple authors, duplicate the `authors:` entry in
`CITATION.cff` and add matching entries under `preferred-citation.authors`.

## One-time setup tasks

These are not placeholders but recommended once before tagging:

- [ ] Bump `version` in both `pyproject.toml` and `CITATION.cff` to the
  paper-release version (e.g. `1.0.0`).
- [ ] Update `date-released` in `CITATION.cff` to the tag date.
- [ ] Create the GitHub repo, push, and confirm the CI workflow
  (`.github/workflows/test.yml`) goes green.
- [ ] (Optional) Mint a Zenodo DOI for the package itself by enabling
  the GitHub-Zenodo integration before tagging `v1.0.0`. Then add the
  Zenodo DOI badge to the top of `README.md`.
- [ ] (Optional) Publish to PyPI (`python -m build && twine upload dist/*`).

## After acceptance

- [ ] Uncomment and fill the `doi`, `volume`, `start`, `end` lines in
  `CITATION.cff` `preferred-citation`.

The README no longer references the paper — it is written as a
standalone open-source package — so no README edit is required after
acceptance. Readers can still cite the paper via the
`preferred-citation` block in `CITATION.cff` (GitHub renders it under
the "Cite this repository" button).
