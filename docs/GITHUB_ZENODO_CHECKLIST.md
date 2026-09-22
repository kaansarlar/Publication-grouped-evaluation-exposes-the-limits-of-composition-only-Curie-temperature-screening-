# GitHub and Zenodo release checklist

1. Create a public GitHub repository named `Tc-targeting-validation`.
2. Upload the **contents** of this directory so that `README.md` is at the repository root.
3. Confirm that no source-article PDFs, manuscript drafts, or personal reference-manager files were added.
4. Run `python scripts/run_pipeline.py --mode verify`; commit the passing `docs/VERIFICATION_REPORT.json`.
5. Run `python scripts/generate_checksums.py`, followed by `sha256sum --check docs/CHECKSUMS.sha256`.
6. Review the split MIT/CC BY 4.0 terms in `LICENSE.md`. Change them before release if a different licensing choice is required.
7. Create a GitHub release tagged `v1.0.0`.
8. Connect the repository to Zenodo and archive the `v1.0.0` release.
9. Add the assigned Zenodo DOI to `CITATION.cff`, the README, and the manuscript's Data and Code Availability statements.
10. If the manuscript changes after peer review, create a new tagged release rather than replacing `v1.0.0` silently.

Do not upload the ZIP file inside the GitHub repository. Upload the extracted files and folders; attach the ZIP only to a release if desired.
