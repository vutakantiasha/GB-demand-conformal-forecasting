# Research Governance and Data Handling

The pipeline uses public, aggregate electricity-system data from NESO. It does not process participant records, household-level smart-meter data, direct identifiers, special-category personal data or confidential operational credentials.

## Data handling principles

- Store only the data required for the stated forecasting experiment.
- Preserve provenance through source URLs, resource identifiers and checksums.
- Do not commit credentials, private keys, tokens or unrelated personal files.
- Do not redistribute raw NESO data in this repository; acquisition remains tied to the first-party source and its licence.
- Report uncertainty honestly: marginal conformal coverage must not be misrepresented as individual conditional certainty.

Project-specific institutional approval records are intentionally kept outside the public software repository.
