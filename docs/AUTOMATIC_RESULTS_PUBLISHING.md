# Automatic research-result generation and GitHub publishing

Every successful click on **Rebuild live results from orbital_data.csv** now generates these files in the app's `results/` directory:

- `live_run_summary.json` — run timestamp, orbital epoch, objects, propagated states, conjunctions, risk counts, and output counts.
- `objective_results_live.json` — evidence and limitations mapped to research Objectives 1–6.
- `hypothesis_results_live.json` — current-run status for H01–H03.
- `research_results_live.md` — readable run report.
- `results_by_research_objectives.md` — the familiar report path, replaced with the newest successful live-run report.

The refresh also regenerates the existing pipeline outputs (such as conjunctions, analytical Pc, predictions, QAE comparison, false-positive analysis, re-entry analysis, and security outputs). Historical benchmark experiments are not silently relabeled as new live trials. Since the live refresh does not run Monte Carlo by default, H02 is marked inconclusive/not tested for that run unless current paired MC data are explicitly available. H01 requires a verified reference-event set; H03 requires a multi-trial statistical benchmark.

## Enable publishing to GitHub

The app can publish the generated files to a separate `live-results` branch. Using a separate branch avoids triggering a redeploy of the Streamlit app on every refresh.

1. Create a GitHub fine-grained personal access token scoped only to this repository, with **Contents: Read and write** permission. Do not paste the token into source code or commit it to this repository.
2. In the deployed Streamlit app, open **App settings → Secrets**.
3. Add the following secrets, replacing the token with your own value:

```toml
ORION_GITHUB_TOKEN = "YOUR_TOKEN_HERE"
ORION_GITHUB_REPOSITORY = "jdnmcortega-boop/orbital-debris-collision-system"
ORION_GITHUB_RESULTS_BRANCH = "live-results"
```

4. Save the secrets and reboot/restart the Streamlit app.
5. Click **Rebuild live results from orbital_data.csv**. After a successful run, the dashboard will report whether publishing succeeded and link to the results branch.

If `live-results` does not exist, the app attempts to create it from `main` on the first publish. If no token is configured, result files are still generated in the app's runtime, but cannot be committed to GitHub. Streamlit's runtime filesystem may be temporary, so configure the token if you need persistent GitHub copies.

## Scientific safeguards

- A successful file write is not proof that the model is accurate.
- The live report distinguishes current-run measurements from fixed benchmark tests and historical CDM comparisons.
- No H01 accuracy decision is made without verified reference data.
- H02 is not statistically tested without paired current-run QAE and Monte Carlo estimates.
- H03 is not statistically retested from a single security run.
- Collision-probability risk labels and response-priority rankings remain separate.
