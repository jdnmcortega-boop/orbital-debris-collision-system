# Orbital Data Source and Sampling Notes

Updated: 2026-10-10

The `data/orbital_data.csv` file contains 500 debris-object records parsed from GP/TLE group files mirrored from CelesTrak by the public `satvisorcom/satvisor-data` repository.

Source files:
- Fengyun-1C debris: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/fengyun-1c-debris.tle
- Iridium 33 debris: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/iridium-33-debris.tle
- Cosmos 2251 debris: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/cosmos-2251-debris.tle
- Original CelesTrak catalog: https://celestrak.org/NORAD/elements/index.php?FORMAT=tle

## Sample composition
This is a stratified 500-object sample, not the complete debris catalog:
- Fengyun-1C debris: 200 records
- Iridium 33 debris: 100 records
- Cosmos 2251 debris: 200 records

The sample was selected at evenly spaced catalog-number positions within each source group to provide broader coverage. TLE epochs and orbital elements were parsed into the existing application CSV schema. Dataset validation confirms 500 unique NORAD catalog IDs. Epoch range in the CSV: 2026-09-15T10:38:42.154 to 2026-10-09T07:46:51.376.

## Limitations
- These are catalogued debris objects with available GP/TLE elements, not every physical fragment.
- The sample is intended for software testing and conjunction-screening research; it does not guarantee every object meets a particular altitude cutoff at every epoch.
- TLEs are approximate and time-sensitive. Refresh from a current source before formal analysis and record the retrieval date.
