# Orbital Data Source and Sampling Notes

Updated: 2026-10-10

The `data/orbital_data.csv` file contains 961 orbital-object records: 500 debris records and 461 records from active satellite groups. GP/TLE data were obtained from files mirrored by the public `satvisorcom/satvisor-data` repository, rather than downloaded directly from CelesTrak during this update.

## Debris source files
- Fengyun-1C debris: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/fengyun-1c-debris.tle
- Iridium 33 debris: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/iridium-33-debris.tle
- Cosmos 2251 debris: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/cosmos-2251-debris.tle

## Satellite source files
- Iridium NEXT: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/iridium-NEXT.tle
- Planet: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/planet.tle
- CubeSats: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/cubesat.tle
- Space stations: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/stations.tle
- Globalstar: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/globalstar.tle
- ORBCOMM: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/orbcomm.tle
- Spire: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/spire.tle
- Weather satellites: https://github.com/satvisorcom/satvisor-data/blob/master/celestrak/tle/weather.tle
- Original CelesTrak catalog: https://celestrak.org/NORAD/elements/index.php?FORMAT=tle

## Sample composition

| Group | Records |
|---|---:|
| Fengyun-1C debris | 200 |
| Iridium 33 debris | 100 |
| Cosmos 2251 debris | 200 |
| Iridium NEXT satellites | 80 |
| Planet satellites | 117 |
| CubeSats | 82 |
| Space stations | 18 |
| Globalstar satellites | 28 |
| ORBCOMM satellites | 14 |
| Spire satellites | 72 |
| Weather satellites | 50 |
| **Total** | **961** |

Satellite records were filtered using mean motion greater than 11.25 revolutions/day as a broad LEO screening criterion, then combined with the existing debris rows. This is a screening approximation; mean motion alone does not guarantee that every orbit remains entirely below 2,000 km. The dataset uses the existing 17-column application CSV schema. NORAD catalog IDs were checked against the existing rows to avoid duplicate IDs.

## Limitations
- This is a selected dataset, not the complete active-satellite or debris catalog.
- The source files are public mirrors and may lag the live catalog. Satellite TLE epochs in the added rows range from 2026-09-29T06:18:58.791 to 2026-10-09T23:33:49.876; refresh from a current source before formal analysis and record the retrieval date.
- TLEs are approximate and time-sensitive. Confirm epoch freshness and propagate the elements with a suitable model before making operational claims.
- Adding satellites allows satellite–satellite and satellite–debris candidate pairs to be considered only if the app's screening logic processes all object categories. The dataset update alone does not prove the application or collision-screening results have been tested.
