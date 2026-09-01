# Iridium 33 / Cosmos 2251 — 30-Day Historical Replay

## Ground-truth event

- Event ID: `IRIDIUM33_COSMOS2251`
- Iridium 33 NORAD Catalog Number: `24946`
- Cosmos 2251 NORAD Catalog Number: `22675`
- Collision epoch used by this project: `2009-02-10T16:56:00Z`
- Rewind target: `2009-01-11T16:56:00Z`
- Replay window: 30 days before the event through the event epoch

CelesTrak's historical collision analysis reports that SOCRATES predicted the conjunction at the collision epoch in all 14 reports during the week before impact. The predicted minimum range varied substantially, from 117 m to 1.812 km, demonstrating the sensitivity of conjunction prediction to the available TLE data.

## Historical orbital-data requirement

The quantitative replay must use archived GP/TLE element sets whose epochs are historical and whose availability is not later than the prediction timestamp being replayed.

For the first experiment, obtain archived GP data for:

```text
24946
22675
```

with a requested date range covering at least:

```text
Start: 2009-01-11
Stop:  2009-02-10
```

CelesTrak's Special Data Request service provides historical GP data for unclassified objects and supports TLE/3LE, 2LE, XML, KVN, JSON, and CSV output. The request service sends the requested archive by email rather than exposing the historical range as an unauthenticated automated download endpoint.

## Files expected locally

After receiving the archive, place the two historical files here:

```text
iridium33_history.tle
cosmos2251_history.tle
```

The replay engine accepts either 3-line records (`name + line 1 + line 2`) or repeated 2-line TLE pairs.

## Leakage rule

For a replay snapshot at time `t`, the replay engine selects the newest archived element set with an epoch `<= t`.

It must never select an element set whose epoch is later than the snapshot. This prevents future orbital information from leaking into the historical prediction.

## Experiment outputs

After the archives are present, generate:

```text
results/historical/iridium33_cosmos2251_states.csv
results/historical/iridium33_cosmos2251_pair.csv
```

The pair file will contain the timestamp, days before the event, relative distance, relative velocity, and historical-event marker. It will then be passed into the existing conjunction/probability/Monte-Carlo/QAE pipeline.

## External references

- CelesTrak — Iridium 33/Cosmos 2251 collision analysis: https://www.celestrak.org/events/collision/
- CelesTrak — historical GP data request service: https://www.celestrak.org/NORAD/archives/request.php
- CelesTrak — GP data formats: https://www.celestrak.org/NORAD/documentation/gp-data-formats.php
- NASA NTRS — Analysis and Consequences of the Iridium 33/Cosmos 2251 Collision: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20100008433.pdf

## Important

Do not populate this directory with a modern/current TLE and label it historical. If the archive does not contain a record at or before the rewind date, the replay should fail rather than silently use future data.
