# Historical collision replay data

This directory stores **historical-event metadata and replay outputs** for the orbital-debris collision backtest.

## Scientific rule

The prediction stage must use only orbital information that was available at or before the replay timestamp. The known collision outcome is **ground truth only** and must not be fed into the risk estimator.

For the planned experiment:

1. Rewind to **30 days before** a confirmed collision.
2. At each snapshot, select the freshest archived GP/TLE element set whose epoch is not later than that snapshot.
3. Confirm separately that the selected element set was published or available by that snapshot; element epoch alone does not prove availability.
4. Propagate the two objects to the snapshot time with SGP4.
5. Calculate relative distance and relative velocity.
6. Run the existing analytic, Monte-Carlo, and QAE probability pipeline.
7. Classify the risk from the calculated probability; do not force a HIGH label.
8. Compare the forecast with the historical collision afterward.

## Historical data acquisition and provenance

The repository does not fabricate historical TLEs. Obtain authentic historical element sets from an authoritative archive and preserve the original source response, query, retrieval time, and publication metadata.

Space-Track documents the GP_HISTORY service for historical orbital element sets and describes CREATION_DATE as the field used to identify when an element set was published to Space-Track. See the [Space-Track API documentation](https://www.space-track.org/documentation). A valid account may be required, and its API usage rules must be followed. CelesTrak also provides current and special-interest GP data, but current element data must not be presented as historical data.

For a GP_HISTORY CSV export containing NORAD_CAT_ID, EPOCH, and CREATION_DATE, convert the source timestamps into the schema required by the provenance gate:

~~~bash
python -m modules.build_tle_availability \
  --input-csv path/to/gp_history.csv \
  --output-csv path/to/record_level_availability.csv
~~~

The resulting CSV has NORAD_CAT_ID, TLE_EPOCH, and AVAILABLE_AT_UTC. It uses the earliest recorded CREATION_DATE for each unique object/epoch. Keep the original source export and verify that every availability row matches the exact element record used in the replay. The converted file alone is not proof of provenance if the source export or query cannot be documented.

## Existing Iridium 33–Cosmos 2251 case

Use NORAD IDs **24946** and **22675** and an archive that reaches at least **2009-01-11 UTC** for the 30-day rewind. The collision occurred at about **2009-02-10 16:56 UTC**. CelesTrak documents historical conjunction reporting for this event, but that does not replace the need to verify when each orbital record became available.

## Second-event candidate: CERISE–Ariane debris

Use NORAD IDs **23606** and **18208**, with records reaching at least **1996-06-24 UTC** and the event at **1996-07-24 09:48 UTC**. The NASA event source and data requirements are documented in [cerise_ariane/README.md](cerise_ariane/README.md). This candidate is **not yet ready for quantitative replay**: its complete orbital history and record-level availability timestamps have not been archived and validated.

## Replay command

After the archived TLE files are available locally, the existing state-replay engine can be run like this:

~~~bash
python -m modules.historical_replay --event-id IRIDIUM33_COSMOS2251 --event-time 2009-02-10T16:56:00Z --norad-a 24946 --norad-b 22675 --tle path/to/iridium33_history.tle path/to/cosmos2251_history.tle --rewind-days 30 --forecast-days 0 --step-hours 24 --output results/historical/iridium33_cosmos2251_states.csv --pair-output results/historical/iridium33_cosmos2251_pair.csv
~~~

The replay engine produces one state row per object per timestamp and a pair-level CSV containing relative distance, relative velocity, and the actual-event flag. The pair-level file is intended to feed the existing probability/risk pipeline without replacing it.

## Event status

Only events with authoritative confirmation and sufficient pre-event orbital data should be used for quantitative backtesting. A catalog entry can remain present while its status is needs_archival_tle or NOT_VERIFIED. Neither a successful propagation nor a close forecast TCA proves operational predictive skill. That requires verified data availability, an independent baseline, and multiple events.
