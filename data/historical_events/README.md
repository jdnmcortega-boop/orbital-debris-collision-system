# Historical collision replay data

This directory stores **historical-event metadata and replay outputs** for the orbital-debris collision backtest.

## Scientific rule

The prediction stage must use only orbital information that was available at or before the replay timestamp. The known collision outcome is **ground truth only** and must not be fed into the risk estimator.

For the planned experiment:

1. Rewind to **30 days before** a confirmed collision.
2. At each snapshot, select the freshest archived GP/TLE element set whose epoch is not later than that snapshot.
3. Propagate the two objects to the snapshot time with SGP4.
4. Calculate relative distance and relative velocity.
5. Run the existing analytic, Monte-Carlo, and QAE probability pipeline.
6. Classify the risk from the calculated probability; do not force a HIGH label.
7. If the system reaches HIGH risk, use the state at that time as the starting point for a forward forecast.
8. Compare the forecast with the historical collision afterward.

## Historical data acquisition

The repository does not fabricate historical TLEs. Obtain archived GP/TLE element sets from an authoritative archive such as CelesTrak's historical GP request service or another permitted authoritative archive, then place the text files in a local/raw directory. CelesTrak's archive request supports NORAD catalog numbers and date ranges.

For Iridium 33/Cosmos 2251, use NORAD IDs **24946** and **22675** and make sure the archive reaches at least **2009-01-11 UTC** so the 30-day rewind has valid input. The collision occurred at about **2009-02-10 16:56 UTC**. CelesTrak documents that SOCRATES predicted the conjunction in each of the 14 reports during the week before the event, with the final report predicting a 584 m close approach.

For CERISE, use NORAD IDs **23606** and **18208** and an archive that reaches at least 30 days before the 1996-07-24 event. Historical analysis of the event documents close approaches between CERISE and the Ariane fragment.

## Replay command

After the archived TLE files are available locally, the replay engine can be run like this:

```text
python -m modules.historical_replay --event-id IRIDIUM33_COSMOS2251 --event-time 2009-02-10T16:56:00Z --norad-a 24946 --norad-b 22675 --tle path/to/iridium33_history.tle path/to/cosmos2251_history.tle --rewind-days 30 --forecast-days 0 --step-hours 24 --output results/historical/iridium33_cosmos2251_states.csv --pair-output results/historical/iridium33_cosmos2251_pair.csv
```

The replay engine produces one state row per object per timestamp and a pair-level CSV containing relative distance, relative velocity, and the actual-event flag. The pair-level file is intended to feed the existing probability/risk pipeline without replacing it.

## Event status

Only events with authoritative confirmation and sufficient pre-event orbital data should be used for quantitative backtesting. An event can remain in the catalog while being marked `needs_archival_tle` until its historical input data are obtained and verified.
