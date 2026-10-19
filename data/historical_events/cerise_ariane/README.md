# CERISE–Ariane debris historical replay candidate

**Event:** CERISE (NORAD 23606) and Ariane 1 debris (NORAD 18208)  
**Reported event time:** 1996-07-24 09:48:03 UTC  
**Catalog status:** confirmed historical event; replay inputs not yet archived in this repository.

NASA's Orbital Debris Quarterly News reported a close encounter between catalogued objects 23606 and 18208 at approximately 09:48 UTC on 24 July 1996, with subsequent confirmation of damage to CERISE. See the [NASA report](https://orbitaldebris.jsc.nasa.gov/quarterly-news/pdfs/ODQNV1i2.pdf). A later NASA historical summary also records the event and gives an example of pre-event CERISE orbital elements; one element set is not a substitute for a complete time series.

## Required archive

Before running this as a second backtest, acquire authentic historical element sets for **both** NORAD IDs covering at least 1996-06-24 through 1996-07-24 09:48 UTC. Preserve the source response and its metadata.

For each element record, the research needs:
- object catalog ID;
- element epoch (EPOCH);
- source publication/creation timestamp (CREATION_DATE) or equivalent independently documented first-availability time;
- original source URL/query and retrieval timestamp;
- the actual element set used by SGP4.

Space-Track documents its GP_HISTORY service as the source for historical element sets and explains that CREATION_DATE distinguishes when element data was published to Space-Track. See the [Space-Track API documentation](https://www.space-track.org/documentation). Access may require a Space-Track account and compliance with its usage rules.

Do **not** create synthetic TLEs from a paper's summary elements, infer publication time from element epoch, or mark the provenance as verified just because the records can be propagated.

## Build the availability table

After obtaining a GP_HISTORY CSV export that contains NORAD_CAT_ID, EPOCH, and CREATION_DATE, run:

~~~bash
python -m modules.build_tle_availability \
  --input-csv path/to/cerise_gp_history.csv \
  --output-csv data/historical_events/cerise_ariane/record_level_availability.csv
~~~

The script retains the earliest documented creation timestamp for each unique object/epoch. Cross-check that the availability rows correspond exactly to the element records used in the replay. Keep the original export and query log alongside the converted data.

The event remains **not ready for quantitative replay** until both object histories, coverage, record matching, and publication timestamps have been independently checked. Even verified data availability alone will not establish predictive skill; that requires independent baseline comparisons and multiple events.
