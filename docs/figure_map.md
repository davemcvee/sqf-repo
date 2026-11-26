# Figure map: report figures to script outputs

## Why the numbers differ

Each written report numbers its figures 1 to n, in the order a reader meets them, so the reports read cleanly on their own. The scripts name their outputs in the order the figures were created while the analysis was being worked out. Some outputs were never used in a report, and some were reworked along the way, so the two numbering schemes do not match.

This table links the two. Run the numbered scripts from the repo root and each output is written to `figures/` under the filename shown. The `images/` folder holds only the report figures used in the README.

## How this map was made

I matched each report figure to its script output by comparing the figure's caption and content in the report with the filename and code of the script that writes it. I did not regenerate the figures to confirm the match visually. The Report 3 rows are the least obvious, since each model run writes a confusion matrix and an importance chart, so those were matched by the model and feature set named in each caption.

## Report 1: Data and Visualization

| Report figure | Shows | Script output | Written by |
|---|---|---|---|
| 1 | Demographics: race, age, sex | `fig1_demographics.png` | `08_visualizations.py` |
| 2 | Reasons cited for the stop | `fig2_reasons_for_stop.png` | `08_visualizations.py` |
| 3 | Additional circumstances | `fig2b_additional_circumstances.png` | `08_visualizations.py` |
| 4 | What happens after a stop | `fig3_outcome_sequence.png` | `08_visualizations.py` |
| 5 | Type of physical force | `fig4_force_type.png` | `08_visualizations.py` |
| 6 | Weapons found | `fig5_weapons_found.png` | `08_visualizations.py` |
| 7 | Officer uniform and self-identification | `fig6_officer_compliance.png` | `08_visualizations.py` |
| 8 | Correlation of numeric attributes with outcomes | `fig8_correlation_heatmap.png` | `09_relationships.py` |
| 9 | Outcome rate by hour of day | `fig14_hour_of_day.png` | `09_relationships.py` |
| 10 | Stop density by location | `fig9_spatial_density.png` | `09_relationships.py` |
| 11 | Outcome rates by race | `fig10_race_outcomes.png` | `09_relationships.py` |
| 12 | Suspicion factor vs weapon or contraband found | `fig11_suspicion_weapon_lift.png` | `09_relationships.py` |
| 13 | Outcome rate by age group | `fig12_age_outcomes.png` | `09_relationships.py` |
| 14 | Reason for stop vs force tier | `fig13_reason_vs_force.png` | `10_reason_vs_force.py` |

## Report 2: Cluster Analysis

| Report figure | Shows | Script output | Written by |
|---|---|---|---|
| 1 | k-distance plot, night subset | `fig15_kdist_night.png` | `11_cluster_dbscan_location.py` |
| 2 | DBSCAN hotspots, night | `fig16_dbscan_night.png` | `11_cluster_dbscan_location.py` |
| 3 | DBSCAN hotspots, day | `fig16_dbscan_day.png` | `11_cluster_dbscan_location.py` |
| 4 | Inertia and silhouette across k (reasons) | `fig17_kmeans_reasons_kselect.png` | `12_cluster_kmeans_reasons.py` |
| 5 | The nine reason clusters, by reason | `fig18_kmeans_reasons_profiles.png` | `12_cluster_kmeans_reasons.py` |
| 6 | Outcomes and race by reason cluster | `fig19_kmeans_reasons_outcomes_demo.png` | `12_cluster_kmeans_reasons.py` |
| 7 | Elbow plot, precinct profiles | `fig20_precinct_kselect.png` | `13_cluster_precinct.py` |
| 8 | The five precinct clusters | `fig21_precinct_profiles.png` | `13_cluster_precinct.py` |
| 9 | Precinct clusters on a map | `fig22_precinct_map.png` | `13_cluster_precinct.py` |

## Report 3: Predictive Modelling

| Report figure | Shows | Script output | Written by |
|---|---|---|---|
| 1 | Confusion matrices, weapon found, Model A | `fig23a_weapon_behavior_confusion.png` | `14_predict_weapon.py` |
| 2 | Feature importances, weapon found, Model B | `fig23b_weapon_demo_importance.png` | `14_predict_weapon.py` |
| 3 | Confusion matrices, arrest made | `fig24_arrest_confusion.png` | `15_predict_arrest.py` |
| 4 | Feature importances, arrest made | `fig24_arrest_importance.png` | `15_predict_arrest.py` |
| 5 | Confusion matrices, race, Bucket 1+2 | `fig25b_race_bucket12_confusion.png` | `16_predict_race.py` |
| 6 | Feature importances, race, Bucket 1+2 | `fig25b_race_bucket12_importance.png` | `16_predict_race.py` |

## Script outputs not used in the reports

These are still written by the scripts. They were part of working out the analysis and were left out of the final reports.

| Script output | Written by |
|---|---|
| `fig15_kdist_day.png` | `11_cluster_dbscan_location.py` |
| `fig23a_weapon_behavior_importance.png` | `14_predict_weapon.py` |
| `fig23b_weapon_demo_confusion.png` | `14_predict_weapon.py` |
| `fig25a_race_bucket1_confusion.png` | `16_predict_race.py` |
| `fig25a_race_bucket1_importance.png` | `16_predict_race.py` |

There is no `fig7` in the script outputs. That number was never used.
