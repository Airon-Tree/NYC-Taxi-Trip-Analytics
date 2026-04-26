# Stage 4 - Temporal + Analytics Outputs

This folder stores Stage 4 temporal analytics output tables.

Run command:

python src/analytics/temporal_analysis.py

Required input:

data/processed/zone_hour_features/

Main output:

outputs/tables/temporal/parquet/

CSV outputs are also generated under outputs/tables/temporal/csv/.

Demand definition:

pickup demand = trip_count

Output tables:

1. kpi_summary
- One-row summary of total trips, revenue, date range, active days, and active zones.

2. hourly_demand
- One row per hour of day.
- Used to analyze 24-hour taxi demand pattern.

3. daily_demand
- One row per pickup date.
- Used to analyze daily demand trend.

4. monthly_demand
- One row per year-month.
- Used to analyze monthly demand trend.

5. weekday_weekend_hourly
- One row per weekday/weekend flag and hour.
- Used to compare weekday and weekend hourly patterns.

6. weekday_hour_heatmap
- One row per weekday and hour.
- Ready for weekday-hour heatmap visualization.

7. borough_hourly_pattern
- One row per pickup borough and hour.
- Used to compare hourly demand across boroughs.

8. rush_hour_summary
- One row per time period.
- Time periods: late_night_00_05, morning_peak_06_10, midday_11_15, evening_peak_16_19, night_20_23.

9. top_zones_by_hour
- Top 10 pickup zones for each hour.

10. top_zones_overall
- Overall top pickup zones by total trips.

Notes:
- Stage 4 does not read raw trip records directly.
- Stage 4 reads the Stage 3 zone_hour_features table.
- Spark writes parquet and CSV outputs as folders containing part files.
