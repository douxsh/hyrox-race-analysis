# Supplementary Reproducibility Note: Data-Quality Thresholds

This note documents the conservative lower-bound thresholds used to flag likely timing artifacts in HYROX split data.

## Thresholds (seconds)

- `1000m SkiErg`: `>= 60`
- `50m Sled Push`: `>= 40`
- `50m Sled Pull`: `>= 40`
- `80m Burpee Broad Jump`: `>= 40`
- `1000m Row`: `>= 60`
- `200m Farmers Carry`: `>= 40`
- `100m Sandbag Lunges`: `>= 40`
- `Wall Balls`: `>= 40`
- `Running 1` to `Running 8`: `>= 60`
- `Roxzone Total`: `>= 30`

## Rationale

- Values below these bounds are physiologically and operationally implausible for official HYROX station distances/standards.
- The objective is artifact detection (e.g., malformed timestamp parsing, dropped digits), not aggressive performance trimming.
- Thresholds are intentionally conservative to reduce false-positive exclusion of valid slow performances.

## Use in This Project

- Primary analyses retain the main cleaned cohort after missing-value filtering.
- Robustness analyses additionally use a plausible-entry subset after applying the threshold rules above.
