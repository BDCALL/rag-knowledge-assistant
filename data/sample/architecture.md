# Platform Architecture Overview

Robots stream telemetry over MQTT to an edge gateway in each warehouse. The gateway batches
messages and forwards them to the cloud ingestion service, which writes to a time-series
database and publishes events to Kafka.

## Services

- **Ingestion service**: validates and deduplicates telemetry, written in Go.
- **Fleet API**: a FastAPI service that exposes robot status to the operations dashboard.
- **Planner**: assigns picking tasks to robots using a constraint solver.

## Data retention

Raw telemetry is kept for 30 days. Hourly aggregates are kept for two years for capacity planning.
