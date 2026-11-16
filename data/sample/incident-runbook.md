# Runbook: Fleet Telemetry Ingestion

## Alert NWR-4031: telemetry backlog above threshold

The ingestion queue holds more than 50,000 unprocessed messages for over ten minutes.

Likely causes:

- A consumer deployment is crash-looping. Check `kubectl get pods -n telemetry`.
- The time-series database is throttling writes. Check the write-latency dashboard.

Mitigation: scale the consumer deployment to 12 replicas, then confirm the backlog drains.
If it keeps growing after 15 minutes, page the database on-call engineer.

## Alert NWR-5102: robot heartbeat missing

A robot has not sent a heartbeat for five minutes. Usually this is a warehouse Wi-Fi outage,
not a software fault. Confirm with the site team before restarting any services.
