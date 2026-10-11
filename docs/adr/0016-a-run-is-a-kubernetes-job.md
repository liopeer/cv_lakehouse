<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0016: A run on Kubernetes is a Kubernetes job

- Status: accepted
- Date: 2026-10-12

## Context

The Kubernetes deployment launched each run with the `DefaultRunLauncher`. A run was a
subprocess of the code server.

On 2026-10-11, the rollout of 0.9.0 restarted the code server. It killed two runs. They
stayed in STARTED, and they held both slots of the run queue. Three runs waited behind
them.

Dagster run monitoring detects a dead run worker only with a launcher other than the
`DefaultRunLauncher`.

## Decision

On Kubernetes, the `K8sRunLauncher` of `dagster-k8s` launches each run as a Kubernetes
job.

- The app image holds `dagster-k8s`, so the daemon and the webserver can launch and
  stop a job.
- The code server reports its own image in `DAGSTER_CURRENT_IMAGE`. A run job uses that
  image.
- A run job uses the multiprocess executor. The `k8s_job_executor` was rejected: one
  pod per step costs a pod start per asset.
- Run monitoring fails a run whose job died, and does not resume it.

The docker compose stack in `deploy/` keeps the `DefaultRunLauncher`.

## Consequences

- A rollout of the code server kills no run. A run finishes on the image that it
  started on.
- A run whose pod dies fails, and frees its slot in the queue.
- A run waits for a pod start before its first step.
- The deployment configures the environment and the resources of a run job in the
  launcher, not in the code server.
- A restart of the node still kills every run.
