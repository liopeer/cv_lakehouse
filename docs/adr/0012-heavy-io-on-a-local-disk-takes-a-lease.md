<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0012: Heavy I/O on a local disk takes a lease

- Status: accepted
- Date: 2026-10-08

## Context

On 2026-10-05, the bronze `open_images` run unpacked its archives on the local HDD of the
lake. At the same time, the silver `pp4av` run read its images. The disk was at 99%
utilisation. After 55 minutes, silver had used 7 s of CPU. See issue #40.

The limit depends on the operation and on the storage backend, not on the layer. An HDD
suffers from concurrent random reads. S3 and an SSD do not.

## Decision

Core has a `DiskLease`. One unit of heavy I/O on a local disk holds it.

- The deployment enables it with `<PREFIX>_DISK_LEASE_PATH`, the path of a lock file.
  Unset means that the lease does nothing. S3 and an SSD leave it unset.
- The setting is a path, not a flag. Two domains on one disk set the same path.
- A target that is not local takes no lease, even when the path is set.
- One lease covers one unit of work: one archive, one checksum, or one silver split.
- A silver split holds the lease while Triton embeds it, because Triton reads the
  images by path.
- `fcntl.flock` implements it. The kernel releases it when a process dies.
- A run that waits logs the host, the pid and the work that hold the disk.

These options were rejected:

- Dagster pools. They need one op per archive, and they do not know the backend.
- A lease inside `LakeStore`. `tarfile`, PIL and Triton read by path, not through the
  store.
- `ionice`. The `mq-deadline` scheduler ignores the I/O class.
- A pool per layer. A bronze download waits on the network. A silver scan is heavy
  random reads.

## Consequences

- A step that waits for the lease shows as running in Dagster, not as queued.
- The lease works on one host only. Waiters do not get the lease in order.
- A new heavy operation must take the lease. Nothing enforces that.
- The lease covers only I/O that the lake code starts or waits for. Another process on
  the disk is not limited.
- The network part of a bronze download takes no lease, so downloads still overlap.
