#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Pass the build of an upstream asset to the asset that reads it.

Every layer of builds reports its build id to Dagster as its data version. A downstream
asset reads the build that Dagster recorded, not the build that a manifest names when
the asset runs. So Dagster's lineage names the exact input of every run.
"""

import dagster as dg

# The tag of a materialization that holds its data version.
DATA_VERSION_TAG = "dagster/data_version"


def read_upstream_build_id(context: dg.AssetExecutionContext, key: dg.AssetKey) -> str:
    event = context.instance.get_latest_materialization_event(key)
    materialization = None if event is None else event.asset_materialization
    build_id = (
        None if materialization is None else materialization.tags.get(DATA_VERSION_TAG)
    )
    if build_id is None:
        raise dg.Failure(
            description=f"{key.to_user_string()} has no build. Materialize it first."
        )
    return build_id
