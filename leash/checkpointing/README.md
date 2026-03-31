# Checkpointing System Specification

## Scope
This document specifies the checkpointing module, its on-disk layout, manifest schema, resume semantics, and design choices. The module targets distributed training with per-rank artifacts, local storage, and optional S3-compatible object storage.

## Goals
- Deterministic, manifest-driven checkpointing.
- Per-rank artifacts (no forced gathering) for scalability.
- Safe resumption with explicit exactness signaling.
- Storage-agnostic keys for local and S3 backends.
- No overwrite of existing runs by default.

## Non-goals
- Perfect determinism for streaming datasets without explicit iterator state.
- Automatic optimizer conversion across optimizer types.
- Tight coupling between filesystem run IDs and logger run IDs.

## Directory Layout
```
runs/
  <run_id>/
    config/
      train.toml
      config.json
    aliases/
      latest.json
      best.json
    versions/
      v000001/
        model.safetensors
        opt_shard_rank0000.bin
        opt_shard_rank0001.bin
        rng_rank0000.json
        rng_rank0001.json
        manifest.json
      v000002/
        ...
```

## Artifacts Per Version
- `model.safetensors`: model weights (rank 0 writes).
- `opt_shard_rankXXXX.bin`: optimizer shard for each rank.
- `rng_rankXXXX.json`: RNG + batcher state per rank.
- `manifest.json`: version manifest describing the snapshot.

## Manifest Keys and Resolution
- Manifest `key` fields are repo-root relative (e.g., `runs/<run_id>/versions/v000420/model.safetensors`).
- These keys are used for local paths and S3 object keys.
- Alias files contain `manifest_key` for direct resolution.

## Version Manifest Schema (Summary)
Required fields (simplified):
- `schema_version`: integer
- `run_id`, `version_id`, `created_at`, `step`
- `paths`: `{layout_version, root_local, root_remote?}`
- `config`: `{key, sha256, bytes}`
- `model`: `{key, sha256, bytes}`
- `optimizer`: `{sharding, shards: [{rank, key, sha256, bytes}]}`
- `rng`: `{per_rank: true, keys: [{rank, key}]}`
- `resume`: `{base_step, exact}`
- `metrics`: arbitrary dict

## Run Manifest Schema (Summary)
- `schema_version`, `run_id`, `created_at`
- `paths`, `config`
- `aliases`: `latest` + `best`
- `versions`: list of `{version_id, step, created_at, model_key, metrics}`

## Aliases
- `aliases/latest.json` always points to the most recent version.
- `aliases/best.json` always exists; `status` is `pending` until the first usable metric appears.
- `manifest_key` is stored for fast resolution without scanning directories.

## Resume Semantics
- Resume is driven by a manifest path or alias (`latest`/`best`).
- `resume.exact = true` only when RNG + batcher state is exact across all ranks.
- If `resume.exact = false`, training is best-effort and loss curves may diverge.
- `checkpointing.resume_optimizer = false` skips optimizer state load (useful for optimizer changes).

## Best-Effort Streaming
- `StreamingBatcher.get_state()` saves the internal buffer.
- The underlying iterator is not fully resumable, so exact replay is not guaranteed.
- The manifest `resume.exact` flag is derived from batcher state fields.

## Storage Backends
- Local filesystem always supported.
- Optional S3-compatible storage is enabled via environment variables only.

Env vars:
```
CHECKPOINTING_S3_BUCKET
CHECKPOINTING_S3_PREFIX
CHECKPOINTING_S3_ENDPOINT_URL
CHECKPOINTING_S3_REGION
CHECKPOINTING_S3_ACCESS_KEY_ID
CHECKPOINTING_S3_SECRET_ACCESS_KEY
CHECKPOINTING_S3_SESSION_TOKEN
```

See `env/checkpointing.env.example` for a template.

## Design Choices
- **Manifest-first**: Every snapshot is self-describing for reproducibility and remote sync.
- **Per-rank shards**: Avoids synchronization overhead and preserves sharded optimizer state.
- **Fresh run IDs**: New runs always write to a new `runs/<run_id>` directory.
- **Decoupled logging**: Filesystem run ID is not forced to match logger run name.
- **Schema-driven safety**: Fields required for resume are explicit and validated.

## Caveats
- Streaming datasets are not fully deterministic without iterator state support.
- Changing optimizer type while resuming will fail unless `resume_optimizer = false`.
- If `runs_path` changes between save and resume, repo-relative keys must still resolve correctly.
