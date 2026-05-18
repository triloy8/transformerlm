#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC2034
CONFIG="${1:-${CONFIG:-config/resources/train.toml}}"
# shellcheck disable=SC2034
EXTRA_ARGS="${2:-${EXTRA_ARGS:-}}"

export PATH="${HOME}/.local/bin:${PATH}"

if [ ! -f env/wandb.env ]; then
	echo "Missing env/wandb.env; run \`just sync-env\` first" >&2
	exit 1
fi

read_env_value() {
	local var_name="$1"
	local file_path="$2"
	grep -Em1 "^[[:space:]]*(export[[:space:]]+)?${var_name}=" "$file_path" \
		| sed -E "s/^[[:space:]]*(export[[:space:]]+)?${var_name}=//" \
		| tr -d '\r\n'
}

WANDB_API_KEY=$(
	read_env_value "WANDB_API_KEY" "env/wandb.env"
)

if [ -z "${WANDB_API_KEY:-}" ]; then
	echo "WANDB_API_KEY is empty" >&2
	exit 1
fi

if [ -f env/checkpointing.env ]; then
	CHECKPOINTING_S3_BUCKET="$(read_env_value "CHECKPOINTING_S3_BUCKET" "env/checkpointing.env")"
	CHECKPOINTING_S3_PREFIX="$(read_env_value "CHECKPOINTING_S3_PREFIX" "env/checkpointing.env")"
	CHECKPOINTING_S3_ENDPOINT_URL="$(read_env_value "CHECKPOINTING_S3_ENDPOINT_URL" "env/checkpointing.env")"
	CHECKPOINTING_S3_REGION="$(read_env_value "CHECKPOINTING_S3_REGION" "env/checkpointing.env")"
	CHECKPOINTING_S3_ACCESS_KEY_ID="$(read_env_value "CHECKPOINTING_S3_ACCESS_KEY_ID" "env/checkpointing.env")"
	CHECKPOINTING_S3_SECRET_ACCESS_KEY="$(read_env_value "CHECKPOINTING_S3_SECRET_ACCESS_KEY" "env/checkpointing.env")"
	CHECKPOINTING_S3_SESSION_TOKEN="$(read_env_value "CHECKPOINTING_S3_SESSION_TOKEN" "env/checkpointing.env")"
fi

if [ -f env/huggingface.env ]; then
	HF_TOKEN="$(read_env_value "HF_TOKEN" "env/huggingface.env")"
	CHECKPOINTING_HF_TOKEN="$(read_env_value "CHECKPOINTING_HF_TOKEN" "env/huggingface.env")"
	CHECKPOINTING_HF_REPO_ID="$(read_env_value "CHECKPOINTING_HF_REPO_ID" "env/huggingface.env")"
	CHECKPOINTING_HF_REPO_TYPE="$(read_env_value "CHECKPOINTING_HF_REPO_TYPE" "env/huggingface.env")"
	CHECKPOINTING_HF_REVISION="$(read_env_value "CHECKPOINTING_HF_REVISION" "env/huggingface.env")"
	CHECKPOINTING_HF_PATH_IN_REPO="$(read_env_value "CHECKPOINTING_HF_PATH_IN_REPO" "env/huggingface.env")"
	CHECKPOINTING_HF_PRIVATE="$(read_env_value "CHECKPOINTING_HF_PRIVATE" "env/huggingface.env")"
	CHECKPOINTING_HF_STRICT="$(read_env_value "CHECKPOINTING_HF_STRICT" "env/huggingface.env")"
fi

if ! command -v tmux >/dev/null 2>&1; then
	echo "tmux not available on remote host" >&2
	exit 1
fi

SESSION="transformerlm-train"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
	tmux kill-session -t "${SESSION}"
fi

CMD="uv run transformerlm-train --config \"${CONFIG}\" ${EXTRA_ARGS}"
tmux new -d -s "${SESSION}" \
	"WANDB_API_KEY=${WANDB_API_KEY} \
CHECKPOINTING_S3_BUCKET=${CHECKPOINTING_S3_BUCKET:-} \
CHECKPOINTING_S3_PREFIX=${CHECKPOINTING_S3_PREFIX:-} \
CHECKPOINTING_S3_ENDPOINT_URL=${CHECKPOINTING_S3_ENDPOINT_URL:-} \
CHECKPOINTING_S3_REGION=${CHECKPOINTING_S3_REGION:-} \
	CHECKPOINTING_S3_ACCESS_KEY_ID=${CHECKPOINTING_S3_ACCESS_KEY_ID:-} \
	CHECKPOINTING_S3_SECRET_ACCESS_KEY=${CHECKPOINTING_S3_SECRET_ACCESS_KEY:-} \
	CHECKPOINTING_S3_SESSION_TOKEN=${CHECKPOINTING_S3_SESSION_TOKEN:-} \
	HF_TOKEN=${HF_TOKEN:-} \
	CHECKPOINTING_HF_TOKEN=${CHECKPOINTING_HF_TOKEN:-} \
	CHECKPOINTING_HF_REPO_ID=${CHECKPOINTING_HF_REPO_ID:-} \
	CHECKPOINTING_HF_REPO_TYPE=${CHECKPOINTING_HF_REPO_TYPE:-} \
	CHECKPOINTING_HF_REVISION=${CHECKPOINTING_HF_REVISION:-} \
	CHECKPOINTING_HF_PATH_IN_REPO=${CHECKPOINTING_HF_PATH_IN_REPO:-} \
	CHECKPOINTING_HF_PRIVATE=${CHECKPOINTING_HF_PRIVATE:-} \
	CHECKPOINTING_HF_STRICT=${CHECKPOINTING_HF_STRICT:-} \
	${CMD}"
echo "Started tmux session ${SESSION}"
