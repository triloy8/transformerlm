set shell := ["bash", "-euo", "pipefail", "-c"]

prime_host := env_var_or_default("PRIME_HOST", "prime-node")
remote_root := env_var_or_default("REMOTE_ROOT", "~/transformerlm")
infer_command_default := env_var_or_default("CMD_INFER", "uv run transformerlm-bench-infer --config config/resources/bench_infer.toml")

bootstrap-remote:
	ssh {{prime_host}} 'bash -s' < scripts/bootstrap_remote.sh

data-remote:
	ssh {{prime_host}} "cd {{remote_root}} && bash -s" < scripts/fetch_data.sh

build-remote:
	ssh {{prime_host}} "cd {{remote_root}} && export PATH=\"\\$HOME/.local/bin:\\$PATH\" && (uv sync --frozen || uv sync)"

train config="config/resources/train.toml" extra="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_train_remote.sh $(printf '%q' '{{config}}') $(printf '%q' '{{extra}}')"

train-mnist extra="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_train_remote.sh $(printf '%q' 'config/resources/train_mnist_flow.toml') $(printf '%q' '{{extra}}')"

train-mnist-categorical-flow extra="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_train_remote.sh $(printf '%q' 'config/resources/train_mnist_categorical_flow.toml') $(printf '%q' '{{extra}}')"

sweep-train config="config/resources/wandb/train_sweep.yaml" extra="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_sweep_train_remote.sh $(printf '%q' '{{config}}') $(printf '%q' '{{extra}}')"

infer command="{{infer_command_default}}" args="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_infer_remote.sh $(printf '%q' '{{command}}') $(printf '%q' '{{args}}')"

infer-mnist-categorical-flow args="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_infer_remote.sh $(printf '%q' 'uv run transformerlm-infer-image --config config/resources/infer_mnist_categorical_flow.toml') $(printf '%q' '{{args}}')"

nvitop:
	ssh -t {{prime_host}} 'export PATH="$HOME/.local/bin:$PATH"; uvx nvitop'

attach-train:
	ssh -t {{prime_host}} 'tmux attach -t transformerlm-train'

attach-sweep:
	ssh -t {{prime_host}} 'tmux attach -t transformerlm-sweep-train'

kill-train:
	ssh {{prime_host}} 'tmux kill-session -t transformerlm-train 2>/dev/null || true'

kill-sweep:
	ssh {{prime_host}} 'tmux kill-session -t transformerlm-sweep-train 2>/dev/null || true'

fetch any_file:
	echo "Fetching {{any_file}} from {{prime_host}}"
	scp -r {{prime_host}}:{{remote_root}}/{{any_file}} {{any_file}}

rsync any_path:
	rsync -av --partial --progress {{prime_host}}:{{remote_root}}/{{any_path}} {{any_path}}

rsync-run run_name:
	rsync -av --partial --progress {{prime_host}}:{{remote_root}}/runs/{{run_name}} runs/

rsync-latest-run:
	latest="$(ssh {{prime_host}} "ls -1t {{remote_root}}/runs | head -n 1")"; \
	if [ -z "$latest" ]; then echo "No runs found on {{prime_host}}:{{remote_root}}/runs" >&2; exit 1; fi; \
	echo "Fetching latest run: $latest"; \
	rsync -av --partial --progress {{prime_host}}:{{remote_root}}/runs/"$latest" runs/

list-runs:
	ssh {{prime_host}} "ls -1 {{remote_root}}/runs"

sync-env:
	if [ ! -f env/wandb.env ]; then echo "Missing env/wandb.env; copy env/wandb.env.example and fill WANDB_API_KEY" >&2; exit 1; fi
	scp env/wandb.env {{prime_host}}:{{remote_root}}/env/wandb.env
	if [ -f env/checkpointing.env ]; then scp env/checkpointing.env {{prime_host}}:{{remote_root}}/env/checkpointing.env; else echo "Skipping env/checkpointing.env (optional)"; fi
	if [ -f env/huggingface.env ]; then scp env/huggingface.env {{prime_host}}:{{remote_root}}/env/huggingface.env; else echo "Skipping env/huggingface.env (optional)"; fi

auto-train: bootstrap-remote data-remote sync-env train

auto-train-mnist: bootstrap-remote data-remote sync-env train-mnist

auto-train-mnist-categorical-flow: bootstrap-remote data-remote sync-env train-mnist-categorical-flow

auto-sweep-train: bootstrap-remote data-remote sync-env sweep-train
