#!/usr/bin/env bash
set -euo pipefail

err() {
    echo "bootstrap_remote: $*" >&2
}

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

export PATH="${HOME}/.local/bin:${PATH}"

apt_updated=false
apt_install() {
    if [ "${apt_updated}" = false ]; then
        ${SUDO} apt-get update
        apt_updated=true
    fi
    ${SUDO} apt-get install -y "$@"
}

if ! command -v curl >/dev/null 2>&1; then
    apt_install curl
fi

if ! command -v git >/dev/null 2>&1; then
    apt_install git
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v just >/dev/null 2>&1; then
    curl -fsSL https://just.systems/install.sh | ${SUDO} bash -s -- --to /usr/local/bin
fi

if ! command -v tmux >/dev/null 2>&1; then
    apt_install tmux
fi

if ! command -v rsync >/dev/null 2>&1; then
    apt_install rsync
fi

if ! command -v nvitop >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        uvx nvitop
    else
        python3 -m pip install --user nvitop
    fi
fi

repo_root="${HOME}/transformerlm"

if [ ! -d "${repo_root}/.git" ]; then
    if [ -d "${repo_root}" ] && [ "$(ls -A "${repo_root}")" ]; then
        err "${repo_root} exists but is not a git repo; aborting clone"
        exit 1
    fi
    git clone https://github.com/triloy8/transformerlm.git "${repo_root}"
else
    (
        cd "${repo_root}"
        git fetch --all --prune
        git pull --ff-only || err "git pull failed; please resolve manually"
    )
fi

mkdir -p "${repo_root}/data" "${repo_root}/runs" "${repo_root}/env"

(
    cd "${repo_root}"
    if command -v uv >/dev/null 2>&1; then
        uv python install 3.11 || true
        uv sync --frozen || uv sync
    fi
)

train_env="${repo_root}/env/train.env"
if [ ! -f "${train_env}" ]; then
    cat > "${train_env}" <<'EOF'
# Reserved for optional non-secret environment variables.
EOF
fi

echo "Bootstrap complete: repo synced, uv environment ready, tooling installed."
