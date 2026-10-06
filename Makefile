# The tensor-runtime integration profile resolves unisim-core from the relative
# sibling checkout. Repository-owned test entrypoints therefore provide the
# explicit approved-source sentinel. `make check-workspace` verifies the pinned
# commits; direct uv users can run scripts/tools/sync_tensor_workspace.py.
UNILAB_LOCAL_UNISIM ?= $(abspath ../unisim)
export UNILAB_LOCAL_UNISIM

.PHONY: sync-workspace
sync-workspace:
	uv run --no-project python scripts/tools/sync_tensor_workspace.py --sync

.PHONY: check-workspace
check-workspace:
	uv run --no-project python scripts/tools/sync_tensor_workspace.py

.PHONY: sync
sync: sync-workspace
	uv sync --extra mujoco --extra uni_rl

.PHONY: setup
setup: sync
	uv run --no-sync unilab-complete install

.PHONY: install-completion
install-completion:
	uv run --no-sync unilab-complete install

.PHONY: sync-rocm
sync-rocm:
	@cp pyproject.rocm.toml pyproject.toml
	@if [ -f uv.rocm.lock ]; then cp uv.rocm.lock uv.lock; fi
	uv sync --extra mujoco --extra uni_rl
	cp uv.lock uv.rocm.lock

.PHONY: sync-xpu
sync-xpu: sync-workspace
	uv sync --extra mujoco --extra uni_rl --no-install-package torch
	uv pip install torch==2.7.0 --torch-backend xpu

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run mypy src/unilab
	uv run pyright

.PHONY: check
check: format type check-tests

.PHONY: check-tests
check-tests:
	uv run ruff check tests --select F401,F821,F811,F841 --output-format concise

.PHONY: test
test:
	uv run pytest -m "not slow"

.PHONY: test-cov
test-cov:
	uv run pytest -m "not slow" --cov=src/unilab --cov-report=term-missing

.PHONY: test-slow
test-slow:
	uv run pytest -m "slow" -v

.PHONY: test-benchmark-smoke
test-benchmark-smoke:
	uv run python scripts/benchmark/smoke_test.py

.PHONY: test-all
test-all: check test-cov test-benchmark-smoke

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -f train_appo.log train_sac.log train_flashsac.log train_rsl_rl.log MUJOCO_LOG.TXT
	find src/unilab/assets/.cache -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find src/unilab/assets/caches -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find src/unilab/assets/checkpoints -type f ! -name '.gitkeep' -delete 2>/dev/null || true
	find src/unilab/assets/scenes -type f ! -name '.gitkeep' -delete 2>/dev/null || true

.PHONY: setup-drake
setup-drake:
	uv run --no-sync bash scripts/tools/setup_drake_env.sh --download-drake
