# Convenience targets. Set IMAGE to your own registry/name, e.g.
#   make image IMAGE=ghcr.io/me/connex-vllm:0.1.0 VLLM_FORK_REPO=https://...
IMAGE          ?= connex-vllm:dev
BASE_IMAGE     ?= vllm/vllm-openai:v0.11.2
VLLM_FORK_REPO ?=
VLLM_FORK_REF  ?= 380186d57
FORK_INSTALL   ?= overlay
COMPOSE        ?= docker compose
PYTHON         ?= python3

.PHONY: help image push up down smoke cpu-demo test check traces

help:
	@echo "image     build the stage image (needs VLLM_FORK_REPO)"
	@echo "push      push IMAGE to its registry"
	@echo "up/down   single-host 2-stage compose demo (2 GPUs)"
	@echo "smoke     wait for the compose demo and send one completion"
	@echo "cpu-demo  CPU-only end-to-end: router, fake engines, resume proxy"
	@echo "test      unit tests;  check  release scanner"
	@echo "traces    generate sample gamma traces into benchmarks/traces/"

image:
	@test -n "$(VLLM_FORK_REPO)" || { echo "set VLLM_FORK_REPO=<git URL of the external-PP vLLM fork>"; exit 1; }
	docker build -f docker/Dockerfile -t $(IMAGE) \
	  --build-arg BASE_IMAGE=$(BASE_IMAGE) --build-arg VLLM_FORK_REPO=$(VLLM_FORK_REPO) \
	  --build-arg VLLM_FORK_REF=$(VLLM_FORK_REF) --build-arg FORK_INSTALL=$(FORK_INSTALL) .

push:
	docker push $(IMAGE)

up:
	CONNEX_IMAGE=$(IMAGE) $(COMPOSE) -f docker/docker-compose.yml up -d

down:
	$(COMPOSE) -f docker/docker-compose.yml --profile proxy down

smoke:
	docker/smoke_test.sh

cpu-demo:
	examples/cpu_demo.sh

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider tests

check:
	$(PYTHON) scripts/check_release.py

traces:
	for q in 2 4 8; do $(PYTHON) benchmarks/k8s/gen_trace.py --qps $$q --duration 240 \
	  --out benchmarks/traces/gamma_qps$$q.0.csv; done
