# Convenience targets. Set IMAGE to your own registry/name, e.g.
#   make image IMAGE=ghcr.io/me/connex-vllm:dev
# builds latest vLLM main with the three proposed PP changes (PP_ENGINE=upstream).
# Legacy external-PP fork:
#   make image PP_ENGINE=fork VLLM_FORK_REPO=https://... BASE_IMAGE=<base matching the fork>
IMAGE          ?= connex-vllm:dev
PP_ENGINE      ?= upstream
BASE_IMAGE     ?= vllm/vllm-openai:latest
VLLM_REF       ?= main
VLLM_INSTALL   ?= precompiled
VLLM_FORK_REPO ?=
VLLM_FORK_REF  ?= 380186d57
FORK_INSTALL   ?= overlay
COMPOSE        ?= docker compose
COMPOSE_FILE   ?= $(if $(filter fork,$(PP_ENGINE)),docker/docker-compose.fork.yml,docker/docker-compose.yml)
PYTHON         ?= python3

.PHONY: help image push up down smoke cpu-demo test check traces

help:
	@echo "image     build the stage image (vLLM main + PP changes; PP_ENGINE=fork: legacy fork)"
	@echo "push      push IMAGE to its registry"
	@echo "up/down   single-host 2-stage compose demo (3 GPUs; PP_ENGINE=fork: 4)"
	@echo "smoke     wait for the compose demo and send one completion"
	@echo "cpu-demo  CPU-only end-to-end: router, fake engines, resume proxy"
	@echo "test      unit tests;  check  release scanner"
	@echo "traces    generate sample gamma traces into benchmarks/traces/"

image:
	@test "$(PP_ENGINE)" != fork || test -n "$(VLLM_FORK_REPO)" || { echo "PP_ENGINE=fork: set VLLM_FORK_REPO=<git URL of the external-PP vLLM fork>"; exit 1; }
	docker build -f docker/Dockerfile -t $(IMAGE) \
	  --build-arg PP_ENGINE=$(PP_ENGINE) --build-arg BASE_IMAGE=$(BASE_IMAGE) \
	  --build-arg VLLM_REF=$(VLLM_REF) --build-arg VLLM_INSTALL=$(VLLM_INSTALL) \
	  --build-arg VLLM_FORK_REPO=$(VLLM_FORK_REPO) \
	  --build-arg VLLM_FORK_REF=$(VLLM_FORK_REF) --build-arg FORK_INSTALL=$(FORK_INSTALL) .

push:
	docker push $(IMAGE)

up:
	CONNEX_IMAGE=$(IMAGE) $(COMPOSE) -f $(COMPOSE_FILE) up -d

down:
	$(COMPOSE) -f $(COMPOSE_FILE) --profile proxy down

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
