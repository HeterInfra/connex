#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
# Block until every stage of pipeline PREFIX is serving: stage 0's readiness
# probe passes and stages >0 have entered the external-PP loop.
# usage: wait_ready.sh PREFIX [TIMEOUT_S] [EXTRA_LABEL_SELECTOR e.g. replica=1]
# env: CONNEX_NAMESPACE (default connex)
NS=${CONNEX_NAMESPACE:-connex}
P=$1; T=${2:-900}; SEL=${3:+,$3}; end=$((SECONDS + T))
while [ $SECONDS -lt $end ]; do
  ok=1
  for pod in $(kubectl -n "$NS" get pods -l pipeline=$P$SEL -o name 2>/dev/null); do
    st=$(kubectl -n "$NS" get $pod -o jsonpath='{.metadata.labels.stage}')
    if [ "$st" = 0 ]; then
      kubectl -n "$NS" get $pod --no-headers | grep -q "1/1" || ok=0
    else
      kubectl -n "$NS" logs $pod 2>/dev/null | grep -qE "entering tight pipeline loop|CONNEX_PP_STAGE_READY" || ok=0
    fi
  done
  [ "$(kubectl -n "$NS" get pods -l pipeline=$P$SEL --no-headers 2>/dev/null | wc -l)" -gt 0 ] || ok=0
  [ $ok = 1 ] && { echo "pipeline $P ready"; exit 0; }
  sleep 5
done
echo "pipeline $P NOT ready after ${T}s"; exit 1
