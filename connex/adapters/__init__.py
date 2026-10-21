# SPDX-License-Identifier: Apache-2.0
"""Engine adapters on top of the engine-neutral core (connex.pp).

Each subpackage imports its engine only in its own modules, so
``import connex.adapters.<engine>`` works without that engine installed.
"""
