# Security policy

## Supported versions

Security fixes are made on the `main` branch and in the latest release
(currently 0.1.x).

## Reporting a vulnerability

Please do not report security problems in public issues, pull requests or
discussions. Report them privately through GitHub instead: open a private
security advisory at
https://github.com/HeterInfra/connex/security/advisories/new
(the **Security** tab of this repository, then **Report a vulnerability**).

Include what is affected (component, version or commit), how to reproduce it,
and the impact you expect. We will acknowledge the report, keep you informed
while we investigate, and credit you in the advisory unless you prefer
otherwise.

## Scope

connex is research infrastructure meant to run inside a trusted cluster network.
Its control and data planes (ZeroMQ, UCX, HTTP) have no authentication or
encryption of their own; exposing their ports to untrusted networks is outside
the supported deployment model. Reports that show a way to break that
boundary, or bugs that matter even inside it (memory safety, crashes from
malformed messages), are welcome.
