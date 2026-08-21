# Security Policy

## Supported Versions

We release security updates and bug fixes for actively maintained versions of Shiva and the CSCP connector.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2.0 | :x:                |

---

## Reporting a Vulnerability

If you discover a security vulnerability, memory safety issue, or data corruption flaw in Shiva or CSCP, **please do not report it in a public issue**.

Instead, report it responsibly via one of the following channels:

* **GitHub Security Advisory**: Open a draft security advisory via the repository's [Security Tab](../../security/advisories/new).
* **Private Email**: Send the details to `adityadhruva2003@gmail.com`.

### What to Include

To help us triage and resolve the issue quickly, include:

1. **Description**: A clear summary of the vulnerability and its potential impact.
2. **Affected Component**: Specify the module involved (e.g., `protocol/`, `ffi.rs`, shared memory IPC, Seqlock, C/C++/Python bindings).
3. **Reproduction Steps**: A minimal reproducible example, proof of concept (PoC), or script demonstrating the flaw.
4. **Environment**: OS/architecture, Rust toolchain version, and compiler versions for any C/C++ bindings used.

---

## Areas of Critical Concern

Given Shiva's focus on low-latency, real-time physical control and foreign function interfaces (FFI), we prioritize vulnerabilities in:

* **Memory Safety & FFI (`ffi.rs`, `bindings/`)**: Undefined behavior, use-after-free, pointer misalignments, or buffer overflows across language boundaries.
* **Shared Memory & IPC (`protocol/`)**: Race conditions, torn reads/writes across asynchronous reader/writer processes, or memory corruption in the lock-free Seqlock.
* **Denial of Service (DoS) in Control Loops**: Non-deterministic execution stalls or panic loops that could disrupt real-time hardware execution deadlines.

---

## Response & Disclosure Process

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within **48 hours**.
2. **Assessment & Triage**: Maintainers will investigate, reproduce, and determine severity within **5 business days**.
3. **Fix & Verification**: A patch will be developed in a private branch and tested against regression suites and benchmarks.
4. **Coordinated Disclosure**: Once patched and released in a minor/patch version, we will publish a public advisory crediting your discovery (unless requested otherwise).

---

## Deployment Security Best Practices

When deploying Shiva and CSCP on physical systems:

* **Shared Memory Permissions**: Restrict POSIX shared memory access permissions (`/dev/shm`) to authorized user groups to prevent unauthorized processes from reading or injecting actuator commands.
* **Hardware E-Stop Mechanisms**: Always maintain hardwired, physical fail-safe mechanisms independent of software control loops for safety-critical robotics.
