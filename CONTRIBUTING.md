# Contributing to Shiva

Thank you for your interest in contributing to Shiva. Shiva is an open-source real-time control and multi-engine reinforcement learning framework built in Rust with polyglot bindings for C, C++, and Python.

We welcome contributions of all kinds: bug fixes, performance optimizations, new policy engines, algorithmic improvements, documentation updates, and hardware/simulator bindings.

---

## Code of Conduct

Please help us keep this project open, respectful, and welcoming. We expect all contributors and maintainers to adhere to standard open-source collaboration etiquette.

---

## Getting Started

### Prerequisites

To build and test the full repository, ensure you have the following installed:

* **Rust**: Latest stable toolchain (`rustup install stable`)
* **Cargo Tools**: `clippy` and `rustfmt` (`rustup component add clippy rustfmt`)
* **C/C++ Toolchain**: `gcc` / `clang`, `cmake`, and a C++17 compliant compiler
* **Python**: Python 3.9+ with `pip`, `venv`, and `numpy`

### Local Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/shiva.git
   cd shiva

```

2. **Build the Rust Core:**
```bash
cargo build --all-targets

```


3. **Run the Test Suite:**
```bash
cargo test

```


4. **Test Polyglot Examples:**
```bash
# Run Rust example
cargo run --example basic_control_loop

# Test Python bindings
cd bindings/python
pip install -e .
python ../../examples/python_example.py

```



---

## Repository Architecture

Familiarize yourself with the workspace layout before making changes:

```
shiva/
├── bindings/             # Foreign Function Interfaces
│   ├── c/                # C header (shiva.h)
│   ├── cpp/              # C++ header-only wrapper (shiva.hpp)
│   └── python/           # Python ctypes / C-ABI wrappers (shiva.py)
├── examples/             # Reference loops for Rust, C, C++, and Python
├── src/
│   ├── algorithms/       # Core math & RL (TD3, IQN, SAC, RND, CPO)
│   ├── brain/            # High-level domain facades & trait definitions
│   ├── environment/      # Actuator signals, environment matrix abstractions
│   ├── nodes/            # Multi-engine consensus & orchestration nodes
│   ├── protocol/         # Zero-copy CSCP shared-memory transport
│   ├── ffi.rs            # C-ABI export functions
│   └── lib.rs            # Framework root export
└── tests/                # Integration and regression tests

```

---

## Contribution Workflow

### 1. Reporting Issues

* Search existing issues to verify your bug or feature request has not already been logged.
* When reporting bugs, include:
* Operating system and architecture (`uname -a`).
* Rust/Python/C++ compiler version.
* Minimal reproducible example reproducing the bug or memory fault.



### 2. Making Changes

* **Branch Naming**: Use descriptive branch names:
* `feat/add-diffusion-policy`
* `fix/seqlock-torn-read`
* `docs/update-c-abi-guide`


* **Atomicity**: Keep commits focused and logically grouped. Write clear commit messages.

### 3. Code Standards & Style

#### Rust

* Format all code with `rustfmt` before committing:
```bash
cargo fmt --all

```


* Ensure no `clippy` warnings remain:
```bash
cargo clippy --all-targets --all-features -- -D warnings

```


* File names must use `snake_case` (e.g., `implicit_quantile_networks.rs`).
* Maintain zero-allocation guarantees on hot execution paths (`mothership.rs`, `protocol/`).

#### C / C++

* Keep header interfaces clean and strictly compliant with C99 / C++17 standards.
* Avoid adding heavy external dependencies to the FFI boundary.

#### Python

* Adhere to PEP 8. Format Python code using `black` and `ruff`.
* Type annotations (`typing`) are strongly encouraged across all public SDK methods.

---

## Performance & Safety Benchmarks

Shiva guarantees sub-10 microsecond state-action roundtrips. Any pull request touching `protocol/`, `ffi.rs`, or the consensus orchestrator (`mothership.rs`) must not introduce latency regressions:

* Run benchmarks before and after changes:
```bash
cargo bench

```


* If modifying lock-free Seqlocks or shared memory structures, verify safety against data races across threads and processes.

---

## Submitting a Pull Request (PR)

1. Ensure all tests and lint checks pass locally:
```bash
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test

```


2. Push your branch to your fork.
3. Open a Pull Request against the `main` branch.
4. Provide a clear description of:
* What was changed or added.
* Why the change is necessary.
* How the change was tested (unit tests, benchmarks, hardware/sim validation).


5. Address any code review feedback from maintainers.

---

## License

By contributing to Shiva, you agree that your contributions will be licensed under the project's [MIT License](https://www.google.com/search?q=LICENSE).
