# earthkit-workflows-pproc

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/sandbox_badge.svg" alt="Static Badge">
  </a>

<a href="https://codecov.io/gh/ecmwf/earthkit-workflows-pproc">
    <img src="https://codecov.io/gh/ecmwf/earthkit-workflows-pproc/branch/develop/graph/badge.svg" alt="Code Coverage">
  </a>

<a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
  </a>

<a href="https://github.com/ecmwf/earthkit-workflows-pproc/releases">
    <img src="https://img.shields.io/github/v/release/ecmwf/earthkit-workflows-pproc?color=blue&label=Release&style=flat-square" alt="Latest Release">
  </a>
</p>

> \[!IMPORTANT\]
> This software is **Sandbox** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

Earthkit-Workflows-PProc is a Python library for constructing task graphs associated to forecast post-processing products provided by PProc(https://github.com/ecmwf/pproc). These task graphs are an extension of the [earthkit-workflows](https://github.com/ecmwf/earthkit-workflows) framework, in which the graphs can be executed.

## Installation

The workflows defined in this plugin depend on PProc. Please follow the instructions in https://github.com/ecmwf/pproc for installing PProc and it's dependencies, after which this plugin can be installed via `pip` with:

```bash
pip install 'earthkit-workflows-pproc'
```

For development, you can use `pip install -e .` 
