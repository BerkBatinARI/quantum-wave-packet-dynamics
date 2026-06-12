# Quantum Wave Packet Dynamics in Structured Potentials

This project studies the numerical time evolution of one-dimensional quantum wave packets using the time-dependent Schrödinger equation. The aim is not only to produce animations, but to use simulation as a way of analysing how quantum states spread, scatter, tunnel, and interfere under controlled potential landscapes.

The project begins with the free evolution of a Gaussian wave packet and then extends toward potential barriers, wells, and double-barrier structures. Each stage is designed to connect three parts of the same problem: the underlying physics, the numerical method, and the interpretation of the results.

## Research Motivation

Wave-packet dynamics provides a useful bridge between formal quantum mechanics and computational physics. Instead of treating scattering only through stationary-state calculations, this project follows the real-time evolution of a localised quantum state and studies how probability density changes as the packet encounters different potentials.

This makes it possible to investigate questions such as:

- how dispersion changes the shape of a free quantum wave packet;
- how reflection and transmission emerge during barrier scattering;
- how tunnelling appears in the time-dependent probability density;
- how structured potentials can produce interference and resonant behaviour.

## Scientific Goals

The main goals of this project are to:

- implement a reproducible Python framework for one-dimensional wave-packet evolution;
- compare free-particle spreading with scattering in structured potentials;
- analyse probability conservation as a numerical consistency check;
- visualise the relationship between the wave function, probability density, and potential energy landscape;
- build a research-style workflow connecting theory, computation, and interpretation.

## Current Stage

The current stage focuses on constructing the numerical framework and testing it on the free evolution of a Gaussian wave packet. Later stages will introduce potential barriers and double-barrier systems to study tunnelling and resonance effects.

## Methods

The project will use finite-difference methods and matrix-based time evolution to approximate the one-dimensional time-dependent Schrödinger equation. Numerical results will be checked through probability conservation, stability behaviour, and comparison with physical expectations.

## Tools

- Python
- NumPy
- SciPy
- Matplotlib
- Jupyter