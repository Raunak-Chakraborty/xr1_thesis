# XR-1 Force Architecture

Status: Awaiting source-level architecture audit.

## Research objective

Introduce force as an explicitly modeled observation
modality for contact-rich assembly.

## Candidate approaches

1. Wrench through existing XR-1 state conditioning.
2. Independent force encoder with DiT fusion.
3. ForceVLA-inspired Mixture-of-Experts fusion.

## Requirements

- Preserve pretrained checkpoint compatibility.
- Preserve the force-free baseline.
- Preserve Doosan action semantics.
- Establish explicit force normalization.
- Support CPU-compatible validation.
- Measure effects of force ablation.

## Architecture evidence

To be populated from actual source-code inspection.

Do not invent file-level implementation details.
