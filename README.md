# Cyberdelia Samplers

Custom sampling methods for [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), based on the Ralston RK2 (Runge-Kutta 2nd order) method.

## Samplers

### Cyberdelia Ralston (RK2)

A crisp 2nd-order Runge-Kutta sampler using Ralston's method — an optimal RK2 variant that minimizes truncation error.

Each step evaluates the model twice:

1. **k1** at the current sigma `s0`
2. **k2** at an interior point `s0 + ⅔(s1 − s0)`

Then combines them with the optimal Ralston weights: **¼·k1 + ¾·k2**.

This produces cleaner, more stable results than Euler while keeping things simple (no ancestral noise, no SDE paths). Good general-purpose sampler for 20–30+ steps.

### Cyberdelia LCM Ralston (RK2)

An LCM-style denoise → re-noise loop, stabilized with a Ralston RK2 blended prediction. Designed for **LCM / DMD2-style LoRAs** where you want few steps (4–8) at low CFG, but with cleaner output than plain LCM sampling.

The denoise prediction is a blend between a single-eval LCM prediction and the full Ralston RK2 prediction, controlled by `rk2_blend`.

#### Tunable parameters (via `extra_args`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cd_lcm_s_noise` | `1.0` | Noise scale on re-noise step |
| `cd_lcm_eps` | `1e-8` | Sigma floor clamp |
| `cd_lcm_rk2_blend` | `0.5` | 0 = plain LCM, 1 = full Ralston RK2 stabilization |

## Installation

Copy the `cyberdelia_samplers` folder into your WebUI's `extensions/` directory and restart. Both samplers will appear in the sampler dropdown.

Alternatively, clone directly:

```bash
cd extensions
git clone https://github.com/YOUR_USERNAME/cyberdelia-samplers.git cyberdelia_samplers
```

## Requirements

No additional dependencies — uses only PyTorch and the standard WebUI sampler API.

## License

MIT
