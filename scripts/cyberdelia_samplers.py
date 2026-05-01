# cyberdelia_samplers.py — Cyberdelia custom samplers for Forge / A1111
#
# Two Ralston RK2-based samplers in one extension:
#
#   1. Cyberdelia Ralston (RK2)
#      A crisp 2nd-order Runge-Kutta sampler using Ralston's method.
#      k1 at s0, k2 at s0 + 2/3*(s1-s0), weighted 1/4 + 3/4.
#
#   2. Cyberdelia LCM Ralston (RK2)
#      LCM-style denoise→re-noise loop, stabilized with a Ralston RK2
#      blended prediction. Works well with LCM / DMD2-style LoRAs at
#      few steps + low CFG.
#
#      Optional extra_args:
#        cd_lcm_s_noise:    float  (default 1.0)   noise scale on re-noise
#        cd_lcm_eps:        float  (default 1e-8)   sigma clamp
#        cd_lcm_rk2_blend:  0..1   (default 0.0)    1 = full RK2, 0 = plain LCM
#                                                    Raise only if you see
#                                                    instability with non-DMD2
#                                                    LoRAs or odd schedulers.

from __future__ import annotations

import torch
from modules import sd_samplers, sd_samplers_common
from modules.sd_samplers_kdiffusion import KDiffusionSampler

__VER__ = "2.1"
_TAG = "[Cyberdelia Samplers]"
print(f"{_TAG} v{__VER__} loaded:", __file__)


# ---------------------------------------------------------------------------
#  Shared utilities
# ---------------------------------------------------------------------------

def _normalize_sigmas(sigmas, device, dtype) -> torch.Tensor:
    if sigmas is None:
        return None
    if not torch.is_tensor(sigmas):
        sigmas = torch.tensor(sigmas, device=device, dtype=dtype)
    else:
        sigmas = sigmas.to(device=device, dtype=dtype)
    if sigmas.ndim != 1:
        sigmas = sigmas.flatten()
    return sigmas.contiguous()


def _to_d(x: torch.Tensor, sigma, denoised: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    else:
        sigma = sigma.to(device=x.device, dtype=x.dtype)

    sigma = sigma.clamp_min(eps)

    # Per-batch [B] → [B,1,1,1] for NCHW broadcast
    if sigma.ndim == 1:
        sigma = sigma.view(-1, 1, 1, 1)

    return (x - denoised) / sigma


def _nan_guard(x: torch.Tensor, tag: str = "", step: int = -1) -> torch.Tensor:
    if torch.isnan(x).any() or torch.isinf(x).any():
        if tag:
            print(f"{_TAG} NaN/Inf detected in {tag} at step {step} — clamping, generation may be corrupt")
        x = torch.nan_to_num(x)
    return x


def _resolve_sigmas(sigmas, kwargs):
    """Get sigmas from the argument or common Forge keyword variants."""
    if sigmas is None:
        sigmas = kwargs.get("sigmas") or kwargs.get("sigma_sched")
    if sigmas is None:
        raise ValueError(f"{_TAG} missing sigmas schedule")
    return sigmas


def _filter_cd_args(ea: dict) -> dict:
    """Strip cd_* keys so they don't leak into the model call."""
    if not ea:
        return {}
    return {k: v for k, v in ea.items() if not str(k).startswith("cd_")}


def _register_unique(label: str, func, aliases=None, options=None) -> None:
    aliases = list(dict.fromkeys(aliases or []))
    options = options or {}

    existing_names = {s.name for s in sd_samplers.all_samplers}
    existing_aliases = {a for s in sd_samplers.all_samplers for a in (getattr(s, "aliases", []) or [])}

    if label in existing_names or any(a in existing_aliases for a in aliases):
        print(f"{_TAG} '{label}' already registered (or alias collision). Skipping.")
        return

    ctor = (lambda m, f=func: KDiffusionSampler(f, m))
    sdata = sd_samplers_common.SamplerData(label, ctor, aliases, options)

    sd_samplers.all_samplers.append(sdata)
    sd_samplers.all_samplers_map = {s.name: s for s in sd_samplers.all_samplers}

    if hasattr(sd_samplers, "set_samplers"):
        sd_samplers.set_samplers()

    print(f"{_TAG} registered sampler: {label}")


# ---------------------------------------------------------------------------
#  Sampler 1 — Cyberdelia Ralston (RK2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_cyberdelia_ralston(
    model,
    x: torch.Tensor,
    *,
    sigmas=None,
    extra_args=None,
    callback=None,
    disable=False,
    **kwargs,
):
    sigmas = _resolve_sigmas(sigmas, kwargs)
    device, dtype = x.device, x.dtype
    sigmas = _normalize_sigmas(sigmas, device, dtype)

    steps = int(sigmas.numel() - 1)
    if steps <= 0:
        return x

    ea = extra_args or {}
    s_in = torch.ones((x.shape[0],), device=device, dtype=dtype)

    def _cb(i, s0, s_r, s1, denoised):
        if callback is None:
            return
        try:
            callback({"i": i, "sigma": s0, "sigma_hat": s_r, "sigma_next": s1, "x": x, "denoised": denoised})
        except TypeError:
            callback(i, x, s0, s1)

    for i in range(steps):
        s0 = sigmas[i]
        s1 = sigmas[i + 1]
        h = s1 - s0  # negative: sigmas decrease toward 0

        # Denoise at s0
        den0 = model(x, s0 * s_in, **ea)

        # If last step to (near-)zero, do a stable Euler finalize
        if float(s1) < 1e-6:
            d0 = _to_d(x, s0, den0)
            x = x + h * d0
            _cb(i, s0, s1, s1, den0)
            continue

        d0 = _to_d(x, s0, den0)

        # Ralston interior point: 2/3 of the way from s0 to s1
        s_r = torch.clamp(s0 + (2.0 / 3.0) * h, min=1e-8)
        x_r = x + (2.0 / 3.0) * h * d0

        den_r = model(x_r, s_r * s_in, **ea)
        d_r = _to_d(x_r, s_r, den_r)

        # Weighted RK2 combination (optimal 1/4 + 3/4 Ralston weights)
        x = x + h * (0.25 * d0 + 0.75 * d_r)
        x = _nan_guard(x, "Ralston", i)

        _cb(i, s0, s_r, s1, den_r)

    return x


# ---------------------------------------------------------------------------
#  Sampler 2 — Cyberdelia LCM Ralston (RK2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_cyberdelia_lcm_ralston(
    model,
    x: torch.Tensor,
    *,
    sigmas=None,
    extra_args=None,
    callback=None,
    disable=False,
    **kwargs,
):
    sigmas = _resolve_sigmas(sigmas, kwargs)
    device, dtype = x.device, x.dtype
    sigmas = _normalize_sigmas(sigmas, device, dtype)

    steps = int(sigmas.numel() - 1)
    if steps <= 0:
        return x

    ea = extra_args or {}
    ea_model = _filter_cd_args(ea)

    s_noise = float(ea.get("cd_lcm_s_noise", 1.0))
    eps = float(ea.get("cd_lcm_eps", 1e-8))
    rk2_blend = max(0.0, min(1.0, float(ea.get("cd_lcm_rk2_blend", 0.0))))
    use_rk2 = rk2_blend > 1e-6

    s_in = torch.ones((x.shape[0],), device=device, dtype=dtype)
    eps_t = torch.tensor(eps, device=device, dtype=dtype)

    def _cb(i, s0, s_r, s1, den):
        if callback is None:
            return
        try:
            callback({"i": i, "sigma": s0, "sigma_hat": s_r, "sigma_next": s1, "x": x, "denoised": den})
        except TypeError:
            pass

    for i in range(steps):
        s0 = sigmas[i]
        s1 = sigmas[i + 1]

        # First denoise
        den0 = model(x, s0 * s_in, **ea_model)

        if use_rk2:
            # Ralston RK2 stabilization of the denoise prediction
            sigma_r = torch.maximum(s0 + (2.0 / 3.0) * (s1 - s0), eps_t)

            d0 = _to_d(x, s0, den0, eps=eps)
            x_r = x + (2.0 / 3.0) * (s1 - s0) * d0

            den_r = model(x_r, sigma_r * s_in, **ea_model)

            # Ralston weights: 0.25/0.75 on denoise predictions, blended
            den_hat_rk2 = 0.25 * den0 + 0.75 * den_r
            den_hat = (1.0 - rk2_blend) * den0 + rk2_blend * den_hat_rk2
        else:
            # Plain LCM path — skip the second model call entirely
            sigma_r = s0
            den_hat = den0

        # Final step: if sigma_next is 0, return denoised (no re-noise)
        if float(s1) == 0.0:
            x = _nan_guard(den_hat, "LCM Ralston", i)
            _cb(i, s0, sigma_r, s1, den_hat)
            break

        # LCM loop: re-noise to the next sigma
        if s_noise > 0.0:
            x = den_hat + torch.randn_like(x) * (s1 * s_noise)
        else:
            x = den_hat

        x = _nan_guard(x, "LCM Ralston", i)
        _cb(i, s0, sigma_r, s1, den_hat)

    return x


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

_register_unique(
    label="Cyberdelia Ralston (RK2)",
    func=sample_cyberdelia_ralston,
    aliases=["cyberdelia-ralston", "cd-ralston", "rk2-ralston"],
    options={
        "scheduler": "auto",
        "uses_ensd": False,
        "second_order": True,
    },
)

_register_unique(
    label="Cyberdelia LCM Ralston (RK2)",
    func=sample_cyberdelia_lcm_ralston,
    aliases=["cyberdelia-lcm-rk2", "cd-lcm-rk2", "lcm-ralston-cd"],
    options={
        "scheduler": "auto",
        "second_order": True,
    },
)
