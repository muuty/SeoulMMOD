# ODNative Baselines

BasicTS-compatible adapters for OD-native baselines used on SeoulMMOD.

## Layout

```text
baselines/ODNative/
  MPGCN/README.md                  # mirrored upstream README
  ODCRN/README.md                  # mirrored upstream README
  ODMixer/README.md                # mirrored upstream README
  *_SeoulMMOD_District_2024_h24.py  # BasicTS configs
  arch/
    mpgcn.py                        # MPGCN 2D graph convolution core
    odcrn.py                        # ODCRN encoder-decoder core
    odmixer.py                      # ODMixer OD interaction backbone
    od_native_arch.py               # BasicTS adapters and graph builders
```

## Protocol

The adapters use the BasicTS data/scaler/loss/metric pipeline. Inputs are SeoulMMOD OD-pair tensors reshaped from `[B, T, N*N, C]` to `[B, T, N, N, C]`, and predictions are flattened back before BasicTS metric computation.

MPGCN and ODMixer are evaluated with one-step models rolled out autoregressively for the configured horizon. ODCRN keeps its native autoregressive decoder.

ODMixer is extended to multi-modal OD inputs by flattening the mode dimension into the temporal input vector (`T * modes`) before the ODMixer backbone. This keeps joint cross-mode conditioning, but it is an adaptation rather than the original single-mode upstream protocol. The current adapter still uses a zero `prev_od` placeholder because BasicTS samples do not yet provide ODMixer's auxiliary previous-period window.

## Upstream Code

The model cores were consolidated from the repository's `od_baselines/` copies:

- `od_baselines/MPGCN`
- `od_baselines/ODCRN`
- `od_baselines/ODMixer`

The original README files are mirrored under `MPGCN/`, `ODCRN/`, and `ODMixer/` for attribution and citation context.

Keep upstream attribution and license files in mind when moving or publishing these implementations.
