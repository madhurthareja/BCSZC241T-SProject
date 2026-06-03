Video Watermarking (DCT Spread-Spectrum)

This project implements a keyed, DCT-domain spread-spectrum watermark for video frames.
The current implementation focuses on embedding robustness with perceptual safeguards
and deterministic, key-based coefficient selection.

Algorithm Overview

1) Key conditioning
- Input key string K is hashed with SHA-256.
- The 256-bit digest is converted into a binary vector b_i in {0,1}.
- Bits are mapped to a bipolar watermark sequence w_i in {-1,+1} via:

	w_i = 2 * b_i - 1

- The sequence can be repeated to increase length (repeat factor r), producing
	L = 256 * r elements. See key helpers in [src/vidmark/core/algorithms.py](src/vidmark/core/algorithms.py).

2) Frame-level deterministic randomness
- A deterministic PRNG is seeded from the key hash.
- For each frame index t, a mixed seed is derived so coefficient selection is
	reproducible at embed and detect time.

3) Luminance-only processing
- Each input frame is converted to YCrCb, and only the luminance (Y) channel is modified.
- Chrominance channels (Cr, Cb) pass through untouched to reduce visible artifacts.

4) Block DCT
- The Y plane is split into 8x8 blocks.
- Each block is centered by subtracting 128.
- A 2D DCT is applied to each block.

5) Mid-frequency coefficient selection
- A fixed mid-frequency candidate set is used (avoids DC/low and very high frequencies).
- A keyed, per-frame shuffle selects a subset of size M (default 6) from the candidates
	for each block. This reduces energy injection while preserving spread-spectrum behavior.

6) Adaptive embedding strength (perceptual masking)
- For each block, standard deviation sigma is computed.
- Smooth blocks are skipped if sigma < sigma_floor to avoid visible artifacts.
- Adaptive strength is computed and capped:

	alpha_k = min(alpha * (sigma / sigma_scale), alpha_cap)

Defaults are tuned to reduce banding and gray patches on real videos.

7) Embedding rule
- For each selected mid-frequency coefficient C(u,v):

	C'(u,v) = C(u,v) + alpha_k * w_i

- The watermark sequence is consumed sequentially across blocks and coefficients.

8) Inverse DCT and reconstruction
- Each modified block is inverse-transformed and shifted back by +128.
- The Y plane is clipped to [0,255] and converted to uint8.
- The frame is reassembled from Y, Cr, and Cb and returned as BGR.

Detection (implemented, not calibrated)

Detection extracts the same coefficient locations and computes a normalized correlation:

	T = sum_i (X_i - mean(X)) * w_i
	    / (||X - mean(X)|| * ||w||)

The current code returns a per-frame score and averages across frames in
[src/vidmark/utils/result.py](src/vidmark/utils/result.py). Thresholding and calibration
are not yet formalized.

Implementation Details

Primary algorithm class:
- [src/vidmark/core/algorithms.py](src/vidmark/core/algorithms.py)
	- DctSpreadSpectrumWatermark: embedding + detection
	- key_to_bipolar_sequence / expand_sequence helpers

Embedding pipeline:
- [src/vidmark/core/watermarker.py](src/vidmark/core/watermarker.py)
	- Watermarker.embed(): frame iteration and algorithm dispatch

Key Parameters (current defaults)

- block_size: 8
- max_positions (M): 6
- sigma_floor: 5.0
- sigma_scale: 128.0
- alpha_cap: 0.08
- strength -> alpha mapping:
	- low: 0.03
	- medium: 0.06
	- high: 0.10

Parameter Effects

- Higher alpha increases robustness but raises visible distortion.
- Lower max_positions reduces artifacts but may reduce detectability.
- Higher sigma_floor avoids smooth areas (reduces flicker / banding).
- Lower alpha_cap prevents large coefficient spikes in high-variance blocks.

Limitations and Open Work

- Detection threshold tau is not calibrated.
- No PSNR/SSIM or ROC evaluation harness is included yet.
- Block alignment assumes no cropping or scaling.
- Video encoding settings (codec, bitrate) can affect artifact levels.

Test Notes

- The end-to-end MP4 detection test with the correct key can fail on some systems because MP4 encoding (mp4v) can attenuate the DCT-domain watermark in small synthetic clips. See [tests/test_watermarking.py](tests/test_watermarking.py) for the test setup.
- Passed locally: deterministic sequence generation, medium-strength transparency check, incorrect-key detection, and multi-frame aggregation stability.

Screenshot-friendly pytest output (verbose with test names):

```bash
pytest -vv --color=yes | tee pytest_output.txt
```
