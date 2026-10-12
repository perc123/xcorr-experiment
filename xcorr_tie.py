import numpy as np
import matplotlib.pyplot as plt

FS = 48_000
F = 1_000
D_MS = 100
D = int(D_MS * FS / 1000)  # delay in samples
N = int(0.5 * FS)  # number of samples

# original signal
t = np.arange(N) / FS
x = np.sin(2 * np.pi * F * t)

# delayed signal; zero padded at the front; fixed output length
y = np.concatenate((np.zeros(D), x[:-D]))

# linear cross-correlation using NumPy FFTs
nfft = 1 << (2 * N - 2).bit_length()  # next power of 2
r = np.fft.irfft(
    np.fft.rfft(y, nfft) * np.conj(np.fft.rfft(x, nfft)),
    nfft,
)
c = np.concatenate((r[-(N - 1):], r[:N]))
lags = np.arange(-(N - 1), N)
lags_ms = lags / FS * 1000

# inspect the zero-delay and truedelay correlation values
i0 = N - 1
idelay = i0 + D

print(f"Sampling rate: {FS} Hz")
print(f"Tone frequency: {F} Hz")
print (f"True delay: {D_MS} ms ({D} samples)")
print(f"Correlation at 0 ms: {c[i0]:.12f}")
print(f"Correlation at 100 ms: {c[idelay]:.12f}")


# Raw numerical maximum
imax = np.argmax(c)
print(f"Raw np.argmax lag: {lags_ms[imax]:.6f} ms")
print(f"Raw maximum: {c[imax]:.12f}")

# Find all values numerically indistinguishable from the maximum
tol = 1e-8
near_max = np.isclose(c, c.max(), rtol=1e-12, atol=tol)
first_near_max = np.flatnonzero(near_max)[0]

print(
    "First near-maximum lag: "
    f"{lags_ms[first_near_max]:.6f} ms"
)
print(f"Number of near-maximum samples: {near_max.sum()}")

assert near_max[i0], "Zero lag should be a near-maximum"
assert near_max[idelay], "100 ms should be a near-maximum"
assert lags_ms[first_near_max] == 0.0

print(f"Equal maxima: {np.isclose(c[i0], c[idelay])}")



assert np.isclose(c[i0], c[idelay]), (
    "Expected equal correlation values at 0 and 100 ms."
)


fig, ax = plt.subplots(figsize=(12, 6.27))

# Restrict the view to -50 to +150 ms
window = (lags_ms >= -50) & (lags_ms <= 150)
ax.plot(
    lags_ms[window],
    c[window],
    linewidth=0.8,
    color="#3678A8",
    label="Cross-correlation",
)

# Highlight the two equal maxima
ax.scatter(
    [0, D_MS],
    [c[i0], c[idelay]],
    color="#D9534F",
    s=65,
    zorder=5,
    label="Equal maxima",
)
ax.axhline(c[i0], color="#D9534F", linestyle="--", linewidth=1)

ax.annotate(
    "argmax reports 0 ms",
    xy=(0, c[i0]),
    xytext=(-38, c[i0] * 0.88),
    arrowprops=dict(arrowstyle="->", color="#D9534F"),
    fontsize=12,
)
ax.annotate(
    "true delay: 100 ms",
    xy=(D_MS, c[idelay]),
    xytext=(108, c[idelay] * 0.88),
    arrowprops=dict(arrowstyle="->", color="#D9534F"),
    fontsize=12,
)

ax.set(
    title="Two equal maxima: which lag wins?",
    xlabel="Lag (ms)",
    ylabel="Cross-correlation",
    xlim=(-50, 150),
)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="lower left")

fig.tight_layout()
fig.savefig("xcorr_tie.png", dpi=200, bbox_inches="tight")
plt.show()