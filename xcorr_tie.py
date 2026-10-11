import  numpy as np

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
print(f"Correlation at 0 ms: {c[idelay]:.12f}")
print(f"Correlation at 100 ms: {c[idelay]:.12f}")
print(f"np.argmax lag: {lags_ms[np.argmax(c)]:.6f} ms")
print(f"Equal maxima: {np.isclose(c[i0], c[idelay])}")

assert np.isclose(c[i0], c[idelay]), (
    "Expected equal correlation values at 0 and 100 ms."
)