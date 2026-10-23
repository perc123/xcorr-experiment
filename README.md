# xcorr-experiment

A small experiment showing how cross-correlation of a pure tone is ambiguous: when a periodic signal is delayed by a whole number of periods, the correlation has several equal maxima, so `np.argmax` can report the wrong delay.

## What it does

[xcorr_tie.py](xcorr_tie.py):

1. Generates a 1 kHz sine at 48 kHz sampling rate (0.5 s long).
2. Delays it by 100 ms (4800 samples, exactly 100 periods), zero-padded at the front.
3. Computes the linear cross-correlation with NumPy FFTs.
4. Prints the correlation at 0 ms and at the true delay, plus the raw `np.argmax` lag and all near-maximum samples.
5. Asserts that 0 ms and 100 ms are both (numerically) maxima, and that the first near-maximum is at 0 ms.
6. Plots the result (−50 to +150 ms) and saves it as `xcorr_tie.png`.

Because the delay is a whole number of periods, the correlation at 0 ms equals the correlation at 100 ms, and `argmax` picks the earliest one (0 ms) rather than the true delay.

## Usage

Requires Python 3 with `numpy` and `matplotlib`:

```sh
pip install numpy matplotlib
python xcorr_tie.py
```

The script prints the results to the console, writes `xcorr_tie.png`, and opens the plot.

## Parameters

Edit the constants at the top of the script:

| Name   | Default | Meaning                    |
| ------ | ------- | -------------------------- |
| `FS`   | 48000   | Sampling rate (Hz)         |
| `F`    | 1000    | Tone frequency (Hz)        |
| `D_MS` | 100     | Delay (ms)                 |
| `N`    | 24000   | Number of samples (0.5 s)  |
