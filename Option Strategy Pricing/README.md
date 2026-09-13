# Option Strategy Analyzer

This project prices European and American vanilla call and put options and analyzes a few common two-leg strategies:

- Bull call spread
- Bear call spread
- Bull put spread
- Bear put spread
- Straddle
- Strangle

## What Was Added

The project now includes:

- safer handling for zero-volatility and expiry-day pricing
- corrected Greeks for gamma, theta, and vega
- an implied volatility solver from observed market prices
- American option pricing with a CRR binomial tree
- clearer command-line parsing and validation
- exact strategy metrics for max profit, max loss, and breakeven
- a CMake build
- a regression test executable

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run

The analyzer expects two options with the same spot and expiry:

```bash
./build/strategy_analyzer \
  100 95 0.25 0.5 0.01 0.04 C \
  100 105 0.25 0.5 0.01 0.04 C
```

Arguments are:

```text
spot strike volatility years_to_expiry dividend_yield risk_free_rate option_type [exercise_style]
```

Option type must be `C` or `P`. Exercise style may be `E` or `A` and defaults to `E`.

You can also append two observed market prices to recover implied volatilities:

```bash
./build/strategy_analyzer \
  100 100 0.20 1.0 0.00 0.05 C \
  100 100 0.20 1.0 0.00 0.05 P \
  E E 10.4506 5.5735
```

In that mode the tool prints the market price and the implied volatility for each option.
If both legs have valid implied volatilities, the strategy section also switches to market premiums and IV-implied Greeks.

You can price American contracts by inserting styles after the option types:

```bash
./build/strategy_analyzer \
  100 95 0.25 0.5 0.01 0.04 C \
  100 105 0.25 0.5 0.01 0.04 P \
  A A
```

## Example Strategies

Bull call spread:

```bash
./build/strategy_analyzer \
  100 95 0.25 0.5 0.01 0.04 C \
  100 105 0.25 0.5 0.01 0.04 C
```

Bear put spread:

```bash
./build/strategy_analyzer \
  100 95 0.25 0.5 0.01 0.04 P \
  100 105 0.25 0.5 0.01 0.04 P
```

Straddle:

```bash
./build/strategy_analyzer \
  100 100 0.25 0.5 0.01 0.04 C \
  100 100 0.25 0.5 0.01 0.04 P
```

Strangle:

```bash
./build/strategy_analyzer \
  100 110 0.25 0.5 0.01 0.04 C \
  100 90 0.25 0.5 0.01 0.04 P
```

## Good Next Augmentations

- payoff and Greek charts across spot prices
- Monte Carlo pricing for path-independent options
- CSV input for screening many trades at once
- a small web UI with interactive sliders
