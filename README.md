## Installation

1. Clone repo

2. Install required dependencies:
```bash
pip install numpy scipy
```

## Usage

### Basic Usage

Run the calculator with a configuration file:

```bash
python main.py --config config/european_call.json
```

### Command Line Options

```
python main.py --config <config_file> [options]

Options:
  --config, -c     Path to configuration JSON file (required)
  --output, -o     Path to output file (optional, prints to console if not specified)
  --format, -f     Output format: json or txt (default: json)
  --no-greeks      Skip Greeks calculation for faster computation
  --simple         Simple output (price only)
```

### Examples

1. Calc European call (output to console):
```bash
python main.py --config config/european_call.json
```

2. Calc American put and save results to JSON:
```bash
python main.py --config config/american_put.json --output results.json
```

3. Calc Asian option and save as text file:
```bash
python main.py --config config/asian_call.json --output results.txt --format txt
```

4. Quick price calc without Greeks:
```bash
python main.py --config config/barrier_down_out_call.json --no-greeks --simple
```

## Configuration File Format

Config files are in JSON format. Here's an example for a European call option:

```json
{
  "option_style": "european",
  "option_type": "call",
  "underlying_price": 100.0,
  "strike_price": 105.0,
  "time_to_maturity": 0.5,
  "volatility": 0.2,
  "risk_free_rate": 0.05,
  "dividend_yield": 0.02
}
```

### Required Parameters

- `option_style`: Type of option - "european", "american", "asian", or "barrier"
- `option_type`: "call" or "put"
- `underlying_price`: Current price of the underlying (S)
- `strike_price`: Strike price (K)
- `time_to_maturity`: Time to expiration in years (T)
- `volatility`: Volatility of the underlying (σ)
- `risk_free_rate`: Risk-free rate (r)

### Optional Parameters

- `dividend_yield`: Continuous dividend yield (q) (default: 0)
- `num_simulations`: Number of Monte Carlo simulations (default: 10000)
- `num_steps`: Number of time steps in simulation (default: 252)

### Option-Specific Parameters

**Asian Options:**
- `average_type`: "arithmetic" or "geometric" - default: "arithmetic"

**Barrier Options:**
- `barrier_type`: "up-and-out", "up-and-in", "down-and-out", or "down-and-in" (required)
- `barrier_level`: Price level of the barrier (required)
