# Contributing to Invisible Physics

Thank you for helping improve Invisible Physics. Contributions that make the numerical methods more accurate, the visualizations clearer, or the learning experience more useful are welcome.

## Development setup

1. Fork the repository and clone your fork.
2. Create a virtual environment with Python 3.10 or newer.
3. Install the package and development dependencies:

   ```bash
   python -m pip install -e ".[dev]"
   ```

4. Run the test suite before making changes:

   ```bash
   pytest
   ```

## Making changes

- Keep public APIs documented and preserve SI units throughout the physics modules.
- Add or update focused tests for numerical behavior and input validation.
- Keep notebook changes reproducible and avoid committing generated checkpoints.
- Use clear, focused commits and explain any scientific assumptions in the relevant documentation.

Before opening a pull request, run:

```bash
pytest
python examples/example_experiments.py --no-show
```

## Pull requests

Pull requests should describe the problem being solved, summarize the approach, and include validation results. Changes to formulas or numerical algorithms should include a reference to the governing equation or method.

## Reporting issues

Please include the Python version, operating system, installation method, a minimal reproducible example, and the full traceback when reporting a bug.
