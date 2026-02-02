# Tests

How to run
- `pytest`
- `pytest --cov=src --cov-report=term-missing --cov-report=html`
- `pytest -m unit`
- `pytest -m integration`

Notes
- Tests mock all external network calls (Gemini, embeddings, Chroma).
- Assumptions: build prompt includes source markers in the context section (e.g., `Source 1:`).
- Coverage configuration lives in `pytest.ini` and enforces >= 85% coverage.
- To run with coverage, ensure `pytest-cov` is installed in your environment.
