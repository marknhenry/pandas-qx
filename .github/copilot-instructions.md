# Copilot Instructions for pandas-qx

This is a pandas extension library for quantitative/data analytics functions.
Follow these conventions when writing or modifying code in this repo.

## General Python

- Always use type hints on function signatures.
- Use NumPy-style docstrings (`Parameters`, `Returns` sections).
- Only comment code that genuinely needs clarification — avoid redundant comments.

## Code Review

- Always call `.copy()` on the input DataFrame at the start of a function to avoid mutating the caller's data.
- Validate inputs early and raise `ValueError` or `TypeError` with clear messages.
  ```python
  if returns_col not in df.columns:
      raise ValueError(f"Column '{returns_col}' not found in DataFrame.")
  if not pd.api.types.is_numeric_dtype(df[returns_col]):
      raise TypeError(f"Column '{returns_col}' must be numeric.")
  ```
- Prefix all internally generated helper columns with `_q_` so they can be identified and filtered out easily.
- Keep core computation in plain NumPy functions (no pandas dependency), then wrap in a pandas function for labeling. This keeps logic testable without pandas.
- Expose logic as both a standalone function and a pandas accessor method. The accessor must delegate to the standalone function — never duplicate logic.
- Use `df.aggregate()` when a function should work on both `pd.Series` and `pd.DataFrame`.
- Use vectorized pandas operations (no loops, no iterrows)
- Prefer method chaining (.assign, .query, .groupby)
- Use .pipe() for reusable transformations
- Avoid inplace operations
- Write functions as DataFrame -> DataFrame
- Keep functions small and composable


## Testing

- Use `pytest` as the test runner. Run tests with `python -m pytest tests/ -v`.
- Organise tests to mirror source modules: `tests/test_qx_accessor.py`, `tests/test_stats_accessor.py`, `tests/test_data_loads.py`.
- Group related tests in classes (e.g., `class TestDrawdown`).
- Use `conftest.py` for shared fixtures (e.g., reusable DataFrames).
- Every function must have tests covering:
  1. **Return type** — result is a `pd.DataFrame` or expected type.
  2. **No mutation** — input is not modified (check original columns before/after).
  3. **Output structure** — expected columns are present; internal `_q_` columns are absent from outputs where appropriate.
  4. **Correctness** — at least one deterministic assertion on computed values (use `pytest.approx` for floats).
  5. **Edge cases** — invalid column name raises `KeyError`/`ValueError`, bad types raise `TypeError`.
- Use `pd.testing.assert_frame_equal` and `pd.testing.assert_series_equal` for structural comparisons.
- Accessor methods must have a test asserting they produce the same result as the underlying standalone function.
- Test assumptions about data must reflect reality (e.g., historical returns can exceed 100% in a single month — test means, not maximums).

## Project Structure

- Standalone functions live in the relevant module (e.g., `qx_accessor.py`) and are exported via `__init__.py`.
- Accessor classes are registered with `@pd.api.extensions.register_dataframe_accessor(...)`.
- The accessor method calls the standalone function — it does not reimplement it.
