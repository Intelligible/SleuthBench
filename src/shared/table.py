"""Small text-table renderer shared by pipeline CLIs."""

from __future__ import annotations


def print_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    max_last_col: int | None = None,
) -> None:
    """Print aligned rows, optionally truncating the final column."""
    rendered_rows = [list(row) for row in rows]
    if max_last_col is not None:
        for row in rendered_rows:
            if row and len(row[-1]) > max_last_col:
                row[-1] = row[-1][: max_last_col - 3] + "..."

    col_widths = [len(header) for header in headers]
    for row in rendered_rows:
        for index, cell in enumerate(row):
            col_widths[index] = max(col_widths[index], len(cell))

    def format_row(cells: list[str]) -> str:
        return "  " + "  ".join(
            cell.ljust(col_widths[index])
            for index, cell in enumerate(cells)
        )

    print(format_row(headers))
    print("  " + "  ".join("-" * width for width in col_widths))
    for row in rendered_rows:
        print(format_row(row))
