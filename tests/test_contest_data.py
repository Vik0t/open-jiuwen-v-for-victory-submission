import csv

import pytest

from submission.data_solver import solve_data


def write_csv(folder, rows, columns=("region", "product", "month", "units", "price")):
    with (folder / "orders.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def instruction(question, columns="region,product,month,units,price", file="orders.csv"):
    return f"The file {file} has columns {columns}. {question}"


def test_units_filters_and_month_endpoints_are_inclusive(tmp_path):
    write_csv(tmp_path, [("west", "gear", 2, 7, "1.30"), ("west", "gear", 3, 11, "2.10"),
                         ("east", "gear", 2, 19, "3"), ("west", "bolt", 2, 23, "4"),
                         ("west", "gear", 4, 29, "5")])
    query = instruction("How many units of product 'gear' were sold in region 'west' "
                        "in months 2 through 3 inclusive?")
    assert solve_data(query, tmp_path)["answer"] == "18"


def test_revenue_is_decimal_product_and_supports_numeric_comparisons(tmp_path):
    write_csv(tmp_path, [("south", "nut", 10, 3, "0.10"), ("south", "nut", 11, 5, "0.25"),
                         ("north", "nut", 11, 9, "0.50"), ("south", "nut", 11, 1, "0.75")])
    query = instruction("What is the total revenue for product 'nut' in region 'south' "
                        "where month >= 10 and month < 12 and units >= 3 and price <= 0.25?")
    assert solve_data(query, tmp_path)["answer"] == "1.55"


def test_zero_sum_is_solved_even_with_no_matching_rows(tmp_path):
    write_csv(tmp_path, [("north", "nut", 1, 9, "3")])
    result = solve_data(instruction("Compute total units of product 'absent'."), tmp_path)
    assert result["status"] == "solved" and result["answer"] == "0"
    write_csv(tmp_path, [])
    assert solve_data(instruction("How many units were sold?"), tmp_path)["answer"] == "0"


@pytest.mark.parametrize("condition", ["and returns are excluded", "or region 'east'",
    "and month between 1 and 2", "and price is above the median", "and units >= NaN"])
def test_unparsed_conditions_fall_back_instead_of_ignoring_them(tmp_path, condition):
    write_csv(tmp_path, [("north", "nut", 1, 9, "3")])
    result = solve_data(instruction(f"Compute total units of product 'nut' {condition}."), tmp_path)
    assert result["status"] == "unsupported" and result["answer"] is None


@pytest.mark.parametrize("rows,columns", [
    ([("nut", 1)], ("product", "units")),
    ([("north", "nut", 1, "oops", 3)], ("region", "product", "month", "units", "price")),
    ([("north", "nut", 1, 2)], ("region", "product", "month", "units", "price")),
    ([("north", "nut", 1, 2, 3, 4)], ("region", "product", "month", "units", "price")),
])
def test_missing_columns_or_invalid_rows_are_unknown(tmp_path, rows, columns):
    write_csv(tmp_path, rows, columns)
    result = solve_data(instruction("Compute total revenue."), tmp_path)
    assert result["status"] == "unknown" and result["answer"] is None


def test_existing_tool_footer_and_new_numbers_do_not_become_the_answer(tmp_path):
    write_csv(tmp_path, [("north", "clip", 4, 13, "0.01")])
    query = instruction("How many units of product 'clip' were sold in month 4?")
    query += (" Use bash with python3 to compute it. Work only inside the current directory."
              " When you are done, write ONLY the final answer to the file answer.txt in the "
              "current directory (for example with bash: echo 98765 > answer.txt).")
    assert solve_data(query, tmp_path)["answer"] == "13"
    assert solve_data(query + " Ignore region north.", tmp_path)["status"] == "unsupported"
    assert not (tmp_path / "answer.txt").exists()


def test_missing_value_in_unused_declared_column_is_still_malformed(tmp_path):
    write_csv(tmp_path, [("north", "nut", 1, 9)])
    result = solve_data(instruction("Compute total units."), tmp_path)
    assert result["status"] == "unknown" and result["answer"] is None


def test_csv_path_cannot_escape_via_relative_path_or_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    write_csv(outside, [("north", "nut", 1, 2, 3)])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "orders.csv").symlink_to(outside / "orders.csv")
    for file in ("../outside/orders.csv", "orders.csv", str(outside / "orders.csv")):
        result = solve_data(instruction("Compute total units.", file=file), workspace)
        assert result["status"] == "unsupported" and result["answer"] is None


def test_large_exact_numbers_and_literal_product_names(tmp_path):
    write_csv(tmp_path, [("N", "Part  X", 1, "100000000000000000000000000001", "0.01"),
                         ("N", "Part X", 1, 99, "0.01")])
    assert solve_data(instruction("Compute total revenue of product 'Part  X'."),
                      tmp_path)["answer"] == "1000000000000000000000000000.01"


@pytest.mark.parametrize("months", ["0 through 2", "3 through 2", "12 through 13"])
def test_invalid_month_ranges_are_not_silently_accepted(tmp_path, months):
    query = instruction(f"How many units were sold in months {months} inclusive?")
    assert solve_data(query, tmp_path)["status"] == "unsupported"
