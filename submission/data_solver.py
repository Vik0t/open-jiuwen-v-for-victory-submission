"""Conservative CSV sums from a fully parsed public instruction; no task metadata."""
import csv
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path
import re


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
HEADER = (r"The file (?P<file>[\w./-]+\.csv) has columns "
          r"(?P<columns>[a-z_]+(?:\s*,\s*[a-z_]+)*)"
          r"(?:\s*\(\d+ rows\))?\.\s*")
FOOTER = (r"(?:\s*Use bash with python3 to compute it\.)?"
          r"(?:\s*Work only inside the current directory\.)?"
          r"(?:\s*When you are done, write ONLY the final answer to the file "
          r"answer\.txt in the current directory"
          r"(?: \(for example with bash: echo " + NUMBER + r" > answer\.txt\))?\.)?\s*")


def _result(status, reason, answer=None):
    return {"status": status, "answer": answer, "reason": reason}


def _filters(text):
    """Parse conjunctions only. Every character must belong to this grammar."""
    filters = []
    while text.strip():
        text = re.sub(r"^(?:and|where)\s+", "", text.strip(), flags=re.I)
        match = re.match(r"(?:(?:in|during)\s+)?months\s+(\d+)\s+"
                         r"(?:through|to)\s+(\d+)\s+inclusive\b", text, re.I)
        if match:
            low, high = map(int, match.groups())
            if not 1 <= low <= high <= 12:
                return None
            filters.extend([("month", ">=", Decimal(low)),
                            ("month", "<=", Decimal(high))])
        else:
            match = re.match(r"(?:(?:in|during)\s+)?month\s+(\d+)\b", text, re.I)
            if match:
                month = int(match[1])
                if not 1 <= month <= 12:
                    return None
                filters.append(("month", "=", Decimal(month)))
            else:
                match = re.match(r"(?:(?:of|for|in|from)\s+)?(product|region)\s*"
                                 r"(?:(==|!=|=|is|equals)\s*)?(['\"])(.*?)\3",
                                 text, re.I)
                if match:
                    column, op, _, value = match.groups()
                    filters.append((column.lower(), "!=" if op == "!=" else "=", value))
                else:
                    match = re.match(r"(units|price|month)\s*(<=|>=|!=|==|=|<|>)\s*("
                                     + NUMBER + r")\b", text, re.I)
                    if not match:
                        return None
                    filters.append((match[1].lower(), match[2], Decimal(match[3])))
        text = text[match.end():]
    return filters


def _matches(value, op, wanted):
    if op in ("=", "=="):
        return value == wanted
    if op == "!=":
        return value != wanted
    return {"<": value < wanted, "<=": value <= wanted,
            ">": value > wanted, ">=": value >= wanted}[op]


def solve_data(instruction, workspace: Path):
    """Return solved/unknown/unsupported; never read expected or write answer.txt."""
    if not isinstance(instruction, str):
        return _result("unsupported", "instruction must be text")
    parsed = re.fullmatch(HEADER + r"(?P<query>.+?[?.])" + FOOTER,
                          instruction.strip(), re.I | re.S)
    if not parsed:
        return _result("unsupported", "unrecognized instruction structure")
    query = parsed["query"]
    units = re.fullmatch(r"How many units(.*?)\s+were sold(.*?)\?", query, re.I | re.S)
    metric = re.fullmatch(r"(?:What is|Calculate|Compute)\s+(?:the\s+)?"
                          r"(?:total\s+|sum of\s+)?(units|revenue)(.*?)[?.]",
                          query, re.I | re.S)
    if units:
        target, conditions = "units", units[1] + " " + units[2]
    elif metric:
        target, conditions = metric[1].lower(), metric[2]
    else:
        return _result("unsupported", "unrecognized sum metric")
    filters = _filters(conditions)
    if filters is None:
        return _result("unsupported", "unrecognized or invalid filter")
    try:
        root = Path(workspace).resolve(strict=True)
        name = Path(parsed["file"])
        path = (root / name).resolve(strict=True)
        if name.is_absolute() or not path.is_relative_to(root) or not path.is_file():
            return _result("unsupported", "CSV path must stay inside workspace")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            columns = reader.fieldnames or []
            required = {"units"} | {f[0] for f in filters}
            if target == "revenue":
                required.add("price")
            declared = {c.strip().lower() for c in parsed["columns"].split(",")}
            if len(columns) != len(set(columns)) or not (required | declared) <= set(columns):
                return _result("unknown", "missing or duplicate CSV columns")
            rows = list(reader)
        numeric = {"units", "price", "month"} & required
        values = [row.get(c) for row in rows for c in required]
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            return _result("unknown", "malformed CSV row")
        with localcontext() as context:
            context.prec = max(28, 4 * max((len(v) for v in values), default=0)
                               + len(str(len(rows))) + 8)
            total = Decimal(0)
            for row in rows:
                for column in numeric:
                    if not re.fullmatch(NUMBER, row[column].strip()):
                        return _result("unknown", "non-numeric CSV value")
                    row[column] = Decimal(row[column].strip())
                if all(_matches(row[c], op, wanted) for c, op, wanted in filters):
                    total += row["units"] if target == "units" else row["units"] * row["price"]
        answer = format(total, "f")
        if "." in answer:
            answer = answer.rstrip("0").rstrip(".")
        return _result("solved", "fully parsed CSV sum", "0" if total == 0 else answer)
    except (OSError, UnicodeError, csv.Error, DecimalException, ValueError, OverflowError):
        return _result("unknown", "CSV unavailable or invalid")
