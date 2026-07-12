from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


FIELD = re.compile(r"<([^<>]+)>")


@dataclass
class LogFormatParser:
    log_format: str

    def __post_init__(self) -> None:
        fields = FIELD.findall(self.log_format)
        if not fields:
            raise ValueError("log_format must contain at least one <Field>")
        self.fields = fields
        pieces = []
        position = 0
        for match in FIELD.finditer(self.log_format):
            literal = self.log_format[position:match.start()]
            pieces.append(self._literal_pattern(literal))
            name = match.group(1)
            pieces.append(fr"(?P<{name}>.*?)")
            position = match.end()
        pieces.append(self._literal_pattern(self.log_format[position:]))
        # Make the last field greedy so Content may contain separators.
        pattern = "".join(pieces)
        last = fields[-1]
        pattern = pattern.replace(fr"(?P<{last}>.*?)", fr"(?P<{last}>.*)", 1)
        self.regex = re.compile("^" + pattern + "$", re.ASCII)

    @staticmethod
    def _literal_pattern(value: str) -> str:
        if not value:
            return ""
        parts = re.split(r"(\s+)", value)
        return "".join(r"\s+" if x.isspace() else re.escape(x) for x in parts if x)

    def parse_line(self, line: str, line_number: int) -> dict[str, str]:
        match = self.regex.match(line.rstrip("\r\n"))
        if not match:
            excerpt = line.rstrip()[:240]
            raise ValueError(f"line {line_number} does not match log_format: {excerpt!r}")
        return {key: value.strip() for key, value in match.groupdict().items()}

    def parse_file(self, path: Path, encoding: str) -> Iterator[dict[str, str]]:
        with path.open("r", encoding=encoding, errors="strict") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = self.parse_line(line, number)
                row["source_line"] = number
                yield row
