import pytest
from sdhtg.data.format_parser import LogFormatParser


def test_content_is_greedy():
    parser = LogFormatParser("<Date> <Level> <Content>")
    row = parser.parse_line("2024-01-01 INFO message with spaces: and colon", 1)
    assert row == {"Date": "2024-01-01", "Level": "INFO",
                   "Content": "message with spaces: and colon"}


def test_mismatch_fails():
    parser = LogFormatParser("<Date>|<Content>")
    with pytest.raises(ValueError):
        parser.parse_line("no delimiter", 3)
