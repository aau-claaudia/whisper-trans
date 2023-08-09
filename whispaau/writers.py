import csv
from typing import TextIO
import json

from whisper.utils import ResultWriter, format_timestamp


class WriteCSV(ResultWriter):
    extension: str = "csv"

    def write_result(self, result: dict, file: TextIO, options: dict):
        fieldnames: list[str] = list(result["segments"][0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["segments"])


class WriteDOTE(ResultWriter):
    extension: str = "dote.json"

    def format_result(self, result: dict):
        interface = {"lines": []}
        for line in result["segments"]:
            line_add = {
                "startTime": format_timestamp(line["start"], True),
                "endTime": format_timestamp(line["end"], True),
                "speakerDesignation": "",
                "text": line["text"].strip(),
            }
            interface["lines"].append(line_add)
        else:
            return interface

    def write_result(self, result: dict, file: TextIO, options: dict):
        result = self.format_result(result)
        json.dump(result, file)
