import csv
from typing import Callable, TextIO

from whisper.utils import (
    ResultWriter,
    WriteJSON,
    WriteSRT,
    WriteTSV,
    WriteTXT,
    WriteVTT,
)


# Todo usikkert på det her. Måske fjerne for at det skal virke?
class WriteCSV(ResultWriter):
    extension: str = "csv"

    def write_result(self, result: dict, file: TextIO, options: dict):
        fieldnames: list[str] = list(result["segments"][0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["segments"])


def get_writer(output_format: str, output_dir: str) -> Callable[[dict, TextIO], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
        "csv": WriteCSV,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, file: TextIO):
            for writer in all_writers:
                writer(result, file)

        return write_all

    return writers[output_format](output_dir)
