import argparse
import csv
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", nargs="+", type=str, default="", help="File for transcribring"
)
args = parser.parse_args()


files = [Path(file) for file in args.input if Path(file).exists()]

files = [
    "./out/Opfølgning (på workshop med SDU kollegerne)-20221004_141208-Meeting Recording_large.csv",
    "./out/Opfølgning (på workshop med SDU kollegerne)-20221004_142017-Meeting Recording_large.csv",
]

for file in files:
    print(file)
    with open(file, "r") as fp, open(file[:-4] + ".srt", "w") as sub_fp:
        csvreader = csv.DictReader(fp)
        for idx, row in enumerate(csvreader):
            time_from: datetime = datetime.utcfromtimestamp(float(row["start"]))
            time_to: datetime = datetime.utcfromtimestamp(float(row["end"]))
            # print(time_from, time_to)

            tid_fra = time_from.strftime("%H:%M:%S.%f")
            tid_til = time_to.strftime("%H:%M:%S.%f")

            # print(row["id"], tid_fra, tid_til, row["text"])

            section = (
                f"""{row["id"]}\n{tid_fra} --> {tid_til}\n{row["text"].strip()}\n\n"""
            )
            sub_fp.write(section)
