#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Functions for archiving files to .zip """

from pyzipper import AESZipFile, ZipFile, BadZipFile, LargeZipFile, ZIP_DEFLATED, WZ_AES
from pathlib import Path
from typing import Iterator


secret_password = "ss"


def archiving(
    jobname: Path,
    output_file: Path,
    paths: Iterator[Path],
    secret_password: str | None = None,
) -> None:
    with AESZipFile(output_file, compression=ZIP_DEFLATED, mode="w") as archive:
        if secret_password is not None:
            secret_password = bytes(secret_password, "utf-8")
            archive.setpassword(secret_password)
            archive.setencryption(WZ_AES, nbits=128)

        try:
            for file in paths:
                archive.write(file, arcname=jobname / file.name)
        except (BadZipFile, LargeZipFile):
            print(f"Zip Archive: {output_file} failed to be created.")


if __name__ == "__main__":
    files = list(Path("../../out3").glob("*"))
    job_name = Path("transcription")
    output_file = Path("multi_fsiles_.zip")
    print(archiving(job_name, output_file, files, secret_password=secret_password))
    print(archiving(job_name, Path("multi.zip"), files))

    print("Files is in package, see structure here")

    with ZipFile(output_file, mode="r") as fp:
        fp.printdir()
