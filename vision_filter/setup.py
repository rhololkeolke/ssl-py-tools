import os
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class ProtobufExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()


class ProtobufBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["protoc", "--version"])
        except OSError:
            raise RuntimeError(
                "Protoc must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        setup_path = Path(__file__).resolve().parent
        proto_dir = setup_path / "proto"
        output_dir = setup_path / "vision_filter" / "proto"
        protoc_args = [f"-I{proto_dir}", f"--python_out={output_dir}"]
        if not output_dir.is_dir():
            os.makedirs(output_dir)

        proto_files = list(proto_dir.rglob("*.proto"))

        # generate the protobuf python files
        subprocess.check_call(["protoc"] + protoc_args + proto_files)

        # create init files that import the messages without the need
        # for 'from *_pb2 import *' in user code
        self.__create_init_file(output_dir)

    def __create_init_file(self, path):
        # have to create this first or __init__.py will end up in glob
        # and then try to import itself
        py_files = [
            py_file for py_file in path.glob("*.py") if py_file.name != "__init__.py"
        ]

        init_file = path / "__init__.py"
        with init_file.open("w") as f:
            f.write(
                "# AUTOGENERATED by setup.py. Do not manually modify.\n# To regenerate run 'setup.py build'.\n\n"
            )

            # import from python files at this module level
            for py_file in py_files:
                f.write(f"from .{py_file.stem} import *")

            # import submodules and generate init files recursively
            subdirs = [subdir for subdir in path.iterdir() if subdir.is_dir() and subdir.name != '__pycache__']
            for subdir in subdirs:
                f.write(f"from . import {subdir.name}")
                self.__create_init_file(subdir)


setup(
    name="vision_filter",
    version="0.1.0",
    author="Devin Schwab",
    author_email="Devin Schwab <dschwab@andrew.cmu.edu>",
    description="Filtering and visualization for SSL-Vision data",
    ext_modules=[ProtobufExtension("vision_filter")],
    cmdclass=dict(build_ext=ProtobufBuild),
    zip_safe=False,
)
