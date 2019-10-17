import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

import pkg_resources
from grpc_tools.protoc import main as protoc_main
from protobuf_gen.error import TranspilerError
from protobuf_gen.patch import \
    rename_protobuf_imports as rename_from_protobuf_imports
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class ProtobufExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = Path(sourcedir).resolve()


def _build_pb_with_prefix(
    module_prefix: str,
    root_output_dir: str,
    includes: List[str],
    input_proto: List[str],
):
    protoc_output_dir = os.path.join(root_output_dir, module_prefix.replace(".", "/"))
    os.makedirs(protoc_output_dir, exist_ok=True)

    args = ["__main__"]

    includes = [pkg_resources.resource_filename("grpc_tools", "_proto")] + includes

    args += ["-I" + x for x in includes]

    args += [
        f"--python_out={protoc_output_dir}",
        f"--grpc_python_out={protoc_output_dir}",
    ]
    args += [f"--python_grpc_out={protoc_output_dir}"]

    args += [x for x in input_proto]

    r = protoc_main(args)

    if r != 0:
        raise TranspilerError(f"protoc returned {r}, check tool output")

    rename_from_protobuf_imports(protoc_output_dir, module_prefix)
    rename_protobuf_imports(protoc_output_dir, module_prefix)


def rename_protobuf_imports(
    dir_root: str, root: str, do_not_replace: Optional[List[str]] = None
):
    do_not_replace = do_not_replace or ["google.protobuf"]
    pattern = re.compile(r"^import ([^ ]+)_pb2$")

    print("Patching 'import *_pb2' statements")
    for path, _, files in os.walk(dir_root):
        init_file = os.path.join(path, "__init__.py")

        if not os.path.exists(init_file):
            with open(init_file, "w+"):
                pass

        for file in files:
            if not file.endswith(".py"):
                continue

            with open(os.path.join(path, file), "r") as f:
                lines = list(f.readlines())

            changes = 0

            replacements = []
            with open(os.path.join(path, file), "w+") as f:
                for line in lines:
                    match = pattern.match(line)
                    if (
                        match
                        and ".".join(match.group(1).split(".")[:-1])
                        not in do_not_replace
                    ):
                        changes += 1
                        # convert foo.bar syntax to foo_dot_bar
                        new_name = match.group(1).replace(".", "_dot_")
                        f.write(
                            f"import {root}.{match.group(1)}_pb2 as {new_name}__pb2\n"
                        )
                        replacements.append(
                            (f"{match.group(1)}_pb2", f"{new_name}__pb2")
                        )
                    else:
                        # There is probably a better way of doing this
                        for find, replace in replacements:
                            line = line.replace(find, replace)
                        f.write(line)

            print("Patched", file, changes)


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
        if not output_dir.is_dir():
            os.makedirs(output_dir)

        proto_files = [str(proto_file) for proto_file in proto_dir.rglob("*.proto")]

        _build_pb_with_prefix(
            "vision_filter.proto", setup_path, [str(proto_dir)], proto_files
        )


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
