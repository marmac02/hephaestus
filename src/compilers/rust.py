import os
import re
from src.compilers.base import BaseCompiler

class RustCompiler(BaseCompiler):
    ERROR_REGEX = re.compile(
        r"error\[\w+\]: (.*?)\n\s+-->\s+(.*\.rs):(\d+:\d+)", re.MULTILINE)
    CRASH_REGEX = re.compile(r"thread '.*' panicked at '.*'")

    def __init__(self, input_name, filter_patterns=None):
        input_name = os.path.join(input_name, '*', '*.rs')
        super().__init__(input_name, filter_patterns)

    @classmethod
    def get_compiler_version(cls):
        return ['rustc', '--version']

    def get_compiler_cmd(self):
        return ['rustc', '-Awarnings', self.input_name]

    def get_filename(self, match):
        return match[1]

    def error_msg(self, match):
        return match[0]