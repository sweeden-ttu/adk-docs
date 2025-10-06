import os
import importlib


def test_load_instruction_from_file_reads_existing(tmp_path, monkeypatch):
    pkg_path = os.path.dirname(importlib.import_module('agent_samples.youtube_shorts_assistant.util').__file__)
    test_file = os.path.join(pkg_path, 'scriptwriter_instruction.txt')
    assert os.path.exists(test_file)

    util = importlib.import_module('agent_samples.youtube_shorts_assistant.util')
    text = util.load_instruction_from_file('scriptwriter_instruction.txt')
    assert isinstance(text, str)
    assert len(text) > 0


