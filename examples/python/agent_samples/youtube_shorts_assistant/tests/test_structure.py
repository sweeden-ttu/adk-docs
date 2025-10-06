import importlib


def test_agent_exposes_subagents():
    mod = importlib.import_module('agent_samples.youtube_shorts_assistant.agent')
    assert hasattr(mod, 'scriptwriter_agent')
    assert hasattr(mod, 'visualizer_agent')
    assert hasattr(mod, 'formatter_agent')


