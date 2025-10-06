import importlib


def test_agent_module_imports():
    mod = importlib.import_module('agent_samples.youtube_shorts_assistant.agent')
    assert hasattr(mod, 'root_agent')


def test_loop_agent_module_imports():
    mod = importlib.import_module('agent_samples.youtube_shorts_assistant.loop_agent')
    assert hasattr(mod, 'root_agent')


