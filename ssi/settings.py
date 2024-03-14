from yaml import load
from typing import Optional
from jinja2 import Environment
import yaml

# From: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
from typing import Any


class Settings(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def load(filename: str, settings_section: str = "settings", render_as_template: bool = False, **kwargs) -> Optional['Settings']:
        with open(filename) as yaml_filename:
            if render_as_template:
                yaml_filename = Environment().from_string(
                    yaml_filename.read()).render(**kwargs)

            settings_dict = load(yaml_filename, Loader=yaml.Loader)
            return Settings(settings_dict[settings_section])
        return None

    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return Settings(value)
        return value

    def __getitem__(self, key: Any) -> Any:
        value = self.get(key)
        if isinstance(value, dict):
            return Settings(value)
        return value
