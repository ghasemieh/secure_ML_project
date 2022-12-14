"""
The configuration service loads a configuration file into a
ConfigParser through which configuration values can be accessed

The ``ACTIVE_PROFILE`` environment variable must be set to
a valid profile because there is no default configuration

A valid profile is one for which a ``drive-{profile}.config`` file
exists in the project root
"""

import os
from configparser import ConfigParser


class __Config:
    __path_to_config = 'drive-{}.config'

    __env_var_name = 'ACTIVE_PROFILE'

    __error_no_active_profile_set = """

        Cannot continue without an active configuration profile. 

        To set an active profile, add the environment variable 

            'ACTIVE_PROFILE'

        A valid profile is one for which a 'drive-{profile}.config file
        exists in the project root
        """

    __error_no_file_matching_active_profile = """

        The active profile is set to '{}' but no file named '{}' exists in '{}'
        """

    @property
    def get(self) -> ConfigParser:
        cp = ConfigParser()
        cp.read(self.__path_to_config.format('dev'))
        return cp


__config = __Config()


def get() -> ConfigParser:
    return __config.get
