"""
Test config options.
"""

from aesara.configparser import THEANO_FLAGS_DICT, AddConfigVar, ConfigParam


class TestConfig:
    def test_invalid_default(self):
        # Ensure an invalid default value found in the Aesara code only causes
        # a crash if it is not overridden by the user.

        def filter(val):
            if val == "invalid":
                raise ValueError()
            else:
                return val

        try:
            # This should raise a ValueError because the default value is
            # invalid.
            AddConfigVar(
                "T_config.test_invalid_default_a",
                doc="unittest",
                configparam=ConfigParam("invalid", filter=filter),
                in_c_key=False,
            )
            raise AssertionError()
        except ValueError:
            pass

        THEANO_FLAGS_DICT["T_config.test_invalid_default_b"] = "ok"
        # This should succeed since we defined a proper value, even
        # though the default was invalid.
        AddConfigVar(
            "T_config.test_invalid_default_b",
            doc="unittest",
            configparam=ConfigParam("invalid", filter=filter),
            in_c_key=False,
        )

        # Check that the flag has been removed
        assert "T_config.test_invalid_default_b" not in THEANO_FLAGS_DICT

        # TODO We should remove these dummy options on test exit.
