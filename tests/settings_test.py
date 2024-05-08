from ssi.settings import Settings
import unittest


class SettingsTest(unittest.TestCase):

    def test_settings_without_parameters(self):
        settings = Settings.load(
            "tests/test_data/settings_example.yaml", "settings_example")
        self.assertTrue(settings.first_setting)
        self.assertEqual(1, settings.second_setting)
        self.assertEqual(3, settings.third_setting)
        self.assertEqual("test", settings.fourth_setting.coicop_name)
        self.assertEqual(1, settings.fourth_setting.coicop_label)

    def test_settings_load_with_parameters(self):
        settings = Settings.load(
            "tests/test_data/settings_with_parameters.yaml", "settings", True, product_id="product123", coicop_code="01")
        self.assertTrue(settings.first_setting)
        self.assertEqual(settings.second_setting, "product123")
        self.assertEqual(settings.third_setting, 3)
        self.assertEqual(settings.fourth_setting, "01")

    def test_settings_with_nested_parameters(self):
        settings = Settings.load(
            "tests/test_data/settings_nested_parameters.yaml", "settings", True, product_id="product123", coicop_code="01")
        self.assertTrue(settings.first_setting)
        self.assertEqual(settings.second_setting, "product123")
        self.assertEqual(settings.third_setting, 3)
        self.assertEqual(settings.fourth_setting.coicop_name, "test")
        self.assertEqual(settings.fourth_setting.coicop_label, "01")
