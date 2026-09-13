import unittest

from ibkr_duration import duration_from_days


class IBKRDurationTests(unittest.TestCase):
    def test_days_at_or_below_ibkr_limit_remain_days(self):
        self.assertEqual(duration_from_days(1), "1 D")
        self.assertEqual(duration_from_days(365), "365 D")

    def test_durations_above_limit_are_rounded_up_to_years(self):
        self.assertEqual(duration_from_days(366), "2 Y")
        self.assertEqual(duration_from_days(500), "2 Y")
        self.assertEqual(duration_from_days(1530), "5 Y")

    def test_nonpositive_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            duration_from_days(0)


if __name__ == "__main__":
    unittest.main()
