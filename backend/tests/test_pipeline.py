import unittest
from app.pipeline import observation_time, summarize


class TrackSummaryTests(unittest.TestCase):
    def test_summary_aggregates_multiple_frames(self):
        observations = [
            {'frame_index': 0, 'seconds': 0., 'bbox': [0, 0, 10, 10], 'confidence': .8},
            {'frame_index': 1, 'seconds': .1, 'bbox': [10, 5, 20, 15], 'confidence': .9},
            {'frame_index': 2, 'seconds': .2, 'bbox': [20, 10, 30, 20], 'confidence': .7},
        ]
        summary = summarize(observations)
        self.assertEqual(summary['displacement_pixels'], [20, 10])
        self.assertEqual(summary['observation_count'], 3)
        self.assertAlmostEqual(summary['mean_detection_confidence'], .8)
        self.assertFalse(summary['quality_warnings'])
        self.assertEqual(summary['representative_observations'][0]['confidence'], .9)

    def test_empty_track_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])

    def test_single_observation_flags_insufficient_track(self):
        result = summarize([{'seconds': 0, 'bbox': [0, 0, 5, 5], 'confidence': .9}])
        self.assertEqual(result['quality_warnings'], ['short_track'])


class ObservationTimeTests(unittest.TestCase):
    def test_valid_decoder_time_preserved(self):
        self.assertEqual(observation_time(2.5, 2.4, 25, 10), (2.5, 'decoder'))

    def test_reset_cannot_move_time_backwards(self):
        seconds, source = observation_time(0, 12, 20, 10)
        self.assertAlmostEqual(seconds, 12.1)
        self.assertEqual(source, 'fps_fallback')

    def test_nonfinite_time_falls_back(self):
        seconds, source = observation_time(float('nan'), .1, 2, 10)
        self.assertAlmostEqual(seconds, .2)
        self.assertEqual(source, 'fps_fallback')

    def test_first_frame_time_zero(self):
        self.assertEqual(observation_time(0, -1, 0, 25), (0, 'decoder'))
