"""Hand-calculated matching, metrics and CSV validation fixtures."""
from pathlib import Path
import tempfile
import unittest
import math
from helpers import comparison as c, detection as det


class ComparisonTests(unittest.TestCase):
    """Verify reference-relative matching and CSV contracts with hand-calculated fixtures."""
    def test_exact_match(self):
        """Match coincident centroids even at zero pixel tolerance."""
        self.assertEqual(c.match_centroids([det()], [det()], 0), [(0,0,0.0)])

    def test_inclusive_distance(self):
        """Accept a 3-4-5 displacement exactly at the five-pixel matching boundary."""
        self.assertEqual(c.match_centroids([det(0,0)], [det(3,4)], 5), [(0,0,5.0)])

    def test_beyond_distance(self):
        """Reject the same displacement when tolerance is below five pixels."""
        self.assertEqual(c.match_centroids([det(0,0)], [det(3,4)], 4.99), [])

    def test_one_to_one(self):
        """Prevent two reference detections from sharing one prediction."""
        self.assertEqual(len(c.match_centroids([det(),det()], [det()], 1)), 1)

    def test_maximum_cardinality_before_distance(self):
        """Prefer two valid pairs over a closer assignment that leaves a valid match unused."""
        matches = c.match_centroids([det(0,0),det(2,0)], [det(1,0),det(0,2)], 2)
        self.assertEqual({(i,j) for i,j,_ in matches}, {(0,1),(1,0)})

    def test_precision_recall_f1(self):
        """Check TP=2, FP=1, FN=2 against independent fraction calculations."""
        metrics = c.agreement(2,1,2)
        for key, value in [('precision',2/3),('recall',1/2),('f1',4/7)]:
            self.assertAlmostEqual(metrics[key+'_relative_to_rfdetr'], value)

    def test_count_errors(self):
        """Check signed bias, MAE and RMSE for frame errors of minus two and plus one."""
        _,_,metrics = c.evaluate(['a','b'], {'a':[det(),det(8,8,2)]},
                                 {'b':[det()]}, 0)
        self.assertEqual(metrics['count_mae'], 1.5)
        self.assertAlmostEqual(metrics['count_rmse'], math.sqrt(2.5))
        self.assertEqual(metrics['mean_signed_count_error'], -.5)

    def test_inclusive_iou(self):
        """Count boundary pixels when computing identical, overlapping and disjoint box IoUs."""
        a = dict(x1=0,y1=0,x2=1,y2=1)
        b = dict(x1=1,y1=0,x2=2,y2=1)
        self.assertAlmostEqual(c.bbox_iou(a,b), 1/3)
        self.assertEqual(c.bbox_iou(a,a), 1)
        self.assertEqual(c.bbox_iou(a,dict(x1=3,y1=3,x2=4,y2=4)), 0)

    def test_empty_detections_and_zero_image(self):
        """Retain a processed empty frame; undefined F1 is zero and distance is absent."""
        rows,pairs,m = c.evaluate(['zero.png'], {}, {}, 30)
        self.assertEqual(pairs, [])
        for key in ('count_mae','count_rmse','mean_signed_count_error','f1_relative_to_rfdetr'):
            self.assertEqual(m[key], 0)
        self.assertIsNone(m['mean_centroid_error_px'])
        self.assertEqual(m['count_exact_match_rate'], 1)
        self.assertEqual(rows[0]['traditional_count'], 0)

    def test_exact_filename_matching(self):
        """Treat differently capitalised basenames as different frame identities."""
        _,_,m = c.evaluate(['frame.png'], {'frame.png':[det()]}, {'Frame.png':[det()]}, 0)
        self.assertEqual(m['tp_relative_to_rfdetr'], 0)
        self.assertEqual(m['fn_relative_to_rfdetr'], 1)

    def test_invalid_thresholds(self):
        """Reject negative, infinite and NaN matching tolerances even for empty lists."""
        for value in (-1, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.match_centroids([], [], value)

    def test_csv_round_trip_and_geometry_duplicates(self):
        """Retain duplicate geometry with distinct IDs while reporting the duplicate."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'instances.csv'
            c.write_csv(path,[det(),det(apple_id=2)])
            grouped, duplicates = c.load_instances(path)
            self.assertEqual(len(grouped['frame.png']), 2)
            self.assertEqual(len(duplicates), 1)

    def test_duplicate_ids_rejected(self):
        """Reject repeated apple IDs within the same image."""
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'instances.csv'
            c.write_csv(path,[det(),det()])
            with self.assertRaisesRegex(ValueError, 'duplicate or invalid apple_id'):
                c.load_instances(path)

    def test_invalid_detection_records(self):
        """Reject nonfinite coordinates and invalid area, overlap, confidence, boxes and IDs."""
        for change in (dict(cx_px=float('nan')),dict(area_px=0),dict(branch_overlap=2),
                       dict(confidence=1.1),dict(x1=6),dict(image='folder/frame.png'),dict(apple_id=0)):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'instances.csv'
                c.write_csv(path,[dict(det(),**change)])
                with self.assertRaises(ValueError):
                    c.load_instances(path)

    def test_counts_include_zero_and_validate_discrepancy(self):
        """Preserve zero-count records and reject a conflicting instance record."""
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'counts.csv'
            c.write_csv(path,[dict(image='zero.png',kept_apple_count=0)])
            self.assertEqual(c.load_counts(path,{}), {'zero.png':0})
            with self.assertRaisesRegex(ValueError, 'discrepancy'):
                c.load_counts(path,{'zero.png':[det()]})

    def test_missing_csv_header(self):
        """Reject an empty CSV instead of treating missing headers as an empty result."""
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'empty.csv'
            path.write_text('')
            with self.assertRaisesRegex(ValueError, 'missing header'):
                c.read_csv(path)
