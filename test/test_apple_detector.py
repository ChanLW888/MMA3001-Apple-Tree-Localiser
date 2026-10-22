"""Synthetic RGB-D detector and temporary export tests."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from helpers import detector as d


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        self.rgb[4:8, 5:9] = [200, 10, 10]
        self.depth = np.ones((16, 16), dtype=np.uint16)
        self.gt = np.ones((16, 16), dtype=np.uint8)
        context = patch.multiple(d, MORPH_KERNEL_SIZE=1, BRANCH_DILATION=1,
                                 MIN_APPLE_AREA=4)
        context.start()
        self.addCleanup(context.stop)

    def candidates(self):
        return d.find_apple_candidates(self.rgb, self.depth, self.gt)

    def test_red_dominance_and_non_red(self):
        rgb = np.array([[[200,10,10], [10,200,10], [10,10,200],
                         [50,0,0], [150,100,0], [255,255,255]]], dtype=np.uint8)
        np.testing.assert_array_equal(d.detect_red_mask(rgb, np.ones((1,6))),
                                      [[1,0,0,0,0,0]])

    def test_invalid_depth(self):
        rgb = np.full((1,3,3), [200,0,0], dtype=np.uint8)
        np.testing.assert_array_equal(d.detect_red_mask(rgb, np.array([[0,-1,1]])), [[0,0,1]])

    def test_morphology_removes_isolated_noise(self):
        mask = np.zeros((17,17), dtype=np.uint8)
        mask[4:13,4:13] = 1
        mask[1,1] = 1
        with patch.object(d, 'MORPH_KERNEL_SIZE', 3):
            cleaned = d.clean_red_mask(mask)
        self.assertEqual(cleaned[1,1], 0)
        self.assertEqual(cleaned[8,8], 1)
        self.assertEqual(cleaned.dtype, np.uint8)

    def test_morphology_closes_internal_hole(self):
        mask = np.zeros((17,17), dtype=np.uint8)
        mask[3:14,3:14] = 1
        mask[8,8] = 0
        with patch.object(d, 'MORPH_KERNEL_SIZE', 3):
            self.assertEqual(d.clean_red_mask(mask)[8,8], 1)

    def test_minimum_area_boundary(self):
        with patch.object(d, 'MIN_APPLE_AREA', 16):
            self.assertEqual(len(self.candidates()), 1)
        with patch.object(d, 'MIN_APPLE_AREA', 17):
            self.assertEqual(self.candidates(), [])

    def test_bbox(self):
        self.assertEqual(self.candidates()[0]['bbox'], (5,4,4,4))

    def test_centroid(self):
        self.assertEqual(self.candidates()[0]['centroid'], (6.5,5.5))

    def test_branch_rejection(self):
        self.gt[:] = 0
        self.assertEqual(self.candidates(), [])

    def test_branch_dilation(self):
        self.gt[:] = 0
        self.gt[8,8] = 1
        with patch.object(d, 'BRANCH_DILATION', 3):
            zone = d.create_branch_zone(self.gt)
        self.assertEqual(int(zone.sum()), 5)
        self.assertEqual(zone[8,9], 1)
        self.assertEqual(zone[8,10], 0)

    def test_shape_validation(self):
        with self.assertRaisesRegex(ValueError, 'aligned'):
            d.find_apple_candidates(self.rgb, self.depth[:2], self.gt)

    def test_zero_apples(self):
        self.rgb[:] = 0
        self.assertEqual(self.candidates(), [])
        self.assertEqual(list(d.instance_rows(dict(instances=[]))), [])

    def test_instance_csv_export_and_zero_count(self):
        result = dict(image='frame.png', instances=self.candidates(), image_width=16,
                      image_height=16, tree='R1N1', season='Summer', time='12',
                      apple_count=1, annotated_path='frame.png', read_ms=1,
                      detection_ms=2, save_ms=3, total_ms=6)
        zero = dict(result, image='zero.png', instances=[], apple_count=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.multiple(d, OUTPUT_ROOT=root, COUNTS_CSV=root/'counts.csv',
                                INSTANCES_CSV=root/'instances.csv', TIMINGS_CSV=root/'times.csv'):
                d.save_csv([result, zero])
            with (root/'instances.csv').open(newline='') as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, d.INSTANCE_FIELDS)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual([rows[0][k] for k in ('apple_id','cx_px','cy_px','x1','y1','x2','y2')],
                             ['1','6.5','5.5','5','4','8','7'])
            self.assertEqual(rows[0]['confidence'], '')
            with (root/'counts.csv').open(newline='') as stream:
                self.assertEqual([r['Apple_Count'] for r in csv.DictReader(stream)], ['1','0'])
