import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import apple_predictor_v1 as detector
import compare_apple_detectors as compare


def apple(x,y=0):
    return dict(image='a.png',apple_id=1,confidence='',area_px=1,branch_overlap=1,
                cx_px=x,cy_px=y,x1=x,y1=y,x2=x,y2=y,red_score=1)


class EvaluationTests(unittest.TestCase):
    def test_exact_threshold_and_one_to_one(self):
        self.assertEqual(len(compare.match_centroids([apple(0)],[apple(10)],10)),1)
        self.assertEqual(compare.match_centroids([apple(0)],[apple(10.01)],10),[])
        self.assertEqual(len(compare.match_centroids([apple(0),apple(1)],[apple(0)],10)),1)

    def test_cardinality_before_distance(self):
        # r0 can use either prediction; r1 can use only p0.
        matches=compare.match_centroids([apple(0),apple(-9)],[apple(-1),apple(9)],10)
        self.assertEqual({(i,j) for i,j,d in matches},{(0,1),(1,0)})

    def test_empty_and_metrics(self):
        self.assertEqual(compare.match_centroids([],[],10),[])
        rows,pairs,s=compare.evaluate({'zero','missing'},{'missing':[apple(0)]},{},10)
        self.assertEqual(s['count_mae'],.5)
        self.assertAlmostEqual(s['count_rmse'],np.sqrt(.5))
        self.assertEqual(s['mean_signed_count_error'],-.5)
        self.assertEqual(s['fn_relative_to_rfdetr'],1)
        self.assertEqual(s['precision_relative_to_rfdetr'],0)
        self.assertEqual(compare.agreement(2,1,2)['f1_relative_to_rfdetr'],4/7)
        self.assertEqual(compare.agreement(2,1,2)['precision_relative_to_rfdetr'],2/3)
        self.assertEqual(compare.agreement(2,1,2)['recall_relative_to_rfdetr'],.5)

    def test_exact_filenames_and_row_order(self):
        ref={'a.png':[apple(0)],'b.png':[]}
        pred={'b.png':[],'a.png':[apple(0)]}
        _,_,s=compare.evaluate(set(ref)&set(pred),ref,pred,0)
        self.assertEqual(s['tp_relative_to_rfdetr'],1)
        self.assertEqual(set(ref)&{'A.png'},set())

    def test_duplicates_validation_counts_and_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'instances.csv'
            a=apple(0); b=dict(a,apple_id=2)
            compare.write_csv(p,[a,b])
            grouped,duplicates=compare.load_instances(p)
            self.assertEqual(len(duplicates),1)
            self.assertEqual(len(grouped['a.png']),2)
            c=Path(tmp)/'counts.csv'
            compare.write_csv(c,[dict(image='a.png',kept_apple_count=2),dict(image='zero.png',kept_apple_count=0)])
            self.assertEqual(compare.load_counts(c,grouped)['zero.png'],0)
            compare.write_csv(c,[dict(image='a.png',kept_apple_count=1)])
            with self.assertRaisesRegex(ValueError,'discrepancy'):
                compare.load_counts(c,grouped)
            compare.write_csv(p,[a,a])
            with self.assertRaisesRegex(ValueError,'duplicate'):
                compare.load_instances(p)
            compare.write_csv(p,[dict(a,cx_px=float('nan'))])
            with self.assertRaises(ValueError):
                compare.load_instances(p)
            p.write_text('image\na.png\n')
            with self.assertRaisesRegex(ValueError,'missing columns'):
                compare.load_instances(p)

    def test_export_coordinate_centroid_and_csv(self):
        rgb=np.zeros((40,50,3),dtype=np.uint8)
        rgb[10:20,20:30,0]=255
        depth=np.ones((40,50),dtype=np.uint8)
        gt=np.ones((40,50),dtype=np.uint8)*255
        candidates=detector.find_apple_candidates(rgb,depth,gt)
        self.assertEqual(len(candidates),1)
        ys,xs=np.nonzero(detector.clean_red_mask(detector.detect_red_mask(rgb,depth)))
        result=dict(image='a.png',instances=candidates,image_width=50,image_height=40,
                    tree='R01N01',season='Summer',time='t',apple_count=1,annotated_path='a.png')
        row=list(detector.instance_rows(result))[0]
        self.assertEqual((row['x1'],row['y1'],row['x2'],row['y2']),
                         (xs.min(),ys.min(),xs.max(),ys.max()))
        self.assertEqual(row['cx_px'],xs.mean())
        self.assertEqual(row['cy_px'],ys.mean())
        self.assertEqual(row['confidence'],'')
        self.assertEqual(compare.bbox_iou(row,row),1)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(detector,'OUTPUT_ROOT',Path(tmp)),patch.object(detector,'CSV_PATH',Path(tmp)/'apple_counts.csv'):
                detector.save_csv([result])
                groups,_=compare.load_instances(Path(tmp)/'apple_instances.csv')
                self.assertEqual(compare.load_counts(Path(tmp)/'apple_counts.csv',groups),{'a.png':1})

    def test_original_detector_regression_all_frames(self):
        original=subprocess.check_output(['git','show','HEAD:src/apple_predictor_v1.py'],cwd=ROOT).decode('utf-8')
        namespace={'__file__':str(ROOT/'src/apple_predictor_v1.py'),'__name__':'original_detector'}
        exec(compile(original,'original_detector','exec'),namespace)
        frames=detector.find_all_frames()
        self.assertTrue(frames)
        for path,info in frames:
            depth_path,gt_path=detector.get_matching_files(path,info)
            rgb=detector.load_rgb(path); depth=detector.load_depth(depth_path)
            gt=detector.load_gt(gt_path,depth.shape)
            expected=namespace['find_apple_candidates'](rgb,depth,gt)
            actual=detector.find_apple_candidates(rgb,depth,gt)
            self.assertEqual(expected,actual,path.name)

    def test_actual_instance_count_consistency(self):
        for directory in ['apple_detection','apple_rfdetr_boxes_only']:
            path=ROOT/'outputs'/directory
            instances,_=compare.load_instances(path/'apple_instances.csv')
            counts=compare.load_counts(path/'apple_counts.csv',instances)
            self.assertTrue(counts)

if __name__=='__main__':
    unittest.main()
