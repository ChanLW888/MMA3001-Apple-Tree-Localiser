"""Lightweight grid tests: no dataset access or production main invocation."""
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from helpers import grid as g, detector as d, comparison as c


class GridTests(unittest.TestCase):
    def test_full_cartesian_grid(self):
        combinations=list(g.combinations())
        expected=math.prod(map(len,g.GRID.values()))
        self.assertEqual(expected,5184)
        self.assertEqual(len(combinations),expected)
        self.assertEqual(len({run for run,_ in combinations}),expected)
        self.assertEqual(len({tuple(params.items()) for _,params in combinations}),expected)
        for key,values in g.GRID.items():
            self.assertEqual({params[key] for _,params in combinations},set(values))

    def test_baseline_matches_actual_detector_and_occurs_once(self):
        self.assertEqual(g.BASELINE,{key:getattr(d,key) for key in g.GRID})
        self.assertEqual(sum(params == g.BASELINE for _,params in g.combinations()),1)

    def test_only_traditional_parameters_vary(self):
        self.assertEqual(set(g.GRID), {'RED_THRESHOLD','RED_DOMINANCE','MIN_APPLE_AREA',
            'MAX_APPLE_AREA','MIN_CIRCULARITY','MAX_ASPECT_RATIO','MORPH_KERNEL_SIZE'})
        self.assertNotIn('BRANCH_DILATION',g.GRID)
        self.assertNotIn('MIN_BRANCH_OVERLAP',g.GRID)
        self.assertEqual((d.BRANCH_DILATION,d.MIN_BRANCH_OVERLAP),(100,.05))

    def test_tiny_cartesian_configuration(self):
        with patch.object(g,'GRID',{'RED_THRESHOLD':[30,50],'MORPH_KERNEL_SIZE':[3,5]}):
            self.assertEqual(list(g.combinations()),[
                ('grid_0000',{'RED_THRESHOLD':30,'MORPH_KERNEL_SIZE':3}),
                ('grid_0001',{'RED_THRESHOLD':30,'MORPH_KERNEL_SIZE':5}),
                ('grid_0002',{'RED_THRESHOLD':50,'MORPH_KERNEL_SIZE':3}),
                ('grid_0003',{'RED_THRESHOLD':50,'MORPH_KERNEL_SIZE':5})])

    def test_objective_directions(self):
        mapping={'f1':'f1_relative_to_rfdetr','precision':'precision_relative_to_rfdetr',
                 'recall':'recall_relative_to_rfdetr','mae':'count_mae','rmse':'count_rmse',
                 'centroid':'mean_centroid_error_px','bbox_iou':'mean_matched_bbox_iou'}
        for metric,key in mapping.items():
            with self.subTest(metric=metric):
                low,high=g.score({key:1},metric),g.score({key:2},metric)
                self.assertEqual(high>low,metric in ('f1','precision','recall','bbox_iou'))
        self.assertGreater(g.score({'mean_signed_count_error':-.1},'count_bias'),
                           g.score({'mean_signed_count_error':2},'count_bias'))

    def test_invalid_numeric_scores_rank_last(self):
        for value in ('',None,float('nan'),float('inf')):
            self.assertEqual(g.numeric({'x':value},'x'),float('-inf'))
            self.assertEqual(g.numeric({'x':value},'x',False),float('-inf'))

    def test_baseline_wins_tie(self):
        baseline=dict(g.BASELINE,run_id='z',f1_relative_to_rfdetr=.5)
        other=dict(baseline,run_id='a',RED_THRESHOLD=30)
        self.assertIs(g.select_best([other,baseline],'f1'),baseline)

    def test_tree_partition_is_grouped_and_deterministic(self):
        groups={}
        for tree in range(100):
            parts={g.tree_partition(f'R1N{tree}_Summer_RGB-{time}.png') for time in range(3)}
            self.assertEqual(len(parts),1)
            groups[tree]=parts.pop()
        self.assertEqual(set(groups.values()),{'development','test'})
        self.assertFalse({t for t,p in groups.items() if p=='test'} &
                         {t for t,p in groups.items() if p=='development'})

    def test_result_persistence_schema_and_resume_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'results.csv'
            self.assertEqual(g.load_rows(path),[])
            for run,params in list(g.combinations())[:2]:
                g.append_row(path,dict(run_id=run,split='development',**params,count_mae=1.5))
            rows=g.load_rows(path)
            self.assertEqual({r['run_id'] for r in rows},{'grid_0000','grid_0001'})
            self.assertEqual(list(rows[0]),g.SUMMARY_FIELDS)
            self.assertEqual(float(rows[0]['count_mae']),1.5)
            for key,value in next(g.combinations())[1].items():
                self.assertEqual(float(rows[0][key]),value)
            path.write_text('wrong\n')
            with self.assertRaisesRegex(ValueError,'incompatible columns'):
                g.load_rows(path)

    def test_configuration_evaluation_records_metrics_without_outputs(self):
        predictor=SimpleNamespace(find_apple_candidates=Mock(return_value=[]),
                                  instance_rows=d.instance_rows,BRANCH_DILATION=100,
                                  MIN_BRANCH_OVERLAP=.05)
        rgb=np.zeros((2,2,3),dtype=np.uint8)
        metrics,duration=g.evaluate_configuration({'RED_THRESHOLD':30},
            [('zero.png',{},rgb,np.ones((2,2)),np.ones((2,2)))],['zero.png'],
            predictor,c,{},30)
        self.assertEqual(predictor.RED_THRESHOLD,30)
        self.assertEqual((predictor.BRANCH_DILATION,predictor.MIN_BRANCH_OVERLAP),(100,.05))
        self.assertEqual(metrics['count_mae'],0)
        self.assertGreaterEqual(duration,0)
        predictor.find_apple_candidates.assert_called_once()

    def test_missing_dynamic_module(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            g.load_module(Path(tmp)/'missing.py','missing_grid_test')
    def test_changed_objective_reuses_heldout_known_limitation(self):
        """Reproduce stale held-out reuse with tiny mocked data and temporary files."""
        import contextlib
        import io
        import json
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            tiny={key:[value] for key,value in g.BASELINE.items()}
            fake=SimpleNamespace(**g.BASELINE,BRANCH_DILATION=100,MIN_BRANCH_OVERLAP=.05)
            fake.find_all_frames=lambda: [(Path('dev.png'),{}),(Path('held.png'),{})]
            fake.load_rgb=lambda path: np.zeros((2,2,3),dtype=np.uint8)
            fake.get_matching_files=lambda path,info: (path,path)
            fake.load_depth=lambda path: np.ones((2,2))
            fake.load_gt=lambda path,shape: np.ones(shape)
            comparator=SimpleNamespace(load_instances=lambda path: ({},[]),
                load_counts=lambda path,reference: {'dev.png':0,'held.png':0})
            metrics={key:0 for key in g.SUMMARY_FIELDS if key not in g.GRID and
                     key not in ('run_id','split','detection_seconds')}
            evaluator=Mock(return_value=(metrics,0.0))
            arguments=['grid','--output-dir',str(root)]
            with patch.object(g,'GRID',tiny), patch.object(g,'load_module',
                    side_effect=lambda path,alias: fake if 'predictor' in alias else comparator), \
                 patch.object(g,'file_hash',return_value='synthetic'), \
                 patch.object(g,'tree_partition',side_effect=lambda name: 'test' if name=='held.png' else 'development'), \
                 patch.object(g,'evaluate_configuration',evaluator), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys,'argv',arguments+['--select-by','f1']):
                    g.main()
                heldout=(root/'heldout_comparison.csv').read_bytes()
                metadata=json.loads((root/'grid_metadata.json').read_text())
                evaluator.reset_mock()
                with patch.object(sys,'argv',arguments+['--select-by','mae']):
                    g.main()
                self.assertEqual((root/'heldout_comparison.csv').read_bytes(),heldout)
                self.assertNotIn('select_by',metadata)
                evaluator.assert_not_called()
