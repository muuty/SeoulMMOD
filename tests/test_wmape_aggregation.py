"""Regression checks for global wMAPE across batches and runner paths.

Run with: python -m unittest discover -s tests -p 'test_wmape_aggregation.py'
"""
import functools
import json
import math
import tempfile
import unittest

import torch

from basicts.metrics import masked_mae, masked_rmse, masked_wape, unmasked_wmape
from basicts.runners.base_tsf_runner import BaseTimeSeriesForecastingRunner
from basicts.runners.od_native_runner import ODNativeRunner


class ForecastRunner(BaseTimeSeriesForecastingRunner):
    def forward(self, data, **kwargs):
        return data


class ODRunner(ODNativeRunner):
    def forward(self, data, **kwargs):
        return data


class GlobalWmapeTest(unittest.TestCase):
    def setUp(self):
        # Two wholly zero-target samples, highly unequal volumes, and a short final batch.
        self.target = torch.tensor([
            [0, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4],
            [100, 200, 300, 400], [2, 0, 5, 0],
        ], dtype=torch.float64).reshape(5, 2, 1, 2)
        self.prediction = self.target + torch.arange(1, 21, dtype=torch.float64).reshape_as(self.target)

    def runner(self, cls=ForecastRunner, batch_size=2, metrics=None):
        # Exercise the real iteration/aggregation methods without a model or dataset.
        runner = object.__new__(cls)
        runner.metrics = metrics or {'MAE': masked_mae, 'RMSE': masked_rmse, 'wMAPE': unmasked_wmape}
        runner.loss = masked_mae
        runner.loss_args = {}
        runner.null_val = float('nan')
        runner.meter_pool = None
        runner.cl_param = None
        runner.iter_per_epoch = 5
        runner.if_evaluate_on_gpu = False
        runner.evaluation_horizons = [0, 1]
        runner.mode_names = ('a', 'b')
        runner.test_data_loader = [
            {'prediction': p, 'target': t}
            for p, t in zip(self.prediction.split(batch_size), self.target.split(batch_size))
        ]
        for phase in ('train', 'val', 'test'):
            runner.register_epoch_meter(f'{phase}/loss', phase)
            for key in runner.metrics:
                runner.register_epoch_meter(f'{phase}/{key}', phase)
                if phase == 'test':
                    for suffix in ('h1', 'h2', 'a', 'b'):
                        runner.register_epoch_meter(f'{phase}/{key}@{suffix}', phase)
        return runner

    def assert_metrics(self, actual, prediction, target):
        error = (prediction - target).abs()
        expected = {
            'wMAPE': error.sum().item() / target.abs().sum().item(),
            'MAE': error.mean().item(),
            'RMSE': error.square().mean().sqrt().item(),
        }
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, places=10)
        self.assertAlmostEqual(unmasked_wmape(prediction, target).item(), expected['wMAPE'], places=10)

    def test_test_loops_are_batch_invariant(self):
        for cls in (ForecastRunner, ODRunner):
            for batch_size in (1, 2, 3, 5):
                with self.subTest(runner=cls.__name__, batch_size=batch_size), tempfile.TemporaryDirectory() as tmp:
                    runner = self.runner(cls, batch_size)
                    runner.ckpt_save_dir = tmp
                    runner.test(save_metrics=True)
                    with open(f'{tmp}/test_metrics.json') as f:
                        results = json.load(f)
                    self.assert_metrics(results['overall'], self.prediction, self.target)
                    for horizon in (0, 1):
                        self.assert_metrics(results[f'horizon_{horizon+1}'], self.prediction[:, horizon], self.target[:, horizon])
                    if cls is ODRunner:
                        for index, mode in enumerate(runner.mode_names):
                            self.assert_metrics(results['per_mode'][mode], self.prediction[..., index], self.target[..., index])

    def test_training_validation_and_reset(self):
        runner = self.runner()
        for _ in range(2):
            runner.meter_pool.reset()
            for i, data in enumerate(runner.test_data_loader):
                runner.train_iters(1, i, data)
                runner.val_iters(i, data)
            for phase in ('train', 'val'):
                actual = {key: runner.meter_pool.get_value(f'{phase}/{key}') for key in runner.metrics}
                self.assert_metrics(actual, self.prediction, self.target)

    def test_wmape_includes_zeros_with_partial_null_value(self):
        runner = self.runner(metrics={'wMAPE': functools.partial(unmasked_wmape, null_val=0)})
        runner.test()
        expected = (self.prediction - self.target).abs().sum() / self.target.abs().sum()
        self.assertAlmostEqual(runner.meter_pool.get_value('test/wMAPE'), expected.item())

    def test_upstream_masked_wape_keeps_its_semantics(self):
        runner = self.runner(metrics={'WAPE': masked_wape})
        runner.test()
        expected = sum(masked_wape(**data).item() * data['target'].numel()
                       for data in runner.test_data_loader) / self.target.numel()
        self.assertAlmostEqual(runner.meter_pool.get_value('test/WAPE'), expected)

    def test_zero_total_volume_is_undefined(self):
        self.target.zero_()
        runner = self.runner()
        runner.test()
        self.assertTrue(math.isnan(runner.meter_pool.get_value('test/wMAPE')))
        self.assertTrue(torch.isnan(unmasked_wmape(self.prediction, self.target)).item())


if __name__ == '__main__':
    unittest.main()
