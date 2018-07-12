Changes 2.5 => 2.6
============================================

1. Implemented a method for setting seed to curand random number generator. :py:meth:`renom.cuda.curand_set_seed`

2. Added argument `ignore_bias` to all parameterized class. As example, see doc of Dense class. :py:meth:`renom.layers.function.dense.Dense`.

3. Added argument `reduce_sum` to all loss function class. As example, see MeanSquaredError class. :py:meth:`renom.layers.loss.mean_squared_error.MeanSquaredError`.

4. Added ``Weight Normalize``. See :py:meth:`renom.layers.function.weight_normalize.WeightNormalize`

5. Added ``Layer Normalize``. See :py:meth:`renom.layers.function.layer_normalize.LayerNormalize`

6. Added ``GPUDistributor``. See :py:meth:`renom.utility.distributor.distributor.GPUDistributor`

7. Added ``Gru``. See :py:meth:`renom.layers.function.gru.Gru`

8. Added ``Conv3d``.

9. Added ``ConvNd``.

10. Added ``Pool3d``.

11. Added ``PoolNd``.

12. Added ``Unpool2d``

