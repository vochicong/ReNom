Changes 2.5 => 2.6
============================================

1. Implemented a method for setting seed to curand random number generator. :py:meth:`renom.cuda.curand_set_seed`

2. Added argument `ignore_bias` to all parameterized class. As example, see doc of Dense class. :py:meth:`renom.layers.function.dense.Dense`.

3. Added argument `reduce_sum` to all loss function class. As example, see MeanSquaredError class. :py:meth:`renom.layers.loss.mean_squared_error.MeanSquaredError`.

4. Added ``Weight Normalize``. See :py:meth:`renom.layers.function.weight_normalize.WeightNormalize`

5. Added ``Layer Normalize``. See :py:meth:`renom.layers.function.layer_normalize.LayerNormalize`

6. Added ``GPUDistributor``. See :py:meth:`renom.utility.distributor.distributor.GPUDistributor`

7. Added ``Gru``. See :py:meth:`renom.layers.function.gru.Gru`

8. Added ``Conv3d``. See :py:meth:`renom.layers.function.convnd.Conv3d`

9. Added ``ConvNd``. See :py:meth:`renom.layers.function.convnd.ConvNd`

10. Added ``Average, Max Pool3d``.
    See :py:meth:`renom.layers.function.poolnd.AvgPool3d` and
    See :py:meth:`renom.layers.function.poolnd.MaxPool3d`

11. Added ``Average, Max PoolNd``. See
    :py:meth:`renom.layers.function.poolnd.AvgPoolNd` and 
    :py:meth:`renom.layers.function.poolnd.MaxPoolNd`.

12. Added ``Average, Max Unpool2d``. See 
    :py:meth:`renom.layers.function.poolnd.AvgUnPool2d` and 
    :py:meth:`renom.layers.function.poolnd.MaxUnPool2d`.
