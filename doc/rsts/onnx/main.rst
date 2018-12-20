ONNX Support
=============

renom.utility.onnx
-------------------------

.. automodule:: renom.utility.onnx
    :members:
    :undoc-members:


How to use.
-----------

Save Model
__________

  .. code-block:: python

    import renom as rm
    import renom.utility.onnx as onnx

    # Define a CNN
    cnn = rm.Sequential([
        rm.Conv2d(channel=32, filter=3, padding=1),
        rm.Relu(),
        rm.Conv2d(channel=64, filter=3, padding=1),
        rm.Relu(),
        rm.MaxPool2d(filter=2, stride=2),
        rm.Dropout(0.5),
        rm.Flatten(),
        rm.Dense(128),
        rm.Relu(),
        rm.Dense(10)
    ])

    # Train the CNN on some dataset
    # ... CNN Train ...

    # Save the trained model as ONNX fomrat.
    # Note: This requires ``dummy_input`` for build a computational graph.
    dummy_input = np.random.random((1, 1, 28, 28))
    onnx.export_onnx("mnist", cnn, dummy_input, "mnist.onnx") 


Load Model
__________

Supported Functions.
--------------------

Operations
___________

- __neg__
- __add__
- __abs__
- __sub__
- __mul__
- __div__


Activation functions
___________


- Relu


Layers
___________

- Dense(:py:meth:`renom.layers.function.dense.Dense`)
- Conv2d(:py:meth:`renom.layers.function.conv2d.Conv2d`)
- MaxPool2d(:py:meth:`renom.layers.function.pool2d.MaxPool2d`)
- Dropout(:py:meth:`renom.layers.function.dropout.Dropout`)


Others
-------

- reshape(:py:meth:`renom.core.Node`)
- flatten(:py:meth:`renom.layers.function.flatten.Flatten`)

