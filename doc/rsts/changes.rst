Changes 2.3.1 => 2.4.1
============================================

Modified
^^^^^^^^

1. GPU accelerated __getitem__, __setitem__ is available.

2. renom.concat() accepts variable arguments.

3. renom.sum() accepts 'axis' argument.

4. Auto differentiation is enabled in functions 'T', 'transpose' and 'reshape' that are implemented in Node object.

5. In the Yolo class, redundant argument 'image_size' of the function '__init__' is removed.


New feature
^^^^^^^^^^^

1. New auto differentiation available operation: renom.amax, renom.amin.

2. New method for releasing gpu memory pool: renom.cuda.release_mem_pool.

Removed module
^^^^^^^^^^^^^^

Nothing

Bug fix
^^^^^^^

1. Broadcasting matrix arithmetic operations.

    In the previous version, broadcasted calculations, as exampled bellow, with gpu are
    not correctly calculated.

    Example:
        >>> 

2. Batch normalization inference calculation.



