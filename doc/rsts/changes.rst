Change History
==============

-------------------------
version 2.0.0 ⇒ version 2.0.1
-------------------------

Added Trainer class.

-------------------------
version 1.2 ⇒ version 2.0.0
-------------------------

The changes of the version from 1.2 to 2.0 are below.

=============================================== ================ ===================================
Changes                                         version 1.2      version 2.0
=============================================== ================ ===================================
1- Name of Model class                          Model            Sequential
2- Input layer has deleted                      Need Input layer Not need
3- The timing of weight initialization          Model creation   First forward propagation
4- Name of the attribute of weight parameter    parameters       params
5- Need "with block" in train loop              Not needed       Need to manage computational graph 
6- Let out loss functions from Model definition Defined in model Defined out of model
7- Name of back propagation method              backward         grad
8- Which class has update method                Model            Node
9- How to use of optimizer class                With model class With update method
=============================================== ================ ===================================

**1- Name of Model class**

The name of Model class has changed. 
In version 1.2 for building a neural network, you are supposed to use Model class.
In version 2.0, it renamed to Sequential.

**Example**

.. code-block:: python
   
   # Version.2
   model = Model([
           Input(10),
           Dense(100),
           Relu(),
           Dense(10),
           MeanSquaredError()
        ])
        

.. code-block:: python
   
   # Version.0
   model = Sequential([
           # Input(10), Input layer has deleted.
           Dense(100),
           Relu(),
           Dense(10),
           # MeanSquaredError() Loss function will be defined at train loop.
        ])

        
**2- Input layer has deleated**

Showed in above example, the input layer class was removed in version 2.0 .

**3- The timing of weight initialization**

Because of the input layer was removed, the timing of weight initialization was changed.
In version 1.2, weight parameters are initialized when a model object is instantiated.
In version 2.0, weight parameters are initialized at the first forward propagation,
or if the argument input_size has been passed, weights are initialized same timing as version 2.0. 
See tutorial 0.2 for more detail of weight initialization.

**Example**

.. code-block:: python

   # Version 1.2
   model = Model([
           Input(10),
           Dense(100),
           Relu(),
           Dense(10),
           MeanSquaredError()
        ])
   # You can access the weight parameters because weights are already initialized.
   print(model[1].parameters["w"])
   
.. code-block:: python

   # Version 2.0
   model = Sequential([
           Dense(100),
           Relu(),
           Dense(10),
        ])
   # Weight parameters have not been initialized yet.
   z = model(x)   # Execute forward propagation
   print(model[0].params["w"])  # After execution, parameters can be accessed.
   
   
**4- Name of the weight parameter attribute**

The attribute name has changed to params from parameters.


**5- Need "with block" in train loop**

In version 2.0, because of the auto-differentiation was implemented,
users has responsible for managing building computational graphs.
Computational graphs will continue to extend accordance with the
execution of each operations.

ReNom limits the extension of computational graphs only in **with train block**.
You have to execute the operation in "with train block" if you want to build a
computational graph. The operation which was done out side of the "with block" will not
extend the computational graph. This means the operation which was done out side of the
with block is not concerned at the calculation of the gradient.

**Example**

.. code-block:: python

   # Version 1.2
   loss = model.forward(x, y)


.. code-block:: python

   # Version 2.0
   with model.train():
      z = model(x)
      loss = mean_squared_error(z, y)
   

**6- Let out loss functions from Model definition**

Loss functions turned out to be set out side of model definition.
As showed above, the loss function is used in out of model to evaluate the difference
between output of the model and the target value. 

   
**7- Name of backpropagation method**

The name of back propagation method has been changed. And also the object which has the method was changed.
In version 1.2, model object has the back propagation method as a name of "backward".
In version 2.0, node object has the back propagation method as a name of "grad".

Both methods executes back propagation, but the name and which object has the method are
different.

**Example**

.. code-block:: python

   # Version 1.2
   loss = model.forward(x, y)
   model.backward()
   model.update()

.. code-block:: python

   # Version 2.0
   with model.train():
      z = model(x)
      loss = mean_squared_error(z, y)
   gradients = loss.grad()
   gradients.update()
   

**8- Which object has update method**

The method "update" has been moved to Node class same as above example.

   
**9- How to use of optimizer class**

In version 1.2, optimizer class, for instance Sgd class, is used with model object.
In version 2.0, optimizer class is added to update method as an argument.

**Example**

.. code-block:: python

   # Version 1.2
   # Optimizer object wraps model object. 
   model = Sgd(Model([
           Input(10),
           Dense(100),
           Relu(),
           Dense(10),
           MeanSquaredError()
        ]), lr=0.1)
   loss = model.forward(x, y)
   model.backward()
   model.update()

.. code-block:: python

   # Version 2.0
   optimizer = Sgd(lr=0.1)
   with model.train():
      z = model(x)
      loss = mean_squared_error(z, y)
   gradients = loss.grad()
   # Optimizer object is passed to update method.
   gradients.update(optimizer)

