
ReNom TDA
=========================

In ReNom version 2.2, TDA modules has been added.

Introduction to ReNom TDA
---------------

In machine learning and deep learning, we model data with various algorithms.
At that time, think about how to understand the complex data structure and model it.
Because learning may not proceed well unless a correct approach is taken.

The data generated on IoT every day has a very complicated structure,
which is different from the data set whose results are guaranteed in papers.

That means that engineers who develop and analyze must proceed with searching for clues
by themselves and the that is really difficult.

When we analysis data, modeling without understanding the complex multidimensional data structure
and modeling by understanding the data structure can't achieve the same accuracy and results.

And we think that the time required to derive them is overwhelmingly different.
So, we thought about a method to solve it.

ReNom TDA is a module for mapping and visualizing and analyzing high dimensional data in topological space.
By understanding the shape of the data and understanding the relationship between the variables,
it helps engineers who analyze the data.

And ReNom TDA can be used not only as preprocessing of data or understanding of data structure
as an advanced profiling tool.

For example, by visualizing the connection between complicated data, it becomes possible to profile various data
such as customer data, machine data, financial data, unauthorized access, cyber security, etc.

We aim to provide new discoveries and applications that create new business through various algorithms.
Please try out various ideas by all means.


**What is TDA?**

Topological data analysis (TDA) is a new data analysis method using Topology,
which makes it possible to visualize the shape of data in topological space and extract the meanings of the data.

Topology is a field of mathematics that considers connections in topological space,
focusing on properties that are retained even when continuously deformed without cutting and pasting.

By visualizing the data in the topological space, it becomes possible to visualize the data
without losing the features that were lost when reduce dimension of the data in the old method.

There are 2 ways to analyze by TDA.

The first one is clustering.
In ReNom TDA, one node contains data with similar characteristics.
Therefore, by looking at the data contained in the node, you can find the feature of data.

Second is comparing colors.
you can find variables with high correlation with the target variables by comparing colors.

We colorize the topology with various variables and look for a variable
that shows the same color distribution as the target variable.
By doing this, you can intuitively find variables with high correlation with target variables.


**About Topology**

We use metrics and lenses to project high dimensional data into a low dimensional space called a point cloud.

Data is mapped to topological space by dividing the point cloud and clustering the data in the partition.
We create a topology, by connecting nodes that have common data to each other.

At ReNom TDA, we finally colorize the topology and visualize it.

Metric is a measure of the distance, for example, Euclidean distance, cosine distance, Hamming distance etc.

The lens represents the axis of projection and you can use algorithms
not only PCA, T-SNE but auto encoder as a dimension reduction method.

When mapping to topological space, you can decide the number of divisions by resolution and width of overlap,
and the cluster is divided according to that parameter.

In Python APIs, you can combine multiple lenses to create a point cloud.
In the future expansion, users can add various algorithms themselves.



**ReNom TDA GUI**

You can visualize topology in web application with ReNom TDA.
And you can launch the application with one command, and you can visualize the shape of the data without programming.

It is possible to easily visualize and analyze complex data simply
by importing data from CSV and specifying an algorithm and executing it.

Regarding how to handle data and how to operate it, please see it there as it is published in the ReNom catalog.

In the next version, we can extract data sets obtained from ReNom TDA be learned by deep learning and
other machine learning algorithms, and visualize arranging multiple target variables side by side.

So, we will promote the development of easy-to-use functions for data analysts.

Installation
------------

First, you have to install the python.
There are many web pages that explain how to intall the python.
And, you can download ReNom TDA from following link.

URL: https://github.com/ReNom-dev-team/ReNomTDA

.. code-block:: sh

   git clone https://github.com/ReNom-dev-team/ReNomTDA.git
   cd ReNomTDA
   pip install -e .

**Requirements**

ReNom requires following libraries.

- Python 2.7, 3.4
- Numpy 1.13.0, 1.12.1 http://www.numpy.org/
- bottle 0.12.13 https://bottlepy.org/docs/dev/
- matplotlib 2.0.2
- networkx 1.11
- pandas 0.20.3
- scikit\-learn 0.18.2
- scipy 0.19.0
- pytest 3.0.7
