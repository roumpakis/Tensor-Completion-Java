# Tensor Completion Java Implementation
TMac: Tensor Completion by parallel Matrix factorization (https://xu-yangyang.github.io/TMac/)  implemented in Java.
This project provides a mathematical tool for missing data reconstruction and managing three-dimensional structures.



# Table of contents
1. [Introduction](#introduction)
2. [Modules Description](#modules)
3. [Datasets](#datasets)
4. [Useful methods](#useful)
5. [Installation Instructions](#execution)
6. [Licence](#licence)

## Introduction <a name="introduction"></a>
One problem we often encounter in wireless sensor networks is that of non-recorded measurements.
Tensor Completion provides a way of completing these measurements,especially for data streams characterized by high correlation.  


**Tensor**


In real-world applications, it is often necessary to store the measurements in higher-order structures,
apart from 2D matrices.Tensors are third-dimensional data structure which can be consindered as as a generalization of scalars (zero-order
tensors), vectors (first-order tensors) and matrices (second-order tensors).

**Tensor Fold/Unfolding**

A commonly used way for recovering missing entries in high-order tensors is first to reduce them
to low-rank matrices via appropriate unfolding and then apply matrix completion techniques.
Unfolding is a transformation that reorders the elements of a tensor into a matrix and simplifies 
subsequent matrix-based processing . Such transformation is not unique, since different ways exist for stacking the horizontal, lateral and frontal slices of a tensor in either column-wise or row-wise arrays.
The reverse process, where a matrix  transform into a 3D Tensor is called folding.

![Tensor Unfolding](https://github.com/roumpakis/TCJ/blob/master/images/Capture.PNG)



**Tensor Completion (TC)**

Parallel matrix factorization (PMF)  has been introduced as an efficient alternative for solving the TC problem.
Focusing on the 3D case, we are interested in fully recovering a tensor T <font face="Symbol">&#206;</font>
 R<sup>(N * S * P)</sup>
from M  &lt; &lt; N * S * T measurements.
Specifically, T is unfolded across all of its modes to a set of matrix factors $X_n, Y_n$, such that $$Tn = XnYn$$, where n = 1; 2; 3 indicates the corresponding
mode. Introducing a common variable Z to relate these matrix factorizations, we solve the following
problem to recover T


![Tensor Unfolding](https://github.com/roumpakis/TCJ/blob/master/images/formula.PNG)

where X = (X1;X2;X3), Y = (Y1;Y2;Y3), and Zi, i = 1; 2; 3, corresponds to the unfolding of the
three-way tensor. The parameters Zi are introduced in order to properly weight the contribution
of each unfolding.

## Modules Description <a name="modules"></a>
The Project consists of the following classes.

* **Tensor:** Class for the construction and  management of 3D data structures, the Tensors
* **TMacPar:** Class for reconstructing low-rank tensor measurements using the TMac algorithm.
* **Matlab:** Class that provides useful mathematical functions, operations in three-dimensional structures as well as type conversion between its various mathematical libraries
* **IO:**  Class for reading and writing .csv files. Methods of this class are used to record the reconstruction results in the res\ folder, for post-processing evaluation purposes.
* **Main:** Class that includes the functions for executing TC in predefined data as well as a function for  user-defined data reconstruction


## Datasets <a name="datasets"></a>
The data streams are real measurements of pressure sensors located in the greater Malevizi Municipality area. 
Specifically in the folder data\ there are measurements from 10 regions stored in .csv files. The same folder contains the files needed to run a test.

* **DiIn.csv**	: Input water stream pressure measurement from i-th area
* **DiOut.csv** : Output water stream pressure measurement from i-th area
* **DiInMiss.csv**: Input water stream pressure measurement from i-th area with missing entries
* **rdnxj.csv**: Rank increase j-th vector for testing
* **X_j.csv**: Initialization of j-th Unfolding  of X matrix for testing
* **Y_j.csv**: Initialization of j-th Unfolding  of Y matrix for testing

## Useful Methods <a name="useful"></a>

* **Matlab.Fold:** Method that provides folding functionality, where a 2D matrix  transform into a 3D Tensor.
* **Matlab.Unfold:** Method that provides folding functionality, where a 3D Tensor  transform into a 2D matrix.
* **TCJTest.Test:** Method that runs a TC test, with predefined data 
* **Main.TMaacParTest:** Method that runs TC with user defined  data and random matrices
* **Tensor.TensorObservedElements:** The measurements set that we have observed, a useful function for data reconstruction purpose
* **Tensor.TensorObservedIndecies:** The indices where observed measurements are stored, a useful function for data reconstruction purpose

## Installation Instructions <a name="execution"></a>
1. Download the project source files.
2. Import the project to Netbeans or other IDE.
3. Call TCJTest.Test for predefined or Main.TMaacParTest for user defined data.


## License <a name="licence"></a>
Use of this source code in publications must be acknowledged by referencing the following publication:

* S. Roubakis, G. Tzagkarakis, and P. Tsakalides, "Real-Time Prototyping of Matlab-Java Code Integration for Water Sensor Networks Applications," in Proc. 27th European Signal Processing Conference (EUSIPCO '19), A Coruna, Spain, September 2-6, 2019.  