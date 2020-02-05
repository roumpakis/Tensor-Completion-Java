# Tensor Completion Java Implementation
TMac: Tensor Completion by Parallel Matrix Factorization (https://xu-yangyang.github.io/TMac/) implemented in Java.
This project provides a mathematical tool for missing data reconstruction by managing three-dimensional structures.



# Table of Contents
1. [Introduction](#introduction)
2. [Modules Description](#modules)
3. [Datasets](#datasets)
4. [Useful Methods](#useful)
5. [Installation Instructions](#execution)
6. [Licence](#licence)

## Introduction <a name="introduction"></a>
One problem we often encounter in (wireless) sensor networks is the existence of non-recorded measurements, e.g. due to sensors' malfuntion or network failures.
Tensor Completion provides a framework for recovering missing measurements with high accuracy, especially for data streams characterized by high correlation.  


**Tensor**


In real-world applications, it is often necessary to store the measurements in higher-order structures,
apart from 2D matrices. Tensors are third-dimensional data structures which can be considered as a generalization of scalars (zero-order
tensors), vectors (first-order tensors) and matrices (second-order tensors).

**Tensor Folding/Unfolding**

A commonly used procedure for recovering missing entries in high-order tensors is first to reduce them
to low-rank matrices via appropriate unfolding and then apply matrix completion techniques.
Unfolding is a transformation that reorders the elements of a tensor into a matrix and simplifies 
subsequent matrix-based processing. Such a transformation is not unique, since different ways exist for stacking the horizontal, lateral and frontal slices of a tensor in either column-wise or row-wise arrays.
The reverse process, where a matrix is transformed into a 3D tensor is called folding.


![Tensor Unfolding](https://github.com/roumpakis/TCJ/blob/master/images/Capture.PNG)



**Tensor Completion (TC)**

Parallel matrix factorization (PMF) has been introduced as an efficient alternative for solving the TC problem.
Focusing on the 3D case, we are interested in fully recovering a tensor T <font face="Symbol">&#8712;</font>
 R<sup>N x S x P </sup>  from M  <font face="Symbol">&#8810;</font> N * S * P measurements.
   
   ![Tensor Unfolding](https://github.com/roumpakis/TCJ/blob/master/images/formula.PNG)
   
Specifically, T is unfolded across all of its modes to a set of matrix factors X<sub>n</sub> , Y<sub>n</sub>
 such that T<sub>n</sub> <font face="Symbol">&#8776;</font> X<sub>n</sub> * Y<sub>n</sub>, where n = 1, 2, 3 indicates the corresponding
mode. A common variable Z is introduced to relate these matrix factorizations, and the tensor T is recovered by solving the following
optimization problem <br>
![Tensor Unfolding](https://github.com/roumpakis/TCJ/blob/master/images/min.PNG)




where X = (X<sub>1</sub>;X<sub>2</sub>;X<sub>3</sub>), Y = (Y<sub>1</sub>;Y<sub>2</sub>;Y<sub>3</sub>), 
and Z<sub>i</sub>, i = 1, 2, 3, correspond to the unfolding of the
three-way tensor. The parameters Z<sub>i</sub> are introduced in order to properly weight the contribution
of each unfolding.

## Modules Description <a name="modules"></a>
This project consists of the following classes:

* **Tensor:** Class for the construction and management of 3D data structures (the tensors).
* **TMacPar:** Class for reconstructing low-rank tensor measurements using the TMac algorithm.
* **Matlab:** Class that provides useful mathematical functions, operations in three-dimensional structures, as well as type conversion between its various mathematical libraries.
* **IO:**  Class for reading and writing .csv files. Methods of this class are used to record the reconstruction results in the res\ folder, for post-processing evaluation purposes.
* **Main:** Class that includes the functions for executing TC in predefined data, as well as a function for user-defined data reconstruction.


## Datasets <a name="datasets"></a>
The data streams are real measurements of pressure sensors located in the greater Malevizi Municipality area in Crete, Greece. 
Specifically, the folder data\ contains measurements from 10 regions stored in .csv files. The same folder contains the files required to execute a test case.

* **DiIn.csv**	: Input water stream pressure measurements from i-th area
* **DiOut.csv** : Output water stream pressure measurements from i-th area
* **DiInMiss.csv**: Input water stream pressure measurements from i-th area with missing entries
* **rdnxj.csv**: Rank increase j-th vector for testing
* **X_j.csv**: Initialization of j-th Unfolding of X matrix for testing
* **Y_j.csv**: Initialization of j-th Unfolding of Y matrix for testing

## Useful Methods <a name="useful"></a>

* **Matlab.Fold:** Method that provides the folding functionality, where a 2D matrix is transformed into a 3D tensor.
* **Matlab.Unfold:** Method that provides the unfolding functionality, where a 3D tensor is transformed into a 2D matrix.
* **TCJTest.Test:** Method that executes a TC test with predefined data. 
* **Main.TMaacParTest:** Method that performs TC with user-defined data and random matrices.
* **Tensor.TensorObservedElements:** The set of observed measurements; the function is used for data reconstruction purposes.
* **Tensor.TensorObservedIndecies:** The indices where observed measurements are stored; the function is used for data reconstruction purposes.

## Installation Instructions <a name="execution"></a>
1. Download the project's source files.
2. Import the project to Netbeans or other IDE.
3. Call TCJTest.Test for predefined or Main.TMaacParTest for user-defined data.


## License <a name="licence"></a>
This source code can be used for non-commercial purposes only. Its utilization must acknowledge and cite the following publication:

* S. Roubakis, G. Tzagkarakis, and P. Tsakalides, "Real-Time Prototyping of Matlab-Java Code Integration for Water Sensor Networks Applications," in Proc. 27th European Signal Processing Conference (EUSIPCO '19), A Coruna, Spain, September 2-6, 2019.  