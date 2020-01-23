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
Unfolding is a transformation that reorders the elements of a tensor into a matrix and simplifies subsequent matrix-based processing . Such transformation is not unique, since different ways exist for stacking the horizontal, lateral and frontal slices of a tensor in either column-wise or row-wise arrays.
The reverse process, where a 3D Tensor transform into a matrix is called folding.
**Tensor Completion (TC)**



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
* **rdnxj.csv**: I
* **X_j.csv**: I
* **Y_j.csv**: I
## Useful Methods <a name="useful"></a>

* HAPT dataset
* FORTH-TRACE dataset (collective scenario -- all devices of 10 random participants)
* FORTH-TRACE dataset (single-device scenario -- for each separate device of 10 random participants) 

## Installation Instructions <a name="execution"></a>
1. Download the project source files.
2. Import the project to Netbeans or other IDE.
3. Add the following libraries into the build path (located in the libs/ directory of the project):
4. Specify the path of your dataset files in the Main.java file of the project.


## License <a name="licence"></a>
Use of this source code in publications must be acknowledged by referencing the following publications:

* Roumpakis, Stylianos ; Tzagkarakis, George ; Tsakalides, Panagiotis Real-Time Prototyping of Matlab-Java Code Integration for Water Sensor Networks Applications,European Signal Processing Conference (EUSIPCO), 2019.