# python-scala-rust-LogisticRegression-with-SGD
This Repo aims to compare performance of Python, Scala and Rust on Moons data with Logistic Regression using Stochastic Gradient Decent.

This projects aims to implement a Logistic Regression algorithms with SGD from scratch with Rust, Scala and Python3 and compare CPU and Memory Performances.

rustc 1.43.1 (8d69840ab 2020-05-04)

Scala code runner version 2.11.12 -- Copyright 2002-2017, LAMP/EPFL
OpenJDK Runtime Environment (build 11.0.7+10-post-Ubuntu-2ubuntu218.04)

Python 3.6.9 - Cpython

Hardware Specs:
CPU:
Intel(R) Core(TM) i7-4850HQ CPU @ 2.30GHz
CPU max MHz:         3500.0000
CPU min MHz:         800.0000

RAM:
Type: DDR3
Type Detail: Synchronous
Speed: 1600 MT/s

Hard Drive:
APPLE SSD SM0512F

Data: Moon Data set # Features has dimensions of (N,2), Binary Classification

args: 
--n_size[1000/10000/100000/1000000] #Data Size 
--l_rate[Float]
--n_epochs[Integer]
