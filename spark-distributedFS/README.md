Spark implementation of the FS methods RELIEF-F, CFS and SVM-RFE.
=====================================================

This project implements three methods of Feature Selection (FS) on Spark for its application on Big Data problems. The methods implemented are RELIEF-F, CFS and SVM-RFE.

This work has associated a submitted contributions to a Spanish congress (CAEPIA 2015 [1]) an international journal, which will be attached to this package as soon as they are accepted. This software has been proved with several large real-world datasets, such as:

- Epsilon dataset: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon. 400K instances and 2K attributes.
- Higgs dataset: http://archive.ics.uci.edu/ml/datasets/HIGGS. 11M instances and 28 attributes.
- KDD2010 Bridge to algebra: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html. 19M instances and 30M attributes.
- Poker Hand: http://archive.ics.uci.edu/ml/datasets/Poker+Hand. 1M instances and 11 attributes.

## Building:
	To build simply run:
		mvn clean package

## Usage:
	- RELIEFF:
		spark-submit --class org.apache.spark.mllib.feature.ReliefFFeatureSelector PATH_TO_COMPILED_JAR PATH_TO_LIBSVM_FILE
	- CFS:
		spark-submit --class org.apache.spark.mllib.feature.CFSFeatureSelector PATH_TO_COMPILED_JAR PATH_TO_LIBSVM_FILE
	- SVM-RFE:
		spark-submit --class org.apache.spark.mllib.feature.SVMRFEFeatureSelector PATH_TO_COMPILED_JAR PATH_TO_LIBSVM_FILE


## Prerequisites:

Files must be in libsvm format. 

## Contributors

- Carlos Eiras-Franco (carlos.eiras.franco@udc.es) (main contributor and maintainer).

##References

[1] Eiras-Franco, C., Bolón-Canedo, V., Ramos, S., González-Domınguez, J., Alonso-Betanzos, A., & Tourino, J. 2015 Paralelización de algoritmos de selección de caracterısticas en la plataforma Weka.
[2] Eiras-Franco, C., Bolón-Canedo, V., Ramos, S., González-Domınguez, J., Alonso-Betanzos, A., & Tourino, J. 2015 Multithreaded and Spark parallelization of feature selection filters
