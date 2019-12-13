Threaded Implementation of InfoGain, RELIEF-F and SVM-RFE Feature Selection methods for Weka
=====================================================

This package contains Java classes to be added to the Weka source code in order to provide multithreaded versions of three feature selection methods: InfoGain and RELIEF-F (rankers) and SVM-RFE (subset evaluator).

Additionally a multithreaded version of the Discretize algorithm included with Weka is provided. For this class to compile, minor changes were made to existing Weka classes. All the files needed are included in the src folder.

This work has associated a submitted contributions to a Spanish congress (CAEPIA 2015 [1]) an international journal, which will be attached to this package as soon as they are accepted. This software has been proved with several large real-world datasets, such as:

- Epsilon dataset: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon. 400K instances and 2K attributes.
- Higgs dataset: http://archive.ics.uci.edu/ml/datasets/HIGGS. 11M instances and 28 attributes.

## Installation: 

Add contents of the src folder to the source folder of the current Weka distribution (tested for Weka 3.7.12) overwritting any existing files and then compile. Included in this archive is the jar compiled with Weka version 3.7.12.

## Usage: 

	InfoGain
		java -cp PATH_TO_JAR weka.attributeSelection.AttributeSelection weka.attributeSelection.InfoGainThreadedAttributeEval -s "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1" -c last -i PATH_TO_ARFF_FILE
	RELIEF-F
		java -cp PATH_TO_JAR weka.attributeSelection.AttributeSelection weka.attributeSelection.ReliefFThreadedAttributeEval -s "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1" -c last -i PATH_TO_ARFF_FILE
	SVM-RFE
		java -cp PATH_TO_JAR weka.attributeSelection.AttributeSelection weka.attributeSelection.SVMThreadedAttributeEval -s "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1" -c last -i PATH_TO_ARFF_FILE
	Discretize
		java -cp PATH_TO_JAR weka.filters.supervised.attribute.DiscretizeThreaded -i PATH_TO_ARFF_FILE -o PATH_TO_OUTPUT_FILE -c last -E -Y
        
## Prerequisites:

As with other Weka FS methods, files must be in .arff format.

## Contributors

- Carlos Eiras-Franco (carlos.eiras.franco@udc.es) (main contributor and maintainer).

##References

[1] Eiras-Franco, C., Bolón-Canedo, V., Ramos, S., González-Domınguez, J., Alonso-Betanzos, A., & Tourino, J. 2015 Paralelización de algoritmos de selección de caracterısticas en la plataforma Weka.
[2] Eiras-Franco, C., Bolón-Canedo, V., Ramos, S., González-Domınguez, J., Alonso-Betanzos, A., & Tourino, J. 2015 Multithreaded and Spark parallelization of feature selection filters
