/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    InfoGainAttributeEval.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.attributeSelection;

import java.util.Enumeration;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToBinary;

/**
 * <!-- globalinfo-start --> InfoGainAttributeEval :<br/>
 * <br/>
 * Evaluates the worth of an attribute by measuring the information gain with
 * respect to the class.<br/>
 * <br/>
 * InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).<br/>
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -M
 *  treat missing values as a separate value.
 * </pre>
 * 
 * <pre>
 * -B
 *  just binarize numeric attributes instead 
 *  of properly discretizing them.
 * </pre>
 * 
 * <pre>
 * -P &lt;num threads&gt;
 * Specify the number of threads to be
 * used in the computation. If too large
 * or 0 all threads available will be used.
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @version $Revision: 1 $
 * @see Discretize
 * @see NumericToBinary
 */
public class InfoGainThreadedAttributeEval extends ASEvaluation implements AttributeEvaluator, OptionHandler
{
	private class InstanceCounter extends Thread
	{
		private Instances data;
		private int firstInstanceIndex;
		private int lastInstanceIndex;
		private int classIndex;
		private int numClasses;
		private double[][][] counts;
		
		public double[][][] getCounts()
		{
			return counts;
		}

		public InstanceCounter(Instances data, int firstInstanceIndex, int lastInstanceIndex)
		{
			this.data=data;
			this.firstInstanceIndex=firstInstanceIndex;
			this.lastInstanceIndex=lastInstanceIndex;
			this.classIndex=data.classIndex();
			this.numClasses=data.attribute(classIndex).numValues();

			// Reserve space and initialize counters
			counts = new double[data.numAttributes()][][];
			for (int k = 0; k < data.numAttributes(); k++)
				if (k != classIndex)
				{
					int numValues = data.attribute(k).numValues();
					counts[k] = new double[numValues + 1][numClasses + 1];
				}
		}
		
		public void run()
		{
			performCount();
		}
		
		private void performCount()
		{
			// Initialize counters
		    //Can this operation be integrated in the counting loop??
		    double[] temp = new double[numClasses + 1];
		    for (int k = firstInstanceIndex; k < lastInstanceIndex; k++)
		    {
		    	Instance inst = data.instance(k);
		    	if (inst.classIsMissing())
		    		temp[numClasses] += inst.weight();
		    	else
		    		temp[(int) inst.classValue()] += inst.weight();
		    }
		    for (int k = 0; k < counts.length; k++)
		    	if (k != classIndex)
		    		for (int i = 0; i < temp.length; i++)
		    			counts[k][0][i] = temp[i];
		    
			// Get counts
		    for (int k = firstInstanceIndex; k < lastInstanceIndex; k++)
		    {
		    	Instance inst = data.instance(k);
		    	for (int i = 0; i < inst.numValues(); i++)
		    	{
		    		if (inst.index(i) != classIndex)
		    		{
		    			if (inst.isMissingSparse(i) || inst.classIsMissing())
		    			{
		    				if (!inst.isMissingSparse(i))
		    				{
		    					counts[inst.index(i)][(int) inst.valueSparse(i)][numClasses] += inst.weight();
		    					counts[inst.index(i)][0][numClasses] -= inst.weight();
		    				}
		    				else
		    					if (!inst.classIsMissing())
		    					{
		    						counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][(int) inst.classValue()] += inst.weight();
		    						counts[inst.index(i)][0][(int) inst.classValue()] -= inst.weight();
		    					}
		    					else
		    					{
		    						counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][numClasses] += inst.weight();
		    						counts[inst.index(i)][0][numClasses] -= inst.weight();
		    					}
		    			}
		    			else
		    			{
		    				counts[inst.index(i)][(int) inst.valueSparse(i)][(int) inst.classValue()] += inst.weight();
		    				counts[inst.index(i)][0][(int) inst.classValue()] -= inst.weight();
		    			}
		    		}
		    	}
		    }
		}
	}
	
	private class MissingMergerAndGainCalculator extends Thread
	{
		private Instances data;
		private int firstAttributeIndex;
		private int lastAttributeIndex;
		private int classIndex;
		private int numClasses;
		private double[][][] counts;
		
		public MissingMergerAndGainCalculator(Instances data, int firstAttributeIndex, int lastAttributeIndex, double[][][] counts)
		{
			this.data=data;
			this.firstAttributeIndex=firstAttributeIndex;
			this.lastAttributeIndex=lastAttributeIndex;
			this.classIndex=data.classIndex();
			this.numClasses=data.attribute(classIndex).numValues();
			this.counts=counts;
		}
		
		public void run()
		{
			performCalculations();
		}
		
		public void performCalculations()
		{
			// distribute missing counts if required
		    if (m_missing_merge)
		    {	
		    	for (int k = firstAttributeIndex; k < lastAttributeIndex; k++)
		    	{
		    		if (k != classIndex)
		    		{
		    			int numValues = data.attribute(k).numValues();

		    			// Compute marginals
		    			double[] rowSums = new double[numValues];
		    			double[] columnSums = new double[numClasses];
		    			double sum = 0;
		    			for (int i = 0; i < numValues; i++)
		    			{
		    				for (int j = 0; j < numClasses; j++)
		    				{
		    					rowSums[i] += counts[k][i][j];
		    					columnSums[j] += counts[k][i][j];
		    				}
		    				sum += rowSums[i];
		    			}

		    			if (Utils.gr(sum, 0))
		    			{
		    				double[][] additions = new double[numValues][numClasses];

		    				// Compute what needs to be added to each row
		    				for (int i = 0; i < numValues; i++)
		    					for (int j = 0; j < numClasses; j++)
		    						additions[i][j] = (rowSums[i] / sum) * counts[k][numValues][j];

		    				// Compute what needs to be added to each column
		    				for (int i = 0; i < numClasses; i++)
		    					for (int j = 0; j < numValues; j++)
		    						additions[j][i] += (columnSums[i] / sum) * counts[k][j][numClasses];

		    				// Compute what needs to be added to each cell
		    				for (int i = 0; i < numClasses; i++)
		    					for (int j = 0; j < numValues; j++)
		    						additions[j][i] += (counts[k][j][i] / sum) * counts[k][numValues][numClasses];

		    				// Make new contingency table
		    				double[][] newTable = new double[numValues][numClasses];
		    				for (int i = 0; i < numValues; i++)
		    					for (int j = 0; j < numClasses; j++)
		    						newTable[i][j] = counts[k][i][j] + additions[i][j];
		    				
		    				counts[k] = newTable;
		    			}
		    		}
		    	}
		    }

		    // Compute info gains
		    m_InfoGains = new double[data.numAttributes()];
		    for (int i = 0; i < data.numAttributes(); i++)
		    	if (i != classIndex)
		    		m_InfoGains[i] = (ContingencyTables.entropyOverColumns(counts[i]) - ContingencyTables.entropyConditionedOnRows(counts[i]));
		}
	}
  /** for serialization */
  static final long serialVersionUID = -1949849544589218930L;

  /** Treat missing values as a separate value */
  private boolean m_missing_merge;

  /** Just binarize numeric attributes */
  private boolean m_Binarize;

  /** The info gain for each attribute */
  private double[] m_InfoGains;

  /** Number of threads to be used in the computation */
  private int n_threads=Integer.MAX_VALUE;
  
  /**
   * Returns a string describing this attribute evaluator
   * 
   * @return a description of the evaluator suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return "InfoGainThreadedAttributeEval :\n\nEvaluates the worth of an attribute "
      + "by measuring the information gain with respect to the class.\n\n"
      + "InfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).\n";
  }

  /**
   * Constructor
   */
  public InfoGainThreadedAttributeEval() {
    resetOptions();
  }

  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options.
   **/
  @Override
  public Enumeration<Option> listOptions() {
    Vector<Option> newVector = new Vector<Option>(2);
    newVector.addElement(new Option("\ttreat missing values as a separate "
      + "value.", "M", 0, "-M"));
    newVector.addElement(new Option(
      "\tjust binarize numeric attributes instead \n"
        + "\tof properly discretizing them.", "B", 0, "-B"));
    newVector.addElement(new Option("\tSpecify the number of threads to be\n"
    	      + "\tused in the computation. If too\n"
    	      + "\tlarge or 0 all threads available will\n"
    	      + "\tbe used.", "P", 1,
    	      "-P <num threads>"));
    return newVector.elements();
  }

  /**
   * Parses a given list of options.
   * <p/>
   * 
   * <!-- options-start --> Valid options are:
   * <p/>
   * 
   * <pre>
   * -M
   *  treat missing values as a separate value.
   * </pre>
   * 
   * <pre>
   * -B
   *  just binarize numeric attributes instead 
   *  of properly discretizing them.
   * </pre>
   * 
   * <pre>
   * -P &lt;num threads&gt;
   * Specify the number of threads to be
   * used in the computation. If too large
   * or 0 all threads available will be used.
   * </pre>
   * 
   * <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {

    resetOptions();
    setMissingMerge(!(Utils.getFlag('M', options)));
    setBinarizeNumericAttributes(Utils.getFlag('B', options));
    
    String optionString = Utils.getOption('P', options);

    if (optionString.length() != 0) {
      setN_threads(Integer.parseInt(optionString));
    }
  }

  /**
   * Gets the current settings of WrapperSubsetEval.
   * 
   * @return an array of strings suitable for passing to setOptions()
   */
  @Override
  public String[] getOptions() {
    String[] options = new String[3];
    int current = 0;

    if (!getMissingMerge()) {
      options[current++] = "-M";
    }
    if (getBinarizeNumericAttributes()) {
      options[current++] = "-B";
    }
    
    options[current++] = " -P "+((getN_threads()==Integer.MAX_VALUE)?0:getN_threads());

    while (current < options.length) {
      options[current++] = "";
    }

    return options;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String binarizeNumericAttributesTipText() {
    return "Just binarize numeric attributes instead of properly discretizing them.";
  }

  /**
   * Binarize numeric attributes.
   * 
   * @param b true=binarize numeric attributes
   */
  public void setBinarizeNumericAttributes(boolean b) {
    m_Binarize = b;
  }

  /**
   * get whether numeric attributes are just being binarized.
   * 
   * @return true if missing values are being distributed.
   */
  public boolean getBinarizeNumericAttributes() {
    return m_Binarize;
  }

  /**
   * Returns the tip text for this property
   * 
   * @return tip text for this property suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String missingMergeTipText() {
    return "Distribute counts for missing values. Counts are distributed "
      + "across other values in proportion to their frequency. Otherwise, "
      + "missing is treated as a separate value.";
  }

  /**
   * distribute the counts for missing values across observed values
   * 
   * @param b true=distribute missing values.
   */
  public void setMissingMerge(boolean b) {
    m_missing_merge = b;
  }

  /**
   * get whether missing values are being distributed or not
   * 
   * @return true if missing values are being distributed.
   */
  public boolean getMissingMerge() {
    return m_missing_merge;
  }
  
  public int getN_threads()
  {
	  return n_threads;
  }
  
  public void setN_threads(int n_threads)
  {
	  if (n_threads<=0)
		  this.n_threads=Integer.MAX_VALUE;
	  else
		  this.n_threads = n_threads;
  }

  /**
   * Returns the capabilities of this evaluator.
   * 
   * @return the capabilities of this evaluator
   * @see Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }

  /**
   * Initializes an information gain attribute evaluator. Discretizes all
   * attributes that are numeric.
   * 
   * @param data set of instances serving as training data
   * @throws Exception if the evaluator has not been generated successfully
   */
  @Override
  public void buildEvaluator(Instances data) throws Exception {

    // can evaluator handle data?
    getCapabilities().testWithFail(data);

    //int classIndex = data.classIndex();
    int numInstances = data.numInstances();

    if (!m_Binarize)
    {
    	Discretize disTransform = new Discretize();
    	disTransform.setUseBetterEncoding(true);
    	disTransform.setInputFormat(data);
    	data = Filter.useFilter(data, disTransform);
    } 
    else
    {
    	NumericToBinary binTransform = new NumericToBinary();
    	binTransform.setInputFormat(data);
    	data = Filter.useFilter(data, binTransform);
    }
    //int numClasses = data.attribute(classIndex).numValues();

    // Reserve space and initialize counters
    // Get counts
    int numThreads = Runtime.getRuntime().availableProcessors();
    if (numThreads>n_threads)
    	numThreads=n_threads;
    InstanceCounter[] counters=new InstanceCounter[numThreads];    
    int firstInstanceIndex=0;
    int step=numInstances/numThreads;
    for (int i=0; i<counters.length; i++)
    {
    	if (i>=counters.length-1)
    		step=numInstances-firstInstanceIndex;
    	counters[i]=new InstanceCounter(data, firstInstanceIndex, firstInstanceIndex+step);
    	firstInstanceIndex+=step;
    	counters[i].start();
    }
    
    double[][][] counts = new double[data.numAttributes()][][];
    
    for (int i=0; i<counters.length; i++)
    {
    	boolean exito=false;
    	try{
    		counters[i].join();
    		exito=true;
    	} catch (InterruptedException e) {
			System.out.println("Problem joining thread "+ i);
    	}
    	
    	if (exito)
    	{
    		//Add up the results from the threads
    	    double[][][] itemCounts=counters[i].getCounts();
    	    for (int j=0; j<itemCounts.length; j++)
    	    {
    	    	if (counts[j]==null)
    	    		counts[j]=itemCounts[j];
    	    	else
    	    	{
	    	    	for (int k=0; k<itemCounts[j].length; k++)
	    	    		for (int l=0; l<itemCounts[j][k].length; l++)
	    	    			counts[j][k][l]+=itemCounts[j][k][l];
    	    	}
    	    }
    	}
    	counters[i]=null;
    }
    
    // distribute missing counts if required
    // Compute info gains
    numThreads = Runtime.getRuntime().availableProcessors(); //Check again since it may vary during execution
    if (numThreads>n_threads)
    	numThreads=n_threads;
    MissingMergerAndGainCalculator[] workers=new MissingMergerAndGainCalculator[numThreads];
    
    int firstAttributeIndex=0;
    step=data.numAttributes()/numThreads;
    for (int i=0; i<workers.length; i++)
    {
    	if (i>=counters.length-1)
    		firstInstanceIndex=numInstances-step;
    	workers[i]=new MissingMergerAndGainCalculator(data, firstAttributeIndex, firstAttributeIndex+step, counts);
    	firstAttributeIndex+=step;
    	workers[i].start();
    }
    
    for (int i=0; i<workers.length; i++)
    {
    	boolean exito=false;
    	try{
    		workers[i].join();
    		exito=true;
    	} catch (InterruptedException e) {
			System.out.println("Problem joining thread "+ i);
    	}
    	
    	if (exito)
    	{
    		//Add up the results from the threads
    	    //Nothing to be done
    	}
    	workers[i]=null;
    }
  }
  
  /**
   * Reset options to their default values
   */
  protected void resetOptions() {
    m_InfoGains = null;
    m_missing_merge = true;
    m_Binarize = false;
  }

  /**
   * evaluates an individual attribute by measuring the amount of information
   * gained about the class given the attribute.
   * 
   * @param attribute the index of the attribute to be evaluated
   * @return the info gain
   * @throws Exception if the attribute could not be evaluated
   */
  @Override
  public double evaluateAttribute(int attribute) throws Exception {

    return m_InfoGains[attribute];
  }

  /**
   * Describe the attribute evaluator
   * 
   * @return a description of the attribute evaluator as a string
   */
  @Override
  public String toString() {
    StringBuffer text = new StringBuffer();

    if (m_InfoGains == null) {
      text.append("Information Gain attribute evaluator has not been built");
    } else {
      text.append("\tInformation Gain Ranking Filter");
      text.append("\n\tUsing "+((getN_threads()==Integer.MAX_VALUE)?("all ("+Runtime.getRuntime().availableProcessors()+")"):getN_threads())+" threads");
      if (!m_missing_merge) {
        text.append("\n\tMissing values treated as separate");
      }
      if (m_Binarize) {
        text.append("\n\tNumeric attributes are just binarized");
      }
      
    }

    text.append("\n");
    return text.toString();
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 1 $");
  }

  // ============
  // Test method.
  // ============
  /**
   * Main method for testing this class.
   * 
   * @param args the options
   */
  public static void main(String[] args) {
    runEvaluator(new InfoGainThreadedAttributeEval(), args);
  }
}
