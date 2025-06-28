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
 *    RandomTreeMSU.java
 *    Copyright (C) 2001-2012 University of Waikato, Hamilton, New Zealand
 *    Copyright (C) 2023-2023 University Pablo de Olavide, Seville, Spain
 *
 */
package weka.classifiers.trees;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import upo.jcu.math.MatrixUtils;
import upo.jcu.math.data.dataset.DataType;
import upo.jcu.math.stat.MultivariateStatUtils;
import upo.jml.data.dataset.ClassificationDataset;
import upo.jml.data.dataset.DatasetUtils;

/**
 * <!-- globalinfo-start --> Class for constructing a tree that considers K
 * randomly chosen attributes at each node. Performs no pruning. Also has an
 * option to allow estimation of class probabilities (or target mean in the
 * regression case) based on a hold-out set (backfitting). It uses a
 * Multivariate Symmetrical Uncertainty (MSU) meassure, instead of information
 * gain.<br>
 * <br>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p>
 *
 * <pre>
 * -K &lt;number of attributes&gt;
 *  Number of attributes to randomly investigate. (default 0)
 *  (&lt;1 = int(log_2(#predictors)+1)).
 * </pre>
 *
 * <pre>
 * -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 1)
 * </pre>
 *
 * <pre>
 * -V &lt;minimum variance for split&gt;
 *  Set minimum numeric class variance proportion
 *  of train variance for split (default 1e-3).
 * </pre>
 *
 * <pre>
 * -S &lt;num&gt;
 *  Seed for random number generator.
 *  (default 1)
 * </pre>
 *
 * <pre>
 * -depth &lt;num&gt;
 *  The maximum depth of the tree, 0 for unlimited.
 *  (default 0)
 * </pre>
 *
 * <pre>
 * -N &lt;num&gt;
 *  Number of folds for backfitting (default 0, no backfitting).
 * </pre>
 *
 * <pre>
 * -U
 *  Allow unclassified instances.
 * </pre>
 *
 * <pre>
 * -B
 *  Break ties randomly when several attributes look equally good.
 * </pre>
 *
 * <pre>
 * -output-debug-info
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 *
 * <pre>
 * -do-not-check-capabilities
 *  If set, classifier capabilities are not checked before classifier is built
 *  (use with caution).
 * </pre>
 *
 * <pre>
 * -num-decimal-places
 *  The number of decimal places for the output of numbers in the model (default 2).
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Paco Saucedo (fsaufer@alu.upo.es)
 * @author Miguel Garcia Torres (mgarciat@upo.es)
 * @version $Revision$
 */
public class RandomTreeMSU extends RandomTree {

    private boolean debug = false;
    /**
     * for serialization
     */
    private static final long serialVersionUID = -9051129597407396724L;

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {

        return "Class for constructing a tree that considers K randomly "
                + " chosen attributes at each node. Performs no pruning. Also has"
                + " an option to allow estimation of class probabilities (or target mean "
                + "in the regression case) based on a hold-out set (backfitting)."
                + "It uses a Multivariate Symmetrical Uncertainty (MSU) meassure, instead of information gain.";
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);

        return result;
    }

    /**
     * The inner class for dealing with the tree.
     */
    protected class Tree extends RandomTree.Tree implements Serializable {

        /**
         * For serialization
         */
        private static final long serialVersionUID = 3549573538656522569L;

        protected Tree() {
            super();
        }

        /**
         * Recursively generates a tree.
         *
         * @param data the data to work with
         * @param classProbs the class distribution
         * @param attIndicesWindow the attribute window to choose attributes
         * from
         * @param random random number generator for choosing random attributes
         * @param depth the current depth
         * @param msuSubset Multivariate Symmetrical Uncertainty (MSU)
         * attributes subset,
         * @throws Exception if generation fails
         */
        protected void buildTree(Instances data, double[] classProbs,
                int[] attIndicesWindow, double totalWeight, Random random, int depth,
                double minVariance, int[] msuSubset) throws Exception {
/*
            if (msuSubset != null) {
                System.out.println("MSU subset: " + Arrays.toString(msuSubset));
            } else {
                System.out.println("MSU subset: ---");
            }*/
            // Make leaf if there are no training instances
            if (data.numInstances() == 0) {
                m_Attribute = -1;
                m_ClassDistribution = null;
                m_Prop = null;

                if (data.classAttribute().isNumeric()) {
                    m_Distribution = new double[2];
                }
                return;
            }

            double priorVar = 0;
            if (data.classAttribute().isNumeric()) {

                // Compute prior variance
                double totalSum = 0, totalSumSquared = 0, totalSumOfWeights = 0;
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance inst = data.instance(i);
                    totalSum += inst.classValue() * inst.weight();
                    totalSumSquared
                            += inst.classValue() * inst.classValue() * inst.weight();
                    totalSumOfWeights += inst.weight();
                }
                priorVar = singleVariance(totalSum, totalSumSquared, totalSumOfWeights);
            }

            // Check if node doesn't contain enough instances or is pure
            // or maximum depth reached
            if (data.classAttribute().isNominal()) {
                totalWeight = Utils.sum(classProbs);
            }
            // System.err.println("Total weight " + totalWeight);
            // double sum = Utils.sum(classProbs);
            if (totalWeight < 2 * m_MinNum
                    || // Nominal case
                    (data.classAttribute().isNominal() && Utils.eq(
                    classProbs[Utils.maxIndex(classProbs)], Utils.sum(classProbs)))

                    || // Numeric case
                    (data.classAttribute().isNumeric() && priorVar / totalWeight < minVariance)

                    || // check tree depth
                    ((getMaxDepth() > 0) && (depth >= getMaxDepth()))) {

                // Make leaf
                m_Attribute = -1;
                m_ClassDistribution = classProbs.clone();
                if (data.classAttribute().isNumeric()) {
                    m_Distribution = new double[2];
                    m_Distribution[0] = priorVar;
                    m_Distribution[1] = totalWeight;
                }

                m_Prop = null;
                return;
            }

            // Compute class distributions and value of splitting
            // criterion for each attribute
            double val = -Double.MAX_VALUE;
            double split = -Double.MAX_VALUE;
            double[][] bestDists = null;
            double[] bestProps = null;
            int bestIndex = 0;

            // Handles to get arrays out of distribution method
            double[][] props = new double[1][0];
            double[][][] dists = new double[1][0][0];
            double[][] totalSubsetWeights = new double[data.numAttributes()][0];

            // Investigate K random attributes
            if (debug) System.out.println("==========================");
            int attIndex = 0;
            int windowSize = attIndicesWindow.length;
            int k = m_KValue;
            if (debug) System.out.println("\tatt indices window: " + Arrays.toString(attIndicesWindow));
            if (debug) System.out.println("\twindow size: " + windowSize);
            if (debug) System.out.println("\tk: " + k);
            boolean gainFound = false;
            double[] tempNumericVals = new double[data.numAttributes()];

            ClassificationDataset msuClassificationDataset = ClassificationDatasetAdapter.newInstance(data);
            int[][] msuData = MatrixUtils.transpose(msuClassificationDataset.getCategoricalData());
            int arrayIndexNewAttribute = msuSubset == null ? 0 : msuSubset.length;
            int[] msuTrialSubset = ClassificationDatasetAdapter.extendMsuSubset(msuSubset);
            int[] msuNewSelectedSubset = msuTrialSubset.clone();
            if (debug) System.out.println("\tmsu new selected subset: " + Arrays.toString(msuNewSelectedSubset));
            if (debug) System.out.println("\tmsu trial subset: " + Arrays.toString(msuTrialSubset));
            
            int idx = 0;
            while ((windowSize > 0) && (k-- > 0 || !gainFound)) {
                if (debug) System.out.println("\twindow size=" + windowSize + " (>0), k=" + k + " (>0)");
                int chosenIndex = random.nextInt(windowSize);
                if (debug) System.out.println("\t\tchosen index: " + chosenIndex + ", att indices window: " + Arrays.toString(attIndicesWindow));
                attIndex = attIndicesWindow[chosenIndex];
                if (debug) System.out.println("\t\tatt index: " + attIndex);
                // shift chosen attIndex out of window
                attIndicesWindow[chosenIndex] = attIndicesWindow[windowSize - 1];
                attIndicesWindow[windowSize - 1] = attIndex;
                if (debug) System.out.println("\t\tatt indices window: " + Arrays.toString(attIndicesWindow));
                windowSize--;
                if (debug) System.out.println("\t\twindow size: " + windowSize);
                
                double currSplit
                        = data.classAttribute().isNominal() ? distribution(props, dists,
                        attIndex, data) : numericDistribution(props, dists, attIndex,
                                totalSubsetWeights, data, tempNumericVals);

                msuTrialSubset[arrayIndexNewAttribute] = ClassificationDatasetAdapter.findAttIndex(data.attribute(attIndex).name(),
                        msuClassificationDataset.getCategoricalHeader());
                if (debug) System.out.println("\t\tcurrent split: " + currSplit);
                double currVal
                        = data.classAttribute().isNominal()
                        ? MultivariateStatUtils.symmetricalUncertainty(msuData,
                                msuClassificationDataset.getCategoricalCardinalities(),
                                msuTrialSubset,
                                msuClassificationDataset.getLabels(),
                                classProbs.length)
                        : tempNumericVals[attIndex];
                if (debug) System.out.println("\t\tvalue: " + val);
                if (debug) System.out.println("\t\tcurrent value: " + currVal);
                if (Utils.gr(currVal, 0)) {
                    gainFound = true;
                }

                if ((currVal > val)
                        || ((!getBreakTiesRandomly()) && (currVal == val) && (attIndex < bestIndex))) {
                    val = currVal;
                    bestIndex = attIndex;
                    split = currSplit;
                    bestProps = props[0];
                    bestDists = dists[0];
                    msuNewSelectedSubset[arrayIndexNewAttribute] = msuTrialSubset[arrayIndexNewAttribute];
                    if (debug) System.out.println("\t\tMSU new selected subset: " + Arrays.toString(msuNewSelectedSubset));
                }
            }

            
            // Taking into account that it's a recurive process, prepare to free memory through GC  
            msuClassificationDataset = null;
            msuData = null;

            // Find best attribute
            m_Attribute = bestIndex;
            if (debug) System.out.println("Best attribute: " + m_Attribute);
            //System.exit(-1);
            // Any useful split found?
            if (Utils.gr(val, 0)) {
                if (debug) System.out.println("\t\tinner node");
                
                if (m_computeImpurityDecreases) {
                    m_impurityDecreasees[m_Attribute][0] += val;
                    m_impurityDecreasees[m_Attribute][1]++;
                }

                // Build subtrees
                m_SplitPoint = split;
                m_Prop = bestProps;
                Instances[] subsets = splitData(data);
                m_Successors = new RandomTreeMSU.Tree[bestDists.length];
                double[] attTotalSubsetWeights = totalSubsetWeights[bestIndex];
                
                for (int i = 0; i < bestDists.length; i++) {
                    if (debug) System.out.println("\t\t\tsubtree i=" + i + " MSU new selected subset: " + Arrays.toString(msuNewSelectedSubset));
                    m_Successors[i] = new RandomTreeMSU.Tree();
                    ((RandomTreeMSU.Tree) m_Successors[i]).buildTree(subsets[i], bestDists[i], attIndicesWindow,
                            data.classAttribute().isNominal() ? 0 : attTotalSubsetWeights[i],
                            random, depth + 1, minVariance, msuNewSelectedSubset);
                }

                // If all successors are non-empty, we don't need to store the class
                // distribution
                boolean emptySuccessor = false;
                for (int i = 0; i < subsets.length; i++) {
                    if (m_Successors[i].m_ClassDistribution == null) {
                        emptySuccessor = true;
                        break;
                    }
                }
                if (emptySuccessor) {
                    m_ClassDistribution = classProbs.clone();
                }
            } else {
                if (debug) System.out.println("\t\tleaf node");
                // Make leaf
                m_Attribute = -1;
                m_ClassDistribution = classProbs.clone();
                if (data.classAttribute().isNumeric()) {
                    m_Distribution = new double[2];
                    m_Distribution[0] = priorVar;
                    m_Distribution[1] = totalWeight;
                }
            }
        }
    }

    /**
     * Adapter between Weka and MSU data structures
     */
    protected static class ClassificationDatasetAdapter implements Serializable {

        private static final ClassificationDatasetAdapter adapter = new ClassificationDatasetAdapter();

        private ClassificationDatasetAdapter() {
        }

        protected static ClassificationDataset newInstance(Instances instances) {
            int[] categoricalLabels = new int[instances.size()];
            int indexCategoricalLabels = 0;
            String[] labelNames = null;
            List<String> categoricalHeader = null;
            List<String[]> categoricalValueNames = null;
            List<List<Integer>> categoricalData = null;
            List<String> numericHeader = null;
            List<List<Double>> numericData = null;
            boolean isFirstInstance = true;

            for (Instance instance : instances) {
                List<Integer> instanceCategoricalData = null;
                List<Double> instanceNumericData = null;

                if (isFirstInstance) {
                    if (instance.classAttribute().isNominal()) {
                        labelNames = adapter.toStringArray(instance.classAttribute().enumerateValues());
                    } else {
                        throw new UnsupportedOperationException("Only nominal class attributes are supported");
                    }

                    Enumeration<Attribute> attributes = instance.enumerateAttributes();
                    Attribute attribute;
                    while (attributes.hasMoreElements()) {
                        attribute = attributes.nextElement();
                        if (attribute.isNominal()) {
                            if (categoricalHeader == null) {
                                categoricalHeader = new ArrayList<>();
                            }
                            if (categoricalValueNames == null) {
                                categoricalValueNames = new ArrayList<>();
                            }
                            if (categoricalData == null) {
                                categoricalData = new ArrayList<>();
                            }

                            categoricalHeader.add(attribute.name());
                            categoricalValueNames.add(adapter.toStringArray(attribute.enumerateValues()));
                        } else if (attribute.isNumeric()) {
                            if (numericHeader == null) {
                                numericHeader = new ArrayList<>();
                            }
                            if (numericData == null) {
                                numericData = new ArrayList<>();
                            }

                            numericHeader.add(attribute.name());
                        } else {
                            throw new UnsupportedOperationException("Only nominal and numeric attributes are supported");
                        }
                    }

                    isFirstInstance = false;
                }

                if (categoricalHeader != null && instanceCategoricalData == null) {
                    instanceCategoricalData = new ArrayList<>();
                }

                if (numericHeader != null && instanceNumericData == null) {
                    instanceNumericData = new ArrayList<>();
                }

                double[] instanceAttributesValues = instance.toDoubleArray();
                for (int i = 0; i < instanceAttributesValues.length; i++) {
                    if (i == instance.classIndex()) {
                        categoricalLabels[indexCategoricalLabels++] = (int) instanceAttributesValues[i];
                    } else if (instance.attribute(i).isNominal()) {
                        instanceCategoricalData.add((int) instanceAttributesValues[i]);
                    } else {
                        instanceNumericData.add(instanceAttributesValues[i]);
                    }
                }

                if (instanceCategoricalData != null) {
                    categoricalData.add(instanceCategoricalData);
                }

                if (instanceNumericData != null) {
                    numericData.add(instanceNumericData);
                }
            }

            ClassificationDataset dataset = new ClassificationDataset();
            dataset.setLabels(categoricalLabels);
            dataset.setLabelNames(labelNames);
            if (categoricalData != null && numericData == null) {
                dataset.setDataType(DataType.CATEGORICAL);

                dataset.setCategoricalHeader(adapter.toStringArray(categoricalHeader));
                dataset.setCategoricalValueNames(adapter.toStringMatrix(categoricalValueNames));
                dataset.setCategoricalData(adapter.toIntegerMatrix(categoricalData));
            } else if (categoricalData == null && numericData != null) {
                dataset.setDataType(DataType.NUMERICAL);

                dataset.setNumericalHeader(adapter.toStringArray(numericHeader));
                dataset.setNumericalData(adapter.toDoubleMatrix(numericData));

                dataset = adapter.discretize(dataset);
            } else {
                dataset.setDataType(DataType.NUMERICAL_CATEGORICAL);

                dataset.setCategoricalHeader(adapter.toStringArray(categoricalHeader));
                dataset.setCategoricalData(adapter.toIntegerMatrix(categoricalData));
                dataset.setCategoricalValueNames(adapter.toStringMatrix(categoricalValueNames));

                dataset.setNumericalHeader(adapter.toStringArray(numericHeader));
                dataset.setNumericalData(adapter.toDoubleMatrix(numericData));

                dataset = adapter.discretize(dataset);
            }

            return dataset;
        }

        protected static int findAttIndex(String attName, String[] categoricalHeader) {
            for (int i = 0; i < categoricalHeader.length; i++) {
                if (attName.equals(categoricalHeader[i])) {
                    return i;
                }
            }

            throw new RuntimeException("Attribute not found while looking for MSU equivalent");
        }

        protected static int[] extendMsuSubset(int[] msuSubset) {
            int[] msuNewSubset;

            if (msuSubset == null) {
                msuNewSubset = new int[1];
            } else {
                msuNewSubset = new int[msuSubset.length + 1];

                System.arraycopy(msuSubset, 0, msuNewSubset, 0, msuSubset.length);
            }

            return msuNewSubset;
        }

        private ClassificationDataset discretize(ClassificationDataset dataset) {
            try {
                if (dataset.getDataType().equals(DataType.NUMERICAL)) {
                    return DatasetUtils.dicretizeNumericalDatasetViaFayyad(dataset);
                } else {
                    return DatasetUtils.dicretizeMixedDatasetViaFayyad(dataset);
                }
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        }

        private int[][] toIntegerMatrix(List<List<Integer>> data) {
            int[][] result = new int[data.size()][];
            int rowIndex = 0;
            int columnIndex;

            for (List<Integer> row : data) {
                columnIndex = 0;
                result[rowIndex] = new int[row.size()];

                for (Integer columnValue : row) {
                    result[rowIndex][columnIndex++] = columnValue;
                }

                rowIndex++;
            }

            return result;
        }

        private double[][] toDoubleMatrix(List<List<Double>> data) {
            double[][] result = new double[data.size()][];
            int rowIndex = 0;
            int columnIndex;

            for (List<Double> row : data) {
                columnIndex = 0;
                result[rowIndex] = new double[row.size()];

                for (Double columnValue : row) {
                    result[rowIndex][columnIndex++] = columnValue;
                }

                rowIndex++;
            }

            return result;
        }

        private String[][] toStringMatrix(List<String[]> data) {
            String[][] result = new String[data.size()][];

            for (int i = 0; i < data.size(); i++) {
                result[i] = data.get(i);
            }

            return result;
        }

        private String[] toStringArray(Enumeration<Object> values) {
            List<String> stringList = new ArrayList<>();

            while (values.hasMoreElements()) {
                stringList.add((String) values.nextElement());
            }

            return toStringArray(stringList);
        }

        private String[] toStringArray(List<String> stringList) {
            String[] stringArray = new String[stringList.size()];
            for (int i = 0; i < stringArray.length; i++) {
                stringArray[i] = stringList.get(i);
            }

            return stringArray;
        }

    }

    protected Tree getTree() {
        return new RandomTreeMSU.Tree();
    }

    /**
     * first call*
     */
    protected void buildTree(Instances train, double[] classProbs, int[] attIndicesWindow, double totalWeight, Random rand, double trainVariance) throws Exception {
        ((RandomTreeMSU.Tree) m_Tree).buildTree(train, classProbs, attIndicesWindow, totalWeight, rand, 0,
                m_MinVarianceProp * trainVariance, null);
    }

    /**
     * Main method for this class.
     *
     * @param argv the commandline parameters
     */
    public static void main(String[] argv) {
        runClassifier(new RandomTreeMSU(), argv);
    }
}
