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
 *    RandomForestMSU.java
 *    Copyright (C) 2001-2012 University of Waikato, Hamilton, New Zealand
 *    Copyright (C) 2023-2023 University Pablo de Olavide, Seville, Spain
 *
 */

package weka.classifiers.trees;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.gui.ProgrammaticProperty;

import weka.classifiers.trees.RandomForest;

/**
 * <!-- globalinfo-start --> Class for constructing a forest of random trees. 
 * It uses a Multivariate Symmetrical Uncertainty (MSU) meassure, instead of information gain.<br>
 * <br>
 * For more information see: <br>
 * <br>
 * Leo Breiman (2001). Random Forests. Machine Learning. 45(1):5-32. <br>
 * <br>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;article{Breiman2001,
 *    author = {Leo Breiman},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {5-32},
 *    title = {Random Forests},
 *    volume = {45},
 *    year = {2001}
 * }
 * </pre>
 * 
 * <br>
 * <br>
 * <!-- technical-bibtex-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p>
 * 
 * <pre>
 * -P
 *  Size of each bag, as a percentage of the
 *  training set size. (default 100)
 * </pre>
 * 
 * <pre>
 * -O
 *  Calculate the out of bag error.
 * </pre>
 * 
 * <pre>
 * -store-out-of-bag-predictions
 *  Whether to store out of bag predictions in internal evaluation object.
 * </pre>
 * 
 * <pre>
 * -output-out-of-bag-complexity-statistics
 *  Whether to output complexity-based statistics when out-of-bag evaluation is performed.
 * </pre>
 * 
 * <pre>
 * -print
 *  Print the individual classifiers in the output
 * </pre>
 * 
 * <pre>
 * -attribute-importance
 *  Compute and output attribute importance (mean impurity decrease method)
 * </pre>
 * 
 * <pre>
 * -I &lt;num&gt;
 *  Number of iterations (i.e., the number of trees in the random forest).
 *  (current value 100)
 * </pre>
 * 
 * <pre>
 * -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)
 *  (use 0 to auto-detect number of cores)
 * </pre>
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
 * <pre>
 * -batch-size
 *  The desired batch size for batch prediction  (default 100).
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Paco Saucedo (fsaufer@alu.upo.es)
 * @version $Revision$
 */
public class RandomForestMSU extends RandomForest {

  /** for serialization */
  static final long serialVersionUID = 1116839470761428688L;
  
  /**
   * Constructor that sets base classifier for bagging to RandomTre and default
   * number of iterations to 100.
   */
  public RandomForestMSU() {
    RandomTreeMSU rTree = new RandomTreeMSU();
    rTree.setDoNotCheckCapabilities(true);
    super.setClassifier(rTree);
    super.setRepresentCopiesUsingWeights(true);
    setNumIterations(defaultNumberOfIterations());
  }

  /**
   * Returns default capabilities of the base classifier.
   *
   * @return the capabilities of the base classifier
   */
  public Capabilities getCapabilities() {

    // Cannot use the main RandomTreeMSU object because capabilities checking has
    // been turned off
    // for that object.
    return (new RandomTreeMSU()).getCapabilities();
  }

  /**
   * String describing default classifier.
   *
   * @return the default classifier classname
   */
  @Override
  protected String defaultClassifierString() {

    return "weka.classifiers.trees.RandomTreeMSU";
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing detailed
   * information about the technical background of this class, e.g., paper
   * reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  @Override
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation result;

    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "Gustavo Sosa-Cabrera, Miguel García-Torres, Santiago Gómez-Guerrero, Christian E. Schaerer, Federico Divina");
    result.setValue(Field.YEAR, "2019");
    result.setValue(Field.TITLE, "A multivariate approach to the symmetrical uncertainty measure: Application to feature selection problem");
    result.setValue(Field.JOURNAL, "Science Direct");
    
    return result;
  }
  
  /**
   * This method only accepts RandomTreeMSU arguments.
   *
   * @param newClassifier the RandomTreeMSU to use.
   * @exception if argument is not a RandomTreeMSU
   */
  @Override
  @ProgrammaticProperty
  public void setClassifier(Classifier newClassifier) {
    if (!(newClassifier instanceof RandomTreeMSU)) {
      throw new IllegalArgumentException(
        "RandomForest: Argument of setClassifier() must be a RandomTreeMSU.");
    }
    super.setClassifier(newClassifier);
  }

  /**
   * Main method for this class.
   * 
   * @param argv the options
   */
  public static void main(String[] argv) {
    runClassifier(new RandomForestMSU(), argv);
  }
}
