package edu.gatech.shashir.td

import java.io.File
import java.io.PrintWriter
import java.util.Random

import edu.gatech.shashir.td.Types.TrainingSet

/**
 * Driver for generating data sets for the figures 3, 4, 5 in Sutton 1988.
 */
object Driver {
  // Random number stuff.
  val SEED_1: Long = 918798907L
  val SEED_2: Long = System.currentTimeMillis()
  val RANDOM: Random = new Random(SEED_2)

  // Training set sizes.
  val NUM_SETS: Int = 100
  val NUM_SEQUENCES: Int = 10
  // Training data.
  val TRAINING_SETS: Seq[TrainingSet] = RandomWalkGenerator(SEED_1).generateTrainingSets(NUM_SETS, NUM_SEQUENCES)

  // Training termination conditions.
  val MAX_ITERATIONS: Int = 10000
  val TOLERANCE: Double = 0.0001

  // Values used to generate chart data.
  val LAMBDAS: Seq[Double] = Seq(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
  val LAMBDAS_FIGURE_4: Seq[Double] = Seq(0.0, 0.3, 0.8, 1.0)
  val ALPHAS: Seq[Double] = (0 to 12).map { a: Int => a / 20.}.toSeq
  val ALPHA_FIGURE_3: Double = 0.02
  val INITIAL_VALUE_FIGURE_3: Double = 0.5

  // File locations.
  val FIGURE_3_FILE: String = "/tmp/figure3_data.csv"
  val FIGURE_4_FILE: String = "/tmp/figure4_data.csv"
  val FIGURE_5_FILE: String = "/tmp/figure5_data.csv"

  /**
   * Generates data from figures 3, 4, 5 in Sutton 1988.
   *
   * @param args unused.
   */
  def main(args: Array[String]): Unit = {
    println("Generating figure 3 data at: " + FIGURE_3_FILE)
    generateFigure3(FIGURE_3_FILE)
    println("Generating figure 4 data at: " + FIGURE_4_FILE)
    generateFigure4(FIGURE_4_FILE)
    println("Generating figure 5 data at: " + FIGURE_5_FILE)
    generateFigure5(FIGURE_5_FILE)
    println("Complete!")
  }

  /**
   * Writes data for figure 3 to given file.
   * Computes error at each lambda using alpha of 0.02.
   * Trains in batch till convergence.
   *
   * @param file to write data.
   */
  def generateFigure3(file: String): Unit = {
    val writer: PrintWriter = new PrintWriter(new File(file))
    // Header.
    writer.println("lambda,rmse,std")

    // Go through the lambdas and the alpha to compute RMSEs for the data set.
    for (l: Double <- LAMBDAS) {
      // Train and compute RMSEs
      val td: TD = TD(l, ALPHA_FIGURE_3)
      val (rmse, std): (Double, Double) = td.getErrorOverTrainingSets(
        TRAINING_SETS,
        // Initialize randomly.
        w = RandomWalkVector(Seq(
          0.,
          RANDOM.nextDouble(),
          RANDOM.nextDouble(),
          RANDOM.nextDouble(),
          RANDOM.nextDouble(),
          RANDOM.nextDouble(),
          1.
        )),
        maxIterations = MAX_ITERATIONS,
        tolerance = TOLERANCE, // Train till convergence.
        online = false // Batch
      )
      // Write to file.
      writer.println(l + "," + rmse + "," + std)
      writer.flush()
    }
    writer.close()
  }

  /**
   * Writes data for figure 4 to given file.
   * Computes error at each alpha for 4 lambdas: 0.0, 0.3, 0.8, 1.0.
   * Trains online for 1 pass through the data.
   *
   * @param file to write data.
   */
  def generateFigure4(file: String): Unit = {
    val writer: PrintWriter = new PrintWriter(new File(file))
    // Header.
    writer.println("alpha,0.0,0.3,0.8,1.0")

    // Go through the lambdas and the alphas to compute RMSEs for the data set.
    for (a: Double <- ALPHAS) {
      val rmse: Seq[Double] = for (l: Double <- LAMBDAS_FIGURE_4) yield {
        val td: TD = TD(l, a)
        val (rmse, std): (Double, Double) = td.getErrorOverTrainingSets(
          TRAINING_SETS,
          // Initialize at 0.5
          w = RandomWalkVector(Seq(
            0.,
            INITIAL_VALUE_FIGURE_3,
            INITIAL_VALUE_FIGURE_3,
            INITIAL_VALUE_FIGURE_3,
            INITIAL_VALUE_FIGURE_3,
            INITIAL_VALUE_FIGURE_3,
            1.
          )),
          maxIterations = 1,
          tolerance = -1.,
          online = true
        )
        rmse
      }
      // Write to file.
      writer.println(a + "," + rmse.map(_.toString).mkString(","))
      writer.flush()
    }
    writer.close()
  }

  /**
   * Writes data for figure 5 to given file.
   * Computes error for each lambda at the best corresponding alpha for that lambda.
   * Trains online for 1 pass through the data.
   *
   * @param file to write data.
   */
  def generateFigure5(file: String): Unit = {
    // Compute map of the best alphas for each lambda.
    val bestAlphaMap: Map[Double, Double] = {for (l: Double <- LAMBDAS) yield {
      // Compute errors for each alpha, sort by error and pick best alpha to minimize error.
      val bestAlpha: Double = {for (a: Double <- ALPHAS) yield {
        val td: TD = TD(l, a)
        val (rmse, std): (Double, Double) = td.getErrorOverTrainingSets(
          TRAINING_SETS,
          w = RandomWalkVector(Seq(
            0.,
            INITIAL_VALUE_FIGURE_3,
            INITIAL_VALUE_FIGURE_3,
            INITIAL_VALUE_FIGURE_3,
            INITIAL_VALUE_FIGURE_3,
            INITIAL_VALUE_FIGURE_3,
            1.
          )),
          maxIterations = 1, // Single pass.
          tolerance = -1., // Do not terminate on convergence
          online = true
        )
        (a, rmse)
      }}.sortBy(_._2).head._1
      // Emit lambda and best alpha pair.
      (l, bestAlpha)
    }}.toMap


    val writer: PrintWriter = new PrintWriter(new File(file))
    // Header.
    writer.println("lambda,rmse,std")

    // Go through the lambdas and their best alphas to compute RMSEs for the data set.
    for (l: Double <- LAMBDAS) {
      val td: TD = TD(l, bestAlphaMap.get(l).get)
      val (rmse, std): (Double, Double) = td.getErrorOverTrainingSets(
        TRAINING_SETS,
        w = RandomWalkVector(Seq(
          0.,
          INITIAL_VALUE_FIGURE_3,
          INITIAL_VALUE_FIGURE_3,
          INITIAL_VALUE_FIGURE_3,
          INITIAL_VALUE_FIGURE_3,
          INITIAL_VALUE_FIGURE_3,
          1.
        )),
        maxIterations = 1, // Single pass.
        tolerance = -1., // Do not terminate on convergence
        online = true
      )
      // Write to file.
      writer.println(l + "," + rmse + "," + std)
      writer.flush()
    }
    writer.close()
  }
}
