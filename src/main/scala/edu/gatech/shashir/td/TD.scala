package edu.gatech.shashir.td

import scala.annotation.tailrec

import edu.gatech.shashir.td.Types._
import edu.gatech.shashir.td.Types.RandomWalkStates._

/**
 * Performs temporal difference learning.
 *
 * @param lambda history weighting parameter.
 * @param alpha learning rate.
 */
final case class TD(
  lambda: Double,
  alpha: Double
) {
  import TD._

  /**
   * Get average root-mean-square error (RMSE) between the learned weights and the ideal weights.
   * Each RMSE is computed per training set and then averaged across all training sets.
   * This method also returns standard deviation of the RMSEs across all training sets.
   *
   * @param trainingSets to train on.
   * @param w initial weights vector.
   * @param maxIterations maximum iterations to run for.
   * @param tolerance when the difference of subsequent weights vector is below tolerance, terminate.
   * @param online perform learning in an "online" fashion, i.e. up update weights vector after every walk.
   * @return a pair of RMSE average and RMSE standard deviation.
   */
  def getErrorOverTrainingSets(
    trainingSets: Seq[TrainingSet],
    w: RandomWalkVector,
    maxIterations: Int,
    tolerance: Double,
    online: Boolean
  ): (Double, Double) = {
    // Compute RMSE per trainin set.
    val rmses: Seq[Double] = for (trainingSet <- trainingSets) yield {
      val diff = TARGET_WEIGHTS - train(trainingSet, w, maxIterations, tolerance, online)
      // Root of the mean of the square.
      Math.sqrt(diff * diff / 5)
    }
    // Compute average RMSE.
    val avg = rmses.sum / rmses.size
    // Compute standard deviation of RMSEs.
    val std = Math.sqrt(rmses.map { a => Math.pow(a - avg, 2)}.sum / rmses.size)

    return (avg, std)
  }

  /**
   * Recursively, trains on a training set of random walks and returns a weights vector.
   *
   * @param trainingSet to train on.
   * @param w initial weights vector.
   * @param maxIterations maximum iterations to run for.
   * @param tolerance when the difference of subsequent weights vector is below tolerance, terminate.
   * @param online perform learning in an "online" fashion, i.e. up update weights vector after every walk.
   * @return trained weights vector.
   */
  @tailrec def train(
    trainingSet: TrainingSet,
    w: RandomWalkVector,
    maxIterations: Int,
    tolerance: Double,
    online: Boolean
  ): RandomWalkVector = {
    if (maxIterations == 0) {
      // Terminate if maximum iterations are exhausted.
      return w
    } else {
      // Compute new weights vector in batch or online as required.
      val newW: RandomWalkVector = trainingSet.foldLeft(w) { case (wPrev: RandomWalkVector, walk: RandomWalk) =>
        // Get deltas for each walk in the training set.
        val dw: RandomWalkVector = this.computeDeltas(
          walk,
          // If online, propagate using new weights, else if batch, propagate initial weights vector.
          if (online) wPrev else w
        )
        // Update weights with the sum of the deltas (delta rule).
        wPrev + dw
      }

      // Check if L_\infty norm exceeds tolerance.
      if ((newW - w).vector.map(Math.abs(_)).max < tolerance) {
        // If L_\infty norm exceeds tolerance, we have converged, now return.
        return newW
      } else {
        // If weights have not converged, keep training.
        return train(trainingSet, newW, maxIterations - 1, tolerance, online)
      }
    }
  }

  /**
   * Recursively, computes the (temporal difference) cumulative delta of the weights vector on a given walk.
   * See (pp. 15-16, Sutton 1988, equation (4)).
   *
   * @param walk random walk to compute delta for.
   * @param w current weights vector.
   * @param dw precomputed delta (used as the accumulator of deltas in recursion).
   * @param e the sum of the weighted gradients (see p. 15, Sutton, 1988)
   * @return delta for the weights vector.
   */
  @tailrec def computeDeltas(
    walk: RandomWalk,
    w: RandomWalkVector,
    dw: RandomWalkVector = RandomWalkVector.zeroes(),
    e: RandomWalkVector = RandomWalkVector.zeroes()
  ): RandomWalkVector = {
    if (walk.length <= 1) {
      // If we have gotten to the end of the walk, then return the accumulated deltas.
      return dw
    } else {
      // Get basis vector of current state.
      val x: RandomWalkVector = RandomWalkVector(walk.head.id)
      // Get basis vector of next state.
      val xNext: RandomWalkVector = RandomWalkVector(walk.tail.head.id)
      // Compute predicted value of the current state.
      val P: Double = w * x
      // Compute predicted value of the next state
      // (unless we reach states A or G, in which case return values 0 and 1,respectively).
      val PNext: Double = if (walk.tail.head.equals(A)) 0. else if (walk.tail.head.equals(G)) 1. else w * xNext
      // Compute the sum of the weighted gradients (see p. 15, Sutton, 1988)
      val eNext: RandomWalkVector = x + e * lambda
      // Compute delta (see equation (4) p, 15, Sutton, 1988).
      val dwNext = alpha * (PNext - P) * eNext
      // Recursively compute more deltas.
      return computeDeltas(
        walk.tail,
        w,
        dw + dwNext,
        eNext
      )
    }
  }
}

/**
 * Companion object for static fields.
 */
object TD {
  // Target weights.
  val TARGET_WEIGHTS: RandomWalkVector = RandomWalkVector(Seq(0., 1/6., 1/3., 1/2., 2/3., 5/6., 1.))
}