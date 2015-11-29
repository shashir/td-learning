package edu.gatech.shashir.td

/**
 * Defines helper types.
 */
object Types {
  import RandomWalkStates.RandomWalkStates

  /**
   * Enum describing random walk states: A, B, C, D, E, F, G.
   */
  object RandomWalkStates extends Enumeration {
    type RandomWalkStates = Value
    val A, B, C, D, E, F, G = Value
  }

  /**
   * Type alias for random walks as a sequence of random walk states.
   */
  type RandomWalk = Seq[RandomWalkStates]

  /**
   * Type alias for training set as a sequence of random walks.
   */
  type TrainingSet = Seq[RandomWalk]

  /**
   * Implicit type conversion to allow scalar vector left-multiplication.
   *
   * @param scalar scalar value to left multiply.
   */
  implicit class Scalar(scalar : Double) {
    def *(vector: RandomWalkVector): RandomWalkVector = vector * scalar
  }
}
