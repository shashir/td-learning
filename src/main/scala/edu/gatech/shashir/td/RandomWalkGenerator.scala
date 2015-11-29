package edu.gatech.shashir.td

import java.util.Random
import Types._
import RandomWalkStates._

/**
 * Generates random walks based on a seed. All walks start from state D and end on either state A or G.
 * Every step of the random walk proceeds to one alphabetical character before or after the current step.
 *
 * @param seed random seed.
 */
final case class RandomWalkGenerator(
  seed: Long = System.currentTimeMillis()
) {
  // Random number generator.
  private val random: Random = new Random(seed)

  /**
   * Recursively generate required number of training sets with required number of sequences each.
   *
   * @param sets number of training sets to generate.
   * @param sequencesPerSet number of walks each set must contain.
   * @return a set of training sets containing the required number of walks each.
   */
  def generateTrainingSets(
    sets: Int,
    sequencesPerSet: Int
  ): Seq[TrainingSet] = {
    return for (set <- 0 until sets) yield {
      for (seq <- 0 until sequencesPerSet) yield generateSequence()
    }
  }

  /**
   * Recursively, generates a random walk.
   *
   * @param walk accumulator for the random walk. Contains D by default.
   * @return walk starting from D and terminating at A or G.
   */
  def generateSequence(walk: RandomWalk = Seq(RandomWalkStates.D)): RandomWalk = {
    val r: Boolean = random.nextBoolean()
    // Proceed one step before or after the current letter. If A or G, return walk.
    return walk.last match {
      case A | G => walk
      case B => generateSequence(walk.:+(if (r) A else C))
      case C => generateSequence(walk.:+(if (r) B else D))
      case D => generateSequence(walk.:+(if (r) C else E))
      case E => generateSequence(walk.:+(if (r) D else F))
      case F => generateSequence(walk.:+(if (r) E else G))
    }
  }
}