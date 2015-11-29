package edu.gatech.shashir.td

/**
 * Implementation of 6-dimensional euclidean vector with +, -, * (dot product), * (scalar multiplication), and norm.
 *
 * These vectors can be used to defined vectorized states of the random walks, the weights vectors, deltas,
 * or even errors.
 *
 * @param vector sequence of double values backing the vector.
 */
final case class RandomWalkVector(vector: Seq[Double]) {
  // Ensure size 6.
  require(vector.length == RandomWalkVector.SIZE)

  // Addition
  def +(that: RandomWalkVector): RandomWalkVector = {
    return RandomWalkVector(this.vector.zip(that.vector).map {
      case (thisComp: Double, thatComp: Double) => thisComp + thatComp
    })
  }

  // Subtract
  def -(that: RandomWalkVector): RandomWalkVector = {
    return RandomWalkVector(this.vector.zip(that.vector).map {
      case (thisComp: Double, thatComp: Double) => thisComp - thatComp
    })
  }

  // Dot product
  def *(that: RandomWalkVector): Double = {
    return this.vector.zip(that.vector).map {
      case (thisComp: Double, thatComp: Double) => thisComp * thatComp
    }.sum
  }

  // Scalar product
  def *(that: Double): RandomWalkVector = {
    return RandomWalkVector(this.vector.map { comp => comp * that })
  }

  // Norm
  def norm(): Double = Math.sqrt(this * this)

  // Printable string.
  override def toString(): String = {
    return vector.mkString(", ")
  }
}

object RandomWalkVector {
  val SIZE: Int = 7

  /**
   * Factory method for constructing i'th basis vector.
   * 0th basis vector is = (1, 0, 0, 0, 0, 0, 0)
   * 1st basis vector is = (0, 1, 0, 0, 0, 0, 0)
   * ...
   *
   * @param i basis vector index.
   * @return i'th basis vector.
   */
  def apply(i: Int): RandomWalkVector = {
    require(i < SIZE && i >= 0)
    return RandomWalkVector((0 until i).map { _ => 0.}.toSeq ++ Seq(1.) ++ (i + 1 until SIZE).map { _ => 0.}.toSeq)
  }

  /**
   * @return zero vector.
   */
  def zeroes(): RandomWalkVector = {
    return RandomWalkVector((0 until SIZE).map { _ => 0.})
  }
}
