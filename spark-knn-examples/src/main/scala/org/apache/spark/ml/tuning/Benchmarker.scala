package org.apache.spark.ml.tuning

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/**
  * Params for [[Benchmarker]] and [[BenchmarkModel]].
  */
private[ml] trait BenchmarkerParams extends ValidatorParams {
  /**
    * Param for number of times for benchmark.  Must be >= 1.
    * Default: 1
    * @group param
    */
  val numTimes: IntParam = new IntParam(this, "numTimes",
    "number of times for benchmark (>= 1)", ParamValidators.gtEq(1))

  /** @group getParam */
  def getNumTimes: Int = $(numTimes)

  setDefault(numTimes -> 1)
}

/**
  * :: Experimental ::
  * Benchmark estimator pipelines.
  */
@Experimental
class Benchmarker(override val uid: String) extends Estimator[BenchmarkModel]
with BenchmarkerParams {

  def this() = this(Identifiable.randomUID("benchmark"))

  private val f2jBLAS = new F2jBLAS

  /** @group setParam */
  def setEstimator(value: Estimator[_]): this.type = set(estimator, value)

  /** @group setParam */
  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  /** @group setParam */
  def setEvaluator(value: Evaluator): this.type = set(evaluator, value)

  /** @group setParam */
  def setNumTimes(value: Int): this.type = set(numTimes, value)

  override def fit(dataset: Dataset[_]): BenchmarkModel = {
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val sqlCtx = dataset.sqlContext
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val models = new Array[Model[_]](epm.length)
    val trainingRuntimes = new Array[Double](epm.length)
    val evaluationRuntimes = new Array[Double](epm.length)
    (1 to getNumTimes).foreach { index =>
      // multi-model training
      logDebug(s"Train $index times with multiple sets of parameters.")
      var i = 0
      while (i < numModels) {
        var tic = System.currentTimeMillis()
        models(i) = est.fit(dataset, epm(i)).asInstanceOf[Model[_]]
        trainingRuntimes(i) += System.currentTimeMillis() - tic

        tic = System.currentTimeMillis()
        val metric = eval.evaluate(models(i).transform(dataset, epm(i)))
        evaluationRuntimes(i) += System.currentTimeMillis() - tic

        logDebug(s"Got metric $metric for model trained with ${epm(i)}.")
        i += 1
      }
    }

    f2jBLAS.dscal(numModels, 1.0 / $(numTimes), trainingRuntimes, 1)
    f2jBLAS.dscal(numModels, 1.0 / $(numTimes), evaluationRuntimes, 1)
    logInfo(s"Average training runtimes: ${trainingRuntimes.toSeq}")
    logInfo(s"Average evaluation runtimes: ${evaluationRuntimes.toSeq}")
    val (fastestRuntime, fastestIndex) = trainingRuntimes.zipWithIndex.minBy(_._1)
    logInfo(s"Fastest set of parameters:\n${epm(fastestIndex)}")
    logInfo(s"Fastest training runtime: $fastestRuntime.")

    copyValues(new BenchmarkModel(uid, models(fastestIndex), trainingRuntimes, evaluationRuntimes).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    $(estimator).transformSchema(schema)
  }

  override def validateParams(): Unit = {
    super.validateParams()
    val est = $(estimator)
    for (paramMap <- $(estimatorParamMaps)) {
      est.copy(paramMap).validateParams()
    }
  }

  override def copy(extra: ParamMap): Benchmarker = {
    val copied = defaultCopy(extra).asInstanceOf[Benchmarker]
    if (copied.isDefined(estimator)) {
      copied.setEstimator(copied.getEstimator.copy(extra))
    }
    if (copied.isDefined(evaluator)) {
      copied.setEvaluator(copied.getEvaluator.copy(extra))
    }
    copied
  }
}

/**
  * :: Experimental ::
  * Model from benchmark runs.
  */
@Experimental
class BenchmarkModel private[ml](
                                  override val uid: String,
                                  val fastestModel: Model[_],
                                  val avgTrainingRuntimes: Array[Double],
                                  val avgEvaluationRuntimes: Array[Double])
  extends Model[BenchmarkModel] with BenchmarkerParams {

  override def validateParams(): Unit = {
    fastestModel.validateParams()
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    fastestModel.transform(dataset)
  }

  override def transformSchema(schema: StructType): StructType = {
    fastestModel.transformSchema(schema)
  }

  override def copy(extra: ParamMap): BenchmarkModel = {
    val copied = new BenchmarkModel(
      uid,
      fastestModel.copy(extra).asInstanceOf[Model[_]],
      avgTrainingRuntimes.clone(),
      avgEvaluationRuntimes.clone())
    copyValues(copied, extra).setParent(parent)
  }
}
