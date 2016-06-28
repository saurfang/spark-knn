from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.param.shared import *
from pyspark.mllib.common import inherit_doc
from pyspark.ml.util import keyword_only


@inherit_doc
class KNNRegression(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol,
                    HasInputCols, HasThresholds, HasSeed, HasWeightCol):
    @keyword_only
    def __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                 seed=None, topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30, bufferSize=-1.0,
                 bufferSizeSampleSize=list(range(100, 1000 + 1, 100)), balanceThreshold=0.7,
                 k=5, neighborsCol="neighbors", maxNeighbors=float("inf")):
        super(KNNRegression, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.regression.KNNRegression", self.uid)

        self.topTreeSize = Param(self, "topTreeSize", "todo")
        self.topTreeLeafSize = Param(self, "topTreeLeafSize", "todo")
        self.subTreeLeafSize = Param(self, "subTreeLeafSize", "todo")
        self.bufferSize = Param(self, "bufferSize", "todo")
        self.bufferSizeSampleSize = Param(self, "bufferSizeSampleSize", "todo")
        self.balanceThreshold = Param(self, "balanceThreshold", "todo")
        self.k = Param(self, "k", "todo")
        self.neighborsCol = Param(self, "neighborsCol", "todo")
        self.maxNeighbors = Param(self, "maxNeighbors", "todo")

        self._setDefault(topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30, bufferSize=-1.0,
                         bufferSizeSampleSize=list(range(100, 1000 + 1, 100)), balanceThreshold=0.7,
                         k=5, neighborsCol="neighbors", maxNeighbors=float("inf"))

        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                  seed=None, topTreeSize=1000, topTreeLeafSize=10, subTreeLeafSize=30, bufferSize=-1.0,
                  bufferSizeSampleSize=list(range(100, 1000 + 1, 100)), balanceThreshold=0.7,
                  k=5, neighborsCol="neighbors", maxNeighbors=float("inf")):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return KNNRegressionModel(java_model)


class KNNRegressionModel(JavaModel):
    """
    Model fitted by KNNRegression.
    """
    def __init__(self, java_model):
        super(KNNRegressionModel, self).__init__(java_model)

        # note: look at https://issues.apache.org/jira/browse/SPARK-10931 in the future
        self.bufferSize = Param(self, "bufferSize", "todo")
        self.k = Param(self, "k", "todo")
        self.neighborsCol = Param(self, "neighborsCol", "todo")
        self.maxNeighbors = Param(self, "maxNeighbors", "todo")

        self._transfer_params_from_java()
