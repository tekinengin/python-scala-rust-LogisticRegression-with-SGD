
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object lrwithsgd{
	def load_csv(filename:String) : ArrayBuffer[ArrayBuffer[Double]] = {
	  var dateset = ArrayBuffer[ArrayBuffer[Double]]()
	  val file = Source.fromFile(filename)
	  for (line <- file.getLines) {
	    val cols = line.split(",")
	    var temp = ArrayBuffer[Double]()
	    for (x <- cols) {
	      temp.append(x.toFloat)
	    }
	    dateset.append(temp)
	  }
	  file.close()
	  dateset
	}

	def train_test_split(dataset : ArrayBuffer[ArrayBuffer[Double]],labels : ArrayBuffer[ArrayBuffer[Double]], ratio:Double = 0.3): Array[ArrayBuffer[ArrayBuffer[Double]]] = {
	  var (dataset_train, dataset_test) = (dataset.clone(), ArrayBuffer[ArrayBuffer[Double]]())
	  var (label_train, label_test) = (labels.clone(), ArrayBuffer[ArrayBuffer[Double]]())
	  while (dataset_test.length < (dataset.length * ratio)){
	    var index = Random.nextInt(dataset_train.length)
	    dataset_test.append(dataset_train.remove(index))
	    label_test.append(label_train.remove(index))
	  }
	  Array[ArrayBuffer[ArrayBuffer[Double]]](dataset_train, label_train,dataset_test, label_test)
	}

	def accuracy_metric(actual:ArrayBuffer[ArrayBuffer[Double]], predicted:ArrayBuffer[Double]) :Double = {
	  var correct = 0
	  for (i <- 0 until actual.length) {
	    if (actual(i)(0) == predicted(i)) {
	      correct += 1
	    }
	  }
	  correct / actual.length.toDouble * 100.0
	}
	// Make a prediction with coefficients
	def predict(row:ArrayBuffer[Double], coefficients:ArrayBuffer[Double]):Double = {
	  var yhat = 0.0
	  for (i <- 0 until row.length) {
	    yhat += coefficients(i) * row(i)
	  }

	  1.0 / (1.0 + Math.exp(-yhat))
	}

	// Estimate logistic regression coefficients using stochastic gradient descent
	def coefficients_sgd(X_train:ArrayBuffer[ArrayBuffer[Double]],y_train:ArrayBuffer[ArrayBuffer[Double]], l_rate:Double, n_epoch:Int): ArrayBuffer[Double] = {
	  var coef = ArrayBuffer.fill[Double](X_train(0).length)(0)
	  for (_ <- 0 until n_epoch) {
	    for ((row,i) <- X_train.zipWithIndex) {
	      var yhat = predict(row, coef)
	      var error = y_train(i)(0) - yhat
	      for (i <- 0 until row.length) {
	        coef(i) = coef(i) + l_rate * error * yhat * (1.0 - yhat) * row(i)
	      }
	    }
	  }
	  coef
	}

	// Linear Regression Algorithm With Stochastic Gradient Descent
	def logistic_regression(data:Array[ArrayBuffer[ArrayBuffer[Double]]], l_rate:Double, n_epoch:Int): Double = {
	  var predictions = ArrayBuffer[Double]()
	  var coef = coefficients_sgd(data(0),data(1), l_rate, n_epoch)
	  for (row <- data(2)) {
	    var yhat = Math.round(predict(row, coef))
	    predictions.append(yhat)
	  }
	  accuracy_metric(data(3), predictions)
	}

	def main(args: Array[String]) {
		var n_size = 100000
		var l_rate = 0.1
		var n_epoch = 1

		for (arg <- args){
			val var_ = arg.split("=")
			var_(0) match {
				case "--n_size" => n_size = var_(1).toInt
				case "--l_rate" => l_rate = var_(1).toDouble
				case "--n_epoch" => n_epoch = var_(1).toInt
			}
		}

		Random.setSeed(1)

		val data_file = "data/moon_data_%d.csv".format(n_size)
		val label_file = "data/moon_labels_%d.csv".format(n_size)

		var dataset = load_csv(data_file)
		var labels = load_csv(label_file)

		var data = train_test_split(dataset,labels)
		var score = logistic_regression(data, l_rate, n_epoch)
		println(score)
	}

}


