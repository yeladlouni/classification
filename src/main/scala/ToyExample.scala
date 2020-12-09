import org.apache.spark.sql.functions.round
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession


object ToyExample extends App {
  val spark = SparkSession
    .builder()
    .appName("ToyExample")
    .config("spark.master", "local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val df_train= spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("data/train.csv")

  val df_test= spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("data/test.csv")

  df_train.printSchema()
  df_test.printSchema()

  df_train.show()
  df_test.show()

  val supervised = new RFormula()
    .setFormula("salary ~ age")  // 2 x + 2000 + delta, pour notre cas j'ai utilisé un delta à 0 pour avoir un modèle parfait

  val fittedRF = supervised.fit(df_train)  // entraînement du modèle: algorithme
  val prepared = fittedRF.transform(df_train) // On applique le modèle pour les données training
  val predicted = fittedRF.transform(df_test) // On applique le modèle sur les données test

  println(fittedRF)

  prepared.show()
  predicted.show()

  val lr = new LinearRegression()  // On utilise un modèle de régression linéaire pour prédire

  val model = lr.fit(prepared)

  println(model.coefficients(0).round)
  println(model.intercept.round)
  

}