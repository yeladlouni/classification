import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator


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

  val df_validation = spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("data/validation.csv")

  val df_test = spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("data/test.csv")


  df_train.printSchema()
  df_validation.printSchema()

  df_train.show()
  df_validation.show()

  val formula = new RFormula()
    .setFormula("salary ~ age")  // 2 x + 2000 + delta, pour notre cas j'ai utilisé un delta à 0 pour avoir un modèle parfait

  val fittedRF = formula.fit(df_train)  // entraînement du modèle: algorithme
  val df_train_prep = fittedRF.transform(df_train) // On applique le modèle pour les données training
  val df_validation_prep = fittedRF.transform(df_validation) // On applique le modèle sur les données de validation


  df_train_prep.show()
  df_validation_prep.show()

  val lr = new LinearRegression()  // On utilise un modèle de régression linéaire pour prédire

  val model = lr.fit(df_train_prep)

  println(model.coefficients)
  println(model.intercept)

  val predictions = model.transform(df_validation_prep)

  predictions.show()

  // On peut utiliser le même modèle pour prédire de nouvelles valeurs

  val new_employee_age = 77
  val new_employee_salary = model.predict(Vectors.dense(new_employee_age))
  println(new_employee_salary)

  df_test.foreach(r => {
    val id = r.get(0)
    val age = r.get(1).asInstanceOf[Int].toDouble

    println(age)

    val predicted_salary = model.predict(Vectors.dense(age))

    println(id, age, predicted_salary)
  })


}