import org.apache.spark.sql.SparkSession

object ToyExample2 extends App {
  val spark = SparkSession
    .builder()
    .appName("ToyExample")
    .config("spark.master", "local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val df = spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("data/train.csv")

  df.show()

  val Array(df_train, df_validation, df_test) = df.randomSplit(Array(0.6, 0.2, 0.2))  // les valeurs indiquées sont les proportions qui vont être utilisée pour le split

  // Ce split sera utilisé afin de trainer/entraîner le modèle
  // Ce split sera utilisé afin de valider le modèle ==> de finetuner les hyperparameters/weights/poids/coefficients+intercept
  // Ce split sera utiliser pour test le modèle ==> mesurer sa précision/accuracy

  df_train.show()
  df_validation.show()
  df_test.show()
}