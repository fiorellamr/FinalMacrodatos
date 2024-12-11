import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import scala.sys.process._

object EntrenamientoLLM {
  def main(args: Array[String]): Unit = {
    // =======================
    // 1. Configuración de Spark
    // =======================
    val spark = SparkSession.builder()
      .appName("Entrenamiento y Consulta de LLM con Scala y Spark")
      .master("local[*]") // Cambiar según el entorno
      .getOrCreate()

    // =======================
    // 2. Configuración de conexión a PostgreSQL
    // =======================
    val jdbcUrl = "jdbc:postgresql://localhost:5432/diabetes" // Cambiar <host> y <puerto>
    val tabla = "datos_pacientes"
    val user = "postgres"
    val password = "mani"

    // Leer datos desde PostgreSQL
    val dfPacientes = spark.read
      .format("jdbc")
      .option("url", jdbcUrl)
      .option("dbtable", s"(SELECT * FROM $tabla ORDER BY random() LIMIT (SELECT COUNT(*) * 0.002 FROM $tabla)) as subquery")
      .option("user", user)
      .option("password", password)
      .option("driver", "org.postgresql.Driver")
      .load()

    println("Datos cargados aleatoriamente desde PostgreSQL (20% de la tabla)")

    // =======================
    // 3. Preprocesamiento de los datos
    // =======================
    val dfTexto = dfPacientes.select("texto")

    val dfLimpio = dfTexto
      .withColumn("texto_limpio", regexp_replace(col("texto"), "Ã±", "ñ")) // Reemplazar caracteres
      .withColumn("texto_limpio", regexp_replace(col("texto_limpio"), "\\s+", " ")) // Normalizar espacios
      .withColumn("texto_limpio", trim(col("texto_limpio")))

    val dfFiltrado = dfLimpio.filter(length(col("texto_limpio")) > 10)

    // Guardar el corpus en un archivo de texto
    val outputPath = "C:\\Users\\USUARIO\\datos_llm_corpus.txt" // Cambiar a HDFS o S3 si es necesario
    dfFiltrado.select("texto_limpio").write.mode("overwrite").text(outputPath)

    println(s"Corpus generado y guardado en: $outputPath")

    // =======================
    // 4. Entrenamiento del modelo
    // =======================
    val trainScriptPath = "train_llm.py" // Ruta al script de entrenamiento en Python
    val modeloOutputPath = "C:\\Users\\USUARIO\\modelo_entrenado" // Cambiar según el directorio de salida deseado

    //val trainCommand = s"python $trainScriptPath --input $outputPath --model_output $modeloOutputPath"
    val trainCommand = s"python $trainScriptPath --input C:\\Users\\USUARIO\\datos_llm_corpus.txt\\*.txt --model_output $modeloOutputPath"

    println(s"Ejecutando el entrenamiento del modelo con: $trainCommand")
    val trainExitCode = trainCommand.!

    if (trainExitCode == 0) {
      println("Entrenamiento completado exitosamente.")
    } else {
      println("Error durante el entrenamiento.")
      sys.exit(1) // Detener ejecución en caso de error
    }

    // =======================
    // 5. Consultar el modelo
    // =======================
    val inferenceScriptPath = "inferencia.py" // Ruta al script de inferencia en Python
    val consultaPrompt = "El paciente fue diagnosticado con diabetes mellitus tipo 2."
    val consultaCommand = s"python $inferenceScriptPath --prompt '$consultaPrompt' --model_path $modeloOutputPath"

    println(s"Ejecutando la consulta al modelo con: $consultaCommand")
    val consultaExitCode = consultaCommand.!

    if (consultaExitCode == 0) {
      println("Consulta al modelo completada exitosamente.")
    } else {
      println("Error durante la consulta.")
    }

    // =======================
    // 6. Finalizar Spark
    // =======================
    spark.stop()
  }
}
