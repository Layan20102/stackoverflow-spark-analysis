import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.avg

object RDD_Operations {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("StackOverflow RDD Analysis")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // ============================================================
    // LOAD PREPROCESSED DATA
    // ============================================================
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("stackoverflow_final_transformed.csv")

    println("=== Dataset Loaded ===")
    println(s"Rows: ${df.count()} | Cols: ${df.columns.length}")

    val rdd = df.rdd
    println(s"RDD created successfully.\n")

    // ============================================================
    // TRANSFORMATIONS
    // ============================================================

    // --- Transformation 1: map ---
    println("--- Transformation 1: map ---")
    val countryToSalary = rdd.map(row => (
      row.getAs[String]("Country"),
      row.getAs[Double]("Salary_num")
    ))
    println(f"${"Country"}%-25s ${"Salary (USD)"}%15s")
    println("-" * 42)
    countryToSalary.take(5).foreach(x => println(f"${x._1}%-25s $$${x._2}%14.2f"))

    // --- Transformation 2: filter ---
    println("\n--- Transformation 2: filter ---")
    val avgSalary = df.agg(avg("Salary_num")).first().getDouble(0)
    val highEarners = rdd.filter(row => row.getAs[Double]("Salary_num") > avgSalary)
    println(f"Average Salary : $$$avgSalary%.2f")
    println(f"High Earners   : ${highEarners.count()}%d")
    println(f"Percentage     : ${highEarners.count() * 100.0 / rdd.count()}%.1f%%")

    // --- Transformation 3: flatMap ---
    println("\n--- Transformation 3: flatMap ---")
    val languages = rdd.flatMap(row =>
      row.getAs[String]("LanguageWorkedWith").split(";").map(_.trim)
    )
    println("Sample languages:")
    languages.take(10).foreach(lang => println(s"  $lang"))

    // --- Transformation 4: reduceByKey ---
    println("\n--- Transformation 4: reduceByKey (Language Count) ---")
    val topLanguages = languages
      .map(lang => (lang, 1))
      .reduceByKey(_ + _)
      .sortBy(_._2, ascending = false)
    println(f"${"Language"}%-20s ${"Count"}%10s")
    println("-" * 32)
    topLanguages.take(10).foreach(x => println(f"${x._1}%-20s ${x._2}%10d"))

    // --- Transformation 5: reduceByKey (Avg Salary) ---
    println("\n--- Transformation 5: reduceByKey (Average Salary) ---")
    val salaryByDevType = rdd
      .map(row => (
        row.getAs[String]("DevType_segment"),
        (row.getAs[Double]("Salary_num"), 1)
      ))
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .map(x => (x._1, x._2._1 / x._2._2))
      .sortBy(_._2, ascending = false)
    println(f"${"DevType"}%-22s ${"Avg Salary (USD)"}%18s")
    println("-" * 42)
    salaryByDevType.collect().foreach(x => println(f"${x._1}%-22s $$${x._2}%17.2f"))

    // ============================================================
    // ACTIONS
    // ============================================================
    println("\n" + "=" * 60)
    println("RDD ACTIONS")
    println("=" * 60)

    // --- Action 1: count ---
    println("\n--- Action 1: count ---")
    println(f"Total records: ${rdd.count()}%,d")

    // --- Action 2: first ---
    println("\n--- Action 2: first ---")
    val firstRow = rdd.first()
    println(s"Country    : ${firstRow.getAs[String]("Country")}")
    println(s"DevType    : ${firstRow.getAs[String]("DevType_segment")}")
    println(s"Employment : ${firstRow.getAs[String]("Employment")}")
    println(f"Salary     : $$${firstRow.getAs[Double]("Salary_num")}%.2f")

    // --- Action 3: reduce ---
    println("\n--- Action 3: reduce ---")
    val totalSalary = rdd.map(row => row.getAs[Double]("Salary_num")).reduce(_ + _)
    println(f"Total salary: $$$totalSalary%,.2f")
    println(f"(~$$${ totalSalary / 1e9 }%.2f billion USD)")

    // --- Action 4: take ---
    println("\n--- Action 4: take ---")
    println("First 10 Data developers:")
    println(f"${"Country"}%-18s ${"Languages"}%s")
    println("-" * 55)
    rdd.filter(row => row.getAs[String]("DevType_segment") == "Data")
      .take(10)
      .foreach(row => println(f"${row.getAs[String]("Country")}%-18s ${row.getAs[String]("LanguageWorkedWith").take(35)}..."))

    // --- Action 5: collect ---
    println("\n--- Action 5: collect ---")
    println("Complete salary ranking:")
    println(f"${"Rank"}%6s ${"DevType"}%-22s ${"Avg Salary"}%15s")
    println("-" * 45)
    salaryByDevType.collect().zipWithIndex.foreach { case ((devtype, sal), i) =>
      println(f"  #${i+1}   ${devtype}%-22s $$$sal%14.2f")
    }

    spark.stop()
  }
}
