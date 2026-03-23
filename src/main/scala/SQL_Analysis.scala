import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object SQL_Analysis {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("StackOverflow SQL Analysis")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // Load the preprocessed data
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("stackoverflow_final_transformed.csv")
    
    println("=== Dataset Loaded for SQL Analysis ===")
    println(s"Rows: ${df.count()} | Cols: ${df.columns.length}")
    println()

    // Register as temporary view
    df.createOrReplaceTempView("developer_survey")
    
    // ============================================================
    // QUERY 1: Average Salary by Developer Type
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 1: Average Salary by Developer Type")
    println("=" * 60)
    
    val query1 = """
      SELECT 
        DevType_segment,
        COUNT(*) as developer_count,
        ROUND(AVG(Salary_num), 2) as avg_salary_usd
      FROM developer_survey
      GROUP BY DevType_segment
      ORDER BY avg_salary_usd DESC
    """
    
    spark.sql(query1).show(10, truncate = false)
    
    // ============================================================
    // QUERY 2: Educational Pathways to AI/ML Roles
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 2: Educational Pathways to AI/ML Roles")
    println("=" * 60)
    
    val query2 = """
      SELECT 
        UndergradMajor,
        COUNT(*) as total_developers,
        SUM(CASE WHEN DevType_segment IN ('Data', 'AI/ML') THEN 1 ELSE 0 END) as ai_ml_count,
        ROUND(SUM(CASE WHEN DevType_segment IN ('Data', 'AI/ML') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_in_ai_ml
      FROM developer_survey
      WHERE UndergradMajor IS NOT NULL
      GROUP BY UndergradMajor
      HAVING total_developers >= 30
      ORDER BY pct_in_ai_ml DESC
      LIMIT 10
    """
    
    spark.sql(query2).show(truncate = false)
    
    // ============================================================
    // QUERY 3: Self-Learning vs Formal Education Impact
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 3: Self-Learning vs Formal Education Impact")
    println("=" * 60)
    
    val query3 = """
      SELECT 
        CASE 
          WHEN EducationTypes LIKE '%Taught yourself%' 
               AND EducationTypes NOT LIKE '%Bachelor%' 
               AND EducationTypes NOT LIKE '%Master%'
            THEN 'Self-Learning Only'
          WHEN (EducationTypes LIKE '%Bachelor%' OR EducationTypes LIKE '%Master%') 
               AND EducationTypes LIKE '%Taught yourself%'
            THEN 'Formal + Self-Learning'
          WHEN (EducationTypes LIKE '%Bachelor%' OR EducationTypes LIKE '%Master%') 
            THEN 'Formal Only'
          ELSE 'Mixed/Other'
        END as learning_path,
        COUNT(*) as developer_count,
        ROUND(AVG(Salary_num), 0) as avg_salary
      FROM developer_survey
      WHERE Salary_num > 0
      GROUP BY learning_path
      ORDER BY avg_salary DESC
    """
    
    spark.sql(query3).show(truncate = false)
    
    // ============================================================
    // QUERY 4: Top Programming Languages by Developer Type (CORRECTED)
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 4: Top Programming Languages per Developer Type")
    println("=" * 60)
    
    val query4 = """
      WITH language_counts AS (
        SELECT 
          DevType_segment,
          language,
          COUNT(*) as lang_count
        FROM (
          SELECT 
            DevType_segment,
            EXPLODE(SPLIT(LanguageWorkedWith, ';')) as language
          FROM developer_survey
        ) t
        GROUP BY DevType_segment, language
      ),
      ranked AS (
        SELECT 
          DevType_segment,
          language,
          lang_count,
          ROW_NUMBER() OVER (PARTITION BY DevType_segment ORDER BY lang_count DESC) as rank
        FROM language_counts
      )
      SELECT DevType_segment, language, lang_count
      FROM ranked
      WHERE rank <= 3
      ORDER BY DevType_segment, rank
    """
    
    spark.sql(query4).show(20, truncate = false)
    
    // ============================================================
    // QUERY 5: Skills Gap Analysis (Simplified)
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 5: Skills Gap Analysis for AI/ML")
    println("=" * 60)
    
    val query5 = """
      SELECT 
        language,
        COUNT(*) as demand_in_ai_ml,
        SUM(CASE WHEN DevType_segment NOT IN ('Data', 'AI/ML', 'Student/Academic') THEN 1 ELSE 0 END) as supply_in_non_ai
      FROM (
        SELECT 
          DevType_segment,
          EXPLODE(SPLIT(LanguageWorkedWith, ';')) as language
        FROM developer_survey
        WHERE Salary_num > 0
      ) t
      WHERE DevType_segment IN ('Data', 'AI/ML')
      GROUP BY language
      ORDER BY demand_in_ai_ml DESC
      LIMIT 10
    """
    
    spark.sql(query5).show(truncate = false)

    // ============================================================
    // QUERY 6: Salary Distribution by Age and Experience
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 6: Salary Distribution by Age and Experience Groups")
    println("=" * 60)
    
    val query6 = """
      SELECT 
        CASE 
          WHEN Age LIKE '18-24%' THEN '18-24 (Early Career)'
          WHEN Age LIKE '25-34%' THEN '25-34 (Mid Career)'
          WHEN Age LIKE '35-44%' THEN '35-44 (Established)'
          WHEN Age LIKE '45-54%' THEN '45-54 (Senior)'
          ELSE '55+ (Veteran)'
        END as age_group,
        CASE 
          WHEN YearsCoding LIKE '0-2%' THEN '0-2 years (Junior)'
          WHEN YearsCoding LIKE '3-5%' THEN '3-5 years (Intermediate)'
          WHEN YearsCoding LIKE '6-8%' THEN '6-8 years (Mid-Level)'
          WHEN YearsCoding LIKE '9-11%' THEN '9-11 years (Senior)'
          ELSE '12+ years (Expert)'
        END as experience_group,
        COUNT(*) as respondent_count,
        ROUND(AVG(Salary_num), 0) as avg_salary_usd,
        ROUND(PERCENTILE(Salary_num, 0.5), 0) as median_salary_usd
      FROM developer_survey
      WHERE Salary_num > 0 AND Salary_num < 500000
      GROUP BY age_group, experience_group
      HAVING respondent_count >= 10
      ORDER BY age_group, avg_salary_usd DESC
      LIMIT 20
    """
    
    spark.sql(query6).show(truncate = false)
    
    // ============================================================
    // QUERY 7: Global Talent Distribution
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 7: Global Talent Distribution - Top 15 Countries")
    println("=" * 60)
    
    val query7 = """
      SELECT 
        Country,
        COUNT(*) as developer_count,
        ROUND(AVG(Salary_num), 0) as avg_salary_usd,
        SUM(CASE WHEN DevType_segment IN ('Data', 'AI/ML') THEN 1 ELSE 0 END) as ai_ml_count,
        ROUND(SUM(CASE WHEN DevType_segment IN ('Data', 'AI/ML') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_ai_ml
      FROM developer_survey
      WHERE Country IS NOT NULL AND Salary_num > 0
      GROUP BY Country
      HAVING developer_count >= 50
      ORDER BY developer_count DESC
      LIMIT 15
    """
    
    spark.sql(query7).show(truncate = false)
    
    // ============================================================
    // SUMMARY
    // ============================================================
    println("\n" + "=" * 60)
    println("SUMMARY OF INSIGHTS")
    println("=" * 60)
    println("KEY FINDINGS:")
    println("-" * 60)
    println("1. SALARY BY ROLE: 'Other' roles have highest avg salary ($85,622), followed by")
    println("   Data roles ($83,978). Student/Academic roles lowest ($77,283).")
    println()
    println("2. EDUCATIONAL PATHWAYS: Mathematics/Statistics graduates have highest")
    println("   transition rate to AI/ML roles (16%), followed by Social Sciences (12.3%).")
    println()
    println("3. LEARNING IMPACT: Mixed/Other learning paths yield highest salaries ($86,406),")
    println("   while Self-Learning Only averages $77,430 (10.4% lower).")
    println()
    println("4. TOP LANGUAGES: JavaScript, Python, and SQL dominate across most developer roles.")
    println()
    println("5. CAREER INSIGHTS: Educational background significantly influences career trajectory,")
    println("   with Mathematics/Statistics graduates showing strongest AI/ML alignment.")
    println("=" * 60)
    
    spark.stop()
  }
}