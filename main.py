from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import sys
from math import sqrt
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.functions import lag, avg, stddev, row_number
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, FloatType

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_INPUT_PATH', 'S3_OUTPUT_PATH'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


class StocksMetrics():
    def __init__(self, args):
        self.s3_input_path = args['S3_INPUT_PATH']
        self.s3_output_path = args['S3_OUTPUT_PATH']
        self.df = self._prepare_data()

    def write_data_to_s3(self, df, obj):
        df.write.mode("overwrite").parquet(f"{self.s3_output_path}{obj}")

    def _prepare_data(self):
        """
        0 - read the data
        1 - convert date columnfrom type string to date format
        2 - filled missing prices based on the closest available date
        3 - fillna as 0 for volume
        4 - find the previous price
        """
        df = spark.read.csv(self.s3_input_path, header=True, inferSchema=True, dateFormat="M/d/yyyy") \
            .select("date", "close", "volume", "ticker") \
            .withColumnRenamed('close', 'closing_price')
        df = df.withColumn("date", F.to_date(F.col("date"), "M/d/yyyy")) \
            .withColumn("date", F.date_format(F.col("date"), "yyyy-MM-dd"))
        windowSpec = Window.partitionBy("ticker").orderBy("date")
        df = df.withColumn("filled_closing_price", F.last("closing_price", True).over(windowSpec))
        df = df.fillna({"volume": 0})
        df = df.withColumn("prev_closing_price", F.lag("filled_closing_price", 1).over(windowSpec))
        # is being used multiple times, so made accessible to all, to reduce time compute
        df = df.withColumn("daily_return", F.when(F.isnull(F.col("prev_closing_price")), 0)
                           .otherwise((F.col("filled_closing_price") - F.col("prev_closing_price")) / \
                                      F.col("prev_closing_price") \
                                      * 100))
        return df

    def q1(self):
        """
        write data for the avg daily return
            * return can be trivially computed as the % difference of two prices
            * since there's no previous day's price to compare against, you can explicitly set the return for the first day to zero
        """
        print('------------------------- Create data for avg_daily_return table ---------------------------')
        df = self.df.select('date', 'daily_return')
        df_daily_return = df.groupBy("date").agg(F.avg("daily_return").alias("average_return"))
        df_daily_return = df_daily_return.withColumn("date", F.to_date("date", "yyyy-MM-dd"))
        self.write_data_to_s3(df_daily_return, "q1")

    def q2(self):
        """
        write data for the stock with the highest average traded value
        """
        print('------------------------- Create data for stock\'s highest avg_traded_value ---------------------------')
        df = self.df.select('filled_closing_price', 'volume', 'ticker')
        df = df.withColumn("traded_value", F.col("filled_closing_price") * F.col("volume"))
        avg_traded_value = df.groupBy("ticker").agg(F.avg("traded_value").alias("frequency"))
        most_traded_stock = avg_traded_value.orderBy(F.desc("frequency")).first()  # return a Row type

        schema = StructType([
            StructField("ticker", StringType(), True),
            StructField("frequency", DoubleType(), True)
        ])

        most_traded_stock_df = spark.createDataFrame([most_traded_stock], schema=schema)
        self.write_data_to_s3(most_traded_stock_df, "q2")

    def q3(self, trading_days_per_year=252):
        """
        write data for the most volatile as measured by the annualized standard deviation of daily returns
         * The annualization factor (sqrt(252)) assumes 252 trading days in a year,
           which is a common convention in financial markets.
        """
        print('------------------------- Create data for volatile std of daily returns ---------------------------')
        df = self.df.select('ticker', 'daily_return')
        std_dev_daily_return = df.groupBy("ticker").agg(F.stddev("daily_return").alias("stddev_daily_return"))
        std_dev_annualized = std_dev_daily_return.withColumn("standard deviation", (
                    F.col("stddev_daily_return") * F.sqrt(F.lit(trading_days_per_year)))) \
            .select('ticker', 'standard deviation')

        most_volatile_stock = std_dev_annualized.orderBy(F.desc("standard deviation")).first()  # return a Row type
        schema = StructType([
            StructField("ticker", StringType(), True),
            StructField("standard deviation", FloatType(), True)
        ])

        most_volatile_stock_df = spark.createDataFrame([most_volatile_stock], schema=schema)
        self.write_data_to_s3(most_volatile_stock_df, "q3")

    def q4(self):
        """
        write data for the top three 30-day return dates
        """
        print('------------------------- Create data for top three 30-day return dates ---------------------------')
        df = self.df.select('date', 'daily_return', 'ticker')
        df = df.withColumn("date", F.to_date("date", "yyyy-MM-dd"))
        windowSpec30Days = Window.partitionBy("ticker").orderBy("date").rowsBetween(-30, -1)
        df = df.withColumn("30_day_return", F.avg("daily_return").over(windowSpec30Days))
        windowSpecRank = Window.partitionBy("ticker").orderBy(F.desc("30_day_return"))
        df_ranked = df.withColumn("rank", F.rank().over(windowSpecRank))
        top_3_dates = df_ranked.filter(F.col("rank") <= 3) \
            .groupBy("ticker") \
            .agg(F.collect_list("date").alias("date combinations"))
        self.write_data_to_s3(top_3_dates, "q4")

    def run(self):
        try:
            self.q1()
            self.q2()
            self.q3()
            self.q4()
        except Exception as e:
            raise e


sm = StocksMetrics(args)
sm.run()

job.commit()


