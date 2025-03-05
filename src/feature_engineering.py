from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, expr

class FeatureEngineering:
    def __init__(self, df):
        """ Initialize with the preprocessed DataFrame. """
        self.df = df

    def encode_transaction_type(self):
        """ Convert transaction type into numerical labels"""
        self.df = self.df.withColumn("type_encoded",
                                     when(col("type") == "PAYMENT", 1).
                                     when(col("type") == "TRANSFER", 2).
                                     when(col("type") == "CASH_OUT", 3).
                                     when(col("type") == "DEBIT", 4).
                                     when(col("type") == "CASH_IN",5)
                                     .otherwise(0),)
        return self.df 

    def create_balance_features(self):
        """ Feature Engineering implementing new features """
        self.df = self.df.withColumn(
            "balance_change_orig", col("oldbalanceOrg") - col("newbalanceOrig"),
        )       
        self.df = self.df.withColumn(
            "balance_change_dest", col("oldbalanceDest") - col("newbalanceDest"),
        )
        return self.df
    
    def normalize_amount(self):
        """ Normalize the amount column by using min-max scaling"""
        max_amount = self.df.agg({"amount": "max"}).collect()[0][0]
        min_amount = self.df.agg({"amount": "min"}).collect()[0][0]

        self.df = self.df.withColumn("normalized_amount", (col("amount")- min_amount) / (max_amount - min_amount),
                                     )
        return self.df
    
    def engineer_features(self):
        """ Run full feature engineering pipeline """
        self.encode_transaction_type()
        self.create_balance_features()
        self.normalize_amount()
        return self.df
    
if __name__ == "__main__":
    spark = SparkSession.builder.appName("Feature Engineering").getOrCreate()

    # Load Preprocessed data 
    df = spark.read.csv("data/PS_20174392719_1491204439457_log.csv",header=True,inferSchema=True)

    # Preprocess features

    fe = FeatureEngineering(df)
    df_transformed = fe.engineer_features()

    df_transformed.show(5)