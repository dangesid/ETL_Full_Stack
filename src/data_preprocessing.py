from pyspark.sql import SparkSession
from pyspark.sql.functions import col 
import os 
import shutil

class DataPreprocessing:

    def __init__(self, file_path):
        """ Initialize the Spark Session and loading the dataset. """
        self.spark = SparkSession.builder.appName("Fraud Detection Data Preprocessing").getOrCreate()
        self.file_path = file_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        """ Load the CSV data into pyspark DataFrame """
        try:
            self.df = self.spark.read.csv(self.file_path, header=True, inferSchema=True)
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    def clean_data(self):
        """ Clean the data by dropping missing values and unwanted columns"""
        try:
            self.df = self.df.dropna()
            return self.df
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return None

    def convert_types(self):
        """ Converting the columns to correct data types."""
        try:
            self.df = self.df.withColumn("amount",col("amount").cast("double"))
            return self.df
        except Exception as e:
            print("Error converting types")
            return None
        
    def save_data(self):
        """Save preprocessed data as a single CSV file."""
        temp_path = self.output_path + "_temp"
        self.df.coalesce(1).write.csv(temp_path, header=True, mode="overwrite")

        # Find the generated CSV file and rename it
        for file in os.listdir(temp_path):
            if file.startswith("part-") and file.endswith(".csv"):
                shutil.move(os.path.join(temp_path, file), self.output_path)
                break
        print(f"Data saved successfully at {self.output_path}")        
        shutil.rmtree(temp_path)
    
    def preprocess(self):
        "Run full preprocessing pipeline"
        self.load_data()
        if self.df is None:
            return None 
        self.clean_data()
        self.convert_types()
        self.save_data()
        return self.df  
    
if __name__ == "__main__":
    file_path = os.path.abspath("data/PS_20174392719_1491204439457_log.csv")
    output_path=os.path.abspath("data/preprocessed_data.csv")

    processor = DataPreprocessing(file_path)
    df = processor.preprocess()
    if df is not None:
        df.show(5)
    else:
        print("\n Error in Preprocessing")

