import airflow
from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago

from kubernetes.client import models as k8s


default_args = {
    'owner': 'nsuraeva',    
    #'start_date': airflow.utils.dates.days_ago(2),
    # 'end_date': datetime(),
    # 'depends_on_past': False,
    #'email': ['airflow@example.com'],
    #'email_on_failure': False,
    #'email_on_retry': False,
    # If a task fails, retry it once after waiting
    # at least 5 minutes
    #'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

IMAGE='registry.neomsa.ru/docker-mlops/mlops/airflow:2.2.5-demo-v5'

with DAG(
    dag_id = "batch_predict",
    default_args=default_args,
    # schedule_interval='0 0 * * *',
    schedule_interval='@once',
    dagrun_timeout=timedelta(minutes=60),
    description='batch model predict',
    start_date = airflow.utils.dates.days_ago(1),
    catchup=False
) as dag:
    
    import os
    import yaml

    fs_yaml = os.getenv("FEATURE_STORE_YAML")
    templates = yaml.safe_load(fs_yaml)
    project = templates.get("project").lower()
    def init_feature_store_path():
        from feast import FeatureStore
        with open("feature_store.yaml", "w") as yaml_file:
            yaml_file.write(os.getenv("FEATURE_STORE_YAML"))
        return FeatureStore(repo_path=".")

    def get_boto3_client():
        import boto3
        return boto3.client("s3",
                    endpoint_url=os.environ.get("FEAST_S3_ENDPOINT_URL"),
                    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
    
    def load_batch():
        s3_client = get_boto3_client()
        bucket_name = os.environ.get("S3_BUCKET")
        s3_client.download_file(bucket_name, 'data_for_batch_predict.parquet', 'data_for_batch_predict.parquet')

    def upload_batch(file_name):
        s3_client = get_boto3_client()
        bucket_name = os.environ.get("S3_BUCKET")
        s3_client.upload_file(file_name, bucket_name, file_name)

    def prepare_dataset(**kwargs):
        store = init_feature_store_path()
        load_batch()

        import pandas as pd
        batch = pd.read_parquet('data_for_batch_predict.parquet')

        date= datetime.utcnow() - datetime(1970, 1, 1)
        seconds =(date.total_seconds())
        milliseconds = round(seconds*1000)
        dataset_name = f"{project}_demo_predict_dataset_{milliseconds}"

        from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import SavedDatasetPostgreSQLStorage
        
        loan_features = store.get_feature_service(f'{project}_loan_features')
        enriched_job = store.get_historical_features(
            entity_df=batch,
            features=loan_features,
        )
        dataset = store.create_saved_dataset(
            from_=enriched_job,
            name=dataset_name,
            storage=SavedDatasetPostgreSQLStorage(table_ref=dataset_name),
        )
        kwargs['ti'].xcom_push(key='dataset name', value=dataset_name)
    
    def predict(**kwargs):
        store = init_feature_store_path()
        
        import pandas as pd

        dataset_name = kwargs['ti'].xcom_pull(key='dataset name', task_ids='create_dataset')

        enriched_df = store.get_saved_dataset(dataset_name).to_df()
        
        enriched_df = enriched_df[enriched_df.columns.drop("event_timestamp")]
    
        import mlflow.pyfunc

        model_name = "batch_model"
        model_version = 1

        loaded_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )

        # Predict on a Pandas DataFrame.
        result_array = loaded_model.predict(enriched_df.copy())
        result_df = pd.DataFrame(result_array, columns = ["zipcode", "dob_ssn", "prediction"])
        input_with_prediction_df = pd.merge(result_df, enriched_df, on=["zipcode", "dob_ssn"])
        print(input_with_prediction_df)
        input_with_prediction_df.to_parquet(f"result_{dataset_name}.parquet")
        upload_batch(f"result_{dataset_name}.parquet")
    
    create_dataset = PythonOperator(
        task_id="create_dataset",
        python_callable=prepare_dataset,
        provide_context=True,
        executor_config = {
        "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
    },
    )
    
    batch_predict = PythonOperator(
        task_id="batch_predict",
        python_callable=predict,
        provide_context=True,
        executor_config = {
        "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
    },
    )


create_dataset >> batch_predict
