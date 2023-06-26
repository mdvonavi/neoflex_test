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
    dag_id = "materialization",
    default_args=default_args,
    # schedule_interval='0 0 * * *',
    schedule_interval='@once',
    dagrun_timeout=timedelta(minutes=60),
    description='train model credit history demo',
    start_date = airflow.utils.dates.days_ago(1),
    catchup=False
) as dag:
    import yaml
    import os
    fs_yaml = os.getenv("FEATURE_STORE_YAML")
    templates = yaml.safe_load(fs_yaml)
    project = templates.get("project").lower()
    
    def init_feature_store_path():
        import os
        from feast import FeatureStore
        with open("feature_store.yaml", "w") as yaml_file:
            yaml_file.write(os.getenv("FEATURE_STORE_YAML"))
        return FeatureStore(repo_path=".")

    def materialize_ch():
        print("materialize_ch")
        store = init_feature_store_path()
        store.materialize(
            feature_views=[f"{project}_credit_history"],
            start_date=datetime.utcnow() - timedelta(days=3650), end_date=datetime.utcnow() - timedelta(minutes=10)
        )
        
    def materialize_zip():
        print("materialize_zip")
        store = init_feature_store_path()
        store.materialize(
            feature_views=[f"{project}_zipcode_features"],
            start_date=datetime.utcnow() - timedelta(days=3650), end_date=datetime.utcnow() - timedelta(minutes=10)
        )
    
    def materialize_cf():
        print("materialize_cf")
        store = init_feature_store_path()
        store.materialize(
            feature_views=[f"{project}_client_features"],
            start_date=datetime.utcnow() - timedelta(days=3650), end_date=datetime.utcnow() - timedelta(minutes=10)
        )
    
    dummy_task = DummyOperator(task_id='dummy_task', retries=3)
    
    materialize_credit_history = PythonOperator(
        task_id="material_cr_hist",
        python_callable=materialize_ch,
        provide_context=True,
        executor_config = {
        "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
    },
    )
    
    materialize_zipcode = PythonOperator(
        task_id="material_zip",
        python_callable=materialize_zip,
        provide_context=True,
        executor_config = {
        "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
    },
    )
    
    materialize_client_features = PythonOperator(
        task_id="material_cl_feat",
        python_callable=materialize_cf,
        provide_context=True,
        executor_config = {
        "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
    },
    )


dummy_task >> [materialize_credit_history, materialize_zipcode, materialize_client_features]

