import airflow
from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
from airflow.utils.dates import days_ago

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

with DAG(
    dag_id = "etl_demo",
    default_args=default_args,
    # schedule_interval='0 0 * * *',
    schedule_interval='@once',
    dagrun_timeout=timedelta(minutes=60),
    description='etl credit history demo',
    start_date = airflow.utils.dates.days_ago(1),
    catchup=False
) as dag:
    import yaml
    import os
    fs_yaml = os.getenv("FEATURE_STORE_YAML")
    templates = yaml.safe_load(fs_yaml)
    project = templates.get("project").lower()
    

    def load_data(**kwargs): #parquet_name: str, table: str):
        parquet_name = kwargs['parquet_name']
        table = kwargs['table']

        import boto3
        import os

        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("FEAST_S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        def get_sqlalchemy_engine():
            import sqlalchemy
            import yaml
            fs_yaml = os.getenv("FEATURE_STORE_YAML")
            templates = yaml.safe_load(fs_yaml)
            of_store = templates.get("offline_store")
            host = of_store.get("host")
            port = of_store.get("port")
            user = of_store.get("user")
            password = of_store.get("password")
            db_name = of_store.get("database")
            url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
            return sqlalchemy.create_engine(url, client_encoding='utf8', connect_args={'options': '-c search_path={}'.format("public")})

        import pandas as pd
        con = None
        raw_con = None
        try:
            con = get_sqlalchemy_engine()
            s3_client.download_file(os.getenv("S3_BUCKET"), parquet_name, parquet_name)
            df = pd.read_parquet(parquet_name)

            con.execute("DROP TABLE IF EXISTS " + table)
            create_table_sql = pd.io.sql.get_schema(df, table, con=con)
            print(create_table_sql)
            con.execute(create_table_sql)
            from io import StringIO
            buffer = StringIO()
            df.to_csv(buffer, header=False, index=False, na_rep="\\N")
            buffer.seek(0)
            raw_con = con.raw_connection()
            with raw_con.cursor() as cursor:
                cursor.copy_from(buffer, table, sep=",")
            raw_con.commit()
        finally:
            if raw_con is not None:
                raw_con.close()

    
    dummy_task = DummyOperator(task_id='dummy_task', retries=3)

    load_credit_history_task = PythonOperator(
        task_id='load_credit_history', 
        python_callable=load_data,
        op_kwargs={'parquet_name': 'credit_history.parquet', 'table': f'{project}_credit_history'})

    load_zipcode_task = PythonOperator(
        task_id='load_zipcode', 
        python_callable=load_data,
        op_kwargs={'parquet_name': 'zipcode_table.parquet', 'table': f'{project}_zipcode'})

    load_incomes_task = PythonOperator(
        task_id='load_income', 
        python_callable=load_data,
        op_kwargs={'parquet_name': 'income_history.parquet', 'table': f'{project}_income_history'})




dummy_task >> [load_credit_history_task, load_zipcode_task, load_incomes_task]
