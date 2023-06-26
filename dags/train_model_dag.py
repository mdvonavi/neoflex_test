import airflow
from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
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

IMAGE='registry.neomsa.ru/docker-mlops/mlops/airflow:2.2.5-demo-v4'

with DAG(
    dag_id = "train_model_demo",
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
        with open("feature_store.yaml", "w") as yaml_file:
            yaml_file.write(os.getenv("FEATURE_STORE_YAML"))

    def train_model():
        from sklearn import tree
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, make_scorer
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import pandas as pd
        import dill
        import sklearn

        from feast import FeatureStore
        
        init_feature_store_path()
        store = FeatureStore(repo_path=".")
        training_df = store.get_saved_dataset(f'{project}_demo_training_dataset').to_df()
        
        encoder = OrdinalEncoder()
        categorical_features = [
            "person_home_ownership",
            "loan_intent",
            "city",
            "state",
            "location_type",
        ]
        encoder.fit(training_df[categorical_features])
        transform_training_df=training_df.copy()
        transform_training_df[categorical_features] = encoder.transform(
            training_df[categorical_features]
        )

        target = "loan_status"
        train_X = transform_training_df[
            transform_training_df.columns.drop(target)
            .drop("event_timestamp")
            .drop("created_timestamp")
            .drop("loan_id")
            .drop("zipcode")
            .drop("dob_ssn")
        ]
        train_X = train_X.reindex(sorted(train_X.columns), axis=1)
        train_Y = transform_training_df.loc[:, target]

        x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.10)
        
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(train_X[sorted(train_X)], train_Y)
        max_depth = 30
        classifier = tree.DecisionTreeClassifier(max_depth = max_depth)
        classifier.fit(x_train[sorted(x_train)], y_train)
        predictions = classifier.predict(x_test)
        
        accuracy = accuracy_score(y_true=y_test, y_pred = predictions)
        
        def eval_metrics(actual, pred):
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            return rmse, mae, r2
        
        (rmse, mae, r2) = eval_metrics(y_test, predictions)
        print(rmse)
        print(mae)
        print(r2)
        
        import joblib

        model_name = "model.bin"
        encoder_name = "encoder.bin"

        joblib.dump(classifier, model_name)
        joblib.dump(encoder, encoder_name)
        
        artifacts = {
            "model": model_name,
            "encoder": encoder_name,
        }
        
        import mlflow
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, ColSpec
        import logging

        class ModelWrapper(mlflow.pyfunc.PythonModel):

            def load_context(self, context):
                import joblib
                self.encoder = joblib.load(context.artifacts["encoder"])            
                self.model = joblib.load(context.artifacts["model"]) 
                self.categorical_features = [
                    "person_home_ownership",
                    "loan_intent",
                    "city",
                    "state",
                    "location_type",
                ]
            
            def predict(self, context, inp):
        
                import pandas as pd

                logger = logging.getLogger(__name__)
                
                if isinstance(inp, dict):
                    features_df = pd.DataFrame.from_dict(inp)
                elif isinstance(inp, pd.DataFrame):
                    features_df = inp

                logger.debug("Enriching features DataFrame: %s", features_df)

                # Apply ordinal encoding to categorical features
                features_df[self.categorical_features] = self.encoder.transform(
                    features_df[self.categorical_features]
                )

                logger.debug("features DataFrame after encoging: %s", features_df)

                # Sort columns
                features_df = features_df.reindex(sorted(features_df.columns), axis=1)

                # Drop unnecessary columns
                df_for_predict = features_df.copy()
                df_for_predict = df_for_predict[df_for_predict.columns.drop("zipcode").drop("dob_ssn")]

                logger.debug("features DataFrame before predict: %s", df_for_predict)

                # Make prediction
                df_for_predict["prediction"] = self.model.predict(df_for_predict)
                
                result = pd.merge(features_df, df_for_predict, left_index=True, right_index=True)[["zipcode", "dob_ssn", "prediction"]]
                
                logger.debug("result of predict: %s", result)

                # return result of credit scoring
                return result.to_numpy()

        conda_env={
            'channels': ['defaults'],
            'dependencies': [
              'python=3.8.10',
              'pip',
              {
                'pip': [
                  'mlflow=={}'.format(mlflow.__version__),
                  'numpy==1.23.4',
                  'scikit-learn=={}'.format(sklearn.__version__),
                  'joblib=={}'.format(joblib.__version__),
                  'dill=={}'.format(dill.__version__),
                ],
              },
            ],
            'name': 'demo_env'
        }

        input_schema = Schema([
            ColSpec("long", "zipcode"),
            ColSpec("string", "dob_ssn"),
            ColSpec("long", "population"),
            ColSpec("long", "total_wages"),
            ColSpec("string", "state"),
            ColSpec("long", "tax_returns_filed"),
            ColSpec("string", "city"),
            ColSpec("string", "location_type"),
            ColSpec("long", "bankruptcies"),
            ColSpec("long", "credit_card_due"),
            ColSpec("long", "missed_payments_6m"),
            ColSpec("long", "mortgage_due"),
            ColSpec("long", "vehicle_loan_due"),
            ColSpec("long", "hard_pulls"),
            ColSpec("long", "missed_payments_2y"),
            ColSpec("long", "student_loan_due"),
            ColSpec("long", "missed_payments_1y"),
            ColSpec("double", "incomeamount12m"),
            ColSpec("long", "person_age"),
            ColSpec("string", "person_home_ownership"),
            ColSpec("double", "person_emp_length"),
            ColSpec("long", "person_income"),
            ColSpec("string", "loan_intent"),
            ColSpec("long", "loan_amnt"),
            ColSpec("double", "loan_int_rate"),
        ])
        output_schema = Schema([ColSpec("long", "zipcode"),ColSpec("string", "dob_ssn"),ColSpec("integer", "prediction")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        mlflow.set_experiment("demo_prod")

        with mlflow.start_run(run_name='user guide') as current_run:

            mlflow.log_param("max_depth", max_depth)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)
            
            mlflow.set_tag("dataset_name", f"{project}_demo_training_dataset")

            mlflow.pyfunc.log_model(artifact_path="model", python_model=ModelWrapper(),
                                    artifacts=artifacts, conda_env=conda_env, signature=signature, registered_model_name='batch_model')

            run_id = current_run.info.run_id
            print(run_id)
    
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
        executor_config = {
        "pod_override": k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name="base", image=IMAGE)]))
    },
    )


train_model_task
