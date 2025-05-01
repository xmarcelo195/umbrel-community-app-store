from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# Define default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='bash_operator_example',
    default_args=default_args,
    schedule_interval=None,  # Run manually or according to your schedule
    catchup=False,
) as dag:

    # Use BashOperator and pass the command
    run_bash_task = BashOperator(
        task_id='run_python_script',
        bash_command="python /home/umbrel/umbrel/app-data/xmrcl-airflow/scripts/modelo_recessao.py",
        env = {
            "FRED_API_KEY": "{{ var.value.fred_api_key }}"
        }
    )

    run_bash_task
