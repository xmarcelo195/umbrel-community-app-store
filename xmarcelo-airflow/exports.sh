#!/bin/bash
export APP_AIRFLOW_FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")