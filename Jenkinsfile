pipeline {
    agent any
    triggers {
        // Poll SCM every minute for new changes
        pollSCM('* * * * *')
    }
    options {
       // add timestamps to output
       timestamps()
    }
    environment { 
        MLFLOW_TRACKING_URL = 'http://mlflow:5000'
    }
    stages {
        stage('Install dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        // stage('Run tests') {
        //     steps {
        //         sh './run_tests.sh'
        //     }
        // }
        stage('Run ML pipeline') {
            steps {
                sh 'python3 test.py'
            }
       }
    }
}