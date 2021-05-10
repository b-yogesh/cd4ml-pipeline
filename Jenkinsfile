pipeline {
  agent any
  stages {
    stage('Install dependencies') {
      steps {
        sh 'pip3 install -r requirements.txt'
      }
    }

    stage('Run ML pipeline') {
      steps {
        sh 'python3 test.py'
      }
    }

  }
  environment {
    MLFLOW_TRACKING_URL = 'http://mlflow:5000'
  }
  options {
    timestamps()
  }
  triggers {
    pollSCM('* * * * *')
  }
}