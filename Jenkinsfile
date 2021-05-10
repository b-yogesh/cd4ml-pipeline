pipeline {
  agent none
    options {
       // add timestamps to output
       timestamps()
    }
     triggers {
    pollSCM('* * * * *')
    }
    environment { 
        MLFLOW_TRACKING_URL = 'http://mlflow:5000'
    }
    
    stages {
        stage('Install dependencies') {
        agent {
                docker {
                    image 'python:3.9.5-windowsservercore-ltsc2016'
                }
            }
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
            agent any

            steps {
                    sh 'python3 test.py'
            }
       }
    }

  }
 