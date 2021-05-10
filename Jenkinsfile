pipeline {
  agent any
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
    agent {
                docker {
                    image 'python:3-alpine'
                }
            }
    stages {
        stage('Install dependencies') {
            steps {
                withEnv(["HOME=${env.WORKSPACE}"]){
                    sh 'pip install -r requirements.txt'
                }
            }
        }
        // stage('Run tests') {
        //     steps {
        //         sh './run_tests.sh'
        //     }
        // }
        stage('Run ML pipeline') {
            steps {
                withEnv(["HOME=${env.WORKSPACE}"]){
                    sh 'python3 test.py'
                }
            }
       }
    }

  }
 