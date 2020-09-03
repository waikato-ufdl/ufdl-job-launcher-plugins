#!/bin/bash

function check_executable()
{
  echo "Checking $EXEC..."
  if [ ! -x "/usr/bin/$EXEC" ]
  then
    AVAILABLE=false
    if [ "$REQUIRED" = "true" ]
    then
      echo
      echo "$EXEC executable not present!"
      echo "Install on Debian systems with:"
      echo "sudo apt-get install $EXEC $ADDITIONAL"
      echo
      exit 1
    else
      echo "...NOT present"
    fi
  else
    echo "...is present"
    AVAILABLE=true
  fi
}

function check_repository()
{
  echo "Checking repo $REPO..."
  if [ ! -d "../$REPO" ]
  then
    echo	  
    echo "Directory ../$REPO does not exist!"
    echo "Check out repo as follows:"
    echo "cd .."
    echo "git clone https://github.com/waikato-ufdl/$REPO.git"
    echo	  
    exit 2
  else
    echo "...is present"
  fi
}

echo "Performing checks"

EXEC="virtualenv"
ADDITIONAL=""
REQUIRED=true
check_executable

EXEC="python3.7"
ADDITIONAL="python3.7-dev"
REQUIRED=false
check_executable
PYTHON37_AVAILABLE=$AVAILABLE

EXEC="python3.8"
ADDITIONAL="python3.8-dev"
REQUIRED=false
check_executable
PYTHON38_AVAILABLE=$AVAILABLE

if [ "$PYTHON37_AVAILABLE" = "false" ] && [ "$PYTHON38_AVAILABLE" = "false" ]
then
  echo
  echo "Neither Python 3.7 nor Python3.8 are available!"
  echo "Install on Debian systems with:"
  echo "  sudo apt-get install python3.7 python3.7-dev"
  echo "or"
  echo "  sudo apt-get install python3.8 python3.8-dev"
  echo
  exit 1
fi

if [ "$PYTHON37_AVAILABLE" = "true" ]
then
  PYTHON=python3.7
elif [ "$PYTHON38_AVAILABLE" = "true" ]
then
  PYTHON=python3.8
else
  echo "Don't know what Python executable to use!"
  exit 2
fi


REPO="ufdl-json-messages"
check_repository

REPO="ufdl-python-client"
check_repository

REPO="ufdl-job-launcher"
check_repository

echo
echo "Press any key to start setup of 'venv' for UFDL job launcher (using $PYTHON)..."
read -s -n 1 key

# delete old directory
if [ -d "./venv" ]
then
  echo "Removing old virtual environment..."
  rm -rf ./venv
fi

echo "Creating new virtual environment..."
virtualenv -p /usr/bin/$PYTHON ./venv

echo "Installing dependencies..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install --upgrade setuptools
./venv/bin/pip install Cython
./venv/bin/pip install numpy
./venv/bin/pip install tensorflow
./venv/bin/pip install ../ufdl-json-messages
./venv/bin/pip install ../ufdl-python-client
./venv/bin/pip install wai.lazypip
./venv/bin/pip install wai.annotations
./venv/bin/pip install pika
./venv/bin/pip install psutil
./venv/bin/pip install pyyaml
./venv/bin/pip install ../ufdl-job-launcher
