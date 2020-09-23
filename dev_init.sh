#!/bin/bash

#############
# FUNCTIONS #
#############

# the usage of this script
function usage()
{
   echo
   echo "${0##*/} [-y] [-u] [-h]"
   echo
   echo "Initializes the virtual environment for the UFDL job-launcher plugins."
   echo
   echo " -h   this help"
   echo " -y   do not prompt user, assume 'yes'"
   echo " -u   update any repositories first"
   echo
}

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
    exit 1
  else
    echo "...is present"
  fi
}

function update_repository()
{
  echo "Updating repo $REPO..."
  CURRENT="`pwd`"
  cd "../$REPO"
  git pull
  cd "$CURRENT"
}

##############
# PARAMETERS #
##############

VENV="venv.dev"
PROMPT="yes"
UPDATE="no"
while getopts ":hyu" flag
do
   case $flag in
      y) PROMPT="no"
         ;;
      u) UPDATE="yes"
         ;;
      h) usage
         exit 0
         ;;
      *) usage
         exit 1
         ;;
   esac
done

##########
# CHECKS #
##########

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
  exit 1
fi

REPO="ufdl-json-messages"
check_repository

REPO="ufdl-python-client"
check_repository

REPO="ufdl-job-launcher"
check_repository

#############
# EXECUTION #
#############

if [ "$PROMPT" = "yes" ]
then
  echo
  echo "Press any key to start setup of '$VENV' for 'UFDL job-launcher-plugins' (using $PYTHON)..."
  read -s -n 1 key
fi

# update repos
if [ "$UPDATE" = "yes" ]
then
  REPO="ufdl-json-messages"
  update_repository

  REPO="ufdl-python-client"
  update_repository

  REPO="ufdl-job-launcher"
  update_repository

  git pull
fi

# delete old directory
if [ -d "./$VENV" ]
then
  echo "Removing old virtual environment..."
  rm -rf ./$VENV
fi

echo "Creating new virtual environment $VENV..."
virtualenv -p /usr/bin/$PYTHON ./$VENV

echo "Installing dependencies..."
./$VENV/bin/pip install --upgrade pip
./$VENV/bin/pip install --upgrade setuptools
./$VENV/bin/pip install Cython
./$VENV/bin/pip install numpy
./$VENV/bin/pip install tensorflow
./$VENV/bin/pip install ../ufdl-json-messages
./$VENV/bin/pip install ../ufdl-python-client
./$VENV/bin/pip install wai.lazypip
./$VENV/bin/pip install wai.annotations
./$VENV/bin/pip install psutil
./$VENV/bin/pip install pyyaml
./$VENV/bin/pip install ../ufdl-job-launcher
./$VENV/bin/pip install .
