all:

install:
	#install python
	sudo apt-get -y install python 2.7.6
	sudo apt-get -y install python-pip
	sudo apt-get -y install python-virtualenv
	sudo apt-get -y install build-essential python-setuptools python-dev python-numpy python-numpy-dev libjpeg-dev python-scipy libatlas-dev libatlas3gf-base g++ python-matplotlib ipython
	sudo apt-get -y install libjpeg-dev libfreetype6 libfreetype6-dev zlib1g-dev
	sudo apt-get -y install libblas-dev liblapack-dev libatlas-base-dev gfortran
	sudo apt-get -y install libhdf5-dev

	# creating virtual environment
	virtualenv --system-site-packages env	
	env/bin/pip install -r requirements.txt

