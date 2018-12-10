# Read me

All code is written in python, some are written regular .py-files and others in jupyter-notebooks.

## Required packages

These are the packages used in the project. In hindsight it would have been a good idea to use a virtual environment. 

* pandas
* numpy
* matplotlib
* seaborn
* dateutil
* gmplot
* IPython
* tensorflow
* keras
* hyperas
* hyperopt
* gpflow
* GPy
* sklearn
* scipy
* mpl_toolkits

As well as some standard python libraries.

## Pre-processing and data exploration

Under the folder `pre-processing/`

The file `events_types.txt` shows the format of the events in the raw log files.
The file `events_lengths.txt` shows the number of fields within different events.

The script `parse_all_logs.py` is used to extract a complete journey, defined as all journeys (in one direction) from station X to station Y.
The fields in the file `bus_information.py` are used to define the stations and bus line to extract in the `parse_all_logs.py` script.

The notebook `inspect_journey.ipynb` is used to further examine and filter out bad data from the extracted journeys retrieved by the `parse_all_logs.py` script.

The notebook `busline_data_distribution.ipynb` is used to plot distributions of travel time of extracted journeys within different time-of-day and time-of-year intervals.


## Baseline models

Under the folder `baseline_models/`

The notebook `baseline_model_203.ipynb` contains the baseline model for bus 203.
The notebook `baseline_model_211.ipynb` contains the baseline model for bus 211.
