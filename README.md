# Read me

All code is written in python, some are written regular .py-files and others in jupyter-notebooks. Below the required packages are listed as well as descriptions of the implementations.

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

## Artificial Neural Networks

Implementations for neural networks are found under the `ANN` folder. 

Models are created in a 4 step process denoted by the number in the name of the notebook.

As an example, the notebook `1. Pre-processing` is the first notebook to be run.

The only notebook that will differ across models is the notebooks regarding step 2 where models are created and evaluated. The other notebooks will be the same, the only thing that needs change is variables regarding what models is used.

The notebook `2. Extra - parameter optimization` is a special step used for deciding the best parameters, in this case made for model M2.

## Gaussian Process Regression (GPR)

Under the folder `GP/`

The numbered notebooks contain the final code used in the project. The other notebooks contain some legacy code and experiments done in order to perform the project.

* `Arrival Time Prediction 1 - Pre Processing`: Extracts and pre-processes data for the GPR model.
* `Arrival Time Prediction 2 - Synchronisation`: Training of synchronisation GP, exploration of priors, and relevant plots.
* `Arrival Time Prediction 3 - Training`: Training of prediction GPs and some predictions.
* `Arrival Time Prediction 4 - Performance`: Evaluates performance of the GPR model.

List of legacy notebooks:

* `Gaussian Processes with GPflow`: Exploration of GPflow, contains basic experiments to get acquainted with the library.
* `Arrival Time Prediction - Model`: Early attempt at synchronising trajectories.
* `Arrival Time Prediction - Pointwise`: Early attempt at pointwise predictions.
* `Arrival Time Prediction - Synchronisation`: Early attempt at synchronisation, contains previous approach to enforcing smoothness by duplicating and shifting the trajectory.

Custom modules:
* `gp_gpflow`: GPflow implementations for GPR model
* `gp_gpy`: GPy implementations for GPR model
* `plot`: Plotting related functionality
* `synch`: Synchronisation related functionality