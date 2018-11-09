import pandas as pd
import numpy as np
import pickle
import os
import glob
import bus_information

# global counter for journey numbers
journey_number = 1

# events we care about
events = {
	'ObservedPositionEvent',
	'EnteredEvent',
	'ExitedEvent',
	'PassedEvent',
	'ArrivedEvent',
	'DepartedEvent',
	'ParameterChangedEvent',
	'JourneyStartedEvent',
	'JourneyCompletedEvent',
	'JourneyAssignedEvent'
}

def parse_line(line):
	"""Clean and split strings by pipe-symbol, returning a list"""
	return [x for x in str(line.strip()).strip("'b").strip("'").split('|') if x]

def extract_fields(x):
	"""Extract the relevant fields from the list returned by parse_line()"""
	if (x[2] in {'ParameterChangedEvent', 'JourneyAssignedEvent'} and len(x) != 14)\
	or x[2] not in events\
	or x[6] != 'Bus':
		return None

	return [
		x[0], # timestamp
		x[2], # event
		x[9], # vehicle_id
		x[11][7:11] if x[11][:4] in {'9011', '9012', '9015'} else 0, # bus line number
		x[10].split(",")[0], # longitude 
		x[10].split(",")[1], # latitude
		x[12] if x[2] == 'ObservedPositionEvent' else -1, # direction
		x[13] if x[2] == 'ObservedPositionEvent' else -1, # speed
		x[15] if x[2] not in {
			'ObservedPositionEvent',
			'ParameterChangedEvent',
			'JourneyStartedEvent',
			'JourneyCompletedEvent',
			'JourneyAssignedEvent'
		} else None # bus station number
	]

def split_log(filename):
	"""Parse a .log file and split it into dataframes of 2M rows each"""

	cols = ['timestamp', 'event', 'vehicle_id', 'line', 'longitude', 'latitude', 'direction', 'speed', 'station']

	df = pd.DataFrame(columns=cols).astype(dtype={
		'timestamp': 'object', 
		'event': 'object',
		'vehicle_id': 'int64',
		'line': 'int64',
		'longitude': 'float64', 
		'latitude': 'float64',
		'direction': 'float64',
		'speed': 'float64',
		'station': 'object'
	})

	lines = list()
	flush = 1e5
	i = 1

	with open(filename, 'rb') as f:
		while True:
			try:
				read = f.readline()
				line = parse_line(read) if read else ""
				
				# parse relevant events
				extracted_fields = extract_fields(line) if line else None
				if extracted_fields:
					lines.append(extracted_fields)
					
				# clear the memory of the list and write to our dataframe when we have read 100k lines
				if len(lines) == flush or line == "":
					df = df.append(pd.DataFrame(data=lines, columns=cols)).astype(dtype={
						'timestamp': 'object', 
						'event': 'object',
						'vehicle_id': 'int64',
						'line': 'int64',
						'longitude': 'float64', 
						'latitude': 'float64',
						'direction': 'float64',
						'speed': 'float64',
						'station': 'object'
					})
					lines = list()
					
				# if we have reached 2M rows in our dataframe, or if we reached EOF
				# serialize & save the dataframe and clear from memory
				if len(df.index) >= 2e6 or line == "":
					pickle.dump(df, open(f'log_splits/split-{i}.p', 'wb'))
					print(f'processed and serialized {i} dataframes')
					print(f'the latest df has {len(df.index)} rows')
					df = pd.DataFrame(columns=cols)
					i += 1
					
				if line == "":
					print('EOF reached for this log!')
					break
						
			except Exception as e:
				print(f'this line caused exception:\n{line}')
				print(e)

def extract_journey(bus_line, first_station, second_station, last_station, date):
    """Extracts the journeys a bus line from first_station to last_station"""

    global journey_number

    journey_cols = [
    'timestamp', 
    'event', 
    'vehicle_id', 
    'line', 
    'longitude', 
    'latitude', 
    'direction', 
    'speed', 
    'station',
    'journey_number',
    'segment_number'
    ]

    busline_df = pd.DataFrame(columns=journey_cols).astype(dtype={
        'timestamp': 'object', 
        'event': 'object',
        'vehicle_id': 'int64',
        'line': 'int64',
        'longitude': 'float64', 
        'latitude': 'float64',
        'direction': 'float64',
        'speed': 'float64',
        'station': 'object',
        'journey_number': 'int64',
        'segment_number': 'int64'
        })

    # iterate over all pickled dataframes
    files = os.listdir('log_splits/')
    for j in range(len(files)):
        file = f'split-{j+1}.p' # workaround to get correct ordering of files
        print(f'** Starting on new file: {file} **')
        df = pickle.load(open(f'log_splits/{file}', 'rb'))
        # get the unique vehicle ids that drove bus line of interest
        vehicle_ids = df[(df['line'] == bus_line)]['vehicle_id'].unique()
        # iterate over all vehicle ids
        for vid in vehicle_ids:
            vals = list()
            started = False # flag for starting to collect data
            first_entered_event = False # flag for checking the first bus stop after journey started
            # extract slice of df for this vehicle id and sort by timestamp
            df_ = df[df['vehicle_id'] == vid].copy().sort_values('timestamp')
            for row in df_.itertuples():
                # start collecting data at when JourneyStartedEvent fires
                # if the started event was fired at the starting position of interest
                # (longitude difference is less than 0.001)
                if not started \
                and row[2] == 'JourneyStartedEvent' \
                and row[4] == bus_line:
                    started = True
                    first_entered_event = True
                    segment_number = 1
                    vals.append([x for x in row[1:]] + [journey_number, segment_number])
                    continue

                # as long as the bus line is the same, we are on the same route, 
                # since we are only looking at a single vehicle and events are sorted by timestamp
                # keep collecting data until we hit a the last station, where we write to dataframe
                # and start over
                if started and (row[4] == bus_line or row[4] == 0):
                    # if we hit a JourneyStartedEvent or JourneyCompletedEvent after our initial
                    # JourneyStartedEvent, we have passed the last station without firing an EnteredEvent
                    # in that case, scrap this data and start over
                    if row[2] in {'JourneyStartedEvent', 'JourneyCompletedEvent'}:
                        started = False
                        first_entered_event = False
                        vals = list()
                        continue

                    # if we have just begun to collect data, we need to verify that we are going in
                    # the indended direction for this bus line by checking the station of the first EnteredEvent
                    # (this could be the first or the second station depending on when the EnteredEvent fired)
                    if first_entered_event and row[2] == 'EnteredEvent':
                        if row[9] in {first_station, second_station}:
                            first_entered_event = False
                        else:
                            started = False
                            first_entered_event = False
                            vals = list()
                            continue

                    # if all is good, keep adding rows to the list for this journey
                    vals.append([x for x in row[1:]] + [journey_number, segment_number])
                    # if we hit EnteredEvent, increment segment_number if the station is not the last station,
                    # otherwise, if it is the last station, we have reached the end for this journey
                    if row[2] == 'EnteredEvent':
                        if row[9] == last_station:
                            busline_df = busline_df.append(pd.DataFrame(data=vals, columns=journey_cols)).astype(dtype={
                                'timestamp': 'object', 
                                'event': 'object',
                                'vehicle_id': 'int64',
                                'line': 'int64',
                                'longitude': 'float64', 
                                'latitude': 'float64',
                                'direction': 'float64',
                                'speed': 'float64',
                                'station': 'object',
                                'journey_number': 'int64',
                                'segment_number': 'int64'
                            })
                            print(f'Successfully collected {journey_number} journeys!')
                            started = False
                            first_entered_event = False
                            vals = list()
                            journey_number += 1
                        # since sometimes EnteredEvent fires right after the JourneyStartedEvent,
                        # we don't want to increment segment number if we are still at the starting station.
                        # If we hit an EnteredEvent not being the starting station, increment the segment counter
                        elif row[9] != first_station:
                            segment_number += 1
                            
                # if bus line changed anywhere in the middle of this sequence, something went wrong
                # we then scrap the collected data and start over
                elif started:
                    started = False
                    first_entered_event = False
                    vals = list()

    pickle.dump(busline_df, open(f'buslines/{bus_line}_{date}.p', 'wb'))

def main():
	files = glob.glob('/home/max/bus_logs/data/*')
	for f in sorted(files):
		split_log(f)
		print(f'------- Finished splitting {f} -------')
		extract_journey(
			bus_information.line_number,
			bus_information.first_station,
			bus_information.second_station,
			bus_information.last_station,
			f[-12:-4] # date
			)
		print(f'------- Finished extracting journeys for {f} -------')
		to_delete = glob.glob('/home/max/det-ar-lugnt/log_splits/*')
		for d in to_delete:
			os.remove(d)
		print(f'------- Finished processing {f} -------')

	print("DET AR LUGNT!")

if __name__ == "__main__":
	main()