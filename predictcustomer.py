import csv
from datetime import datetime
import difflib
from sklearn.linear_model import LinearRegression

# Define headers used in CSV files
headers = ["Description", "Job Date", "Address", "Document Number", "Ward", "Client", "Agent", "Status", "Response"]

# declare a dictionary to hold the wards
wards = {}
#Read the wards CSV file
with open('wards.csv') as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		next(readCSV)
		for row in readCSV:
			# row[0] is the name of the ward. row[1] is the average income
			wards[row[0]] = float(row[1])

# Declare this for wards not found in the wards CSV file
averageIncome = sum(wards.values()) / float(len(wards.values()))

# filename: CSV file to open, data: Where the csv data is read into, averageIncome: to be used when ward cannot be found
# learning: boolean determining if the csv file provided contains learning or testing data, normalize: Should the data be normalized before savewd into 'data'
def readCSVFile(filename, data, averageIncome, learning, normalize):
	with open(filename) as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			# Skip the first line containing the column title
			next(readCSV)
			# Create dictionary containing empty lists for each header
			for header in headers:
				data[header] = []

			if normalize:
				for row in readCSV:
					# Read and convert Description colomn. Get length of string. For better results extract relevant keywords.
					data["Description"].append(len(row[0]))
					# Read and convert Job Date colomn. Convert from date to duration in days from current date.
					date = datetime.strptime(row[1], '%a %d %b %Y')
					current_date = datetime.now()
					data["Job Date"].append((current_date - date).days)
					# Read and convert Address colomn. Leave as it is.
					data["Address"].append(row[2])
					# Read and convert Document colomn. Extract integer from string.
					data["Document Number"].append([int(s) for s in row[3].split() if s.isdigit()][0])
					# Read and convert Ward colomn. Leave as it is.
					data["Ward"].append(row[4])
					# Read and convert Address colomn. Leave as it is.
					data["Client"].append(row[5])
					# Read and convert Agent colomn. null -> 0 otherwise 1
					data["Agent"].append(0 if row[6] == "null" else 1)
					# Read and convert Status colomn. Pending -> 1 otherwise 0
					data["Status"].append(1 if row[7] == "Pending" else 0)
					# Read and convert Response colomn. TRUE -> 1 otherwise 0
					if learning == True:
						data["Response"].append(1 if row[8] == "TRUE" else 0)

				# Normalize Description
				data["Description"] = [float(i)/max(data["Description"]) for i in data["Description"]]
				# Normalize Job Date
				data["Job Date"] = [(max(data["Job Date"]) - float(i))/max(data["Job Date"]) for i in data["Job Date"]]
				# Normalize Document Number
				data["Document Number"] = [float(i)/max(data["Document Number"]) for i in data["Document Number"]]
				# Normalize Wards:
				# Search for most similar ward name from the wards dictionary and replace the ward name with the average income
				for index, ward in enumerate(data["Ward"]):
					match = difflib.get_close_matches(ward, wards.keys())
					if match:
						data["Ward"][index] = wards[match[0]]
					else:
						data["Ward"][index] = averageIncome
				# Then normalize
				data["Ward"] = [float(i)/max(data["Ward"]) for i in data["Ward"]]
			else:
				for row in readCSV:
					data["Description"].append(row[0])
					data["Job Date"].append(row[1])
					data["Address"].append(row[2])
					data["Document Number"].append(row[3])
					data["Ward"].append(row[4])
					data["Client"].append(row[5])
					data["Agent"].append(row[6])
					data["Status"].append(row[7])

learningData = {}

#Read the learning CSV file
readCSVFile('100Jobs - Learn.csv', learningData, averageIncome, True, True)

testingData = {}
#Read the testing CSV file
readCSVFile('100Jobs - Test.csv', testingData, averageIncome, False, True)

testingDataOriginal = {}
#Read the testing CSV file again, without normalizing
readCSVFile('100Jobs - Test.csv', testingDataOriginal, averageIncome, False, False)

# Let's try to find a pattern in our data.
# Zip the learningData to an accepted format
combineTestData = list(zip(learningData["Description"], learningData["Job Date"], learningData["Document Number"], learningData["Ward"], learningData["Agent"], learningData["Status"]))
# Use sklearn to build a prediciton model
model = LinearRegression()
model.fit(combineTestData, learningData["Response"])

# Add colomn to testing data for the prediction

# Get the test data for easier iteration
combineTestData = list(zip(testingData["Description"], testingData["Job Date"], testingData["Document Number"], testingData["Ward"], testingData["Agent"], testingData["Status"]))

for testData in combineTestData:
	X_predict = [[testData[0], testData[1], testData[2], testData[3], testData[4], testData[5]]]
	y_predict = model.predict(X_predict)

	# Save prediction in original testing data variable
	testingDataOriginal["Response"].append(y_predict[0])

#print(testingDataOriginal["Description"])
#print(testingDataOriginal["Job Date"])
#print(testingDataOriginal["Address"])
#print(testingDataOriginal["Document Number"])
#print(testingDataOriginal["Ward"])
#print(testingDataOriginal["Client"])
#print(testingDataOriginal["Status"])
#print(testingDataOriginal["Response"])


outputHeaders = ['', "Description", "Job Date", "Address", "Document Number", "Ward", "Client", "Agent", "Status", "Prediction"]

# Write prediction CSV file
with open('100Jobs - Prediction.csv', "w") as csv_file:
	writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
	# Write first row of headers
	writer.writerow(outputHeaders)
	strs = ["" for x in range(len(testingDataOriginal["Description"]))]
	# Convert colomns to rows for easier writing
	rows = zip(strs, testingDataOriginal["Description"], testingDataOriginal["Job Date"], testingDataOriginal["Address"],
	 testingDataOriginal["Document Number"], testingDataOriginal["Ward"],
	 testingDataOriginal["Client"],testingDataOriginal["Agent"], testingDataOriginal["Status"], testingDataOriginal["Response"])
	for value in rows:
		writer.writerow(value)