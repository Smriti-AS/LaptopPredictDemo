import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

data = pd.read_csv(r"C:\College GLIM All Data\DATA SCIENCE\LAB Assignment 1\laptop_data.csv")
data.shape
data.head(10)
data.tail()
data.info()
data.isnull().sum()
data.duplicated().sum()
data.drop(columns=["Unnamed: 0"],inplace=True)
data["Ram"]=data["Ram"].str.replace("GB","")
data["Ram"]=data["Ram"].astype("int")
data["Weight"]=data["Weight"].str.replace("kg","")
data["Weight"]=data["Weight"].astype("float")

categorical_columns = data.select_dtypes(include=['object']).columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns

for col in categorical_columns:
    print(f"\nColumn: {col}")
    print(data[col].value_counts())
    
plt.figure(figsize=(10, 6))
sns.histplot(data['Price'], bins=50, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Company', data=data, color='lightgreen')
plt.xticks(rotation=90)
plt.title('Laptops per Company')
plt.xlabel('Company')
plt.ylabel('Number of Laptops')
plt.show()

# Calculate the average price per company
average_price_per_company = data.groupby('Company')['Price'].mean().reset_index()

# Create the bar plot using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='Company', y='Price', data=average_price_per_company)
plt.xticks(rotation=90)
plt.title('Average Laptop Price per Company')
plt.xlabel('Company')
plt.ylabel('Average Price')
plt.show()

data["Touchscreen"]=data["ScreenResolution"].apply(lambda x:1 if "Touchscreen" in x else 0)

data["IPS"]=data["ScreenResolution"].apply(lambda x:1 if "IPS" in x else 0)

data['ScreenResolution'] = data['ScreenResolution'].astype(str)
data['Resolution_Width'] = data['ScreenResolution'].str.extract('(\d+)x\d+').astype(int)
data['Resolution_Height'] = data['ScreenResolution'].str.extract('\d+x(\d+)').astype(int)

data.head()

data['PPI'] = (((data['Resolution_Width']**2) + (data['Resolution_Height']**2))**0.5/data['Inches']).astype('float')

print(data['Cpu'].unique())

data['Cpu Name'] = data['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
        
data['Cpu brand'] = data['Cpu Name'].apply(fetch_processor)

# Convert 'Memory' column to string and clean up
data['Memory'] = data['Memory'].astype(str).replace('\.0', '', regex=True)
data['Memory'] = data['Memory'].str.replace('GB', '').str.replace('TB', '000')

# Split and process 'Memory' into components
memory_split = data['Memory'].str.split("+", expand=True).fillna("0")

# Extract numeric values from each component
memory_split = memory_split.apply(lambda col: col.str.extract(r'(\d+)')[0].fillna(0).astype(int))

# Create storage type binary indicators and calculate capacities
storage_types = ['HDD', 'SSD', 'Hybrid', 'Flash Storage']
for i, layer in enumerate(memory_split.columns, start=1):
    for storage in storage_types:
        data[f'Layer{i}{storage.replace(" ", "_")}'] = memory_split[layer].where(
            data['Memory'].str.contains(storage, na=False, regex=True), 0
        )

# Combine capacities for each storage type
for storage in storage_types:
    data[storage.replace(" ", "_")] = sum(
        data[f'Layer{i}{storage.replace(" ", "_")}'] for i in range(1, len(memory_split.columns) + 1)
    )

# Drop intermediate columns
data.drop(columns=[col for col in data if col.startswith('Layer')], inplace=True)

# Handle missing or malformed values
data['Gpu brand'] = data['Gpu'].apply(lambda x: x.split()[0] if isinstance(x, str) and len(x.split()) > 0 else None)

data['Gpu brand'].value_counts() 
data = data[data['Gpu brand'] != 'ARM']

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

data['os'] = data['OpSys'].apply(cat_os)

# Defining features (X) and target variable (y)
X = data[['Ram', 'Weight', 'Touchscreen', 'IPS', 'PPI', 'Cpu brand', 'HDD', 'SSD', 'Hybrid', 'Flash_Storage', 'Gpu brand', 'os']]
y = data['Price']

# Converting categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Cpu brand', 'Gpu brand', 'os'])

# Min-Max scaling for X
min_max_scaler_X = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler_X.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Min-Max scaling for y
y = np.array(y).reshape(-1,1)
min_max_scaler_y = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler_y.fit_transform(y)

with open('models/min_max_scaler_y.pkl', 'wb') as file:
    pickle.dump(min_max_scaler_y, file)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


from sklearn.ensemble import RandomForestRegressor

# Initialize and train a Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.ravel())

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R-squared: {r2_rf}")

# Save the model to a file
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Save the scaler for X
with open('min_max_scaler_X.pkl', 'wb') as file:
    pickle.dump(min_max_scaler_X, file)

# Save the scaler for y (optional, if needed for scaling predictions)
with open('min_max_scaler_y.pkl', 'wb') as file:
    pickle.dump(min_max_scaler_y, file)
    
# Save the exact columns after one-hot encoding
dummy_columns = X.columns.tolist()
with open('dummy_columns.pkl', 'wb') as file:
    pickle.dump(dummy_columns, file)

