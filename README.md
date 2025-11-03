# Rail-Mileage-Calculator
Using FRA's North American Rail Network (NARN) data, create a basic app to calculate mileage between nodes, highlighting rail ownership, calculating rail diversion distance, and some cost information.

The basic GUI is built within a tkinter window. The "app" requires the user to separately download the csv files containing edge and node data from the FRA's NARN maps (linked below).
The Python code then extracts key columns from the .csv, constructs a graph using the NetworkX library. 




FRA Geospatial Data: https://railroads.dot.gov/rail-network-development/maps-and-data/maps-geographic-information-system/maps-geographic

venv/
__pycache__/
*.pyc
.DS_Store
*.sqlite
