import sqlite3

# Verify the sample data
conn = sqlite3.connect('potholes.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM potholes")
rows = cursor.fetchall()
conn.close()

for row in rows:
    print(f"ID: {row[0]}, Lat: {row[1]}, Lng: {row[2]}, Severity: {row[3]}")
