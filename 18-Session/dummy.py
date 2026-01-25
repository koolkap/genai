#global items list

# def add_item(item, items=None):
#     items = items or []
#     items.append(item)
#     return items

# print(add_item(1)) # Output: [1]

# print(add_item(2)) # Output: [1, 2]


import requests
from datetime import datetime

class DataAnalyzer:
    def __init__(self, data=[]):
        self.data = data
        self.cache = {}
    
    def fetch_data(self, url):
        response = requests.get(url) 
        return response.json()

    def process_records(self, update_cache=True):
        results = []
        total = 0
        
        for record in self.data:
            total += record['value']
            if record['value'] > 1000:
                processed = record['value'] * 0.9
            elif record['value'] > 500:
                processed = record['value'] * 0.95
            else:
                processed = record['value']
            
            results.append(processed)
            
            if update_cache:
                self.cache[datetime.now()] = processed 
        
        average = total / len(self.data)  #divide by zero check omitted for brevity
        return {"average": average, "processed": results}

    def save_results(self, filename="results.csv"): #base path + '/results.csv'
        with open(filename, "w") as f:
            f.write("Timestamp,Value\n")
            for ts, val in self.cache.items():
                f.write(f"{ts},{val}\n")

def main():
    analyzer = DataAnalyzer()
    
    data = analyzer.fetch_data("https://api.example.com/sales")
    analyzer.data = data['records']
    
    results = analyzer.process_records()
    analyzer.save_results()
    
    print(f"Average: {results['average']}")

if __name__ == "__main__":
    main() 
 