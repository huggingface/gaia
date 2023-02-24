import json
import os
import pprint

import requests

pp = pprint.PrettyPrinter(indent=4)

query = "don't go chasing waterfalls"

# run the following in commandline, replace address with your server address:
# export address="http://12.345.678.910:8080"

output = requests.post(
    os.environ.get("address"),
    headers={"Content-type": "application/json"},
    data=json.dumps({"query": query, "corpus": "c4"}),
    timeout=60,
)

results = json.loads(output.text)

for result in results["results"]:
    print(result)
    print()
    print()

print("Num results:", len(results["results"]))
