import requests

# Local API URL
API_URL = "http://127.0.0.1:5000/similarity"

# Test cases
test_cases = [
    {
        "text1": "The new legislation aims to reduce carbon emissions by 30% over the next decade.",
        "text2": "The bill proposes cutting greenhouse gases by a third within ten years."
    },
    {
        "text1": "Taylor Swift announces world tour with 30 stops across 5 continents.",
        "text2": "Scientists discover new species of frogs in Amazon rainforest."
    },
    {
        "text1": "nuclear body seeks new tech to detect secret nuclear sites.",
        "text2": "Terror suspects face arrest in Spain and Italy."
    }
]

# Test each case
for case in test_cases:
    response = requests.post(API_URL, json=case)
    print(f"Text 1: {case['text1']}")
    print(f"Text 2: {case['text2']}")
    print(f"Similarity Score: {response.json()['similarity score']}")
    print("-" * 80)