import re
import pandas as pd
def extract_digital_ucc(filepath):

    digital_keywords =[
        "computer","software","internet","hardware","network","online","video games","streaming","downloading",
        "digital","applications","satellite television","cellular phone","portable","accessories","television",
        "telephone","ebook","ovens","speakers","Photographic equipment"
    ]

    digital_ucc =[]

    try:
        with open(filepath,'r') as file:
            for line in file:
                match = re.search(r'^\d+\s+\d+\s+(.*?)\s{2,}(\d{6})\s+I', line)

                if match:
                    description = match.group(1).strip()
                    ucc_code = match.group(2)
                
                    if any(key in description.lower() for key in digital_keywords):
                        digital_ucc.append({
                            "UCC": ucc_code,
                            "Description": description
                        })
    except FileNotFoundError:
        return "file not found"
    
    return digital_ucc


if __name__ == "__main__":
    results = extract_digital_ucc("/home/alain/Documents/Data science notes/stubs/CE-HG-Inter-2023.txt")
    
    for item in results:
        print(f"UCC: {item['UCC']} | {item['Description']}")