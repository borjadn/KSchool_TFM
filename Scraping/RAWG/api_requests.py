import requests
import pandas as pd

api_key = YOUR_API_KEY
url = 'https://api.rawg.io/api/games?key=' + api_key + '&page='

hasNextPage = True
page_n = 1
games = pd.DataFrame()

##### Run the code below again if you get a 'JSONDecodeError: Expecting value: line 1 column 1 (char 0) error' #####

while hasNextPage:
    response = requests.get(url+str(page_n)).json()
    
    aux = pd.DataFrame.from_dict(response['results'], orient = 'columns')
    
    games = pd.concat([games, aux])
        
    print('Page: ' + str(page_n))
    
    page_n += 1
    
    if response['next'] != (url + str(page_n)):
        
        hasNextPage = False
    
print('Finished')

