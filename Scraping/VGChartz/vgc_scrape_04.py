from bs4 import BeautifulSoup, element
import urllib
import pandas as pd
import numpy as np

title = []
platform = []
publisher = []
developer = []
vgc_score = []
critic_score = []
user_score = []
total_shipped = []
total_sales = []
sales_na = []
sales_pal = []
sales_jp = []
sales_ot = []
release = []
update = []

urlhead = 'https://www.vgchartz.com/games/games.php?page='
urltail = '&console=&region=All&developer=&publisher=&genre=&boxart=Both&ownership=Both'
urltail += '&results=1000&order=Sales&showtotalsales=1&shownasales=1&showpalsales=1&showjapansales=1'
urltail += '&showothersales=1&showpublisher=1&showdeveloper=1&showreleasedate=1&showlastupdate=1'
urltail += '&showvgchartzscore=1&showcriticscore=1&showuserscore=1&showshipped=1'


# We may need to fragment the number of pages to avoid server kicks
pages = 21
rec_count = 0

for page in range(16, pages):
    surl = urlhead + str(page) + urltail
    r = urllib.request.urlopen(surl).read()
    soup = BeautifulSoup(r, features="lxml")
    print(f"Page: {page}")
    
    chart = soup.find('div', id='generalBody').find('table')
    
    for row in chart.find_all('tr')[3:]:
        try:
            col = row.find_all('td')

            # extract data into column data
           
            column_2 = col[2].find('a').string.strip()                # Title
            column_3 = col[3].find('img')['alt'].strip()    # Platform
            column_4 = col[4].string.strip()                # Publisher
            column_5 = col[5].string.strip()                # Developer
            column_6 = col[6].string.strip()                # VGChartz Score
            column_7 = col[7].string.strip()                # Critic Score
            column_8 = col[8].string.strip()                # User Score
            column_9 = col[9].string.strip()                # Total Shipped
            column_10 = col[10].string.strip()              # Total Sales
            column_11 = col[11].string.strip()              # NA Sales
            column_12 = col[12].string.strip()              # PAL Sales (EU)
            column_13 = col[13].string.strip()              # Japan Sales
            column_14 = col[14].string.strip()              # Other Sales
            column_15 = col[15].string.strip()              # Release Date
            column_16 = col[16].string.strip()              # Last Update

            # Add Data to columns
            # Adding data only if able to read all of the columns
            
            
            title.append(column_2)
            platform.append(column_3)
            publisher.append(column_4)
            developer.append(column_5)
            vgc_score.append(column_6)
            critic_score.append(column_7)
            user_score.append(column_8)
            total_shipped.append(column_9)
            total_sales.append(column_10)
            sales_na.append(column_11)
            sales_pal.append(column_12)
            sales_jp.append(column_13)
            sales_ot.append(column_14)
            release.append(column_15)
            update.append(column_16)

            rec_count += 1

        except:
            print('Got Exception')
            continue

columns = {'Title': title, 'Platform': platform, 'Publisher': publisher, 'Developer': developer, 'VGC_Score': vgc_score,
           'Critic_Score': critic_score, 'User_Score': user_score, 'Total_Shipped': total_shipped, 'Total_Sales': total_sales, 'NA_Sales': sales_na,
           'EU_Sales': sales_pal, 'JP_Sales': sales_jp, 'Other_Sales': sales_ot, 'Release': release, 'Last_Update': update}

print (rec_count)
df = pd.DataFrame(columns)
print(df)
df = df[['Title', 'Platform', 'Publisher', 'Developer', 'VGC_Score', 'Critic_Score', 'User_Score', 'Total_Shipped', 'Total_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
         'Release', 'Last_Update']]
# del df.index.name
df.to_csv("vgsales_04.csv", sep=",", encoding='utf-8')

print('done!')