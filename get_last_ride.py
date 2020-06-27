from zwift import Client
import json
import pandas as pd


with open('auth', 'r') as f:
    auth = json.load(f)
    
client = Client(auth['user'], auth['password'])
p = client.get_profile()
pdata = p.profile
adata = p.latest_activity

start_time = pd.Timestamp(adata['startDate'])
title = adata['name']
assert start_time.tzinfo.tzname(0) == 'UTC'
start_time = start_time.astimezone(None) + pd.Timedelta(hours=7)
moving_time = pd.Timedelta(seconds=adata['movingTimeInMs'] / 1000)
distance = adata['distanceInMeters'] / 1000
elevation = adata['totalElevation']
avg_watt = adata['avgWatts']
calories = adata['calories']

level = pdata['achievementLevel'] / 100
total_distance = pdata['totalDistance'] / 1000
total_climb = pdata['totalDistanceClimbed']
total_xp = pdata['totalExperiencePoints']
total_drops = pdata['totalGold']
ftp = pdata['ftp']
weight = pdata['weight'] / 1000

print(f'Last activity: {start_time}')
print(f'Title: {title}')
print(f'Distance: {distance}')
print(f'Level: {level}')
print(f'Total XP: {total_xp}')

rides = pd.read_excel('rides.xlsx', parse_dates=['dtime'])
last_saved_ride = rides['dtime'].max()
last_xp = rides['totalxp'].max()
last_drops = rides['totaldrops'].max()
last_calories = rides['totalcalories'].max()
if last_saved_ride >= start_time or (start_time - last_saved_ride) < pd.Timedelta(seconds=300):
    print(f'The rides.xlsx is up to date (latest ride: {last_saved_ride})')
else:
    answer = input("Update rides? (Y/N) ")
    if 'Y' in answer:
        secs = moving_time.seconds
        duration = f'{secs//3600:02d}:{(secs % 3600)//60:02d}:{secs % 60:02d}'
        row = pd.Series({
            'dtime': start_time,
            'title': title,
            'type': 'interval',
            'distance': distance,
            'totaldistance': total_distance,
            'elevation': elevation,
            'totalelevation': total_climb,
            'calories': calories,
            'totalcalories': last_calories + calories,
            'xp': total_xp - last_xp,
            'totalxp': total_xp,
            'drops': total_drops - last_drops,
            'totaldrops': total_drops,
            'ftp': ftp,
            'weight': weight,
            'time': duration,
            'minutes': secs / 60,
            'avg_watt': avg_watt,
            })
        rides = rides.append(row, ignore_index=True)
        rides.to_excel('rides.xlsx', index=False)
        print('rides.xlsx saved')
