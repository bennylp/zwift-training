import datetime
from fitparse import FitFile
import glob
import json
import numpy as np
import os
import pandas as pd
import pytz
from xml.dom import minidom


def path_val(element, tag_names, default=None):
    tag_names = tag_names.split('|')
    for tag_name in tag_names:
        nodes = element.getElementsByTagName(tag_name)
        if len(nodes) == 0:
            if default is not None:
                return default
            else:
                raise KeyError(f'{tag_name} not found')
        if len(nodes) > 1:
            raise RecursionError(f'Multiple {tag_name}s found')
        element = nodes[0]
    
    rc = []
    for node in element.childNodes:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)
    
    
def tcx_to_csv(path):
    with open(path, 'r') as f:
        doc = f.read().strip()
    doc = minidom.parseString(doc)
    
    sport = doc.getElementsByTagName('Activities')[0] \
               .getElementsByTagName('Activity')[0] \
               .attributes['Sport'].value \
               .lower()
    title = path_val(doc, 'Activities|Activity|Notes', '')
    meta = dict(sport=sport, title=title)
    
    trackpoints = doc.getElementsByTagName('Trackpoint')
    rows = []
    for trackpoint in trackpoints:
        raw_time = pd.Timestamp(path_val(trackpoint, 'Time'))
        try:
            latt = path_val(trackpoint, 'LatitudeDegrees', np.NaN)
            long = path_val(trackpoint, 'LongitudeDegrees', np.NaN)
            elevation = path_val(trackpoint, 'AltitudeMeters', np.NaN)
            distance = path_val(trackpoint, 'DistanceMeters', np.NaN) # not always specified
            hr = path_val(trackpoint, 'HeartRateBpm|Value', np.NaN) # not always specified
            cadence = path_val(trackpoint, 'Cadence', np.NaN) # not always specified
            speed = path_val(trackpoint, 'Speed', np.NaN) # not always specified
            power = path_val(trackpoint, 'Watts', np.NaN) # not always specified
        except Exception as e:
            raise e.__class__(f'Error processing {raw_time}: {str(e)}')
                   
        row = dict(dtime=raw_time, latt=latt, long=long,
                   elevation=elevation, distance=distance, hr=hr, cadence=cadence,
                   speed=speed, power=power
                   )
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df['dtime'] = df['dtime'].dt.tz_convert(pytz.timezone("Asia/Jakarta")).dt.tz_localize(None)
    return df


def gpx_to_csv(path):
    with open(path, 'r') as f:
        doc = f.read().strip()
    doc = minidom.parseString(doc)
    
    sport = path_val(doc, 'trk|type').lower()
    title = path_val(doc, 'trk|name', '')
    meta = dict(sport=sport, title=title)
    
    trackpoints = doc.getElementsByTagName('trkpt')
    rows = []
    for trackpoint in trackpoints:
        raw_time = pd.Timestamp(path_val(trackpoint, 'time'))
        try:
            latt = trackpoint.attributes['lat'].value
            long = trackpoint.attributes['lon'].value
            elevation = path_val(trackpoint, 'ele', np.NaN)
            cadence = path_val(trackpoint, 'extensions|gpxtpx:cad', np.NaN) # not always specified
            power = path_val(trackpoint, 'extensions|power', np.NaN) # not always specified
        except Exception as e:
            raise e.__class__(f'Error processing {raw_time}: {str(e)}')
                   
        row = dict(dtime=raw_time, latt=latt, long=long,
                   elevation=elevation, distance=np.NaN, hr=np.NaN, cadence=cadence,
                   speed=np.NaN, power=power
                   )
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df['dtime'] = df['dtime'].dt.tz_convert(pytz.timezone("Asia/Jakarta")).dt.tz_localize(None)
    return df


def fit_to_csv(path):
    fitfile = FitFile(path)
    has_power = False
    meta = {}
    
    rows = []
    for record in fitfile.get_messages('record'):
        data =  record.get_values()
        raw_time = pd.Timestamp(data['timestamp'])
        latt = data.get('position_lat', np.NaN)
        long = data.get('position_long', np.NaN)
        elevation = data.get('altitude', np.NaN)
        distance = data.get('distance', np.NaN)
        hr = data.get('heart_rate', np.NaN)
        cadence = data.get('cadence', np.NaN)
        speed = data.get('speed', np.NaN)
        power = data.get('power', np.NaN)
        
        if not has_power and not pd.isnull(power):
            has_power = True
            
        row = dict(dtime=raw_time, latt=latt, long=long,
                   elevation=elevation, distance=distance, hr=hr, cadence=cadence,
                   speed=speed, power=power
                   )
        rows.append(row)
        
    df = pd.DataFrame(rows)
    # Time is naive UTC. Convert to WIB
    df['dtime'] = df['dtime'] + pd.Timedelta(hours=7)
    return df


def calc_max_powers(filename, df):
    if not len(df):
        return {}
    
    dtime = df.iloc[0]['dtime']
    
    periods = []
    periods += list(range(1, 30, 1))
    periods += list(range(30, 60, 5))
    periods += list(range(60, 120, 10))
    periods += list(range(120, 300, 30))
    periods += list(range(300, 1200, 60))
    periods += list(range(1200, 7200, 300))
    periods += list(range(7200, 12*3600+600, 600))
    
    df = df[['dtime', 'power']].dropna()
    df['power'] = df['power'].astype('float')
    if not len(df):
        return {}
    
    MIN_POWER = 20
    MAX_POWER = 3000
    df = df[ (df['power'] >= MIN_POWER) & (df['power'] <= MAX_POWER)]
    if not len(df):
        return {}
    
    result = {'dtime': dtime, 'filename': filename}
    for p in periods:
        result[str(p)] = df['power'].rolling(p).mean().max()
    return result
    

def calc_power_curve(src_dir, from_date=None, to_date=None, max_hr=None):
    src_pattern = os.path.join(src_dir, '20*.csv')
    files = glob.glob(src_pattern)
    curve_df = None
    MIN_POWER = 20
    MAX_POWER = 3000
    
    if to_date:
        to_date = to_date.replace(hour=23, minute=23, second=23)
    
    for file in sorted(files):
        filepart = os.path.split(file)[1].split('.')[0]
        dtime = datetime.datetime.strptime(filepart, '%Y-%m-%d_%H-%M-%S')
        if from_date is not None and dtime < from_date:
            continue
        if to_date is not None and dtime > to_date:
            continue
        df = pd.read_csv(file, parse_dates=['dtime'])
        df = df[['dtime', 'power', 'hr']].dropna()
        df = df[ (df['power'] >= MIN_POWER) & (df['power'] <= MAX_POWER)]
        if max_hr is not None:
            df = df[ df['hr'] <= max_hr ]
        if len(df) == 0:
            continue
        
        power = calc_max_powers(filepart, df)
        if not len(power):
            continue

        if curve_df is None:
            curve_df = pd.DataFrame([power]).set_index('dtime')
        else:
            del power['dtime']
            curve_df.loc[dtime] = power

        curve_df = curve_df.sort_values('dtime')                
    
    return curve_df
        
    
def scan_update(src_dir, out_dir):
    done_path = os.path.join(out_dir, 'done-files.json')
    if os.path.exists(done_path):
        with open(done_path, 'r') as f:
            done_files = json.load(f)
        done_files = [f for f in done_files if 'csv' not in f]
        done_files = set(done_files)
    else:
        done_files = set([])

    curve_path = os.path.join(out_dir, 'power-curve.csv')
    if os.path.exists(curve_path):
        curve_df = pd.read_csv(curve_path, parse_dates=['dtime'])
        curve_df = curve_df.sort_values('dtime').set_index('dtime')
    else:
        curve_df = None

    pats = [('*.tcx', tcx_to_csv), ('*.fit', fit_to_csv), ('*.gpx', gpx_to_csv)]
    
    for pat, convert_func in pats:
        for file in glob.glob(os.path.join(src_dir, pat)):
            filename = os.path.split(file)[1]
            if filename in done_files:
                continue
            
            print(f'Processing {file}..')
            df = convert_func(file)
            
            dtime = df.loc[0, 'dtime']
            csv_filename = dtime.strftime('%Y-%m-%d_%H-%M-%S.csv')
            csv_filename = os.path.join(out_dir, csv_filename)
            df.to_csv(csv_filename, index=False)

            power = calc_max_powers(filename, df)
            if len(power):
                if curve_df is None:
                    curve_df = pd.DataFrame([power]).set_index('dtime')
                else:
                    del power['dtime']
                    curve_df.loc[dtime] = power

                curve_df = curve_df.sort_values('dtime')                
                curve_df.to_csv(curve_path)
            
            done_files.add(filename)
            with open(done_path, 'wt') as f:
                json.dump(list(done_files), f, indent='  ')
            

    
if __name__ == '__main__':
    #scan_update('/home/bennylp/Desktop/project/zwift-planner/activities/raw/garmin-fix', 'activities/rides')
    #calc_power_curve('/home/bennylp/Desktop/project/zwift-planner/activities/rides', max_hr=128,
    #                 from_date=pd.Timestamp('2020-06-08'), to_date=pd.Timestamp('2020-06-08'))
    pass
