from collections import OrderedDict
import datetime
from fitparse import FitFile, FitParseError
from geopy import distance
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz
import re
from sklearn.linear_model import LinearRegression
import sys
from xml.dom import minidom
from zwift import Client


def xml_get_text(element):
    rc = []
    for node in element.childNodes:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

    
def xml_path_val(element, tag_names, default=None):
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
    
    return xml_get_text(element)


class ZwiftTraining:
    
    DEFAULT_PROFILE_DIR = "my-ztraining-data"
    
    def __init__(self, conf_file, quiet=False):
        with open(conf_file) as f:
            self.conf = json.load(f)
            self.profile_dir = self.conf.get('dir', self.DEFAULT_PROFILE_DIR)
            
            if self.conf.get('zwift-user', None) and self.conf.get('zwift-password', None):
                self.zwift_client = Client(self.conf['zwift-user'], self.conf['zwift-password'])
                del self.conf['zwift-password']
            else:
                self.zwift_client = None
            self._zwift_profile = None
            
        if not quiet:
            print(f'Zwift user: {self.conf.get("zwift-user")}')
            print(f'Profile data directory: {self.profile_dir}')
        
    @property
    def zwift_profile_updates_csv(self):
        return os.path.join(self.profile_dir, 'zwift-profile-updates.csv')
    
    @property
    def activity_file(self):
        return os.path.join(self.profile_dir, 'activities.csv')
    
    @property
    def profile_history(self):
        if os.path.exists(self.zwift_profile_updates_csv):
            df = pd.read_csv(self.zwift_profile_updates_csv, parse_dates=['dtime'])
            return df.sort_values('dtime')
        else:
            sys.stderr.write('Error: Zwift profile not updated yet. Call update()\n')
            return None
        
    @property
    def profile_info(self):
        df = self.profile_history
        return df.iloc[-1] if df is not None else None

    @property
    def zwift_profile(self):
        if self._zwift_profile is None:
            self._zwift_profile = self.zwift_client.get_profile().profile
            # Not sure what to do if metric is not used
            assert self._zwift_profile['useMetric'], "Not sure what to change if metric is not used"
        return self._zwift_profile
    
    def plot_profile_history(self, field, interval='W-SUN', from_dtime=None, to_dtime=None):
        df = self.profile_history
        if df is None or not len(df):
            print('Error: no profile history or profile history is empty')
            return
        
        df = df.set_index('dtime')
        df['diff'] = df[field].diff().fillna(0)
        
        if from_dtime:
            df = df.loc[from_dtime:,:]
        if to_dtime:
            df = df.loc[:to_dtime,:]
        
        last_value = np.NaN
        
        def func(group):
            nonlocal last_value
            if len(group)==0:
                return pd.Series({'diff': 0, field: last_value})
            else:
                last_value = group[field].iloc[-1]
                return pd.Series({'diff': group['diff'].sum(), field: last_value})
            
        df = df.groupby(pd.Grouper(freq=interval, label='left')).apply(func)
        title = field.replace('_', ' ').title()

        if len(df) > 10:
            width = 0.8
        elif len(df) > 5:
            width = 3
        else:
            width = 3
            
        fig, ax = plt.subplots(1, 1, figsize=(15,6))

        ax.set_title(title)
        ax.bar(df.index, df['diff'], color='darkgreen', width=width, alpha=0.5, zorder=5)
        ax.set_ylabel(f'{title}')
        handles = [plt.Rectangle((0,0),1,1, color='darkgreen', alpha=0.5)]
        
        ax2 = ax.twinx()
        handles.extend( ax2.plot(df.index, df[field], color='C1', alpha=1, zorder=10) )
        ax2.set_ylabel(f'Accumulation')
        
        ax.set_xticks(df.index)
        ax.set_xticklabels(df.index.strftime('%y-%m-%d'))
        if len(df) >= 10:
            plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
        
        ax.grid()
        ax.legend(handles, [f'{title}', f'Accummulation'])
        plt.show()
        
    def get_activities(self, from_dtime=None, to_dtime=None, sport=None):
        df = pd.read_csv(self.activity_file, parse_dates=['dtime'])
        df['title'] = df['title'].fillna('')
        df['duration'] = pd.to_timedelta(df['duration'])
        df['mov_duration'] = pd.to_timedelta(df['mov_duration'])
        
        if sport:
            df = df[ df['sport']==sport ]
        if from_dtime is not None:
            df = df[ df['dtime'] >= pd.Timestamp(from_dtime) ]
        if to_dtime is not None:
            to_dtime = pd.Timestamp(to_dtime)
            if to_dtime.hour == 0:
                to_dtime = to_dtime.replace(hour=23, minute=59, second=59)
            df = df[ df['dtime'] <= to_dtime ]
            
        return df

    def get_activity_data(self, dtime=None, src_file=None):
        assert dtime or src_file, "Either dtime and/or src_file must be specified"

        df = self.get_activities()
        if dtime:
            df = df[ df['dtime']==dtime ]
        if src_file:
            df = df[ df['src_file']==src_file ]
        
        if not len(df):
            sys.stderr.write('Error: no matching activity found\n')
            return None

        dtime = df['dtime'].iloc[0]
        activities_dir = os.path.join(self.profile_dir, 'activities')
        csv_filename = dtime.strftime('%Y-%m-%d_%H-%M-%S.csv')
        csv_filename = os.path.join(activities_dir, csv_filename)

        return pd.read_csv(csv_filename, parse_dates=['dtime'])
            
    def activity_exists(self, dtime=None, src_file=None, tolerance=90):
        if os.path.exists(self.activity_file):
            df = pd.read_csv(self.activity_file, parse_dates=['dtime'])
            if dtime is not None:
                df['end_time'] = df['dtime'] + pd.to_timedelta(df['duration'])
                df['dtime'] = df['dtime'] - pd.Timedelta(seconds=tolerance)
                df['end_time'] = df['end_time'] + pd.Timedelta(seconds=tolerance)
                found = df[(df['dtime'] <= dtime) & (df['end_time'] >= dtime)]
            elif src_file is not None:
                # Only need the filename, not the full path
                assert '/' not in src_file and '\\' not in src_file
                found = df[ df['src_file'].str.lower()==src_file.lower() ]
            else:
                assert False, "Either dtime or src_file must be specified"
            return len(found) > 0
        else:
            return False

    def plot_activity(self, dtime=None, src_file=None, x='mov_duration'):
        assert x in ['distance', 'mov_duration', 'duration', 'dtime']
        
        df = self.get_activity_data(dtime=dtime, src_file=src_file)
        df['mov_duration'] = pd.to_timedelta(df['mov_duration'], unit='s')
        df['duration'] = pd.to_timedelta(df['duration'], unit='s')
        
        cols = ['speed', 'elevation', 'hr', 'power', 'cadence', 'temp']
        cols = [col for col in cols if not pd.isnull(df[col].iloc[0])]
        
        fig, axs = plt.subplots(nrows=len(cols), ncols=1, figsize=(15,3*len(cols)))
        for i_r, col in enumerate(cols):
            ax = axs[i_r]
            df.plot(x=x, y=col, ax=ax, color=f'C{i_r}', zorder=10)
            ax.axhline(df[col].mean(), color=f'C{i_r}', linestyle='--', zorder=1)
            ax.set_ylabel(col)
            ax.set_title(f'{col} avg: {df[col].mean():.1f}, max: {df[col].max()}')
            ax.grid()
            
        fig.tight_layout()
        plt.show()

    def plot_activities(self, field, interval='W-SUN', sport=None, from_dtime=None, to_dtime=None):
        df = self.get_activities(from_dtime=from_dtime, to_dtime=to_dtime, sport=sport)
        if df is None or not len(df):
            print('Error: no activities found')
            return
        
        df = df.set_index('dtime')
        df = df.groupby(pd.Grouper(freq=interval, closed='left', label='left')).agg({field: 'sum'})
        
        fig, ax = plt.subplots(1, 1, figsize=(15,6))
        
        title = field.replace('_', ' ').title()
        if field in ['duration', 'mov_duration']:
            df[field] = df[field].dt.total_seconds() / 3600
            title += ' (Hours)'
        
        ax.set_title(title)
        x = pd.to_datetime(df.index)
        if len(x) > 10:
            width = 0.8
        elif len(x) > 5:
            width = 3
        else:
            width = 3
            
        ax.bar(x, df[field], label=title, width=width, align='center', color='darkgreen', alpha=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(x.strftime('%Y-%m-%d'))
        if len(x) >= 10:
            plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
            
        ax.set_ylabel(title)
        ax.grid()
        ax.legend()
        plt.show()
        
    def delete_activity(self, dtime=None, src_file=None, dry_run=False, quiet=False):
        assert dtime or src_file, "Either dtime and/or src_file must be specified"
        
        activities = self.get_activities()
        df = activities.copy()
        if dtime:
            df = df[ df['dtime']==dtime ]
        if src_file:
            df = df[ df['src_file']==src_file ]
        
        if not len(df):
            sys.stderr.write('Error: no matching activity found\n')
            return False
        
        dtime = df['dtime'].iloc[0]
        activities_dir = os.path.join(self.profile_dir, 'activities')
        csv_filename = dtime.strftime('%Y-%m-%d_%H-%M-%S.csv')
        csv_filename = os.path.join(activities_dir, csv_filename)

        if not os.path.exists(csv_filename):
            sys.stderr.write(f'Warning: activity file {csv_filename} not found\n')
        else:
            if not quiet:
                print(f'Deleting {csv_filename}')
            if not dry_run:
                os.remove(csv_filename)

        activities = activities.drop(index=df.index)
        activities = activities.sort_values('dtime')
        if not dry_run:
            activities.to_csv(self.activity_file, index=False)
            
        return df

    def import_files(self, dir, max=None, from_dtime=None, to_dtime=None, 
                     overwrite=False, quiet=False):
        files = glob.glob(os.path.join(dir, '*'))
        updates = []
        
        if from_dtime:
            from_dtime = pd.Timestamp(from_dtime)
        if to_dtime:
            to_dtime = pd.Timestamp(to_dtime)
            if to_dtime.hour==0 and to_dtime.minute==0:
                to_dtime = to_dtime.replace(hour=23, minute=59, second=59)
                
        if not quiet:
            print(f'Found {len(files)} files in {dir}')
            
        for file in files:
            filename = os.path.split(file)[1]
            extension = filename.split('.')[-1].lower()
            
            if extension not in ['tcx', 'gpx', 'fit']:
                continue
            
            if self.activity_exists(src_file=filename) and not overwrite:
                if not quiet:
                    print(f'Skipping {filename} (already processed).. ')
                    pass
                continue
            
            df, meta = ZwiftTraining.parse_file(file)
            
            if from_dtime and meta['dtime'] < from_dtime:
                continue
            if to_dtime and meta['dtime'] > to_dtime:
                continue

            if not quiet:
                print(f'Importing {filename}..')
                
            self.save_activity(df, meta, overwrite=overwrite, quiet=quiet)
            
            updates.append(file)
            if max and len(updates) >= max:
                break
            
        return len(updates)

    def import_activity_file(self, path, sport=None, overwrite=False, quiet=False):
        """
        Import activity data and metadata from a TCX/GPX/FILE file.
        
        Parameters:
        - path:       Path to file to import
        - sport:      Give a hint about the sport of this activity.
        - overwrite:  False (the default) means do not save the activity if previous
                      activity with the same filename has been imported.
        - quiet:      Do not print messages if True
        """
        df, meta = ZwiftTraining.parse_file(path)
        if not meta['sport']:
            meta['sport'] = sport
        self.save_activity(df, meta, overwrite=overwrite, quiet=quiet)
        
    def save_activity(self, df, meta, overwrite=False, quiet=False):
        activities_dir = os.path.join(self.profile_dir, 'activities')
        if not os.path.exists(activities_dir):
            os.makedirs(activities_dir)
        
        # Save to CSV file
        dtime = meta['dtime']
        csv_filename = dtime.strftime('%Y-%m-%d_%H-%M-%S.csv')
        csv_filename = os.path.join(activities_dir, csv_filename)
        df.to_csv(csv_filename, index=False)

        # Update activities.csv
        if os.path.exists(self.activity_file):
            df = pd.read_csv(self.activity_file, parse_dates=['dtime'])
            existing = df[ df['src_file'] == meta['src_file'] ]
            if len(existing):
                if not overwrite:
                    if not quiet:
                        print('Row already exists')
                else:
                    if not quiet:
                        print(f'Overwriting {meta["src_file"]} ({meta["dtime"]})')
                    df = df.drop(index=existing.index)
                    df = df.append(meta, ignore_index=True)
            else:
                df = df.append(meta, ignore_index=True)
        else:
            df = pd.DataFrame([meta])
            
        df = df.sort_values('dtime')
        df.to_csv(self.activity_file, index=False)
        
    def zwift_update(self, start=0, max=0, batch=10, from_dtime=None, to_dtime=None, 
                     profile=True, overwrite=False, quiet=False):
        """
        Update local profile and statistics and optionally scan and update new activities 
        from the online Zwift account.
        
        Parameters:
        - start:      Start number of activity index to import (zero is the latest) 
        - max:        Maximum number of activities to scan.
        - batch:      How many activities to scan from Zwift website for each loop
        - from_dtime: Only import activities from this datetime. We still have to scan
                      the activities one by one starting from the latest activity,
                      so the start and max parameters are still used.
        - to_dtime:   Only import activities older than this datetime.
        - profile:    True to check for profile updates.
        - overwrite:  True to force overwriting already saved activities. This is only
                      usable if previous import was corrupt.
        - quiet:      True to silence the update.
        
        Returns:
          Number of updates performed
        """
        n_updates = 0
        
        if not os.path.exists(self.profile_dir):
            os.makedirs(self.profile_dir)
            
        if profile and self.zwift_client is not None:
            n_updates += self._zwift_update_profile(quiet=quiet)
            
        if max > 0:
            n_updates += self._zwift_update_activities(start=start, max=max, batch=batch,
                                                       from_dtime=from_dtime, to_dtime=to_dtime, 
                                                       overwrite=overwrite, quiet=quiet)
            
        return n_updates

    def zwift_list_activities(self, start=0, max=10, batch=10):
        player_id = self.zwift_profile['id']
        activity_client = self.zwift_client.get_activity(player_id)
        count = 0
        
        metas = []
        
        while count < max:
            activities = activity_client.list(start=start, limit=batch)
            for activity in activities:
                meta = ZwiftTraining._parse_meta_from_zwift_activity(activity, extended=True)
                del meta['src_file']
                metas.append(meta)
                count += 1
            start += batch
        
        return pd.DataFrame(metas)
        
    def _zwift_update_profile(self, quiet=False):
        if os.path.exists(self.zwift_profile_updates_csv):
            df = pd.read_csv(self.zwift_profile_updates_csv, parse_dates=['dtime'])
            latest = df.sort_values('dtime').iloc[-1]
        else:
            df = None
            latest = None
        
        pdata = self.zwift_profile
        
        cycling_level = pdata['achievementLevel'] / 100
        cycling_distance = pdata['totalDistance'] / 1000
        cycling_elevation = pdata['totalDistanceClimbed']
        cycling_xp = pdata['totalExperiencePoints']
        cycling_drops = pdata['totalGold']
        ftp = pdata['ftp']
        weight = pdata['weight'] / 1000
        
        running_level = pdata['runAchievementLevel'] / 100
        running_distance = pdata['totalRunDistance'] / 1000
        running_minutes = pdata['totalRunTimeInMinutes']
        running_xp = pdata['totalRunExperiencePoints']
        running_calories = pdata['totalRunCalories']
        
        if (latest is None or cycling_xp > latest['cycling_xp'] or cycling_drops != latest['cycling_drops'] or
            ftp != latest['ftp'] or weight != latest['weight'] or cycling_level > latest['cycling_level'] or
            running_level != latest['running_level'] or running_distance != latest['running_distance'] or
            running_xp != latest['running_xp']):
            row = OrderedDict(dtime=pd.Timestamp.now().replace(microsecond=0), 
                              cycling_level=cycling_level, cycling_distance=cycling_distance,
                              cycling_elevation=cycling_elevation, cycling_calories=np.NaN,
                              cycling_xp=cycling_xp, cycling_drops=cycling_drops, ftp=ftp, weight=weight,
                              running_level=running_level, running_distance=running_distance,
                              running_minutes=running_minutes, running_xp=running_xp,
                              running_calories=running_calories)
            if df is None:
                df = pd.DataFrame([row])
            else:
                df = df.append(row, ignore_index=True)
                
            df = df.sort_values('dtime')
            df.to_csv(self.zwift_profile_updates_csv, index=False)
            
            if not quiet:
                print('Zwift local profile updated')
                
            return True
        else:
            if not quiet:
                print(f'Zwift local profile is up to date (last update: {latest["dtime"]})')
            return False
    
    def get_cycling_level_xp(self, level):
        levels = pd.read_csv('data/levels.csv')
        return levels.loc[ levels['level'] == level, 'xp' ].iloc[0]
    
    @staticmethod
    def _parse_meta_from_zwift_activity(activity, extended=False):
        src_file = activity['id_str'] + '.zwift'
        dtime = pd.Timestamp(activity['startDate']) \
                    .tz_convert(pytz.timezone("Asia/Jakarta")) \
                    .tz_localize(None) \
                    .replace(microsecond=0)
        calories = round(activity['calories'], 1) if 'calories' in activity else np.NaN
        meta = OrderedDict(dtime=dtime, sport=activity['sport'].lower(), 
                           title=activity['name'], src_file=src_file,
                           calories=calories)
        if extended:
            meta['id'] = activity['id']
            meta['duration'] = pd.Timestamp(activity['endDate']) - pd.Timestamp(activity['startDate'])
            meta['distance'] = round(activity['distanceInMeters'] / 1000, 1)
            meta['elevation'] = round(activity['totalElevation'], 1)
            meta['power_avg'] = round(activity['avgWatts'], 1)
        
        return meta
        
    def _zwift_update_activities(self, start=0, max=20, batch=10,
                                 from_dtime=None, to_dtime=None,
                                 overwrite=False, quiet=False):
        player_id = self.zwift_profile['id']
        activity_client = self.zwift_client.get_activity(player_id)
        n_updates = 0
        
        if overwrite and not max:
            raise ValueError("'overwrite' without 'max' will retrieve too many activities")
        
        if from_dtime:
            from_dtime = pd.Timestamp(from_dtime)
            
        if to_dtime:
            to_dtime = pd.Timestamp(to_dtime)
            if to_dtime.hour==0 and to_dtime.minute==0:
                to_dtime = to_dtime.replace(hour=23, minute=59, second=59)
                
        while start < max:
            limit = start+batch
            if not quiet:
                print(f'Querying start: {start}, limit: {limit}')
            
            activities = activity_client.list(start=start, limit=limit)
            if not quiet:
                print(f'Fetched {len(activities)} activities metadata')
            
            if not activities:
                break
            
            for activity in activities:
                if start >= max:
                    break
                
                meta = ZwiftTraining._parse_meta_from_zwift_activity(activity)
                if not quiet:
                    print(f'Found activity {start}: {meta["title"]} ({meta["dtime"]}) (id: {activity["id"]})')
                    
                if self.activity_exists(src_file=meta['src_file']) and not overwrite:
                    start += 1
                    continue
                if from_dtime and meta['dtime'] < from_dtime:
                    start += 1
                    continue
                if to_dtime and meta['dtime'] > to_dtime:
                    start += 1
                    continue
                
                try:
                    df, meta = self.parse_zwift_activity(activity['id'], meta=meta, quiet=quiet)
                    self.save_activity(df, meta, overwrite=overwrite, quiet=quiet)
                    n_updates += 1
                except FitParseError as e:
                    sys.stderr.write(f'Import error: error parsing activity index: {start}, id: {activity["id"]}, datetime: {meta["dtime"]}, title: "{meta["title"]}", duration: {activity["duration"]}: FitParseError: {str(e)} \n')
                
                start += 1
            
        return n_updates

    def parse_zwift_activity(self, activity_id, meta=None, quiet=False):
        player_id = self.zwift_profile['id']
        activity_client = self.zwift_client.get_activity(player_id)
        if not meta:
            activity = activity_client.get_activity(activity_id)
            meta = ZwiftTraining._parse_meta_from_zwift_activity(activity)
        if not quiet:
            print(f'Getting activity {meta["title"]} ({meta["dtime"]})')
        records = activity_client.get_data(activity_id)
        return ZwiftTraining.parse_fit_records(records, meta) 
            
    def plot_power_curves(self, periods, min_interval=None, max_interval=None, max_hr=None, title=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
        
        def sec_to_str(sec):
            s = f'{sec//3600:02d}:{(sec%3600)//60:02d}:{sec % 60:02d}'
            s = re.sub(r'^00:', '', s)
            s = re.sub(r'^00:', '', s)
            return s
        
        power_intervals = None
        for i_period, (from_date, to_date) in enumerate(periods):
            from_date = pd.Timestamp(from_date)
            to_date = pd.Timestamp(to_date)
            
            df = self.calc_power_curve(from_date=from_date, to_date=to_date, max_hr=max_hr)
            if df is None:
                continue
            df = df.drop(columns=['7',
                                  '11', '12', '13', '14',
                                  '16', '17', '18', '19',
                                  '21', '22', '23', '24', '25',
                                  '26', '27', '28', '29',
                                  '35', '40', '55', 
                                  '70', '80', '100', '110'])
            
            if min_interval is not None:
                cols = [col for col in df.columns if int(col) >= min_interval]
                df = df[cols]
    
            if max_interval is not None:
                cols = [col for col in df.columns if int(col) <= max_interval]
                df = df[cols]
    
            power_cols = [col for col in df.columns]
            power_intervals = [int(col) for col in power_cols]
            power_interval_labels = [sec_to_str(i) for i in power_intervals]
                
            values = [df[col].max() for col in power_cols]
            label = f"{from_date.strftime('%d %b %Y')} - {to_date.strftime('%d %b %Y')}"
            ax.plot(range(len(power_cols)), values, label=label,
                    zorder=10+i_period)
        
        if power_intervals is None:
            sys.stderr.write('Error: no graph can be generated. Check the directory, or the dates\n')
            return
        
        tick_posses = dict([(1, '1s'), (5, '5s'), (10, '10s'), (30, '30s'),
                            (60, '1m'), (5*60, '5m'), (20*60, '20m'), 
                            (1*3600, '1h'), (2*3600, '2h'), (3*3600, '3h'), 
                            (4*3600, '3h'), (4*3600, '4h'), (5*3600, '5h'), 
                            (6*3600, '6h'), (7*3600, '2h'), (7*3600, '2h'), 
                            (8*3600, '8h'), (11*3600, '11h')])
        
        xticks = []
        for i_p, interval in enumerate(power_intervals):
            if interval in tick_posses:
                xticks.append( (i_p, tick_posses[interval]) )
        
        ax.set_xticks([elem[0] for elem in xticks])
        ax.set_xticklabels([elem[1] for elem in xticks])
        
        ax.set_ylabel('Power')
        ax.set_xlabel('Interval')
        ax.grid()
        ax.legend()
        if title:
            ax.set_title(title)
        plt.show()

    def calc_power_curve(self, from_date=None, to_date=None, max_hr=None):
        if from_date:
            from_date = pd.Timestamp(from_date)
        if to_date:
            to_date = pd.Timestamp(to_date)
            
        src_pattern = os.path.join(self.profile_dir, 'activities', '20*.csv')
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
            
            power = ZwiftTraining.calc_max_powers(df)
            if not len(power):
                continue
    
            if curve_df is None:
                curve_df = pd.DataFrame([power]).set_index('dtime')
            else:
                del power['dtime']
                curve_df.loc[dtime] = power
    
            curve_df = curve_df.sort_values('dtime')                
        
        return curve_df

    def _train_duration_predictor1(self, quiet=False):
        df = self.get_activities(sport='cycling')
        df = df[ (df['distance'] > 5) & (df['elevation'] > 1) & (df['power_avg'] > 5)]
        df = df.sort_values('dtime').tail(10)
        df = df.copy()
        if len(df) < 10:
            sys.stderr.write('Not enough data to make prediction\n')
            return None
        
        if not quiet:
            print(f'Training with {len(df)} datapoints from {df["dtime"].iloc[0]}')
        
        df['minutes'] = pd.to_timedelta(df['mov_duration']).dt.total_seconds() / 60
        
        X = df[['distance', 'elevation', 'power_avg']]
        y = df['minutes']
        reg = LinearRegression().fit(X, y)
        
        if not quiet:
            pred = reg.predict(X)
            mse = np.mean( (pred - y)**2 )
            me = np.mean( mse ** 0.5 )
            me_pct = np.mean( np.abs(pred-y) / y )
            print(f'Mean error: {me:.1f} minutes')
            print(f'Mean error: {me_pct:.1%}')
        
        return reg        
    
    @staticmethod
    def _convert_to_segments(df, segment_length_meters=100):
        df['segment_num'] = (df['distance'] * 1000 / segment_length_meters).astype('int')
        agg_rules = {'distance': 'last', 'elevation': 'last', 'mov_duration': 'last'}
        if 'power' in df.columns:
            agg_rules['power'] = 'mean'
        segments = df.groupby('segment_num').agg(agg_rules)
        if 'power' not in df.columns:
            segments['power_avg'] = np.NaN
        segments.columns = ['last_distance', 'last_elevation', 'last_mov_duration', 'power_avg']
        segments['distance'] = segments['last_distance'].diff()
        segments['elevation'] = segments['last_elevation'].diff()
        segments['seconds'] = segments['last_mov_duration'].diff()
        segments = segments.iloc[1:]
        return segments[['distance', 'elevation', 'seconds', 'power_avg']]
        
    def _train_duration_predictor2(self, activity_dtime, quiet=False):
        """
        Train linear regression model to predict duration using the specified activity
        """
        df = self.get_activity_data(dtime=activity_dtime)
        segments = ZwiftTraining._convert_to_segments(df)

        X = segments[['distance', 'elevation', 'power_avg']]
        y = segments['seconds']
        model = LinearRegression().fit(X, y)
        
        if not quiet:
            pred = model.predict(X)
            y_dur1 = df['mov_duration'].iloc[-1]
            y_dur2 = segments['seconds'].sum()
            pred_dur = pred.sum()
            diff = abs(pred_dur - y_dur2)
            err = diff / y_dur2
            print(f'Regression: duration: {y_dur2}, prediction: {pred_dur}. error: {diff} secs ({err:.1%})')
        
        return model

    def _predict_duration2(self, model, route_file, avg_power, quiet=False):
        r_df, r_meta = ZwiftTraining.parse_file(route_file)
        if not quiet:
            print(f"Distance: {r_meta['distance']:.1f} km, elevation: {r_meta['elevation']:.0f}m")
        segments = ZwiftTraining._convert_to_segments(r_df)
        if not quiet:
            print(f"Distance: {segments['distance'].sum():.1f} km, elevation: {segments['elevation'].sum():.0f}m")
        segments['power_avg'] = avg_power
        X = segments[['distance', 'elevation', 'power_avg']]
        pred = model.predict(X)
        dur = pd.Timedelta(seconds=pred.sum())
        if not quiet:
            print(f"Predicted duration: {dur}")
        return dur
    
    def _load_routes(self, sport=None):
        route = pd.read_csv('data/routes.csv')
        route['total distance'] = route['distance'] + route['lead-in']
        route['done'] = 0
        route['restriction'] = route['restriction'].fillna('')
        route['badge'] = route['badge'].fillna(0)
        route['elevation'] = route['elevation'].fillna(0)
        
        route = route[ ~route['restriction'].str.contains('Event') ]
        
        inventories_file = os.path.join(self.profile_dir, 'inventories.csv')
        if os.path.exists(inventories_file):
            inventories = pd.read_csv(inventories_file)
            inventories = inventories[ inventories['type']=='route' ]
            for idx, row in inventories.iterrows():
                route.loc[ route['name']==row['name'], 'done' ] = 1
                
        if sport == 'cycling':
            route = route[ ~route['restriction'].str.contains('Run') ]
        elif sport:
            assert False, f"Unknown sport: {sport}"

        route = route.set_index(['world', 'route'])
        route = route.drop(columns=['name'])
            
        return route
    
    @staticmethod
    def tss_to_avg_watt(ftp, tss, duration):
        # TSS = hour * avg_watt**2 / ftp**2 * 100
        # avg_watt = (tss / hour * ftp**2 / 100) ** 0.5
        hour = pd.Timedelta(duration).total_seconds() / 3600
        return (tss * (ftp**2) / hour / 100) ** 0.5
        
    @staticmethod
    def avg_watt_to_tss(ftp, avg_watt, duration):
        #np = avg_watt
        #IF = np / ftp
        #tss = (pd.Timedelta(duration).total_seconds()  * np * IF) / (ftp * 3600) * 100
        tss = pd.Timedelta(duration).total_seconds() / 3600 * (avg_watt ** 2) / (ftp ** 2) * 100
        return tss

    def best_cycling_route(self, max_duration, avg_watt=None, tss=None, ftp=None, min_duration=None, 
                           kind=None, done=None, quiet=False):
        assert (avg_watt is not None) or (tss is not None and ftp is not None)
        assert kind is None or kind in ['ride', 'interval', 'workout']
        max_minutes = pd.Timedelta(max_duration).total_seconds() / 60
        
        if avg_watt is None:
            avg_watt = ZwiftTraining.tss_to_avg_watt(ftp, tss, max_duration)
            if not quiet:
                print(f'Avg watt: {avg_watt:.0f}')
        
        regressor = self._train_duration_predictor1(quiet=quiet)
        df = self._load_routes(sport='cycling')
        
        # In "ride" mode, Zwift awards 20 xp per km
        XP_PER_KM = 20
        
        # In "interval", reward is 12 XP per minute. But there will be other
        # blocks such as warmups and free rides so usually we won't get the full XPs
        XP_PER_MIN_ITV = 12 * 0.95
        
        # In "workout" blocks, reward is 10 XP per minute for workout blocks and
        # 5-6 XP per minute for warmup/rampup/cooldown/rampdown blocks.
        XP_PER_MIN_WRK = 10 * 0.8 + 5.5 * 0.2
        
        df['power_avg'] = avg_watt
        
        # Set for better display
        df[['done', 'elevation', 'badge']] = df[['done', 'elevation', 'badge']].astype(int)
        
        # Clear badge if route is done
        df.loc[df['done'] != 0, 'badge'] = 0
        
        # Predict ride duration using linear regressor
        df['pred minutes'] = regressor.predict(df[['total distance', 'elevation', 'power_avg']]).round(1)
        df['pred avg speed'] = (df['total distance'] / (df['pred minutes'] / 60.0)).round(1)
        
        # Predict the XPs received for ride and interval activities, assuming that we finish
        # the full route.
        df['pred xp (ride)'] = (np.floor(df['total distance']) * XP_PER_KM + df['badge']).astype(int)
        df['pred xp (interval)'] = ( np.floor( df['pred minutes'] ) * XP_PER_MIN_ITV + df['badge']).astype(int)
        df['pred xp (workout)'] = (np.floor(df['pred minutes']) * XP_PER_MIN_WRK + df['badge']).astype(int)
        
        # Set which is the best activity (ride or interval), unless it's forced
        if not kind:
            df['best activity'] = df[['pred xp (ride)', 'pred xp (interval)', 'pred xp (workout)']].idxmax(axis=1)
            df['best activity'] = df['best activity'].str.extract(r'\((.*)\)')
    
            # Set the best XP received if the best activity is selected
            df['best pred xp'] = df[['pred xp (ride)', 'pred xp (interval)', 'pred xp (workout)']].max(axis=1)
        else:
            df['best activity'] = kind
            df['best pred xp'] = df[f'pred xp ({kind})']
            
        df['best pred xp/minutes'] = (df['best pred xp'] / df['pred minutes']).round(1)
        
        # Filter only routes less than the specified duration
        df = df[ df['pred minutes'] <= max_minutes]
        
        # Filter only routes with at least the specified duration
        if min_duration:
            min_minutes = pd.Timedelta(min_duration).total_seconds() / 60
            df = df[ df['pred minutes'] > min_minutes]
            
        # Display result
        columns = ['done', 'total distance', 'distance', 'lead-in', 'elevation', 'badge', 
                   'best activity', 'best pred xp', 'power_avg',  'pred avg speed', 
                   'pred minutes', 'best pred xp/minutes']

        df = df[columns]
        df['power_avg'] = df['power_avg'].astype(int)

        #df = df.sort_values(['best pred xp/minutes'], ascending=False)
        df = df.sort_values(['best pred xp'], ascending=False)
        if done is not None:
            df = df[ df['done']==done ]
            
        return df

    def load_routes(self):
        route = pd.read_csv('data/routes.csv').set_index('route')
        route['done'] = False
        
        inventory = pd.read_csv(os.path.join(self.profile_dir, 'inventories.csv'))
        inventory = inventory[ inventory['type'] == 'route' ]
        
        for idx, row in inventory.iterrows():
            route.loc[row['name'], 'done'] = True
            
        return route

    @staticmethod
    def calc_max_powers(df):
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
        
        result = {'dtime': dtime}
        for p in periods:
            result[str(p)] = round(df['power'].rolling(p).mean().max(), 1)
        return result
    
    @staticmethod
    def _process_activity(df, meta, min_kph=3, copy=True):
        if copy:
            df = df.copy()
        
        MAX_ELE = 9000
        MAX_HR = 250
        MAX_POWER = 2500
        MAX_CADENCE = 210
        MAX_SPEED = 100
        MAX_TEMP = 55
        
        df['latt'] = df['latt'].astype('float')
        df['long'] = df['long'].astype('float')
        df['elevation'] = df['elevation'].astype('float').clip(upper=MAX_ELE)
        df['distance'] = df['distance'].astype('float')
        df['hr'] = df['hr'].astype('float').clip(upper=MAX_HR)
        df['power'] = df['power'].astype('float').clip(upper=MAX_POWER)
        df['cadence'] = df['cadence'].astype('float').clip(upper=MAX_CADENCE)
        df['speed'] = df['speed'].astype('float').clip(upper=MAX_SPEED)
        df['temp'] = df['temp'].astype('float').clip(upper=MAX_TEMP)
        
        mpos = list(df.columns).index('distance')
        if pd.isnull(df['distance'].iloc[0]):
            if pd.isnull(df['latt'].iloc[0]):
                assert False, "Unable to calculate distance because GPS coordinates are null"
            df['prev-latt'] = df['latt'].shift()
            df['prev-long'] = df['long'].shift()
            func = lambda row: ZwiftTraining.measure_distance(row['prev-latt'], row['prev-long'], 
                                                              row['latt'], row['long'])
            df.insert(mpos, 'movement', df.apply(func, axis=1).fillna(0))
            df['distance'] = df['movement'].cumsum() / 1000
            df = df.drop(columns=['prev-latt', 'prev-long'])
        else:
            df.insert(mpos, 'movement', (df['distance'] * 1000).diff())

        df.loc[0, 'movement'] = df.loc[0, 'distance'] * 1000
        df['movement'] = df['movement'].round(3)
        
        max_movement = MAX_SPEED * 1000 / 3600
        df['movement'] = df['movement'].clip(upper=max_movement)
        
        # absolute duration
        start_time = df.iloc[0]['dtime']
        df.insert(1, 'duration', (df['dtime'] - start_time).dt.total_seconds())
        
        # recalculate speed.
        tick_elapsed = df['duration'].diff().fillna(1)
        df['speed'] = df['movement'] * 3600 / 1000 / tick_elapsed
        df['speed'] = df['speed'].replace(np.inf, np.NaN)
        df.loc[ df['speed'] > 100, 'speed'] = 100
        
        # Smoothen speed, power, hr
        df['speed'] = df['speed'].rolling(3, min_periods=1).mean()
        df['power'] = df['power'].rolling(2, min_periods=1).mean()
        df['hr'] = df['hr'].rolling(2, min_periods=1).mean()
        df['cadence'] = df['cadence'].rolling(2, min_periods=1).mean()
        df['temp'] = df['temp'].rolling(2, min_periods=1).mean()
        
        # remove non-movement
        min_movement = min_kph*1000 / 3600
        df = df[ df['movement'] >= min_movement]
        
        # moving time
        df.insert(2, 'mov_duration', range(0, len(df)))
        
        sports = {
            'biking': 'cycling',
            'cycling': 'cycling',
            'cycling_transportation': 'cycling',
            'cycling_sport': 'cycling',
            'ride': 'cycling',
            'virtualride': 'cycling',
            'virtualrun': 'running',
            'run': 'running',
            'running': 'running',
            'other': 'other',
        }
        meta['sport'] = sports[ meta['sport'] ]
        
        if len(df):
            meta['distance'] = np.round(df["distance"].iloc[-1], 3)
            meta['duration'] = pd.Timedelta(seconds=df.iloc[-1]['duration'])
            meta['mov_duration'] = pd.Timedelta(seconds=df.iloc[-1]['mov_duration'])
            if False:
                climb = df['elevation'].diff()
                meta['elevation'] = np.round(climb[ climb > 0.05 ].sum(), 1)
            else:
                climb = df['elevation'].rolling(6).mean().diff()
                meta['elevation'] = np.round(climb[ climb > 0 ].sum(), 1)
            meta['speed_avg'] = np.round(meta['distance'] / (meta['mov_duration'].total_seconds() / 3600), 1)
            meta['speed_max'] = np.round(df["speed"].max(), 1)
            meta['hr_avg'] = np.round(df["hr"].mean(), 2)
            meta['hr_max'] = df['hr'].max()
            meta['power_avg'] = np.round(df["power"].mean(), 2)
            meta['power_max'] = df["power"].max()
            cadence = df["cadence"]
            cadence = cadence[ cadence > 0 ]
            meta['cadence_avg'] = np.round(cadence.mean(), 2)
            meta['cadence_max'] = np.ceil(df["cadence"].max())
            meta['temp_avg'] = round(df["temp"].mean(), 1)
            meta['temp_max'] = df["temp"].max()
        else:
            meta['distance'] = np.NaN
            meta['duration'] = np.NaN
            meta['mov_duration'] = np.NaN
            meta['elevation'] = np.NaN
            meta['speed_avg'] = np.NaN
            meta['speed_max'] = np.NaN
            meta['hr_avg'] = np.NaN
            meta['hr_max'] = np.NaN
            meta['power_avg'] = np.NaN
            meta['power_max'] = np.NaN
            meta['cadence_avg'] = np.NaN
            meta['cadence_max'] = np.NaN
            meta['temp_avg'] = np.NaN
            meta['temp_max'] = np.NaN
        
        # Move calories to end of dictionary
        if 'calories' in meta:
            calories = meta['calories']
            del meta['calories']
        else:
            calories = np.NaN
        meta['calories'] = calories
        
        # Round some values
        df = df.copy()
        df['elevation'] = df['elevation'].astype('float').round(2)
        df['distance'] = df['distance'].astype('float').round(3)
        df['speed'] = df['speed'].astype('float').round(2)
            
        return df, meta
            
    @staticmethod
    def measure_distance(lat1, lon1, lat2, lon2):
        """
        Measure distance between two coordinates, in meters.
        """
        if pd.isnull(lat1) or pd.isnull(lat2):
            return np.NaN
        #if lat1 < -90 or lat1 > 90:
        #    # just in case coordinates in .fit file are not converted
        #    lat1 *= 180/(2**31)
        #    lon1 *= 180/(2**31)
        #    lat2 *= 180/(2**31)
        #    lon2 *= 180/(2**31)
        return distance.distance((lat1, lon1), (lat2, lon2)).m
    
    @staticmethod
    def parse_file(file):
        if file[-4:].lower() == '.tcx':
            return ZwiftTraining.parse_tcx_file(file)
        elif file[-4:].lower() == '.fit':
            return ZwiftTraining.parse_fit_file(file)
        elif file[-4:].lower() == '.gpx':
            return ZwiftTraining.parse_gpx_file( file)
        else:
            assert False, f"Unsupported file extension {file[-3:]}"
        
    @staticmethod    
    def parse_tcx_file(path):
        """
        Convert TCX file to CSV
        """
        with open(path, 'r') as f:
            doc = f.read().strip()
        doc = minidom.parseString(doc)
        
        sport = doc.getElementsByTagName('Activities')[0] \
                   .getElementsByTagName('Activity')[0] \
                   .attributes['Sport'].value \
                   .lower()
        title = xml_path_val(doc, 'Activities|Activity|Notes', '')
        
        calories = 0
        calories_nodes = doc.getElementsByTagName('Calories')
        for node in calories_nodes:
            s = xml_get_text(node)
            if s.strip():
                calories += float(s.strip())
        if not calories:
            calories = np.NaN
        
        trackpoints = doc.getElementsByTagName('Trackpoint')
        rows = []
        for trackpoint in trackpoints:
            raw_time = pd.Timestamp(xml_path_val(trackpoint, 'Time'))
            try:
                latt = xml_path_val(trackpoint, 'LatitudeDegrees', np.NaN)
                long = xml_path_val(trackpoint, 'LongitudeDegrees', np.NaN)
                elevation = xml_path_val(trackpoint, 'AltitudeMeters', np.NaN)
                distance = xml_path_val(trackpoint, 'DistanceMeters', np.NaN) # not always specified
                hr = xml_path_val(trackpoint, 'HeartRateBpm|Value', np.NaN) # not always specified
                cadence = xml_path_val(trackpoint, 'Cadence', np.NaN) # not always specified
                speed = xml_path_val(trackpoint, 'Speed', np.NaN) # not always specified
                if pd.isnull(speed):
                    speed = xml_path_val(trackpoint, 'ns3:Speed', np.NaN) # not always specified
                power = xml_path_val(trackpoint, 'Watts', np.NaN) # not always specified
                if pd.isnull(power):
                    power = xml_path_val(trackpoint, 'ns3:Watts', np.NaN) # not always specified
            except Exception as e:
                raise e.__class__(f'Error processing {raw_time}: {str(e)}')
                       
            row = OrderedDict(dtime=raw_time, latt=latt, long=long,
                              elevation=elevation, distance=distance, hr=hr, cadence=cadence,
                              speed=speed, power=power, temp=np.NaN
                              )
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df['dtime'] = df['dtime'].dt.tz_convert(pytz.timezone("Asia/Jakarta")).dt.tz_localize(None)
        df['distance'] = df['distance'].astype('float') / 1000
        meta = OrderedDict(dtime=df['dtime'].iloc[0], sport=sport, title=title, 
                           src_file=os.path.split(path)[-1], calories=calories)
        return ZwiftTraining._process_activity(df, meta, copy=False)
    
    @staticmethod
    def parse_gpx_file(path):
        """
        Convert GPX file to CSV
        """
        with open(path, 'r') as f:
            doc = f.read().strip()
        doc = minidom.parseString(doc)
        
        title = xml_path_val(doc, 'trk|name', '')
        if 'ride' in title.lower():
            sport = 'biking'
        elif 'run' in title.lower():
            sport = 'running'
        else:
            sport = xml_path_val(doc, 'trk|type').lower()
        
        trackpoints = doc.getElementsByTagName('trkpt')
        rows = []
        for trackpoint in trackpoints:
            raw_time = pd.Timestamp(xml_path_val(trackpoint, 'time'))
            try:
                latt = trackpoint.attributes['lat'].value
                long = trackpoint.attributes['lon'].value
                elevation = xml_path_val(trackpoint, 'ele', np.NaN)
                cadence = xml_path_val(trackpoint, 'extensions|gpxtpx:cad', np.NaN) # not always specified
                power = xml_path_val(trackpoint, 'extensions|power', np.NaN) # not always specified
                hr = xml_path_val(trackpoint, 'extensions|gpxtpx:hr', np.NaN) # not always specified
                temp = xml_path_val(trackpoint, 'extensions|gpxtpx:atemp', np.NaN) # not always specified
            except Exception as e:
                raise e.__class__(f'Error processing {raw_time}: {str(e)}')
                       
            row = OrderedDict(dtime=raw_time, latt=latt, long=long,
                              elevation=elevation, distance=np.NaN, hr=hr, cadence=cadence,
                              speed=np.NaN, power=power, temp=temp
                              )
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df['dtime'] = df['dtime'].dt.tz_convert(pytz.timezone("Asia/Jakarta")).dt.tz_localize(None)
        
        meta = OrderedDict(dtime=df['dtime'].iloc[0], sport=sport, title=title, src_file=os.path.split(path)[-1])
        return ZwiftTraining._process_activity(df, meta, copy=False)
    
    @staticmethod
    def parse_fit_file(path):
        """
        Convert FIT file to CSV
        """
        fitfile = FitFile(path)
        messages = fitfile.get_messages('record')
        records = [m.get_values() for m in messages]
        meta = OrderedDict(dtime=None, sport='', title='', src_file=os.path.split(path)[-1])
        return ZwiftTraining.parse_fit_records(records, meta)

    @staticmethod
    def parse_fit_records(records, meta):
        has_power = False
        
        rows = []
        for data in records:
            raw_time = data.get('timestamp', data.get('time', None))
            assert raw_time, "Unable to get time information in fit record"
            latt = data.get('position_lat', data.get('lat', np.NaN))
            long = data.get('position_long', data.get('lng', np.NaN))
            elevation = data.get('altitude', np.NaN)
            distance = data.get('distance', np.NaN)
            hr = data.get('heart_rate', data.get('heartrate', np.NaN))
            cadence = data.get('cadence', np.NaN)
            speed = data.get('speed', np.NaN)
            power = data.get('power', np.NaN)
            temp = data.get('temperature', np.NaN) 
    
            if not has_power and not pd.isnull(power):
                has_power = True
                       
            row = OrderedDict(dtime=raw_time, latt=latt, long=long,
                              elevation=elevation, distance=distance, hr=hr, cadence=cadence,
                              speed=speed, power=power, temp=temp
                              )
            rows.append(row)
            
        df = pd.DataFrame(rows)
        # Time is naive UTC. Convert to WIB
        df['dtime'] = df['dtime'] + pd.Timedelta(hours=7)
        if not meta.get('dtime', None):
            meta['dtime'] = df['dtime'].iloc[0]
        if not meta.get('sport', None):
            meta['sport'] = ''
        if not meta.get('title', None):
            meta['title'] = ''
        if not meta.get('src_file', None):
            meta['src_file'] = ''
        
        # Some adjustments
        if len(df):
            if 'timestamp' in records[0]:
                if pd.isnull(df['latt'].iloc[0]):
                    # Garmin .fit format on trainer.
                    #This below is correct, but disabling this as we'll use general heuristic later
                    #df['distance'] /= 1000
                    pass
                else:
                    # Garmin .fit format with GPS
                    if df['latt'].max() > 90 or df['latt'].min() < -90:
                        df['latt'] *= 180/(2**31)
                        df['long'] *= 180/(2**31)
                    
                    #Strava (or possibly Zwift) doesn't need this below
                    #df['elevation'] /= 5
                    
                    # Clear distance. Some .fit files are in meters, some in km, some in cm
                    #df['distance'] /= (100 * 1000)  # orginally in cm
                    # Nope! Sometimes the GPS is messed up but the distance is good
                    # (e.g. on trainer session with GPS on. Example: 1873571076.fit)
                    #df['distance'] = np.NaN
                    
                # Some heuristic until we know the rule
                max_dist = df['distance'].max()
                if not pd.isnull(max_dist):
                    if max_dist < 1000:
                        # the distance is probably alright
                        pass
                    elif max_dist < 1000000:
                        # in meters
                        df['distance'] /= 1000
                    else:
                        # in cm
                        df['distance'] /= 100000
                    
            elif 'time' in records[0]:
                # Zwift .fit format:
                # - latt and long is correct
                # - distance is in km
                # - elevation is in m
                # - so nothing to do then!
                pass
            
        # Hack
        if not meta['sport']:
            if has_power or len(df[ df['speed'] > 20 ]) > 120:
                meta['sport'] = 'cycling'
            else:
                meta['sport'] = 'running'
            
        return ZwiftTraining._process_activity(df, meta, copy=False)

    def _zwift_update_calories(self, start=0, max=0, batch=10):
        player_id = self.zwift_profile['id']
        activity_client = self.zwift_client.get_activity(player_id)
        
        calories_updates = {}
        
        while start < max:
            limit = start+batch
            print(f'Querying start: {start}, limit: {limit}')
            
            activities = activity_client.list(start=start, limit=limit)
            print(f'Fetched {len(activities)} activities metadata')
            
            if not activities:
                break
            
            for activity in activities:
                if start >= max:
                    break
                
                meta = ZwiftTraining._parse_meta_from_zwift_activity(activity)
                calories_updates[meta['src_file']] = meta['calories']
                start += 1

        print(f'Updating {len(calories_updates)} activities')            
        df = pd.read_csv(self.activity_file, parse_dates=['dtime'])
        for src_file, cal in calories_updates.items():
            found = df[ df['src_file']==src_file ]
            if not len(found):
                print(f'Error: {src_file} not found')
                continue
            if len(found) > 1:
                print(f'Warning: found {len(found)} rows for {src_file}')
            df.loc[ found.index, 'calories' ] = cal
            print(f'Row {found.index} updated')
        
        df = df.sort_values('dtime')
        df.to_csv(self.activity_file, index=False)

    def _update_tcx_calories(self, import_dir, start=0, max=0):
        df = pd.read_csv(self.activity_file, parse_dates=['dtime'])
        
        tcx_df = df[ df['src_file'].str.contains('.tcx') ]
        calories_updates = {}
        
        tcx_df = tcx_df.sort_values('dtime', ascending=False)
        
        for idx, row in tcx_df.iterrows():
            print(f'\rProcessing activity {idx}   ', end='')
            start -= 1
            if start >= 0:
                continue
            
            path = os.path.join(import_dir, row['src_file'])
            if os.path.exists(path):
                _, meta = self.parse_tcx_file(path)
                if not pd.isnull(meta['calories']) and meta['calories']:
                    calories_updates[ row['src_file'] ] = meta['calories']
            
            if len(calories_updates) >= max:
                break
            
        print(f'Updating {len(calories_updates)} activities')            
        df = pd.read_csv(self.activity_file, parse_dates=['dtime'])
        for src_file, cal in calories_updates.items():
            found = df[ df['src_file']==src_file ]
            if not len(found):
                print(f'Error: {src_file} not found')
                continue
            if len(found) > 1:
                print(f'Warning: found {len(found)} rows for {src_file}')
            df.loc[ found.index, 'calories' ] = cal
            print(f'Row {found.index} updated')
        
        df = df.sort_values('dtime')
        df.to_csv(self.activity_file, index=False)
            