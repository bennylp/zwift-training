from collections import OrderedDict
import datetime
import glob
import json
import math
import os
import re
import sys
from xml.dom import minidom

import pytz
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from fitparse import FitFile, FitParseError
from geopy import distance
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zwift import Client


def sec_to_str(sec, full=False):
    s = f'{sec//3600:02d}:{(sec%3600)//60:02d}:{sec % 60:02d}'
    if not full:
        s = re.sub(r'^00:', '', s)
        s = re.sub(r'^00:', '', s)
    return s


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


class FTPHistory:
    MAX_VALIDITY = 3*30
    MAX_PRIOR_VALIDITY = 30
    
    def __init__(self, profile_history, default_ftp=None, max_validity=MAX_VALIDITY):
        df = profile_history[['dtime', 'ftp']].sort_values('dtime')
        df['date'] = df['dtime'].dt.date
        self.ftp_history = df[['date', 'ftp']].set_index('date')['ftp']
        self.default_ftp = default_ftp
        self.max_validity = max_validity
        self.max_prior_validity = self.MAX_PRIOR_VALIDITY
    
    def get_ftp(self, dtime):
        date = pd.Timestamp(dtime).date()
        min_date = date - pd.Timedelta(days=self.max_validity)
        max_date = date + pd.Timedelta(days=self.max_prior_validity)
        ftp = self.ftp_history.loc[min_date:max_date]
        if not len(ftp):
            return self.default_ftp
        
        while len(ftp) >= 2 and ftp.index[1] <= date:
            ftp = ftp.iloc[1:]
        while len(ftp) >= 2 and ftp.index[-2] >= date:
            ftp = ftp.iloc[:-1]
        
        assert len(ftp) <= 2
        return ftp.iloc[0]
        
        
class ZwiftTraining:
    
    DEFAULT_PROFILE_DIR = "my-ztraining-data"
    POWER_ZONES = [0.55, 0.75, 0.9, 1.05, 1.2, 1.5]
    POWER_LABELS = ['Active Recovery', 'Endurance', 'Tempo', 'Lactate Threshold', 
                    'VO2Max', 'Anaerobic', 'Neuromoscular']
    HR_ZONES = [0.6, 0.72, 0.8, 0.9 ]
    HR_LABELS = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5']
    
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

    def modify_activity(self, dtime=None, src_file=None, note=None, route=None, bike=None, wheel=None):
        if not os.path.exists(self.activity_file):
            raise RuntimeError("Activity file does not exist. Update or import some activities first")
        
        selector = None
        activities = pd.read_csv(self.activity_file, parse_dates=['dtime'])
        if dtime is not None:
            dtime = pd.Timestamp(dtime)
            dtime_selector = activities['dtime']==dtime
            if selector is None:
                selector = dtime_selector
            else:
                selector = selector & dtime_selector
        elif src_file is not None:
            # Only need the filename, not the full path
            assert '/' not in src_file and '\\' not in src_file
            file_selector = activities['src_file'].str.lower()==src_file.lower()
            if selector is None:
                selector = file_selector
            else:
                selector = selector & file_selector
                            
        if selector is None:
            raise ValueError("Either dtime or src_file must be specified")
            
        if len(activities[selector])==0:
            raise ValueError(f'Unable to find matching activity')
        elif len(activities[selector])>1:
            raise ValueError(f'Found more than one matching activities')

        activity = activities[selector].iloc[0]        
        if route:
            route_names = self._load_routes(sport=activity['sport'])['name']
            if '*' in route:
                route_names = route_names[ route_names.str.contains(route.replace('*', '')) ]
                return sorted(list(route_names))
            else:
                route_names = route_names[ route_names==route ]
                if len(route_names)==0:
                    raise ValueError(f'The specified route is not found in database')
                elif len(route_names)>1:
                    raise ValueError(f'{len(route_names)} matching routes found')
            activities.loc[selector, 'route'] = route
            
        return activities
        

    def plot_activity(self, dtime=None, src_file=None, x='mov_duration', ftp=None, max_hr=182):
        assert x in ['distance', 'mov_duration', 'duration', 'dtime']
        
        df = self.get_activity_data(dtime=dtime, src_file=src_file)
        df['mov_duration'] = pd.to_timedelta(df['mov_duration'], unit='s')
        df['duration'] = pd.to_timedelta(df['duration'], unit='s')
        
        if dtime is None:
            dtime = df['dtime'].iloc[0]
        activity = self.get_activities(from_dtime=dtime, to_dtime=dtime).iloc[0]
        print(activity['title'])
        
        cols = ['speed', 'elevation', 'hr', 'power', 'cadence', 'temp']
        cols = [col for col in cols if not pd.isnull(df[col].iloc[0])]
        
        ncols = 3
        nrows = len(cols) + 2
        
        fig = plt.figure(figsize=(15,3*nrows))
        #fig.suptitle(activity['title'])
        
        # Power histogram
        from_dtime = dtime
        to_dtime = df['dtime'].iloc[-1]
        ax = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=2)
        self.plot_power_zones_duration(from_dtime=from_dtime, to_dtime=to_dtime, labels=None, ftp=ftp, 
                                       title='Power Zones', ax=ax, label_type='simple', 
                                       show=False)

        # HR histogram
        ax = plt.subplot2grid((nrows, ncols), (0, 1), rowspan=2)
        self.plot_hr_zones_duration(from_dtime, to_dtime, max_hr, title='HR Zones', ax=ax, 
                                    label_type='simple', show=False)

        # Power curve
        ax = plt.subplot2grid((nrows, ncols), (0, 2), rowspan=2)
        self.plot_power_curves([(from_dtime, to_dtime)], min_interval=1, max_interval=3*3600, 
                               title='Power Curve', ax=ax, show=False)
        
        # Timeline
        for i_r, col in enumerate(cols):
            ax = plt.subplot2grid((nrows, ncols), (i_r+2, 0), colspan=3)
            df.plot(x=x, y=col, ax=ax, color=f'C{i_r}', zorder=10)
            ax.axhline(df[col].mean(), color=f'C{i_r}', linestyle='--', zorder=1)
            ax.set_ylabel(col)
            ax.set_title(f'{col} avg: {df[col].mean():.1f}, max: {df[col].max()}')
            ax.grid()

        fig.tight_layout()
        plt.show()

    def plot_activities(self, field, interval='W-SUN', sport=None, from_dtime=None, to_dtime=None,
                        return_df=False):
        df = self.get_activities(from_dtime=from_dtime, to_dtime=to_dtime, sport=sport)
        if df is None or not len(df):
            print('Error: no activities found')
            return
        
        if field=='tss':
            ph = self.profile_history
            ftph = FTPHistory(ph)
            df['ftp'] = df['dtime'].apply(ftph.get_ftp)
            calc_tss = lambda row: ZwiftTraining.avg_watt_to_tss(row['ftp'], row['power_avg'], row['mov_duration'])
            df['tss'] = df.apply(calc_tss, axis=1)
        
        df = df.set_index('dtime')
        df = df.groupby(pd.Grouper(freq=interval, closed='left', label='left')).agg({field: 'sum'})

        title = field.replace('_', ' ').title()
        if field in ['duration', 'mov_duration']:
            df[field] = df[field].dt.total_seconds() / 3600
            title += ' (Hours)'
        
        ax = df.plot.bar(y=field, title=title, align='center', color='darkgreen', alpha=0.5,
                         figsize=(15, 6), rot=45)
        @mpl.ticker.FuncFormatter
        def major_formatter_x(x, pos):
            return f'{df.index[x].date()}'
        ax.xaxis.set_major_formatter(major_formatter_x)
        
        ax.set_ylabel(title)
        ax.grid()
        ax.legend()
        plt.show()
        
        return df if return_df else None
        
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
                           title=activity['name'], src_file=src_file, route='', bike='', 
                           wheel='', note='', calories=calories)
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
                    print(f'Import error ignored: error parsing activity index: {start}, id: {activity["id"]}, datetime: {meta["dtime"]}, title: "{meta["title"]}", duration: {activity["duration"]}: FitParseError: {str(e)}')
                
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
            
    def plot_power_curves(self, periods, min_interval=None, max_interval=None, max_hr=None, title=None, 
                          ax=None, show=True):
        if ax is None:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
        
        power_intervals = None
        for i_period, (from_date, to_date) in enumerate(periods):
            from_date = pd.Timestamp(from_date)
            to_date = pd.Timestamp(to_date)
            
            df = self.calc_power_curve(from_date=from_date, to_date=to_date, max_hr=max_hr)
            if df is None:
                print(f'No power data for period {from_date} - {to_date}')
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

            for col in df.columns:
                arr = df[col]
                nc = sum(arr.isnull())
                if nc == len(df):
                    df = df.drop(columns=[col])
                    
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
        
        max_y = ax.get_ylim()[1]
        y_grid = 50
        ax.set_yticks(np.arange(0, (max_y+y_grid-1)//y_grid*y_grid, y_grid))
        ax.set_yticks(np.arange(0, max_y, 25), minor=True)
        
        ax.grid()
        #ax.grid(True, which='both', axis='y')
        ax.legend()
        if title:
            ax.set_title(title)
        if show:
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
        
        if to_date and to_date.hour==0 and to_date.minute==0:
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
    
    def calc_power_zones_duration(self, from_dtime, to_dtime, ftp=None, 
                                  zones=POWER_ZONES, labels=POWER_LABELS):
        #if from_dtime:
        from_dtime = pd.Timestamp(from_dtime)
            
        #if to_dtime:
        to_dtime = pd.Timestamp(to_dtime)
        if to_dtime.hour==0 and to_dtime.minute==0:
            to_dtime = to_dtime.replace(hour=23, minute=59, second=59)
        
        if labels and len(labels) != len(zones)+1:
            raise ValueError('Length of labels must be len(zones)+1 (extra label for power greater than the last zone)')
        
        activities = self.get_activities(from_dtime=from_dtime, to_dtime=to_dtime, sport='cycling')
        activities = activities.set_index('dtime', drop=True)
        if len(activities)==0:
            print(f'Error: no cycling activities found between {from_dtime} - {to_dtime}')
            return
        
        ph = self.profile_history
        if len(ph)==0:
            print(f'Error: no profile history')
            return
        
        ftph = FTPHistory(ph, default_ftp=ftp)
        
        empty = pd.DataFrame({'dummy': [0]*(len(zones)+1)}, index=range(1, len(zones)+2))
        
        results = [empty]
        ftps = [] # for averaging
        zones = [0] + zones
        for dtime, adf in activities.iterrows():
            ftp_at_that_time = ftph.get_ftp(dtime)
            if not ftp_at_that_time:
                print(f'No FTP at {dtime}')
                continue
            ftps.append(ftp_at_that_time)
            data = self.get_activity_data(dtime=dtime, src_file=adf['src_file'])
            if len(data)==0:
                sys.stderr.write(f'Error: unable to find activity on {dtime}\n')
                continue
            data = data[ (data['power'].notnull()) & (data['power'] != 0) ]
            data['%ftp'] = data['power'] / ftp_at_that_time
            data['power_zone'] = None
            for i_z in range(len(zones)):
                mi = zones[i_z]
                ma = zones[i_z+1] if i_z < len(zones)-1 else 1e10
                data.loc[ ((data['%ftp'] > mi) & (data['%ftp'] <= ma)), 'power_zone' ] = i_z+1

            secs_in_zone = data.groupby('power_zone').agg({'dtime': 'count'})
            secs_in_zone = secs_in_zone.rename(columns={'dtime': dtime})
            results.append(secs_in_zone)
                
        tmp = pd.concat(results, axis=1).fillna(0)
        duration = tmp.sum(axis=1)
        
        if not labels:
            labels = [f'Zone {i+1}' for i in range(len(duration))]
        duration.index = labels
        
        avg_ftp = np.mean(ftps)
        if pd.isnull(avg_ftp):
            raise ValueError('No FTP value is found for the specified period. Please manually specify ftp argument')
        
        pct_min = pd.Series(zones, index=labels)
        pct_max = pct_min.shift(-1).fillna(np.inf)
        power_min = (pct_min * avg_ftp).round(0).astype('int')
        power_max = (pct_max * avg_ftp).round(0)
        
        result = pd.DataFrame({'pct_min': pct_min,
                               'pct_max': pct_max,
                               'ftp': round(avg_ftp),
                               'power_min': power_min,
                               'power_max': power_max,
                               'duration': duration})
        return result
    
    @staticmethod
    def power_color_gradient(pct_ftp, opacity=1, output='css'):
        # POWER_ZONES = [0.55, 0.75, 0.9, 1.05, 1.2, 1.5]
        if pct_ftp < 0.3:
            #return '#0000aa'
            c = [0, 0, 0xaa/0xff]
        elif pct_ftp < 0.75:
            mix = (pct_ftp - 0.3) / (0.75 - 0.3)
            c1=np.array(mpl.colors.to_rgb('#0000aa'))
            c2=np.array(mpl.colors.to_rgb('#00aa00'))
            #return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
            c = (1-mix)*c1 + mix*c2
        elif pct_ftp < 0.9:
            mix = (pct_ftp - 0.75) / (0.9 - 0.75)
            c1=np.array(mpl.colors.to_rgb('#00aa00'))
            c2=np.array(mpl.colors.to_rgb('yellow'))
            #return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
            c = (1-mix)*c1 + mix*c2
        else:
            pct_ftp = min(pct_ftp, 1.2)
            mix = (pct_ftp - 0.9) / (1.2 - 0.9)
            c1=np.array(mpl.colors.to_rgb('yellow'))
            c2=np.array(mpl.colors.to_rgb('red'))
            #return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
            c = (1-mix)*c1 + mix*c2
        
        if output=='css':
            c = np.round(c * 0xff, 0)
            return f'rgba({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}, {opacity:.1f})'
        else:
            c = np.round(c * 0xff, 0).astype(int)
            #return f'({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}, {opacity:.1f})'
            return f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}{int(opacity*0xff):02x}'
        
    
    def plot_power_zones_duration(self, from_dtime, to_dtime, ftp=None, zones=POWER_ZONES, labels=POWER_LABELS,
                                  title=None, ax=None, show=True, label_type='default'):
        z = self.calc_power_zones_duration(from_dtime, to_dtime, ftp=ftp, zones=zones, labels=labels)
        if z is None or not len(z):
            return
        
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 5))
        
        # convert seconds to hour
        z['duration'] = z['duration'] / 3600
        total_hours = z['duration'].sum()
        
        
        x = range(len(z))
        zns = [0] + zones
        colors = [ZwiftTraining.power_color_gradient(zns[i], output='mpl') for i in range(len(zns))]
        bars = ax.bar(x, z['duration'], color=colors, alpha=0.6)
        
        # text percentage
        for i_b, rect in enumerate(bars):
            break
            height = rect.get_height()
            txt = f'{z["duration"].iloc[i_b]/total_hours:.0%}'
            ax.text(rect.get_x() + rect.get_width()/2.0, height, 
                    txt, ha='center', va='bottom')            
        
        ax.set_xticks(x)
        if label_type=='default':
            xlabels = [f"{idx}\n{r['pct_min']:.0%} - {r['pct_max']:.0%}\n{r['power_min']:.0f} - {r['power_max']:.0f} watt" for idx,r in z.iterrows()]
        elif label_type=='simple':
            xlabels = z.index
        else:
            assert False, 'Invalid label_type parameter'
        ax.set_xticklabels(xlabels)
        ax.set_ylabel('Duration (Hours)')
        dur_fmt = mpl.ticker.FuncFormatter(lambda y, pos: f'{int(y):02d}:{int((y-int(y))*60):02d}')
        ax.yaxis.set_major_formatter(dur_fmt)
        if title:
            ax.set_title(title)
        ax.grid()
        if show:
            plt.show()
    
    def plot_power_zones_duration2(self, from_dtime=None, to_dtime=None, freq='W-MON', zones=POWER_ZONES,
                                   labels=POWER_LABELS):
        if not to_dtime:
            to_dtime = pd.Timestamp.now()
        if not from_dtime:
            from_dtime = to_dtime - pd.Timedelta(weeks=20)
        dates = list(pd.date_range(from_dtime, end=to_dtime, freq=freq, normalize=True, closed='left'))
        dates.append(pd.Timestamp.now().normalize())
        periods = zip(dates[0:-1], dates[1:])
        
        rows = []
        for start_date, end_date in periods:
            df = self.calc_power_zones_duration(from_dtime=start_date, to_dtime=end_date, ftp=None, 
                                  zones=zones, labels=labels)
            #ser = pd.Series(pd.to_timedelta(df['duration'], unit='S').transpose(), name=start_date)
            ser = pd.Series(df['duration'].transpose(), name=start_date.date())
            rows.append(ser)
            
        df = pd.DataFrame(rows)
        
        zns = [0] + zones
        colors = [self.power_color_gradient(zns[i], output='mpl') for i in range(len(zns))]

        if False:
            fig, ax = plt.subplots(figsize=(15, 6))
            margin_bottom = np.zeros(len(df))
            #print(f'dates: {df.index}')
            #print(f'margin_bottom: {margin_bottom}')
            
            for zonenum, col in enumerate(df.columns):
                values = df[col]
                #print(f'  {col}: {values}')
                ax.bar(df.index, values, align='center', label=col, color=colors[zonenum],
                       bottom=margin_bottom, alpha=0.6, width=5)
                margin_bottom += values
        else:
            ax = df.plot.bar(stacked=True, color=colors, alpha=0.6, figsize=(15, 6), rot=45)
                        
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(3600))
        
        @mpl.ticker.FuncFormatter
        def major_formatter_y(val, pos):
            return f'{int(val//3600)}'
        ax.yaxis.set_major_formatter(major_formatter_y)
        ax.set_ylabel("Duration (Hours)")
        
        ax.grid()
        ax.legend(loc=2)
        plt.show()
        
    def calc_hr_zones_duration(self, from_dtime, to_dtime, max_hr, 
                               zones=HR_ZONES, labels=HR_LABELS):
        from_dtime = pd.Timestamp(from_dtime)
        to_dtime = pd.Timestamp(to_dtime)
        if to_dtime.hour==0 and to_dtime.minute==0:
            to_dtime = to_dtime.replace(hour=23, minute=59, second=59)
        
        if labels and len(labels) != len(zones)+1:
            raise ValueError('Length of labels must be len(zones)+1 (extra label for power greater than the last zone)')
        
        activities = self.get_activities(from_dtime=from_dtime, to_dtime=to_dtime, sport='cycling')
        activities = activities.set_index('dtime', drop=True)
        if len(activities)==0:
            print(f'Error: no cycling activities found between {from_dtime} - {to_dtime}')
            return
        
        empty = pd.DataFrame({'dummy': [0]*(len(zones)+1)}, index=range(1, len(zones)+2))
        
        results = [empty]
        zones = [0] + zones
        for dtime, adf in activities.iterrows():
            max_hr_at_that_time = max_hr # TODO: adjust based on age at that time?
            data = self.get_activity_data(dtime=dtime, src_file=adf['src_file'])
            if len(data)==0:
                sys.stderr.write(f'Error: unable to find activity on {dtime}\n')
                continue
            #if pd.isnull(data['hr'].iloc[0]):
            #    continue
            data = data[ data['hr'].notnull() ]
            data['%hr'] = data['hr'] / max_hr_at_that_time
            data['hr_zone'] = None
            for i_z in range(len(zones)):
                mi = zones[i_z]
                ma = zones[i_z+1] if i_z < len(zones)-1 else 1e10
                data.loc[ ((data['%hr'] > mi) & (data['%hr'] <= ma)), 'hr_zone' ] = i_z+1

            secs_in_zone = data.groupby('hr_zone').agg({'dtime': 'count'})
            secs_in_zone = secs_in_zone.rename(columns={'dtime': dtime})
            results.append(secs_in_zone)
                
        tmp = pd.concat(results, axis=1).fillna(0)
        duration = tmp.sum(axis=1).astype(int)
        
        if not labels:
            labels = [f'Zone {i+1}' for i in range(len(duration))]
        duration.index = labels
        
        pct_min = pd.Series(zones, index=labels)
        pct_max = pct_min.shift(-1).fillna(np.inf)
        hr_min = (pct_min * max_hr).round(0).astype('int')
        hr_max = (pct_max * max_hr).round(0)
        
        result = pd.DataFrame({'pct_min': pct_min,
                               'pct_max': pct_max,
                               'hr_min': hr_min,
                               'hr_max': hr_max,
                               'duration': duration})
        return result
    
    def plot_hr_zones_duration(self, from_dtime, to_dtime, max_hr, zones=HR_ZONES, labels=HR_LABELS,
                               title=None, ax=None, show=True, label_type='default'):
        z = self.calc_hr_zones_duration(from_dtime, to_dtime, max_hr, zones=zones, labels=labels)
        if z is None or not len(z):
            return
        
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 5))
        
        # convert seconds to hour
        z['duration'] = z['duration'] / 3600
        total_hours = z['duration'].sum()
        
        
        x = range(len(z))
        zns = [0] + zones
        colors = [ZwiftTraining.power_color_gradient(zns[i], output='mpl') for i in range(len(zns))]
        bars = ax.bar(x, z['duration'], color=colors, alpha=0.6)
        
        # text percentage
        for i_b, rect in enumerate(bars):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2.0, height, 
                    f'{z["duration"].iloc[i_b]/total_hours:.0%}', ha='center', va='bottom')            
        
        ax.set_xticks(x)
        if label_type=='default':
            xlabels = [f"{idx}\n{r['pct_min']:.0%} - {r['pct_max']:.0%}\n{r['hr_min']:.0f} - {r['hr_max']:.0f}" for idx,r in z.iterrows()]
        elif label_type=='simple':
            xlabels = z.index
        else:
            assert False, 'Invalid label_type parameter'
        ax.set_xticklabels(xlabels)
        ax.set_ylabel('Duration (Hours)')
        dur_fmt = mpl.ticker.FuncFormatter(lambda y, pos: f'{int(y):02d}:{int((y-int(y))*60):02d}')
        ax.yaxis.set_major_formatter(dur_fmt)
        if title:
            ax.set_title(title)
        ax.grid()
        if show:
            plt.show()
    
    def calc_training_form(self, sport='cycling', from_dtime=None, to_dtime=None, 
                           fatigue_period=7, fitness_period=42):
        assert sport=='cycling', "Only 'cycling' is supported for now"
        
        activities = self.get_activities(sport=sport, to_dtime=to_dtime)
        activities = activities[['dtime', 'mov_duration', 'power_avg']].set_index('dtime').dropna()
    
        ftph = FTPHistory(self.profile_history)
        def _calc_tss(row):
            d = row.name
            return self.avg_watt_to_tss(ftph.get_ftp(d), row['power_avg'], row['mov_duration'])
        
        activities['tss'] = activities.apply(_calc_tss, axis=1)
        
        #if activities.index[-1] < pd.Timestamp.now().normalize():
        # Add present
        activities.loc[pd.Timestamp.now(), 'tss'] = 0
        
        activities = activities[['tss']].resample('1D').sum()
        activities['Fitness (CTL)'] = activities['tss'].ewm(span=fitness_period).mean()
        activities['Fatigue (ATL)'] = activities['tss'].ewm(span=fatigue_period).mean()
        activities['Form (TSB)'] = activities['Fitness (CTL)'] - activities['Fatigue (ATL)']
        
        if from_dtime:
            activities = activities.loc[from_dtime:,:]
        
        return activities
    
    def plot_training_form(self, sport='cycling', from_dtime=None, to_dtime=None, 
                           fatigue_period=7, fitness_period=42, ax=None, show=True):
        df = self.calc_training_form(sport=sport, from_dtime=from_dtime, to_dtime=to_dtime,
                                     fatigue_period=fatigue_period, fitness_period=fitness_period)
        
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        print('Current form:', df.index[-1], df.iloc[-1]['Form (TSB)'].round(2))
        
        # TSS bar
        handles = [plt.Rectangle((0,0),1,1, color='gray', alpha=0.1)]
        ax.bar(df.index, df['tss'], alpha=0.1, color='gray', zorder=1)
        
        # metrics
        metrics = ['Fitness (CTL)', 'Fatigue (ATL)', 'Form (TSB)']
        for m in metrics:
            handles.extend( ax.plot(df.index, df[m], zorder=10) )
        
        ax.grid()
        ax.set_title('Form (Training Stress Balance)')
        ax.legend(handles, ['Daily TSS']+metrics)
        if show:
            plt.show()
        
    def _train_duration_predictor1(self, n=10, quiet=False):
        df = self.get_activities(sport='cycling')
        df = df[ (df['distance'] > 5) & (df['elevation'] > 1) & (df['power_avg'] > 5)]
        df = df.sort_values('dtime').tail(n)
        df = df.copy()
        if len(df) < n:
            sys.stderr.write(f'Not enough activities to make prediction (requires {n})\n')
            return None
        
        if not quiet:
            print(f'Training with {len(df)} datapoints from {df["dtime"].iloc[0]}')
        
        df['minutes'] = pd.to_timedelta(df['mov_duration']).dt.total_seconds() / 60
        df['dist/power'] = df['distance'] / df['power_avg']
        df['ele/power'] = df['elevation'] / df['power_avg']
        features = ['distance', 'elevation', 'dist/power']

        X = df[features]
        y = df['minutes']
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        
        if not quiet:
            pred = reg.predict(X)
            err = np.abs(pred - y)
            rep = pd.DataFrame({'y': np.round(y,1), 'pred': np.round(pred,1), 'err': np.round(err,1), 'pcterr': np.round(err/y, 2)})
            #print(rep)
            me = np.mean(err)
            me_pct = np.mean( np.abs(pred-y) / y )
            coefs = list(zip(features, [round(c, 3) for c in reg.coef_]))
            #coefs.append(('intercept', round(reg.intercept_, 2)))
            print(f'Coefs: {coefs}')
            print(f'Mean error: {rep["err"].mean():.1f} minutes ({rep["pcterr"].mean():.1%})')
            print(f'Max error : {rep["err"].max():.1f} minutes ({rep["pcterr"].max():.1%})')
        
        return reg        
    
    @staticmethod
    def _predict_duration1(regressor, df):
        df = df.copy()
        assert len(df.columns) == 3
        df.columns = ['distance', 'elevation', 'power_avg']
        df['dist/power'] = df['distance'] / df['power_avg']
        df = df.drop(columns=['power_avg'])
        #df['ele/power'] = df['elevation'] / df['power_avg']
        return regressor.predict(df).round(1)
    
    @staticmethod
    def _convert_to_segments(df, segment_length_meters=75):
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

        # Shuffle
        segments = segments.sample(frac=1)

        test_len = min(50, int(len(segments)/4))

        X = segments[['distance', 'elevation', 'power_avg']]
        y = segments['seconds']
        
        X_train, y_train = X.iloc[:-test_len], y.iloc[:-test_len]
        X_test, y_test = X.iloc[-test_len:], y.iloc[-test_len:]
        
        if not quiet:
            print(f'Train: {len(X_train)} samples, test: {len(X_test)} samples')
        
        model = LinearRegression().fit(X_train, y_train)
        #model = SVR(kernel='poly', cache_size=1000, degree=3).fit(X_train, y_train)
        #model = SVR(kernel='linear', cache_size=1000).fit(X_train, y_train)
        
        if not quiet:
            datasets = [('Train', X_train, y_train), ('Test', X_test, y_test)]
            for title, ds_x, ds_y in datasets:
                pred = model.predict(ds_x)
                pred_total = pred.sum()
                y_total = ds_y.sum()
                diff = abs(pred_total - y_total)
                err = diff / y_total
                print(f'{title}: duration: {y_total:.0f} secs, prediction: {pred_total:.0f} secs')
                print(f'{title} error: {diff:.0f} secs ({err:.0%})')
        
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
    
    def _load_routes(self, sport=None, allow_events=False, worlds=[]):
        route = pd.read_csv('data/routes.csv')
        route['total distance'] = route['distance'] + route['lead-in']
        route['done'] = 0
        route['restriction'] = route['restriction'].fillna('')
        route['badge'] = route['badge'].fillna(0)
        route['elevation'] = route['elevation'].fillna(0)
        
        if not allow_events:
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

        if worlds:
            for world in worlds:
                route = route[ route['world'].str.lower().str.contains(world.lower()) ]
        route = route.set_index(['world', 'route'])
            
        return route
    
    @staticmethod
    def tss_to_avg_watt(ftp, tss, duration):
        # TSS = hour * avg_watt**2 / ftp**2 * 100
        # avg_watt = (tss / hour * ftp**2 / 100) ** 0.5
        hour = pd.Timedelta(duration).total_seconds() / 3600
        return (tss * (ftp**2) / hour / 100) ** 0.5
        
    @staticmethod
    def avg_watt_to_tss(ftp, avg_watt, duration):
        if not ftp or not avg_watt:
            return None
        if pd.isnull(duration):
            return None
        #np = avg_watt
        #IF = np / ftp
        #tss = (pd.Timedelta(duration).total_seconds()  * np * IF) / (ftp * 3600) * 100
        tss = pd.Timedelta(duration).total_seconds() / 3600 * (avg_watt ** 2) / (ftp ** 2) * 100
        return tss

    def best_cycling_route(self, max_duration, avg_watt=None, tss=None, ftp=None, min_duration=None, 
                           kind=None, worlds=[], done=None, train_n=20, meetup=False, 
                           allow_events=False, quiet=False):
        assert (avg_watt is not None) or (tss is not None and ftp is not None)
        assert kind is None or kind in ['ride', 'interval', 'workout']
        max_minutes = pd.Timedelta(max_duration).total_seconds() / 60
        
        if avg_watt is None:
            avg_watt = ZwiftTraining.tss_to_avg_watt(ftp, tss, max_duration)
            if not quiet:
                print(f'Avg watt: {avg_watt:.0f}')
        
        regressor = self._train_duration_predictor1(n=train_n, quiet=quiet)
        df = self._load_routes(sport='cycling', worlds=worlds, allow_events=allow_events)
        df = df.drop(columns=['name'])
        if meetup:
            df['total distance'] = df['distance']
        
        # In "ride" mode, Zwift awards 20 xp per km
        XP_PER_KM = 20
        
        # In "interval", reward is 12 XP per minute. But there will be other
        # blocks such as warmups and free rides so usually we won't get the full XPs
        XP_PER_MIN_ITV = 12
        
        # In "workout" blocks, reward is 10 XP per minute for workout blocks and
        # 5-6 XP per minute for warmup/rampup/cooldown/rampdown blocks.
        XP_PER_MIN_WRK = 10 * 0.8 + 5.5 * 0.2
        
        df['power_avg'] = avg_watt
        
        # Set for better display
        df[['done', 'elevation', 'badge']] = df[['done', 'elevation', 'badge']].astype(int)
        
        # Clear badge if route is done
        df.loc[df['done'] != 0, 'badge'] = 0
        
        # Predict ride duration using linear regressor
        #df['route minutes'] = regressor.predict(df[['total distance', 'elevation', 'power_avg']]).round(1)
        df['route minutes'] = ZwiftTraining._predict_duration1(regressor, df[['total distance', 'elevation', 'power_avg']])
        df['pred avg speed'] = (df['total distance'] / (df['route minutes'] / 60.0)).round(1)
        df['pred distance'] = df['pred avg speed']/60 * max_minutes
        
        # Predict the XPs received for ride and interval activities, assuming that we finish
        # the full route.
        df['route minutes'] = np.floor(df['route minutes']).astype(int)
        df['pred xp (ride)'] = (df['pred distance'] * XP_PER_KM + df['badge']).astype(int)
        df['pred xp (interval)'] = (max_minutes * XP_PER_MIN_ITV + df['badge']).astype(int)
        df['pred xp (workout)'] = (max_minutes * XP_PER_MIN_WRK + df['badge']).astype(int)
        
        # Set which is the best activity (ride or interval), unless it's forced
        if not kind:
            df['best activity'] = df[['pred xp (ride)', 'pred xp (interval)', 'pred xp (workout)']].idxmax(axis=1)
            df['best activity'] = df['best activity'].str.extract(r'\((.*)\)')
    
            # Set the best XP received if the best activity is selected
            df['best pred xp'] = df[['pred xp (ride)', 'pred xp (interval)', 'pred xp (workout)']].max(axis=1)
        else:
            df['best activity'] = kind
            df['best pred xp'] = df[f'pred xp ({kind})']
        
        # Filter only routes less than the specified duration
        df = df[ df['route minutes'] <= max_minutes]
        df = df.sort_values(['best pred xp'], ascending=False)
        
        df['best pred xp /minutes'] = (df['best pred xp'] / max_minutes).round(1)
        
        # Filter only routes with at least the specified duration
        if min_duration:
            min_minutes = pd.Timedelta(min_duration).total_seconds() / 60
            df = df[ df['route minutes'] > min_minutes]
        
        df['route time'] = df['route minutes'].apply(lambda mnt: sec_to_str(mnt*60, full=True))
        
        # Display result
        columns = ['done', 'total distance', 'distance', 'lead-in', 'elevation', 'badge', 
                   'best activity', 'best pred xp', 'pred distance', 'pred avg speed',   
                   'route time', 'best pred xp /minutes']

        df = df[columns].rename(columns={'elevation': 'elev'})
        #df['power_avg'] = df['power_avg'].astype(int)

        #df = df.sort_values(['best pred xp/minutes'], ascending=False)
        df = df.sort_values(['best pred xp'], ascending=False)
        if done is not None:
            df = df[ df['done']==done ]
            
        return df

    def list_routes(self, world=None, route=None, done=None):
        df = pd.read_csv('data/routes.csv').set_index('name')
        df['done'] = False
        
        inventory = pd.read_csv(os.path.join(self.profile_dir, 'inventories.csv'))
        inventory = inventory[ inventory['type'] == 'route' ]
        
        for idx, row in inventory.iterrows():
            df.loc[row['name'], 'done'] = True
        
        if world:
            df = df[ df['world'].str.lower().str.contains(world.lower()) ]
        if route:
            df = df[ df['route'].str.lower().str.contains(route.lower()) ]
        if done is not None:
            df = df[ df['done']==done ]
            
        return df

    def list_inventory(self, kind, value):
        if '*' not in value:
            value += '*'
        return self.set_inventory(kind, value)

    def get_inventory(self, kind=None):
        df = pd.read_csv(os.path.join(self.profile_dir, 'inventories.csv'),
                                      parse_dates=['dtime'])
        if kind:
            df = df[ df['type']==kind ]
            
        return df
                
    def set_inventory(self, kind, value):
        assert kind in ['route', 'frame', 'wheels']

        if kind == 'route':
            master = pd.read_csv('data/routes.csv')
        elif kind == 'frame':
            master = pd.read_csv('data/frames.csv')
        elif kind == 'wheels':
            master = pd.read_csv('data/wheels.csv')
        else:
            assert False, f"Invalid kind '{kind}'"
        
        if '*' in value:
            master = master[ master['name'].str.lower().str.contains(value.replace('*', '').lower()) ]
            return master
        
        found = master[ master['name']==value ]
        if not len(found):
            raise ValueError(f'{kind} "{value}" not found')

        inventory = pd.read_csv(os.path.join(self.profile_dir, 'inventories.csv'),
                                parse_dates=['dtime'])

        found = inventory[ inventory['name']==value ]
        if len(found):
            raise ValueError(f'{kind} "{value}" already exist')
        
        inventory = inventory.append({'type': kind, 
                                      'name': value, 
                                      'dtime': pd.Timestamp.now().replace(microsecond=0)}, 
                                     ignore_index=True)
        inventory.to_csv(os.path.join(self.profile_dir, 'inventories.csv'), index=False)
        print(f'{kind} {value} successfully added to inventory')
        return inventory

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
            '17': 'cycling', # strava GPX export
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
                           src_file=os.path.split(path)[-1], route='', bike='', 
                           wheel='', note='', calories=calories)
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
        
        meta = OrderedDict(dtime=df['dtime'].iloc[0], sport=sport, title=title, src_file=os.path.split(path)[-1],
                           route='', bike='', wheel='', note='')
        return ZwiftTraining._process_activity(df, meta, copy=False)
    
    @staticmethod
    def parse_fit_file(path):
        """
        Convert FIT file to CSV
        """
        fitfile = FitFile(path)
        messages = fitfile.get_messages('record')
        records = [m.get_values() for m in messages]
        meta = OrderedDict(dtime=None, sport='', title='', src_file=os.path.split(path)[-1],
                           route='', bike='', wheel='', note='', )
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
    
    @staticmethod
    def display_zwo(path, ftp, watt='watt'):
        from IPython.display import display, clear_output, Markdown, HTML
        
        def _getText(childNodes):
            rc = []
            for node in childNodes:
                if node.nodeType == node.TEXT_NODE:
                    rc.append(node.data)
            return ''.join(rc)
    
        def _to_time(sec):
            sec = int(float(sec))
            if sec < 3600:
                return f'{sec//60:02d}:{sec%60:02d}'
            else:
                return f'{sec//3600:}:{(sec%3600)//60:02d}:{sec%60:02d}'
        
        
        # https://stackoverflow.com/a/50784012/7975037
        def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
            c1=np.array(mpl.colors.to_rgb(c1))
            c2=np.array(mpl.colors.to_rgb(c2))
            return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

        def colorFader3(c1, c2, c3, val, mid=0.5):
            c1=np.array(mpl.colors.to_rgb(c1))
            c2=np.array(mpl.colors.to_rgb(c2))
            c3=np.array(mpl.colors.to_rgb(c3))
            if val < mid:
                mix = val / mid
                return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
            else:
                val = min(val, 1)
                mix = (val - mid) / (1 - mid)
                return mpl.colors.to_hex((1-mix)*c2 + mix*c3)
                
        def power_color(s):
            #s = s.astype(float) / (ftp * 1.2)
            #clr = s.apply(lambda v: colorFader3("green", "yellow", "red", int(v)/(ftp*1.2), mid=0.75) if v else '')
            clr = s.apply(lambda v: ZwiftTraining.power_color_gradient(float(v)/ftp, opacity=0.5) if v else '')
            return [f'background-color: {c}; color: black;' for c in clr]

        doc = minidom.parse(path)
        
        workout = doc.getElementsByTagName('workout')[0]
        time = 0
        rows = []
        for node in workout.childNodes:
            if node.nodeType == node.TEXT_NODE:
                pass
            elif node.tagName.lower() == "freeride":
                duration = int(node.attributes['Duration'].value)
                text = ''
                for child in node.childNodes:
                    if child.nodeType != node.TEXT_NODE and child.tagName.lower()=='textevent':
                        when = child.attributes['timeoffset'].value
                        msg = child.attributes['message'].value
                        text += f"[{_to_time(when)}] {msg}\n"
                rows.append( {"time": time, 
                              "type": 'freeride', 
                              "duration": duration,
                              "repeat": '', 
                              "on watt": 0, 
                              "on duration": '',
                              "on rpm": '',
                              "off watt": 0, 
                              "off duration": '',
                              "off rpm": '',
                              "total watt second": int(0.5*ftp*duration),
                              "cum avg watt": 0,
                              "text": text} )            
                
                time += duration
            elif node.tagName.lower() == "intervalst":
                repeat = int(node.attributes['Repeat'].value)
                on_duration = int(node.attributes['OnDuration'].value)
                off_duration = int(node.attributes['OffDuration'].value)
                if watt=='watt':
                    on_power = float(node.attributes['OnPower'].value) * ftp
                    off_power = float(node.attributes['OffPower'].value) * ftp
                elif watt=='%ftp':
                    on_power = round(float(node.attributes['OnPower'].value), 2)
                    off_power = round(float(node.attributes['OffPower'].value), 2)
                else:
                    assert False, f'Invalid watt parameter "{watt}"'
                if 'Cadence' in node.attributes:
                    on_cadence = f"{node.attributes['Cadence'].value}"
                    off_cadence = f"{node.attributes['CadenceResting'].value}"
                else:
                    on_cadence = ''
                    off_cadence = ''
                duration = repeat * (on_duration + off_duration)
                text = ''
                for child in node.childNodes:
                    if child.nodeType != node.TEXT_NODE and child.tagName.lower()=='textevent':
                        when = child.attributes['timeoffset'].value
                        msg = child.attributes['message'].value
                        text += f"[{_to_time(when)}] {msg}\n"
                        if float(when)+20 > duration:
                            sys.stderr.write(f'Error: text event exceeds duration ([{_to_time(when)}] {msg})\n')
                rows.append( {"time": time, 
                              "type": 'interval', 
                              "duration": duration,
                              "repeat": repeat, 
                              "on watt": int(on_power), 
                              "on duration": on_duration,
                              "on rpm": on_cadence,
                              "off watt": int(off_power), 
                              "off duration": off_duration,
                              "off rpm": off_cadence,
                              "total watt second": int((on_power*on_duration + off_power*off_duration)*repeat),
                              "cum avg watt": 0,
                              "text": text} )
                time += duration
                
        df = pd.DataFrame(rows)
        
        watt_second = df['total watt second'].cumsum()
        cum_duration = df['duration'].cumsum()
        df['cum avg watt'] = (watt_second / cum_duration).astype(int)
        df = df.drop(columns=['total watt second'])
        
        total_secs = df['duration'].sum()
        
        df['time'] = df['time'].apply(_to_time)
        df['duration'] = df['duration'].apply(_to_time)
        #df['on watt'] = df['on watt'].round(2)
        #df['off watt'] = df['off watt'].round(2)
        
        print(f'Title      : {_getText(doc.getElementsByTagName("name")[0].childNodes)}')
        print(f'Author     : {_getText(doc.getElementsByTagName("author")[0].childNodes)}')
        print(f'Description: {_getText(doc.getElementsByTagName("description")[0].childNodes)}')
        print(f'Duration   : {_to_time(total_secs)}')
        print('Workouts:')
        
        display( HTML( df.style.apply(power_color, subset=['on watt', 'off watt']).render().replace("\\n","<br>") ) )
        #display( df.style.background_gradient(cmap='viridis', subset=['on watt', 'off watt']) )
    
        return
            