import os
import pandas as pd
import shutil
import sys
import time
import unittest

if True:
    sys.path.insert(0, os.path.abspath('..'))
    from ztraining import ZwiftTraining
    
    
class TestZwiftTraining(unittest.TestCase):
    def verify_gpx1(self, meta):
        self.assertEqual(meta['dtime'], pd.Timestamp('2019-01-26 07:41:53'))
        self.assertEqual(meta['sport'], 'cycling')
        self.assertAlmostEqual(meta['distance'], 3.16, delta=0.2)
        self.assertAlmostEqual(meta['mov_duration'].total_seconds(), 12*60+11, delta=60)
        self.assertAlmostEqual(meta['elevation'], 9, delta=10)
        self.assertAlmostEqual(meta['speed_avg'], 15.6, delta=1)
        self.assertAlmostEqual(meta['speed_max'], 25.6, delta=2)
        self.assertAlmostEqual(meta['power_avg'], 70, delta=5)
        self.assertAlmostEqual(meta['power_max'], 325, delta=7)
        self.assertAlmostEqual(meta['cadence_avg'], 39, delta=2)
        self.assertAlmostEqual(meta['cadence_max'], 137, delta=35) ## due to smoothing
        self.assertAlmostEqual(meta['temp_avg'], 31, delta=1)
        self.assertAlmostEqual(meta['temp_max'], 32, delta=1.5)
        
    def test_parse_gpx(self):
        # https://www.strava.com/activities/2103263896/overview
        df, meta = ZwiftTraining.parse_gpx_file('tcx_gpx_fit_files/2246203970.gpx')
        self.verify_gpx1(meta)
        
        self.assertAlmostEqual(df['mov_duration'].iloc[-1], 12*60+11, delta=60)
        self.assertAlmostEqual(df['distance'].iloc[-1], 3.16, delta=0.2)
    
    def verify_tcx1(self, meta):
        self.assertEqual(meta['dtime'], pd.Timestamp('2013-11-09 05:04:11'))
        self.assertEqual(meta['sport'], 'cycling')
        self.assertAlmostEqual(meta['distance'], 215.06, delta=1)
        self.assertAlmostEqual(meta['duration'].total_seconds(), pd.Timedelta('10:41:09').total_seconds(), delta=120)
        self.assertAlmostEqual(meta['mov_duration'].total_seconds(), pd.Timedelta('08:02:24').total_seconds(), delta=5*60)
        self.assertAlmostEqual(meta['elevation'], 2272, delta=30) # <-- !
        self.assertAlmostEqual(meta['speed_avg'], 26.7, delta=3)
        self.assertAlmostEqual(meta['speed_max'], 56.5, delta=5)
        self.assertAlmostEqual(meta['hr_avg'], 145, delta=1)
        self.assertAlmostEqual(meta['hr_max'], 193, delta=5)
        self.assertAlmostEqual(meta['power_avg'], 121, delta=2)
        self.assertAlmostEqual(meta['power_max'], 577, delta=20)
        self.assertAlmostEqual(meta['cadence_avg'], 82, delta=2)
        self.assertAlmostEqual(meta['cadence_max'], 123, delta=10)
        
    def test_parse_tcx(self):
        # https://www.strava.com/activities/94032423
        df, meta = ZwiftTraining.parse_tcx_file('tcx_gpx_fit_files/102574211.tcx')
        self.verify_tcx1(meta)
        
        self.assertAlmostEqual(df['duration'].iloc[-1], pd.Timedelta('10:41:09').total_seconds(), delta=120)
        self.assertAlmostEqual(df['mov_duration'].iloc[-1], pd.Timedelta('08:02:24').total_seconds(), delta=5*60)
        self.assertAlmostEqual(df['distance'].iloc[-1], 215.06, delta=1)
    
    def verify_fit1(self, meta):
        self.assertEqual(meta['dtime'].replace(second=0), pd.Timestamp('2020-06-27 06:39:00'))
        self.assertEqual(meta['sport'], 'cycling')
        self.assertAlmostEqual(meta['distance'], 54.94, delta=1)
        self.assertAlmostEqual(meta['duration'].total_seconds(), pd.Timedelta('2:09:53').total_seconds(), delta=2*60)
        self.assertAlmostEqual(meta['mov_duration'].total_seconds(), pd.Timedelta('2:01:06').total_seconds(), delta=2*60)
        self.assertAlmostEqual(meta['elevation'], 273, delta=25) # <-- !
        self.assertAlmostEqual(meta['speed_avg'], 27.2, delta=1)
        self.assertAlmostEqual(meta['speed_max'], 46.4, delta=3)
        self.assertAlmostEqual(meta['hr_avg'], 127, delta=1)
        self.assertAlmostEqual(meta['hr_max'], 146, delta=5)
        self.assertAlmostEqual(meta['power_avg'], 117, delta=1)
        self.assertAlmostEqual(meta['power_max'], 154, delta=5)
        self.assertAlmostEqual(meta['cadence_avg'], 67, delta=1)
        self.assertAlmostEqual(meta['cadence_max'], 117, delta=3)
        
    def test_parse_fit_strava(self):
        # https://www.strava.com/activities/3676214123
        # This is a Zwift fit file, but exported through Strava
        df, meta = ZwiftTraining.parse_fit_file('tcx_gpx_fit_files/3925200538.fit')
        self.verify_fit1(meta)
        
        self.assertAlmostEqual(df['duration'].iloc[-1], pd.Timedelta('2:09:53').total_seconds(), delta=2*60)
        self.assertAlmostEqual(df['mov_duration'].iloc[-1], pd.Timedelta('2:01:06').total_seconds(), delta=2*60)
        self.assertAlmostEqual(df['distance'].iloc[-1], 54.94, delta=1)
    
    def test_parse_fit_zwift(self):
        # The same activity as above, but with .fit file exported directly from Zwift
        df, meta = ZwiftTraining.parse_fit_file('tcx_gpx_fit_files/2020-06-27-06-38-50.fit')
        self.verify_fit1(meta)
        self.assertAlmostEqual(df['duration'].iloc[-1], pd.Timedelta('2:09:53').total_seconds(), delta=2*60)
        self.assertAlmostEqual(df['mov_duration'].iloc[-1], pd.Timedelta('2:01:06').total_seconds(), delta=2*60)
        self.assertAlmostEqual(df['distance'].iloc[-1], 54.94, delta=1)
    
    def verify_fit2(self, meta, skip_hr=False, skip_temp=False):
        self.assertEqual(meta['dtime'].replace(second=0), pd.Timestamp('2020-05-17 16:23:00'))
        self.assertEqual(meta['sport'], 'cycling')
        self.assertAlmostEqual(meta['distance'], 14.8, delta=0.2)
        self.assertAlmostEqual(meta['duration'].total_seconds(), pd.Timedelta('00:36:51').total_seconds(), delta=3*60)
        self.assertAlmostEqual(meta['mov_duration'].total_seconds(), pd.Timedelta('00:36:46').total_seconds(), delta=1*60)
        self.assertAlmostEqual(meta['elevation'], 0, delta=0)
        self.assertAlmostEqual(meta['speed_avg'], 24.1, delta=0.5)
        self.assertAlmostEqual(meta['speed_max'], 24.2, delta=11.5) # <-- !
        if not skip_hr:
            self.assertAlmostEqual(meta['hr_avg'], 135, delta=1)
            self.assertAlmostEqual(meta['hr_max'], 157, delta=5)
        self.assertAlmostEqual(meta['power_avg'], 131, delta=1)
        self.assertAlmostEqual(meta['power_max'], 251, delta=30)
        self.assertAlmostEqual(meta['cadence_avg'], 79, delta=1)
        self.assertAlmostEqual(meta['cadence_max'], 102, delta=5)
        if not skip_temp:
            self.assertAlmostEqual(meta['temp_avg'], 34, delta=0.2)
            self.assertAlmostEqual(meta['temp_max'], 35, delta=0.5)
        
    def test_parse_fit_gc_trainer(self):
        # Trainer .fit activity exported from Garmin Connect site
        df, meta = ZwiftTraining.parse_fit_file('tcx_gpx_fit_files/4944741403.fit')
        self.verify_fit2(meta)
        self.assertAlmostEqual(df['duration'].iloc[-1], pd.Timedelta('00:36:51').total_seconds(), delta=3*60)
        self.assertAlmostEqual(df['mov_duration'].iloc[-1], pd.Timedelta('00:36:46').total_seconds(), delta=1*60)
        self.assertAlmostEqual(df['distance'].iloc[-1], 14.8, delta=0.2)
    
    def test_parse_tcx_gc_trainer(self):
        # Same as above, but downloaded as TCX file from Garmin Connect site
        df, meta = ZwiftTraining.parse_tcx_file('tcx_gpx_fit_files/activity_4944741403.tcx')
        self.verify_fit2(meta, skip_temp=True)
        self.assertAlmostEqual(df['duration'].iloc[-1], pd.Timedelta('00:36:51').total_seconds(), delta=3*60)
        self.assertAlmostEqual(df['mov_duration'].iloc[-1], pd.Timedelta('00:36:46').total_seconds(), delta=1*60)
        self.assertAlmostEqual(df['distance'].iloc[-1], 14.8, delta=0.2)
        
    def test_parse_tcx_strava_trainer(self):
        # Still same as above, but downloaded as TCX file from Strava
        df, meta = ZwiftTraining.parse_tcx_file('tcx_gpx_fit_files/Afternoon_Trainer_Ride.tcx')
        self.verify_fit2(meta, skip_hr=True, skip_temp=True)
        self.assertAlmostEqual(df['duration'].iloc[-1], pd.Timedelta('00:36:51').total_seconds(), delta=3*60)
        self.assertAlmostEqual(df['mov_duration'].iloc[-1], pd.Timedelta('00:36:46').total_seconds(), delta=1*60)
        self.assertAlmostEqual(df['distance'].iloc[-1], 14.8, delta=0.2)
        
    def test_parse_zwift(self):
        # STILL the same activity as above, but pull directly from ZWIFT.COM
        zt = ZwiftTraining('test.json', quiet=False)
        df, meta = zt.parse_zwift_activity(581382235002805376, quiet=False)
        self.verify_fit1(meta)
        self.assertAlmostEqual(df['duration'].iloc[-1], pd.Timedelta('2:09:53').total_seconds(), delta=2*60)
        self.assertAlmostEqual(df['mov_duration'].iloc[-1], pd.Timedelta('2:01:06').total_seconds(), delta=2*60)
        self.assertAlmostEqual(df['distance'].iloc[-1], 54.94, delta=1)
        
    def test_parse_fit_bromo100k(self):
        # Older .fit file from Strava
        # https://www.strava.com/activities/120989151
        df, meta = ZwiftTraining.parse_fit_file('tcx_gpx_fit_files/132442327.fit')
        self.assertEqual(meta['sport'], 'cycling')
        self.assertAlmostEqual(meta['distance'], 109.89, delta=2)
        self.assertAlmostEqual(meta['duration'].total_seconds(), pd.Timedelta('8:13:36 ').total_seconds(), delta=1*60)
        self.assertAlmostEqual(meta['mov_duration'].total_seconds(), pd.Timedelta('6:15:09').total_seconds(), delta=10*60)
        self.assertAlmostEqual(meta['elevation'], 1932, delta=225) # <-- !!
        self.assertAlmostEqual(meta['speed_avg'], 17.6, delta=0.5)
        self.assertAlmostEqual(meta['speed_max'], 44.6, delta=0.5)
        self.assertAlmostEqual(meta['hr_avg'], 143, delta=1)
        self.assertAlmostEqual(meta['hr_max'], 173, delta=1)
        self.assertAlmostEqual(meta['power_avg'], 117, delta=1)
        self.assertAlmostEqual(meta['power_max'], 428, delta=10)
        self.assertAlmostEqual(meta['cadence_avg'], 74, delta=1.5)
        self.assertAlmostEqual(meta['cadence_max'], 210, delta=20) # <-- !
        self.assertAlmostEqual(meta['temp_avg'], 25, delta=0.5)
        self.assertAlmostEqual(meta['temp_max'], 37, delta=0.5)

    def test_parse_fit_trainer_with_gps(self):
        # We're training on trainer but left the GPS on! Any distance calculation based on GPS
        # will be ruined.
        # https://www.strava.com/activities/1746273899/overview
        df, meta = ZwiftTraining.parse_fit_file('tcx_gpx_fit_files/1873571076.fit')
        self.assertEqual(meta['sport'], 'cycling')
        self.assertAlmostEqual(meta['distance'], 14, delta=1)
        self.assertAlmostEqual(meta['duration'].total_seconds(), pd.Timedelta('0:37:50 ').total_seconds(), delta=1*60)
        self.assertAlmostEqual(meta['mov_duration'].total_seconds(), pd.Timedelta('0:35:37').total_seconds(), delta=1*60)
        self.assertAlmostEqual(meta['elevation'], 3, delta=27) # <-- !
        self.assertAlmostEqual(meta['speed_avg'], 23.6, delta=0.5)
        self.assertAlmostEqual(meta['speed_max'], 37.8, delta=6) # <-- !
        self.assertAlmostEqual(meta['power_avg'], 107, delta=1)
        self.assertAlmostEqual(meta['power_max'], 167, delta=10)
        self.assertAlmostEqual(meta['cadence_avg'], 82, delta=1.5)
        self.assertAlmostEqual(meta['cadence_max'], 132, delta=20) # <-- !
        self.assertAlmostEqual(meta['temp_avg'], 27, delta=0.5)
        self.assertAlmostEqual(meta['temp_max'], 27, delta=0.5)
        
    def test_import_files(self):
        if os.path.exists(ZwiftTraining.DEFAULT_PROFILE_DIR):
            shutil.rmtree(ZwiftTraining.DEFAULT_PROFILE_DIR, ignore_errors=True)
            
        zt = ZwiftTraining('test.json', quiet=False)
        n_updates = zt.import_files('tcx_gpx_fit_files', quiet=False)
        self.assertEqual(n_updates, 9)
        
        df = zt.get_activities(from_dtime='2013-11-09', to_dtime='2020-06-27', sport='cycling')
        self.assertEqual(len(df), 9)
        
        time.sleep(0.5)
        n_updates = zt.import_files('tcx_gpx_fit_files', quiet=False)
        self.assertEqual(n_updates, 0)

        df = zt.get_activities()
        self.assertEqual(len(df), 9)

    def test_zwift_update(self):
        if os.path.exists(ZwiftTraining.DEFAULT_PROFILE_DIR):
            shutil.rmtree(ZwiftTraining.DEFAULT_PROFILE_DIR, ignore_errors=True)

        zt = ZwiftTraining('test.json', quiet=False)
        n_updates = zt.zwift_update(start=0, max=1, quiet=False)
        self.assertEqual(n_updates, 2)

        df = zt.get_activities(to_dtime=pd.Timestamp.now())
        self.assertEqual(len(df), 1)

    def not_test_benny(self):
        zt = ZwiftTraining('../benny.json')
        n_updates = zt.update('/home/bennylp/Desktop/Google Drive/My Drive/Personal/Cycling/activities/raw')
        print(f'Done {n_updates} updates')


if __name__ == '__main__':
    if False:
        suite = unittest.TestSuite()
        suite.addTest(TestZwiftTraining("test_parse_fit_trainer_with_gps"))
        #suite.addTest(TestZwiftTraining("test_import_files"))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        unittest.main(warnings='ignore')

