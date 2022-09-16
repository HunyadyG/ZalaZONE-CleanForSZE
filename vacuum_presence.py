# ===========================================================================
# Copyright (C) 2021-2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import argparse
from ifxAvian import Avian
from fft_spectrum import *
import numpy as np
from scipy import signal
from collections import namedtuple

import queue
import threading
import time
from miio import RoborockVacuum

class PresenceAntiPeekingAlgo:
    def __init__(self, num_samples_per_chirp, num_chirps_per_frame):
        """Presence and Anti-Peeking Algorithm
        
        This is a simple use case of an a presence detection and
        anti-peeking demo.
        
        Parameters:
        num_samples_per_chirp: Number of samples per chirp
        """
        self.num_samples_per_chirp = num_samples_per_chirp
        self.num_chirps_per_frame = num_chirps_per_frame

        # Algorithm Parameters
        self.detect_start_sample = num_samples_per_chirp//8
        self.detect_end_sample = num_samples_per_chirp//2
        self.peek_start_sample = num_samples_per_chirp//2
        self.peek_end_sample = num_samples_per_chirp

        self.threshold_presence = 0.0007
        self.threshold_peeking = 0.0006

        self.alpha_slow = 0.001
        self.alpha_med = 0.05
        self.alpha_fast = 0.6

        # Initialize state
        self.presence_status = False
        self.peeking_status = False
        self.first_run = True
        
        # Use Blackmann-Harris as window function
        self.window = signal.blackmanharris(num_samples_per_chirp).reshape(1,num_samples_per_chirp)

    def presence(self, mat):
        """Run the presence and anti-peeking algorithm on the current frame.
        
        Parameters:
            - mat: Radar data for one antenna as returned by Frame.get_mat_from_antenna
        
        Returns:
            - Tuple consisting of the state for presence detection and peeking.
              The first bool indicates if a target was detected. The second bool
              indicates if peeking was detected.
        """
        # copy values into local variables to keep names short
        alpha_slow = self.alpha_slow
        alpha_med = self.alpha_med
        alpha_fast = self.alpha_fast

        # Compute range FFT
        range_fft = fft_spectrum(mat, self.window)

        # Average absolute FFT values over number of chirps
        fft_spec_abs = abs(range_fft)
        fft_norm = np.divide(fft_spec_abs.sum(axis=0), self.num_chirps_per_frame)

        # Presence sensing
        if self.first_run: # initialize averages
            self.slow_avg = fft_norm
            self.fast_avg = fft_norm
            self.slow_peek_avg = fft_norm
            self.first_run = False

        if self.presence_status == False:
            alpha_used = alpha_med
        else:
            alpha_used = alpha_slow

        self.slow_avg = self.slow_avg*(1-alpha_used) + fft_norm*alpha_used
        self.fast_avg = self.fast_avg*(1-alpha_fast) + fft_norm*alpha_fast
        data = self.fast_avg-self.slow_avg

        self.presence_status = np.max(data[self.detect_start_sample:self.detect_end_sample]) > self.threshold_presence

        # Peeking sensing
        if self.peeking_status == False:
            alpha_used = self.alpha_med
        else:
            alpha_used = self.alpha_slow

        self.slow_peek_avg = self.slow_peek_avg*(1-alpha_used) + fft_norm*alpha_used
        data_peek = self.fast_avg-self.slow_peek_avg

        self.peeking_status = np.max(data_peek[self.peek_start_sample:self.peek_end_sample]) > self.threshold_peeking
        
        return namedtuple("state", ["presence", "peeking"])(self.presence_status, self.peeking_status)

###################################################

def goToSpot(zone: list):
    """Go to clean spot"""
    vac.set_fan_speed_preset(101)
    time.sleep(1)
    vac.zoned_clean([zone])

    lastClean = str(vac.last_clean_details())
    while lastClean == str(vac.last_clean_details()):
        time.sleep(1)

def getPresenceData():
    global askForPresenceData

    data = [0]*20
    dataCounter = 0
    while True:
        # Get radar data for the first RX antenna
        frame = device.get_next_frame()

        # matrix of dimension num_chirps_per_frame x num_samples_per_chirp for RX1
        mat = frame[0, :, :]
        presence_status, _ = algo.presence(mat)
        print(presence_status)
        data[dataCounter] = presence_status
        dataCounter += 1
        if askForPresenceData:
            askForPresenceData = False
            presenceDataQueue.put(data.copy())
        if dataCounter >= 20:
            dataCounter = 0

def checkForPresence(device, algo) -> bool:
    global askForPresenceData
    
    # vac.set_fan_speed_preset(101)
    # for i in range(8):
    #     vac.manual_control_once(45, 0, 1000)
    #     time.sleep(5)
    askForPresenceData = True

    pd = presenceDataQueue.get()
    
    presenceCount = 0
    for presenceValue in pd:
        if presenceValue:
            presenceCount += 1
    

    if presenceCount > 12:
        return False
    return True

if __name__ == "__main__":
    askForPresenceData = False
    presenceDataQueue = queue.Queue()

    vac = RoborockVacuum("192.168.10.103", "5137354e646d4865616c52596f72676a")

    config = Avian.DeviceConfig(
        sample_rate_Hz = 1e6,                   # ADC sample rate of 1MHz
        rx_mask = 1,                            # RX antenna 1 activated
        tx_mask = 1,                            # TX antenna 1 activated
        tx_power_level = 31,                    # TX power level of 31
        if_gain_dB = 33,                        # 33dB if gain
        start_frequency_Hz = 59_133_931_281,    # start frequency: 59.133931281 GHz
        end_frequency_Hz = 62_366_068_720,      # end frequency: 62.366068720 GHz
        num_samples_per_chirp = 64,             # 64 samples per chirp
        num_chirps_per_frame = 32,              # 32 chirps per frame
        chirp_repetition_time_s = 0.000411238,  # Chirp repetition time (or pulse repetition time) of 411.238us
        frame_repetition_time_s = 1/5, # Frame repetition time default 0.2s (frame rate of 5Hz)
        mimo_mode = "off")                      # MIMO disabled

    with Avian.Device() as device:
        # set device config for presence sensing
        device.set_config(config)

        # Read back the current configuration and initialize the algorithm
        config = device.get_config()
        algo = PresenceAntiPeekingAlgo(config.num_samples_per_chirp, config.num_chirps_per_frame)

        readThread = threading.Thread(target=getPresenceData)
        readThread.daemon = True
        readThread.start()

        rooms = [16, 17] # room ids to clean
        zones = [[27600, 25200, 27500, 25100, 1], [29200, 25500, 29100, 25400, 1]] # room enter spots

        i = 0
        cleanStarted = False
        while i < len(rooms):
            print("gotospot")
            goToSpot(zones[i]) # go to next room
            print("arrived")
            lastClean = str(vac.last_clean_details())
            
            vac.set_fan_speed_preset(108)
            while lastClean == str(vac.last_clean_details()): # clean until the room is cleaned or a human is detected
                if cleanStarted:
                    vac.pause() # pause cleaning for human detection
                    time.sleep(5)
                    print("pause")
                
                print("spin&check")
                if not checkForPresence(device, algo): # check for human presence
                    print("HUMAN PRESENCE DETECTED")
                    vac.stop()
                    break
                else:
                    print("No human presence")
                    
                    if not cleanStarted: # resume cleaning if presence is NOT detected
                        time.sleep(1)
                        vac.segment_clean([rooms[i]])
                        cleanStarted = True
                    else:
                        vac.resume_segment_clean()
                    print("wait")
                    time.sleep(60)
            
            i += 1

        vac.home() # at the end send the cleaner home (dock)
