#----------------------------------------------------------#
#  Q-learning for caching optimization in CCN enabled WSN  #
#----------------------------------------------------------#

import signal, os
import threading
import socket
import random
import math
import sys
from threading import Thread
from os import path

import queue
import time
from datetime import datetime

from collections import defaultdict

from model import *
from dqn_agent import *

#====================================================================#
#                             Tunable parameters                     #
#====================================================================#

RECEPTION_RANGE = 100               # used to simulate the limited transmission range of nodes
CACHE_CAPACITY = 4

NUM_NODES = 21 #DEBUG#

#====================================================================#
#                             Constants                              #
#====================================================================#

DEFAULT_DIRECTORY = "/home/"
CONFIG_PATH = "config_ccwsn.txt"
DATA_PATH = "data.txt"
INTEREST_PATH = "interest.txt"
#KNOWLEDGE_PATH = "knowledge.txt"
#SARS_PATH = "sars.txt"
TRACE_PATH = "trace.txt"
CACHE_PATH = "cache.txt"
HIT_MISS_PATH = "hit_miss.txt"


# Types of packets
INTEREST = 1
CONTENT_OBJECT = 2

# communication ports
INTEREST_CLIENT_PORT = 37020
INTEREST_SERVER_PORT = 44444
DATA_CLIENT_PORT = 37021
DATA_SERVER_PORT = 44445

# Learning parameters
EPS_START = 1           # starting value of epsilon
EPS_DECAY = 0.99        # decaying factor of epsilon

#====================================================================#
#                             Environment                            #
#====================================================================#

class env:
    # state info
    state_size = 6*(CACHE_CAPACITY+1)
    dist2req, dist2prov, betweenness_centrality = [0,1,2]
    # action info
    action_size = 2
    notcache, cache = range(action_size)
    # global information table
    knowledge = defaultdict(lambda: [[],[],[]])
    # last "slide_length" requests
    slide_length = 1000
    # replace it with a deque (maxlen=slide_length)
    slide = [0]*slide_length

    def add_request(content_name):
        env.slide.pop(0)
        env.slide.append(content_name)

    def update_knowledge(content_name, ihc, dhc, provDist):
        if len(env.knowledge[content_name][0]) == env.slide_length:
            [env.knowledge[content_name][_].pop(0) for _ in range(len(env.knowledge[content_name]))]
        env.knowledge[content_name][env.dist2req].append(ihc)
        env.knowledge[content_name][env.dist2prov].append(provDist-dhc)
        env.knowledge[content_name][env.betweenness_centrality].append(ihc-dhc)


    def state(new_data=None):
        # returns knowledge of cached data and new_data
        current_state = []
        for _cn in CS.LRU_cache:
            substate=[]
            substate.append(np.mean(env.knowledge[_cn][env.dist2req]))               # distance to requesters mean
            substate.append(np.var(env.knowledge[_cn][env.dist2req]))                # distance to requesters variance
            substate.append(np.mean(env.knowledge[_cn][env.dist2prov]))              # distance to providers mean
            substate.append(np.var(env.knowledge[_cn][env.dist2prov]))               # distance to providers variance
            substate.append(np.mean(env.knowledge[_cn][env.betweenness_centrality])) # mean betweenness centrality
            substate.append(env.slide.count(_cn)/len(env.slide))                         # request rate
            current_state.append(substate)

        if new_data:
            substate=[]
            substate.append(np.mean(env.knowledge[new_data][env.dist2req]))               # distance to requesters mean
            substate.append(np.var(env.knowledge[new_data][env.dist2req]))                # distance to requesters variance
            substate.append(np.mean(env.knowledge[new_data][env.dist2prov]))              # distance to providers mean
            substate.append(np.var(env.knowledge[new_data][env.dist2prov]))               # distance to providers variance
            substate.append(np.mean(env.knowledge[new_data][env.betweenness_centrality])) # mean betweenness centrality
            substate.append(env.slide.count(new_data)/len(env.slide))                         # request rate
            current_state.append(substate)

        current_state = np.array(current_state).flatten()
        return current_state

    def show_knowledge():
        kn = "\nKNOWLEDGE: \n"
        for _cn in env.knowledge:
            kn += "\t-> " + _cn + ": " + str(env.knowledge[_cn]) + " <-\n"
        return kn

#////////////////////////  UDP Sockets for communication ////////////////////////#

class InterestSocketClient(Thread):   # Socket for interest packet reception
    def __init__(self):
        Thread.__init__(self)
    def run(self):
        sock_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock_client.bind(("", INTEREST_CLIENT_PORT))
        except sock_client.error as msg:
            print("Interest Client: Error in bind", msg, "\n")

        while True:
            data, addr = sock_client.recvfrom(1024)
            interest_pkt = convert_to_interest(data)
            # If the received packet cannot be heard, because of transmission range, it is discarded
            # This test enables to virtualize the real deployment of nodes (which have real coordinates)
            dist = my_location.distance(interest_pkt.sender_location)
            if ((dist <= RECEPTION_RANGE) and (dist > 0)):                        # pkt in accepted range ==> received and treat pkt
                InterestReceptionQueue.put(interest_pkt)        # Queue for receiving interests

class InterestSocketServer(Thread):  # Socket for interest packet transmission
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        interest_sock_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        interest_sock_server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            interest_sock_server.bind(("", INTEREST_SERVER_PORT))
        except interest_sock_server.error as msg:
            print("Interest Server: Error in bind", msg, "\n")

        while True:
            interest_pkt = InterestTransmissionQueue.get(block=True)    # Queue for transmitting interests
            lockStdOut.acquire()
            print("interest transmission: ", interest_pkt.content_name)
            lockStdOut.release()
            byte_format = interest_pkt.to_bytes()
            interest_sock_server.sendto(byte_format, (Broadcast_IP_Addr, INTEREST_CLIENT_PORT))

class DataSocketClient(Thread):  # Socket for data packet reception from neighbors
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        data_sock_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_sock_client.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            data_sock_client.bind(("", DATA_CLIENT_PORT))
        except data_sock_client.error as msg:
            print("Data Client: Error in bind", msg, "\n")

        while True:
            data, addr = data_sock_client.recvfrom(1024)
            co_pkt = convert_to_co(data)
            # If the received packet cannot be heard, because of transmission range, it is discarded
            # This test enables to virtualize the real deployment of nodes (which have real coordinates)
            dist = my_location.distance(co_pkt.sender_location)
            if ((dist <= RECEPTION_RANGE) and (dist > 0)):                        # pkt in accepted range ==> receive and treat pkt
                DataReceptionQueue.put(co_pkt)                  # Queue for receiving data

class DataSocketServer(Thread):  # Socket for data packet transmission to neighbors
    def __init__(self):
        Thread.__init__(self)
    def run(self):
        data_sock_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        data_sock_server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        try:
            data_sock_server.bind(("", DATA_SERVER_PORT))
        except data_sock_server.error as msg:
            print("Data Server: Error in bind", msg, "\n")
        while True:
            co_pkt = DataTransmissionQueue.get(block=True)
            lockStdOut.acquire()
            print("data transmission: ", co_pkt.content_name)
            lockStdOut.release()
            byte_format = co_pkt.to_bytes()
            data_sock_server.sendto(byte_format, (Broadcast_IP_Addr, DATA_CLIENT_PORT))

#////////////////////////  END Sockets for communication ////////////////////////#

#===================================================================#
#                       Data class definitions                      #
#===================================================================#
# CCN packets were inspired from "Ndn specification. http://named-data.net. Online; acessed 15 July 2018."

class ContentObjectPacket:
    def __init__(self, content_name, data,  sender_location, dataHopCount=0, ttc=0, provDist=0):
        self.content_name = content_name          # Content Name of the data
        self.data = data                        # User data included in packet
        self.sender_location = sender_location
        self.dataHopCount = dataHopCount
        self.ttc = ttc
        self.provDist = provDist                         # hopcount from provider

    def to_string(self):
        s = "" + self.content_name + ";" + self.data + ";" + self.sender_location.to_string() \
            + ";" + str(self.dataHopCount) + ";" + str(self.ttc) + ";" + str(self.provDist) + ""
        return s

    def to_bytes(self):
        byte_format = bytes((self.to_string()),'utf-8')
        return byte_format


class InterestPacket:
    def __init__(self, content_name, sender_location, nonce, selector="", interestHopCount=0):
        self.content_name = content_name          # Content Name of the data
        self.sender_location = sender_location
        self.selector = selector                # Selector (Preference, order, publisher, filter, scope,... )
        self.nonce = nonce                      # Nonce (used interest identifier, to treat duplicated interests)
        self.interestHopCount = interestHopCount

    def to_string(self):
        s = "" + self.content_name + ";" + self.sender_location.to_string() + ";" + str(self.nonce) + ";" + self.selector \
            + ";" + str(self.interestHopCount) + ""
        return s

    def to_bytes(self):
        byte_format = bytes((self.to_string()),'utf-8')
        return byte_format

class CSEntry:
    def __init__(self, co_pkt):
        self.co_pkt = co_pkt
        self.prev = None
        self.next = None

class CS:
    data_provided = {}
    LRU_cache = {}
    capacity = CACHE_CAPACITY
    current_size = 0
    head = None
    end = None

    def exists(content_name):
        if (content_name in CS.data_provided):
            return 1
        elif (content_name in CS.LRU_cache):
            return 1
        else:
            return 0

    def get(content_name):
        if (content_name in CS.data_provided):
            return CS.data_provided[content_name]
        elif (content_name not in CS.LRU_cache):
            return 0

        entry = CS.LRU_cache[content_name]

        if CS.head == entry:
            return entry.co_pkt
        CS.remove(entry)
        CS.set_head(entry)
        return entry.co_pkt

    def _add_data_provided(co_pkt):
        CS.data_provided[co_pkt.content_name] = co_pkt

    def add(co_pkt):
        if co_pkt.content_name in CS.LRU_cache:
            entry = CS.LRU_cache[co_pkt.content_name]
            entry.co_pkt = co_pkt

            if CS.head != entry:
                CS.remove(entry)
                CS.set_head(entry)

        else:
            new_entry = CSEntry(co_pkt)
            if CS.current_size == CS.capacity:
                del CS.LRU_cache[CS.end.co_pkt.content_name]
                CS.remove(CS.end)
            CS.set_head(new_entry)
            CS.LRU_cache[co_pkt.content_name] = new_entry

    def set_head(entry):
        if not CS.head:
            CS.head = entry
            CS.end = entry
        else:
            entry.prev = CS.head
            CS.head.next = entry
            CS.head = entry
        CS.current_size += 1

    def remove(entry):
        if not CS.head:
            return 0

        if entry.prev:
            entry.prev.next = entry.next
        if entry.next:
            entry.next.prev = entry.prev

        if not entry.next and not entry.prev:
            CS.head = None
            CS.end = None

        if CS.end == entry:
            CS.end = entry.next
            CS.end.prev = None
        CS.current_size -= 1
        return entry

    def show_cache():
        s = ""
        #print("@@@ CS @@@ ")
        for _cn in CS.data_provided:
            #print("-> " + _cn + " <-", end="")
            s += "-> " + _cn + " <-"
        entry = CS.head
        while entry:
            #print("-> " + entry.co_pkt.content_name + " -", end="")
            s += "-> " + entry.co_pkt.content_name + " -"
            entry = entry.prev
        #print("\n")
        return s

class PITEntry:
    def __init__(self, interest_pkt):
        self.content_name = interest_pkt.content_name
        self.nonces = {interest_pkt.nonce}
        self.hopcounts = [interest_pkt.interestHopCount,interest_pkt.interestHopCount]
        self.time = time.time()

class PIT:
    table = dict()

    @staticmethod
    def add(interest):
        recent_access = 10              # get current time
        if (interest.content_name not in PIT.table.keys()):
            PIT.table[interest.content_name] = PITEntry(interest)
            return 1
        return 0

    @staticmethod
    def append(interest):
        if (not PIT.is_duplicate(interest)):
            PIT.table[interest.content_name].nonces.add(interest.nonce)
            hc_min = min(interest.interestHopCount, PIT.table[interest.content_name].hopcounts[0])
            hc_max = max(interest.interestHopCount, PIT.table[interest.content_name].hopcounts[1])
            PIT.table[interest.content_name].hopcounts = [hc_min,hc_max]
            return 1
        return 0

    @staticmethod
    def remove(content_name):
        if (content_name in PIT.table.keys()):
            interest = PIT.table[content_name]
            del PIT.table[content_name]
            return interest
        return 0

    @staticmethod
    def exists(content_name):
        if (content_name in PIT.table.keys()):
            return 1
        else:
            return 0

    @staticmethod
    def is_duplicate(interest):
        if PIT.exists(interest.content_name):
            if ((time.time() - PIT.table[interest.content_name].time) <= 3.0):
                return 1
            elif (interest.nonce in PIT.table[interest.content_name].nonces):
                return 1
        return 0

    @staticmethod
    def get_ihc(content_name):
        if PIT.exists(content_name):
            return sum(PIT.table[content_name].hopcounts)/2
        else:
            return -1

    @staticmethod
    def show_pit():
        print("@@@ PIT @@@ ")
        for content_name in PIT.table.keys():
            print("-> "+ content_name + " -", end="")
        print("\n")

class FIB:
    table = dict()

    @staticmethod
    def get(content_name):
        if (content_name in FIB.table.keys()):
            return FIB.table[content_name]
        else:
            return 0

    @staticmethod
    def add(content_name, ttc=0):
        c_count=0
        if (content_name not in FIB.table.keys()):
            FIB.table[content_name] = [ttc, c_count]
            return 1
        else:
            current_ttc, current_c_count = FIB.table[content_name]
            ttc = max(ttc, current_ttc)
            c_count = current_c_count + 1
            FIB.table[content_name] = [ttc, c_count]
            return 0

    @staticmethod
    def remove(content_name):
        if (content_name in FIB.table.keys()):
            FIB_entry = FIB.table[content_name]
            del FIB.table[content_name]
            return FIB_entry
        return 0

class Location:
    def __init__(self, loc):            # loc is in the format: '[loc_x, loc_y]'
        self.loc = [ float(i) for i in loc[1:-1].split(",") ]

    def distance(self, sender_location):
        dist = math.sqrt( math.pow((self.loc[0] - sender_location.loc[0]),2) \
                        + math.pow((self.loc[1] - sender_location.loc[1]),2))
        return math.ceil(dist)

    def to_string(self):
        s = str(self.loc)
        return s

class InterestGenerator:
    gen_queue = queue.Queue()
    waiting_list = defaultdict(lambda: [datetime.now().strftime("%H:%M:%S.%f")])

    @staticmethod
    def add(content_name, delay):
        InterestGenerator.gen_queue.put( (content_name,delay) )

    @staticmethod
    def generate(content_name):
        global reward, hit_miss_file
        if (not CS.exists(content_name)):
            interest_pkt = InterestPacket(content_name, my_location, random.random(), interestHopCount=0)
            if InterestGenerator.check(content_name):
                InterestGenerator.waiting_list[content_name].append(datetime.now().strftime("%H:%M:%S.%f"))
            else:
                _ = InterestGenerator.waiting_list[content_name]
            PIT.add(interest_pkt)
            interest_pkt.interestHopCount += 1
            InterestTransmissionQueue.put(interest_pkt)
        else:
            env.add_request(content_name)
            reward = reward + 1                                     # reward: +1 for cache hit
            hit_miss_file.write("hit\n")
            env.update_knowledge(content_name, 0, 0, 0)
            now = datetime.now().strftime("%H:%M:%S.%f")
            tracing(type="rcpt", content_name=content_name, provDist=0, t_rqst=now, t_rcpt=now)

        InterestGenerator.start()

    @staticmethod
    def start():
        #print( time.strftime("%M:%S", time.localtime()) )
        if (not InterestGenerator.gen_queue.empty()):
            content_name, delay = InterestGenerator.gen_queue.get()
            gen = threading.Timer(delay, InterestGenerator.generate, [content_name])
            gen.start()

    @staticmethod
    def check(content_name):
        if (content_name in InterestGenerator.waiting_list.keys()):
            return InterestGenerator.waiting_list[content_name]
        else:
            return None

    @staticmethod
    def remove(content_name):
        if InterestGenerator.check(content_name):
            InterestGenerator.waiting_list.pop(content_name)
            return 1
        return 0


#====================================================================#
#                               Functions                            #
#====================================================================#

def convert_to_co(byte_format):
    s = byte_format.decode('utf-8').split(";")
    co_pkt = ContentObjectPacket(s[0], s[1], Location(s[2]), int(s[3]), float(s[4]), int(s[5]))
    return co_pkt

def convert_to_interest(byte_format):
    s = byte_format.decode('utf-8').split(";")
    interest_pkt = InterestPacket(s[0], Location(s[1]), float(s[2]), s[3], int(s[4]))
    return interest_pkt

def interest_packet_handler(interest_pkt):
    global reward, hit_miss_file
    if (interest_pkt.interestHopCount > NUM_NODES):
        return -1
    if CS.exists(interest_pkt.content_name):
        co_pkt = CS.get(interest_pkt.content_name)
        co_pkt.dataHopCount = interest_pkt.interestHopCount - 1
        co_pkt.provDist = interest_pkt.interestHopCount
        DataTransmissionQueue.put(co_pkt)
        env.add_request(interest_pkt.content_name)
        reward = reward + 1                                     # reward: +1 for cache hit
        # increment cache hit
        hit_miss_file.write("hit\n")
        return 2
    elif not PIT.exists(interest_pkt.content_name):
        PIT.add(interest_pkt)
        interest_pkt.sender_location = my_location
        interest_pkt.interestHopCount += 1
        InterestTransmissionQueue.put(interest_pkt)
        env.add_request(interest_pkt.content_name)
        return 1
    elif not PIT.is_duplicate(interest_pkt):
        PIT.append(interest_pkt)
        interest_pkt.sender_location = my_location
        interest_pkt.interestHopCount += 1
        InterestTransmissionQueue.put(interest_pkt)
        env.add_request(interest_pkt.content_name)
        # update recent_access on the PIT entry
        return 0

def co_packet_handler(co_pkt):
    global state, action, reward, tmp_state0, eps, hit_miss_file, cache_file
    if not CS.exists(co_pkt.content_name):
        if PIT.exists(co_pkt.content_name):
            ihc = PIT.get_ihc(co_pkt.content_name)
            env.update_knowledge(co_pkt.content_name, ihc, co_pkt.dataHopCount, co_pkt.provDist)
            co_pkt.sender_location = my_location
            co_pkt.dataHopCount -= 1
            DataTransmissionQueue.put(co_pkt)
            PIT.remove(co_pkt.content_name)
            if (ihc>0):
                reward = reward - co_pkt.provDist/ihc - max(0,(5 - abs(ihc-co_pkt.dataHopCount)))               # reward: penalty for cache miss
            else:
                reward = reward - co_pkt.provDist                   # reward: case ihc = 0
            hit_miss_file.write("miss\n")
            t = InterestGenerator.check(co_pkt.content_name)
            if (t):
                _ = InterestGenerator.remove(co_pkt.content_name)
                now = datetime.now().strftime("%H:%M:%S.%f")
                for t_part in t:
                    tracing(type="rcpt", content_name=co_pkt.content_name, provDist=co_pkt.provDist, t_rqst=t_part, t_rcpt=now)

        if CS.current_size == CS.capacity:
            # retrieving next state
            next_state = env.state(co_pkt.content_name)
            # adding new experience tuple (s,a,r,s') to the replay buffer
            if tmp_state0:
                agent.step(state, action, reward, next_state)
                print("=====> STEP <=====")
                #with open(DEFAULT_DIRECTORY + SARS_PATH, 'a') as sars_file:
                #    sars_file.write("\ntuple: " + str(state) + "#" + str(action) + "#" + str(reward) + "#" + str(next_state) + "\n")
                #with open(DEFAULT_DIRECTORY + CACHE_PATH, 'a') as cache_file:
                cache_file.write("\n" + CS.show_cache() + "\nreward: " + str(reward) + "\n")
            tmp_state0 = 1                                      # to avoid the case where state is not yet defined
            # preparing for next experience
            reward = 0
            state = next_state
            action = agent.act(state, eps)
            eps = EPS_DECAY*eps
            if (action==env.cache):
                CS.add(co_pkt)
        else:                                                   # Cache memory is not full yet
            CS.add(co_pkt)

        return 1
    else:
        return 0

def tracing(type, content_name, provDist, t_rqst, t_rcpt):
    global trace_file, trace_line, hit_miss_file, cache_file
    lockStdOut.acquire()
    #with open(DEFAULT_DIRECTORY + TRACE_PATH, 'a') as trace_file:
    trace_file.write("\n" + type + "#" + content_name + "#" + str(provDist) + "#" + t_rqst + "#" + t_rcpt + "")
    trace_line+=1
    if (trace_line%500 == 0):
        trace_file.close()
        trace_file = open(DEFAULT_DIRECTORY + TRACE_PATH, "a")
        hit_miss_file.close()
        hit_miss_file = open(DEFAULT_DIRECTORY + HIT_MISS_PATH, "a")
        cache_file.close()
        cache_file = open(DEFAULT_DIRECTORY + CACHE_PATH, "a")
    lockStdOut.release()
    return 0


#------------------------------------------------------#
#           *------------------------------*           #
#           *            MAIN              *           #
#           *------------------------------*           #
#------------------------------------------------------#

print("===== Starting ===== \n")

#====================================================================#
#                       Main variables and main thread               #
#====================================================================#

# Open trace file
trace_file = open(DEFAULT_DIRECTORY + TRACE_PATH, "w")
#trace_file.close()
# Open knowledge file
#knowledge_file = open(DEFAULT_DIRECTORY + KNOWLEDGE_PATH, "w")
#knowledge_file.close()
# Open State-Action-Reward-nestState file
#sars_file = open(DEFAULT_DIRECTORY + SARS_PATH, "w")
#sars_file.close()
# Open cache file
cache_file = open(DEFAULT_DIRECTORY + CACHE_PATH, "w")
#cache_file.close()

hit_miss_file = open(DEFAULT_DIRECTORY + HIT_MISS_PATH, "w")
#hit_miss_file.close()

# Read node configuration
f_config = open(DEFAULT_DIRECTORY + CONFIG_PATH,"r")
Broadcast_IP_Addr = f_config.readline().replace("\n", "")   # IP address to bradcast packets in the ad hoc network
my_location = f_config.readline().replace("\n", "")         # Node coordinates: '[x,y]'
my_location = Location(my_location)
f_config.close()

# Read provided data (if file exists)
if (path.exists(DEFAULT_DIRECTORY + DATA_PATH)):
    data_file = open(DEFAULT_DIRECTORY + DATA_PATH,"r")
    for _line in data_file:
        _data = _line.replace("\n", "").split(";")
        CS._add_data_provided(ContentObjectPacket(_data[0], _data[1], my_location))
    CS.show_cache()
#else:
#    time.sleep(5)


# Read scheduled interests (if file exists)
if (path.exists(DEFAULT_DIRECTORY + INTEREST_PATH)):
    interest_file = open(DEFAULT_DIRECTORY + INTEREST_PATH,"r")
    for _line in interest_file:
        _interest = _line.split(";")
        InterestGenerator.add(_interest[0], float(_interest[1]))   # _interest == [content_name, delay]

###TOCHECK###
lockStdOut = threading.Lock()                               # for locking output stream, which is shared by multiple threads
lockSharedData = threading.Lock()                           # for protecting shared data access


################################################################

InterestReceptionQueue = queue.Queue()
InterestTransmissionQueue = queue.Queue()
DataReceptionQueue = queue.Queue()
DataTransmissionQueue = queue.Queue()


# Sockets declaration and opening
InterestClient = InterestSocketClient()  # Socket for receiving interest packets
InterestClient.start()
InterestServer = InterestSocketServer()  # Socket for sending interest packets
InterestServer.start()
DataClient = DataSocketClient()          # Socket for receiving data packets
DataClient.start()
DataServer = DataSocketServer()          # Socket for sending data packets
DataServer.start()

# Initializing our agent
agent = Agent(state_size=env.state_size, action_size=env.action_size, seed=0)
scores = []                 # list of scores to evaluate agent's performance
scores_window = deque(maxlen=100)  # last 100 scores
eps = EPS_START                    # initialize epsilon
timestep = 0
state = None
tmp_state0 = 0                                      # to avoid the case where state is not yet defined
reward = 0

trace_line = 0

InterestGenerator.start()

while (True):
    if not DataReceptionQueue.empty():
        co_pkt = DataReceptionQueue.get(block=False)
        co_packet_handler(co_pkt)
        #lockStdOut.acquire()
        #print("### Data received: "+ co_pkt.to_string())
        #CS.show_cache()
        #with open(DEFAULT_DIRECTORY + TRACE_PATH, 'a') as trace_file:
            #trace_file.write("\n" + env.show_knowledge() + "\n")
            #trace_file.write("\n SLIDE: " + str(env.slide) + "\n")
        #lockStdOut.release()

    if not InterestReceptionQueue.empty():
        interest_pkt = InterestReceptionQueue.get(block=False)
        interest_packet_handler(interest_pkt)
        #lockStdOut.acquire()
        #print("### Interest received: "+ interest_pkt.to_string())
        #PIT.show_pit()
        #lockStdOut.release()
