import benchmark_timer as bm

t = bm.benchmark_timer()

def startTiming(name = None):
    t.startTiming(name)

def endTiming(name = None):
    t.endTiming(name)

def newTiming(name = None):
    t.newTiming(name)

def getTimes():
    t.getTimes()
