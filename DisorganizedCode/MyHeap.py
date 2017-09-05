import heapq

class MinHeap(object):
    def __init__(self, data=[], key=None):
        self.key = key
        if key == None:
            self.heap = [d for d in data]
        else:
            self.heap = [(self.key(d), d) for d in data]
        self.length = len(self.heap)
        heapq.heapify(self.heap)
        
    def push(self, item):
        if self.key == None:
            heapq.heappush(self.heap, item)
        else:
            decorated = self.key(item), item
            heapq.heappush(self.heap, decorated)
        self.length += 1

    def pop(self):
        if self.length == 0:
            return None
        elif self.key == None:
            self.length -= 1
            return heapq.heappop(self.heap)
        else:
            decorated = heapq.heappop(self.heap)
            self.length -= 1
            return decorated[1]
    def top(self):
        if self.length == 0:
            return None
        elif self.key == None:
            return self.heap[0]
        else:
            decorated = self.heap[0][1]
            return decorated
        
    def __getitem__(self,i):
        if self.key == None:
            return self.heap[i]
        else:
            return self.heap[i][1]

    def __len__(self): return self.length
    
class MaxHeapObj():
    def __init__(self,val): self.val = val
    def __lt__(self,other): return self.val > other.val
    def __eq__(self,other): return self.val == other.val
    def __str__(self):      return str(self.val)

class MaxHeap(MinHeap):
    def __init__(self, data=[], key=None):
        new_key = None
        if key == None:
            new_key = lambda x: MaxHeapObj(x)
        else:
            new_key = lambda x: MaxHeapObj(key(x))
        MinHeap.__init__(self, data, new_key)

class TopN:
    def __init__(self, n, data=[], key=None):
        self.key = key
        self.min_heap = MinHeap(data, key)
        self.max_length = n

    def tryPush(self, item):
        if len(self.min_heap) < self.max_length:
            self.min_heap.push(item)
        elif (self.key == None and self.min_heap.top() < item) or (self.key != None and self.key(self.min_heap.top()) < self.key(item)):
            self.min_heap.pop()
            self.min_heap.push(item)

    def pop(self):
        return self.min_heap.pop()

    def extractToList(self):
        result = []
        heap_len = len(self.min_heap)
        for i in range(heap_len):
            result.append(self.min_heap.pop())
        result.reverse()
        return result


    def __len__(self): return len(self.min_heap)

class BottomN:
    def __init__(self, n, data=[], key=None):
        self.key = key
        self.max_heap = MaxHeap(data, key)
        self.max_length = n

    def tryPush(self, item):
        if len(self.max_heap) < self.max_length:
            self.max_heap.push(item)
        elif (self.key == None and self.max_heap.top() > item) or (self.key != None and self.key(self.max_heap.top()) > self.key(item)):
            self.max_heap.pop()
            self.max_heap.push(item)

    def pop(self):
        return self.max_heap.pop()

    def extractToList(self):
        result = []
        heap_len = len(self.max_heap)
        for i in range(heap_len):
            result.append(self.max_heap.pop())
        result.reverse()
        return result


    def __len__(self): return len(self.max_heap)



