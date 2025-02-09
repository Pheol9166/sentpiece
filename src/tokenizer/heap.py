from typing import Dict


class MaxHeap:
    """
    Binary Max Heap class

    For efficient maximum searching and removal, we use binary max heap
    """

    def __init__(self):
        self.heap = []

    def swap(self, x: int, y: int):
        self.heap[x], self.heap[y] = self.heap[y], self.heap[x]

    def insert(self, value):
        self.heap.append(value)
        self._sift_up(len(self.heap) - 1)

    def extract_max(self):
        if not self.heap:
            return None

        max_value = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()

        if self.heap:
            self._sift_down(0)
            return max_value

    def _sift_up(self, index):
        """
        sort heap after insert value
        """
        parent = (index - 1) // 2
        while index > 0 and self.heap[index][0] > self.heap[parent][0]:
            self.swap(index, parent)
            index = parent
            parent = (index - 1) // 2

    def _sift_down(self, index: int = 0):
        """
        sort heap after extract max value.
        """
        left = 2 * index + 1
        right = 2 * index + 2
        largest = index

        if left < len(self.heap) and self.heap[left][0] > self.heap[largest][0]:
            largest = left
        if right < len(self.heap) and self.heap[right][0] > self.heap[largest][0]:
            largest = right
        if largest != index:
            self.swap(index, largest)
            self._sift_down(largest)

    def clear(self):
        self.heap.clear()

    def update(self, data: Dict[str, int]):
        for key, value in data.items():
            if (value, key) not in self.heap:
                self.insert((value, key))

    def __len__(self):
        return len(self.heap)
