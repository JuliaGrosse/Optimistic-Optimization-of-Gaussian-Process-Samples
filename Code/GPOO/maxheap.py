# Python3 implementation of Max Heap
# (Code from https://www.geeksforgeeks.org/max-heap-in-python/)
import sys


class MaxHeap:
    def __init__(self, maxsize, keyelement):

        self.maxsize = maxsize
        self.size = 0
        self.Heap = [(0, 0, 0, 0, 0, 0)] * (self.maxsize + 1)
        self.Heap[0] = (sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize)
        self.FRONT = 1
        self.keyelement = keyelement

    # Function to return the position of
    # parent for the node currently
    # at pos
    def parent(self, pos):

        return pos // 2

    # Function to return the position of
    # the left child for the node currently
    # at pos
    def leftChild(self, pos):

        return 2 * pos

    # Function to return the position of
    # the right child for the node currently
    # at pos
    def rightChild(self, pos):

        return (2 * pos) + 1

    # Function that returns true if the passed
    # node is a leaf node
    def isLeaf(self, pos):

        if pos >= (self.size // 2) and pos <= self.size:
            return True
        return False

    # Function to swap two nodes of the heap
    def swap(self, fpos, spos):

        self.Heap[fpos], self.Heap[spos] = (self.Heap[spos], self.Heap[fpos])

    # Function to heapify the node at pos
    def maxHeapify(self, pos):

        # If the node is a non-leaf node and smaller
        # than any of its child
        if not self.isLeaf(pos):
            try:
                if (
                    self.Heap[pos][self.keyelement]
                    < self.Heap[self.leftChild(pos)][self.keyelement]
                    or self.Heap[pos][self.keyelement]
                    < self.Heap[self.rightChild(pos)][self.keyelement]
                ):

                    # Swap with the left child and heapify
                    # the left child
                    if (
                        self.Heap[self.leftChild(pos)][self.keyelement]
                        > self.Heap[self.rightChild(pos)][self.keyelement]
                    ):
                        self.swap(pos, self.leftChild(pos))
                        self.maxHeapify(self.leftChild(pos))

                    # Swap with the right child and heapify
                    # the right child
                    else:
                        self.swap(pos, self.rightChild(pos))
                        self.maxHeapify(self.rightChild(pos))
            except IndexError:
                print(self.Heap[pos])

    # Function to insert a node into the heap based on the indexed part of element
    def insert(self, element):

        if self.size >= self.maxsize:
            return
        self.size += 1
        self.Heap[self.size] = element

        current = self.size
        while (
            self.Heap[current][self.keyelement]
            > self.Heap[self.parent(current)][self.keyelement]
        ):
            self.swap(current, self.parent(current))
            current = self.parent(current)

    # Function to print the contents of the heap
    def Print(self):

        for i in range(1, (self.size // 2) + 1):
            print(
                " PARENT : "
                + str(self.Heap[i])
                + " LEFT CHILD : "
                + str(self.Heap[2 * i])
                + " RIGHT CHILD : "
                + str(self.Heap[2 * i + 1])
            )

    # Function to remove and return the maximum
    # element from the heap
    def extractMax(self):

        popped = self.Heap[self.FRONT]
        self.Heap[self.FRONT] = self.Heap[self.size]
        self.size -= 1
        self.maxHeapify(self.FRONT)

        return popped

    def contains(self, element, pos):
        for i in range(1, self.size + 1):
            if self.Heap[i][pos] == element:
                return True
        return False


# Driver Code
if __name__ == "__main__":

    print("The maxHeap is ")

    maxHeap = MaxHeap(15, 2)
    maxHeap.insert((5, [6, 6, 6], 5, 5))
    maxHeap.insert((3, [6, 6, 6], 3, 3))
    maxHeap.insert((7, [6, 6, 6], 7, 7))
    maxHeap.insert((7, [6, 6, 6], 8, 7))

    maxHeap.Print()

    print("The Max val is " + str(maxHeap.extractMax()))
