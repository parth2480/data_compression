import heapq
import numpy as np
import scipy as sp
import pandas as pd
import random
import sys

t = sp.arange(0, 5, 0.015625)
x = np.array([np.int16(random.randint(-65534, 65534)) for i in range(len(t))], dtype='uint16')
y = np.array([np.int16(random.randint(-65534, 65534)) for i in range(len(t))], dtype='uint16')
z = np.array([np.int16(random.randint(-65534, 65534)) for i in range(len(t))], dtype='uint16')
uncomp_sum = sys.getsizeof(t) + sys.getsizeof(x) + sys.getsizeof(y) + sys.getsizeof(z)
# print(type(x[2]), type(y[2]), type(z[2]))

# 1 kilobyte for approximately every 0.5 s
# for full range of 16 bit integers
#%%

df = pd.DataFrame(zip(t, x, y, z), columns=['Time', 'X', 'Y', 'Z'])
print(sys.getsizeof(df), sys.getsizeof(x), sys.getsizeof(y), sys.getsizeof(z), sys.getsizeof(t))
print(df.memory_usage(index=False).sum())
print('sum =', sys.getsizeof(t)+sys.getsizeof(x)+sys.getsizeof(y)+sys.getsizeof(z))
print('number of readings = ', len(t))
print(bytes(x))
# for 5 s of random data at 64readings/s, individual array add up to 5 kilobytes
# this equals about 1 kilobyte per second

#%%
import heapq
import os
from functools import total_ordering

"""
Code for Huffman Coding, compression and decompression. 
Explanation at http://bhrigu.me/blog/2017/01/17/huffman-coding-python-implementation/
"""


@total_ordering
class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if (other == None):
            return False
        if (not isinstance(other, HeapNode)):
            return False
        return self.freq == other.freq


class HuffmanCoding:
    def __init__(self, path):
        self.path = path
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    # functions for compression:

    def make_frequency_dict(self, text):
        frequency = {}
        for character in text:
            if not character in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        if (len(padded_encoded_text) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))
        return b

    def compress(self):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + ".bin"

        with open(self.path, 'r+') as file, open(output_path, 'wb') as output:
            text = file.read()
            text = text.rstrip()

            frequency = self.make_frequency_dict(text)
            self.make_heap(frequency)
            self.merge_nodes()
            self.make_codes()

            encoded_text = self.get_encoded_text(text)
            padded_encoded_text = self.pad_encoded_text(encoded_text)

            b = self.get_byte_array(padded_encoded_text)
            output.write(bytes(b))

        print("Compressed")
        return output_path

    """ functions for decompression: """

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-1 * extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""

        for bit in encoded_text:
            current_code += bit
            if (current_code in self.reverse_mapping):
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""

        return decoded_text

    def decompress(self, input_path):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + "_decompressed" + ".txt"

        with open(input_path, 'rb') as file, open(output_path, 'w') as output:
            bit_string = ""

            byte = file.read(1)
            while (len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.remove_padding(bit_string)

            decompressed_text = self.decode_text(encoded_text)

            output.write(decompressed_text)

        print("Decompressed")
        return output_path


path = 'C1_160308.FS.FULLRES.txt'
h = HuffmanCoding(path)
output_path = h.compress()

h.decompress(output_path)
#%%
import zlib
import binascii

# data = x
compressed_t = zlib.compress(t, 6)
compressed_x = zlib.compress(x, 6)
compressed_y = zlib.compress(y, 6)
compressed_z = zlib.compress(z, 6)
comp_sum = sys.getsizeof(compressed_t) + sys.getsizeof(compressed_x) + sys.getsizeof(compressed_y) + sys.getsizeof(compressed_z)

# print(binascii.hexlify(compressed), type(binascii.hexlify(compressed)))
print('t:', sys.getsizeof(compressed_t), sys.getsizeof(t))
print('x:', sys.getsizeof(compressed_x), sys.getsizeof(x))
print('y:', sys.getsizeof(compressed_y), sys.getsizeof(y))
print('z:', sys.getsizeof(compressed_z), sys.getsizeof(z))
print('Uncompressed Sum:', uncomp_sum)
print('Compressed Sum:', comp_sum)
#%%
data_df = pd.read_csv('C1_160313.txt', sep=",", engine='python')
data_df.columns = ['Time', 'X', 'Y', 'Z']
data_df.astype({'Y': 'float64'})
print(data_df.dtypes)
#%%
df_chunk = data_df[0:int(64*1.5)]
# print(df_chunk)
dtype = 'float32'

time = np.array(np.array(df_chunk['Time'].tolist(), dtype=dtype)*1000, dtype='int16')
x_values = np.array(df_chunk['X'].tolist(), dtype=dtype)
y_values = np.array(df_chunk['Y'].tolist(), dtype=dtype)
z_values = np.array(df_chunk['Z'].tolist(), dtype=dtype)
print(z_values)

comp_t = zlib.compress(time, 9)
comp_x = zlib.compress(x_values, 9)
comp_y = zlib.compress(y_values, 9)
comp_z = zlib.compress(z_values, 9)
comp_sum = sys.getsizeof(comp_t) + sys.getsizeof(comp_x) + sys.getsizeof(comp_y) + sys.getsizeof(comp_z)
uncomp_sum = sys.getsizeof(time) + sys.getsizeof(x_values) + sys.getsizeof(y_values) + sys.getsizeof(z_values)

print(len(data_df)/64)
# print('time:', sys.getsizeof(comp_t), sys.getsizeof(time))
# print('x values:', sys.getsizeof(comp_x), sys.getsizeof(x_values))
# print('y values:', sys.getsizeof(comp_y), sys.getsizeof(y_values))
# print('z values:', sys.getsizeof(comp_z), sys.getsizeof(z_values))
print('Uncompressed Sum:', uncomp_sum)
print('Compressed Sum:', comp_sum)
print(comp_sum/uncomp_sum)

'''
Zlib compression ranges from approx 1025 kilobytes for level 3 compression to 880 kilobytes for level 9 compression
'''
#%%
df_chunk = data_df[0:int(64*2)]
# print(df_chunk)
dtype = 'float64'

print(df_chunk['Z'].tolist())
time = np.array(np.array(df_chunk['Time'].tolist(), dtype=dtype)*10000, dtype='int16')
x_values = np.array(np.array(df_chunk['X'].tolist(), dtype=dtype)*10000, dtype='int16')
y_values = np.array(np.array(df_chunk['Y'].tolist(), dtype=dtype)*10000, dtype='int16')
z_values = np.array(np.array(df_chunk['Z'].tolist(), dtype=dtype)*10000, dtype='int16')
# print(z_values)

comp_t = zlib.compress(time, 9)
comp_x = zlib.compress(x_values, 9)
comp_y = zlib.compress(y_values, 9)
comp_z = zlib.compress(z_values, 9)
comp_sum = sys.getsizeof(comp_t) + sys.getsizeof(comp_x) + sys.getsizeof(comp_y) + sys.getsizeof(comp_z)
uncomp_sum = sys.getsizeof(time) + sys.getsizeof(x_values) + sys.getsizeof(y_values) + sys.getsizeof(z_values)

print(len(data_df)/64)
# print('time:', sys.getsizeof(comp_t), sys.getsizeof(time))
# print('x values:', sys.getsizeof(comp_x), sys.getsizeof(x_values))
# print('y values:', sys.getsizeof(comp_y), sys.getsizeof(y_values))
# print('z values:', sys.getsizeof(comp_z), sys.getsizeof(z_values))
print('Uncompressed Sum:', uncomp_sum)
print('Compressed Sum:', comp_sum)
print(comp_sum/uncomp_sum)
#%%
print(1/64)
