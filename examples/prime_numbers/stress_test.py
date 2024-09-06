#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:00:58 2024

@author: awspiv
"""

import multiprocessing as mp


def is_prime(n):
    """
    Check if a number is prime.
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def get_primes(start, end):
    """
    Find all prime numbers in the given range.
    """
    primes = []
    for num in range(start, end + 1):
        if is_prime(num):
            primes.append(num)
    return primes

def parallel_prime_finder(start, end, n_workers):
    """
    Use parallel processing to find all prime numbers in the given range.
    """
    chunk_size = (end - start + 1) // n_workers
    ranges = [(start + i * chunk_size, min(start + (i + 1) * chunk_size, end + 1)) for i in range(n_workers)]
    pool = mp.Pool(processes=n_workers)
    results = pool.starmap(get_primes, ranges)
    pool.close()
    pool.join()
    primes = []
    for result in results:
        primes.extend(result)
    return primes


#%%
if __name__ == "__main__":

    print("Inside stress test")

    n_workers = mp.cpu_count() # Number of workers (equal to the number of CPU cores)

    start = 1
    #end = int(1e8)
    end = int(1e7)
    parallel_prime_finder(start, end, n_workers)

    print("End")

